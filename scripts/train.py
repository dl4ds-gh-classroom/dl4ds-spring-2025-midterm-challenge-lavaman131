import torch
from torchvision.transforms import v2
from torchvision import datasets
import wandb
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from cv.data.dataset import CIFAR100
from cv.data.transforms import (
    make_classification_eval_transform,
    make_classification_train_transform,
)
from cv.utils.build import build_model, build_optimizer, build_scheduler, scale_lr
from cv.utils.config import parse_config
from pathlib import Path
from cv.utils.misc import get_decay_parameters, set_seed
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
import torch.optim as optim
from omegaconf import SCMode
import logging
from argparse import ArgumentParser, Namespace

from cv.utils.save import load_ckpt, save_ckpt
from cv.train import train_one_epoch, validate

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train")
logger.setLevel(logging.INFO)


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--base_config_dir", type=str, default="config")
    parser.add_argument("--model_config", type=str, default="convnet.yml")
    return parser.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = get_args()
    base_config_path = Path(args.base_config_dir)

    config = parse_config(
        base_config_path.joinpath("train.yml"),
        base_config_path.joinpath("models", args.model_config),
        base_config_path.joinpath("wandb.yml"),
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    transform_train = make_classification_train_transform(
        transform_config=config.transforms,
        crop_size=config.input_size,
    )
    transform_val = make_classification_eval_transform(
        resize_size=config.transforms.resize_size,
        crop_size=config.input_size,
    )

    base_train_dataset = CIFAR100(
        root=config.data_dir, train=True, download=True, transform=transform_train
    )
    base_val_dataset = CIFAR100(
        root=config.data_dir, train=True, download=True, transform=transform_val
    )

    # Generate indices for the split
    train_size = int(0.8 * len(base_train_dataset))
    val_size = len(base_train_dataset) - train_size
    indices = torch.randperm(len(base_train_dataset))

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create train and validation datasets with their respective transforms
    train_dataset = Subset(base_train_dataset, train_indices)
    val_dataset = Subset(base_val_dataset, val_indices)

    ### TODO -- define loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
    )

    cutmix_or_mixup = None
    if config.cutmix_or_mixup:
        cutmix = v2.CutMix(num_classes=config.num_classes)
        mixup = v2.MixUp(num_classes=config.num_classes)
        cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
    )

    ############################################################################
    #   Instantiate model and move to target device
    ############################################################################
    model = build_model(config).to(config.device)

    logger.info(f"Model: {model}")

    ############################################################################
    # Loss Function, Optimizer and optional learning rate scheduler
    ############################################################################
    loss_fn = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    config.lr = scale_lr(config.base_lr, config.batch_size)
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer)
    scaler = torch.amp.GradScaler(device=config.device) if config.use_fp16 else None

    epoch = 0
    if config.resume:
        epoch = load_ckpt(
            ckpt_path=config.ckpt_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )

    # Initialize wandb
    run = wandb.init(
        name=config.experiment_name,
        project=config.project,
        entity=config.entity,
        config=OmegaConf.to_container(
            config,
            resolve=True,
            structured_config_mode=SCMode.DICT,
        ),  # type: ignore
    )
    run.watch(model)  # watch the model gradients

    ############################################################################
    # --- Training Loop (Example - Students need to complete) ---
    ############################################################################
    best_val_acc = 0.0

    for epoch in range(config.epochs):
        train_loss, train_acc = train_one_epoch(
            config,
            epoch,
            model,
            train_loader,
            optimizer,
            loss_fn=loss_fn,
            cutmix_or_mixup=cutmix_or_mixup,
        )
        val_loss, val_acc = validate(
            config=config, model=model, val_loader=val_loader, loss_fn=loss_fn
        )
        scheduler.step()

        # log to wandb
        run.log(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],  # Log learning rate
            },
        )

        # Save the best model (based on validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir.joinpath("best_model.pth")
            save_ckpt(
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                ckpt_path=ckpt_path,
            )

    run.finish()


if __name__ == "__main__":
    main()
