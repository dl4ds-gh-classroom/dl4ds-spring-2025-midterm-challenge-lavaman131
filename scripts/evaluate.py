import torch
from cv.data.dataset import CIFAR100
from cv.data.transforms import make_classification_eval_transform
from cv.evaluate.cifar100 import evaluate_cifar100_test, evaluate_cifar100_ttc
from cv.evaluate.ood import evaluate_ood_test, create_ood_df
import logging
from argparse import ArgumentParser
from pathlib import Path
from cv.utils.build import build_model
from cv.utils.config import parse_config
from cv.utils.misc import get_dtype, set_seed
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("evaluate")
logger.setLevel(logging.INFO)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--base_config_dir", type=str, default="config")
    parser.add_argument("--model_config", type=str, default="convnet.yml")
    return parser.parse_args()


def main() -> None:
    torch.set_float32_matmul_precision("high")
    args = get_args()
    base_config_path = Path(args.base_config_dir)

    config = parse_config(
        base_config_path.joinpath("inference.yml"),
        base_config_path.joinpath("models", args.model_config),
        base_config_path.joinpath("wandb.yml"),
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    model = build_model(config).to(config.device)

    state_dict = torch.load(config.ckpt_path, weights_only=False)

    model.load_state_dict(state_dict["model"])

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    ############################################################################
    # Evaluation -- shouldn't have to change the following code
    ############################################################################

    transform_test = make_classification_eval_transform(
        resize_size=config.transforms.resize_size,
        crop_size=config.input_size,
    )
    # (Create validation and test loaders)
    test_dataset = CIFAR100(
        root=config.data_dir, train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_mem,
        persistent_workers=config.persistent_workers,
    )

    if config.ttc:
        evaluate_fn = evaluate_cifar100_ttc
    else:
        evaluate_fn = evaluate_cifar100_test

    # --- Evaluation on Clean CIFAR-100 Test Set ---
    predictions, clean_accuracy = evaluate_fn(
        config=config,
        model=model,
        test_loader=test_loader,
        device=config.device,
    )
    logger.info(f"Clean CIFAR-100 Test Accuracy: {clean_accuracy:.2f}%")

    # --- Evaluation on OOD ---
    all_predictions = evaluate_ood_test(
        config=config,
        model=model,
    )

    # --- Create Submission File (OOD) ---
    submission_df_ood = create_ood_df(all_predictions)

    submission_df_ood.to_csv(output_dir.joinpath("submission_ood.csv"), index=False)
    logger.info("submission_ood.csv created successfully.")


if __name__ == "__main__":
    main()
