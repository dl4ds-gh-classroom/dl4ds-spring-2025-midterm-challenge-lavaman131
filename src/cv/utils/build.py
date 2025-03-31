from typing import Callable, Dict, Any, List
import torch.nn as nn
from omegaconf import DictConfig
import torch.optim as optim
from functools import partial
from cv.models import ConvNet, ResNeXt, ResNet
from timm.optim import create_optimizer_v2
from cv.utils.misc import CosineScheduler
from torch.optim.lr_scheduler import LRScheduler


def scale_lr(lr: float, batch_size: int) -> float:
    return lr * batch_size / 256.0


def get_act_layer(act_layer: str) -> Callable[..., nn.Module]:
    if act_layer == "ReLU":
        return partial(nn.ReLU, inplace=True)
    elif act_layer == "LeakyReLU":
        return partial(nn.LeakyReLU, inplace=True)
    elif act_layer == "GELU":
        return nn.GELU
    elif act_layer == "Mish":
        return partial(nn.Mish, inplace=True)
    else:
        raise NotImplementedError(f"Activation function {act_layer} not supported")


def build_model(
    config: DictConfig,
) -> nn.Module:
    act_layer = get_act_layer(config.get("act_layer", "ReLU"))

    if config.base_model == "convnet":
        model = ConvNet(
            num_classes=config.num_classes,
            input_size=config.input_size,
            act_layer=act_layer,
            drop_rate=config.drop_rate,
        )
    elif "resnet" in config.base_model:
        model = ResNet(
            model_name=config.base_model,
            num_classes=config.num_classes,
            input_size=config.input_size,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            act_layer=act_layer,
            drop_rate=config.drop_rate,
            drop_block_rate=config.drop_block_rate,
        )
    elif "resnext" in config.base_model:
        model = ResNeXt(
            model_name=config.base_model,
            num_classes=config.num_classes,
            input_size=config.input_size,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            act_layer=act_layer,
            drop_rate=config.drop_rate,
            drop_block_rate=config.drop_block_rate,
            cardinality=config.cardinality,
            base_width=config.base_width,
        )
    else:
        raise NotImplementedError(f"Model {config.base_model} not supported")

    return model


def build_optimizer(
    config: DictConfig,
    parameters: List[Dict[str, Any]],
) -> optim.Optimizer:
    optimizer = create_optimizer_v2(
        parameters,
        opt=config.optim,
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
    )
    return optimizer


def build_scheduler(
    config: DictConfig,
    optimizer: optim.Optimizer,
) -> LRScheduler:
    return CosineScheduler(
        optimizer,
        T_max=config.epochs,
        eta_min=config.min_lr,
        warmup_epochs=config.warmup_epochs,
        warmup_start_lr=config.warmup_start_lr,
    )
