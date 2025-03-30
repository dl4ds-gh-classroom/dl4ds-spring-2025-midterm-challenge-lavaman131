from typing import Callable, Dict, Any, List
import torch.nn as nn
from omegaconf import DictConfig
import torch.optim as optim
from functools import partial
from cv.models import ConvNet, ResNet
from cv.models.layers import resnet18, resnet34, resnet50
from timm.optim import create_optimizer_v2
import torch


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
    elif "res" in config.base_model:
        model = ResNet(
            model_name=config.base_model,
            num_classes=config.num_classes,
            input_size=config.input_size,
            pretrained=config.pretrained,
            freeze_backbone=config.freeze_backbone,
            act_layer=act_layer,
            drop_rate=config.drop_rate,
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
) -> optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )
