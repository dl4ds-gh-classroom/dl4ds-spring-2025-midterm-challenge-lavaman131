from typing import Callable, Dict, Any
import torch.nn as nn
from omegaconf import DictConfig
import torch.optim as optim
from functools import partial
from cv.models import ConvNet


def get_act_layer(act_layer: str) -> Callable[..., nn.Module]:
    if act_layer == "ReLU":
        return partial(nn.ReLU, inplace=True)
    else:
        raise NotImplementedError(f"Activation function {act_layer} not supported")


def build_model(
    config: DictConfig,
) -> nn.Module:
    act_layer = get_act_layer(config.act_layer)

    if config.base_model == "ConvNet":
        return ConvNet(
            config.num_classes,
            config.blocks,
            config.input_size,
            act_layer=act_layer,
        )
    else:
        raise NotImplementedError(f"Model {config.base_model} not supported")


def build_optimizer(
    config: DictConfig,
    model: nn.Module,
) -> optim.Optimizer:
    if config.optim == "adamw":
        return optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas)
    elif config.optim == "adam":
        return optim.Adam(model.parameters(), lr=config.lr, betas=config.betas)
    elif config.optim == "sgd":
        return optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum)
    else:
        raise ValueError(f"Optimizer {config.optim} not supported")


def build_scheduler(
    config: DictConfig,
    optimizer: optim.Optimizer,
) -> optim.lr_scheduler.LRScheduler:
    return optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.min_lr
    )
