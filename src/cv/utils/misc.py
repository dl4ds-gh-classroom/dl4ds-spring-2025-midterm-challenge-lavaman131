import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from omegaconf import ListConfig
from typing import Union, Tuple, Sequence, List, Dict, Any
import torch.nn as nn
import torch.optim as optim
import math
import warnings
from torch.optim.lr_scheduler import LRScheduler, _warn_get_lr_called_within_step


def set_seed(
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    # set webdataset seed
    os.environ["WDS_SEED"] = str(seed)


def make_2tuple(
    x: Union[int, Tuple[int, int], ListConfig],
) -> Tuple[int, int]:
    if isinstance(x, Sequence) and len(x) == 2:
        return tuple(x)

    assert isinstance(x, int)
    return (x, x)


DTYPE_MAPPING = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def get_dtype(dtype: str) -> torch.dtype:
    if dtype not in DTYPE_MAPPING:
        raise ValueError(f"Invalid dtype: {dtype}")
    return DTYPE_MAPPING[dtype]


def get_decay_parameters(
    model: nn.Module,
    lr: float,
    lr_mult: float,
) -> List[Dict[str, Any]]:
    """
    Get decay parameters for a model.

    From: https://kozodoi.me/blog/20220329/discriminative-lr
    """
    layer_names = []
    for idx, (name, param) in enumerate(model.named_parameters()):
        layer_names.append(name)
    layer_names.reverse()

    # placeholder
    parameters = []
    prev_group_name = layer_names[0].split(".")[0]

    # store params & learning rates
    for idx, name in enumerate(layer_names):
        # parameter group name
        cur_group_name = name.split(".")[0]

        # update learning rate
        if cur_group_name != prev_group_name:
            lr *= lr_mult
        prev_group_name = cur_group_name

        # display info
        # print(f"{idx}: lr = {lr:.6f}, {name}")

        # append layer parameters
        parameters += [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if n == name and p.requires_grad
                ],
                "lr": lr,
            }
        ]

    return parameters


class CosineScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: optim.Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        warmup_epochs: int = 0,
        warmup_start_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        """Cosine learning rate scheduler with optional warmup period.

        Args:
            optimizer: Wrapped optimizer.
            T_max: Maximum number of iterations/epochs.
            eta_min: Minimum learning rate.
            warmup_epochs: Number of epochs to linearly increase learning rate from warmup_start_lr to base_lr.
            warmup_start_lr: Initial learning rate during warmup period.
            last_epoch: The index of the last epoch. Default: -1.
            verbose: If True, prints a message to stdout for each update.
        """
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        _warn_get_lr_called_within_step(self)

        # During warmup period
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # During cosine annealing period
        else:
            effective_epoch = self.last_epoch - self.warmup_epochs
            cosine_period = self.T_max - self.warmup_epochs

            if effective_epoch >= cosine_period:
                return [self.eta_min for _ in self.base_lrs]

            return [
                self.eta_min
                + 0.5
                * (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * effective_epoch / cosine_period))
                for base_lr in self.base_lrs
            ]

    def _get_closed_form_lr(self) -> List[float]:
        """Get the closed-form learning rate for specific cases."""
        # During warmup period
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / self.warmup_epochs
            return [
                self.warmup_start_lr + alpha * (base_lr - self.warmup_start_lr)
                for base_lr in self.base_lrs
            ]

        # During cosine annealing period
        else:
            effective_epoch = self.last_epoch - self.warmup_epochs
            cosine_period = self.T_max - self.warmup_epochs

            if effective_epoch >= cosine_period:
                return [self.eta_min for _ in self.base_lrs]

            return [
                self.eta_min
                + 0.5
                * (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * effective_epoch / cosine_period))
                for base_lr in self.base_lrs
            ]
