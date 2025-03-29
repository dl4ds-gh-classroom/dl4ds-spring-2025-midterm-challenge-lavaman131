import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from omegaconf import ListConfig
from typing import Union, Tuple, Sequence, List, Dict, Any
import torch.nn as nn


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
