import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Union


def save_ckpt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_path: Union[str, Path],
) -> None:
    ckpt_path = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        },
        ckpt_path,
    )


def load_ckpt(
    ckpt_path: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
) -> int:
    ckpt_path = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    epoch = state_dict["epoch"]
    return epoch + 1
