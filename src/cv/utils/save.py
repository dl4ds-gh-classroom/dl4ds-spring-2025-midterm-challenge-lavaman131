import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Tuple, Union, Optional


def save_ckpt(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    ckpt_path: Union[str, Path],
    scaler: Optional[torch.amp.GradScaler] = None,
) -> None:
    ckpt_path = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "epoch": epoch,
        },
        ckpt_path,
    )


def load_ckpt(
    ckpt_path: Union[str, Path],
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> int:
    ckpt_path = Path(ckpt_path) if isinstance(ckpt_path, str) else ckpt_path
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(state_dict["scaler"])
    epoch = state_dict["epoch"]
    return epoch + 1
