import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os
from omegaconf import ListConfig
from typing import Union, Tuple, Sequence


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
