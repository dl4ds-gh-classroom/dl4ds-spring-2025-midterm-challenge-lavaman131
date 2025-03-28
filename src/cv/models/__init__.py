import torch.nn as nn
import torch

from cv.utils.misc import make_2tuple
from .layers import ConvBlock
from functools import partial
from typing import Callable, Mapping, List, Any, Tuple, Union

################################################################################
# Model Definition (Simple Example - You need to complete)
# For Part 1, you need to manually define a network.
# For Part 2 you have the option of using a predefined network and
# for Part 3 you have the option of using a predefined, pretrained network to
# finetune.
################################################################################


class ConvNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        input_size: Union[Tuple[int, int], int] = (32, 32),  # defaults to CIFAR size
        act_layer: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_size = make_2tuple(input_size)
        self.num_classes = num_classes
        self.blocks = [
            {
                "in_channels": 3,
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "in_channels": 64,
                "out_channels": 64,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "in_channels": 64,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        ]
        self.conv_blocks = nn.Sequential(
            *[
                ConvBlock(**block, act_layer=act_layer, dropout=dropout)
                for block in self.blocks
            ]
        )
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
