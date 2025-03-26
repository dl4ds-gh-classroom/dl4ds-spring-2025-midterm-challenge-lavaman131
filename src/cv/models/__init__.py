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
        blocks: List[Mapping[str, Any]],
        input_size: Union[Tuple[int, int], int] = (32, 32),  # defaults to CIFAR size
        act_layer: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    ) -> None:
        super().__init__()
        self.input_size = make_2tuple(input_size)
        self.num_classes = num_classes
        self.conv_blocks = nn.Sequential(
            *[ConvBlock(**block, act_layer=act_layer) for block in blocks]
        )
        in_features = ConvNet.compute_fc_in_features(self.input_size, blocks)
        self.fc = nn.Linear(in_features, num_classes)

    @staticmethod
    def compute_fc_in_features(
        input_size: Tuple[int, int], blocks: List[Mapping[str, Any]]
    ) -> int:
        h, w = input_size
        for block in blocks:
            h = (
                (h + 2 * block["padding"] - block["kernel_size"]) // block["stride"]
            ) + 1  # type: ignore
            w = (
                (w + 2 * block["padding"] - block["kernel_size"]) // block["stride"]
            ) + 1  # type: ignore
        return h * w * blocks[-1]["out_channels"]  # type: ignore

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
