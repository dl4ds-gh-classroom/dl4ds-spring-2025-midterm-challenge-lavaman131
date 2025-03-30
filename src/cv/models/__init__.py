import torch.nn as nn
import torch
from cv.utils.misc import make_2tuple
from .layers import DoubleConvBlock, ConvBlock
from functools import partial
from typing import Callable, Any, Literal, Tuple, Union
from collections import OrderedDict
import torch.nn.functional as F
from timm import create_model

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
        in_channels: int = 3,
        act_layer: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_size = make_2tuple(input_size)
        self.num_classes = num_classes
        self.blocks = [
            {
                "in_channels": in_channels,
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
            {
                "in_channels": 128,
                "out_channels": 128,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "in_channels": 128,
                "out_channels": 256,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
            {
                "in_channels": 256,
                "out_channels": 512,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
            },
        ]
        self.conv_blocks = nn.Sequential(
            *[
                DoubleConvBlock(**block, act_layer=act_layer, drop_rate=drop_rate)
                for block in self.blocks
            ]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(self.blocks[-1]["out_channels"], 128)
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class ResNet(nn.Module):
    def __init__(
        self,
        model_name: Literal["resnet18", "resnet34", "resnet50"],
        num_classes: int,
        input_size: Union[Tuple[int, int], int] = (32, 32),  # defaults to CIFAR size
        in_channels: int = 3,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        drop_rate: float = 0.0,
        act_layer: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.model = create_model(
            model_name=model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            act_layer=act_layer,
            **kwargs,
        )
        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()
        self.model.avgpool = nn.AvgPool2d(kernel_size=4)
        self.model.fc1 = nn.Linear(self.model.num_features, 128)
        self.model.dropout = nn.Dropout(p=drop_rate)
        self.model.fc2 = nn.Linear(128, num_classes)
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc1.weight.requires_grad = True
        self.model.fc1.bias.requires_grad = True
        self.model.fc2.weight.requires_grad = True
        self.model.fc2.bias.requires_grad = True

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.model.fc1(x)
        x = self.model.dropout(x)
        x = self.model.fc2(x)
        return x

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.forward_features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
