import torch
import torch.nn as nn
from typing import Callable


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act_layer: Callable[..., nn.Module],
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = act_layer()
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class DoubleConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        act_layer: Callable[..., nn.Module],
        drop_rate: float,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = act_layer()
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = act_layer()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.bn2(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x
