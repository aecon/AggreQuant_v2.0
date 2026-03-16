"""
Residual blocks for ResUNet architectures.

This module provides residual connection blocks inspired by ResNet (He et al., 2016)
adapted for UNet encoder/decoder paths. Residual connections help with gradient
flow and enable training of deeper networks.
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """Residual block with skip connection.

    Implements: output = ReLU(input + F(input))

    where F is a two-convolution path. If input and output channels differ,
    a 1x1 convolution is used to match dimensions for the skip connection.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of channels between the two convolutions
            (default: same as out_channels)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = ResidualBlock(64, 128)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if mid_channels is None:
            mid_channels = out_channels

        # Main convolution path
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
        )

        # Skip connection: 1x1 conv if channels change, otherwise identity
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return self.relu(out)


class BottleneckResidualBlock(nn.Module):
    """Bottleneck residual block for deeper networks.

    Uses 1x1 -> 3x3 -> 1x1 convolution pattern to reduce computation.
    The bottleneck reduces channels by a factor of 4 before the 3x3 conv.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: Channel reduction factor for bottleneck (default: 4)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = BottleneckResidualBlock(256, 256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
    ) -> None:
        super().__init__()

        bottleneck_channels = out_channels // reduction

        # 1x1 reduce
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(bottleneck_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # 3x3 conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                bottleneck_channels,
                bottleneck_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(bottleneck_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        # 1x1 expand
        self.conv3 = nn.Sequential(
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
        )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels, affine=True),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return self.relu(out)
