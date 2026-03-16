"""
Squeeze-and-Excitation blocks for channel recalibration.

This module implements SE blocks as described in:
"Squeeze-and-Excitation Networks" (Hu et al., 2018)

SE blocks adaptively recalibrate channel-wise feature responses by
explicitly modeling interdependencies between channels.
"""

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    SE blocks perform channel-wise recalibration in three steps:
    1. Squeeze: Global average pooling to get channel statistics
    2. Excitation: Two FC layers to learn channel weights
    3. Scale: Multiply original features by learned weights

    Arguments:
        channels: Number of input/output channels
        reduction: Channel reduction ratio for bottleneck (default: 16)

    Returns:
        Recalibrated features of shape (B, channels, H, W)

    Example:
        >>> se = SEBlock(channels=256, reduction=16)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = se(x)
        >>> out.shape
        torch.Size([1, 256, 64, 64])

    References:
        Hu et al., "Squeeze-and-Excitation Networks"
        https://arxiv.org/abs/1709.01507
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        # Ensure reduction doesn't make bottleneck too small
        reduced_channels = max(channels // reduction, 1)

        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape

        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, C)
        squeezed = self.squeeze(x).view(batch_size, channels)

        # Excitation: (B, C) -> (B, C)
        weights = self.excitation(squeezed)

        # Scale: (B, C) -> (B, C, 1, 1) and multiply
        weights = weights.view(batch_size, channels, 1, 1)
        return x * weights


class SEConvBlock(nn.Module):
    """Convolution block with integrated SE attention.

    Combines double convolution with SE block for enhanced feature learning.
    The SE block is applied after the convolutions.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: SE reduction ratio (default: 16)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = SEConvBlock(64, 128)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.se = SEBlock(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        return out


class SEResidualBlock(nn.Module):
    """Residual block with SE attention (SE-ResNet style).

    Combines residual connections with SE channel attention.
    SE is applied at the end of the residual path before adding the skip.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: SE reduction ratio (default: 16)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = SEResidualBlock(64, 128)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
        )

        self.se = SEBlock(out_channels, reduction)

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
        out = self.se(out)

        out = out + identity
        out = self.relu(out)

        return out
