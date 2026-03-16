"""
Efficient Channel Attention (ECA) block.

This module implements ECA as described in:
"ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
(Wang et al., 2020)

ECA replaces the FC bottleneck of SE with a 1D convolution across channels,
avoiding the dimensionality reduction that loses information.
"""

import math
from typing import Optional

import torch
import torch.nn as nn


class ECABlock(nn.Module):
    """Efficient Channel Attention block.

    ECA performs channel attention using a 1D convolution instead of
    SE's fully-connected bottleneck. Each channel interacts with its
    k nearest neighbors, where k is adaptively determined from the
    channel count.

    SE:  z -> FC(C->C/r) -> ReLU -> FC(C/r->C) -> sigmoid   (2C²/r params)
    ECA: z -> Conv1D(kernel=k) -> sigmoid                    (k params)

    Arguments:
        channels: Number of input/output channels
        gamma: Hyperparameter for adaptive kernel size (default: 2)
        b: Hyperparameter for adaptive kernel size (default: 1)
        kernel_size: Override adaptive kernel size (default: None, auto-compute)

    Example:
        >>> eca = ECABlock(channels=256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = eca(x)
        >>> out.shape
        torch.Size([1, 256, 64, 64])

    References:
        Wang et al., "ECA-Net: Efficient Channel Attention"
        https://arxiv.org/abs/1910.03151
    """

    def __init__(
        self,
        channels: int,
        gamma: int = 2,
        b: int = 1,
        kernel_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        # Adaptive kernel size: k = |log2(C) / gamma + b / gamma|_odd
        if kernel_size is None:
            t = int(abs(math.log2(channels) / gamma + b / gamma))
            kernel_size = max(t if t % 2 == 1 else t + 1, 3)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1, 1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: (B, C, H, W) -> (B, C, 1, 1) -> (B, 1, C)
        weights = self.avg_pool(x).squeeze(-1).transpose(-1, -2)

        # 1D convolution across channels: (B, 1, C) -> (B, 1, C)
        weights = self.conv(weights)

        # Sigmoid and reshape: (B, 1, C) -> (B, C, 1, 1)
        weights = self.sigmoid(weights).transpose(-1, -2).unsqueeze(-1)

        return x * weights
