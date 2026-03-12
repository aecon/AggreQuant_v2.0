"""
ConvNeXt block for modernized UNet encoder/decoder.

This module implements the ConvNeXt block as described in:
"A ConvNet for the 2020s" (Liu et al., 2022)

ConvNeXt modernizes standard convolution blocks by borrowing design
choices from Vision Transformers: large depthwise kernels, LayerNorm,
GELU activation, and inverted bottleneck.
"""

import torch
import torch.nn as nn


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W).

    Standard nn.LayerNorm operates on the last dimension, so we permute
    to (B, H, W, C), normalize, and permute back.

    Arguments:
        channels: Number of channels to normalize

    References:
        Liu et al., "A ConvNet for the 2020s" (2022)
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: modernized convolution block.

    Architecture:
        output = x + gamma * Linear_up(GELU(Linear_down(LayerNorm(DWConv7x7(x)))))

    Key design choices vs ResidualBlock:
    - 7x7 depthwise conv: larger receptive field, fewer parameters
    - LayerNorm instead of BatchNorm: more stable for small batches
    - GELU instead of ReLU: smoother gradients
    - Inverted bottleneck: expand channels (C -> 4C) then squeeze back
    - LayerScale: learnable per-channel residual scaling

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        expansion: Bottleneck expansion ratio (default: 4)
        kernel_size: Depthwise convolution kernel size (default: 7)
        layer_scale_init: Initial value for layer scale (default: 1e-6)

    Example:
        >>> block = ConvNeXtBlock(64, 128)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = block(x)
        >>> out.shape
        torch.Size([1, 128, 128, 128])

    References:
        Liu et al., "A ConvNet for the 2020s"
        https://arxiv.org/abs/2201.03545
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 4,
        kernel_size: int = 7,
        layer_scale_init: float = 1e-6,
    ) -> None:
        super().__init__()

        mid_channels = out_channels * expansion
        padding = kernel_size // 2

        # Depthwise convolution (spatial mixing, channel-independent)
        self.dwconv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_channels,
            bias=True,
        )

        self.norm = LayerNorm2d(out_channels)

        # Inverted bottleneck: expand then squeeze
        self.pwconv1 = nn.Conv2d(out_channels, mid_channels, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=1)

        # LayerScale: learnable per-channel scaling of the residual
        self.layer_scale = nn.Parameter(
            layer_scale_init * torch.ones(out_channels, 1, 1)
        )

        # Channel projection for skip connection if needed
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ConvNeXt block.

        Arguments:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        identity = self.skip(x)

        out = self.dwconv(identity)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        out = self.layer_scale * out

        return identity + out
