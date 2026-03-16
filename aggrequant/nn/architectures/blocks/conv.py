"""
Basic convolution blocks for UNet architectures.

This module provides the fundamental building blocks used in the encoder
and decoder paths of UNet-style networks.
"""

import torch
import torch.nn as nn
from typing import Optional


class SingleConv(nn.Module):
    """Single convolution block: Conv2d -> BatchNorm -> ReLU.

    This is the atomic unit for building UNet encoder/decoder blocks.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size (default: 3)
        padding: Padding for convolution (default: 1 for same padding)

    Returns:
        Tensor of shape (B, out_channels, H, W)
            where B: batch size
            H, W: height and width of image
        
    Example:
        >>> conv = SingleConv(64, 128)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> out = conv(x)
        >>> out.shape
        torch.Size([1, 128, 128, 128])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1, # adds a 1-pixel border of zeros so output H×W stays same as input
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False, # redundant before BatchNorm (which has its own learnable bias via affine=True)
            ),
            nn.BatchNorm2d(out_channels, affine=True), # mean≈0, std≈1 across batch; affine adds learnable scale+bias
            nn.ReLU(inplace=True), # modifies the tensor in-place to save memory
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through single convolution block.

        Arguments:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        return self.conv(x)


class DoubleConv(nn.Module):
    """Double convolution block: two consecutive SingleConv blocks.

    This is the standard UNet block (Ronneberger et al., 2015) consisting
    of two 3x3 convolutions, each followed by BatchNorm and ReLU.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        mid_channels: Number of channels between the two convolutions
            (default: same as out_channels)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = DoubleConv(64, 128)
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

        self.double_conv = nn.Sequential(
            SingleConv(in_channels, mid_channels),
            SingleConv(mid_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)
