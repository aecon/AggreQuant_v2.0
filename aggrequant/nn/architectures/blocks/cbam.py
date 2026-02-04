"""Convolutional Block Attention Module (CBAM).

This module implements CBAM as described in:
"CBAM: Convolutional Block Attention Module" (Woo et al., 2018)

CBAM sequentially applies channel attention and spatial attention to
refine feature maps along both dimensions.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Channel attention module for CBAM.

    Computes channel-wise attention using both average-pooled and max-pooled
    features through a shared MLP, then combines them.

    Arguments:
        channels: Number of input/output channels
        reduction: Channel reduction ratio for MLP (default: 16)

    Returns:
        Channel attention weights of shape (B, channels, 1, 1)

    Example:
        >>> ca = ChannelAttention(256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> weights = ca(x)
        >>> weights.shape
        torch.Size([1, 256, 1, 1])

    References:
        Woo et al., "CBAM: Convolutional Block Attention Module"
        https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
    ) -> None:
        super().__init__()

        reduced_channels = max(channels // reduction, 1)

        # Shared MLP for both pooled features
        self.mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights.

        Arguments:
            x: Input tensor of shape (B, channels, H, W)

        Returns:
            Attention weights of shape (B, channels, 1, 1)
        """
        batch_size, channels, _, _ = x.shape

        # Average pooling path
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_out)

        # Max pooling path
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_out)

        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out)
        return attention.view(batch_size, channels, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module for CBAM.

    Computes spatial attention by applying convolution to channel-wise
    statistics (max and average across channels).

    Arguments:
        kernel_size: Convolution kernel size (default: 7)

    Returns:
        Spatial attention weights of shape (B, 1, H, W)

    Example:
        >>> sa = SpatialAttention()
        >>> x = torch.randn(1, 256, 64, 64)
        >>> weights = sa(x)
        >>> weights.shape
        torch.Size([1, 1, 64, 64])
    """

    def __init__(
        self,
        kernel_size: int = 7,
    ) -> None:
        super().__init__()

        assert kernel_size in (3, 7), "kernel_size must be 3 or 7"
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(
            2,  # avg + max channels
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights.

        Arguments:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention weights of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)

        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = self.conv(combined)

        return self.sigmoid(attention)


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Sequential application of channel attention followed by spatial attention.
    Each attention refines the features independently.

    Arguments:
        channels: Number of input/output channels
        reduction: Channel reduction ratio for channel attention (default: 16)
        spatial_kernel_size: Kernel size for spatial attention conv (default: 7)

    Returns:
        Attention-refined features of shape (B, channels, H, W)

    Example:
        >>> cbam = CBAM(256)
        >>> x = torch.randn(1, 256, 64, 64)
        >>> out = cbam(x)
        >>> out.shape
        torch.Size([1, 256, 64, 64])

    References:
        Woo et al., "CBAM: Convolutional Block Attention Module"
        https://arxiv.org/abs/1807.06521
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()

        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply CBAM attention to input features.

        Arguments:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Attention-refined tensor of shape (B, C, H, W)
        """
        # Apply channel attention
        ca_weights = self.channel_attention(x)
        x = x * ca_weights

        # Apply spatial attention
        sa_weights = self.spatial_attention(x)
        x = x * sa_weights

        return x


class CBAMConvBlock(nn.Module):
    """Convolution block with integrated CBAM attention.

    Combines double convolution with CBAM for enhanced feature learning.
    CBAM is applied after the convolutions.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: CBAM channel reduction ratio (default: 16)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = CBAMConvBlock(64, 128)
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.cbam = CBAM(out_channels, reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBAM convolution block.

        Arguments:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)
        return out


class CBAMResidualBlock(nn.Module):
    """Residual block with CBAM attention.

    Combines residual connections with CBAM attention.
    CBAM is applied at the end of the residual path before adding the skip.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        reduction: CBAM channel reduction ratio (default: 16)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> block = CBAMResidualBlock(64, 128)
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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.cbam = CBAM(out_channels, reduction)

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CBAM residual block.

        Arguments:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            Output tensor of shape (B, out_channels, H, W)
        """
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cbam(out)

        out = out + identity
        out = self.relu(out)

        return out
