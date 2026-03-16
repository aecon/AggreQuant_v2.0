"""
Atrous Spatial Pyramid Pooling (ASPP) for multi-scale context.

This module implements ASPP as described in:
"Rethinking Atrous Convolution for Semantic Image Segmentation" (Chen et al., 2017)

ASPP captures multi-scale context by applying parallel atrous (dilated)
convolutions with different dilation rates, useful for the bridge/bottleneck
of UNet architectures.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ASPPConv(nn.Module):
    """Single ASPP convolution branch with specified dilation.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilation: Dilation rate for atrous convolution

    Returns:
        Tensor of shape (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ASPPPooling(nn.Module):
    """Global average pooling branch for ASPP.

    Captures image-level features by pooling to 1x1 and broadcasting back.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels

    Returns:
        Tensor of shape (B, out_channels, H, W) (upsampled to match input)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[2:]
        x = self.pool(x)
        x = self.conv(x)
        return nn.functional.interpolate(
            x, size=size, mode="bilinear", align_corners=False
        )


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling module.

    Applies parallel branches with different dilation rates to capture
    multi-scale context:
    1. 1x1 convolution (no dilation)
    2. 3x3 convolutions with various dilation rates
    3. Global average pooling (image-level features)

    All branches are concatenated and projected to output channels.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels (default: same as in_channels)
        dilations: Tuple of dilation rates (default: (6, 12, 18) for DeepLabV3)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> aspp = ASPP(512, 256)
        >>> x = torch.randn(1, 512, 16, 16)
        >>> out = aspp(x)
        >>> out.shape
        torch.Size([1, 256, 16, 16])

    References:
        Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation"
        https://arxiv.org/abs/1706.05587
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dilations: Tuple[int, ...] = (6, 12, 18),
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        # Internal channel count for each branch
        branch_channels = out_channels // (len(dilations) + 2)
        # Adjust for rounding
        branch_channels = max(branch_channels, 1)

        modules = []

        # 1x1 convolution branch (no dilation)
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Dilated convolution branches
        for dilation in dilations:
            modules.append(ASPPConv(in_channels, branch_channels, dilation))

        # Global pooling branch
        modules.append(ASPPPooling(in_channels, branch_channels))

        self.convs = nn.ModuleList(modules)

        # Calculate total channels from all branches
        total_channels = branch_channels * len(modules)

        # Project concatenated features to output channels
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all branches
        branch_outputs = [conv(x) for conv in self.convs]

        # Concatenate along channel dimension
        concatenated = torch.cat(branch_outputs, dim=1)

        # Project to output channels
        return self.project(concatenated)


class ASPPBridge(nn.Module):
    """ASPP-based bridge for UNet bottleneck.

    Designed as a drop-in replacement for the standard DoubleConv bridge,
    incorporating multi-scale context capture via ASPP.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilations: Dilation rates for ASPP (default: (6, 12, 18))

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> bridge = ASPPBridge(512, 1024)
        >>> x = torch.randn(1, 512, 8, 8)
        >>> out = bridge(x)
        >>> out.shape
        torch.Size([1, 1024, 8, 8])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dilations: Tuple[int, ...] = (6, 12, 18),
    ) -> None:
        super().__init__()

        # For small feature maps, reduce dilations
        self.aspp = ASPP(in_channels, out_channels, dilations)

        # Additional refinement convolution
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ASPP bridge.

        Arguments:
            x: Input tensor from encoder bottom

        Returns:
            Multi-scale context features
        """
        x = self.aspp(x)
        x = self.refine(x)
        return x


class LightASPP(nn.Module):
    """Lightweight ASPP with depthwise separable convolutions.

    Memory-efficient version of ASPP using depthwise separable convolutions,
    suitable for resource-constrained environments.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        dilations: Dilation rates (default: (3, 6, 9) - reduced for efficiency)

    Returns:
        Tensor of shape (B, out_channels, H, W)

    Example:
        >>> aspp = LightASPP(256, 256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> out = aspp(x)
        >>> out.shape
        torch.Size([1, 256, 32, 32])
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        dilations: Tuple[int, ...] = (3, 6, 9),
    ) -> None:
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        branch_channels = out_channels // (len(dilations) + 1)
        branch_channels = max(branch_channels, 1)

        modules = []

        # 1x1 convolution branch
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
        )

        # Depthwise separable dilated convolution branches
        for dilation in dilations:
            modules.append(
                nn.Sequential(
                    # Depthwise
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        groups=in_channels,
                        bias=False,
                    ),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(inplace=True),
                    # Pointwise
                    nn.Conv2d(in_channels, branch_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )

        self.convs = nn.ModuleList(modules)

        total_channels = branch_channels * len(modules)

        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [conv(x) for conv in self.convs]
        concatenated = torch.cat(branch_outputs, dim=1)
        return self.project(concatenated)
