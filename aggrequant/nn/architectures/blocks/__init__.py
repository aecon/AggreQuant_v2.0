"""Building blocks for modular UNet architectures.

This module provides pluggable components that can be swapped via configuration
to enable systematic benchmarking of different UNet variants.

Author: Athena Economides, 2026, UZH

Available blocks:

Basic Convolution:
    - SingleConv: Conv2d -> BatchNorm -> ReLU
    - DoubleConv: Two SingleConv blocks (standard UNet block)

Residual:
    - ResidualBlock: Input + Conv path with skip connection
    - BottleneckResidualBlock: 1x1 -> 3x3 -> 1x1 pattern for deeper networks

Attention Gates (Attention U-Net):
    - AttentionGate: Standard attention gate for skip connections
    - MultiHeadAttentionGate: Multi-head variant

Squeeze-and-Excitation:
    - SEBlock: Channel recalibration
    - SEConvBlock: DoubleConv + SE
    - SEResidualBlock: Residual + SE

CBAM (Channel + Spatial Attention):
    - ChannelAttention: Channel-wise attention
    - SpatialAttention: Spatial attention
    - CBAM: Combined channel + spatial
    - CBAMConvBlock: DoubleConv + CBAM
    - CBAMResidualBlock: Residual + CBAM

ASPP (Atrous Spatial Pyramid Pooling):
    - ASPP: Multi-scale context capture
    - ASPPBridge: ASPP for UNet bottleneck
    - LightASPP: Lightweight depthwise separable version

Example:
    >>> from aggrequant.nn.architectures.blocks import DoubleConv, ResidualBlock
    >>> encoder_block = DoubleConv(64, 128)
    >>> res_block = ResidualBlock(128, 256)
"""

# Basic convolution blocks
from .conv import SingleConv, DoubleConv

# Residual blocks
from .residual import ResidualBlock, BottleneckResidualBlock

# Attention gates
from .attention import AttentionGate, MultiHeadAttentionGate

# Squeeze-and-Excitation blocks
from .se import SEBlock, SEConvBlock, SEResidualBlock

# CBAM blocks
from .cbam import (
    ChannelAttention,
    SpatialAttention,
    CBAM,
    CBAMConvBlock,
    CBAMResidualBlock,
)

# ASPP blocks
from .aspp import ASPP, ASPPBridge, ASPPConv, ASPPPooling, LightASPP

__all__ = [
    # Basic convolution
    "SingleConv",
    "DoubleConv",
    # Residual
    "ResidualBlock",
    "BottleneckResidualBlock",
    # Attention gates
    "AttentionGate",
    "MultiHeadAttentionGate",
    # SE blocks
    "SEBlock",
    "SEConvBlock",
    "SEResidualBlock",
    # CBAM blocks
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "CBAMConvBlock",
    "CBAMResidualBlock",
    # ASPP blocks
    "ASPP",
    "ASPPBridge",
    "ASPPConv",
    "ASPPPooling",
    "LightASPP",
]
