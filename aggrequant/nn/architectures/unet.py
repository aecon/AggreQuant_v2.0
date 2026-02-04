"""Modular UNet architecture with pluggable components.

This module provides a configurable UNet implementation that allows systematic
benchmarking of different architectural improvements (residual blocks, attention
gates, SE blocks, CBAM, deep supervision).

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04

Example:
    >>> from aggrequant.nn.architectures.unet import ModularUNet
    >>> model = ModularUNet(
    ...     in_channels=1,
    ...     out_channels=1,
    ...     encoder_block="residual",
    ...     use_attention_gates=True,
    ... )
    >>> x = torch.randn(1, 1, 256, 256)
    >>> out = model(x)
    >>> out.shape
    torch.Size([1, 1, 256, 256])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from .blocks import (
    DoubleConv,
    ResidualBlock,
    AttentionGate,
    SEBlock,
    CBAM,
    ASPPBridge,
)


class EncoderBlock(nn.Module):
    """Encoder block with optional SE/CBAM attention.

    Combines a convolution block (DoubleConv or ResidualBlock) with
    optional channel attention (SE or CBAM).

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        block_type: Type of convolution block ("double_conv" or "residual")
        use_se: Whether to add SE block after convolution
        use_cbam: Whether to add CBAM block after convolution
        se_reduction: Reduction ratio for SE/CBAM (default: 16)

    Returns:
        Tensor of shape (B, out_channels, H, W)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_type: str = "double_conv",
        use_se: bool = False,
        use_cbam: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()

        # Select convolution block type
        if block_type == "residual":
            self.conv = ResidualBlock(in_channels, out_channels)
        else:
            self.conv = DoubleConv(in_channels, out_channels)

        # Optional attention modules (mutually exclusive)
        self.attention = None
        if use_cbam:
            self.attention = CBAM(out_channels, reduction=se_reduction)
        elif use_se:
            self.attention = SEBlock(out_channels, reduction=se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder block."""
        x = self.conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection.

    Combines upsampling, skip connection concatenation, and convolution
    with optional attention gate and SE/CBAM.

    Arguments:
        in_channels: Number of input channels from previous decoder level
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        block_type: Type of convolution block ("double_conv" or "residual")
        use_attention_gate: Whether to apply attention gate to skip connection
        use_se: Whether to add SE block after convolution
        use_cbam: Whether to add CBAM block after convolution
        se_reduction: Reduction ratio for SE/CBAM (default: 16)
        upsample_mode: Upsampling mode ("transpose" or "bilinear")

    Returns:
        Tensor of shape (B, out_channels, H*2, W*2)
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        block_type: str = "double_conv",
        use_attention_gate: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
        se_reduction: int = 16,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()

        # Upsampling
        if upsample_mode == "transpose":
            self.upsample = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            up_channels = in_channels // 2
        else:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            )
            up_channels = in_channels // 2

        # Attention gate for skip connection
        self.attention_gate = None
        if use_attention_gate:
            self.attention_gate = AttentionGate(
                gate_channels=up_channels,
                skip_channels=skip_channels,
            )

        # Convolution block after concatenation
        if block_type == "residual":
            self.conv = ResidualBlock(up_channels + skip_channels, out_channels)
        else:
            self.conv = DoubleConv(up_channels + skip_channels, out_channels)

        # Optional attention modules
        self.attention = None
        if use_cbam:
            self.attention = CBAM(out_channels, reduction=se_reduction)
        elif use_se:
            self.attention = SEBlock(out_channels, reduction=se_reduction)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through decoder block.

        Arguments:
            x: Input tensor from previous decoder level
            skip: Skip connection tensor from encoder

        Returns:
            Output tensor after upsampling, concatenation, and convolution
        """
        # Upsample
        x = self.upsample(x)

        # Handle size mismatch due to odd input dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        # Apply attention gate to skip connection if enabled
        if self.attention_gate is not None:
            skip = self.attention_gate(x, skip)

        # Concatenate and convolve
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)

        # Apply channel attention if enabled
        if self.attention is not None:
            x = self.attention(x)

        return x


class ModularUNet(nn.Module):
    """Configurable UNet with pluggable modules for benchmarking.

    This implementation allows systematic A/B testing of individual architectural
    improvements by enabling/disabling specific modules via configuration.

    Arguments:
        in_channels: Number of input channels (default: 1 for grayscale)
        out_channels: Number of output channels (default: 1 for binary segmentation)
        features: List of feature counts per encoder level (default: [64, 128, 256, 512])
        encoder_block: Block type for encoder ("double_conv" or "residual")
        decoder_block: Block type for decoder ("double_conv" or "residual")
        bridge_type: Bridge/bottleneck type ("double_conv", "residual", or "aspp")
        use_attention_gates: Add attention gates to skip connections
        use_se: Add SE blocks after conv blocks
        use_cbam: Add CBAM blocks (mutually exclusive with SE)
        use_deep_supervision: Return multi-scale outputs for deep supervision loss
        se_reduction: Reduction ratio for SE/CBAM blocks (default: 16)
        upsample_mode: Upsampling mode ("transpose" or "bilinear")

    Returns:
        During training with deep_supervision: tuple of (main_output, [deep_outputs])
        Otherwise: main output tensor of shape (B, out_channels, H, W)

    Example:
        >>> model = ModularUNet(
        ...     in_channels=1,
        ...     out_channels=1,
        ...     encoder_block="residual",
        ...     use_attention_gates=True,
        ...     use_se=True,
        ... )
        >>> x = torch.randn(2, 1, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([2, 1, 256, 256])

    References:
        - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
          Image Segmentation" (2015)
        - Oktay et al., "Attention U-Net" (2018)
        - He et al., "Deep Residual Learning" (2016)
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: Optional[List[int]] = None,
        encoder_block: str = "double_conv",
        decoder_block: str = "double_conv",
        bridge_type: str = "double_conv",
        use_attention_gates: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
        use_deep_supervision: bool = False,
        se_reduction: int = 16,
        upsample_mode: str = "transpose",
        **kwargs,  # Accept additional kwargs for compatibility
    ) -> None:
        super().__init__()

        me = "ModularUNet.__init__"

        # Default features if not provided
        if features is None:
            features = [64, 128, 256, 512]

        self.features = features
        self.use_deep_supervision = use_deep_supervision
        self.use_attention_gates = use_attention_gates

        # Validate configuration
        if use_se and use_cbam:
            raise ValueError(f"({me}) use_se and use_cbam are mutually exclusive")

        # Build encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        in_ch = in_channels
        for feature in features:
            self.encoders.append(
                EncoderBlock(
                    in_channels=in_ch,
                    out_channels=feature,
                    block_type=encoder_block,
                    use_se=use_se,
                    use_cbam=use_cbam,
                    se_reduction=se_reduction,
                )
            )
            self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_ch = feature

        # Build bridge/bottleneck
        bridge_in = features[-1]
        bridge_out = features[-1] * 2

        if bridge_type == "aspp":
            self.bridge = ASPPBridge(bridge_in, bridge_out)
        elif bridge_type == "residual":
            self.bridge = ResidualBlock(bridge_in, bridge_out)
        else:
            self.bridge = DoubleConv(bridge_in, bridge_out)

        # Build decoder path
        self.decoders = nn.ModuleList()

        reversed_features = list(reversed(features))
        in_ch = bridge_out

        for i, feature in enumerate(reversed_features):
            self.decoders.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=feature,
                    out_channels=feature,
                    block_type=decoder_block,
                    use_attention_gate=use_attention_gates,
                    use_se=use_se,
                    use_cbam=use_cbam,
                    se_reduction=se_reduction,
                    upsample_mode=upsample_mode,
                )
            )
            in_ch = feature

        # Final output convolution
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Deep supervision heads (for intermediate outputs)
        if use_deep_supervision:
            # Create output heads for each decoder level except the last
            self.deep_supervision_heads = nn.ModuleList()
            for feature in reversed_features[:-1]:  # Skip last (finest) level
                self.deep_supervision_heads.append(
                    nn.Conv2d(feature, out_channels, kernel_size=1)
                )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Forward pass through the UNet.

        Arguments:
            x: Input tensor of shape (B, in_channels, H, W)

        Returns:
            If use_deep_supervision and training:
                Tuple of (main_output, [deep_outputs])
            Otherwise:
                Main output tensor of shape (B, out_channels, H, W)
        """
        # Encoder path - collect skip connections
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bridge
        x = self.bridge(x)

        # Decoder path - use skip connections in reverse order
        skip_connections = skip_connections[::-1]
        deep_outputs = []

        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])

            # Collect deep supervision outputs
            if self.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_outputs.append(self.deep_supervision_heads[i](x))

        # Final output
        output = self.final_conv(x)

        # Return with deep supervision outputs during training
        if self.use_deep_supervision and self.training:
            return output, deep_outputs

        return output

    def get_config(self) -> dict:
        """Return model configuration as dictionary."""
        return {
            "in_channels": self.encoders[0].conv.double_conv[0].conv.in_channels
            if hasattr(self.encoders[0].conv, "double_conv")
            else self.encoders[0].conv.conv1[0].in_channels,
            "features": self.features,
            "use_deep_supervision": self.use_deep_supervision,
            "use_attention_gates": self.use_attention_gates,
        }

    def count_parameters(self) -> int:
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation with parameter count."""
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__}("
            f"features={self.features}, "
            f"params={params:,}, "
            f"deep_supervision={self.use_deep_supervision}, "
            f"attention_gates={self.use_attention_gates})"
        )


# Alias for backward compatibility
UNet = ModularUNet
