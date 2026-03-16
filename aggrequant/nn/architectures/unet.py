"""Modular UNet architecture with pluggable components.

This module provides a configurable UNet implementation that allows systematic
testing of different architectural improvements. Start with baseline UNet
(Ronneberger 2015) and incrementally add modules to test their effect.

Example:
    >>> from aggrequant.nn.architectures.unet import UNet
    >>>
    >>> # Baseline UNet
    >>> model = UNet()
    >>>
    >>> # Add residual connections
    >>> model = UNet(encoder_block="residual", decoder_block="residual")
    >>>
    >>> # Add attention gates
    >>> model = UNet(use_attention_gates=True)
    >>>
    >>> # ConvNeXt encoder with ECA attention
    >>> model = UNet(encoder_block="convnext", use_eca=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

from aggrequant.nn.architectures.blocks.conv import DoubleConv
from aggrequant.nn.architectures.blocks.residual import ResidualBlock
from aggrequant.nn.architectures.blocks.attention import AttentionGate
from aggrequant.nn.architectures.blocks.se import SEBlock
from aggrequant.nn.architectures.blocks.cbam import CBAM
from aggrequant.nn.architectures.blocks.eca import ECABlock
from aggrequant.nn.architectures.blocks.convnext import ConvNeXtBlock
from aggrequant.nn.architectures.blocks.aspp import ASPPBridge


def _make_conv_block(
    in_channels: int,
    out_channels: int,
    block_type: str,
) -> nn.Module:
    """Create a convolution block by type name.

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        block_type: "double_conv", "residual", or "convnext"

    Returns:
        Convolution block module
    """
    if block_type == "residual":
        return ResidualBlock(in_channels, out_channels)
    elif block_type == "convnext":
        return ConvNeXtBlock(in_channels, out_channels)
    else:
        return DoubleConv(in_channels, out_channels)


def _make_channel_attention(
    channels: int,
    use_se: bool = False,
    use_cbam: bool = False,
    use_eca: bool = False,
    se_reduction: int = 16,
) -> Optional[nn.Module]:
    """Create a channel attention module.

    Arguments:
        channels: Number of channels
        use_se: Use Squeeze-and-Excitation
        use_cbam: Use CBAM (channel + spatial)
        use_eca: Use Efficient Channel Attention
        se_reduction: Reduction ratio for SE/CBAM

    Returns:
        Attention module or None
    """
    if use_cbam:
        return CBAM(channels, reduction=se_reduction)
    elif use_se:
        return SEBlock(channels, reduction=se_reduction)
    elif use_eca:
        return ECABlock(channels)
    return None


class EncoderBlock(nn.Module):
    """Encoder block with optional channel attention.

    Combines a convolution block (DoubleConv, ResidualBlock, or ConvNeXtBlock)
    with optional channel attention (SE, CBAM, or ECA).

    Arguments:
        in_channels: Number of input channels
        out_channels: Number of output channels
        block_type: "double_conv", "residual", or "convnext"
        use_se: Whether to add SE block after convolution
        use_cbam: Whether to add CBAM block after convolution
        use_eca: Whether to add ECA block after convolution
        se_reduction: Reduction ratio for SE/CBAM (default: 16)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        block_type: str = "double_conv",
        use_se: bool = False,
        use_cbam: bool = False,
        use_eca: bool = False,
        se_reduction: int = 16,
    ) -> None:
        super().__init__()
        self.conv = _make_conv_block(in_channels, out_channels, block_type)
        self.attention = _make_channel_attention(
            out_channels, use_se=use_se, use_cbam=use_cbam,
            use_eca=use_eca, se_reduction=se_reduction,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder block."""
        x = self.conv(x)
        if self.attention is not None:
            x = self.attention(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder block with upsampling and skip connection.

    Combines upsampling, skip connection concatenation, and convolution
    with optional attention gate and channel attention (SE/CBAM/ECA).

    Arguments:
        in_channels: Number of input channels from previous decoder level
        skip_channels: Number of channels from skip connection
        out_channels: Number of output channels
        block_type: "double_conv", "residual", or "convnext"
        use_attention_gate: Whether to apply attention gate to skip connection
        use_se: Whether to add SE block after convolution
        use_cbam: Whether to add CBAM block after convolution
        use_eca: Whether to add ECA block after convolution
        se_reduction: Reduction ratio for SE/CBAM (default: 16)
        upsample_mode: Upsampling mode ("transpose" or "bilinear")
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
        use_eca: bool = False,
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
        self.conv = _make_conv_block(
            up_channels + skip_channels, out_channels, block_type,
        )

        # Optional channel attention
        self.attention = _make_channel_attention(
            out_channels, use_se=use_se, use_cbam=use_cbam,
            use_eca=use_eca, se_reduction=se_reduction,
        )

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


class UNet(nn.Module):
    """UNet with optional architectural improvements for benchmarking.

    Start with baseline UNet (Ronneberger 2015) and incrementally add modules
    to test their effect on segmentation performance.

    Arguments:
        in_channels: Input channels (default: 1 for grayscale microscopy)
        out_channels: Output channels (default: 1 for binary segmentation)
        features: Channel sizes per encoder level (default: [64, 128, 256, 512])
        encoder_block: "double_conv" (default), "residual", or "convnext"
        decoder_block: "double_conv" (default), "residual", or "convnext"
        bridge_type: "double_conv" (default), "residual", or "aspp"
        use_attention_gates: Add attention gates on skip connections
        use_se: Add Squeeze-and-Excitation channel attention
        use_cbam: Add CBAM attention (mutually exclusive with use_se/use_eca)
        use_eca: Add Efficient Channel Attention (mutually exclusive with use_se/use_cbam)
        use_deep_supervision: Return multi-scale outputs for training
        se_reduction: Reduction ratio for SE/CBAM (default: 16)
        upsample_mode: "transpose" (default) or "bilinear"

    Returns:
        If use_deep_supervision and training: tuple of (main_output, [aux_outputs])
        Otherwise: output tensor of shape (B, out_channels, H, W)

    References:
        - Ronneberger et al., "U-Net" (2015) - baseline architecture
        - He et al., "Deep Residual Learning" (2016) - residual blocks
        - Oktay et al., "Attention U-Net" (2018) - attention gates
        - Hu et al., "Squeeze-and-Excitation Networks" (2018) - SE blocks
        - Woo et al., "CBAM" (2018) - channel + spatial attention
        - Wang et al., "ECA-Net" (2020) - efficient channel attention
        - Chen et al., "DeepLab" (2017) - ASPP dilated convolutions
        - Liu et al., "A ConvNet for the 2020s" (2022) - ConvNeXt blocks
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
        use_eca: bool = False,
        use_deep_supervision: bool = False,
        se_reduction: int = 16,
        upsample_mode: str = "transpose",
    ) -> None:
        super().__init__()

        # Default features if not provided
        if features is None:
            features = [64, 128, 256, 512]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.encoder_block = encoder_block
        self.decoder_block = decoder_block
        self.bridge_type = bridge_type
        self.use_attention_gates = use_attention_gates
        self.use_se = use_se
        self.use_cbam = use_cbam
        self.use_eca = use_eca
        self.use_deep_supervision = use_deep_supervision
        self.se_reduction = se_reduction
        self.upsample_mode = upsample_mode

        # Validate configuration: at most one channel attention type
        attn_count = sum([use_se, use_cbam, use_eca])
        if attn_count > 1:
            raise ValueError(
                "use_se, use_cbam, and use_eca are mutually exclusive"
            )

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
                    use_eca=use_eca,
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
                    use_eca=use_eca,
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
        """Return model configuration as dictionary.

        The returned dict can be passed directly to UNet(**config) to
        reconstruct the same architecture (before loading weights).
        """
        return {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "features": self.features,
            "encoder_block": self.encoder_block,
            "decoder_block": self.decoder_block,
            "bridge_type": self.bridge_type,
            "use_attention_gates": self.use_attention_gates,
            "use_se": self.use_se,
            "use_cbam": self.use_cbam,
            "use_eca": self.use_eca,
            "use_deep_supervision": self.use_deep_supervision,
            "se_reduction": self.se_reduction,
            "upsample_mode": self.upsample_mode,
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
