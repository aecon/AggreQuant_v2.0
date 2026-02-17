"""UNet architectures for aggregate segmentation.

This module provides a modular UNet that can be configured to test different
architectural improvements. Start with baseline and add modules incrementally.

Example:
    >>> from aggrequant.nn.architectures import UNet
    >>>
    >>> # Baseline UNet (Ronneberger 2015)
    >>> model = UNet()
    >>>
    >>> # Add residual blocks
    >>> model = UNet(encoder_block="residual", decoder_block="residual")
    >>>
    >>> # Add attention gates
    >>> model = UNet(use_attention_gates=True)
    >>>
    >>> # Add SE channel attention
    >>> model = UNet(use_se=True)
    >>>
    >>> # Add ASPP bridge (dilated convolutions)
    >>> model = UNet(bridge_type="aspp")
    >>>
    >>> # Combine modules for A/B testing
    >>> model = UNet(
    ...     encoder_block="residual",
    ...     use_attention_gates=True,
    ...     use_se=True,
    ... )

Available modules to test:
    - encoder_block/decoder_block: "double_conv" or "residual"
    - bridge_type: "double_conv", "residual", or "aspp"
    - use_attention_gates: Attention on skip connections
    - use_se: Squeeze-and-Excitation channel attention
    - use_cbam: Channel + spatial attention (CBAM)
    - use_deep_supervision: Multi-scale auxiliary outputs
"""

from aggrequant.nn.architectures.unet import ModularUNet, UNet

__all__ = [
    "ModularUNet",
    "UNet",
]
