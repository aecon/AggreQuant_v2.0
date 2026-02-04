"""Benchmark configurations for systematic architecture comparison.

This module defines preset configurations for the ModularUNet that allow
systematic A/B testing of different architectural improvements.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04

The configurations are organized as follows:
1. BASELINE: Vanilla UNet as reference
2. SINGLE MODULE ADDITIONS: Test each improvement in isolation
3. COMBINATIONS: Best-performing modules together
4. LIGHTWEIGHT: Resource-constrained deployment

Example:
    >>> from aggrequant.nn.architectures.configs.presets import BENCHMARK_CONFIGS
    >>> config = BENCHMARK_CONFIGS['unet_baseline']
    >>> print(config)
    {'encoder_block': 'double_conv', 'decoder_block': 'double_conv', ...}
"""

from typing import Dict, Any

# Base configuration that all presets inherit from
_BASE_CONFIG: Dict[str, Any] = {
    "in_channels": 1,
    "out_channels": 1,
    "features": [64, 128, 256, 512],
    "encoder_block": "double_conv",
    "decoder_block": "double_conv",
    "bridge_type": "double_conv",
    "use_attention_gates": False,
    "use_se": False,
    "use_cbam": False,
    "use_deep_supervision": False,
    "se_reduction": 16,
    "upsample_mode": "transpose",
}


def _merge_config(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Merge overrides with base configuration."""
    return {**_BASE_CONFIG, **overrides}


# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

BENCHMARK_CONFIGS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # BASELINE - Vanilla UNet (Ronneberger et al., 2015)
    # =========================================================================
    "unet_baseline": _merge_config({
        # Uses all defaults - standard double conv blocks
    }),

    # =========================================================================
    # SINGLE MODULE ADDITIONS - Isolate effect of each improvement
    # =========================================================================

    # ResUNet - Residual connections in encoder/decoder blocks
    # Improves gradient flow, enables deeper networks
    "unet_residual": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
    }),

    # Attention UNet - Attention gates on skip connections
    # Focuses on relevant features, suppresses noise
    "unet_attention": _merge_config({
        "use_attention_gates": True,
    }),

    # SE-UNet - Squeeze-and-Excitation channel attention
    # Recalibrates channel-wise feature responses
    "unet_se": _merge_config({
        "use_se": True,
    }),

    # CBAM-UNet - Channel + Spatial attention
    # Sequential channel and spatial attention refinement
    "unet_cbam": _merge_config({
        "use_cbam": True,
    }),

    # Deep Supervision UNet - Multi-scale auxiliary outputs
    # Improves gradient flow and intermediate representations
    "unet_deep_supervision": _merge_config({
        "use_deep_supervision": True,
    }),

    # ASPP Bridge - Multi-scale context in bottleneck
    # Captures features at multiple receptive field sizes
    "unet_aspp": _merge_config({
        "bridge_type": "aspp",
    }),

    # =========================================================================
    # COMBINATIONS - Best modules together
    # =========================================================================

    # Residual + Attention - Most common combination
    "unet_res_attention": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
    }),

    # Residual + SE + Attention - Triple combination
    "unet_res_se_attention": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
        "use_se": True,
    }),

    # Residual + CBAM + Attention
    "unet_res_cbam_attention": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
        "use_cbam": True,
    }),

    # Full configuration - All modules enabled
    # Maximum feature extraction capability
    "unet_full": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
        "use_se": True,
        "use_deep_supervision": True,
    }),

    # =========================================================================
    # LIGHTWEIGHT - For deployment / resource-constrained environments
    # =========================================================================

    # Lightweight baseline - Reduced feature channels
    "unet_light": _merge_config({
        "features": [32, 64, 128, 256],
    }),

    # Ultra-light - Minimal architecture
    "unet_ultra_light": _merge_config({
        "features": [16, 32, 64, 128],
    }),

    # Light with attention - Balance of efficiency and accuracy
    "unet_light_attention": _merge_config({
        "features": [32, 64, 128, 256],
        "use_attention_gates": True,
    }),

    # =========================================================================
    # DEEPER VARIANTS - More encoder/decoder levels
    # =========================================================================

    # Deep UNet - 5 encoder levels
    "unet_deep": _merge_config({
        "features": [64, 128, 256, 512, 1024],
    }),

    # Deep residual with attention
    "unet_deep_res_attention": _merge_config({
        "features": [64, 128, 256, 512, 1024],
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
    }),

    # =========================================================================
    # BILINEAR UPSAMPLING VARIANTS - Smoother outputs
    # =========================================================================

    "unet_bilinear": _merge_config({
        "upsample_mode": "bilinear",
    }),

    "unet_res_attention_bilinear": _merge_config({
        "encoder_block": "residual",
        "decoder_block": "residual",
        "bridge_type": "residual",
        "use_attention_gates": True,
        "upsample_mode": "bilinear",
    }),
}


def get_config(name: str) -> Dict[str, Any]:
    """Get a benchmark configuration by name.

    Arguments:
        name: Name of the configuration

    Returns:
        Configuration dictionary

    Raises:
        KeyError: If configuration name is not found

    Example:
        >>> config = get_config('unet_baseline')
        >>> print(config['features'])
        [64, 128, 256, 512]
    """
    if name not in BENCHMARK_CONFIGS:
        available = list(BENCHMARK_CONFIGS.keys())
        raise KeyError(
            f"Unknown configuration: '{name}'. "
            f"Available configurations: {available}"
        )
    return BENCHMARK_CONFIGS[name].copy()


def list_configs() -> list:
    """List all available configuration names.

    Returns:
        List of configuration names
    """
    return list(BENCHMARK_CONFIGS.keys())


def get_config_description(name: str) -> str:
    """Get a human-readable description of a configuration.

    Arguments:
        name: Name of the configuration

    Returns:
        Description string
    """
    descriptions = {
        "unet_baseline": "Vanilla UNet with double convolution blocks",
        "unet_residual": "UNet with residual blocks (ResUNet)",
        "unet_attention": "UNet with attention gates on skip connections",
        "unet_se": "UNet with Squeeze-and-Excitation channel attention",
        "unet_cbam": "UNet with CBAM (channel + spatial) attention",
        "unet_deep_supervision": "UNet with deep supervision auxiliary outputs",
        "unet_aspp": "UNet with ASPP multi-scale bridge",
        "unet_res_attention": "Residual UNet with attention gates",
        "unet_res_se_attention": "Residual UNet with SE blocks and attention gates",
        "unet_res_cbam_attention": "Residual UNet with CBAM and attention gates",
        "unet_full": "Full configuration with all modules enabled",
        "unet_light": "Lightweight UNet with reduced feature channels",
        "unet_ultra_light": "Ultra-lightweight UNet for deployment",
        "unet_light_attention": "Lightweight UNet with attention gates",
        "unet_deep": "Deep UNet with 5 encoder levels",
        "unet_deep_res_attention": "Deep residual UNet with attention gates",
        "unet_bilinear": "UNet with bilinear upsampling",
        "unet_res_attention_bilinear": "Residual attention UNet with bilinear upsampling",
    }
    return descriptions.get(name, f"Configuration: {name}")


def print_configs() -> None:
    """Print all available configurations with descriptions."""
    print("Available Benchmark Configurations:")
    print("=" * 60)
    for name in BENCHMARK_CONFIGS:
        desc = get_config_description(name)
        print(f"  {name:30s} - {desc}")
    print("=" * 60)


# Exported configurations for easy access
__all__ = [
    "BENCHMARK_CONFIGS",
    "get_config",
    "list_configs",
    "get_config_description",
    "print_configs",
]
