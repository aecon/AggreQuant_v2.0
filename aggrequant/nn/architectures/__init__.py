"""Neural network architectures for aggregate segmentation.

This module provides modular UNet architectures with pluggable components
for systematic benchmarking of different architectural improvements.

Author: Athena Economides

Example:
    >>> from aggrequant.nn.architectures import create_model, list_architectures
    >>> print(list_architectures())
    ['unet_baseline', 'unet_residual', 'unet_attention', ...]
    >>> model = create_model('unet_baseline', in_channels=1, out_channels=1)
"""

from .unet import ModularUNet, UNet
from .factory import (
    create_model,
    list_architectures,
    register,
    get_architecture_info,
    ARCHITECTURES,
)
from .configs import (
    BENCHMARK_CONFIGS,
    get_config,
    list_configs,
    get_config_description,
    print_configs,
)

__all__ = [
    # UNet classes
    "ModularUNet",
    "UNet",
    # Factory functions
    "create_model",
    "list_architectures",
    "register",
    "get_architecture_info",
    "ARCHITECTURES",
    # Config functions
    "BENCHMARK_CONFIGS",
    "get_config",
    "list_configs",
    "get_config_description",
    "print_configs",
]
