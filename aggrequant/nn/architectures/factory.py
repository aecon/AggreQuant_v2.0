"""Model registry and factory for creating architectures.

This module provides a simple registry pattern for model selection,
allowing models to be created by name from configuration dictionaries.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04

Example:
    >>> from aggrequant.nn.architectures.factory import create_model, list_architectures
    >>> print(list_architectures())
    ['unet_baseline', 'unet_residual', ...]
    >>> model = create_model('unet_baseline', in_channels=1, out_channels=1)
"""

from typing import Dict, Callable, Optional, Any, List
import torch.nn as nn

# Global registry for architectures
ARCHITECTURES: Dict[str, Callable[..., nn.Module]] = {}


def register(name: str) -> Callable:
    """Decorator to register an architecture in the global registry.

    Arguments:
        name: Name to register the architecture under

    Returns:
        Decorator function

    Example:
        >>> @register("my_custom_unet")
        ... def create_my_unet(**kwargs):
        ...     return ModularUNet(**kwargs)
    """
    def decorator(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in ARCHITECTURES:
            raise ValueError(f"Architecture '{name}' is already registered")
        ARCHITECTURES[name] = fn
        return fn
    return decorator


def create_model(name: str, **kwargs) -> nn.Module:
    """Create a model by name from the registry.

    Arguments:
        name: Registered name of the architecture
        **kwargs: Arguments to pass to the model constructor

    Returns:
        Instantiated model

    Raises:
        ValueError: If architecture name is not found in registry

    Example:
        >>> model = create_model('unet_baseline', in_channels=1, out_channels=1)
        >>> model = create_model('unet_residual', in_channels=3, out_channels=2)
    """
    me = "create_model"

    if name not in ARCHITECTURES:
        available = list(ARCHITECTURES.keys())
        raise ValueError(
            f"({me}) Unknown architecture: '{name}'. "
            f"Available architectures: {available}"
        )

    return ARCHITECTURES[name](**kwargs)


def list_architectures() -> List[str]:
    """List all registered architecture names.

    Returns:
        List of registered architecture names

    Example:
        >>> archs = list_architectures()
        >>> print(archs)
        ['unet_baseline', 'unet_residual', 'unet_attention', ...]
    """
    return list(ARCHITECTURES.keys())


def get_architecture_info(name: str) -> Optional[Dict[str, Any]]:
    """Get information about a registered architecture.

    Arguments:
        name: Name of the architecture

    Returns:
        Dictionary with architecture info, or None if not found
    """
    if name not in ARCHITECTURES:
        return None

    fn = ARCHITECTURES[name]
    return {
        "name": name,
        "factory": fn,
        "docstring": fn.__doc__,
    }


def clear_registry() -> None:
    """Clear all registered architectures.

    This is mainly useful for testing.
    """
    ARCHITECTURES.clear()


# Register the preset architectures from configs
def _register_presets() -> None:
    """Register all preset architectures from configs.presets."""
    from .unet import ModularUNet
    from .configs.presets import BENCHMARK_CONFIGS

    for name, config in BENCHMARK_CONFIGS.items():
        # Create a factory function for each preset
        def make_factory(cfg: dict):
            def factory(**kwargs) -> nn.Module:
                # Merge preset config with user kwargs (user kwargs take precedence)
                merged = {**cfg, **kwargs}
                return ModularUNet(**merged)
            return factory

        # Register the factory
        if name not in ARCHITECTURES:
            ARCHITECTURES[name] = make_factory(config)


# Auto-register presets on module import
try:
    _register_presets()
except ImportError:
    # Presets module not yet available, will be registered later
    pass
