"""Named model presets for the architecture benchmark.

Each entry maps a variant name to its constructor class and kwargs,
enabling instantiation by name for benchmarking scripts and configs.

Example:
    >>> from aggrequant.nn.architectures.registry import create_model, list_models
    >>>
    >>> model = create_model("baseline")
    >>> model = create_model("se_attention_resunet", features=[32, 64, 128, 256])
    >>>
    >>> for name in list_models():
    ...     print(name)
"""

from typing import Dict, Any, List, Optional

import torch.nn as nn


# ---------------------------------------------------------------------------
# Registry: ablation variants 1-5 (incremental) + 6-7 (structural alternatives)
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    # --- Incremental ablation (each adds one module) ---
    "baseline": {
        "class": "UNet",
        "description": "Baseline UNet (Ronneberger 2015)",
        "kwargs": {
            "encoder_block": "double_conv",
            "decoder_block": "double_conv",
        },
    },
    "resunet": {
        "class": "UNet",
        "description": "ResUNet — +residual blocks (He 2016)",
        "kwargs": {
            "encoder_block": "residual",
            "decoder_block": "residual",
        },
    },
    "attention_resunet": {
        "class": "UNet",
        "description": "Attention ResUNet — +attention gates (Oktay 2018)",
        "kwargs": {
            "encoder_block": "residual",
            "decoder_block": "residual",
            "use_attention_gates": True,
        },
    },
    "se_attention_resunet": {
        "class": "UNet",
        "description": "SE Attention ResUNet — +SE channel attention (Hu 2018)",
        "kwargs": {
            "encoder_block": "residual",
            "decoder_block": "residual",
            "use_attention_gates": True,
            "use_se": True,
        },
    },
    "aspp_se_attention_resunet": {
        "class": "UNet",
        "description": "ASPP SE Attention ResUNet — +multi-scale bridge (Chen 2017)",
        "kwargs": {
            "encoder_block": "residual",
            "decoder_block": "residual",
            "use_attention_gates": True,
            "use_se": True,
            "bridge_type": "aspp",
        },
    },
    # --- Structural alternatives ---
    # UNet++ is deferred (not yet implemented)
    "convnext_unet": {
        "class": "UNet",
        "description": "ConvNeXt UNet — modern encoder blocks (Liu 2022)",
        "kwargs": {
            "encoder_block": "convnext",
            "use_attention_gates": True,
        },
    },
    # --- Side comparisons (not main ablation, but available) ---
    "eca_attention_resunet": {
        "class": "UNet",
        "description": "ECA Attention ResUNet — ECA replaces SE (Wang 2020)",
        "kwargs": {
            "encoder_block": "residual",
            "decoder_block": "residual",
            "use_attention_gates": True,
            "use_eca": True,
        },
    },
}


def create_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    **override_kwargs,
) -> nn.Module:
    """Create a model by registry name.

    Arguments:
        name: Model variant name (see list_models())
        in_channels: Input channels (default: 1 for grayscale)
        out_channels: Output channels (default: 1 for binary segmentation)
        **override_kwargs: Override any default kwargs from the registry

    Returns:
        Instantiated model

    Raises:
        ValueError: If name is not in the registry

    Example:
        >>> model = create_model("baseline")
        >>> model = create_model("resunet", features=[32, 64, 128, 256])
    """
    if name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Unknown model: '{name}'. Available: {available}"
        )

    entry = MODEL_REGISTRY[name]
    class_name = entry["class"]

    # Merge registry defaults with overrides
    kwargs = {
        "in_channels": in_channels,
        "out_channels": out_channels,
        **entry["kwargs"],
        **override_kwargs,
    }

    # Import the appropriate class
    if class_name == "UNet":
        from aggrequant.nn.architectures.unet import UNet
        return UNet(**kwargs)
    else:
        raise ValueError(f"Unknown model class: '{class_name}'")


def list_models() -> List[str]:
    """Return list of available model names.

    Example:
        >>> for name in list_models():
        ...     print(name)
    """
    return list(MODEL_REGISTRY.keys())


def describe_models() -> Dict[str, str]:
    """Return dict mapping model names to descriptions.

    Example:
        >>> for name, desc in describe_models().items():
        ...     print(f"{name}: {desc}")
    """
    return {name: entry["description"] for name, entry in MODEL_REGISTRY.items()}
