"""Neural network module for aggregate segmentation.

This module provides modular UNet architectures for aggregate segmentation.
Training infrastructure and evaluation utilities are available in submodules.

Example:
    >>> from aggrequant.nn import UNet
    >>>
    >>> # Baseline UNet
    >>> model = UNet()
    >>>
    >>> # With residual blocks and attention
    >>> model = UNet(
    ...     encoder_block="residual",
    ...     use_attention_gates=True,
    ... )

Submodules:
    - architectures: UNet with pluggable modules
    - data: Dataset classes and augmentation
    - training: Loss functions and training loops
    - evaluation: Segmentation metrics
"""

# Version
__version__ = "2.0.0"

# Core architecture
from .architectures import ModularUNet, UNet

# Training (optional - for model development)
from .training import (
    DiceLoss,
    DiceBCELoss,
    FocalLoss,
    TverskyLoss,
    Trainer,
    train_model,
)

# Evaluation
from .evaluation import (
    dice_score,
    iou_score,
    evaluate_model,
)

# Utilities
from .utils import get_device

__all__ = [
    # Version
    "__version__",
    # Architectures
    "ModularUNet",
    "UNet",
    # Training
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "TverskyLoss",
    "Trainer",
    "train_model",
    # Evaluation
    "dice_score",
    "iou_score",
    "evaluate_model",
    # Utilities
    "get_device",
]
