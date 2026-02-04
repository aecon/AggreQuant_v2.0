"""Neural network module for aggregate segmentation.

This module provides modular neural network architectures, training
infrastructure, and evaluation utilities for aggregate segmentation.

Author: Athena Economides

Submodules:
    - architectures: Modular UNet and building blocks
    - data: Dataset classes and augmentation pipelines
    - training: Loss functions and training loops
    - evaluation: Segmentation metrics

Example:
    >>> from aggrequant.nn.architectures import create_model, list_architectures
    >>> from aggrequant.nn.training import DiceBCELoss, Trainer
    >>> from aggrequant.nn.evaluation import evaluate_model
    >>>
    >>> # Create model
    >>> model = create_model('unet_baseline', in_channels=1, out_channels=1)
    >>>
    >>> # Train
    >>> criterion = DiceBCELoss()
    >>> trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    >>> history = trainer.fit(epochs=100)
    >>>
    >>> # Evaluate
    >>> metrics = evaluate_model(model, test_loader)
"""

# Version
__version__ = "2.0.0"

# Convenience imports from submodules
from .architectures import (
    ModularUNet,
    UNet,
    create_model,
    list_architectures,
    BENCHMARK_CONFIGS,
)
from .training import (
    DiceLoss,
    DiceBCELoss,
    FocalLoss,
    TverskyLoss,
    Trainer,
    train_model,
)
from .evaluation import (
    dice_score,
    iou_score,
    evaluate_model,
)

__all__ = [
    # Version
    "__version__",
    # Architectures
    "ModularUNet",
    "UNet",
    "create_model",
    "list_architectures",
    "BENCHMARK_CONFIGS",
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
]
