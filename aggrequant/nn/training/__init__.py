"""Training infrastructure for neural networks.

This module provides loss functions and training loops for
segmentation model training.

Author: Athena Economides, 2026, UZH

Example:
    >>> from aggrequant.nn.training import DiceBCELoss, Trainer
    >>> criterion = DiceBCELoss()
    >>> trainer = Trainer(model, train_loader, val_loader, criterion, optimizer)
    >>> history = trainer.fit(epochs=100)
"""

from aggrequant.nn.training.losses import (
    DiceLoss,
    DiceBCELoss,
    FocalLoss,
    TverskyLoss,
    FocalTverskyLoss,
    DeepSupervisionLoss,
    BoundaryLoss,
    get_loss_function,
)
from aggrequant.nn.training.trainer import (
    Trainer,
    TrainingHistory,
    train_model,
)

__all__ = [
    # Loss functions
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "DeepSupervisionLoss",
    "BoundaryLoss",
    "get_loss_function",
    # Training
    "Trainer",
    "TrainingHistory",
    "train_model",
]
