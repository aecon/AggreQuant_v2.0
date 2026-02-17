"""
Evaluation metrics and utilities for segmentation models.

This module provides metrics for evaluating segmentation performance,
including Dice, IoU, precision, recall, and F1 score.

Example:
    >>> from aggrequant.nn.evaluation import dice_score, evaluate_model
    >>> dice = dice_score(predictions, targets)
    >>> metrics = evaluate_model(model, test_loader)
"""

from aggrequant.nn.evaluation.metrics import (
    dice_score,
    iou_score,
    precision,
    recall,
    f1_score,
    specificity,
    accuracy,
    confusion_matrix,
    SegmentationMetrics,
    find_optimal_threshold,
    evaluate_model,
)

__all__ = [
    "dice_score",
    "iou_score",
    "precision",
    "recall",
    "f1_score",
    "specificity",
    "accuracy",
    "confusion_matrix",
    "SegmentationMetrics",
    "find_optimal_threshold",
    "evaluate_model",
]
