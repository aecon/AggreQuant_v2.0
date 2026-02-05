"""Evaluation metrics for segmentation.

This module provides metrics commonly used for evaluating segmentation models,
including Dice, IoU, precision, recall, and F1 score.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04

Example:
    >>> from aggrequant.nn.evaluation.metrics import dice_score, iou_score
    >>> dice = dice_score(predictions, targets)
    >>> iou = iou_score(predictions, targets)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Optional, Tuple

from aggrequant.common.logging import get_logger
from aggrequant.nn.utils import get_device

logger = get_logger(__name__)


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute Dice coefficient (F1 for segmentation).

    Dice = 2 * |X ∩ Y| / (|X| + |Y|)

    Arguments:
        predictions: Model predictions (B, C, H, W), probabilities or logits
        targets: Ground truth masks (B, C, H, W)
        threshold: Threshold for binarizing predictions (default: 0.5)
        smooth: Smoothing factor to avoid division by zero
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Dice score (0-1, higher is better)

    Example:
        >>> preds = torch.rand(4, 1, 128, 128)
        >>> targets = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> dice = dice_score(preds, targets)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    # Binarize predictions
    predictions = (predictions > threshold).float()

    # Flatten spatial dimensions
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Compute Dice
    intersection = (predictions * targets).sum()
    dice = (2.0 * intersection + smooth) / (
        predictions.sum() + targets.sum() + smooth
    )

    return dice


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute Intersection over Union (Jaccard Index).

    IoU = |X ∩ Y| / |X ∪ Y|

    Arguments:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        IoU score (0-1, higher is better)

    Example:
        >>> preds = torch.rand(4, 1, 128, 128)
        >>> targets = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> iou = iou_score(preds, targets)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    # Binarize predictions
    predictions = (predictions > threshold).float()

    # Flatten spatial dimensions
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Compute IoU
    intersection = (predictions * targets).sum()
    union = predictions.sum() + targets.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)

    return iou


def precision(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute precision.

    Precision = TP / (TP + FP)

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Precision score (0-1)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Compute TP and FP
    tp = (predictions * targets).sum()
    fp = (predictions * (1 - targets)).sum()

    precision_val = (tp + smooth) / (tp + fp + smooth)
    return precision_val


def recall(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute recall (sensitivity).

    Recall = TP / (TP + FN)

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Recall score (0-1)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Compute TP and FN
    tp = (predictions * targets).sum()
    fn = ((1 - predictions) * targets).sum()

    recall_val = (tp + smooth) / (tp + fn + smooth)
    return recall_val


def f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute F1 score.

    F1 = 2 * (Precision * Recall) / (Precision + Recall)

    Note: F1 is equivalent to Dice coefficient for binary segmentation.

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        F1 score (0-1)
    """
    prec = precision(predictions, targets, threshold, smooth, apply_sigmoid)
    rec = recall(predictions, targets, threshold, smooth, apply_sigmoid)

    f1 = (2 * prec * rec + smooth) / (prec + rec + smooth)
    return f1


def specificity(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute specificity (true negative rate).

    Specificity = TN / (TN + FP)

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Specificity score (0-1)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    # Compute TN and FP
    tn = ((1 - predictions) * (1 - targets)).sum()
    fp = (predictions * (1 - targets)).sum()

    spec = (tn + smooth) / (tn + fp + smooth)
    return spec


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute pixel-wise accuracy.

    Accuracy = (TP + TN) / (TP + TN + FP + FN)

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Accuracy score (0-1)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    correct = (predictions == targets).float().sum()
    total = targets.numel()

    return correct / total


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> Tuple[int, int, int, int]:
    """Compute confusion matrix components.

    Arguments:
        predictions: Model predictions
        targets: Ground truth masks
        threshold: Threshold for binarizing predictions
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    # Flatten
    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = (predictions * targets).sum().int().item()
    tn = ((1 - predictions) * (1 - targets)).sum().int().item()
    fp = (predictions * (1 - targets)).sum().int().item()
    fn = ((1 - predictions) * targets).sum().int().item()

    return tp, tn, fp, fn


class SegmentationMetrics:
    """Container for computing multiple segmentation metrics.

    Arguments:
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor for metrics

    Example:
        >>> metrics = SegmentationMetrics()
        >>> results = metrics(predictions, targets)
        >>> print(results['dice'], results['iou'])
    """

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-6,
    ) -> None:
        self.threshold = threshold
        self.smooth = smooth

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        apply_sigmoid: bool = False,
    ) -> dict:
        """Compute all metrics.

        Arguments:
            predictions: Model predictions
            targets: Ground truth masks
            apply_sigmoid: Whether to apply sigmoid to predictions

        Returns:
            Dictionary with all metrics
        """
        return {
            'dice': dice_score(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'iou': iou_score(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'precision': precision(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'recall': recall(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'f1': f1_score(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'specificity': specificity(
                predictions, targets, self.threshold, self.smooth, apply_sigmoid
            ).item(),
            'accuracy': accuracy(
                predictions, targets, self.threshold, apply_sigmoid
            ).item(),
        }


def find_optimal_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric: str = 'dice',
    thresholds: Optional[torch.Tensor] = None,
    apply_sigmoid: bool = False,
) -> Tuple[float, float]:
    """Find optimal threshold for a given metric.

    Arguments:
        predictions: Model predictions (probabilities)
        targets: Ground truth masks
        metric: Metric to optimize ('dice', 'iou', 'f1')
        thresholds: Thresholds to try (default: 0.1 to 0.9 in 0.05 steps)
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Tuple of (optimal_threshold, best_metric_value)

    Example:
        >>> optimal_thresh, best_dice = find_optimal_threshold(preds, targets)
    """
    if thresholds is None:
        thresholds = torch.arange(0.1, 0.95, 0.05)

    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    metric_fns = {
        'dice': dice_score,
        'iou': iou_score,
        'f1': f1_score,
    }

    if metric not in metric_fns:
        raise ValueError(f"Unknown metric: {metric}. Available: {list(metric_fns.keys())}")

    metric_fn = metric_fns[metric]

    best_threshold = 0.5
    best_value = 0.0

    for thresh in thresholds:
        value = metric_fn(predictions, targets, threshold=thresh.item()).item()
        if value > best_value:
            best_value = value
            best_threshold = thresh.item()

    return best_threshold, best_value


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """Evaluate a model on a dataset.

    Arguments:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to use
        threshold: Threshold for binarizing predictions
        verbose: Print progress

    Returns:
        Dictionary with average metrics

    Example:
        >>> metrics = evaluate_model(model, test_loader)
        >>> print(f"Test Dice: {metrics['dice']:.4f}")
    """
    device = get_device(device)

    model = model.to(device)
    model.eval()

    metrics_computer = SegmentationMetrics(threshold=threshold)

    all_metrics = []

    for batch in dataloader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # Handle deep supervision output
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        # Compute metrics for this batch
        batch_metrics = metrics_computer(outputs, masks, apply_sigmoid=True)
        all_metrics.append(batch_metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    if verbose:
        logger.info("Evaluation Results:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    return avg_metrics


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
