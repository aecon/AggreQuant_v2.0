"""
Evaluation metrics for segmentation.

This module provides metrics commonly used for evaluating segmentation models,
including Dice, IoU, precision, recall, and specificity.

Example:
    >>> from aggrequant.nn.evaluation.metrics import dice_score, iou_score
    >>> dice = dice_score(predictions, targets)
    >>> iou = iou_score(predictions, targets)
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict, List

from aggrequant.common.logging import get_logger
from aggrequant.nn.utils import get_device

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Confusion matrix (single-pass foundation for all metrics)
# ---------------------------------------------------------------------------


def confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = False,
) -> Tuple[int, int, int, int]:
    """Compute confusion matrix components in a single pass.

    Arguments:
        predictions: Model predictions (B, C, H, W)
        targets: Ground truth masks (B, C, H, W)
        threshold: Threshold for binarizing predictions
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Tuple of (TP, TN, FP, FN)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = (predictions > threshold).float()

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    tp = (predictions * targets).sum().int().item()
    tn = ((1 - predictions) * (1 - targets)).sum().int().item()
    fp = (predictions * (1 - targets)).sum().int().item()
    fn = ((1 - predictions) * targets).sum().int().item()

    return tp, tn, fp, fn


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------


def dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute Dice coefficient (equivalent to F1 for binary segmentation).

    Dice = 2 * TP / (2 * TP + FP + FN)

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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    return torch.tensor((2 * tp + smooth) / (2 * tp + fp + fn + smooth))


def soft_dice_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute soft Dice score using probabilities (no binarization).

    Unlike ``dice_score``, this does not threshold predictions. It uses
    the continuous probability values directly, giving a differentiable
    measure of overlap quality.

    Soft Dice = 2 * sum(P * T) / (sum(P) + sum(T))

    Arguments:
        predictions: Model predictions (B, C, H, W), probabilities or logits
        targets: Ground truth masks (B, C, H, W)
        smooth: Smoothing factor to avoid division by zero
        apply_sigmoid: Whether to apply sigmoid to predictions

    Returns:
        Soft Dice score (0-1, higher is better)

    Example:
        >>> preds = torch.rand(4, 1, 128, 128)
        >>> targets = torch.randint(0, 2, (4, 1, 128, 128)).float()
        >>> soft_dice = soft_dice_score(preds, targets)
    """
    if apply_sigmoid:
        predictions = torch.sigmoid(predictions)

    predictions = predictions.view(-1)
    targets = targets.view(-1)

    intersection = (predictions * targets).sum()
    return (2.0 * intersection + smooth) / (predictions.sum() + targets.sum() + smooth)


def iou_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
    apply_sigmoid: bool = False,
) -> torch.Tensor:
    """Compute Intersection over Union (Jaccard Index).

    IoU = TP / (TP + FP + FN)

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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    return torch.tensor((tp + smooth) / (tp + fp + fn + smooth))


def precision_score(
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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    return torch.tensor((tp + smooth) / (tp + fp + smooth))


def recall_score(
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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    return torch.tensor((tp + smooth) / (tp + fn + smooth))


def specificity_score(
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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    return torch.tensor((tn + smooth) / (tn + fp + smooth))


def accuracy_score(
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
    tp, tn, fp, fn = confusion_matrix(predictions, targets, threshold, apply_sigmoid)
    total = tp + tn + fp + fn
    return torch.tensor((tp + tn) / total if total > 0 else 0.0)


# ---------------------------------------------------------------------------
# All-in-one metrics computation
# ---------------------------------------------------------------------------


# Available metrics for SegmentationMetrics and evaluate_model
METRIC_FUNCTIONS = {
    'dice': dice_score,
    'soft_dice': soft_dice_score,
    'iou': iou_score,
    'precision': precision_score,
    'recall': recall_score,
    'specificity': specificity_score,
    'accuracy': accuracy_score,
}

# Default set (excludes soft_dice since it has different semantics)
DEFAULT_METRICS = ['dice', 'iou', 'precision', 'recall', 'specificity', 'accuracy']


class SegmentationMetrics:
    """Compute multiple segmentation metrics in a single pass.

    Uses the confusion matrix (TP, TN, FP, FN) once and derives all
    requested metrics from it.

    Arguments:
        threshold: Threshold for binarizing predictions
        smooth: Smoothing factor for metrics
        metrics: List of metric names to compute (default: all standard metrics).
            Available: 'dice', 'soft_dice', 'iou', 'precision', 'recall',
            'specificity', 'accuracy'.

    Example:
        >>> metrics = SegmentationMetrics(metrics=['dice', 'iou', 'precision'])
        >>> results = metrics(predictions, targets)
        >>> print(results['dice'], results['iou'])
    """

    def __init__(
        self,
        threshold: float = 0.5,
        smooth: float = 1e-6,
        metrics: Optional[List[str]] = None,
    ) -> None:
        self.threshold = threshold
        self.smooth = smooth
        self.metric_names = metrics if metrics is not None else DEFAULT_METRICS

        for name in self.metric_names:
            if name not in METRIC_FUNCTIONS:
                raise ValueError(
                    f"Unknown metric: '{name}'. "
                    f"Available: {list(METRIC_FUNCTIONS.keys())}"
                )

    def __call__(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        apply_sigmoid: bool = False,
    ) -> Dict[str, float]:
        """Compute all requested metrics.

        Arguments:
            predictions: Model predictions
            targets: Ground truth masks
            apply_sigmoid: Whether to apply sigmoid to predictions

        Returns:
            Dictionary mapping metric names to values
        """
        # Compute confusion matrix once for all threshold-based metrics
        tp, tn, fp, fn = confusion_matrix(
            predictions, targets, self.threshold, apply_sigmoid
        )
        s = self.smooth

        # Derive metrics from confusion matrix components
        results = {}
        for name in self.metric_names:
            if name == 'dice':
                results[name] = (2 * tp + s) / (2 * tp + fp + fn + s)
            elif name == 'iou':
                results[name] = (tp + s) / (tp + fp + fn + s)
            elif name == 'precision':
                results[name] = (tp + s) / (tp + fp + s)
            elif name == 'recall':
                results[name] = (tp + s) / (tp + fn + s)
            elif name == 'specificity':
                results[name] = (tn + s) / (tn + fp + s)
            elif name == 'accuracy':
                total = tp + tn + fp + fn
                results[name] = (tp + tn) / total if total > 0 else 0.0
            elif name == 'soft_dice':
                # Soft dice needs probabilities, not confusion matrix
                results[name] = soft_dice_score(
                    predictions, targets, s, apply_sigmoid
                ).item()

        return results


# ---------------------------------------------------------------------------
# Threshold optimization
# ---------------------------------------------------------------------------


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
        metric: Metric to optimize ('dice', 'iou')
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
    }

    if metric not in metric_fns:
        raise ValueError(f"Unknown metric: '{metric}'. Available: {list(metric_fns.keys())}")

    metric_fn = metric_fns[metric]

    best_threshold = 0.5
    best_value = 0.0

    for thresh in thresholds:
        value = metric_fn(predictions, targets, threshold=thresh.item()).item()
        if value > best_value:
            best_value = value
            best_threshold = thresh.item()

    return best_threshold, best_value


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    metrics: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a model on a dataset.

    Arguments:
        model: Model to evaluate
        dataloader: DataLoader with evaluation data
        device: Device to use
        threshold: Threshold for binarizing predictions
        metrics: List of metric names to compute (default: all standard metrics).
            Available: 'dice', 'soft_dice', 'iou', 'precision', 'recall',
            'specificity', 'accuracy'.
        verbose: Print progress

    Returns:
        Dictionary with average metrics

    Example:
        >>> results = evaluate_model(model, test_loader)
        >>> print(f"Test Dice: {results['dice']:.4f}")
        >>>
        >>> # Only compute specific metrics
        >>> results = evaluate_model(model, test_loader, metrics=['dice', 'iou'])
    """
    device = get_device(device)

    model = model.to(device)
    model.eval()

    metrics_computer = SegmentationMetrics(
        threshold=threshold,
        metrics=metrics,
    )

    all_metrics = []

    for batch in dataloader:
        images, masks = batch
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        # Handle deep supervision output
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        batch_metrics = metrics_computer(outputs, masks, apply_sigmoid=True)
        all_metrics.append(batch_metrics)

    # Average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = float(np.mean([m[key] for m in all_metrics]))

    if verbose:
        logger.info("Evaluation Results:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    return avg_metrics
