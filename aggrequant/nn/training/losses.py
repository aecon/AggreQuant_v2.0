"""Loss functions for segmentation training.

This module provides loss functions commonly used for binary and multi-class
segmentation, including support for deep supervision.

Author: Athena Economides

Example:
    >>> from aggrequant.nn.training.losses import DiceBCELoss
    >>> criterion = DiceBCELoss()
    >>> loss = criterion(predictions, targets)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation.

    Dice loss = 1 - Dice coefficient = 1 - 2*|X ∩ Y| / (|X| + |Y|)

    Arguments:
        smooth: Smoothing factor to avoid division by zero (default: 1.0)
        sigmoid: Whether to apply sigmoid to predictions (default: True)

    Example:
        >>> criterion = DiceLoss()
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Dice loss.

        Arguments:
            predictions: Model predictions (B, C, H, W) or (B, 1, H, W)
            targets: Ground truth masks (B, C, H, W) or (B, 1, H, W)

        Returns:
            Dice loss value
        """
        if self.sigmoid:
            predictions = torch.sigmoid(predictions)

        # Flatten tensors
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute Dice coefficient
        intersection = (predictions * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )

        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Combined Dice and Binary Cross Entropy loss.

    Total loss = alpha * Dice + beta * BCE

    This combination often works better than either loss alone:
    - BCE provides pixel-wise supervision
    - Dice optimizes the overlap metric directly

    Arguments:
        alpha: Weight for Dice loss (default: 0.5)
        beta: Weight for BCE loss (default: 0.5)
        smooth: Smoothing factor for Dice (default: 1.0)
        pos_weight: Weight for positive class in BCE (default: None)

    Example:
        >>> criterion = DiceBCELoss(alpha=0.5, beta=0.5)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.dice = DiceLoss(smooth=smooth, sigmoid=False)

        if pos_weight is not None:
            self.bce = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight])
            )
        else:
            self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined Dice + BCE loss.

        Arguments:
            predictions: Model predictions (logits, before sigmoid)
            targets: Ground truth masks

        Returns:
            Combined loss value
        """
        # BCE expects logits
        bce_loss = self.bce(predictions, targets)

        # Dice expects probabilities
        predictions_sigmoid = torch.sigmoid(predictions)
        dice_loss = self.dice(predictions_sigmoid, targets)

        return self.alpha * dice_loss + self.beta * bce_loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focal loss = -alpha * (1 - p)^gamma * log(p)

    Down-weights easy examples and focuses on hard examples,
    useful when there's significant class imbalance.

    Arguments:
        alpha: Balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')

    References:
        Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Example:
        >>> criterion = FocalLoss(alpha=0.25, gamma=2.0)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal loss.

        Arguments:
            predictions: Model predictions (logits)
            targets: Ground truth masks

        Returns:
            Focal loss value
        """
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            predictions, targets, reduction='none'
        )

        # Get probabilities
        p = torch.sigmoid(predictions)
        p_t = p * targets + (1 - p) * (1 - targets)

        # Compute focal weight
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha balancing
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class TverskyLoss(nn.Module):
    """Tversky loss for controlling FP/FN trade-off.

    Tversky loss = 1 - TP / (TP + alpha*FN + beta*FP)

    Generalizes Dice loss with controllable FP/FN weights:
    - alpha = beta = 0.5: equivalent to Dice
    - alpha > beta: penalize FN more (better recall)
    - alpha < beta: penalize FP more (better precision)

    Arguments:
        alpha: Weight for false negatives (default: 0.5)
        beta: Weight for false positives (default: 0.5)
        smooth: Smoothing factor (default: 1.0)
        sigmoid: Whether to apply sigmoid (default: True)

    References:
        Salehi et al., "Tversky Loss Function for Image Segmentation" (2017)

    Example:
        >>> # Emphasize recall (penalize FN)
        >>> criterion = TverskyLoss(alpha=0.7, beta=0.3)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        smooth: float = 1.0,
        sigmoid: bool = True,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.sigmoid = sigmoid

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Tversky loss.

        Arguments:
            predictions: Model predictions
            targets: Ground truth masks

        Returns:
            Tversky loss value
        """
        if self.sigmoid:
            predictions = torch.sigmoid(predictions)

        # Flatten
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # Compute TP, FP, FN
        tp = (predictions * targets).sum()
        fp = (predictions * (1 - targets)).sum()
        fn = ((1 - predictions) * targets).sum()

        # Compute Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fn + self.beta * fp + self.smooth
        )

        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky loss combining Tversky with focal mechanism.

    Focal Tversky = (1 - Tversky)^gamma

    Applies focal mechanism to Tversky loss for better handling
    of hard examples and class imbalance.

    Arguments:
        alpha: Weight for false negatives (default: 0.7)
        beta: Weight for false positives (default: 0.3)
        gamma: Focal parameter (default: 0.75)
        smooth: Smoothing factor (default: 1.0)

    Example:
        >>> criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=0.75)
        >>> loss = criterion(predictions, targets)
    """

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.tversky = TverskyLoss(
            alpha=alpha, beta=beta, smooth=smooth, sigmoid=True
        )
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Focal Tversky loss."""
        tversky_loss = self.tversky(predictions, targets)
        return torch.pow(tversky_loss, self.gamma)


class DeepSupervisionLoss(nn.Module):
    """Wrapper for deep supervision training.

    Combines main output loss with weighted auxiliary output losses.
    Auxiliary outputs are typically upsampled to match target size.

    Arguments:
        base_loss: Loss function to use for all outputs
        weights: Weights for each output level (default: decreasing weights)

    Example:
        >>> base_loss = DiceBCELoss()
        >>> criterion = DeepSupervisionLoss(base_loss, weights=[1.0, 0.5, 0.25])
        >>> # outputs = (main_output, [aux1, aux2])
        >>> loss = criterion(outputs, targets)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        weights: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.weights = weights

    def forward(
        self,
        outputs: Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute deep supervision loss.

        Arguments:
            outputs: Either main output only, or (main_output, [aux_outputs])
            targets: Ground truth masks

        Returns:
            Combined loss value
        """
        # Handle non-deep supervision case
        if isinstance(outputs, torch.Tensor):
            return self.base_loss(outputs, targets)

        main_output, aux_outputs = outputs

        # Compute main loss
        total_loss = self.base_loss(main_output, targets)

        # Set default weights if not provided
        if self.weights is None:
            # Decreasing weights: 0.5, 0.25, 0.125, ...
            weights = [0.5 ** (i + 1) for i in range(len(aux_outputs))]
        else:
            weights = self.weights

        # Add auxiliary losses
        for aux_out, weight in zip(aux_outputs, weights):
            # Upsample auxiliary output to match target size
            if aux_out.shape[2:] != targets.shape[2:]:
                aux_out = F.interpolate(
                    aux_out,
                    size=targets.shape[2:],
                    mode='bilinear',
                    align_corners=False,
                )
            total_loss = total_loss + weight * self.base_loss(aux_out, targets)

        return total_loss


class BoundaryLoss(nn.Module):
    """Boundary-aware loss for better edge segmentation.

    Adds extra weight to boundary pixels using distance transform.

    Arguments:
        base_loss: Base loss function
        boundary_weight: Weight for boundary pixels (default: 5.0)

    Example:
        >>> criterion = BoundaryLoss(DiceLoss(), boundary_weight=5.0)
    """

    def __init__(
        self,
        base_loss: nn.Module,
        boundary_weight: float = 5.0,
    ) -> None:
        super().__init__()
        self.base_loss = base_loss
        self.boundary_weight = boundary_weight

    def _compute_boundary_weights(
        self,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary weights using edge detection."""
        # Use simple Laplacian for boundary detection
        laplacian = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=targets.dtype, device=targets.device).view(1, 1, 3, 3)

        # Detect boundaries
        boundaries = F.conv2d(
            targets, laplacian, padding=1
        ).abs()

        # Normalize and scale
        boundaries = (boundaries > 0).float()
        weights = 1.0 + (self.boundary_weight - 1.0) * boundaries

        return weights

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute boundary-weighted loss."""
        # Get base loss
        base_loss = self.base_loss(predictions, targets)

        # Compute boundary-weighted BCE
        weights = self._compute_boundary_weights(targets)
        weighted_bce = F.binary_cross_entropy_with_logits(
            predictions, targets, weight=weights
        )

        return base_loss + 0.5 * weighted_bce


def get_loss_function(
    name: str,
    **kwargs,
) -> nn.Module:
    """Get loss function by name.

    Arguments:
        name: Loss function name
        **kwargs: Arguments for the loss function

    Returns:
        Loss function module

    Available losses:
        - 'dice': DiceLoss
        - 'bce': BCEWithLogitsLoss
        - 'dice_bce': DiceBCELoss
        - 'focal': FocalLoss
        - 'tversky': TverskyLoss
        - 'focal_tversky': FocalTverskyLoss

    Example:
        >>> criterion = get_loss_function('dice_bce', alpha=0.5, beta=0.5)
    """
    losses = {
        'dice': DiceLoss,
        'bce': nn.BCEWithLogitsLoss,
        'dice_bce': DiceBCELoss,
        'focal': FocalLoss,
        'tversky': TverskyLoss,
        'focal_tversky': FocalTverskyLoss,
    }

    if name not in losses:
        available = list(losses.keys())
        raise ValueError(
            f"Unknown loss function: '{name}'. "
            f"Available: {available}"
        )

    return losses[name](**kwargs)


__all__ = [
    "DiceLoss",
    "DiceBCELoss",
    "FocalLoss",
    "TverskyLoss",
    "FocalTverskyLoss",
    "DeepSupervisionLoss",
    "BoundaryLoss",
    "get_loss_function",
]
