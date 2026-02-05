"""Unit tests for evaluation metrics.

Author: Athena Economides, 2026, UZH
"""

import pytest
import torch
import numpy as np

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
)


class TestDiceScore:
    """Test Dice score metric."""

    def test_perfect_overlap(self):
        """Dice should be 1.0 for perfect overlap."""
        pred = torch.ones(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        dice = dice_score(pred, target, threshold=0.5)
        assert dice.item() > 0.99

    def test_no_overlap(self):
        """Dice should be ~0 for no overlap."""
        pred = torch.zeros(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        dice = dice_score(pred, target, threshold=0.5)
        assert dice.item() < 0.01

    def test_partial_overlap(self):
        """Dice should be between 0 and 1 for partial overlap."""
        pred = torch.zeros(1, 1, 64, 64)
        pred[0, 0, :32, :] = 1.0
        target = torch.ones(1, 1, 64, 64)
        dice = dice_score(pred, target, threshold=0.5)
        assert 0 < dice.item() < 1

    def test_dice_equals_f1(self):
        """Dice should equal F1 for binary segmentation."""
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        dice = dice_score(pred, target)
        f1 = f1_score(pred, target)
        assert abs(dice.item() - f1.item()) < 0.01


class TestIoUScore:
    """Test IoU score metric."""

    def test_perfect_overlap(self):
        """IoU should be 1.0 for perfect overlap."""
        pred = torch.ones(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        iou = iou_score(pred, target, threshold=0.5)
        assert iou.item() > 0.99

    def test_no_overlap(self):
        """IoU should be ~0 for no overlap."""
        pred = torch.zeros(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        iou = iou_score(pred, target, threshold=0.5)
        assert iou.item() < 0.01

    def test_iou_less_than_dice(self):
        """IoU should be less than or equal to Dice for same predictions."""
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        iou = iou_score(pred, target)
        dice = dice_score(pred, target)
        # IoU = Dice / (2 - Dice), so IoU <= Dice
        assert iou.item() <= dice.item() + 0.01


class TestPrecisionRecall:
    """Test precision and recall metrics."""

    def test_precision_perfect(self):
        """Precision should be 1.0 when all predictions are correct."""
        pred = torch.zeros(1, 1, 64, 64)
        pred[0, 0, :32, :32] = 1.0
        target = torch.zeros(1, 1, 64, 64)
        target[0, 0, :32, :32] = 1.0
        prec = precision(pred, target, threshold=0.5)
        assert prec.item() > 0.99

    def test_recall_perfect(self):
        """Recall should be 1.0 when all positives are found."""
        pred = torch.ones(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        target[0, 0, :32, :32] = 1.0
        rec = recall(pred, target, threshold=0.5)
        assert rec.item() > 0.99

    def test_precision_zero(self):
        """Precision should be ~0 when all predictions are wrong."""
        pred = torch.ones(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        prec = precision(pred, target, threshold=0.5)
        assert prec.item() < 0.01

    def test_recall_zero(self):
        """Recall should be ~0 when no positives are found."""
        pred = torch.zeros(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        rec = recall(pred, target, threshold=0.5)
        assert rec.item() < 0.01


class TestSpecificity:
    """Test specificity metric."""

    def test_specificity_perfect(self):
        """Specificity should be 1.0 for perfect TN detection."""
        pred = torch.zeros(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        spec = specificity(pred, target, threshold=0.5)
        assert spec.item() > 0.99


class TestAccuracy:
    """Test accuracy metric."""

    def test_accuracy_perfect(self):
        """Accuracy should be 1.0 for perfect predictions."""
        pred = torch.ones(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        acc = accuracy(pred, target, threshold=0.5)
        assert acc.item() > 0.99

    def test_accuracy_half(self):
        """Accuracy should be 0.5 when half predictions are correct."""
        # Create pattern where pred and target match exactly half the pixels
        pred = torch.zeros(1, 1, 64, 64)
        pred[0, 0, :32, :] = 1.0  # Top half = 1
        target = torch.zeros(1, 1, 64, 64)
        target[0, 0, :16, :] = 1.0  # Top quarter = 1
        target[0, 0, 32:48, :] = 1.0  # Third quarter = 1
        # Now: TP in rows 0-16, FP in rows 16-32, FN in rows 32-48, TN in rows 48-64
        # Correct = rows 0-16 (TP) + rows 48-64 (TN) = 2048 + 1024 = half
        acc = accuracy(pred, target, threshold=0.5)
        assert 0.4 < acc.item() < 0.6


class TestConfusionMatrix:
    """Test confusion matrix computation."""

    def test_confusion_matrix_all_positive(self):
        """Test confusion matrix with all positive predictions."""
        pred = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        tp, tn, fp, fn = confusion_matrix(pred, target, threshold=0.5)
        assert tp == 16
        assert tn == 0
        assert fp == 0
        assert fn == 0

    def test_confusion_matrix_all_negative(self):
        """Test confusion matrix with all negative predictions."""
        pred = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        tp, tn, fp, fn = confusion_matrix(pred, target, threshold=0.5)
        assert tp == 0
        assert tn == 16
        assert fp == 0
        assert fn == 0

    def test_confusion_matrix_mixed(self):
        """Test confusion matrix with mixed results."""
        pred = torch.zeros(1, 1, 4, 4)
        pred[0, 0, :2, :2] = 1.0  # 4 positive predictions
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:3, :2] = 1.0  # 4 positive labels
        tp, tn, fp, fn = confusion_matrix(pred, target, threshold=0.5)
        assert tp == 2
        assert fp == 2
        assert fn == 2
        assert tn == 10


class TestSegmentationMetrics:
    """Test SegmentationMetrics container."""

    def test_returns_dict(self):
        """Should return dictionary with all metrics."""
        metrics = SegmentationMetrics()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        result = metrics(pred, target)
        assert isinstance(result, dict)
        assert "dice" in result
        assert "iou" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_all_values_in_range(self):
        """All metrics should be between 0 and 1."""
        metrics = SegmentationMetrics()
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        result = metrics(pred, target)
        for key, value in result.items():
            assert 0 <= value <= 1, f"{key} out of range: {value}"


class TestFindOptimalThreshold:
    """Test optimal threshold finding."""

    def test_finds_threshold(self):
        """Should find a threshold in valid range."""
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        thresh, value = find_optimal_threshold(pred, target, metric="dice")
        assert 0 <= thresh <= 1
        assert 0 <= value <= 1

    def test_different_metrics(self):
        """Should work with different metrics."""
        pred = torch.rand(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        thresh_dice, _ = find_optimal_threshold(pred, target, metric="dice")
        thresh_iou, _ = find_optimal_threshold(pred, target, metric="iou")
        thresh_f1, _ = find_optimal_threshold(pred, target, metric="f1")

        # All should be valid thresholds
        assert all(0 <= t <= 1 for t in [thresh_dice, thresh_iou, thresh_f1])


class TestApplySigmoid:
    """Test apply_sigmoid parameter."""

    def test_sigmoid_makes_difference(self):
        """apply_sigmoid should affect results for logits."""
        pred_logits = torch.randn(1, 1, 64, 64) * 3  # logits
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        # Without sigmoid (treats logits as probabilities - incorrect)
        dice_no_sig = dice_score(pred_logits, target, apply_sigmoid=False)
        # With sigmoid (correct for logits)
        dice_sig = dice_score(pred_logits, target, apply_sigmoid=True)

        # Results should be different
        assert dice_no_sig.item() != dice_sig.item()

    def test_sigmoid_on_probabilities(self):
        """apply_sigmoid on probabilities should still work."""
        pred = torch.rand(1, 1, 64, 64)  # already probabilities
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        dice = dice_score(pred, target, apply_sigmoid=True)
        assert 0 <= dice.item() <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
