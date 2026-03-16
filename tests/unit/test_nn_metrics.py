"""Tests for nn/ evaluation metrics."""

import torch
import pytest

from aggrequant.nn.evaluation.metrics import (
    dice_score,
    soft_dice_score,
    iou_score,
    precision_score,
    recall_score,
    specificity_score,
    accuracy_score,
    confusion_matrix,
    SegmentationMetrics,
    find_optimal_threshold,
)


# --- Known-value fixtures ---

@pytest.fixture(scope="module")
def perfect_pair():
    """Prediction matches target exactly."""
    t = torch.tensor([[[[1, 1, 0], [0, 1, 0], [0, 0, 1]]]]).float()
    p = t.clone()  # probabilities already binary
    return p, t

@pytest.fixture(scope="module")
def known_pair():
    """Hand-crafted case: TP=2, FP=1, FN=1, TN=5."""
    # target: 3 positives, 6 negatives
    t = torch.tensor([[[[1, 1, 1], [0, 0, 0], [0, 0, 0]]]]).float()
    # pred:   2 correct positives, 1 FP, 1 FN
    p = torch.tensor([[[[1, 1, 0], [1, 0, 0], [0, 0, 0]]]]).float()
    return p, t


# --- dice_score ---

def test_dice_perfect(perfect_pair):
    p, t = perfect_pair
    assert dice_score(p, t).item() == pytest.approx(1.0, abs=1e-5)

def test_dice_no_overlap():
    t = torch.ones(1, 1, 4, 4)
    p = torch.zeros(1, 1, 4, 4)
    assert dice_score(p, t).item() < 0.01

def test_dice_known_value(known_pair):
    p, t = known_pair
    # TP=2, FP=1, FN=1 → Dice = 2*2 / (2*2+1+1) = 4/6 ≈ 0.6667
    assert dice_score(p, t, smooth=0).item() == pytest.approx(4.0 / 6.0, abs=1e-5)


# --- soft_dice_score ---

def test_soft_dice_perfect(perfect_pair):
    p, t = perfect_pair
    assert soft_dice_score(p, t).item() == pytest.approx(1.0, abs=1e-5)

def test_soft_dice_no_overlap():
    t = torch.ones(1, 1, 4, 4)
    p = torch.zeros(1, 1, 4, 4)
    assert soft_dice_score(p, t).item() < 0.01

def test_soft_dice_uses_probabilities():
    """Soft dice should differ from hard dice when predictions are not binary."""
    t = torch.tensor([[[[1, 1, 0], [0, 0, 0], [0, 0, 0]]]]).float()
    p = torch.tensor([[[[0.8, 0.6, 0.3], [0.2, 0.1, 0.0], [0.0, 0.0, 0.0]]]]).float()
    hard = dice_score(p, t, smooth=0).item()
    soft = soft_dice_score(p, t, smooth=0).item()
    # Hard dice thresholds at 0.5, so p becomes [1,1,0,...] → same as target → dice=1
    # Soft dice uses continuous values → different result
    assert hard != pytest.approx(soft, abs=0.01)


# --- iou_score ---

def test_iou_perfect(perfect_pair):
    p, t = perfect_pair
    assert iou_score(p, t).item() == pytest.approx(1.0, abs=1e-5)

def test_iou_known_value(known_pair):
    p, t = known_pair
    # TP=2, FP=1, FN=1 → IoU = 2 / (2+1+1) = 0.5
    assert iou_score(p, t, smooth=0).item() == pytest.approx(0.5, abs=1e-5)


# --- precision / recall ---

def test_precision_known_value(known_pair):
    p, t = known_pair
    # TP=2, FP=1 → precision = 2/3
    assert precision_score(p, t, smooth=0).item() == pytest.approx(2.0 / 3.0, abs=1e-5)

def test_recall_known_value(known_pair):
    p, t = known_pair
    # TP=2, FN=1 → recall = 2/3
    assert recall_score(p, t, smooth=0).item() == pytest.approx(2.0 / 3.0, abs=1e-5)

def test_precision_perfect(perfect_pair):
    p, t = perfect_pair
    assert precision_score(p, t).item() == pytest.approx(1.0, abs=1e-5)

def test_recall_perfect(perfect_pair):
    p, t = perfect_pair
    assert recall_score(p, t).item() == pytest.approx(1.0, abs=1e-5)


# --- specificity ---

def test_specificity_known_value(known_pair):
    p, t = known_pair
    # TN=5, FP=1 → specificity = 5/6
    assert specificity_score(p, t, smooth=0).item() == pytest.approx(5.0 / 6.0, abs=1e-5)


# --- accuracy ---

def test_accuracy_perfect(perfect_pair):
    p, t = perfect_pair
    assert accuracy_score(p, t).item() == pytest.approx(1.0, abs=1e-5)

def test_accuracy_known_value(known_pair):
    p, t = known_pair
    # 7 correct out of 9
    assert accuracy_score(p, t).item() == pytest.approx(7.0 / 9.0, abs=1e-5)


# --- confusion_matrix ---

def test_confusion_matrix_known_values(known_pair):
    p, t = known_pair
    tp, tn, fp, fn = confusion_matrix(p, t)
    assert tp == 2
    assert tn == 5
    assert fp == 1
    assert fn == 1

def test_confusion_matrix_sums_to_total(known_pair):
    p, t = known_pair
    tp, tn, fp, fn = confusion_matrix(p, t)
    assert tp + tn + fp + fn == t.numel()


# --- SegmentationMetrics ---

def test_segmentation_metrics_default_keys(known_pair):
    p, t = known_pair
    m = SegmentationMetrics()
    result = m(p, t)
    expected_keys = {"dice", "iou", "precision", "recall", "specificity", "accuracy"}
    assert set(result.keys()) == expected_keys

def test_segmentation_metrics_custom_keys(known_pair):
    p, t = known_pair
    m = SegmentationMetrics(metrics=["dice", "iou"])
    result = m(p, t)
    assert set(result.keys()) == {"dice", "iou"}

def test_segmentation_metrics_unknown_metric():
    with pytest.raises(ValueError, match="Unknown metric"):
        SegmentationMetrics(metrics=["nonexistent"])

def test_segmentation_metrics_values_are_floats(known_pair):
    p, t = known_pair
    result = SegmentationMetrics()(p, t)
    for v in result.values():
        assert isinstance(v, float)


# --- find_optimal_threshold ---

def test_find_optimal_threshold_returns_tuple():
    # Predictions that clearly favor threshold ~0.5
    p = torch.rand(1, 1, 16, 16)
    t = (p > 0.5).float()
    thresh, value = find_optimal_threshold(p, t)
    assert isinstance(thresh, float)
    assert isinstance(value, float)
    assert 0.0 < thresh < 1.0
    assert 0.0 < value <= 1.0

def test_find_optimal_threshold_finds_good_threshold():
    """When predictions perfectly separate at 0.5, optimal should be near 0.5."""
    torch.manual_seed(0)
    t = torch.randint(0, 2, (1, 1, 32, 32)).float()
    # Make predictions = target values (0 or 1), so any threshold in (0,1) is perfect
    p = t.clone()
    thresh, value = find_optimal_threshold(p, t)
    assert value > 0.99

def test_find_optimal_threshold_unknown_metric():
    p = torch.rand(1, 1, 8, 8)
    t = torch.randint(0, 2, (1, 1, 8, 8)).float()
    with pytest.raises(ValueError, match="Unknown metric"):
        find_optimal_threshold(p, t, metric="nonexistent")


# --- apply_sigmoid flag ---

def test_dice_with_sigmoid():
    t = torch.ones(1, 1, 4, 4)
    # Large positive logits → sigmoid ≈ 1 → should match target
    p = torch.ones(1, 1, 4, 4) * 10.0
    score = dice_score(p, t, apply_sigmoid=True)
    assert score.item() == pytest.approx(1.0, abs=1e-5)
