"""Tests for nn/ training loss functions."""

import torch
import pytest

from aggrequant.nn.training.losses import (
    DiceLoss,
    DiceBCELoss,
    FocalLoss,
    TverskyLoss,
    FocalTverskyLoss,
    DeepSupervisionLoss,
    EdgeWeightedLoss,
    get_loss_function,
)


@pytest.fixture(scope="module")
def pred():
    """Logit predictions (B=2, C=1, H=16, W=16)."""
    torch.manual_seed(0)
    return torch.randn(2, 1, 16, 16)

@pytest.fixture(scope="module")
def target():
    """Binary target mask."""
    torch.manual_seed(1)
    return torch.randint(0, 2, (2, 1, 16, 16)).float()


# --- DiceLoss ---

def test_dice_loss_returns_scalar(pred, target):
    loss = DiceLoss()(pred, target)
    assert loss.shape == ()

def test_dice_loss_perfect_prediction():
    t = torch.ones(1, 1, 8, 8)
    # Large positive logits -> sigmoid ≈ 1
    p = torch.ones(1, 1, 8, 8) * 10.0
    loss = DiceLoss()(p, t)
    assert loss.item() < 0.01

def test_dice_loss_worst_prediction():
    t = torch.ones(1, 1, 8, 8)
    # Large negative logits -> sigmoid ≈ 0
    p = torch.ones(1, 1, 8, 8) * -10.0
    loss = DiceLoss()(p, t)
    assert loss.item() > 0.9

def test_dice_loss_range(pred, target):
    loss = DiceLoss()(pred, target)
    assert 0.0 <= loss.item() <= 1.0


# --- DiceBCELoss ---

def test_dice_bce_loss_returns_scalar(pred, target):
    loss = DiceBCELoss()(pred, target)
    assert loss.shape == ()

def test_dice_bce_loss_with_pos_weight(pred, target):
    loss = DiceBCELoss(pos_weight=2.0)(pred, target)
    assert loss.shape == ()
    assert torch.isfinite(loss)


# --- FocalLoss ---

def test_focal_loss_returns_scalar(pred, target):
    loss = FocalLoss()(pred, target)
    assert loss.shape == ()

def test_focal_loss_reduction_none(pred, target):
    loss = FocalLoss(reduction="none")(pred, target)
    assert loss.shape == pred.shape


# --- TverskyLoss ---

def test_tversky_loss_returns_scalar(pred, target):
    loss = TverskyLoss()(pred, target)
    assert loss.shape == ()

def test_tversky_equals_dice_when_symmetric():
    """alpha=0.5, beta=0.5 approximates Dice (differs only by smooth constant)."""
    torch.manual_seed(42)
    p = torch.randn(1, 1, 8, 8)
    t = torch.randint(0, 2, (1, 1, 8, 8)).float()
    # With smooth=0, Tversky(0.5, 0.5) == Dice exactly
    tversky = TverskyLoss(alpha=0.5, beta=0.5, smooth=0.0)(p, t)
    dice = DiceLoss(smooth=0.0)(p, t)
    assert tversky.item() == pytest.approx(dice.item(), abs=1e-5)


# --- FocalTverskyLoss ---

def test_focal_tversky_loss_returns_scalar(pred, target):
    loss = FocalTverskyLoss()(pred, target)
    assert loss.shape == ()


# --- EdgeWeightedLoss ---

def test_boundary_loss_returns_scalar(pred, target):
    loss = EdgeWeightedLoss(DiceLoss())(pred, target)
    assert loss.shape == ()


# --- DeepSupervisionLoss ---

def test_deep_supervision_plain_tensor(pred, target):
    """When given a plain tensor (no aux), acts like base loss."""
    base = DiceLoss()
    ds = DeepSupervisionLoss(base)
    loss_ds = ds(pred, target)
    loss_base = base(pred, target)
    assert loss_ds.item() == pytest.approx(loss_base.item())

def test_deep_supervision_with_aux(target):
    main = torch.randn(2, 1, 16, 16)
    aux1 = torch.randn(2, 1, 8, 8)    # smaller scale
    aux2 = torch.randn(2, 1, 4, 4)
    ds = DeepSupervisionLoss(DiceLoss(), weights=[0.5, 0.25])
    loss = ds((main, [aux1, aux2]), target)
    assert loss.shape == ()
    assert torch.isfinite(loss)

def test_deep_supervision_default_weights(target):
    main = torch.randn(2, 1, 16, 16)
    aux = [torch.randn(2, 1, 8, 8)]
    ds = DeepSupervisionLoss(DiceLoss())
    loss = ds((main, aux), target)
    assert torch.isfinite(loss)


# --- get_loss_function factory ---

@pytest.mark.parametrize("name", ["dice", "bce", "dice_bce", "focal", "tversky", "focal_tversky"])
def test_get_loss_function_factory(name, pred, target):
    loss_fn = get_loss_function(name)
    loss = loss_fn(pred, target)
    assert loss.shape == ()
    assert torch.isfinite(loss)

def test_get_loss_function_unknown():
    with pytest.raises(ValueError, match="Unknown loss"):
        get_loss_function("nonexistent")


# --- Gradient flow ---

@pytest.mark.parametrize("loss_cls", [DiceLoss, DiceBCELoss, FocalLoss, TverskyLoss, FocalTverskyLoss])
def test_loss_gradient_flows(loss_cls):
    p = torch.randn(1, 1, 8, 8, requires_grad=True)
    t = torch.randint(0, 2, (1, 1, 8, 8)).float()
    loss = loss_cls()(p, t)
    loss.backward()
    assert p.grad is not None
    assert torch.all(torch.isfinite(p.grad))
