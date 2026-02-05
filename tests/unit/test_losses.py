"""Unit tests for loss functions.

Author: Athena Economides, 2026, UZH
"""

import pytest
import torch
import torch.nn as nn

from aggrequant.nn.training.losses import (
    DiceLoss,
    DiceBCELoss,
    FocalLoss,
    TverskyLoss,
    FocalTverskyLoss,
    DeepSupervisionLoss,
    get_loss_function,
)


class TestDiceLoss:
    """Test Dice loss function."""

    def test_perfect_prediction(self):
        """Dice loss should be ~0 for perfect predictions."""
        loss_fn = DiceLoss(sigmoid=False)
        # Perfect match
        pred = torch.ones(1, 1, 64, 64)
        target = torch.ones(1, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_worst_prediction(self):
        """Dice loss should be ~1 for completely wrong predictions."""
        loss_fn = DiceLoss(sigmoid=False)
        pred = torch.ones(1, 1, 64, 64)
        target = torch.zeros(1, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() > 0.99

    def test_with_sigmoid(self):
        """Dice loss with sigmoid should handle logits."""
        loss_fn = DiceLoss(sigmoid=True)
        # High logits (will be ~1 after sigmoid)
        pred = torch.ones(1, 1, 64, 64) * 10
        target = torch.ones(1, 1, 64, 64)
        loss = loss_fn(pred, target)
        assert loss.item() < 0.1

    def test_gradient_flow(self):
        """Loss should have gradients."""
        loss_fn = DiceLoss()
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


class TestDiceBCELoss:
    """Test combined Dice + BCE loss."""

    def test_combined_loss(self):
        """DiceBCELoss should combine both losses."""
        loss_fn = DiceBCELoss(alpha=0.5, beta=0.5)
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_gradient_flow(self):
        """Combined loss should have gradients."""
        loss_fn = DiceBCELoss()
        pred = torch.randn(1, 1, 32, 32, requires_grad=True)
        target = torch.randint(0, 2, (1, 1, 32, 32)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None

    def test_pos_weight(self):
        """DiceBCELoss should accept pos_weight."""
        loss_fn = DiceBCELoss(pos_weight=2.0)
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)


class TestFocalLoss:
    """Test Focal loss function."""

    def test_focal_loss(self):
        """Focal loss should work on random inputs."""
        loss_fn = FocalLoss()
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_gamma_effect(self):
        """Higher gamma should down-weight easy examples more."""
        pred = torch.ones(1, 1, 64, 64) * 5  # confident prediction
        target = torch.ones(1, 1, 64, 64)  # correct

        loss_gamma0 = FocalLoss(gamma=0)(pred, target)
        loss_gamma2 = FocalLoss(gamma=2)(pred, target)

        # With higher gamma, confident correct predictions contribute less
        assert loss_gamma2.item() < loss_gamma0.item()

    def test_reduction_modes(self):
        """Focal loss should support different reduction modes."""
        pred = torch.randn(2, 1, 32, 32)
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss_mean = FocalLoss(reduction="mean")(pred, target)
        loss_sum = FocalLoss(reduction="sum")(pred, target)
        loss_none = FocalLoss(reduction="none")(pred, target)

        assert loss_mean.dim() == 0  # scalar
        assert loss_sum.dim() == 0  # scalar
        assert loss_none.shape == pred.shape  # per-element


class TestTverskyLoss:
    """Test Tversky loss function."""

    def test_tversky_loss(self):
        """Tversky loss should work on random inputs."""
        loss_fn = TverskyLoss()
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert 0 <= loss.item() <= 1
        assert not torch.isnan(loss)

    def test_tversky_equals_dice(self):
        """Tversky with alpha=beta=0.5 should equal Dice."""
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        dice_loss = DiceLoss(smooth=1.0)(pred, target)
        tversky_loss = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)(pred, target)

        # Should be approximately equal
        assert abs(dice_loss.item() - tversky_loss.item()) < 0.01

    def test_alpha_beta_tradeoff(self):
        """Different alpha/beta should change FN/FP sensitivity."""
        # Create prediction with more FN than FP
        pred = torch.zeros(1, 1, 64, 64)  # predict nothing
        target = torch.ones(1, 1, 64, 64)  # but everything is positive

        # High alpha penalizes FN more
        loss_high_alpha = TverskyLoss(alpha=0.7, beta=0.3, sigmoid=False)(pred, target)
        loss_high_beta = TverskyLoss(alpha=0.3, beta=0.7, sigmoid=False)(pred, target)

        assert loss_high_alpha.item() > loss_high_beta.item()


class TestFocalTverskyLoss:
    """Test Focal Tversky loss."""

    def test_focal_tversky(self):
        """Focal Tversky should work on random inputs."""
        loss_fn = FocalTverskyLoss()
        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestDeepSupervisionLoss:
    """Test deep supervision loss wrapper."""

    def test_single_output(self):
        """Should work with single output (no deep supervision)."""
        base_loss = DiceBCELoss()
        loss_fn = DeepSupervisionLoss(base_loss)

        pred = torch.randn(1, 1, 64, 64)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()
        loss = loss_fn(pred, target)
        assert not torch.isnan(loss)

    def test_deep_supervision_output(self):
        """Should handle deep supervision outputs."""
        base_loss = DiceBCELoss()
        loss_fn = DeepSupervisionLoss(base_loss)

        main_out = torch.randn(1, 1, 64, 64)
        aux_out1 = torch.randn(1, 1, 32, 32)
        aux_out2 = torch.randn(1, 1, 16, 16)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        loss = loss_fn((main_out, [aux_out1, aux_out2]), target)
        assert not torch.isnan(loss)

    def test_custom_weights(self):
        """Should accept custom weights."""
        base_loss = DiceLoss()
        loss_fn = DeepSupervisionLoss(base_loss, weights=[0.3, 0.2])

        main_out = torch.randn(1, 1, 64, 64)
        aux_out1 = torch.randn(1, 1, 32, 32)
        aux_out2 = torch.randn(1, 1, 16, 16)
        target = torch.randint(0, 2, (1, 1, 64, 64)).float()

        loss = loss_fn((main_out, [aux_out1, aux_out2]), target)
        assert not torch.isnan(loss)


class TestGetLossFunction:
    """Test loss function factory."""

    def test_get_dice(self):
        """Should create Dice loss."""
        loss_fn = get_loss_function("dice")
        assert isinstance(loss_fn, DiceLoss)

    def test_get_dice_bce(self):
        """Should create DiceBCE loss."""
        loss_fn = get_loss_function("dice_bce", alpha=0.7)
        assert isinstance(loss_fn, DiceBCELoss)

    def test_get_focal(self):
        """Should create Focal loss."""
        loss_fn = get_loss_function("focal", gamma=2.5)
        assert isinstance(loss_fn, FocalLoss)

    def test_get_unknown(self):
        """Should raise error for unknown loss."""
        with pytest.raises(ValueError):
            get_loss_function("unknown_loss")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
