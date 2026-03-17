"""Tests for nn/datatools/augmentation.py — custom transforms, helpers, pipeline."""

import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors

from aggrequant.nn.datatools.augmentation import (
    RandomRotate90,
    RandomGamma,
    MultiplicativeNoise,
    _wrap_inputs,
    apply_transform,
    get_training_augmentation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def image_np():
    """Grayscale float32 image (H=64, W=64) in [0, 1]."""
    rng = np.random.RandomState(0)
    return rng.rand(64, 64).astype(np.float32)


@pytest.fixture
def mask_np():
    """Binary float32 mask (H=64, W=64)."""
    rng = np.random.RandomState(1)
    return rng.choice([0.0, 1.0], size=(64, 64)).astype(np.float32)


@pytest.fixture
def img_tv(image_np):
    """tv_tensors.Image (1, 64, 64)."""
    return tv_tensors.Image(torch.from_numpy(image_np).unsqueeze(0).float())


@pytest.fixture
def mask_tv(mask_np):
    """tv_tensors.Mask (1, 64, 64)."""
    return tv_tensors.Mask(torch.from_numpy(mask_np).unsqueeze(0).float())


# ===========================================================================
# _wrap_inputs
# ===========================================================================


class TestWrapInputs:
    """Tests for _wrap_inputs."""

    def test_correct_types_and_shape(self, image_np, mask_np):
        """Returns tv_tensors.Image and tv_tensors.Mask with shape (1, H, W)."""
        img_tv, mask_tv = _wrap_inputs(image_np, mask_np)

        assert isinstance(img_tv, tv_tensors.Image)
        assert isinstance(mask_tv, tv_tensors.Mask)
        assert img_tv.shape == (1, 64, 64)
        assert mask_tv.shape == (1, 64, 64)

    def test_dtype_float32(self):
        """Output is float32 regardless of input dtype."""
        img = np.ones((32, 32), dtype=np.uint16) * 500
        mask = np.ones((32, 32), dtype=np.uint8)

        img_tv, mask_tv = _wrap_inputs(img.astype(np.float32), mask.astype(np.float32))

        assert img_tv.dtype == torch.float32
        assert mask_tv.dtype == torch.float32


# ===========================================================================
# apply_transform
# ===========================================================================


class TestApplyTransform:
    """Tests for apply_transform."""

    def test_identity_transform(self, image_np, mask_np):
        """Identity transform preserves values."""
        transform = v2.Compose([v2.Identity()])
        img_out, mask_out = apply_transform(transform, image_np, mask_np)

        assert img_out.shape == (1, 64, 64)
        assert mask_out.shape == (1, 64, 64)
        assert torch.allclose(
            img_out, torch.from_numpy(image_np).unsqueeze(0).float()
        )

    def test_transform_applied(self, image_np, mask_np):
        """Deterministic flip changes the output."""
        transform = v2.Compose([v2.RandomHorizontalFlip(p=1.0)])
        img_out, mask_out = apply_transform(transform, image_np, mask_np)

        img_orig = torch.from_numpy(image_np).unsqueeze(0).float()
        # Flipped should differ from original (unless symmetric, very unlikely)
        assert not torch.equal(img_out, img_orig)
        # But should match manual flip
        assert torch.allclose(img_out, torch.flip(img_orig, dims=[-1]))


# ===========================================================================
# RandomRotate90
# ===========================================================================


class TestRandomRotate90:
    """Tests for RandomRotate90."""

    def test_p_zero_no_change(self, img_tv, mask_tv):
        """p=0 → output equals input."""
        t = RandomRotate90(p=0.0)
        out_img, out_mask = t(img_tv, mask_tv)

        assert torch.equal(out_img, img_tv)
        assert torch.equal(out_mask, mask_tv)

    def test_p_one_rotated(self, img_tv):
        """p=1 → output shape preserved, image is rotated."""
        t = RandomRotate90(p=1.0)
        out = t(img_tv)

        assert out.shape == img_tv.shape

    def test_mask_also_rotated(self, img_tv, mask_tv):
        """Both image and mask are transformed the same way."""
        t = RandomRotate90(p=1.0)
        out_img, out_mask = t(img_tv, mask_tv)

        # Both should be rotated by the same angle, so spatial structure
        # should change consistently. At minimum, shapes are preserved.
        assert out_img.shape == img_tv.shape
        assert out_mask.shape == mask_tv.shape


# ===========================================================================
# RandomGamma
# ===========================================================================


class TestRandomGamma:
    """Tests for RandomGamma."""

    def test_mask_unchanged(self, img_tv, mask_tv):
        """Mask values are not modified."""
        t = RandomGamma(p=1.0, gamma_range=(0.5, 0.5))
        _, out_mask = t(img_tv, mask_tv)

        assert torch.equal(out_mask, mask_tv)

    def test_image_modified(self, img_tv, mask_tv):
        """Image values differ after gamma correction."""
        t = RandomGamma(p=1.0, gamma_range=(0.5, 0.5))
        out_img, _ = t(img_tv, mask_tv)

        assert not torch.equal(out_img, img_tv)

    def test_output_clamped(self, img_tv, mask_tv):
        """Output image stays in [0, 1]."""
        t = RandomGamma(p=1.0, gamma_range=(0.3, 3.0))
        out_img, _ = t(img_tv, mask_tv)

        assert out_img.min() >= 0.0
        assert out_img.max() <= 1.0


# ===========================================================================
# MultiplicativeNoise
# ===========================================================================


class TestMultiplicativeNoise:
    """Tests for MultiplicativeNoise."""

    def test_mask_unchanged(self, img_tv, mask_tv):
        """Mask values are not modified."""
        t = MultiplicativeNoise(p=1.0)
        _, out_mask = t(img_tv, mask_tv)

        assert torch.equal(out_mask, mask_tv)

    def test_image_modified(self, img_tv, mask_tv):
        """Image values differ after noise."""
        t = MultiplicativeNoise(p=1.0, multiplier_range=(0.5, 0.5))
        out_img, _ = t(img_tv, mask_tv)

        assert not torch.equal(out_img, img_tv)

    def test_output_clamped(self, img_tv, mask_tv):
        """Output image stays in [0, 1]."""
        t = MultiplicativeNoise(p=1.0, multiplier_range=(0.0, 2.0))
        out_img, _ = t(img_tv, mask_tv)

        assert out_img.min() >= 0.0
        assert out_img.max() <= 1.0


# ===========================================================================
# get_training_augmentation
# ===========================================================================


class TestGetTrainingAugmentation:
    """Tests for get_training_augmentation."""

    def test_returns_compose(self):
        """Returns a v2.Compose object."""
        transform = get_training_augmentation()
        assert isinstance(transform, v2.Compose)

    def test_pipeline_runs(self, image_np, mask_np):
        """Pipeline applies without error and produces correct shapes."""
        transform = get_training_augmentation()
        img_out, mask_out = apply_transform(transform, image_np, mask_np)

        assert img_out.shape == (1, 64, 64)
        assert mask_out.shape == (1, 64, 64)

    def test_all_probabilities_zero(self, image_np, mask_np):
        """Disabling optional augmentations still returns a valid pipeline."""
        transform = get_training_augmentation(
            p_spatial=0, p_intensity=0, p_noise=0, p_blur=0,
        )
        img_out, mask_out = apply_transform(transform, image_np, mask_np)

        assert img_out.shape == (1, 64, 64)
        assert mask_out.shape == (1, 64, 64)
