"""
Unit tests for aggrequant.quality.focus module.

Author: Athena Economides, 2026, UZH
"""

import numpy as np
import pytest
from aggrequant.quality.focus import (
    variance_of_laplacian,
    laplace_energy,
    sobel_metric,
    brenner_metric,
    focus_score,
    generate_blur_mask,
    compute_patch_focus_maps,
)


class TestFocusMetricFunctions:
    """Tests for individual focus metric functions."""

    def test_variance_of_laplacian_sharp(self):
        """Sharp edges should have high variance of Laplacian."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255  # Sharp square
        vol = variance_of_laplacian(img)
        assert vol > 100

    def test_variance_of_laplacian_blurry(self):
        """Blurry/constant image should have low variance of Laplacian."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        vol = variance_of_laplacian(img)
        assert vol < 1

    def test_laplace_energy_sharp(self):
        """Sharp edges should have high Laplacian energy."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255
        energy = laplace_energy(img)
        assert energy > 100

    def test_sobel_metric_sharp(self):
        """Sharp edges should have high Sobel magnitude."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255
        sobel = sobel_metric(img)
        assert sobel > 1

    def test_sobel_metric_constant(self):
        """Constant image should have near-zero Sobel magnitude."""
        img = np.ones((100, 100), dtype=np.uint8) * 128
        sobel = sobel_metric(img)
        assert sobel < 0.1

    def test_brenner_metric_sharp(self):
        """Sharp edges should have high Brenner metric."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255
        brenner = brenner_metric(img)
        assert brenner > 1000

    def test_focus_score_contrast(self):
        """High contrast images should have high focus score."""
        img = np.zeros((100, 100), dtype=np.uint8)
        img[::2, :] = 255  # Striped pattern
        score = focus_score(img)
        assert score > 1


class TestComputePatchFocusMaps:
    """Tests for compute_patch_focus_maps function."""

    def test_returns_correct_maps(self):
        """Should return dict with all expected focus maps."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        maps, ys, xs = compute_patch_focus_maps(img, patch_size=(40, 40))

        expected_keys = {"VarianceLaplacian", "LaplaceEnergy", "Sobel", "Brenner", "FocusScore"}
        assert set(maps.keys()) == expected_keys

    def test_map_shapes_correct(self):
        """Map shapes should match patch grid dimensions."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        maps, ys, xs = compute_patch_focus_maps(img, patch_size=(40, 40))

        expected_shape = (5, 5)  # 200/40 = 5 patches per dimension
        for name, m in maps.items():
            assert m.shape == expected_shape, f"Map '{name}' has wrong shape"

    def test_coordinate_lists_correct(self):
        """Returned coordinate lists should be correct."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        maps, ys, xs = compute_patch_focus_maps(img, patch_size=(40, 40))

        assert ys == [0, 40, 80, 120, 160]
        assert xs == [0, 40, 80, 120, 160]

    def test_rejects_3d_image(self):
        """Should reject 3D images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D image"):
            compute_patch_focus_maps(img)

    def test_rejects_too_small_image(self):
        """Should reject images smaller than patch size."""
        img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small for patch size"):
            compute_patch_focus_maps(img, patch_size=(40, 40))

    def test_sharp_patches_high_variance(self):
        """Patches with edges should have high variance of Laplacian."""
        img = np.zeros((200, 200), dtype=np.uint8)
        # Create checkerboard pattern
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    img[i:i+20, j:j+20] = 255

        maps, ys, xs = compute_patch_focus_maps(img, patch_size=(40, 40))
        # All patches should have high values due to edges
        assert np.mean(maps["VarianceLaplacian"]) > 100

    def test_constant_patches_low_variance(self):
        """Constant patches should have low variance of Laplacian."""
        img = np.ones((200, 200), dtype=np.uint8) * 128
        maps, ys, xs = compute_patch_focus_maps(img, patch_size=(40, 40))
        assert np.max(maps["VarianceLaplacian"]) < 1


class TestGenerateBlurMask:
    """Tests for generate_blur_mask function."""

    def test_returns_boolean_array(self):
        """Should return boolean array of same shape as input."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        mask = generate_blur_mask(img)
        assert mask.dtype == bool
        assert mask.shape == img.shape

    def test_sharp_image_low_mask(self):
        """Sharp image should have mostly False (not blurry) mask."""
        img = np.zeros((200, 200), dtype=np.uint8)
        img[50:150, 50:150] = 255  # Sharp square
        mask = generate_blur_mask(img, threshold=15)
        # Most of the image has sharp edges
        assert np.mean(mask) < 0.8

    def test_constant_image_high_mask(self):
        """Constant image should have mostly True (blurry) mask."""
        img = np.ones((200, 200), dtype=np.uint8) * 128
        mask = generate_blur_mask(img, threshold=15)
        assert np.mean(mask) > 0.9

    def test_rejects_3d_image(self):
        """Should reject 3D images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D image"):
            generate_blur_mask(img)

    def test_custom_metric(self):
        """Should support different metrics for thresholding."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        mask_lap = generate_blur_mask(img, metric="VarianceLaplacian", threshold=50)
        mask_sobel = generate_blur_mask(img, metric="Sobel", threshold=10)
        # Both should return valid masks
        assert mask_lap.dtype == bool
        assert mask_sobel.dtype == bool

    def test_invalid_metric_raises(self):
        """Should raise error for unknown metric."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unknown metric"):
            generate_blur_mask(img, metric="InvalidMetric")

    def test_threshold_affects_mask(self):
        """Higher threshold should result in more blurry patches."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        mask_low = generate_blur_mask(img, threshold=1)
        mask_high = generate_blur_mask(img, threshold=1000)
        # High threshold means more patches are below it (more blurry)
        assert np.sum(mask_high) >= np.sum(mask_low)
