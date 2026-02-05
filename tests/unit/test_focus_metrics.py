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
    compute_focus_metrics,
    generate_blur_mask,
    compute_patch_focus_maps,
    FocusMetrics,
    _prepare_image_for_cv2,
)


class TestPrepareImageForCv2:
    """Tests for the _prepare_image_for_cv2 helper function."""

    def test_uint8_passthrough(self):
        """uint8 images should pass through unchanged."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = _prepare_image_for_cv2(img)
        assert result.dtype == np.uint8
        np.testing.assert_array_equal(result, img)

    def test_uint16_conversion(self):
        """uint16 images should be converted to uint8."""
        img = np.random.randint(0, 65535, (100, 100), dtype=np.uint16)
        result = _prepare_image_for_cv2(img)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_all_zero_image(self):
        """All-zero images should not cause division by zero."""
        img = np.zeros((100, 100), dtype=np.uint16)
        result = _prepare_image_for_cv2(img)
        assert result.dtype == np.uint8
        assert np.all(result == 0)
        assert not np.any(np.isnan(result))

    def test_constant_image(self):
        """Constant-value images should be handled correctly."""
        img = np.ones((100, 100), dtype=np.uint16) * 1000
        result = _prepare_image_for_cv2(img)
        assert result.dtype == np.uint8
        assert np.all(result == 0)  # Constant image maps to zero
        assert not np.any(np.isnan(result))

    def test_float_image(self):
        """Float images should be converted to uint8."""
        img = np.random.rand(100, 100).astype(np.float32)
        result = _prepare_image_for_cv2(img)
        assert result.dtype == np.uint8


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


class TestComputeFocusMetrics:
    """Tests for compute_focus_metrics function."""

    def test_returns_focusmetrics_dataclass(self):
        """Should return FocusMetrics dataclass."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        result = compute_focus_metrics(img)
        assert isinstance(result, FocusMetrics)

    def test_all_zero_uint16(self):
        """All-zero uint16 images should not cause division by zero."""
        img = np.zeros((200, 200), dtype=np.uint16)
        metrics = compute_focus_metrics(img)
        assert metrics.variance_laplacian_mean >= 0
        assert not np.isnan(metrics.variance_laplacian_mean)
        assert metrics.is_likely_blurry  # Zero image is blurry

    def test_constant_image_is_blurry(self):
        """Constant-value images should be detected as blurry."""
        img = np.ones((200, 200), dtype=np.uint16) * 1000
        metrics = compute_focus_metrics(img)
        assert metrics.variance_laplacian_mean < 10
        assert metrics.is_likely_blurry

    def test_sharp_image_not_blurry(self):
        """Sharp image with edges should not be detected as blurry."""
        img = np.zeros((200, 200), dtype=np.uint8)
        # Create a checkerboard pattern (very sharp)
        img[::10, ::10] = 255
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                img[i:i+10, j:j+10] = 255
        metrics = compute_focus_metrics(img)
        assert not metrics.is_likely_blurry

    def test_patch_count(self):
        """Patch count should be correct for image and patch size."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        patch_size = (40, 40)
        metrics = compute_focus_metrics(img, patch_size=patch_size)
        expected_patches = (200 // 40) * (200 // 40)  # 5 * 5 = 25
        assert metrics.n_patches_total == expected_patches

    def test_rejects_3d_image(self):
        """Should reject 3D images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D image"):
            compute_focus_metrics(img)

    def test_rejects_too_small_image(self):
        """Should reject images smaller than patch size."""
        img = np.random.randint(0, 255, (30, 30), dtype=np.uint8)
        with pytest.raises(ValueError, match="too small for patch size"):
            compute_focus_metrics(img, patch_size=(40, 40))

    def test_custom_blur_threshold(self):
        """Custom blur threshold should affect blurry patch count."""
        img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        metrics_low = compute_focus_metrics(img, blur_threshold=1)
        metrics_high = compute_focus_metrics(img, blur_threshold=1000)
        # With very high threshold, more patches should be blurry
        assert metrics_high.n_patches_blurry >= metrics_low.n_patches_blurry


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
        mask = generate_blur_mask(img, blur_threshold=15)
        # Most of the image has sharp edges
        assert np.mean(mask) < 0.8

    def test_constant_image_high_mask(self):
        """Constant image should have mostly True (blurry) mask."""
        img = np.ones((200, 200), dtype=np.uint8) * 128
        mask = generate_blur_mask(img, blur_threshold=15)
        assert np.mean(mask) > 0.9

    def test_rejects_3d_image(self):
        """Should reject 3D images."""
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Expected 2D image"):
            generate_blur_mask(img)


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
