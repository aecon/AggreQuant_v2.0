"""Tests for nn/ inference and utility modules."""

import numpy as np
import torch
import pytest

from aggrequant.nn.inference import (
    _to_tensor,
    _pad_to_multiple,
    _gaussian_kernel_2d,
    predict_full,
    predict_tiled,
    predict,
    postprocess_predictions,
)
from aggrequant.nn.utils import get_device
from aggrequant.nn.architectures.registry import create_model


SMALL_FEATURES = [8, 16, 32, 64]


@pytest.fixture(scope="module")
def model():
    m = create_model("baseline", features=SMALL_FEATURES)
    m.eval()
    return m


@pytest.fixture(scope="module")
def image():
    """Synthetic grayscale image (100x100), not multiple of 16."""
    rng = np.random.RandomState(42)
    return rng.randint(0, 4096, size=(100, 100), dtype=np.uint16)


# --- _to_tensor ---

def test_to_tensor_shape():
    arr = np.zeros((64, 64), dtype=np.float32)
    t = _to_tensor(arr)
    assert t.shape == (1, 1, 64, 64)

def test_to_tensor_dtype():
    arr = np.zeros((64, 64), dtype=np.float64)
    t = _to_tensor(arr)
    assert t.dtype == torch.float32


# --- _pad_to_multiple ---

@pytest.mark.parametrize("h,w,expected_h,expected_w", [
    (64, 64, 64, 64),       # already multiple of 16
    (100, 100, 112, 112),   # needs padding
    (17, 33, 32, 48),       # odd sizes
    (16, 16, 16, 16),       # exact
])
def test_pad_to_multiple_shape(h, w, expected_h, expected_w):
    arr = np.zeros((h, w), dtype=np.float32)
    padded = _pad_to_multiple(arr, multiple=16)
    assert padded.shape == (expected_h, expected_w)

def test_pad_to_multiple_preserves_content():
    arr = np.ones((10, 10), dtype=np.float32) * 42.0
    padded = _pad_to_multiple(arr, multiple=16)
    np.testing.assert_array_equal(padded[:10, :10], arr)

def test_pad_to_multiple_no_copy_when_aligned():
    arr = np.zeros((32, 32), dtype=np.float32)
    padded = _pad_to_multiple(arr, multiple=16)
    assert padded is arr


# --- _gaussian_kernel_2d ---

def test_gaussian_kernel_shape():
    k = _gaussian_kernel_2d(64)
    assert k.shape == (64, 64)

def test_gaussian_kernel_dtype():
    k = _gaussian_kernel_2d(64)
    assert k.dtype == np.float32

def test_gaussian_kernel_peak_at_center():
    k = _gaussian_kernel_2d(64)
    center = k.shape[0] // 2
    assert k[center, center] == pytest.approx(1.0)

def test_gaussian_kernel_values_positive():
    k = _gaussian_kernel_2d(64)
    assert np.all(k > 0)

def test_gaussian_kernel_symmetric():
    k = _gaussian_kernel_2d(64)
    np.testing.assert_array_almost_equal(k, k.T)
    np.testing.assert_array_almost_equal(k, np.flip(k, axis=0))


# --- predict_full ---

def test_predict_full_output_shape(model, image):
    prob = predict_full(model, image, device="cpu")
    assert prob.shape == image.shape

def test_predict_full_output_dtype(model, image):
    prob = predict_full(model, image, device="cpu")
    assert prob.dtype == np.float32

def test_predict_full_output_range(model, image):
    prob = predict_full(model, image, device="cpu")
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0


# --- predict_tiled ---

def test_predict_tiled_output_shape(model, image):
    prob = predict_tiled(model, image, tile_size=32, stride=16, device="cpu")
    assert prob.shape == image.shape

def test_predict_tiled_output_dtype(model, image):
    prob = predict_tiled(model, image, tile_size=32, stride=16, device="cpu")
    assert prob.dtype == np.float32

def test_predict_tiled_output_range(model, image):
    prob = predict_tiled(model, image, tile_size=32, stride=16, device="cpu")
    assert prob.min() >= 0.0
    assert prob.max() <= 1.0


# --- predict (auto mode) ---

def test_predict_auto_output_shape(model, image):
    prob = predict(model, image, device="cpu")
    assert prob.shape == image.shape


# --- postprocess_predictions ---

def test_postprocess_output_dtype():
    prob = np.random.rand(64, 64).astype(np.float32)
    labels = postprocess_predictions(prob, threshold=0.5)
    assert labels.dtype == np.uint32

def test_postprocess_output_shape():
    prob = np.random.rand(64, 64).astype(np.float32)
    labels = postprocess_predictions(prob, threshold=0.5)
    assert labels.shape == (64, 64)

def test_postprocess_background_is_zero():
    prob = np.zeros((64, 64), dtype=np.float32)
    labels = postprocess_predictions(prob, threshold=0.5)
    assert labels.max() == 0

def test_postprocess_labels_consecutive():
    prob = np.random.rand(64, 64).astype(np.float32)
    labels = postprocess_predictions(prob, threshold=0.3)
    unique = np.unique(labels)
    np.testing.assert_array_equal(
        unique, np.arange(labels.max() + 1, dtype=np.uint32)
    )

def test_postprocess_min_size_filter():
    """A single 2x2 block (area=4) should be removed with remove_objects_below=9."""
    prob = np.zeros((64, 64), dtype=np.float32)
    prob[10:12, 10:12] = 1.0  # 4-pixel object
    labels = postprocess_predictions(prob, threshold=0.5, remove_objects_below=9)
    assert labels.max() == 0


# --- get_device ---

def test_get_device_cpu():
    d = get_device("cpu")
    assert d == torch.device("cpu")

def test_get_device_none():
    d = get_device(None)
    assert isinstance(d, torch.device)

def test_get_device_passthrough():
    d = get_device(torch.device("cpu"))
    assert d == torch.device("cpu")
