"""Tests for StarDist nuclei segmentation."""

import numpy as np
import pytest

from aggrequant.segmentation.stardist import StarDistSegmenter


@pytest.fixture(scope="module")
def segmenter():
    return StarDistSegmenter()


@pytest.fixture(scope="module")
def preprocessed(segmenter, nuclei_image):
    return segmenter._preprocess(nuclei_image)


# --- _preprocess ---

def test_preprocess_output_shape(preprocessed, nuclei_image):
    assert preprocessed.shape == nuclei_image.shape

def test_preprocess_output_dtype(preprocessed):
    assert preprocessed.dtype == np.float32

def test_preprocess_output_is_finite(preprocessed):
    assert np.all(np.isfinite(preprocessed))

def test_preprocess_output_is_positive(preprocessed):
    assert np.all(preprocessed > 0)


# --- _postprocess_size_exclusion ---

def make_label_image(shape, nuclei):
    """Synthetic label image with circular nuclei: list of (row, col, radius, label_id)."""
    labels = np.zeros(shape, dtype=np.int32)
    rr, cc = np.mgrid[0:shape[0], 0:shape[1]]
    for row, col, radius, label_id in nuclei:
        labels[(rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2] = label_id
    return labels


def test_size_exclusion_removes_small_nucleus():
    seg = StarDistSegmenter(min_area=300, max_area=15000)
    labels = make_label_image((128, 128), [(64, 64, 5, 1)])  # ~78 px, below min
    result = seg._postprocess_size_exclusion(labels.copy())
    assert result.max() == 0


def test_size_exclusion_removes_large_nucleus():
    seg = StarDistSegmenter(min_area=300, max_area=15000)
    labels = make_label_image((256, 256), [(128, 128, 75, 1)])  # ~17670 px, above max
    result = seg._postprocess_size_exclusion(labels.copy())
    assert result.max() == 0


def test_size_exclusion_keeps_valid_nucleus():
    seg = StarDistSegmenter(min_area=300, max_area=15000)
    labels = make_label_image((128, 128), [(64, 64, 20, 1)])  # ~1256 px, in range
    result = seg._postprocess_size_exclusion(labels.copy())
    assert 1 in np.unique(result)


def test_size_exclusion_drops_only_invalid():
    seg = StarDistSegmenter(min_area=300, max_area=15000)
    labels = make_label_image((128, 128), [
        (32, 32, 20, 1),  # valid
        (96, 96,  5, 2),  # too small
    ])
    result = seg._postprocess_size_exclusion(labels.copy())
    assert 1 in np.unique(result)
    assert 2 not in np.unique(result)


# --- _postprocess_increase_borders ---

def test_border_separation_creates_gap_between_touching_nuclei():
    seg = StarDistSegmenter()
    labels = make_label_image((128, 128), [
        (64, 44, 20, 1),
        (64, 84, 20, 2),
    ])
    result = seg._postprocess_increase_borders(labels.copy())
    assert result[64, 64] == 0  # contact zone zeroed


def test_border_separation_preserves_nucleus_interior():
    seg = StarDistSegmenter()
    labels = make_label_image((128, 128), [(64, 64, 20, 1)])
    result = seg._postprocess_increase_borders(labels.copy())
    assert result[64, 64] == 1


# --- segment (slow: loads StarDist model) ---

@pytest.mark.slow
def test_segment_output_shape(nuclei_labels, nuclei_image):
    assert nuclei_labels.shape == nuclei_image.shape

@pytest.mark.slow
def test_segment_output_dtype(nuclei_labels):
    assert nuclei_labels.dtype == np.uint16

@pytest.mark.slow
def test_segment_detects_nuclei(nuclei_labels):
    assert nuclei_labels.max() > 0

@pytest.mark.slow
def test_segment_labels_are_consecutive(nuclei_labels):
    unique = np.unique(nuclei_labels)
    np.testing.assert_array_equal(unique, np.arange(nuclei_labels.max() + 1, dtype=np.uint16))
