"""Tests for filter-based aggregate segmentation."""

import numpy as np
import pytest

from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter


@pytest.fixture(scope="module")
def segmenter():
    return FilterBasedSegmenter()


@pytest.fixture(scope="module")
def normalized(segmenter, aggregate_image):
    return segmenter._normalize_background(aggregate_image)


@pytest.fixture(scope="module")
def thresholded(segmenter, normalized):
    return segmenter._threshold(normalized)


@pytest.fixture(scope="module")
def aggregate_labels(segmenter, aggregate_image):
    return segmenter.segment(aggregate_image)


# --- _normalize_background ---

def test_normalize_background_output_shape(normalized, aggregate_image):
    assert normalized.shape == aggregate_image.shape

def test_normalize_background_output_dtype(normalized):
    assert normalized.dtype == np.float32

def test_normalize_background_output_is_finite(normalized):
    assert np.all(np.isfinite(normalized))

def test_normalize_background_output_is_positive(normalized):
    assert np.all(normalized >= 0)


# --- _threshold ---

def test_threshold_output_shape(thresholded, aggregate_image):
    assert thresholded.shape == aggregate_image.shape

def test_threshold_output_dtype(thresholded):
    assert thresholded.dtype == np.uint8

def test_threshold_output_is_binary(thresholded):
    assert set(np.unique(thresholded)).issubset({0, 1})


# --- segment ---

def test_segment_output_shape(aggregate_labels, aggregate_image):
    assert aggregate_labels.shape == aggregate_image.shape

def test_segment_output_dtype(aggregate_labels):
    assert aggregate_labels.dtype == np.uint32

def test_segment_background_is_zero(aggregate_labels):
    assert aggregate_labels.min() == 0

def test_segment_detects_aggregates(aggregate_labels):
    assert aggregate_labels.max() > 0

def test_segment_labels_are_consecutive(aggregate_labels):
    unique = np.unique(aggregate_labels)
    np.testing.assert_array_equal(unique, np.arange(aggregate_labels.max() + 1, dtype=np.uint32))
