"""Tests for Cellpose cell segmentation."""

import numpy as np
import pytest

from aggrequant.segmentation.cellpose import CellposeSegmenter


# --- _match_cells_to_nuclei ---
# These tests use synthetic label arrays (pure numpy, no model needed).

def test_match_basic_assignment():
    # Cell 1 contains nucleus 1, cell 2 contains nucleus 2 — no conflict
    cell_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    assert set(np.unique(output)) == {0, 1, 2}
    assert np.all(output[nuclei_labels == 1] == 1)
    assert np.all(output[nuclei_labels == 2] == 2)


def test_match_conflict_best_overlap_wins():
    # Nucleus 1 spans both cells; cell 1 covers more of it → cell 1 wins
    # Cell 2 has no other nucleus → dropped
    cell_labels = np.array([
        [1, 1, 1, 2, 2, 2],
        [1, 1, 1, 2, 2, 2],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [0, 1, 1, 1, 0, 0],  # 2 pixels in cell 1 (cols 1,2), 1 pixel in cell 2 (col 3)
        [0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    matched_ids = set(np.unique(output)) - {0}
    assert matched_ids == {1}  # only cell 1 matched; cell 2 dropped


def test_match_drops_cell_without_nucleus():
    # Cell 2 has no nucleus inside it → dropped
    cell_labels = np.array([
        [1, 1, 2, 2],
        [1, 1, 2, 2],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    assert 2 not in np.unique(output)
    assert 1 in np.unique(output)


def test_match_drops_nucleus_without_cell():
    # Nucleus 2 is not inside any cell → dropped
    cell_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],  # nucleus 2 outside any cell
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    matched_ids = set(np.unique(output)) - {0}
    assert matched_ids == {1}


def test_match_output_ids_are_subset_of_nucleus_ids():
    cell_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    output_ids = set(np.unique(output)) - {0}
    nucleus_ids = set(np.unique(nuclei_labels)) - {0}
    assert output_ids.issubset(nucleus_ids)


def test_segment_zeros_unmatched_nuclei_inplace():
    # nucleus 2 is outside any cell → should be zeroed in nuclei_labels after segment()
    from unittest.mock import patch
    seg = CellposeSegmenter()
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],  # nucleus 2 — no cell will cover it
        [0, 0, 2, 2],
    ], dtype=np.int32)
    mock_cells = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],  # no cell over nucleus 2
        [0, 0, 0, 0],
    ], dtype=np.int32)
    with patch.object(seg, '_segment_with_nuclei', return_value=mock_cells):
        seg.segment(np.zeros((4, 4), dtype=np.uint16), nuclei_labels)
    assert 2 not in np.unique(nuclei_labels)
    assert 1 in np.unique(nuclei_labels)


def test_match_equal_cell_and_nucleus_count():
    cell_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    nuclei_labels = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 2, 2],
        [0, 0, 2, 2],
    ], dtype=np.int32)
    seg = CellposeSegmenter()
    output = seg._match_cells_to_nuclei(cell_labels, nuclei_labels)
    n_matched_cells = len(np.unique(output)) - 1  # exclude background
    n_matched_nuclei = len(set(np.unique(output)) & (set(np.unique(nuclei_labels)) - {0}))
    assert n_matched_cells == n_matched_nuclei


# --- segment (slow: loads Cellpose model) ---

@pytest.mark.slow
def test_segment_output_shape(cell_labels, cell_image):
    assert cell_labels.shape == cell_image.shape


@pytest.mark.slow
def test_segment_output_dtype(cell_labels):
    assert cell_labels.dtype == np.uint16


@pytest.mark.slow
def test_segment_detects_cells(cell_labels):
    assert cell_labels.max() > 0


@pytest.mark.slow
def test_segment_cell_ids_are_valid_nucleus_ids(cell_labels, nuclei_labels):
    # Every cell ID in the output must correspond to a nucleus ID in the input
    cell_ids = set(np.unique(cell_labels)) - {0}
    nucleus_ids = set(np.unique(nuclei_labels)) - {0}
    assert cell_ids.issubset(nucleus_ids)
