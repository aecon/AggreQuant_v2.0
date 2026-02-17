"""Post-processing utilities for segmentation label maps."""

import numpy as np
import skimage.morphology


def remove_border_objects(
    cell_labels: np.ndarray, nuclei_labels: np.ndarray
) -> tuple:
    """
    Remove cells touching the image border and their corresponding nuclei.

    Assumes cell and nuclei labels share IDs (i.e., cell N corresponds
    to nucleus N), as guaranteed by CellposeSegmenter._match_cells_to_nuclei.

    Arguments:
        cell_labels: Instance segmentation of cells
        nuclei_labels: Instance segmentation of nuclei (same IDs as cells)

    Returns:
        Tuple of (cell_labels, nuclei_labels) with border objects zeroed out
    """
    cell_labels = cell_labels.copy()
    nuclei_labels = nuclei_labels.copy()

    border_ids = set()
    border_ids.update(np.unique(cell_labels[0, :]))    # top
    border_ids.update(np.unique(cell_labels[-1, :]))   # bottom
    border_ids.update(np.unique(cell_labels[:, 0]))    # left
    border_ids.update(np.unique(cell_labels[:, -1]))   # right
    border_ids.discard(0)

    for label_id in border_ids:
        cell_labels[cell_labels == label_id] = 0
        nuclei_labels[nuclei_labels == label_id] = 0

    return cell_labels, nuclei_labels


def filter_aggregates_by_cells(
    aggregate_labels: np.ndarray, cell_labels: np.ndarray
) -> np.ndarray:
    """
    Remove aggregates that are outside detected cells and relabel.

    Arguments:
        aggregate_labels: Instance segmentation of aggregates
        cell_labels: Instance segmentation of cells

    Returns:
        Relabeled aggregate array with only in-cell aggregates
    """
    cell_mask = cell_labels > 0

    aggregate_labels = aggregate_labels.copy()
    aggregate_labels[~cell_mask] = 0

    # Relabel to remove gaps from deleted aggregates
    aggregate_labels = skimage.morphology.label(aggregate_labels > 0)
    return aggregate_labels.astype(np.uint32)


def relabel_consecutive(
    nuclei_labels: np.ndarray, cell_labels: np.ndarray
) -> tuple:
    """
    Relabel nuclei and cells to consecutive IDs, maintaining correspondence.

    Builds a lookup table from unique nucleus IDs to consecutive IDs
    (1, 2, 3, ...) and applies it to both arrays. This preserves the
    cell-nucleus ID correspondence guaranteed by CellposeSegmenter.

    Arguments:
        nuclei_labels: Instance segmentation of nuclei
        cell_labels: Instance segmentation of cells (same IDs as nuclei)

    Returns:
        Tuple of (nuclei_labels, cell_labels) with consecutive IDs
    """
    unique_ids = np.unique(nuclei_labels[nuclei_labels > 0])

    max_id = max(nuclei_labels.max(), cell_labels.max())
    lookup = np.zeros(max_id + 1, dtype=nuclei_labels.dtype)
    for new_id, old_id in enumerate(sorted(unique_ids), start=1):
        lookup[old_id] = new_id

    return lookup[nuclei_labels], lookup[cell_labels]
