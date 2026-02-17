"""Colocalization of cell, aggregate, and nuclei segmentation masks."""

import numpy as np
from scipy import sparse

from aggrequant.segmentation.postprocessing import count_labels

# Default parameters
MIN_AGGREGATE_AREA_PIXELS = 9


def build_overlap_table(cell_labels, agg_labels):
    """
    Cross-tabulate cell and aggregate labels at every pixel.

    For each pixel where both a cell and an aggregate are present,
    records a co-occurrence. The result is a sparse matrix where
    table[agg_id, cell_id] = number of overlapping pixels.

    Arguments:
        cell_labels: 2D cell instance segmentation (0 = background)
        agg_labels: 2D aggregate instance segmentation (0 = background)

    Returns:
        Sparse CSR matrix of shape (max_agg_id+1, max_cell_id+1)
    """
    mask = (cell_labels > 0) & (agg_labels > 0)
    agg_ids = agg_labels[mask].astype(np.int64)
    cell_ids = cell_labels[mask].astype(np.int64)

    table = sparse.coo_matrix(
        (np.ones(agg_ids.size, dtype=np.int32), (agg_ids, cell_ids)),
        shape=(int(agg_labels.max()) + 1, int(cell_labels.max()) + 1),
    ).tocsr()  # COO->CSR sums duplicate (agg, cell) entries -> pixel counts

    return table


def count_positive_cells(overlap_table, min_area):
    """
    Count cells containing at least one aggregate with sufficient overlap.

    An aggregate is assigned to a cell only if their overlap is >= min_area
    pixels. A cell is "aggregate-positive" if it has at least one such
    aggregate.

    Arguments:
        overlap_table: sparse matrix from build_overlap_table
        min_area: minimum overlap in pixels for an aggregate to count

    Returns:
        Number of aggregate-positive cells
    """
    coo = overlap_table.tocoo()
    valid_cell_ids = coo.col[coo.data >= min_area]
    if valid_cell_ids.size == 0:
        return 0
    return len(np.unique(valid_cell_ids))


def quantify_field(
    cell_labels,
    aggregate_labels,
    nuclei_labels=None,
    min_aggregate_area=MIN_AGGREGATE_AREA_PIXELS,
):
    """
    Quantify a single field of view from its segmentation masks.

    Counts cells, nuclei, and aggregates, then colocalizes the cell and
    aggregate masks to determine which cells are aggregate-positive.

    Arguments:
        cell_labels: instance segmentation of cells (0 = background)
        aggregate_labels: instance segmentation of aggregates (0 = background)
        nuclei_labels: optional instance segmentation of nuclei
        min_aggregate_area: minimum overlap pixels for an aggregate to count

    Returns:
        Dictionary with keys: n_cells, n_nuclei, n_aggregates,
        n_aggregate_positive_cells, pct_aggregate_positive_cells,
        total_cell_area_px, total_aggregate_area_px
    """
    n_cells, total_cell_area = count_labels(cell_labels)
    n_aggs, total_agg_area = count_labels(aggregate_labels)
    n_nuclei = 0
    if nuclei_labels is not None:
        n_nuclei, _ = count_labels(nuclei_labels)

    if n_cells == 0 or n_aggs == 0:
        n_positive = 0
    else:
        overlap = build_overlap_table(cell_labels, aggregate_labels)
        n_positive = count_positive_cells(overlap, min_aggregate_area)

    pct_positive = (n_positive / n_cells * 100) if n_cells > 0 else 0.0

    return {
        "n_cells": n_cells,
        "n_nuclei": n_nuclei,
        "n_aggregates": n_aggs,
        "n_aggregate_positive_cells": n_positive,
        "pct_aggregate_positive_cells": pct_positive,
        "total_cell_area_px": float(total_cell_area),
        "total_aggregate_area_px": float(total_agg_area),
    }
