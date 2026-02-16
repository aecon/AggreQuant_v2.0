"""
Quantification measurements for aggregate analysis.

Computes quantities of interest (QoI) from segmentation masks.

Author: Athena Economides, 2026, UZH
"""

import numpy as np
import skimage.morphology
from scipy import sparse
from typing import Optional, Tuple, Dict

from .results import FieldResult
from aggrequant.common.logging import get_logger
from aggrequant.common.image_utils import remove_small_holes_compat

logger = get_logger(__name__)


# Default parameters
MIN_AGGREGATE_AREA_PIXELS = 9
SMALL_HOLE_AREA_THRESHOLD = 25


def compute_field_measurements(
    cell_labels: np.ndarray,
    aggregate_labels: np.ndarray,
    nuclei_labels: Optional[np.ndarray] = None,
    blur_mask: Optional[np.ndarray] = None,
    min_aggregate_area: int = MIN_AGGREGATE_AREA_PIXELS,
    verbose: bool = False,
    debug: bool = False,
) -> Tuple[FieldResult, Dict[str, np.ndarray]]:
    """
    Compute all quantities of interest for a single field.

    Quantities computed:
    1. Percentage of aggregate-positive cells
    2. Number of cells per image
    3. Total area of aggregates (as percentage of cell area)
    4. Percentage of ambiguous aggregates (spanning multiple cells)
    5. Number of aggregates per image
    6. Average number of aggregates per aggregate-positive cell

    When blur_mask is provided, additionally computes masked metrics
    (cell counts, aggregate-positive counts, areas) excluding blurry regions.

    Uses a single-pass sparse cross-tabulation instead of per-aggregate
    loops, giving O(pixels) complexity instead of O(n_aggregates * pixels).

    Arguments:
        cell_labels: Instance segmentation of cells (uint16)
        aggregate_labels: Instance segmentation of aggregates (uint32)
        nuclei_labels: Optional instance segmentation of nuclei
        blur_mask: Optional binary mask where 1 = blurry (to be excluded)
        min_aggregate_area: Minimum aggregate area in pixels
        verbose: Print progress messages
        debug: Print detailed debug information

    Returns:
        result: FieldResult with all measurements
        diagnostics: Dictionary with diagnostic images/data
    """
    labels_agg_inside = aggregate_labels

    if debug:
        logger.debug(f"Aggregates inside cells: {labels_agg_inside.max()}")

    # Use bincount for O(max_label) instead of np.unique's O(P log P)
    cell_counts = np.bincount(cell_labels.ravel())
    agg_counts = np.bincount(labels_agg_inside.ravel())

    unique_cells = np.nonzero(cell_counts)[0]
    unique_cells = unique_cells[unique_cells > 0]
    unique_aggregates = np.nonzero(agg_counts)[0]
    unique_aggregates = unique_aggregates[unique_aggregates > 0]

    n_cells = len(unique_cells)
    n_aggregates = len(unique_aggregates)

    total_cell_area = int(cell_counts[1:].sum()) if len(cell_counts) > 1 else 0
    total_agg_area = int(agg_counts[1:].sum()) if len(agg_counts) > 1 else 0

    if verbose:
        logger.info(f"Found {n_cells} cells, {n_aggregates} aggregates inside cells")

    if n_cells == 0 or n_aggregates == 0:
        aggregates_per_cell = np.zeros(n_cells)
        cells_per_aggregate = np.zeros(n_aggregates)
    else:
        # Single-pass sparse cross-tabulation: overlap[agg_id, cell_id] = pixel count
        both_mask = (labels_agg_inside > 0) & (cell_labels > 0)
        agg_flat = labels_agg_inside[both_mask].astype(np.int64)
        cell_flat = cell_labels[both_mask].astype(np.int64)

        max_agg_id = int(labels_agg_inside.max())
        max_cell_id = int(cell_labels.max())

        overlap = sparse.coo_matrix(
            (np.ones(agg_flat.size, dtype=np.int32), (agg_flat, cell_flat)),
            shape=(max_agg_id + 1, max_cell_id + 1),
        ).tocsr()

        # Extract unique pairs and their counts via COO representation
        overlap_coo = overlap.tocoo()

        # Filter pairs by min_aggregate_area threshold
        valid = overlap_coo.data >= min_aggregate_area
        valid_agg_ids = overlap_coo.row[valid]
        valid_cell_ids = overlap_coo.col[valid]

        # Build label-ID → consecutive-index lookup tables
        agg_id_to_idx = np.empty(max_agg_id + 1, dtype=np.intp)
        agg_id_to_idx[unique_aggregates] = np.arange(n_aggregates)

        cell_id_to_idx = np.empty(max_cell_id + 1, dtype=np.intp)
        cell_id_to_idx[unique_cells] = np.arange(n_cells)

        valid_agg_idx = agg_id_to_idx[valid_agg_ids]
        valid_cell_idx = cell_id_to_idx[valid_cell_ids]

        # Fully vectorized counting
        aggregates_per_cell = np.bincount(valid_cell_idx, minlength=n_cells).astype(float)
        cells_per_aggregate = np.bincount(valid_agg_idx, minlength=n_aggregates).astype(float)

    # Diagnostic overlay images (only allocated in debug mode)
    if debug:
        mask_cell = cell_labels > 0
        mask_agg = labels_agg_inside > 0
        max_agg_id = int(labels_agg_inside.max()) if n_aggregates > 0 else 0
        max_cell_id = int(cell_labels.max()) if n_cells > 0 else 0

        # overlay_cells_agg: background=0, cell-no-agg=-1, cell-with-agg=-2, agg=cpa
        cell_overlay_lut = np.zeros(max_cell_id + 1, dtype=np.float32)
        cell_overlay_lut[unique_cells] = -1
        if n_aggregates > 0 and n_cells > 0:
            # Cells with at least one valid aggregate → -2
            positive_cell_ids = unique_cells[aggregates_per_cell > 0]
            cell_overlay_lut[positive_cell_ids] = -2

        overlay_cells_agg = cell_overlay_lut[cell_labels]

        # Paint aggregate pixels with their cells_per_aggregate value
        if n_aggregates > 0:
            agg_cpa_lut = np.zeros(max_agg_id + 1, dtype=np.float32)
            agg_cpa_lut[unique_aggregates] = cells_per_aggregate
            overlay_cells_agg[mask_agg] = agg_cpa_lut[labels_agg_inside[mask_agg]]

        # overlay_nagg_per_cell: background=-1, cell=count, agg=-2
        nagg_lut = np.full(max_cell_id + 1, -1.0, dtype=np.float32)
        nagg_lut[unique_cells] = aggregates_per_cell
        overlay_nagg_per_cell = nagg_lut[cell_labels]
        overlay_nagg_per_cell[mask_agg] = -2

    # Compute final statistics
    n_agg_positive_cells = int(np.sum(aggregates_per_cell > 0))
    pct_agg_positive = (n_agg_positive_cells / n_cells * 100) if n_cells > 0 else 0.0
    pct_area_agg = (total_agg_area / total_cell_area * 100) if total_cell_area > 0 else 0.0

    n_ambiguous = int(np.sum(cells_per_aggregate > 1))
    pct_ambiguous = (n_ambiguous / n_aggregates * 100) if n_aggregates > 0 else 0.0

    positive_cells_agg_counts = aggregates_per_cell[aggregates_per_cell > 0]
    avg_agg_per_positive = float(np.mean(positive_cells_agg_counts)) if len(positive_cells_agg_counts) > 0 else 0.0

    # Count nuclei and compute nuclei area if provided
    n_nuclei = 0
    total_nuclei_area = 0
    if nuclei_labels is not None:
        nuc_counts = np.bincount(nuclei_labels.ravel())
        n_nuclei = int(np.sum(nuc_counts[1:] > 0)) if len(nuc_counts) > 1 else 0
        total_nuclei_area = int(nuc_counts[1:].sum()) if len(nuc_counts) > 1 else 0

    # Blur-masked metrics (when blur_mask is provided)
    n_cells_masked = None
    n_agg_positive_masked = None
    pct_agg_positive_masked = None
    total_cell_area_masked = None
    total_agg_area_masked = None

    if blur_mask is not None:
        cell_labels_masked = cell_labels.copy()
        cell_labels_masked[blur_mask != 0] = 0

        masked_counts = np.bincount(cell_labels_masked.ravel(), minlength=len(cell_counts))

        # Valid cells: at least 50% of area in non-blurry region
        ratios = np.zeros(len(cell_counts), dtype=np.float64)
        nonzero = cell_counts > 0
        ratios[nonzero] = masked_counts[nonzero] / cell_counts[nonzero]
        valid_cell_ids = np.where((ratios >= 0.5) & (np.arange(len(ratios)) > 0))[0]
        n_cells_masked = len(valid_cell_ids)

        # Aggregate mask inside non-blurry cell regions
        mask_agg_masked = (aggregate_labels > 0).astype(np.uint8)
        mask_agg_masked = remove_small_holes_compat(
            mask_agg_masked, area_threshold=SMALL_HOLE_AREA_THRESHOLD, connectivity=2,
        ).astype(np.uint8)
        mask_agg_masked[cell_labels_masked == 0] = 0

        # Aggregate area per cell via weighted bincount
        agg_area_per_cell = np.bincount(
            cell_labels_masked.ravel(),
            weights=mask_agg_masked.ravel().astype(np.float64),
            minlength=len(cell_counts),
        )
        n_agg_positive_masked = int(np.sum(agg_area_per_cell[valid_cell_ids] >= min_aggregate_area))
        pct_agg_positive_masked = (n_agg_positive_masked / n_cells_masked * 100) if n_cells_masked > 0 else 0.0

        total_cell_area_masked = float(int(masked_counts[1:].sum()) if len(masked_counts) > 1 else 0)
        total_agg_area_masked = float(int(np.sum(mask_agg_masked)))

    # Create result
    result = FieldResult(
        plate_name="",  # To be filled by caller
        well_id="",  # To be filled by caller
        row="",  # To be filled by caller
        column=0,  # To be filled by caller
        field=0,  # To be filled by caller
        n_cells=n_cells,
        n_nuclei=n_nuclei,
        total_nuclei_area_px=float(total_nuclei_area),
        total_cell_area_px=float(total_cell_area),
        n_aggregates=n_aggregates,
        n_aggregate_positive_cells=n_agg_positive_cells,
        pct_aggregate_positive_cells=pct_agg_positive,
        total_aggregate_area_px=float(total_agg_area),
        pct_aggregate_area_over_cell=pct_area_agg,
        avg_aggregates_per_positive_cell=avg_agg_per_positive,
        pct_ambiguous_aggregates=pct_ambiguous,
        n_cells_masked=n_cells_masked,
        n_aggregate_positive_cells_masked=n_agg_positive_masked,
        pct_aggregate_positive_cells_masked=pct_agg_positive_masked,
        total_cell_area_masked_px=total_cell_area_masked,
        total_aggregate_area_masked_px=total_agg_area_masked,
    )

    # Diagnostic data
    diagnostics = {
        "labels_agg_inside_cells": labels_agg_inside,
        "aggregates_per_cell": aggregates_per_cell,
        "cells_per_aggregate": cells_per_aggregate,
    }
    if debug:
        diagnostics["overlay_cells_agg"] = overlay_cells_agg
        diagnostics["overlay_nagg_per_cell"] = overlay_nagg_per_cell

    return result, diagnostics


def apply_focus_metrics_to_result(
    result: FieldResult,
    focus_maps: dict,
    blur_threshold: float = 15.0,
) -> FieldResult:
    """
    Add focus quality metrics to a field result.

    Arguments:
        result: FieldResult to update
        focus_maps: Dict from compute_patch_focus_maps containing "VarianceLaplacian" key
        blur_threshold: Threshold for variance of Laplacian

    Returns:
        Updated FieldResult
    """
    var_lap = focus_maps["VarianceLaplacian"]
    n_patches = var_lap.size
    n_blurry = int(np.sum(var_lap < blur_threshold))
    pct_blurry = (n_blurry / n_patches * 100) if n_patches > 0 else 0.0

    result.focus_variance_laplacian_mean = float(np.mean(var_lap))
    result.focus_variance_laplacian_min = float(np.min(var_lap))
    result.focus_pct_patches_blurry = pct_blurry
    result.focus_pct_area_blurry = pct_blurry  # Same since non-overlapping patches
    result.focus_is_likely_blurry = pct_blurry > 50
    result.blur_threshold_used = blur_threshold

    return result
