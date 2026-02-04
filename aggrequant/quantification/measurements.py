"""
Quantification measurements for aggregate analysis.

Computes quantities of interest (QoI) from segmentation masks.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np
import skimage.morphology
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .results import FieldResult


# Default parameters
MIN_AGGREGATE_AREA_PIXELS = 9
SMALL_HOLE_AREA_THRESHOLD = 25


@dataclass
class AggregateStats:
    """Statistics for a single aggregate."""
    aggregate_id: int
    total_area: int
    n_cells_overlapping: int
    is_ambiguous: bool  # overlaps multiple cells
    cell_areas: Dict[int, int]  # cell_id -> overlap area


def compute_aggregate_mask_inside_cells(
    aggregate_labels: np.ndarray,
    cell_labels: np.ndarray,
) -> np.ndarray:
    """
    Create aggregate mask excluding regions outside cells.

    Arguments:
        aggregate_labels: Instance labels for aggregates
        cell_labels: Instance labels for cells

    Returns:
        mask: Binary mask of aggregates inside cells
    """
    # Create binary mask from aggregate labels
    mask = (aggregate_labels > 0).astype(np.uint8)

    # Fill small holes in aggregate mask
    try:
        mask = skimage.morphology.remove_small_holes(
            mask.astype(bool), max_size=SMALL_HOLE_AREA_THRESHOLD, connectivity=2
        ).astype(np.uint8)
    except TypeError:
        mask = skimage.morphology.remove_small_holes(
            mask.astype(bool), area_threshold=SMALL_HOLE_AREA_THRESHOLD, connectivity=2
        ).astype(np.uint8)

    # Exclude regions outside cells
    mask[cell_labels == 0] = 0

    return mask


def compute_field_measurements(
    cell_labels: np.ndarray,
    aggregate_labels: np.ndarray,
    nuclei_labels: Optional[np.ndarray] = None,
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

    Arguments:
        cell_labels: Instance segmentation of cells (uint16)
        aggregate_labels: Instance segmentation of aggregates (uint32)
        nuclei_labels: Optional instance segmentation of nuclei
        min_aggregate_area: Minimum aggregate area in pixels
        verbose: Print progress messages
        debug: Print detailed debug information

    Returns:
        result: FieldResult with all measurements
        diagnostics: Dictionary with diagnostic images/data
    """
    me = "compute_field_measurements"

    # Create aggregate mask inside cells
    mask_agg = compute_aggregate_mask_inside_cells(aggregate_labels, cell_labels)

    # Re-label aggregates inside cells (connected components)
    labels_agg_inside = skimage.morphology.label(mask_agg, connectivity=2)

    if debug:
        print(f"({me}) Aggregates inside cells: {labels_agg_inside.max()}")

    # Cell mask
    mask_cell = (cell_labels > 0).astype(np.uint8)

    # Get unique cell and aggregate IDs
    unique_cells = np.unique(cell_labels[cell_labels > 0])
    unique_aggregates = np.unique(labels_agg_inside[labels_agg_inside > 0])

    n_cells = len(unique_cells)
    n_aggregates = len(unique_aggregates)

    if verbose:
        print(f"({me}) Found {n_cells} cells, {n_aggregates} aggregates inside cells")

    # Initialize tracking arrays
    aggregates_per_cell = np.zeros(n_cells)  # Number of aggregates per cell
    cells_per_aggregate = np.zeros(n_aggregates)  # Number of cells per aggregate

    # Diagnostic overlay images
    overlay_cells_agg = np.zeros(cell_labels.shape, dtype=np.float32)
    overlay_cells_agg[mask_cell > 0] = -1

    overlay_nagg_per_cell = np.ones(cell_labels.shape, dtype=np.float32) * -1
    overlay_nagg_per_cell[mask_cell > 0] = 0

    # Process each aggregate
    for ia, agg_id in enumerate(unique_aggregates):
        # Get aggregate region
        idx_agg = labels_agg_inside == agg_id
        total_agg_area = np.sum(idx_agg)

        # Find cells under this aggregate
        cell_ids_under_agg = cell_labels[idx_agg]
        unique_cell_ids = np.unique(cell_ids_under_agg[cell_ids_under_agg > 0])

        if len(unique_cell_ids) == 0:
            continue

        # Track overlap with each cell
        for cell_id in unique_cell_ids:
            # Area of aggregate over this cell
            agg_area_in_cell = np.sum(cell_ids_under_agg == cell_id)

            # Only count if aggregate has minimum area in this cell
            if agg_area_in_cell >= min_aggregate_area:
                # Find cell index in unique_cells
                cell_idx = np.where(unique_cells == cell_id)[0]
                if len(cell_idx) > 0:
                    aggregates_per_cell[cell_idx[0]] += 1
                    cells_per_aggregate[ia] += 1

                    # Update diagnostic overlays
                    cell_mask = cell_labels == cell_id
                    overlay_cells_agg[cell_mask & ~idx_agg] = -2
                    overlay_nagg_per_cell[cell_mask] = aggregates_per_cell[cell_idx[0]]

        # Color aggregate by number of cells it overlaps
        overlay_cells_agg[idx_agg] = cells_per_aggregate[ia]

    # Clean up diagnostic images
    overlay_cells_agg[mask_cell == 0] = 0
    overlay_nagg_per_cell[mask_agg > 0] = -2

    # Compute final statistics
    total_cell_area = np.sum(mask_cell)
    total_agg_area = np.sum(mask_agg)

    # Number of aggregate-positive cells
    n_agg_positive_cells = np.sum(aggregates_per_cell > 0)

    # Percentage of aggregate-positive cells
    pct_agg_positive = (n_agg_positive_cells / n_cells * 100) if n_cells > 0 else 0.0

    # Percentage of cell area covered by aggregates
    pct_area_agg = (total_agg_area / total_cell_area * 100) if total_cell_area > 0 else 0.0

    # Percentage of ambiguous aggregates
    n_ambiguous = np.sum(cells_per_aggregate > 1)
    pct_ambiguous = (n_ambiguous / n_aggregates * 100) if n_aggregates > 0 else 0.0

    # Average aggregates per positive cell
    positive_cells_agg_counts = aggregates_per_cell[aggregates_per_cell > 0]
    avg_agg_per_positive = float(np.mean(positive_cells_agg_counts)) if len(positive_cells_agg_counts) > 0 else 0.0

    # Count nuclei if provided
    n_nuclei = 0
    if nuclei_labels is not None:
        n_nuclei = len(np.unique(nuclei_labels[nuclei_labels > 0]))

    # Create result
    result = FieldResult(
        plate_name="",  # To be filled by caller
        well_id="",  # To be filled by caller
        row="",  # To be filled by caller
        column=0,  # To be filled by caller
        field=0,  # To be filled by caller
        n_cells=n_cells,
        n_nuclei=n_nuclei,
        total_cell_area_px=float(total_cell_area),
        n_aggregates=n_aggregates,
        n_aggregate_positive_cells=n_agg_positive_cells,
        pct_aggregate_positive_cells=pct_agg_positive,
        total_aggregate_area_px=float(total_agg_area),
        pct_aggregate_area_over_cell=pct_area_agg,
        avg_aggregates_per_positive_cell=avg_agg_per_positive,
        pct_ambiguous_aggregates=pct_ambiguous,
    )

    # Diagnostic data
    diagnostics = {
        "overlay_cells_agg": overlay_cells_agg,
        "overlay_nagg_per_cell": overlay_nagg_per_cell,
        "labels_agg_inside_cells": labels_agg_inside,
        "aggregates_per_cell": aggregates_per_cell,
        "cells_per_aggregate": cells_per_aggregate,
    }

    return result, diagnostics


def compute_masked_measurements(
    cell_labels: np.ndarray,
    aggregate_labels: np.ndarray,
    blur_mask: np.ndarray,
    min_aggregate_area: int = MIN_AGGREGATE_AREA_PIXELS,
) -> Dict[str, float]:
    """
    Compute measurements excluding blurry regions.

    Arguments:
        cell_labels: Instance segmentation of cells
        aggregate_labels: Instance segmentation of aggregates
        blur_mask: Binary mask where 1 = blurry (to be excluded)
        min_aggregate_area: Minimum aggregate area in pixels

    Returns:
        Dictionary with masked measurements
    """
    # Create valid (non-blurry) mask
    valid_mask = blur_mask == 0

    # Apply mask to cell labels
    cell_labels_masked = cell_labels.copy()
    cell_labels_masked[~valid_mask] = 0

    # Get cells that still have significant area after masking
    unique_cells = np.unique(cell_labels[cell_labels > 0])
    valid_cells = []

    for cell_id in unique_cells:
        original_area = np.sum(cell_labels == cell_id)
        masked_area = np.sum(cell_labels_masked == cell_id)

        # Keep cell if at least 50% of its area is in valid region
        if masked_area >= 0.5 * original_area:
            valid_cells.append(cell_id)

    n_cells_masked = len(valid_cells)

    # Create aggregate mask inside valid cells
    mask_agg = compute_aggregate_mask_inside_cells(aggregate_labels, cell_labels_masked)
    labels_agg_inside = skimage.morphology.label(mask_agg, connectivity=2)

    # Count aggregate-positive cells among valid cells
    n_agg_positive_masked = 0

    for cell_id in valid_cells:
        cell_region = cell_labels_masked == cell_id
        agg_area_in_cell = np.sum(mask_agg[cell_region])

        if agg_area_in_cell >= min_aggregate_area:
            n_agg_positive_masked += 1

    # Calculate percentages
    pct_agg_positive_masked = (n_agg_positive_masked / n_cells_masked * 100) if n_cells_masked > 0 else 0.0

    # Calculate masked areas
    total_cell_area_masked = np.sum(cell_labels_masked > 0)
    total_agg_area_masked = np.sum(mask_agg)

    return {
        "n_cells_masked": n_cells_masked,
        "n_aggregate_positive_cells_masked": n_agg_positive_masked,
        "pct_aggregate_positive_cells_masked": pct_agg_positive_masked,
        "total_cell_area_masked_px": float(total_cell_area_masked),
        "total_aggregate_area_masked_px": float(total_agg_area_masked),
    }


def apply_focus_metrics_to_result(
    result: FieldResult,
    focus_metrics: "FocusMetrics",
    blur_threshold: float = 15.0,
) -> FieldResult:
    """
    Add focus quality metrics to a field result.

    Arguments:
        result: FieldResult to update
        focus_metrics: FocusMetrics from quality module
        blur_threshold: Threshold for variance of Laplacian

    Returns:
        Updated FieldResult
    """
    result.focus_variance_laplacian_mean = focus_metrics.variance_of_laplacian_mean
    result.focus_variance_laplacian_min = focus_metrics.variance_of_laplacian_min
    result.focus_pct_patches_blurry = focus_metrics.pct_patches_below_threshold
    result.focus_pct_area_blurry = focus_metrics.pct_patches_below_threshold  # Approximation
    result.focus_is_likely_blurry = focus_metrics.is_likely_blurry
    result.blur_threshold_used = blur_threshold

    return result
