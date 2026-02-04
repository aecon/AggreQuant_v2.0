"""
Well-level statistics aggregation.

Aggregates field-level measurements to well-level statistics.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np
from typing import List, Optional

from ..quantification.results import FieldResult, WellResult


def aggregate_field_to_well(
    field_results: List[FieldResult],
    plate_name: str,
    well_id: str,
    row: str,
    column: int,
    control_type: Optional[str] = None,
) -> WellResult:
    """
    Aggregate multiple field results into a single well result.

    The aggregation strategy:
    - Cell counts: Sum across all fields
    - Percentage of positive cells: Weighted by cell count
    - Area measurements: Sum across all fields
    - Focus metrics: Average across fields

    Arguments:
        field_results: List of FieldResult for each field in the well
        plate_name: Name of the plate
        well_id: Well identifier (e.g., "A01")
        row: Row letter (e.g., "A")
        column: Column number (e.g., 1)
        control_type: Optional control type assignment

    Returns:
        WellResult with aggregated statistics
    """
    n_fields = len(field_results)

    if n_fields == 0:
        return WellResult(
            plate_name=plate_name,
            well_id=well_id,
            row=row,
            column=column,
            n_fields=0,
            control_type=control_type,
        )

    # Extract arrays for aggregation
    n_cells_arr = np.array([f.n_cells for f in field_results])
    n_nuclei_arr = np.array([f.n_nuclei for f in field_results])
    cell_area_arr = np.array([f.total_cell_area_px for f in field_results])
    n_agg_arr = np.array([f.n_aggregates for f in field_results])
    n_agg_pos_arr = np.array([f.n_aggregate_positive_cells for f in field_results])
    pct_pos_arr = np.array([f.pct_aggregate_positive_cells for f in field_results])
    agg_area_arr = np.array([f.total_aggregate_area_px for f in field_results])
    pct_area_arr = np.array([f.pct_aggregate_area_over_cell for f in field_results])
    avg_agg_per_cell_arr = np.array([f.avg_aggregates_per_positive_cell for f in field_results])

    # Focus metrics
    focus_values = [f.focus_variance_laplacian_mean for f in field_results if f.focus_variance_laplacian_mean is not None]
    blurry_flags = [f.focus_is_likely_blurry for f in field_results if f.focus_is_likely_blurry is not None]

    # Masked metrics
    n_cells_masked_arr = [f.n_cells_masked for f in field_results if f.n_cells_masked is not None]
    pct_pos_masked_arr = [f.pct_aggregate_positive_cells_masked for f in field_results if f.pct_aggregate_positive_cells_masked is not None]

    # Total counts (simple sums)
    total_n_cells = int(np.sum(n_cells_arr))
    total_n_nuclei = int(np.sum(n_nuclei_arr))
    total_cell_area = float(np.sum(cell_area_arr))
    total_n_agg = int(np.sum(n_agg_arr))
    total_n_agg_pos = int(np.sum(n_agg_pos_arr))
    total_agg_area = float(np.sum(agg_area_arr))

    # Weighted percentage of positive cells
    # (Sum of positive cells) / (Sum of all cells)
    pct_aggregate_positive = (total_n_agg_pos / total_n_cells * 100) if total_n_cells > 0 else 0.0

    # Weighted percentage of aggregate area over cell area
    # (Sum of aggregate area) / (Sum of cell area)
    pct_aggregate_area = (total_agg_area / total_cell_area * 100) if total_cell_area > 0 else 0.0

    # Average aggregates per positive cell (weighted by count)
    # For cells with aggregates, compute weighted average
    total_agg_in_pos_cells = np.sum(n_agg_pos_arr * avg_agg_per_cell_arr)
    avg_agg_per_positive = (total_agg_in_pos_cells / total_n_agg_pos) if total_n_agg_pos > 0 else 0.0

    # Focus metrics aggregation
    avg_focus = float(np.mean(focus_values)) if focus_values else None
    n_blurry = sum(1 for b in blurry_flags if b) if blurry_flags else 0
    pct_blurry = (n_blurry / len(blurry_flags) * 100) if blurry_flags else 0.0

    # Masked metrics aggregation
    total_n_cells_masked = int(np.sum(n_cells_masked_arr)) if n_cells_masked_arr else None

    # For masked percentage, compute weighted average
    pct_pos_masked = None
    if n_cells_masked_arr and pct_pos_masked_arr:
        # Weighted by number of masked cells per field
        weighted_sum = sum(n * p for n, p in zip(n_cells_masked_arr, pct_pos_masked_arr))
        total_masked = sum(n_cells_masked_arr)
        pct_pos_masked = (weighted_sum / total_masked) if total_masked > 0 else 0.0

    return WellResult(
        plate_name=plate_name,
        well_id=well_id,
        row=row,
        column=column,
        n_fields=n_fields,
        control_type=control_type,
        total_n_cells=total_n_cells,
        total_n_nuclei=total_n_nuclei,
        total_cell_area_px=total_cell_area,
        total_n_aggregates=total_n_agg,
        total_n_aggregate_positive_cells=total_n_agg_pos,
        pct_aggregate_positive_cells=pct_aggregate_positive,
        total_aggregate_area_px=total_agg_area,
        pct_aggregate_area_over_cell=pct_aggregate_area,
        avg_aggregates_per_positive_cell=avg_agg_per_positive,
        avg_focus_variance_laplacian=avg_focus,
        n_blurry_fields=n_blurry,
        pct_blurry_fields=pct_blurry,
        total_n_cells_masked=total_n_cells_masked,
        pct_aggregate_positive_cells_masked=pct_pos_masked,
        field_results=field_results,
    )


def compute_well_statistics_from_qoi_arrays(
    pct_agg_pos_cells: np.ndarray,
    n_cells: np.ndarray,
) -> float:
    """
    Compute percentage of aggregate-positive cells in a well.

    This function handles the per-field QoI arrays and computes
    the correct well-level statistic.

    Arguments:
        pct_agg_pos_cells: Array of % positive cells per field
        n_cells: Array of cell counts per field

    Returns:
        Percentage of aggregate-positive cells in the well
    """
    total_n_cells = np.sum(n_cells)
    if total_n_cells == 0:
        return 0.0

    # Convert percentage to count, sum, then back to percentage
    n_agg_pos_cells = np.multiply(pct_agg_pos_cells / 100.0, n_cells)
    total_agg_pos = np.sum(n_agg_pos_cells)

    return total_agg_pos / total_n_cells * 100.0


def compute_area_percentage_from_arrays(
    pct_area_agg: np.ndarray,
    cell_area: np.ndarray,
) -> float:
    """
    Compute percentage of cell area occupied by aggregates in a well.

    Arguments:
        pct_area_agg: Array of % aggregate area per field
        cell_area: Array of cell area per field

    Returns:
        Percentage of cell area occupied by aggregates
    """
    total_cell_area = np.sum(cell_area)
    if total_cell_area == 0:
        return 0.0

    # Convert percentage to absolute area, sum, then back to percentage
    agg_area = np.multiply(pct_area_agg / 100.0, cell_area)
    total_agg_area = np.sum(agg_area)

    return total_agg_area / total_cell_area * 100.0
