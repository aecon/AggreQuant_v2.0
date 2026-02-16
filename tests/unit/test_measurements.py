"""
Regression tests for compute_field_measurements.

Compares the optimized implementation against a known-good reference
implementation using real label masks from the test data directory.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from aggrequant.quantification.measurements import compute_field_measurements

# Path to pre-saved label masks produced by the pipeline
LABELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test" / "output" / "labels"

REQUIRES_TEST_DATA = pytest.mark.skipif(
    not (LABELS_DIR / "B02_f1_cells.tif").exists(),
    reason="Test label masks not found",
)


# ---------------------------------------------------------------------------
# Reference (original) implementation — kept verbatim for regression testing
# ---------------------------------------------------------------------------

def _reference_compute_field_measurements(
    cell_labels: np.ndarray,
    aggregate_labels: np.ndarray,
    nuclei_labels=None,
    min_aggregate_area: int = 9,
):
    """Exact copy of the original loop-based implementation."""
    mask_agg = (aggregate_labels > 0).astype(np.uint8)
    mask_cell = (cell_labels > 0).astype(np.uint8)

    unique_cells = np.unique(cell_labels[cell_labels > 0])
    unique_aggregates = np.unique(aggregate_labels[aggregate_labels > 0])

    n_cells = len(unique_cells)
    n_aggregates = len(unique_aggregates)

    aggregates_per_cell = np.zeros(n_cells)
    cells_per_aggregate = np.zeros(n_aggregates)

    for ia, agg_id in enumerate(unique_aggregates):
        idx_agg = aggregate_labels == agg_id
        cell_ids_under_agg = cell_labels[idx_agg]
        unique_cell_ids = np.unique(cell_ids_under_agg[cell_ids_under_agg > 0])

        if len(unique_cell_ids) == 0:
            continue

        for cell_id in unique_cell_ids:
            agg_area_in_cell = np.sum(cell_ids_under_agg == cell_id)
            if agg_area_in_cell >= min_aggregate_area:
                cell_idx = np.where(unique_cells == cell_id)[0]
                if len(cell_idx) > 0:
                    aggregates_per_cell[cell_idx[0]] += 1
                    cells_per_aggregate[ia] += 1

    total_cell_area = int(np.sum(mask_cell))
    total_agg_area = int(np.sum(mask_agg))
    n_agg_positive_cells = int(np.sum(aggregates_per_cell > 0))
    pct_agg_positive = (n_agg_positive_cells / n_cells * 100) if n_cells > 0 else 0.0
    pct_area_agg = (total_agg_area / total_cell_area * 100) if total_cell_area > 0 else 0.0
    n_ambiguous = int(np.sum(cells_per_aggregate > 1))
    pct_ambiguous = (n_ambiguous / n_aggregates * 100) if n_aggregates > 0 else 0.0
    positive_cells_agg_counts = aggregates_per_cell[aggregates_per_cell > 0]
    avg_agg_per_positive = float(np.mean(positive_cells_agg_counts)) if len(positive_cells_agg_counts) > 0 else 0.0

    n_nuclei = 0
    total_nuclei_area = 0
    if nuclei_labels is not None:
        n_nuclei = len(np.unique(nuclei_labels[nuclei_labels > 0]))
        total_nuclei_area = int(np.sum(nuclei_labels > 0))

    return {
        "n_cells": n_cells,
        "n_nuclei": n_nuclei,
        "total_nuclei_area_px": float(total_nuclei_area),
        "total_cell_area_px": float(total_cell_area),
        "n_aggregates": n_aggregates,
        "n_aggregate_positive_cells": n_agg_positive_cells,
        "pct_aggregate_positive_cells": pct_agg_positive,
        "total_aggregate_area_px": float(total_agg_area),
        "pct_aggregate_area_over_cell": pct_area_agg,
        "avg_aggregates_per_positive_cell": avg_agg_per_positive,
        "pct_ambiguous_aggregates": pct_ambiguous,
        "aggregates_per_cell": aggregates_per_cell,
        "cells_per_aggregate": cells_per_aggregate,
    }


# ---------------------------------------------------------------------------
# Helper to compare FieldResult against reference dict
# ---------------------------------------------------------------------------

def _assert_results_match(ref: dict, result, rtol=1e-12):
    """Assert every measurement in FieldResult matches the reference dict."""
    assert result.n_cells == ref["n_cells"]
    assert result.n_nuclei == ref["n_nuclei"]
    assert result.total_nuclei_area_px == ref["total_nuclei_area_px"]
    assert result.total_cell_area_px == ref["total_cell_area_px"]
    assert result.n_aggregates == ref["n_aggregates"]
    assert result.n_aggregate_positive_cells == ref["n_aggregate_positive_cells"]
    assert result.pct_aggregate_positive_cells == pytest.approx(ref["pct_aggregate_positive_cells"], rel=rtol)
    assert result.total_aggregate_area_px == ref["total_aggregate_area_px"]
    assert result.pct_aggregate_area_over_cell == pytest.approx(ref["pct_aggregate_area_over_cell"], rel=rtol)
    assert result.avg_aggregates_per_positive_cell == pytest.approx(ref["avg_aggregates_per_positive_cell"], rel=rtol)
    assert result.pct_ambiguous_aggregates == pytest.approx(ref["pct_ambiguous_aggregates"], rel=rtol)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@REQUIRES_TEST_DATA
class TestComputeFieldMeasurementsRegression:
    """Regression: optimized implementation must match original on real data."""

    @pytest.fixture()
    def label_triplets(self):
        """Load several real label triplets for thorough coverage."""
        triplets = []
        for stem in ["B02_f1", "B02_f5", "B03_f3"]:
            cell_path = LABELS_DIR / f"{stem}_cells.tif"
            agg_path = LABELS_DIR / f"{stem}_aggregates.tif"
            nuc_path = LABELS_DIR / f"{stem}_nuclei.tif"
            if cell_path.exists() and agg_path.exists() and nuc_path.exists():
                triplets.append((
                    stem,
                    tifffile.imread(str(cell_path)),
                    tifffile.imread(str(agg_path)),
                    tifffile.imread(str(nuc_path)),
                ))
        assert len(triplets) > 0, "No label triplets found"
        return triplets

    def test_optimized_matches_reference(self, label_triplets):
        """All scalar measurements must match between implementations."""
        for stem, cell_labels, agg_labels, nuc_labels in label_triplets:
            ref = _reference_compute_field_measurements(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            result, diag = compute_field_measurements(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            _assert_results_match(ref, result)

    def test_diagnostic_arrays_match(self, label_triplets):
        """Per-cell and per-aggregate tracking arrays must match."""
        for stem, cell_labels, agg_labels, nuc_labels in label_triplets:
            ref = _reference_compute_field_measurements(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            _, diag = compute_field_measurements(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            np.testing.assert_array_equal(
                diag["aggregates_per_cell"], ref["aggregates_per_cell"],
                err_msg=f"{stem}: aggregates_per_cell mismatch",
            )
            np.testing.assert_array_equal(
                diag["cells_per_aggregate"], ref["cells_per_aggregate"],
                err_msg=f"{stem}: cells_per_aggregate mismatch",
            )

    def test_no_cells_no_aggregates(self):
        """Empty labels should produce zero results."""
        empty = np.zeros((100, 100), dtype=np.uint16)
        result, _ = compute_field_measurements(empty, empty.astype(np.uint32))
        assert result.n_cells == 0
        assert result.n_aggregates == 0
        assert result.pct_aggregate_positive_cells == 0.0

    def test_single_cell_single_aggregate(self):
        """Minimal case: one cell fully containing one aggregate."""
        cell_labels = np.zeros((50, 50), dtype=np.uint16)
        cell_labels[10:40, 10:40] = 1

        agg_labels = np.zeros((50, 50), dtype=np.uint32)
        agg_labels[15:25, 15:25] = 1  # 100 pixels, well above min_area=9

        ref = _reference_compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        result, diag = compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        _assert_results_match(ref, result)
        assert result.n_cells == 1
        assert result.n_aggregates == 1
        assert result.n_aggregate_positive_cells == 1
        assert result.pct_aggregate_positive_cells == 100.0

    def test_aggregate_below_min_area_ignored(self):
        """An aggregate smaller than min_aggregate_area should not count."""
        cell_labels = np.zeros((50, 50), dtype=np.uint16)
        cell_labels[10:40, 10:40] = 1

        agg_labels = np.zeros((50, 50), dtype=np.uint32)
        agg_labels[15:17, 15:17] = 1  # 4 pixels — below min_area=9

        ref = _reference_compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        result, _ = compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        _assert_results_match(ref, result)
        assert result.n_aggregate_positive_cells == 0

    def test_ambiguous_aggregate_spanning_two_cells(self):
        """Aggregate overlapping two cells should be counted as ambiguous."""
        cell_labels = np.zeros((50, 100), dtype=np.uint16)
        cell_labels[10:40, 10:50] = 1
        cell_labels[10:40, 50:90] = 2

        # Aggregate spans boundary: 20 px in cell 1, 20 px in cell 2
        agg_labels = np.zeros((50, 100), dtype=np.uint32)
        agg_labels[20:30, 40:60] = 1  # 10×20 = 200 px total, 100 in each cell

        ref = _reference_compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        result, _ = compute_field_measurements(
            cell_labels, agg_labels, min_aggregate_area=9,
        )
        _assert_results_match(ref, result)
        assert result.pct_ambiguous_aggregates == 100.0
        assert result.n_aggregate_positive_cells == 2
