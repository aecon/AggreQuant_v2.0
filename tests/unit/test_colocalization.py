"""Tests for colocalization module and label counting utilities."""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from aggrequant.segmentation.postprocessing import count_labels
from aggrequant.quantification.colocalization import (
    build_overlap_table,
    count_positive_cells,
    quantify_field,
)

# Path to pre-saved label masks produced by the pipeline
LABELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "test" / "output" / "labels"

REQUIRES_TEST_DATA = pytest.mark.skipif(
    not (LABELS_DIR / "B02_f1_cells.tif").exists(),
    reason="Test label masks not found",
)


# ---------------------------------------------------------------------------
# Reference implementation — kept for regression testing
# ---------------------------------------------------------------------------

def _reference_count_positive_cells(cell_labels, aggregate_labels, min_aggregate_area=9):
    """Loop-based reference: count cells with at least one aggregate >= min_area."""
    unique_cells = np.unique(cell_labels[cell_labels > 0])
    unique_aggs = np.unique(aggregate_labels[aggregate_labels > 0])

    positive = set()
    for agg_id in unique_aggs:
        idx_agg = aggregate_labels == agg_id
        cell_ids_under = cell_labels[idx_agg]
        for cell_id in np.unique(cell_ids_under[cell_ids_under > 0]):
            if np.sum(cell_ids_under == cell_id) >= min_aggregate_area:
                positive.add(cell_id)

    return len(positive)


# ---------------------------------------------------------------------------
# count_labels
# ---------------------------------------------------------------------------

class TestCountLabels:

    def test_empty_image(self):
        labels = np.zeros((50, 50), dtype=np.uint16)
        n, area = count_labels(labels)
        assert n == 0
        assert area == 0

    def test_single_object(self):
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[10:20, 10:20] = 1  # 100 pixels
        n, area = count_labels(labels)
        assert n == 1
        assert area == 100

    def test_multiple_objects(self):
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[0:10, 0:10] = 1   # 100 px
        labels[20:30, 20:30] = 3  # 100 px (non-consecutive ID)
        n, area = count_labels(labels)
        assert n == 2
        assert area == 200

    def test_ignores_background(self):
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[0:10, 0:10] = 1
        n, area = count_labels(labels)
        # Background (label 0) covers 2500-100=2400 pixels, should not count
        assert n == 1
        assert area == 100


# ---------------------------------------------------------------------------
# build_overlap_table
# ---------------------------------------------------------------------------

class TestBuildOverlapTable:

    def test_no_overlap(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[0:10, :] = 1
        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[40:50, :] = 1  # aggregate in a different region, no cell underneath

        table = build_overlap_table(cells, aggs)
        assert table.nnz == 0

    def test_full_overlap(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:40, 10:40] = 1  # 900 px

        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[15:25, 15:25] = 1  # 100 px, fully inside cell 1

        table = build_overlap_table(cells, aggs)
        assert table[1, 1] == 100

    def test_aggregate_spanning_two_cells(self):
        cells = np.zeros((50, 100), dtype=np.uint16)
        cells[10:40, 10:50] = 1
        cells[10:40, 50:90] = 2

        aggs = np.zeros((50, 100), dtype=np.uint32)
        aggs[20:30, 40:60] = 1  # spans boundary: 100 px in each cell

        table = build_overlap_table(cells, aggs)
        assert table[1, 1] == 100
        assert table[1, 2] == 100


# ---------------------------------------------------------------------------
# count_positive_cells
# ---------------------------------------------------------------------------

class TestCountPositiveCells:

    def test_one_cell_one_aggregate_above_threshold(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:40, 10:40] = 1
        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[15:25, 15:25] = 1  # 100 px

        table = build_overlap_table(cells, aggs)
        assert count_positive_cells(table, min_area=9) == 1

    def test_aggregate_below_threshold(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:40, 10:40] = 1
        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[15:17, 15:17] = 1  # 4 px, below min_area=9

        table = build_overlap_table(cells, aggs)
        assert count_positive_cells(table, min_area=9) == 0

    def test_spanning_aggregate_counts_both_cells(self):
        cells = np.zeros((50, 100), dtype=np.uint16)
        cells[10:40, 10:50] = 1
        cells[10:40, 50:90] = 2
        aggs = np.zeros((50, 100), dtype=np.uint32)
        aggs[20:30, 40:60] = 1  # 100 px in each cell

        table = build_overlap_table(cells, aggs)
        assert count_positive_cells(table, min_area=9) == 2

    def test_spanning_aggregate_one_side_below_threshold(self):
        cells = np.zeros((50, 100), dtype=np.uint16)
        cells[10:40, 10:50] = 1
        cells[10:40, 50:90] = 2
        aggs = np.zeros((50, 100), dtype=np.uint32)
        aggs[20:30, 47:53] = 1  # 30 px in cell 1, 30 px in cell 2

        table = build_overlap_table(cells, aggs)
        # Both sides are above 9
        assert count_positive_cells(table, min_area=9) == 2

        # Raise threshold so neither side qualifies
        assert count_positive_cells(table, min_area=50) == 0


# ---------------------------------------------------------------------------
# quantify_field
# ---------------------------------------------------------------------------

class TestQuantifyField:

    def test_empty_labels(self):
        empty = np.zeros((100, 100), dtype=np.uint16)
        result = quantify_field(empty, empty.astype(np.uint32))
        assert result["n_cells"] == 0
        assert result["n_aggregates"] == 0
        assert result["pct_aggregate_positive_cells"] == 0.0

    def test_single_cell_single_aggregate(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:40, 10:40] = 1

        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[15:25, 15:25] = 1  # 100 px

        result = quantify_field(cells, aggs, min_aggregate_area=9)
        assert result["n_cells"] == 1
        assert result["n_aggregates"] == 1
        assert result["n_aggregate_positive_cells"] == 1
        assert result["pct_aggregate_positive_cells"] == 100.0
        assert result["total_cell_area_px"] == 900.0
        assert result["total_aggregate_area_px"] == 100.0

    def test_aggregate_below_min_area(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:40, 10:40] = 1
        aggs = np.zeros((50, 50), dtype=np.uint32)
        aggs[15:17, 15:17] = 1  # 4 px

        result = quantify_field(cells, aggs, min_aggregate_area=9)
        assert result["n_aggregate_positive_cells"] == 0
        assert result["pct_aggregate_positive_cells"] == 0.0
        # Aggregate still counted (it exists), just doesn't make the cell positive
        assert result["n_aggregates"] == 1

    def test_nuclei_count(self):
        cells = np.zeros((50, 50), dtype=np.uint16)
        cells[10:20, 10:20] = 1
        cells[30:40, 30:40] = 2

        nuclei = np.zeros((50, 50), dtype=np.uint16)
        nuclei[12:18, 12:18] = 1
        nuclei[32:38, 32:38] = 2

        aggs = np.zeros((50, 50), dtype=np.uint32)

        result = quantify_field(cells, aggs, nuclei_labels=nuclei)
        assert result["n_nuclei"] == 2
        assert result["n_cells"] == 2

    def test_returns_expected_keys(self):
        empty = np.zeros((10, 10), dtype=np.uint16)
        result = quantify_field(empty, empty.astype(np.uint32))
        expected_keys = {
            "n_cells", "n_nuclei", "n_aggregates",
            "n_aggregate_positive_cells", "pct_aggregate_positive_cells",
            "total_cell_area_px", "total_aggregate_area_px",
        }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Regression: compare against reference on real data
# ---------------------------------------------------------------------------

@REQUIRES_TEST_DATA
class TestQuantifyFieldRegression:

    @pytest.fixture()
    def label_triplets(self):
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

    def test_positive_cells_match_reference(self, label_triplets):
        """Vectorized implementation must match loop-based reference."""
        for stem, cell_labels, agg_labels, nuc_labels in label_triplets:
            ref_n_positive = _reference_count_positive_cells(
                cell_labels, agg_labels, min_aggregate_area=9,
            )
            result = quantify_field(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            assert result["n_aggregate_positive_cells"] == ref_n_positive, (
                f"{stem}: expected {ref_n_positive}, got {result['n_aggregate_positive_cells']}"
            )

    def test_counts_and_areas(self, label_triplets):
        """Basic counts and areas should be consistent."""
        for stem, cell_labels, agg_labels, nuc_labels in label_triplets:
            result = quantify_field(
                cell_labels, agg_labels, nuc_labels, min_aggregate_area=9,
            )
            # Sanity: positive cells <= total cells
            assert result["n_aggregate_positive_cells"] <= result["n_cells"]
            # Areas must be positive if objects exist
            if result["n_cells"] > 0:
                assert result["total_cell_area_px"] > 0
            if result["n_aggregates"] > 0:
                assert result["total_aggregate_area_px"] > 0
