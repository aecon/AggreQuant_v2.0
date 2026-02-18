"""Tests for aggrequant.visualization.heatmaps."""

import numpy as np
import pandas as pd
import pytest

from aggrequant.visualization.heatmaps import (
    aggregate_per_well,
    compute_ratio_per_well,
    well_values_to_plate_grid,
    make_plate_heatmap,
    load_field_measurements,
    generate_all_heatmaps,
    detect_focus_columns,
    plot_metric,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    """DataFrame mimicking field_measurements.csv with 2 wells, 2 fields each."""
    return pd.DataFrame({
        "plate_name": ["plate1"] * 4,
        "well_id": ["A01", "A01", "B02", "B02"],
        "field": [1, 2, 1, 2],
        "n_cells": [100, 120, 80, 90],
        "n_nuclei": [110, 130, 85, 95],
        "n_aggregates": [10, 15, 5, 8],
        "n_aggregate_positive_cells": [30, 40, 10, 20],
        "pct_aggregate_positive_cells": [30.0, 33.3, 12.5, 22.2],
        "total_cell_area_px": [50000.0, 60000.0, 40000.0, 45000.0],
        "total_aggregate_area_px": [5000.0, 7000.0, 2000.0, 3000.0],
    })


@pytest.fixture
def sample_csv(tmp_path, sample_df):
    """Write sample_df to a CSV and return its path."""
    path = tmp_path / "field_measurements.csv"
    sample_df.to_csv(path, index=False)
    return path


# ── aggregate_per_well ────────────────────────────────────────────────

def test_aggregate_sum(sample_df):
    result = aggregate_per_well(sample_df, "n_nuclei", "sum")
    assert result == {"A01": 240, "B02": 180}


def test_aggregate_mean(sample_df):
    result = aggregate_per_well(sample_df, "n_nuclei", "mean")
    assert result["A01"] == pytest.approx(120.0)
    assert result["B02"] == pytest.approx(90.0)


def test_aggregate_max(sample_df):
    result = aggregate_per_well(sample_df, "n_cells", "max")
    assert result == {"A01": 120, "B02": 90}


# ── compute_ratio_per_well ────────────────────────────────────────────

def test_ratio_per_well(sample_df):
    # A01: (30+40)/(100+120)*100 = 70/220*100 = 31.818...
    # B02: (10+20)/(80+90)*100  = 30/170*100 = 17.647...
    result = compute_ratio_per_well(
        sample_df, "n_aggregate_positive_cells", "n_cells", scale=100.0,
    )
    assert result["A01"] == pytest.approx(70 / 220 * 100)
    assert result["B02"] == pytest.approx(30 / 170 * 100)


def test_ratio_per_well_zero_denominator():
    df = pd.DataFrame({
        "well_id": ["X01"],
        "n_aggregate_positive_cells": [5],
        "n_cells": [0],
    })
    result = compute_ratio_per_well(df, "n_aggregate_positive_cells", "n_cells")
    assert result["X01"] == 0.0


# ── well_values_to_plate_grid ────────────────────────────────────────

def test_grid_96well():
    values = {"A01": 100, "H12": 200}
    grid = well_values_to_plate_grid(values, "96")
    assert grid.shape == (8, 12)
    assert grid[0, 0] == 100.0  # A01
    assert grid[7, 11] == 200.0  # H12
    assert np.isnan(grid[0, 1])  # A02 — empty


def test_grid_384well():
    values = {"A01": 10, "P24": 20}
    grid = well_values_to_plate_grid(values, "384")
    assert grid.shape == (16, 24)
    assert grid[0, 0] == 10.0   # A01
    assert grid[15, 23] == 20.0  # P24


def test_grid_empty():
    grid = well_values_to_plate_grid({}, "96")
    assert grid.shape == (8, 12)
    assert np.all(np.isnan(grid))


# ── make_plate_heatmap ───────────────────────────────────────────────

def test_make_heatmap_returns_figure():
    grid = np.random.rand(8, 12)
    fig = make_plate_heatmap(grid, title="Test", plate_format="96")
    assert fig is not None
    assert fig.layout.title.text == "Test"


def test_make_heatmap_384():
    grid = np.random.rand(16, 24)
    fig = make_plate_heatmap(grid, plate_format="384")
    # x-axis should have 24 column labels
    heatmap = fig.data[0]
    assert len(heatmap.x) == 24
    assert len(heatmap.y) == 16


# ── load_field_measurements ──────────────────────────────────────────

def test_load_csv(sample_csv):
    df = load_field_measurements(sample_csv)
    assert len(df) == 4
    assert "n_nuclei" in df.columns


# ── plot_metric (end-to-end) ─────────────────────────────────────────

def test_generate_all_heatmaps(sample_csv, tmp_path):
    plots_dir = generate_all_heatmaps(sample_csv, plate_format="96")
    assert plots_dir.exists()
    expected = [
        "n_nuclei.html",
        "n_aggregates.html",
        "n_aggregate_positive_cells.html",
        "total_cell_area_px.html",
        "total_aggregate_area_px.html",
        "pct_aggregate_positive_cells.html",
        "pct_aggregate_area_over_cell.html",
    ]
    for name in expected:
        assert (plots_dir / name).exists(), f"Missing {name}"


def test_generate_all_heatmaps_custom_output_dir(sample_csv, tmp_path):
    out = tmp_path / "custom_output"
    plots_dir = generate_all_heatmaps(sample_csv, plate_format="96", output_dir=out)
    assert plots_dir == out / "plots"
    assert (plots_dir / "n_nuclei.html").exists()


def test_plot_metric_no_show(sample_csv):
    fig = plot_metric(sample_csv, "n_nuclei", plate_format="96", show=False)
    assert fig is not None
    heatmap_data = fig.data[0]
    # A01 is row 0, col 0 → should be 240
    assert heatmap_data.z[0][0] == pytest.approx(240.0)


# ── detect_focus_columns ─────────────────────────────────────────────

def test_detect_focus_columns_patch():
    cols = ["well_id", "n_cells", "nuclei_patch_VarianceLaplacian_mean",
            "nuclei_patch_VarianceLaplacian_min", "cells_patch_Sobel_max"]
    result = detect_focus_columns(cols)
    assert result == [
        "cells_patch_Sobel_max",
        "nuclei_patch_VarianceLaplacian_mean",
        "nuclei_patch_VarianceLaplacian_min",
    ]


def test_detect_focus_columns_global():
    cols = ["well_id", "nuclei_power_log_log_slope", "cells_high_freq_ratio",
            "n_nuclei"]
    result = detect_focus_columns(cols)
    assert result == ["cells_high_freq_ratio", "nuclei_power_log_log_slope"]


def test_detect_focus_columns_mixed():
    cols = ["nuclei_patch_Brenner_mean", "cells_global_variance_laplacian",
            "plate_name", "field"]
    result = detect_focus_columns(cols)
    assert result == ["cells_global_variance_laplacian",
                      "nuclei_patch_Brenner_mean"]


def test_detect_focus_columns_none():
    cols = ["well_id", "n_cells", "n_nuclei", "pct_aggregate_positive_cells"]
    assert detect_focus_columns(cols) == []


# ── focus heatmaps in generate_all_heatmaps ──────────────────────────

@pytest.fixture
def focus_csv(tmp_path):
    """CSV with standard columns + focus quality columns."""
    df = pd.DataFrame({
        "plate_name": ["plate1"] * 4,
        "well_id": ["A01", "A01", "B02", "B02"],
        "field": [1, 2, 1, 2],
        "n_cells": [100, 120, 80, 90],
        "n_nuclei": [110, 130, 85, 95],
        "n_aggregates": [10, 15, 5, 8],
        "n_aggregate_positive_cells": [30, 40, 10, 20],
        "pct_aggregate_positive_cells": [30.0, 33.3, 12.5, 22.2],
        "total_cell_area_px": [50000.0, 60000.0, 40000.0, 45000.0],
        "total_aggregate_area_px": [5000.0, 7000.0, 2000.0, 3000.0],
        "nuclei_patch_VarianceLaplacian_mean": [42.1, 38.5, 40.0, 41.2],
        "nuclei_patch_VarianceLaplacian_min": [10.2, 8.7, 9.5, 11.0],
        "nuclei_power_log_log_slope": [-2.3, -2.1, -2.5, -2.4],
    })
    path = tmp_path / "field_measurements.csv"
    df.to_csv(path, index=False)
    return path


def test_generate_heatmaps_includes_focus(focus_csv):
    plots_dir = generate_all_heatmaps(focus_csv, plate_format="96")
    expected_focus = [
        "focus_nuclei_patch_VarianceLaplacian_mean.html",
        "focus_nuclei_patch_VarianceLaplacian_min.html",
        "focus_nuclei_power_log_log_slope.html",
    ]
    for name in expected_focus:
        assert (plots_dir / name).exists(), f"Missing {name}"


def test_generate_heatmaps_no_focus_without_columns(sample_csv):
    """No focus_*.html files when CSV has no focus columns."""
    plots_dir = generate_all_heatmaps(sample_csv, plate_format="96")
    focus_files = list(plots_dir.glob("focus_*.html"))
    assert focus_files == []
