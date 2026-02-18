"""Tests for aggrequant.visualization.qc_plots."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.figure
import pandas as pd
import pytest

from aggrequant.visualization.qc_plots import plot_control_strip


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """CSV with 2 conditions x 3 wells x 2 fields each."""
    rows = []
    # Condition "negative": wells A01, A02, A03
    for well, pos_cells, total in [("A01", 5, 100), ("A02", 8, 100), ("A03", 3, 100)]:
        for f in [1, 2]:
            rows.append({
                "well_id": well,
                "field": f,
                "n_aggregate_positive_cells": pos_cells,
                "n_cells": total,
            })
    # Condition "NT": wells B01, B02, B03
    for well, pos_cells, total in [("B01", 40, 100), ("B02", 35, 100), ("B03", 45, 100)]:
        for f in [1, 2]:
            rows.append({
                "well_id": well,
                "field": f,
                "n_aggregate_positive_cells": pos_cells,
                "n_cells": total,
            })
    path = tmp_path / "field_measurements.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture
def control_wells():
    return {
        "negative": ["A01", "A02", "A03"],
        "NT": ["B01", "B02", "B03"],
    }


# ── Tests ─────────────────────────────────────────────────────────────

def test_returns_figure(sample_csv, control_wells):
    fig = plot_control_strip(sample_csv, control_wells)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_axes_labels(sample_csv, control_wells):
    fig = plot_control_strip(sample_csv, control_wells)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "% Aggregate-Positive Cells"
    tick_labels = [t.get_text() for t in ax.get_xticklabels()]
    assert tick_labels == ["negative", "NT"]


def test_correct_number_of_points(sample_csv, control_wells):
    from matplotlib.collections import PathCollection
    fig = plot_control_strip(sample_csv, control_wells)
    ax = fig.axes[0]
    # scatter() returns PathCollection; errorbar creates LineCollection
    scatter_collections = [c for c in ax.collections
                           if isinstance(c, PathCollection)]
    total_points = sum(len(c.get_offsets()) for c in scatter_collections)
    assert total_points == 6  # 3 negative + 3 NT


def test_saves_to_file(sample_csv, control_wells, tmp_path):
    out = tmp_path / "plots" / "qc_strip.png"
    fig = plot_control_strip(sample_csv, control_wells, output_path=out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_empty_control_wells(sample_csv):
    fig = plot_control_strip(sample_csv, {})
    assert isinstance(fig, matplotlib.figure.Figure)
    # No scatter data
    ax = fig.axes[0]
    assert len(ax.collections) == 0
