"""Plate heatmap visualization from field_measurements CSV."""

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from aggrequant.focus import ALL_GLOBAL_METRICS
from aggrequant.loaders.plate import PLATE_LAYOUTS, well_id_to_indices

# Patterns for auto-detecting focus quality columns in the CSV.
# Patch metrics: {channel}_patch_{MetricName}_{stat}
# Global metrics: {channel}_{global_metric_name}
_FOCUS_PATCH_RE = re.compile(r"^(nuclei|cells)_patch_.+_(mean|min|max)$")
_FOCUS_GLOBAL_RE = re.compile(
    r"^(nuclei|cells)_(" + "|".join(ALL_GLOBAL_METRICS) + r")$"
)


def load_field_measurements(csv_path):
    """Load a field_measurements.csv into a DataFrame.

    Arguments:
        csv_path: Path to the CSV file produced by the pipeline.

    Returns:
        pandas DataFrame with one row per field.
    """
    return pd.read_csv(csv_path)


def aggregate_per_well(df, metric, agg_func="sum"):
    """Aggregate a field-level metric to well level.

    Arguments:
        df: DataFrame from load_field_measurements.
        metric: Column name to aggregate (e.g. "n_nuclei").
        agg_func: Aggregation function — "sum", "mean", "median", "max", "min",
                  or a callable.

    Returns:
        Dictionary mapping well_id (str) to aggregated value (float).
    """
    grouped = df.groupby("well_id")[metric].agg(agg_func)
    return grouped.to_dict()


def compute_ratio_per_well(df, numerator, denominator, scale=100.0):
    """Compute a ratio metric per well from two field-level count columns.

    Sums numerator and denominator across fields first, then divides.
    Use this for percentage metrics where averaging field-level percentages
    would be incorrect (e.g. pct_aggregate_positive_cells).

    Arguments:
        df: DataFrame from load_field_measurements.
        numerator: Column to sum for the numerator (e.g. "n_aggregate_positive_cells").
        denominator: Column to sum for the denominator (e.g. "n_cells").
        scale: Multiply the ratio by this value (100 for percentage).

    Returns:
        Dictionary mapping well_id (str) to the computed value.
    """
    grouped = df.groupby("well_id")[[numerator, denominator]].sum()
    result = {}
    for well_id, row in grouped.iterrows():
        denom = row[denominator]
        result[well_id] = (row[numerator] / denom * scale) if denom > 0 else 0.0
    return result


def well_values_to_plate_grid(well_values, plate_format="96"):
    """Convert a {well_id: value} dict to a plate-shaped 2D array.

    Arguments:
        well_values: Dictionary mapping well_id (e.g. "A01") to a numeric value.
        plate_format: "96" or "384".

    Returns:
        2D numpy array of shape (n_rows, n_cols) with NaN for empty wells.
    """
    layout = PLATE_LAYOUTS[plate_format]
    n_rows, n_cols = layout["rows"], layout["cols"]
    grid = np.full((n_rows, n_cols), np.nan)

    for well_id, value in well_values.items():
        row, col = well_id_to_indices(well_id, plate_format)
        grid[row, col] = value

    return grid


def make_plate_heatmap(grid, title="", plate_format="96",
                       colorscale="Viridis", fig_width=900, fig_height=500):
    """Create a Plotly heatmap figure shaped as a multi-well plate.

    Arguments:
        grid: 2D numpy array (n_rows x n_cols) from well_values_to_plate_grid.
        title: Figure title.
        plate_format: "96" or "384" (used for axis labels).
        colorscale: Plotly colorscale name.
        fig_width: Figure width in pixels.
        fig_height: Figure height in pixels.

    Returns:
        plotly.graph_objects.Figure
    """
    layout = PLATE_LAYOUTS[plate_format]
    n_rows, n_cols = layout["rows"], layout["cols"]

    row_labels = [chr(ord('A') + i) for i in range(n_rows)]
    col_labels = [str(i + 1) for i in range(n_cols)]

    fig = go.Figure(data=go.Heatmap(
        z=grid,
        x=col_labels,
        y=row_labels,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='Well: %{y}%{x}<br>Value: %{z:.1f}<extra></extra>',
        xgap=1,
        ygap=1,
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(autorange='reversed', scaleanchor='x', dtick=1,
                   constrain='domain'),
        xaxis=dict(side='top', dtick=1, constrain='domain'),
        width=fig_width,
        height=fig_height,
        margin=dict(l=40, r=40, t=60, b=20),
        plot_bgcolor='white',
    )
    return fig


def detect_focus_columns(columns):
    """Return the subset of column names that are focus quality metrics.

    Matches patch metrics ({channel}_patch_{Name}_{stat}) and global
    metrics ({channel}_{global_metric_name}).

    Arguments:
        columns: Iterable of column name strings.

    Returns:
        Sorted list of focus column names.
    """
    focus_cols = []
    for col in columns:
        if _FOCUS_PATCH_RE.match(col) or _FOCUS_GLOBAL_RE.match(col):
            focus_cols.append(col)
    return sorted(focus_cols)


def generate_all_heatmaps(csv_path, plate_format="96", output_dir=None):
    """Generate and save heatmaps for standard metrics and focus quality.

    Saves interactive HTML files to output_dir/plots/. Called automatically
    by the pipeline, but can also be called standalone on any
    field_measurements.csv.

    Arguments:
        csv_path: Path to field_measurements.csv.
        plate_format: "96" or "384".
        output_dir: Directory for the plots/ subfolder. Defaults to the
                    parent directory of csv_path.

    Returns:
        Path to the plots/ directory.
    """
    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent
    else:
        output_dir = Path(output_dir)

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    df = load_field_measurements(csv_path)

    def _save_sum(col, title, filename):
        if col not in df.columns:
            return
        vals = aggregate_per_well(df, col, "sum")
        grid = well_values_to_plate_grid(vals, plate_format)
        fig = make_plate_heatmap(grid, title=title, plate_format=plate_format)
        fig.write_html(plots_dir / filename)

    def _save_ratio(num, denom, scale, title, filename):
        if num not in df.columns or denom not in df.columns:
            return
        vals = compute_ratio_per_well(df, num, denom, scale=scale)
        grid = well_values_to_plate_grid(vals, plate_format)
        fig = make_plate_heatmap(grid, title=title, plate_format=plate_format)
        fig.write_html(plots_dir / filename)

    # Count metrics (summed across fields)
    _save_sum("n_nuclei", "Total nuclei per well", "n_nuclei.html")
    _save_sum("n_aggregates", "Total aggregates per well", "n_aggregates.html")
    _save_sum("n_aggregate_positive_cells",
              "Total aggregate-positive cells per well",
              "n_aggregate_positive_cells.html")
    _save_sum("total_cell_area_px", "Total cell area (px) per well",
              "total_cell_area_px.html")
    _save_sum("total_aggregate_area_px", "Total aggregate area (px) per well",
              "total_aggregate_area_px.html")

    # Ratio metrics (sum numerator / sum denominator per well)
    _save_ratio("n_aggregate_positive_cells", "n_cells", 100.0,
                "% aggregate-positive cells per well",
                "pct_aggregate_positive_cells.html")
    _save_ratio("total_aggregate_area_px", "total_cell_area_px", 100.0,
                "% aggregate area / cell area per well",
                "pct_aggregate_area_over_cell.html")

    # Focus quality metrics (averaged across fields per well)
    for col in detect_focus_columns(df.columns):
        title = col.replace("_", " ").title() + " per well (mean)"
        vals = aggregate_per_well(df, col, "mean")
        grid = well_values_to_plate_grid(vals, plate_format)
        fig = make_plate_heatmap(grid, title=title, plate_format=plate_format)
        fig.write_html(plots_dir / f"focus_{col}.html")

    return plots_dir


def plot_metric(csv_path, metric, agg_func="sum", plate_format="96",
                title=None, colorscale="Viridis", show=True):
    """One-liner: load CSV, aggregate, and plot a plate heatmap.

    Arguments:
        csv_path: Path to field_measurements.csv.
        metric: Column name to plot (e.g. "n_nuclei").
        agg_func: Aggregation function ("sum", "mean", etc.).
        plate_format: "96" or "384".
        title: Figure title. Auto-generated if None.
        colorscale: Plotly colorscale name.
        show: If True, call fig.show() to display interactively.

    Returns:
        plotly.graph_objects.Figure
    """
    df = load_field_measurements(csv_path)
    well_values = aggregate_per_well(df, metric, agg_func)
    grid = well_values_to_plate_grid(well_values, plate_format)

    if title is None:
        title = f"{metric} per well ({agg_func})"

    fig = make_plate_heatmap(grid, title=title, plate_format=plate_format,
                             colorscale=colorscale)
    if show:
        fig.show()
    return fig
