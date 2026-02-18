"""QC strip plot for control well validation."""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from aggrequant.visualization.heatmaps import compute_ratio_per_well


def plot_control_strip(csv_path, control_wells, output_path=None):
    """Strip plot of % aggregate-positive cells by control condition.

    Each dot represents one well. A mean +/- SEM error bar is overlaid
    per condition so the user can quickly check whether controls behaved
    as expected.

    Arguments:
        csv_path: Path to field_measurements.csv.
        control_wells: Dict mapping condition name to list of well IDs,
            e.g. {"negative": ["A01", "A02"], "NT": ["A23", "A24"]}.
        output_path: If given, save figure as PNG to this path.

    Returns:
        matplotlib Figure.
    """
    df = pd.read_csv(csv_path)
    well_pct = compute_ratio_per_well(df, "n_aggregate_positive_cells", "n_cells")

    conditions = list(control_wells.keys())
    fig, ax = plt.subplots(figsize=(max(3, len(conditions) * 1.5), 4))

    for i, condition in enumerate(conditions):
        wells = control_wells[condition]
        values = [well_pct[w] for w in wells if w in well_pct]
        if not values:
            continue

        # Jitter x positions for visibility
        rng = np.random.default_rng(seed=42)
        jitter = rng.uniform(-0.15, 0.15, size=len(values))
        xs = np.full(len(values), i) + jitter

        ax.scatter(xs, values, s=40, zorder=3, alpha=0.8)

        # Mean + SEM error bar
        mean = np.mean(values)
        sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
        ax.errorbar(i, mean, yerr=sem, fmt="_", color="black",
                    markersize=12, capsize=4, linewidth=1.5, zorder=4)

    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_ylabel("% Aggregate-Positive Cells")
    ax.set_title("QC: Control Wells")
    ax.set_xlim(-0.5, max(0.5, len(conditions) - 0.5))
    fig.tight_layout()

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)

    return fig
