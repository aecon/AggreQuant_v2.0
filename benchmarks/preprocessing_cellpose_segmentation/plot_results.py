#!/usr/bin/env python
"""
Plot preprocessing Cellpose segmentation benchmark results.

Generates a publication-ready supplementary figure with five panels:
  A — Line plot of mean cell count per variant across categories
  B — Pairwise count difference box plots
  C — Cell area distribution per variant per category
  D — Solidity comparison per variant per category
  E — Per-image scatter: seeds vs raw DAPI cell counts

Reads counts.csv and timing.csv produced by run_benchmark.py.

Usage:
    python plot_results.py [--results-dir results] [--output-dir results/figures]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

VARIANT_ORDER = [
    "cellpose_cell_only",
    "cellpose_raw_nuclei",
    "cellpose_nuclei_seeds",
]

VARIANT_LABELS = {
    "cellpose_cell_only":    "Cell only",
    "cellpose_raw_nuclei":   "Raw DAPI",
    "cellpose_nuclei_seeds": "StarDist seeds",
}

VARIANT_COLORS = {
    "cellpose_cell_only":    "#E57373",   # red
    "cellpose_raw_nuclei":   "#64B5F6",   # blue
    "cellpose_nuclei_seeds": "#81C784",   # green
}

VARIANT_MARKERS = {
    "cellpose_cell_only":    "o",
    "cellpose_raw_nuclei":   "s",
    "cellpose_nuclei_seeds": "D",
}

CATEGORY_ORDER = [
    "01_low_confluency",
    "02_high_confluency",
    "03_clustered_touching",
    "04_mitotic",
    "05_defocused",
    "06_flatfield_inhomogeneity",
    "07_low_intensity",
    "08_high_intensity",
    "09_debris_artifacts",
]

CATEGORY_LABELS = {
    "01_low_confluency":          "Low confluency",
    "02_high_confluency":         "High confluency",
    "03_clustered_touching":      "Clustered",
    "04_mitotic":                 "Mitotic",
    "05_defocused":               "Defocused",
    "06_flatfield_inhomogeneity": "Flat-field",
    "07_low_intensity":           "Low intensity",
    "08_high_intensity":          "High intensity",
    "09_debris_artifacts":        "Debris",
}

# Pair definitions for Panel B
PAIRS = [
    ("cellpose_nuclei_seeds", "cellpose_cell_only",  "Seeds − Cell only"),
    ("cellpose_nuclei_seeds", "cellpose_raw_nuclei",  "Seeds − Raw DAPI"),
    ("cellpose_raw_nuclei",   "cellpose_cell_only",  "Raw DAPI − Cell only"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_categories(counts_df):
    """Return categories sorted by median cell count across variants."""
    pivot = counts_df.pivot_table(
        values="cell_count", index="variant_id", columns="category", aggfunc="mean",
    ).reindex(columns=CATEGORY_ORDER)
    cat_median = pivot.median(axis=0)
    return cat_median.sort_values().index.tolist()


def _apply_style(ax):
    """Common axis styling."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="#888")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel A — Cell count line plot
# ---------------------------------------------------------------------------

def plot_count_lines(ax, counts_df):
    """Line plot of mean cell count per variant across categories."""
    sorted_cats = _sorted_categories(counts_df)

    pivot_mean = counts_df.pivot_table(
        values="cell_count", index="variant_id", columns="category", aggfunc="mean",
    ).reindex(columns=sorted_cats)
    pivot_std = counts_df.pivot_table(
        values="cell_count", index="variant_id", columns="category", aggfunc="std",
    ).reindex(columns=sorted_cats)

    x = np.arange(len(sorted_cats))
    offsets = np.linspace(-0.08, 0.08, len(VARIANT_ORDER))

    for idx, vid in enumerate(VARIANT_ORDER):
        if vid not in pivot_mean.index:
            continue
        container = ax.errorbar(
            x + offsets[idx], pivot_mean.loc[vid].values,
            yerr=pivot_std.loc[vid].values,
            color=VARIANT_COLORS[vid], linestyle="-",
            marker=VARIANT_MARKERS[vid],
            linewidth=2.0, markersize=7,
            markeredgewidth=0.6, markeredgecolor="white",
            alpha=0.9, capsize=2, elinewidth=0.8, zorder=2,
            label=VARIANT_LABELS[vid],
        )
        for bar_line in container.lines[2]:
            bar_line.set_alpha(0.3)
        for cap in container.lines[1]:
            cap.set_alpha(0.3)

    _apply_style(ax)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5, color="#888")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in sorted_cats],
        rotation=40, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean cell count", fontsize=10)
    ax.set_xlim(-0.3, len(sorted_cats) - 0.7)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, edgecolor="#ccc")
    ax.set_title("A   Mean cell count per variant across difficulty categories",
                 fontsize=11, fontweight="bold", loc="left", pad=10)


# ---------------------------------------------------------------------------
# Panel B — Pairwise count difference
# ---------------------------------------------------------------------------

def plot_pairwise_diff(ax, counts_df):
    """Box plots of per-image count difference between variant pairs."""
    # Pivot to wide: one row per image, columns = variant cell counts
    wide = counts_df.pivot_table(
        values="cell_count", index=["image_name", "category"],
        columns="variant_id", aggfunc="first",
    ).reset_index()

    pair_colors = ["#81C784", "#FFF176", "#64B5F6"]
    box_data = []
    pair_labels = []
    for (v1, v2, label) in PAIRS:
        if v1 in wide.columns and v2 in wide.columns:
            diff = (wide[v1] - wide[v2]).dropna().values
            box_data.append(diff)
            pair_labels.append(label)

    if not box_data:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center")
        return

    positions = np.arange(len(box_data))
    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=4, alpha=0.5, color="#555"),
        medianprops=dict(color="#D32F2F", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch, color in zip(bp["boxes"], pair_colors[:len(box_data)]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#546E7A")
        patch.set_alpha(0.8)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(pair_labels, fontsize=9)
    ax.set_ylabel("Cell count difference", fontsize=10)
    _apply_style(ax)
    ax.set_title("B   Pairwise cell count difference (per image)",
                 fontsize=11, fontweight="bold", loc="left", pad=10)


# ---------------------------------------------------------------------------
# Panel C — Cell area distribution
# ---------------------------------------------------------------------------

def plot_area_distribution(axes_row, counts_df):
    """Box plots of mean cell area per variant, one subplot per category."""
    cats = [c for c in CATEGORY_ORDER if c in counts_df["category"].values]

    for cat_idx, cat in enumerate(cats):
        ax = axes_row[cat_idx]
        cat_data = counts_df[counts_df["category"] == cat]

        box_data = []
        colors = []
        labels = []
        for vid in VARIANT_ORDER:
            vals = cat_data.loc[cat_data["variant_id"] == vid, "mean_area_px"].dropna().values
            box_data.append(vals)
            colors.append(VARIANT_COLORS[vid])
            labels.append(VARIANT_LABELS[vid])

        if not any(len(d) > 0 for d in box_data):
            ax.set_visible(False)
            continue

        positions = np.arange(len(box_data))
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.55, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
            medianprops=dict(color="#D32F2F", linewidth=1.2),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#546E7A")
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(CATEGORY_LABELS[cat], fontsize=9, fontweight="bold")
        _apply_style(ax)
        if cat_idx == 0:
            ax.set_ylabel("Mean cell area (px)", fontsize=9)

    # Hide unused axes
    for i in range(len(cats), len(axes_row)):
        axes_row[i].set_visible(False)


# ---------------------------------------------------------------------------
# Panel D — Solidity comparison
# ---------------------------------------------------------------------------

def plot_solidity(axes_row, counts_df):
    """Box plots of mean solidity per variant, one subplot per category."""
    cats = [c for c in CATEGORY_ORDER if c in counts_df["category"].values]

    for cat_idx, cat in enumerate(cats):
        ax = axes_row[cat_idx]
        cat_data = counts_df[counts_df["category"] == cat]

        box_data = []
        colors = []
        labels = []
        for vid in VARIANT_ORDER:
            vals = cat_data.loc[cat_data["variant_id"] == vid, "mean_solidity"].dropna().values
            box_data.append(vals)
            colors.append(VARIANT_COLORS[vid])
            labels.append(VARIANT_LABELS[vid])

        if not any(len(d) > 0 for d in box_data):
            ax.set_visible(False)
            continue

        positions = np.arange(len(box_data))
        bp = ax.boxplot(
            box_data, positions=positions, widths=0.55, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker=".", markersize=3, alpha=0.5),
            medianprops=dict(color="#D32F2F", linewidth=1.2),
            whiskerprops=dict(linewidth=0.7),
            capprops=dict(linewidth=0.7),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_edgecolor("#546E7A")
            patch.set_alpha(0.8)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(CATEGORY_LABELS[cat], fontsize=9, fontweight="bold")
        _apply_style(ax)
        if cat_idx == 0:
            ax.set_ylabel("Mean solidity", fontsize=9)

    for i in range(len(cats), len(axes_row)):
        axes_row[i].set_visible(False)


# ---------------------------------------------------------------------------
# Panel E — Scatter: seeds vs raw DAPI
# ---------------------------------------------------------------------------

def plot_scatter_seeds_vs_raw(ax, counts_df):
    """Per-image scatter of nuclei_seeds vs raw_nuclei cell counts."""
    wide = counts_df.pivot_table(
        values="cell_count", index=["image_name", "category"],
        columns="variant_id", aggfunc="first",
    ).reset_index()

    v_raw = "cellpose_raw_nuclei"
    v_seeds = "cellpose_nuclei_seeds"
    if v_raw not in wide.columns or v_seeds not in wide.columns:
        ax.text(0.5, 0.5, "Insufficient data", transform=ax.transAxes,
                ha="center", va="center")
        return

    cats = [c for c in CATEGORY_ORDER if c in wide["category"].values]
    cmap = mpl.colormaps["tab10"]
    cat_colors = {cat: cmap(i / max(len(cats) - 1, 1)) for i, cat in enumerate(cats)}

    for cat in cats:
        mask = wide["category"] == cat
        ax.scatter(
            wide.loc[mask, v_raw], wide.loc[mask, v_seeds],
            c=[cat_colors[cat]], s=30, alpha=0.75, edgecolors="white",
            linewidth=0.4, label=CATEGORY_LABELS[cat], zorder=2,
        )

    # Diagonal
    lims = [0, max(wide[v_raw].max(), wide[v_seeds].max()) * 1.05]
    ax.plot(lims, lims, "--", color="#888", linewidth=0.8, alpha=0.5, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Cell count ({VARIANT_LABELS[v_raw]})", fontsize=10)
    ax.set_ylabel(f"Cell count ({VARIANT_LABELS[v_seeds]})", fontsize=10)
    ax.set_aspect("equal")
    _apply_style(ax)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, edgecolor="#ccc",
              ncol=2, markerscale=0.8)
    ax.set_title("E   Per-image cell count: StarDist seeds vs raw DAPI",
                 fontsize=11, fontweight="bold", loc="left", pad=10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot preprocessing Cellpose benchmark results",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory with counts.csv and timing.csv (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for figures (default: <results-dir>/figures)",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure resolution")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(results_dir / "counts.csv")
    timing = pd.read_csv(results_dir / "timing.csv")

    print(f"Counts: {len(counts)} rows, {counts['variant_id'].nunique()} variants")
    print(f"Timing: {len(timing)} rows")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    def save_fig(fig, name):
        for ext in ("pdf", "png"):
            out = output_dir / f"{name}.{ext}"
            fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)

    # Panel A — count lines
    fig_a, ax_a = plt.subplots(figsize=(12, 5.5))
    plot_count_lines(ax_a, counts)
    save_fig(fig_a, "panel_A_counts")

    # Panel B — pairwise difference
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    plot_pairwise_diff(ax_b, counts)
    save_fig(fig_b, "panel_B_pairwise_diff")

    # Panel C — area distribution (3x3 grid)
    n_cats = counts["category"].nunique()
    ncols_grid = min(n_cats, 5)
    nrows_grid = int(np.ceil(n_cats / ncols_grid))
    fig_c, axes_c = plt.subplots(nrows_grid, ncols_grid,
                                  figsize=(ncols_grid * 3.5, nrows_grid * 3.5))
    axes_c_flat = axes_c.ravel() if hasattr(axes_c, "ravel") else [axes_c]
    plot_area_distribution(axes_c_flat, counts)
    fig_c.suptitle("C   Mean cell area per variant across difficulty categories",
                   fontsize=11, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig_c.tight_layout()
    save_fig(fig_c, "panel_C_area")

    # Panel D — solidity (3x3 grid)
    fig_d, axes_d = plt.subplots(nrows_grid, ncols_grid,
                                  figsize=(ncols_grid * 3.5, nrows_grid * 3.5))
    axes_d_flat = axes_d.ravel() if hasattr(axes_d, "ravel") else [axes_d]
    plot_solidity(axes_d_flat, counts)
    fig_d.suptitle("D   Mean solidity per variant across difficulty categories",
                   fontsize=11, fontweight="bold", x=0.01, ha="left", y=1.02)
    fig_d.tight_layout()
    save_fig(fig_d, "panel_D_solidity")

    # Panel E — scatter seeds vs raw
    fig_e, ax_e = plt.subplots(figsize=(7, 7))
    plot_scatter_seeds_vs_raw(ax_e, counts)
    save_fig(fig_e, "panel_E_scatter")

    # --- Quick stats ---
    print("\n" + "=" * 60)
    print("Quick stats")
    print("=" * 60)

    pivot = counts.pivot_table(
        values="cell_count", index="category", columns="variant_id", aggfunc="mean",
    ).reindex(columns=VARIANT_ORDER, index=CATEGORY_ORDER)
    print("\nMean cell count per category:")
    print(pivot.round(1).to_string())

    pivot_sol = counts.pivot_table(
        values="mean_solidity", index="category", columns="variant_id", aggfunc="mean",
    ).reindex(columns=VARIANT_ORDER, index=CATEGORY_ORDER)
    print("\nMean solidity per category:")
    print(pivot_sol.round(4).to_string())

    # Per-image count differences
    wide = counts.pivot_table(
        values="cell_count", index="image_name", columns="variant_id", aggfunc="first",
    )
    for v1, v2, label in PAIRS:
        if v1 in wide.columns and v2 in wide.columns:
            diff = wide[v1] - wide[v2]
            print(f"\n{label}: mean={diff.mean():.1f}, median={diff.median():.1f}, "
                  f"std={diff.std():.1f}")

    valid_t = timing.dropna(subset=["inference_time_s"])
    if not valid_t.empty:
        print("\nCellpose inference time:")
        t_stats = valid_t.groupby("variant_id")["inference_time_s"].agg(["mean", "std"])
        for vid in VARIANT_ORDER:
            if vid in t_stats.index:
                print(f"  {VARIANT_LABELS[vid]:18s}  "
                      f"{t_stats.loc[vid, 'mean']:.3f} ± {t_stats.loc[vid, 'std']:.3f} s")


if __name__ == "__main__":
    main()
