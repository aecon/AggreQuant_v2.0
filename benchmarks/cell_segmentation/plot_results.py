#!/usr/bin/env python
"""
Plot cell segmentation benchmark results.

Generates a publication-ready supplementary figure with five panels:
  A — Line plot of mean cell count per model across categories
  B — Inter-model count agreement (CV) per category
  C — Mean GPU inference time per model
  D — Per-image cell count (one subplot per image rank within category)
  E — Per-category cell count (one subplot per category)

Reads counts.csv and timing.csv produced by run_benchmark.py.

Usage:
    python plot_results.py [--results-dir results] [--output-dir results/figures]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

# Single-channel (FarRed only) — first 4
MODEL_ORDER = [
    "deepcell_mesmer",
    "cellpose_cyto2",
    "cellpose_cyto3",
    "instanseg_fluorescence",
    # Two-channel (FarRed + DAPI nuclear hint)
    "deepcell_mesmer_with_nuc",
    "cellpose_cyto2_with_nuc",
    "cellpose_cyto3_with_nuc",
    "instanseg_fluorescence_with_nuc",
]

N_SINGLE = 4  # first 4 entries are single-channel

MODEL_LABELS = {
    "deepcell_mesmer":                  "Mesmer",
    "cellpose_cyto2":                   "Cellpose cyto2",
    "cellpose_cyto3":                   "Cellpose cyto3",
    "instanseg_fluorescence":           "InstanSeg",
    "deepcell_mesmer_with_nuc":         "Mesmer +nuc",
    "cellpose_cyto2_with_nuc":          "Cellpose cyto2 +nuc",
    "cellpose_cyto3_with_nuc":          "Cellpose cyto3 +nuc",
    "instanseg_fluorescence_with_nuc":  "InstanSeg +nuc",
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

MODEL_FRAMEWORK = {
    "deepcell_mesmer":                  "TensorFlow",
    "deepcell_mesmer_with_nuc":         "TensorFlow",
    "cellpose_cyto2":                   "PyTorch",
    "cellpose_cyto3":                   "PyTorch",
    "cellpose_cyto2_with_nuc":          "PyTorch",
    "cellpose_cyto3_with_nuc":          "PyTorch",
    "instanseg_fluorescence":           "PyTorch",
    "instanseg_fluorescence_with_nuc":  "PyTorch",
}

FW_COLORS = {"TensorFlow": "#CE93D8", "PyTorch": "#64B5F6"}

# Legend order for consistent color assignment
LEGEND_ORDER = list(MODEL_ORDER)

PANEL_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P"]


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _interleaved_indices(n):
    order = []
    lo, hi = 0, n - 1
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1
    return order


def make_model_colors(cmap_name="magma"):
    cmap = mpl.colormaps[cmap_name]
    n = len(LEGEND_ORDER)
    n_slots = n + 2
    seq_colors = [cmap(i / (n_slots - 1)) for i in range(n)]
    interleaved = _interleaved_indices(n)
    return {mid: seq_colors[interleaved[i]] for i, mid in enumerate(LEGEND_ORDER)}


def make_model_colors_sequential(cmap_name="magma"):
    cmap = mpl.colormaps[cmap_name]
    n = len(LEGEND_ORDER)
    n_slots = n + 2
    return {mid: cmap(i / (n_slots - 1)) for i, mid in enumerate(LEGEND_ORDER)}


# ---------------------------------------------------------------------------
# Panel A — Count line plot
# ---------------------------------------------------------------------------

def plot_count_lines(ax, counts_df):
    """Line plot of mean cell count per model across categories."""
    pivot_mean = counts_df.pivot_table(
        values="cell_count", index="model_id", columns="category", aggfunc="mean",
    )
    pivot_std = counts_df.pivot_table(
        values="cell_count", index="model_id", columns="category", aggfunc="std",
    )
    pivot_mean = pivot_mean.reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)
    pivot_std = pivot_std.reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)

    # Sort categories by median count across models (ascending)
    cat_median = pivot_mean.median(axis=0)
    sorted_cats = cat_median.sort_values().index.tolist()
    pivot_mean = pivot_mean[sorted_cats]
    pivot_std = pivot_std[sorted_cats]

    x = np.arange(len(sorted_cats))
    model_colors = make_model_colors("magma")
    offsets = np.linspace(-0.15, 0.15, len(LEGEND_ORDER))

    for idx, mid in enumerate(LEGEND_ORDER):
        if mid not in pivot_mean.index:
            continue
        container = ax.errorbar(
            x + offsets[idx], pivot_mean.loc[mid].values,
            yerr=pivot_std.loc[mid].values,
            color=model_colors[mid], linestyle="-",
            marker=PANEL_MARKERS[idx % len(PANEL_MARKERS)],
            linewidth=1.5, markersize=5,
            markeredgewidth=0.6, markeredgecolor="white",
            alpha=0.9, capsize=1, elinewidth=0.8, zorder=2,
            label=MODEL_LABELS[mid],
        )
        for bar_line in container.lines[2]:
            bar_line.set_alpha(0.2)
        for cap in container.lines[1]:
            cap.set_alpha(0.2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", alpha=0.2, linewidth=0.5, color="#888")
    ax.set_axisbelow(True)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in sorted_cats],
        rotation=40, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean cell count", fontsize=10)
    ax.set_xlim(-0.3, len(sorted_cats) - 0.7)
    ax.set_ylim(bottom=0)

    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    ax.legend(
        ord_h, ord_l,
        fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5,
        borderaxespad=0,
    )
    ax.set_title("A   Mean cell count per model across difficulty categories",
                 fontsize=11, fontweight="bold", loc="left", pad=10)


# ---------------------------------------------------------------------------
# Panel B — Inter-model agreement
# ---------------------------------------------------------------------------

def plot_agreement(ax, counts_df):
    """Box plot of inter-model count CV per category (single-channel models only)."""
    sc_models = MODEL_ORDER[:N_SINGLE]
    sc = counts_df[counts_df["model_id"].isin(sc_models)].copy()

    def cv_func(x):
        m = x.mean()
        return (x.std() / m * 100) if m > 0 else np.nan

    cv = (
        sc.groupby(["image_name", "category"])["cell_count"]
        .agg(cv_func)
        .reset_index(name="cv_pct")
    )

    box_data = [
        cv.loc[cv["category"] == cat, "cv_pct"].dropna().values
        for cat in CATEGORY_ORDER
    ]

    positions = np.arange(len(CATEGORY_ORDER))
    bp = ax.boxplot(
        box_data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=True,
        flierprops=dict(marker=".", markersize=4, alpha=0.5, color="#555"),
        medianprops=dict(color="#D32F2F", linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#B0BEC5")
        patch.set_edgecolor("#546E7A")
        patch.set_alpha(0.8)

    ax.set_xticks(positions)
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in CATEGORY_ORDER],
        rotation=45, ha="right", fontsize=8,
    )
    ax.set_ylabel("Count CV across models (%)", fontsize=9)
    ax.set_title(f"B   Inter-model agreement ({N_SINGLE} single-channel models)",
                 fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(-0.7, len(CATEGORY_ORDER) - 0.3)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Panel C — Inference timing
# ---------------------------------------------------------------------------

def plot_timing(ax, timing_df):
    """Horizontal bar chart of mean inference time per model (GPU)."""
    valid = timing_df.dropna(subset=["inference_time_s"])
    if valid.empty:
        ax.text(0.5, 0.5, "No timing data available",
                transform=ax.transAxes, ha="center", va="center", fontsize=10)
        return

    stats = (
        valid.groupby("model_id")["inference_time_s"]
        .agg(["mean", "std"])
        .reset_index()
    )
    stats = stats[stats["model_id"].isin(MODEL_ORDER)]
    stats = stats.sort_values("mean", ascending=False)

    colors = [FW_COLORS.get(MODEL_FRAMEWORK.get(m, ""), "#999") for m in stats["model_id"]]
    labels = [MODEL_LABELS.get(m, m) for m in stats["model_id"]]

    y_pos = np.arange(len(stats))
    ax.barh(
        y_pos, stats["mean"].values, xerr=stats["std"].values,
        height=0.65, color=colors, edgecolor="none",
        capsize=2, error_kw=dict(lw=0.8, capthick=0.8),
    )

    xmax = (stats["mean"] + stats["std"]).max()
    for i, (_, row) in enumerate(stats.iterrows()):
        ax.text(
            row["mean"] + row["std"] + xmax * 0.02, i,
            f'{row["mean"]:.2f} s', va="center", fontsize=7,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Inference time per image (s)", fontsize=9)
    ax.set_title("C   GPU inference speed (RTX 3090)",
                 fontsize=11, fontweight="bold", loc="left", pad=10)
    ax.set_xlim(0, xmax * 1.25)
    ax.grid(axis="x", alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)

    legend_elements = [Patch(facecolor=c, label=fw) for fw, c in FW_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right", framealpha=0.9)

    missing = set(MODEL_ORDER) - set(stats["model_id"])
    if missing:
        names = ", ".join(MODEL_LABELS.get(m, m) for m in missing)
        print(f"  WARNING: No timing data for: {names}")


# ---------------------------------------------------------------------------
# Panel D — Per-image count lines
# ---------------------------------------------------------------------------

def plot_per_image_lines(counts_df, output_dir, dpi):
    """Grid of subplots: one per within-category image rank."""
    model_colors = make_model_colors("magma")

    img_median = (
        counts_df.groupby(["category", "image_name"])["cell_count"]
        .median()
        .reset_index(name="median_count")
    )
    img_median["rank"] = (
        img_median.groupby("category")["median_count"]
        .rank(method="first").astype(int) - 1
    )

    cat_overall = (
        counts_df.pivot_table(
            values="cell_count", index="model_id", columns="category", aggfunc="mean",
        )
        .reindex(columns=CATEGORY_ORDER)
        .median(axis=0)
    )
    sorted_cats = cat_overall.sort_values().index.tolist()

    max_rank = img_median.groupby("category")["rank"].max().min()
    n_ranks = max_rank + 1

    ncols = 5
    nrows = int(np.ceil(n_ranks / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 3.8),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    x = np.arange(len(sorted_cats))
    counts_ranked = counts_df.merge(
        img_median[["category", "image_name", "rank"]],
        on=["category", "image_name"],
    )

    ymax = 0
    for rank_idx in range(n_ranks):
        ax = axes[rank_idx]
        rank_data = counts_ranked[counts_ranked["rank"] == rank_idx]

        for idx, mid in enumerate(LEGEND_ORDER):
            mdata = rank_data[rank_data["model_id"] == mid]
            if mdata.empty:
                continue
            vals = []
            for cat in sorted_cats:
                row = mdata[mdata["category"] == cat]
                vals.append(row["cell_count"].values[0] if len(row) else np.nan)
            vals = np.array(vals, dtype=float)
            ymax = max(ymax, np.nanmax(vals)) if np.any(~np.isnan(vals)) else ymax
            ax.plot(
                x, vals,
                color=model_colors[mid], linestyle="-",
                marker=PANEL_MARKERS[idx % len(PANEL_MARKERS)],
                linewidth=1.2, markersize=4,
                markeredgewidth=0.4, markeredgecolor="white",
                alpha=0.85,
                label=MODEL_LABELS[mid] if rank_idx == 0 else None,
            )

        ax.set_title(f"Image rank {rank_idx + 1}", fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="both", alpha=0.2, linewidth=0.5, color="#888")
        ax.set_axisbelow(True)
        ax.set_xticks(x)
        if rank_idx >= n_ranks - ncols:
            ax.set_xticklabels(
                [CATEGORY_LABELS[c] for c in sorted_cats],
                rotation=40, ha="right", fontsize=7,
            )
        ax.set_xlim(-0.3, len(sorted_cats) - 0.7)

    for i in range(nrows):
        axes[i * ncols].set_ylabel("Cell count", fontsize=9)
    for ax in axes[:n_ranks]:
        ax.set_ylim(0, ymax * 1.05)
    for i in range(n_ranks, len(axes)):
        axes[i].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    fig.legend(ord_h, ord_l, fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5)

    fig.suptitle(
        "D   Per-image cell count (images ranked within each category by median count)",
        fontsize=11, fontweight="bold", x=0.01, ha="left", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])

    for ext in ("pdf", "png"):
        out = output_dir / f"panel_D_per_image.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel E — Per-category subplots
# ---------------------------------------------------------------------------

def plot_per_category_images(counts_df, output_dir, dpi):
    """3x3 grid: one subplot per category, images on x-axis."""
    model_colors = make_model_colors_sequential("tab20")

    img_median = (
        counts_df.groupby(["category", "image_name"])["cell_count"]
        .median()
        .reset_index(name="median_count")
    )
    img_median["rank"] = (
        img_median.groupby("category")["median_count"]
        .rank(method="first").astype(int) - 1
    )
    counts_ranked = counts_df.merge(
        img_median[["category", "image_name", "rank"]],
        on=["category", "image_name"],
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 13))
    axes = axes.ravel()

    for cat_idx, cat in enumerate(CATEGORY_ORDER):
        ax = axes[cat_idx]
        cat_data = counts_ranked[counts_ranked["category"] == cat]
        n_images = cat_data["rank"].nunique()

        for idx, mid in enumerate(LEGEND_ORDER):
            mdata = cat_data[cat_data["model_id"] == mid].sort_values("rank")
            if mdata.empty:
                continue
            ax.plot(
                mdata["rank"].values, mdata["cell_count"].values,
                color=model_colors[mid], linestyle="-",
                marker=PANEL_MARKERS[idx % len(PANEL_MARKERS)],
                linewidth=1.2, markersize=5.5,
                markeredgewidth=0.4, markeredgecolor="white",
                alpha=0.85,
                label=MODEL_LABELS[mid] if cat_idx == 0 else None,
            )

        ax.set_title(CATEGORY_LABELS[cat], fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5, color="#888")
        ax.set_axisbelow(True)
        ax.set_xlabel("Image (sorted by median count)", fontsize=8)
        ax.set_ylabel("Cell count", fontsize=9)
        ax.set_xlim(-0.3, n_images - 0.7)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(n_images))
        ax.set_xticklabels(np.arange(1, n_images + 1), fontsize=7)

    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    fig.legend(ord_h, ord_l, fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5)

    fig.suptitle(
        "E   Cell count per image within each difficulty category",
        fontsize=12, fontweight="bold", x=0.01, ha="left", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])

    for ext in ("pdf", "png"):
        out = output_dir / f"panel_E_per_category.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot cell segmentation benchmark results",
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="Directory with counts.csv and timing.csv (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for figures (default: <results-dir>/figures)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Figure resolution (default: 300)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    results_dir = Path(args.results_dir) if args.results_dir else script_dir / "results"
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = pd.read_csv(results_dir / "counts.csv")
    timing = pd.read_csv(results_dir / "timing.csv")

    print(f"Counts: {len(counts)} rows, {counts['model_id'].nunique()} models")
    print(f"Timing: {len(timing)} rows, "
          f"{timing['inference_time_s'].notna().sum()} with valid times")

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.pad": 3,
        "ytick.major.pad": 3,
    })

    def save_fig(fig, name):
        for ext in ("pdf", "png"):
            out = output_dir / f"{name}.{ext}"
            fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
            print(f"Saved: {out}")
        plt.close(fig)

    fig_a, ax_a = plt.subplots(figsize=(14, 6))
    plot_count_lines(ax_a, counts)
    save_fig(fig_a, "panel_A_counts")

    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    plot_agreement(ax_b, counts)
    save_fig(fig_b, "panel_B_agreement")

    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    plot_timing(ax_c, timing)
    save_fig(fig_c, "panel_C_timing")

    plot_per_image_lines(counts, output_dir, args.dpi)
    plot_per_category_images(counts, output_dir, args.dpi)

    print("\n" + "=" * 60)
    print("Quick stats")
    print("=" * 60)

    pivot = counts.pivot_table(
        values="cell_count", index="category", columns="model_id", aggfunc="mean",
    ).reindex(columns=MODEL_ORDER, index=CATEGORY_ORDER)

    cat_median = pivot.median(axis=1)
    print("\nCategory median count (across models):")
    for cat in CATEGORY_ORDER:
        if cat in cat_median:
            print(f"  {CATEGORY_LABELS[cat]:18s}  {cat_median[cat]:7.0f}")

    valid_t = timing.dropna(subset=["inference_time_s"])
    if not valid_t.empty:
        t_stats = valid_t.groupby("model_id")["inference_time_s"].mean()
        fastest = t_stats.idxmin()
        slowest = t_stats.idxmax()
        print(f"\nFastest: {MODEL_LABELS.get(fastest, fastest)} ({t_stats[fastest]:.3f} s/img)")
        print(f"Slowest: {MODEL_LABELS.get(slowest, slowest)} ({t_stats[slowest]:.3f} s/img)")


if __name__ == "__main__":
    main()
