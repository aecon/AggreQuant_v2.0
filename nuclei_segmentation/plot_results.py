#!/usr/bin/env python
"""
Plot nuclei segmentation benchmark results.

Generates a publication-ready supplementary figure with three panels:
  A — Line plot of mean nuclei count per model across categories
  B — Inter-model count agreement (CV) per category
  C — Mean GPU inference time per model

Reads counts.csv and timing.csv produced by run_benchmark.py.

Usage:
    python plot_results.py [--results-dir results] [--output-dir results/figures]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    # Single-channel (DAPI only)
    "stardist_2d_fluo",
    "deepcell_nuclear",
    "deepcell_mesmer",
    "cellpose_nuclei",
    "cellpose_cyto2_no_nuc",
    "cellpose_cyto3_no_nuc",
    "instanseg_fluorescence",
    # Two-channel (DAPI + FarRed cell channel)
    "deepcell_mesmer_with_cell",
    "cellpose_cyto2_with_cell",
    "cellpose_cyto3_with_cell",
    "instanseg_fluorescence_with_cell",
]

N_SINGLE = 7  # first 7 entries are single-channel

MODEL_LABELS = {
    "stardist_2d_fluo":                 "StarDist 2D",
    "deepcell_nuclear":                 "DeepCell Nuclear",
    "deepcell_mesmer":                  "Mesmer",
    "cellpose_nuclei":                  "Cellpose nuclei",
    "cellpose_cyto2_no_nuc":            "Cellpose cyto2",
    "cellpose_cyto2_with_nuc":          "Cellpose cyto2 +nuc",
    "cellpose_cyto3_no_nuc":            "Cellpose cyto3",
    "cellpose_cyto3_with_nuc":          "Cellpose cyto3 +nuc",
    "instanseg_fluorescence":           "InstanSeg",
    "deepcell_mesmer_with_cell":        "Mesmer +cell",
    "cellpose_cyto2_with_cell":         "Cellpose cyto2 +cell",
    "cellpose_cyto3_with_cell":         "Cellpose cyto3 +cell",
    "instanseg_fluorescence_with_cell": "InstanSeg +cell",
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
    "01_low_confluency":         "Low confluency",
    "02_high_confluency":        "High confluency",
    "03_clustered_touching":     "Clustered",
    "04_mitotic":                "Mitotic",
    "05_defocused":              "Defocused",
    "06_flatfield_inhomogeneity":"Flat-field",
    "07_low_intensity":          "Low intensity",
    "08_high_intensity":         "High intensity",
    "09_debris_artifacts":       "Debris",
}

MODEL_FRAMEWORK = {
    "stardist_2d_fluo":                 "TensorFlow",
    "deepcell_nuclear":                 "TensorFlow",
    "deepcell_mesmer":                  "TensorFlow",
    "deepcell_mesmer_with_cell":        "TensorFlow",
    "cellpose_nuclei":                  "PyTorch",
    "cellpose_cyto2_no_nuc":            "PyTorch",
    "cellpose_cyto2_with_nuc":          "PyTorch",
    "cellpose_cyto3_no_nuc":            "PyTorch",
    "cellpose_cyto3_with_nuc":          "PyTorch",
    "cellpose_cyto2_with_cell":         "PyTorch",
    "cellpose_cyto3_no_nuc":            "PyTorch",
    "cellpose_cyto3_with_nuc":          "PyTorch",
    "cellpose_cyto3_with_cell":         "PyTorch",
    "instanseg_fluorescence":           "PyTorch",
    "instanseg_fluorescence_with_cell": "PyTorch",
}

FW_COLORS = {"TensorFlow": "#CE93D8", "PyTorch": "#64B5F6"}

# Visual encoding for each model: (color, linestyle, marker)
# Color  = model family (green=StarDist, red=DeepCell, blue=Cellpose, purple=InstanSeg)
# Style  = input type (solid=base, dashdot=+nuc, dashed=+cell)
# Marker = model variant
MODEL_STYLE = {
    "stardist_2d_fluo":                 ("#2E7D32", "-",  "o"),  # green, solid, circle
    "deepcell_nuclear":                 ("#C62828", "-",  "s"),  # dark red, solid, square
    "deepcell_mesmer":                  ("#E65100", "-",  "D"),  # orange, solid, diamond
    "deepcell_mesmer_with_cell":        ("#E65100", "--", "D"),  # orange, dashed, diamond
    "cellpose_nuclei":                  ("#1565C0", "-",  "o"),  # blue, solid, circle
    "cellpose_cyto2_no_nuc":            ("#42A5F5", "-",  "^"),  # light blue, solid, tri-up
    "cellpose_cyto2_with_nuc":          ("#42A5F5", "-.", "^"),  # light blue, dashdot, tri-up
    "cellpose_cyto2_with_cell":         ("#42A5F5", "--", "^"),  # light blue, dashed, tri-up
    "cellpose_cyto3_no_nuc":            ("#0D47A1", "-",  "v"),  # navy, solid, tri-down
    "cellpose_cyto3_with_nuc":          ("#0D47A1", "-.", "v"),  # navy, dashdot, tri-down
    "cellpose_cyto3_with_cell":         ("#0D47A1", "--", "v"),  # navy, dashed, tri-down
    "instanseg_fluorescence":           ("#7B1FA2", "-",  "P"),  # purple, solid, plus
    "instanseg_fluorescence_with_cell": ("#7B1FA2", "--", "P"),  # purple, dashed, plus
}


# ---------------------------------------------------------------------------
# Shared color helpers
# ---------------------------------------------------------------------------

# Legend order used across panels
LEGEND_ORDER = [
    "stardist_2d_fluo",
    "cellpose_nuclei",
    "cellpose_cyto2_no_nuc",
    "cellpose_cyto2_with_cell",
    "cellpose_cyto3_no_nuc",
    "cellpose_cyto3_with_cell",
    "deepcell_nuclear", "deepcell_mesmer", "deepcell_mesmer_with_cell",
    "instanseg_fluorescence", "instanseg_fluorescence_with_cell",
]

PANEL_MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "p", "h", "*", "d"]


def _interleaved_indices(n):
    """Return indices 0..n-1 reordered so adjacent entries are far apart.

    Alternates picking from the low and high ends:
    0, n-1, 1, n-2, 2, n-3, ...
    """
    order = []
    lo, hi = 0, n - 1
    while lo <= hi:
        order.append(lo)
        if lo != hi:
            order.append(hi)
        lo += 1
        hi -= 1
    return order


def make_model_colors_interleaved(cmap_name="magma"):
    """Assign colormap colors to LEGEND_ORDER with maximal contrast."""
    cmap = mpl.colormaps[cmap_name]
    n = len(LEGEND_ORDER)
    n_slots = n + 2  # trim the lightest end
    # Sequential positions in the colormap
    seq_colors = [cmap(i / (n_slots - 1)) for i in range(n)]
    # Re-assign so adjacent legend entries get distant colors
    interleaved = _interleaved_indices(n)
    # interleaved[k] = which sequential color position legend entry k gets
    model_colors = {}
    for legend_idx, mid in enumerate(LEGEND_ORDER):
        model_colors[mid] = seq_colors[interleaved[legend_idx]]
    return model_colors


def make_model_colors_sequential(cmap_name="magma"):
    """Assign colormap colors to LEGEND_ORDER sequentially."""
    cmap = mpl.colormaps[cmap_name]
    n = len(LEGEND_ORDER)
    n_slots = n + 2
    return {
        mid: cmap(i / (n_slots - 1))
        for i, mid in enumerate(LEGEND_ORDER)
    }


# ---------------------------------------------------------------------------
# Panel A — Count line plot
# ---------------------------------------------------------------------------

def plot_count_lines(ax, counts_df):
    """Line plot of mean nuclei count per model across categories.

    X-axis: categories sorted by median count (low → high).
    Y-axis: mean nuclei count with SD error bars.
    One line per model, styled by family / input type.
    """
    pivot_mean = counts_df.pivot_table(
        values="nuclei_count", index="model_id", columns="category",
        aggfunc="mean",
    )
    pivot_std = counts_df.pivot_table(
        values="nuclei_count", index="model_id", columns="category",
        aggfunc="std",
    )
    pivot_mean = pivot_mean.reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)
    pivot_std = pivot_std.reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)

    # Sort categories by median count across models (ascending)
    cat_median = pivot_mean.median(axis=0)
    sorted_cats = cat_median.sort_values().index.tolist()
    pivot_mean = pivot_mean[sorted_cats]
    pivot_std = pivot_std[sorted_cats]

    x = np.arange(len(sorted_cats))

    # --- Legend order determines cividis color assignment ---
    _LEGEND_ORDER = [
        "stardist_2d_fluo",
        "cellpose_nuclei",
        "cellpose_cyto2_no_nuc",
        "cellpose_cyto2_with_cell",
        "cellpose_cyto3_no_nuc",
        "cellpose_cyto3_with_cell",
        "deepcell_nuclear", "deepcell_mesmer", "deepcell_mesmer_with_cell",
        "instanseg_fluorescence", "instanseg_fluorescence_with_cell",
    ]
    cmap = mpl.colormaps["magma"]
    n_models = len(_LEGEND_ORDER)
    n_slots = n_models + 2  # extra slots to trim the yellow end
    model_colors = {
        mid: cmap(i / (n_slots - 1))
        for i, mid in enumerate(_LEGEND_ORDER)
    }

    _MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "p", "h", "*", "d"]

    # Fixed x-offset per model to separate overlapping error bars
    offsets = np.linspace(-0.15, 0.15, n_models)

    # Plot in legend order so zorder is consistent
    for idx, mid in enumerate(_LEGEND_ORDER):
        if mid not in pivot_mean.index:
            continue
        container = ax.errorbar(
            x + offsets[idx], pivot_mean.loc[mid].values,
            yerr=pivot_std.loc[mid].values,
            color=model_colors[mid], linestyle="-",
            marker=_MARKERS[idx % len(_MARKERS)],
            linewidth=1.5,
            markersize=5,
            markeredgewidth=0.6, markeredgecolor="white",
            alpha=0.9,
            capsize=1, elinewidth=0.8,
            zorder=2,
            label=MODEL_LABELS[mid],
        )
        # Make error bar lines and caps transparent
        for bar_line in container.lines[2]:
            bar_line.set_alpha(0.2)
        for cap in container.lines[1]:
            cap.set_alpha(0.2)

    # --- Style: clean spines, grid on both axes ---
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="both", alpha=0.2, linewidth=0.5, color="#888")
    ax.set_axisbelow(True)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CATEGORY_LABELS[c] for c in sorted_cats],
        rotation=40, ha="right", fontsize=9,
    )
    ax.set_ylabel("Mean nuclei count", fontsize=10)
    ax.set_xlim(-0.3, len(sorted_cats) - 0.7)
    ax.set_ylim(bottom=0)

    # --- Legend outside plot on the right ---
    handles, labels = ax.get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in _LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in _LEGEND_ORDER if MODEL_LABELS[m] in label_to_handle]

    ax.legend(
        ord_h, ord_l,
        fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0),
        ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5,
        borderaxespad=0,
    )

    ax.set_title("A   Mean nuclei count per model across difficulty categories",
                 fontsize=11, fontweight="bold", loc="left", pad=10)


# ---------------------------------------------------------------------------
# Panel B — Inter-model agreement
# ---------------------------------------------------------------------------

def plot_agreement(ax, counts_df):
    """Box plot of inter-model count CV per category.

    For each image, the CV of nuclei counts across the 7 single-channel
    models measures how much the models disagree.
    """
    sc_models = MODEL_ORDER[:N_SINGLE]
    sc = counts_df[counts_df["model_id"].isin(sc_models)].copy()

    # Per-image CV across single-channel models
    def cv_func(x):
        m = x.mean()
        return (x.std() / m * 100) if m > 0 else np.nan

    cv = (
        sc.groupby(["image_name", "category"])["nuclei_count"]
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
    ax.set_title("B   Inter-model agreement (7 single-channel)",
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

    # Sort fastest on top (ascending mean → reversed for barh so fastest is at top)
    stats = stats.sort_values("mean", ascending=False)

    colors = [
        FW_COLORS.get(MODEL_FRAMEWORK.get(m, ""), "#999")
        for m in stats["model_id"]
    ]
    labels = [MODEL_LABELS.get(m, m) for m in stats["model_id"]]

    y_pos = np.arange(len(stats))
    ax.barh(
        y_pos, stats["mean"].values, xerr=stats["std"].values,
        height=0.65, color=colors, edgecolor="none",
        capsize=2, error_kw=dict(lw=0.8, capthick=0.8),
    )

    # Annotate bar values
    xmax = (stats["mean"] + stats["std"]).max()
    for i, (_, row) in enumerate(stats.iterrows()):
        offset = xmax * 0.02
        ax.text(
            row["mean"] + row["std"] + offset, i,
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

    # Framework legend
    legend_elements = [Patch(facecolor=c, label=fw) for fw, c in FW_COLORS.items()]
    ax.legend(handles=legend_elements, fontsize=7, loc="upper right",
              framealpha=0.9)

    # Warn about missing models
    missing = set(MODEL_ORDER) - set(stats["model_id"])
    if missing:
        names = ", ".join(MODEL_LABELS.get(m, m) for m in missing)
        print(f"  WARNING: No timing data for: {names}")


# ---------------------------------------------------------------------------
# Panel D — Per-image count lines (one subplot per rank position)
# ---------------------------------------------------------------------------

def plot_per_image_lines(counts_df, output_dir, dpi):
    """Grid of subplots: one per within-category image rank.

    Within each category, images are sorted by their median nuclei count
    across all models.  Subplot k shows each model's count for the k-th
    ranked image in every category.  Categories on x-axis, same layout as
    Panel A.
    """
    _LEGEND_ORDER = [
        "stardist_2d_fluo",
        "cellpose_nuclei",
        "cellpose_cyto2_no_nuc",
        "cellpose_cyto2_with_cell",
        "cellpose_cyto3_no_nuc",
        "cellpose_cyto3_with_cell",
        "deepcell_nuclear", "deepcell_mesmer", "deepcell_mesmer_with_cell",
        "instanseg_fluorescence", "instanseg_fluorescence_with_cell",
    ]
    cmap = mpl.colormaps["magma"]
    n_models = len(_LEGEND_ORDER)
    n_slots = n_models + 2
    model_colors = {
        mid: cmap(i / (n_slots - 1))
        for i, mid in enumerate(_LEGEND_ORDER)
    }
    _MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "p", "h", "*", "d"]

    # --- Compute median count per image across models, rank within category ---
    img_median = (
        counts_df.groupby(["category", "image_name"])["nuclei_count"]
        .median()
        .reset_index(name="median_count")
    )
    img_median["rank"] = (
        img_median.groupby("category")["median_count"]
        .rank(method="first").astype(int) - 1  # 0-based
    )

    # Category sort order: by overall median (same as Panel A)
    cat_overall = (
        counts_df.pivot_table(
            values="nuclei_count", index="model_id", columns="category",
            aggfunc="mean",
        )
        .reindex(columns=CATEGORY_ORDER)
        .median(axis=0)
    )
    sorted_cats = cat_overall.sort_values().index.tolist()

    # Maximum rank that all 9-category groups share (min images per category)
    max_rank = img_median.groupby("category")["rank"].max().min()  # 9 (0-based)
    n_ranks = max_rank + 1  # 10

    # Grid layout
    ncols = 5
    nrows = int(np.ceil(n_ranks / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(ncols * 5.5, nrows * 3.8),
        sharex=True, sharey=True,
    )
    axes = axes.ravel()

    x = np.arange(len(sorted_cats))

    # Merge rank into counts
    counts_ranked = counts_df.merge(
        img_median[["category", "image_name", "rank"]],
        on=["category", "image_name"],
    )

    # Global y-max for shared axis
    ymax = 0

    for rank_idx in range(n_ranks):
        ax = axes[rank_idx]
        rank_data = counts_ranked[counts_ranked["rank"] == rank_idx]

        for idx, mid in enumerate(_LEGEND_ORDER):
            mdata = rank_data[rank_data["model_id"] == mid]
            if mdata.empty:
                continue
            # One value per category for this model at this rank
            vals = []
            for cat in sorted_cats:
                row = mdata[mdata["category"] == cat]
                vals.append(row["nuclei_count"].values[0] if len(row) else np.nan)
            vals = np.array(vals, dtype=float)
            ymax = max(ymax, np.nanmax(vals)) if np.any(~np.isnan(vals)) else ymax
            ax.plot(
                x, vals,
                color=model_colors[mid], linestyle="-",
                marker=_MARKERS[idx % len(_MARKERS)],
                linewidth=1.2,
                markersize=4,
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
        if rank_idx >= n_ranks - ncols:  # bottom row
            ax.set_xticklabels(
                [CATEGORY_LABELS[c] for c in sorted_cats],
                rotation=40, ha="right", fontsize=7,
            )
        ax.set_xlim(-0.3, len(sorted_cats) - 0.7)

    # Shared y-axis label and limits
    for i in range(nrows):
        axes[i * ncols].set_ylabel("Nuclei count", fontsize=9)
    for ax in axes[:n_ranks]:
        ax.set_ylim(0, ymax * 1.05)

    # Hide unused axes
    for i in range(n_ranks, len(axes)):
        axes[i].set_visible(False)

    # Legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in _LEGEND_ORDER
             if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in _LEGEND_ORDER
             if MODEL_LABELS[m] in label_to_handle]
    fig.legend(
        ord_h, ord_l,
        fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5,
    )

    fig.suptitle(
        "D   Per-image nuclei count (images ranked within each category by median count)",
        fontsize=11, fontweight="bold", x=0.01, ha="left", y=1.01,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1.0])

    for ext in ("pdf", "png"):
        out = output_dir / f"panel_D_per_image.{ext}"
        fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Panel E — Per-category subplots (one subplot per category, images on x)
# ---------------------------------------------------------------------------

def plot_per_category_images(counts_df, output_dir, dpi):
    """3x3 grid: one subplot per category.

    Within each subplot, x-axis = images sorted by median count across models,
    y-axis = nuclei count, one line per model.
    """
    model_colors = make_model_colors_sequential("tab20")

    # Rank images within each category by median count across models
    img_median = (
        counts_df.groupby(["category", "image_name"])["nuclei_count"]
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
        x = np.arange(n_images)

        for idx, mid in enumerate(LEGEND_ORDER):
            mdata = cat_data[cat_data["model_id"] == mid].sort_values("rank")
            if mdata.empty:
                continue
            ax.plot(
                mdata["rank"].values, mdata["nuclei_count"].values,
                color=model_colors[mid], linestyle="-",
                marker=PANEL_MARKERS[idx % len(PANEL_MARKERS)],
                linewidth=1.2,
                markersize=5.5,
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
        ax.set_ylabel("Nuclei count", fontsize=9)
        ax.set_xlim(-0.3, n_images - 0.7)
        ax.set_ylim(bottom=0)
        ax.set_xticks(np.arange(n_images))
        ax.set_xticklabels(np.arange(1, n_images + 1), fontsize=7)

    # Legend from first subplot
    handles, labels = axes[0].get_legend_handles_labels()
    label_to_handle = dict(zip(labels, handles))
    ord_h = [label_to_handle[MODEL_LABELS[m]] for m in LEGEND_ORDER
             if MODEL_LABELS[m] in label_to_handle]
    ord_l = [MODEL_LABELS[m] for m in LEGEND_ORDER
             if MODEL_LABELS[m] in label_to_handle]
    fig.legend(
        ord_h, ord_l,
        fontsize=7.5, loc="upper right", bbox_to_anchor=(0.99, 0.99),
        ncol=1, framealpha=0.95, edgecolor="#ccc", handlelength=2.5,
    )

    fig.suptitle(
        "E   Nuclei count per image within each difficulty category",
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
        description="Plot nuclei segmentation benchmark results",
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
    results_dir = (
        Path(args.results_dir) if args.results_dir else script_dir / "results"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir else results_dir / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load data ---
    counts = pd.read_csv(results_dir / "counts.csv")
    timing = pd.read_csv(results_dir / "timing.csv")

    print(f"Counts: {len(counts)} rows, {counts['model_id'].nunique()} models")
    print(f"Timing: {len(timing)} rows, "
          f"{timing['inference_time_s'].notna().sum()} with valid times")

    # --- Style ---
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

    # --- Panel A: count lines ---
    fig_a, ax_a = plt.subplots(figsize=(14, 6))
    plot_count_lines(ax_a, counts)
    save_fig(fig_a, "panel_A_counts")

    # --- Panel B: agreement ---
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    plot_agreement(ax_b, counts)
    save_fig(fig_b, "panel_B_agreement")

    # --- Panel C: timing ---
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    plot_timing(ax_c, timing)
    save_fig(fig_c, "panel_C_timing")

    # --- Panel D: per-image ---
    plot_per_image_lines(counts, output_dir, args.dpi)

    # --- Panel E: per-category ---
    plot_per_category_images(counts, output_dir, args.dpi)

    # --- Console summary ---
    print("\n" + "=" * 60)
    print("Quick stats")
    print("=" * 60)

    pivot = counts.pivot_table(
        values="nuclei_count", index="category", columns="model_id",
        aggfunc="mean",
    ).reindex(columns=MODEL_ORDER, index=CATEGORY_ORDER)

    cat_median = pivot.median(axis=1)
    print("\nCategory median count (across models):")
    for cat in CATEGORY_ORDER:
        print(f"  {CATEGORY_LABELS[cat]:18s}  {cat_median[cat]:7.0f}")

    valid_t = timing.dropna(subset=["inference_time_s"])
    if not valid_t.empty:
        t_stats = valid_t.groupby("model_id")["inference_time_s"].mean()
        fastest = t_stats.idxmin()
        slowest = t_stats.idxmax()
        print(f"\nFastest: {MODEL_LABELS[fastest]} ({t_stats[fastest]:.3f} s/img)")
        print(f"Slowest: {MODEL_LABELS[slowest]} ({t_stats[slowest]:.3f} s/img)")


if __name__ == "__main__":
    main()
