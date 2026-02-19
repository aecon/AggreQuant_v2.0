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
    "cellpose_cyto2_with_nuc",
    "cellpose_cyto3_no_nuc",
    "cellpose_cyto3_with_nuc",
    "instanseg_fluorescence",
    # Two-channel (DAPI + FarRed cell channel)
    "deepcell_mesmer_with_cell",
    "cellpose_cyto2_with_cell",
    "cellpose_cyto3_with_cell",
    "instanseg_fluorescence_with_cell",
]

N_SINGLE = 9  # first 9 entries are single-channel

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
# Panel A — Count line plot
# ---------------------------------------------------------------------------

def plot_count_lines(ax, counts_df):
    """Line plot of mean nuclei count per model across categories.

    X-axis: categories sorted by median count (low → high).
    Y-axis: mean nuclei count.
    One line per model, styled by family / input type.
    """
    pivot = counts_df.pivot_table(
        values="nuclei_count", index="model_id", columns="category",
        aggfunc="mean",
    )
    pivot = pivot.reindex(index=MODEL_ORDER, columns=CATEGORY_ORDER)

    # Sort categories by median count across models (ascending)
    cat_median = pivot.median(axis=0)
    sorted_cats = cat_median.sort_values().index.tolist()
    pivot = pivot[sorted_cats]

    x = np.arange(len(sorted_cats))

    # --- Consensus band (IQR across all models) ---
    q25 = pivot.quantile(0.25, axis=0).values
    q75 = pivot.quantile(0.75, axis=0).values
    ax.fill_between(x, q25, q75, color="#B0B0B0", alpha=0.5, zorder=0,
                    label="_nolegend_")

    # --- Color by model family, shades for variants ---
    _FAMILY_CMAP = {
        "StarDist":      None,
        "DeepCell":      "Greens",
        "CP_nuclei":     None,
        "CP_cyto2":      "RdPu",
        "CP_cyto3":      "Reds",
        "InstanSeg":     "Blues",
    }
    _FAMILY_MEMBERS = {
        "StarDist":  ["stardist_2d_fluo"],
        "DeepCell":  ["deepcell_nuclear", "deepcell_mesmer",
                      "deepcell_mesmer_with_cell"],
        "CP_nuclei": ["cellpose_nuclei"],
        "CP_cyto2":  ["cellpose_cyto2_no_nuc", "cellpose_cyto2_with_nuc",
                      "cellpose_cyto2_with_cell"],
        "CP_cyto3":  ["cellpose_cyto3_no_nuc", "cellpose_cyto3_with_nuc",
                      "cellpose_cyto3_with_cell"],
        "InstanSeg": ["instanseg_fluorescence",
                      "instanseg_fluorescence_with_cell"],
    }

    model_colors = {}
    for fam, members in _FAMILY_MEMBERS.items():
        if fam == "StarDist":
            model_colors[members[0]] = "black"
        elif fam == "CP_nuclei":
            model_colors[members[0]] = "#E65100"
        else:
            cmap = mpl.colormaps[_FAMILY_CMAP[fam]]
            n = len(members)
            positions = np.linspace(0.45, 0.90, n) if n > 1 else [0.65]
            for i, mid in enumerate(members):
                model_colors[mid] = cmap(positions[i])

    # Base models get thicker lines; variants (+nuc, +cell) get thinner
    _BASE_MODELS = {
        "stardist_2d_fluo", "deepcell_nuclear", "deepcell_mesmer",
        "cellpose_nuclei", "cellpose_cyto2_no_nuc", "cellpose_cyto3_no_nuc",
        "instanseg_fluorescence",
    }

    _MARKERS = ["o", "s", "D", "^", "v", "<", ">", "P", "X", "p", "h", "*", "d"]

    for idx, mid in enumerate(MODEL_ORDER):
        if mid not in pivot.index:
            continue
        is_base = mid in _BASE_MODELS
        ax.plot(
            x, pivot.loc[mid].values,
            color=model_colors[mid], linestyle="-",
            marker=_MARKERS[idx % len(_MARKERS)],
            linewidth=2.2 if is_base else 1.2,
            markersize=6 if is_base else 4.5,
            markeredgewidth=0.6, markeredgecolor="white",
            alpha=1.0 if is_base else 0.8,
            zorder=3 if is_base else 2,
            label=MODEL_LABELS[mid],
        )

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
    _LEGEND_ORDER = [
        "stardist_2d_fluo",
        "cellpose_nuclei",
        "cellpose_cyto2_no_nuc", "cellpose_cyto2_with_nuc",
        "cellpose_cyto2_with_cell",
        "cellpose_cyto3_no_nuc", "cellpose_cyto3_with_nuc",
        "cellpose_cyto3_with_cell",
        "deepcell_nuclear", "deepcell_mesmer", "deepcell_mesmer_with_cell",
        "instanseg_fluorescence", "instanseg_fluorescence_with_cell",
    ]
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

    For each image, the CV of nuclei counts across the 9 single-channel
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
    ax.set_title("B   Inter-model agreement (9 single-channel)",
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
