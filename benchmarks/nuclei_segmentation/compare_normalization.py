#!/usr/bin/env python
"""
Compare nuclei segmentation results with and without background normalization.

Generates:
  - Figure 1: CV scatter (raw vs normalized, per image, colored by category)
  - Figure 2: Delta-count box plots stratified by category
  - Figure 3: Median area shift per model (raw vs normalized)
  - Figure 5: Heatmap of median % change in count (model × category)
  - Figure 6: Heatmap of median % change in area (model × category)
  - Console + CSV summary table with per-category statistics and p-values

Usage:
    python compare_normalization.py [--raw-dir results] [--norm-dir results_normalized]
                                    [--output-dir results_normalized/figures]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats


# ---- Constants (shared with plot_results.py) --------------------------------

SINGLE_CHANNEL_MODELS = [
    "stardist_2d_fluo",
    "deepcell_nuclear",
    "deepcell_mesmer",
    "cellpose_nuclei",
    "cellpose_cyto2_no_nuc",
    "cellpose_cyto3_no_nuc",
    "instanseg_fluorescence",
]

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
    "01_low_confluency":          "Low confluency",
    "02_high_confluency":         "High confluency",
    "03_clustered_touching":      "Clustered",
    "04_mitotic":                  "Mitotic",
    "05_defocused":                "Defocused",
    "06_flatfield_inhomogeneity": "Flat-field",
    "07_low_intensity":            "Low intensity",
    "08_high_intensity":           "High intensity",
    "09_debris_artifacts":         "Debris",
}

CATEGORY_COLORS = {
    "01_low_confluency":          "#90CAF9",
    "02_high_confluency":         "#1565C0",
    "03_clustered_touching":      "#AB47BC",
    "04_mitotic":                  "#EF5350",
    "05_defocused":                "#BDBDBD",
    "06_flatfield_inhomogeneity": "#FFB74D",
    "07_low_intensity":            "#78909C",
    "08_high_intensity":           "#FFEE58",
    "09_debris_artifacts":         "#A1887F",
}


# ---- Data loading -----------------------------------------------------------

def load_and_merge(raw_dir, norm_dir):
    """Load both CSVs and merge on (image_name, category, model_id)."""
    raw = pd.read_csv(raw_dir / "counts.csv")
    norm = pd.read_csv(norm_dir / "counts.csv")

    merged = raw.merge(
        norm,
        on=["image_name", "category", "model_id"],
        suffixes=("_raw", "_norm"),
    )
    return merged


def compute_cv_per_image(df, count_col, models):
    """Compute CV of nuclei counts across models for each image."""
    sub = df[df["model_id"].isin(models)]
    grouped = sub.groupby(["image_name", "category"])[count_col]
    cv = grouped.agg(lambda x: x.std() / x.mean() if x.mean() > 0 else np.nan)
    return cv.reset_index().rename(columns={count_col: "cv"})


# ---- Figure 1: CV scatter ---------------------------------------------------

def plot_cv_scatter(cv_raw, cv_norm, output_dir):
    """Scatter CV_raw vs CV_norm per image, colored by category."""
    merged = cv_raw.merge(
        cv_norm, on=["image_name", "category"], suffixes=("_raw", "_norm")
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    for cat in CATEGORY_ORDER:
        sub = merged[merged["category"] == cat]
        if sub.empty:
            continue
        ax.scatter(
            sub["cv_raw"], sub["cv_norm"],
            c=CATEGORY_COLORS[cat],
            label=CATEGORY_LABELS[cat],
            s=40, alpha=0.8, edgecolors="white", linewidths=0.3,
        )

    # Diagonal
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")

    ax.set_xlabel("CV (raw)", fontsize=11)
    ax.set_ylabel("CV (normalized)", fontsize=11)
    ax.set_title("Inter-model agreement: raw vs. normalized", fontsize=12)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"cv_scatter.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved cv_scatter.{{pdf,png}}")


# ---- Figure 2: Delta-count box plots ----------------------------------------

def plot_delta_count_boxes(merged, output_dir):
    """Box plots of (count_norm - count_raw) per category, all models."""
    merged["delta_count"] = merged["nuclei_count_norm"] - merged["nuclei_count_raw"]

    fig, ax = plt.subplots(figsize=(10, 5))

    data_by_cat = []
    labels = []
    colors = []
    for cat in CATEGORY_ORDER:
        sub = merged[merged["category"] == cat]
        if sub.empty:
            continue
        data_by_cat.append(sub["delta_count"].values)
        labels.append(CATEGORY_LABELS[cat])
        colors.append(CATEGORY_COLORS[cat])

    bp = ax.boxplot(
        data_by_cat, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", lw=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel("Δ nuclei count (normalized − raw)", fontsize=11)
    ax.set_title("Effect of background normalization on nuclei counts", fontsize=12)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"delta_count_boxes.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved delta_count_boxes.{{pdf,png}}")


# ---- Figure 3: Area shift per model -----------------------------------------

def plot_area_shift(merged, output_dir):
    """Paired comparison of median area per model (raw vs normalized)."""
    # Use single-channel models only
    sub = merged[merged["model_id"].isin(SINGLE_CHANNEL_MODELS)]

    model_stats = []
    for mid in SINGLE_CHANNEL_MODELS:
        msub = sub[sub["model_id"] == mid]
        raw_med = msub["median_area_px_raw"].median()
        norm_med = msub["median_area_px_norm"].median()
        model_stats.append({
            "model_id": mid,
            "label": MODEL_LABELS[mid],
            "median_area_raw": raw_med,
            "median_area_norm": norm_med,
        })
    mdf = pd.DataFrame(model_stats)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(mdf))
    width = 0.35

    ax.bar(x - width / 2, mdf["median_area_raw"], width, label="Raw",
           color="#64B5F6", edgecolor="white")
    ax.bar(x + width / 2, mdf["median_area_norm"], width, label="Normalized",
           color="#FF8A65", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(mdf["label"], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Median nucleus area (px)", fontsize=11)
    ax.set_title("Nucleus area: raw vs. normalized", fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"area_shift.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved area_shift.{{pdf,png}}")


# ---- Figure 4: Delta-count per model ----------------------------------------

def plot_delta_count_per_model(merged, output_dir):
    """Box plots of (count_norm - count_raw) per model (single-channel)."""
    sub = merged[merged["model_id"].isin(SINGLE_CHANNEL_MODELS)].copy()
    sub["delta_count"] = sub["nuclei_count_norm"] - sub["nuclei_count_raw"]

    fig, ax = plt.subplots(figsize=(8, 5))

    data_by_model = []
    labels = []
    for mid in SINGLE_CHANNEL_MODELS:
        msub = sub[sub["model_id"] == mid]
        data_by_model.append(msub["delta_count"].values)
        labels.append(MODEL_LABELS[mid])

    bp = ax.boxplot(
        data_by_model, patch_artist=True, widths=0.6,
        medianprops=dict(color="black", lw=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#B0BEC5")
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.axhline(0, color="black", ls="--", lw=0.8, alpha=0.5)
    ax.set_ylabel("Δ nuclei count (normalized − raw)", fontsize=11)
    ax.set_title("Per-model effect of background normalization", fontsize=12)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"delta_count_per_model.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved delta_count_per_model.{{pdf,png}}")


# ---- Figure 5: Heatmap of % change in count (model × category) -------------

def plot_heatmap_pct_count(merged, output_dir):
    """Heatmap: median % change in nuclei count per model and category."""
    sub = merged[
        merged["model_id"].isin(SINGLE_CHANNEL_MODELS)
    ].copy()
    sub["pct_delta_count"] = (
        100 * (sub["nuclei_count_norm"] - sub["nuclei_count_raw"])
        / sub["nuclei_count_raw"]
    )

    pt = sub.pivot_table(
        values="pct_delta_count", index="model_id",
        columns="category", aggfunc="median",
    )
    pt = pt.loc[SINGLE_CHANNEL_MODELS, CATEGORY_ORDER]
    row_labels = [MODEL_LABELS[m] for m in pt.index]
    col_labels = [CATEGORY_LABELS[c] for c in pt.columns]

    vmax = max(abs(pt.values.min()), abs(pt.values.max()), 1)
    cnorm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pt.values, cmap="RdBu_r", norm=cnorm, aspect="auto")

    # Annotate cells
    for i in range(pt.shape[0]):
        for j in range(pt.shape[1]):
            val = pt.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Median % change in nuclei count: (normalized − raw) / raw", fontsize=12)
    fig.colorbar(im, ax=ax, label="% change", shrink=0.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"heatmap_pct_count.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved heatmap_pct_count.{{pdf,png}}")


# ---- Figure 6: Heatmap of % change in area (model × category) --------------

def plot_heatmap_pct_area(merged, output_dir):
    """Heatmap: median % change in median nucleus area per model and category."""
    sub = merged[
        merged["model_id"].isin(SINGLE_CHANNEL_MODELS)
    ].copy()
    sub["pct_delta_area"] = (
        100 * (sub["median_area_px_norm"] - sub["median_area_px_raw"])
        / sub["median_area_px_raw"]
    )

    pt = sub.pivot_table(
        values="pct_delta_area", index="model_id",
        columns="category", aggfunc="median",
    )
    pt = pt.loc[SINGLE_CHANNEL_MODELS, CATEGORY_ORDER]
    row_labels = [MODEL_LABELS[m] for m in pt.index]
    col_labels = [CATEGORY_LABELS[c] for c in pt.columns]

    vmax = max(abs(pt.values.min()), abs(pt.values.max()), 1)
    cnorm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(pt.values, cmap="RdBu_r", norm=cnorm, aspect="auto")

    for i in range(pt.shape[0]):
        for j in range(pt.shape[1]):
            val = pt.values[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.1f}%", ha="center", va="center",
                    fontsize=8, color=color)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title("Median % change in nucleus area: (normalized − raw) / raw", fontsize=12)
    fig.colorbar(im, ax=ax, label="% change", shrink=0.8)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"heatmap_pct_area.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved heatmap_pct_area.{{pdf,png}}")


# ---- Summary table -----------------------------------------------------------

def compute_summary_table(merged, cv_raw_df, cv_norm_df):
    """Per-category summary: median delta-count, delta-CV, Wilcoxon p-value."""
    merged["delta_count"] = merged["nuclei_count_norm"] - merged["nuclei_count_raw"]

    cv_merged = cv_raw_df.merge(
        cv_norm_df, on=["image_name", "category"], suffixes=("_raw", "_norm")
    )

    rows = []
    for cat in CATEGORY_ORDER:
        # Delta count (all models)
        cat_delta = merged[merged["category"] == cat]["delta_count"]
        median_delta = cat_delta.median()

        # CV comparison (per image)
        cat_cv = cv_merged[cv_merged["category"] == cat]
        if len(cat_cv) >= 2:
            delta_cv = (cat_cv["cv_norm"] - cat_cv["cv_raw"]).median()
            try:
                stat, p_cv = stats.wilcoxon(cat_cv["cv_raw"], cat_cv["cv_norm"])
            except ValueError:
                p_cv = np.nan
        else:
            delta_cv = np.nan
            p_cv = np.nan

        # Count Wilcoxon (per image, single-channel models only)
        sc = merged[
            (merged["category"] == cat)
            & (merged["model_id"].isin(SINGLE_CHANNEL_MODELS))
        ]
        img_count_raw = sc.groupby("image_name")["nuclei_count_raw"].mean()
        img_count_norm = sc.groupby("image_name")["nuclei_count_norm"].mean()
        if len(img_count_raw) >= 2:
            try:
                _, p_count = stats.wilcoxon(img_count_raw, img_count_norm)
            except ValueError:
                p_count = np.nan
        else:
            p_count = np.nan

        rows.append({
            "category": CATEGORY_LABELS[cat],
            "n_images": merged[merged["category"] == cat]["image_name"].nunique(),
            "median_delta_count": round(median_delta, 1),
            "median_delta_cv": round(delta_cv, 4) if not np.isnan(delta_cv) else np.nan,
            "p_value_cv": round(p_cv, 4) if not np.isnan(p_cv) else np.nan,
            "p_value_count": round(p_count, 4) if not np.isnan(p_count) else np.nan,
        })

    return pd.DataFrame(rows)


# ---- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir", type=Path, default=Path("results"),
        help="Directory with raw (non-normalized) results",
    )
    parser.add_argument(
        "--norm-dir", type=Path, default=Path("results_normalized"),
        help="Directory with background-normalized results",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_comparison"),
        help="Output directory for figures and summaries",
    )
    args = parser.parse_args()

    args.output_dir = args.output_dir
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    merged = load_and_merge(args.raw_dir, args.norm_dir)
    print(f"  {len(merged)} paired rows "
          f"({merged['image_name'].nunique()} images × "
          f"{merged['model_id'].nunique()} models)")

    # CV per image (single-channel models only)
    cv_raw = compute_cv_per_image(merged, "nuclei_count_raw", SINGLE_CHANNEL_MODELS)
    cv_norm = compute_cv_per_image(merged, "nuclei_count_norm", SINGLE_CHANNEL_MODELS)

    print("\nGenerating figures...")
    plot_cv_scatter(cv_raw, cv_norm, args.output_dir)
    plot_delta_count_boxes(merged, args.output_dir)
    plot_delta_count_per_model(merged, args.output_dir)
    plot_area_shift(merged, args.output_dir)
    plot_heatmap_pct_count(merged, args.output_dir)
    plot_heatmap_pct_area(merged, args.output_dir)

    print("\nSummary table:")
    summary = compute_summary_table(merged, cv_raw, cv_norm)
    print(summary.to_string(index=False))

    csv_path = args.output_dir / "normalization_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\n  Saved {csv_path}")


if __name__ == "__main__":
    main()
