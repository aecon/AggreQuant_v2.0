#!/usr/bin/env python
"""
Analyze per-nucleus intensity distributions across models.

For each model's own segmentation mask, measure the median DAPI intensity inside
each detected nucleus. Then plot overlaid histograms of nucleus count vs.
intensity to reveal whether models systematically miss dim or bright nuclei.

Usage:
    python analyze_intensity_detection.py [--category 06_flatfield_inhomogeneity]
                                          [--results-dir results]
                                          [--n-bins 50]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import tifffile


# ---- Constants ---------------------------------------------------------------

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
    "stardist_2d_fluo":      "StarDist 2D",
    "deepcell_nuclear":      "DeepCell Nuclear",
    "deepcell_mesmer":       "Mesmer",
    "cellpose_nuclei":       "Cellpose nuclei",
    "cellpose_cyto2_no_nuc": "Cellpose cyto2",
    "cellpose_cyto3_no_nuc": "Cellpose cyto3",
    "instanseg_fluorescence": "InstanSeg",
}

MODEL_COLORS = {
    "stardist_2d_fluo":      "#2E7D32",
    "deepcell_nuclear":      "#C62828",
    "deepcell_mesmer":       "#E65100",
    "cellpose_nuclei":       "#1565C0",
    "cellpose_cyto2_no_nuc": "#42A5F5",
    "cellpose_cyto3_no_nuc": "#0D47A1",
    "instanseg_fluorescence": "#7B1FA2",
}

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

CATEGORY_ORDER = list(CATEGORY_LABELS.keys())


# ---- Core analysis -----------------------------------------------------------

def measure_nuclei_intensities(data_dir, mask_dir, category, models):
    """For each model, measure median DAPI intensity of every detected nucleus."""
    cat_dir = data_dir / category
    dapi_files = sorted(cat_dir.glob("*390 - Blue*.tif"))

    records = []
    for dapi_path in dapi_files:
        image_name = dapi_path.name
        dapi = tifffile.imread(str(dapi_path)).astype(np.float32)

        for mid in models:
            mask_path = mask_dir / mid / image_name
            if not mask_path.exists():
                continue
            mask = tifffile.imread(str(mask_path))
            label_ids = np.unique(mask)
            label_ids = label_ids[label_ids > 0]

            # Vectorized: compute median intensity per label using ndimage-style
            # For speed, use bincount approach for mean, manual for median
            for lid in label_ids:
                pixels = dapi[mask == lid]
                records.append({
                    "image_name": image_name,
                    "model_id": mid,
                    "label_id": int(lid),
                    "median_intensity": np.median(pixels),
                    "mean_intensity": pixels.mean(),
                    "area_px": len(pixels),
                })

    return pd.DataFrame(records)


# ---- Plotting ----------------------------------------------------------------

def plot_intensity_histogram(df, category, output_dir, n_bins=50, label=""):
    """Overlaid count histogram with dim steps + smooth overlay, log y-axis."""
    cat_label = CATEGORY_LABELS.get(category, category)
    label_suffix = f"_{label}" if label else ""
    label_title = f" [{label}]" if label else ""

    all_intensities = df["median_intensity"]
    lo = all_intensities.min()
    hi = all_intensities.max()
    bin_edges = np.linspace(lo, hi, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mid in SINGLE_CHANNEL_MODELS:
        msub = df[df["model_id"] == mid]
        if msub.empty:
            continue
        counts, _ = np.histogram(msub["median_intensity"], bins=bin_edges)
        counts_float = counts.astype(float)
        counts_masked = np.where(counts > 0, counts_float, np.nan)
        color = MODEL_COLORS[mid]

        # Dim step bars
        ax.plot(
            bin_centers, counts_masked,
            color=color, lw=0.7, alpha=0.25, drawstyle="steps-mid",
        )

        # Smooth overlay (Gaussian filter in log space)
        log_vals = np.log10(np.maximum(counts_float, 0.5))
        smoothed = 10 ** gaussian_filter1d(log_vals, sigma=1.5)
        smoothed_masked = np.where(counts > 0, smoothed, np.nan)
        ax.plot(
            bin_centers, smoothed_masked,
            color=color, label=MODEL_LABELS[mid],
            lw=2.2, alpha=0.9,
        )

        # Endpoint marker
        nonzero = np.where(counts > 0)[0]
        if len(nonzero) > 0:
            last_idx = nonzero[-1]
            ax.plot(
                bin_centers[last_idx], smoothed[last_idx],
                marker="x", color=color,
                markersize=8, markeredgewidth=2, zorder=5,
            )

    ax.set_yscale("log")
    ax.set_xlabel("Median nucleus DAPI intensity", fontsize=11)
    ax.set_ylabel("Number of nuclei (log scale)", fontsize=11)
    ax.set_title(
        f"Nucleus intensity distribution by model — {cat_label}{label_title}",
        fontsize=12,
    )
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    tag = f"intensity_hist_{category}{label_suffix}"
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{tag}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved {tag}.{{pdf,png}}")


def plot_cumulative(df, category, output_dir, label=""):
    """Cumulative distribution of nucleus intensity per model.
    Easier to spot missing dim nuclei: a model that misses dims will have its
    CDF shifted right compared to others."""
    cat_label = CATEGORY_LABELS.get(category, category)
    label_suffix = f"_{label}" if label else ""
    label_title = f" [{label}]" if label else ""

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mid in SINGLE_CHANNEL_MODELS:
        msub = df[df["model_id"] == mid]
        if msub.empty:
            continue
        vals = np.sort(msub["median_intensity"].values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(
            vals, cdf,
            color=MODEL_COLORS[mid], label=MODEL_LABELS[mid],
            lw=2, alpha=0.85,
        )

    ax.set_xlabel("Median nucleus DAPI intensity", fontsize=11)
    ax.set_ylabel("Cumulative fraction of nuclei", fontsize=11)
    ax.set_title(
        f"Cumulative intensity distribution — {cat_label}{label_title}", fontsize=12,
    )
    ax.legend(fontsize=8, loc="lower right", framealpha=0.9)

    tag = f"intensity_cdf_{category}{label_suffix}"
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(output_dir / f"{tag}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  Saved {tag}.{{pdf,png}}")


# ---- Summary -----------------------------------------------------------------

def format_summary(df, category, n_images):
    """Summary statistics per model: count, median intensity, Q10 intensity."""
    cat_label = CATEGORY_LABELS.get(category, category)
    lines = [
        f"Nucleus intensity summary — {cat_label} ({n_images} images)",
        "",
        f"  {'Model':<20s} {'N nuclei':>10s} {'Median int.':>12s} "
        f"{'Q10 int.':>10s} {'Q25 int.':>10s}",
        f"  {'-'*20} {'-'*10} {'-'*12} {'-'*10} {'-'*10}",
    ]

    for mid in SINGLE_CHANNEL_MODELS:
        msub = df[df["model_id"] == mid]
        n = len(msub)
        med = msub["median_intensity"].median()
        q10 = msub["median_intensity"].quantile(0.10)
        q25 = msub["median_intensity"].quantile(0.25)
        lines.append(
            f"  {MODEL_LABELS[mid]:<20s} {n:>10d} {med:>12.0f} "
            f"{q10:>10.0f} {q25:>10.0f}"
        )

    return "\n".join(lines)


# ---- Main --------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--category", type=str, nargs="+",
        default=["06_flatfield_inhomogeneity"],
        help="Category folder name(s) to analyze",
    )
    parser.add_argument(
        "--results-dir", type=Path, nargs="+",
        default=[Path("results"), Path("results_normalized")],
        help="One or more results directories containing masks/",
    )
    parser.add_argument(
        "--label", type=str, nargs="+", default=["raw", "normalized"],
        help="Labels for each results directory (must match --results-dir count)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/images"),
        help="Directory with category subfolders containing DAPI TIFs",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_comparison"),
        help="Output directory for figures and summaries",
    )
    parser.add_argument(
        "--n-bins", type=int, default=50,
        help="Number of histogram bins",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.results_dir) != len(args.label):
        parser.error("--results-dir and --label must have the same number of entries")

    all_summaries = []
    for results_dir, label in zip(args.results_dir, args.label):
        mask_dir = results_dir / "masks"
        print(f"\n{'='*60}")
        print(f"Results: {results_dir} (label: {label})")
        print(f"{'='*60}")

        for category in args.category:
            cat_label = CATEGORY_LABELS.get(category, category)
            print(f"\nAnalyzing {cat_label} ({category})...")

            df = measure_nuclei_intensities(
                args.data_dir, mask_dir, category, SINGLE_CHANNEL_MODELS,
            )
            if df.empty:
                print(f"  No data for {category}, skipping.")
                continue

            n_images = df["image_name"].nunique()
            total = len(df)
            print(f"  {total} nuclei measured across {n_images} images")

            # Summary
            summary_text = format_summary(df, category, n_images)
            summary_text = f"[{label}] {summary_text}"
            print(f"\n{summary_text}")
            all_summaries.append(summary_text)

            # Plots
            plot_intensity_histogram(
                df, category, args.output_dir, args.n_bins, label=label,
            )
            plot_cumulative(df, category, args.output_dir, label=label)

            # Save per-nucleus data
            csv_path = args.output_dir / f"intensity_per_nucleus_{category}_{label}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved {csv_path}")

    # Write all summaries to a single text file
    if all_summaries:
        txt_path = args.output_dir / "intensity_detection_summary.txt"
        txt_path.write_text("\n\n".join(all_summaries) + "\n")
        print(f"\n  Saved {txt_path}")
