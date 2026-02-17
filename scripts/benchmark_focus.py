#!/usr/bin/env python
"""
Benchmark script for focus/blur metrics visualization.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from aggrequant.quality.focus import (
    compute_patch_focus_maps,
    compute_global_focus_metrics,
    power_log_log_slope,
)


# Directories
INPUT_DIR = Path("/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/benchmark_nuclei")
OUTPUT_DIR = INPUT_DIR / "output_focus"

# Focus metrics parameters
#PATCH_SIZE = (80, 80)
#PATCH_SIZE = (136, 136)
PATCH_SIZE = (255, 255)


def load_image(path: Path) -> np.ndarray:
    """Load a TIFF image."""
    return tifffile.imread(str(path))


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range using percentiles."""
    p_low, p_high = np.percentile(image, (1, 99.8))
    if p_high - p_low < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - p_low) / (p_high - p_low), 0, 1).astype(np.float32)


def create_focus_figure(
    image: np.ndarray,
    title: str,
    patch_size: tuple = PATCH_SIZE,
) -> plt.Figure:
    """Create figure showing image and focus metric maps."""
    maps, ys, xs = compute_patch_focus_maps(image, patch_size=patch_size)

    # Compute global metrics
    global_metrics = compute_global_focus_metrics(image)

    n_maps = len(maps)
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(4 * (n_maps + 1), 4))

    # Original image with global metrics
    im = axes[0].imshow(normalize(image), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Focus metric maps
    for k, (name, m) in enumerate(maps.items()):
        im = axes[k + 1].imshow(m, cmap="viridis")
        axes[k + 1].set_title(name)
        axes[k + 1].axis("off")
        fig.colorbar(im, ax=axes[k + 1], fraction=0.046, pad=0.04)

    # Add global metrics to title
    plls = global_metrics["power_log_log_slope"]
    gvol = global_metrics["global_variance_laplacian"]
    hfr = global_metrics["high_freq_ratio"]
    fig.suptitle(
        f"{title}\nPLLS={plls:.2f}  |  GlobalVoL={gvol:.1f}  |  HFRatio={hfr:.2f}",
        fontsize=11,
    )
    fig.tight_layout()

    return fig


def create_summary_figure(results: list) -> plt.Figure:
    """Create summary comparison figure for all images."""
    n = len(results)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    names = [r["name"] for r in results]
    plls_values = [r["plls"] for r in results]
    gvol_values = [r["global_vol"] for r in results]
    hfr_values = [r["hfr"] for r in results]
    mean_vol_values = [r["mean_patch_vol"] for r in results]

    x = np.arange(n)

    # PLLS comparison
    ax = axes[0, 0]
    colors = ["red" if v > -2.0 else "green" for v in plls_values]
    ax.barh(x, plls_values, color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Power Log-Log Slope (more negative = sharper)")
    ax.set_title("PLLS (Global Defocus Detector)")
    ax.axvline(-2.0, color="orange", linestyle="--", label="Threshold suggestion")
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    # Global VoL comparison
    ax = axes[0, 1]
    colors = ["red" if v < 500 else "green" for v in gvol_values]
    ax.barh(x, gvol_values, color=colors, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Global Variance of Laplacian (higher = sharper)")
    ax.set_title("Global VoL")
    ax.invert_yaxis()

    # High Frequency Ratio
    ax = axes[1, 0]
    ax.barh(x, hfr_values, color="steelblue", alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("High/Low Frequency Energy Ratio")
    ax.set_title("High Frequency Ratio")
    ax.invert_yaxis()

    # Mean patch VoL
    ax = axes[1, 1]
    ax.barh(x, mean_vol_values, color="purple", alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Mean Patch Variance of Laplacian")
    ax.set_title("Mean Patch VoL")
    ax.invert_yaxis()

    fig.suptitle("Focus Quality Comparison Across Images", fontsize=14)
    fig.tight_layout()

    return fig


def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please create the directory and add benchmark images.")
        return

    image_files = sorted(INPUT_DIR.glob("*.tif"))
    if not image_files:
        print(f"No .tif files found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nProcessing images...")
    print("=" * 80)

    results = []

    for img_path in image_files:
        image = load_image(img_path)

        # Compute global metrics
        global_metrics = compute_global_focus_metrics(image)

        # Compute patch-based metrics for mean
        maps, _, _ = compute_patch_focus_maps(image, patch_size=PATCH_SIZE)
        mean_patch_vol = maps["VarianceLaplacian"].mean()

        result = {
            "name": img_path.stem[:20],  # Truncate for display
            "plls": global_metrics["power_log_log_slope"],
            "global_vol": global_metrics["global_variance_laplacian"],
            "hfr": global_metrics["high_freq_ratio"],
            "mean_patch_vol": mean_patch_vol,
        }
        results.append(result)

        # Print results
        print(f"  {img_path.name}")
        print(f"    PLLS={result['plls']:.2f}  GlobalVoL={result['global_vol']:.1f}  "
              f"HFRatio={result['hfr']:.2f}  MeanPatchVoL={mean_patch_vol:.1f}")

        # Create and save per-image figure
        fig = create_focus_figure(image, title=img_path.stem)
        output_path = OUTPUT_DIR / f"{img_path.stem}_focus.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("=" * 80)

    # Create and save summary figure
    print("\nCreating summary comparison figure...")
    summary_fig = create_summary_figure(results)
    summary_path = OUTPUT_DIR / "_summary_comparison.png"
    summary_fig.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close(summary_fig)

    # Print ranking by PLLS
    print("\n" + "=" * 80)
    print("RANKING BY PLLS (most negative = sharpest):")
    print("=" * 80)
    sorted_results = sorted(results, key=lambda x: x["plls"])
    for i, r in enumerate(sorted_results, 1):
        status = "SHARP" if r["plls"] < -2.0 else "POTENTIALLY BLURRY"
        print(f"  {i:2d}. {r['name']:25s} PLLS={r['plls']:6.2f}  [{status}]")

    print("=" * 80)
    print(f"Done. Results saved to {OUTPUT_DIR}")
    print(f"Summary figure: {summary_path}")


if __name__ == "__main__":
    main()
