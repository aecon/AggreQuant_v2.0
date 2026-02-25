#!/usr/bin/env python
"""
Nuclei segmentation mask gallery — Part A (raw inference, no normalisation).

Grid layout:
  rows  = 9 difficulty categories  (one randomly selected image per category)
  col 0 = raw DAPI (grayscale, 1st–99.8th percentile stretch)
  cols 1–7 = predicted nuclei masks for the 7 single-channel models
             (uniform fill colour + white instance-boundary contours)

Each panel is a 512×512 centre crop of the original 2040×2040 image.

Output: <output-dir>/panel_F_masks_partA.{pdf,png}

Usage:
    conda activate nuclei-bench
    python plot_masks.py \\
        --data-dir "/media/athena/SpeedDrive/.../NUCLEI-BENCHMARK_AE-CURATED-2026-02-19" \\
        [--masks-dir results/masks] \\
        [--output-dir results/figures] \\
        [--seed 42] \\
        [--crop-size 512] \\
        [--dpi 300]
"""

import argparse
import random
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS = [
    "stardist_2d_fluo",
    "cellpose_nuclei",
    "cellpose_cyto2_no_nuc",
    "cellpose_cyto3_no_nuc",
    "deepcell_nuclear",
    "deepcell_mesmer",
    "instanseg_fluorescence",
]

MODEL_LABELS = {
    "stardist_2d_fluo":      "StarDist 2D",
    "cellpose_nuclei":       "Cellpose nuclei",
    "cellpose_cyto2_no_nuc": "Cellpose cyto2",
    "cellpose_cyto3_no_nuc": "Cellpose cyto3",
    "deepcell_nuclear":      "DeepCell Nuclear",
    "deepcell_mesmer":       "Mesmer",
    "instanseg_fluorescence":"InstanSeg",
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

# Uniform fill colour for all nuclei (R, G, B) in [0, 1]
FILL_COLOR = (0.18, 0.53, 0.72)   # steel blue


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Return a centre crop of a 2-D array."""
    h, w = arr.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return arr[r0:r0 + size, c0:c0 + size]


def load_dapi(path: Path, crop_size: int) -> np.ndarray:
    """Load a uint16 DAPI TIFF, centre-crop, return float32 in [0, 1]."""
    img = tifffile.imread(str(path)).astype(np.float32)
    img = centre_crop(img, crop_size)
    lo, hi = np.percentile(img, [1.0, 99.8])
    return np.clip((img - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def load_mask(path: Path, crop_size: int) -> np.ndarray:
    """Load a uint16 instance-label TIFF and centre-crop."""
    return centre_crop(tifffile.imread(str(path)), crop_size)


def render_mask(mask: np.ndarray) -> np.ndarray:
    """Render instance mask as a single-colour fill with white boundaries.

    Returns an (H, W, 3) float32 RGB array on a black background.
    All foreground pixels are painted FILL_COLOR; pixels on instance
    boundaries (find_boundaries, mode='inner') are painted white.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    fg = mask > 0
    rgb[fg] = FILL_COLOR
    bounds = find_boundaries(mask, mode="inner", background=0)
    rgb[bounds] = (1.0, 1.0, 1.0)
    return rgb


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    data_dir: Path,
    masks_dir: Path,
    seed: int,
    crop_size: int,
) -> plt.Figure:
    rng = random.Random(seed)

    n_rows = len(CATEGORY_ORDER)
    n_cols = 1 + len(MODELS)          # raw DAPI + 7 models

    # Figure geometry (inches)
    cell_inch    = 1.4
    left_margin  = 1.25              # room for row labels
    top_margin   = 0.55              # room for column titles
    right_margin = 0.05
    bot_margin   = 0.05

    fig_w = left_margin + n_cols * cell_inch + right_margin
    fig_h = top_margin  + n_rows * cell_inch + bot_margin

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))

    # Minimal spacing between cells
    fig.subplots_adjust(
        left   = left_margin  / fig_w,
        right  = 1.0 - right_margin / fig_w,
        bottom = bot_margin   / fig_h,
        top    = 1.0 - top_margin   / fig_h,
        wspace = 0.01,
        hspace = 0.01,
    )

    # --- Column headers (top row only) ---
    col_titles = ["Raw DAPI"] + [MODEL_LABELS[m] for m in MODELS]
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(
            title, fontsize=7.5, fontweight="bold", pad=4,
        )

    # --- Row by row ---
    print(f"Image selection  (seed={seed}):")
    for row_idx, cat in enumerate(CATEGORY_ORDER):
        cat_dir    = data_dir / cat
        dapi_files = sorted(cat_dir.glob("*wv 390 - Blue*.tif"))
        if not dapi_files:
            raise FileNotFoundError(
                f"No DAPI images found in: {cat_dir}\n"
                f"Expected files matching '*wv 390 - Blue*.tif'."
            )
        chosen = rng.choice(dapi_files)
        print(f"  {CATEGORY_LABELS[cat]:<20s}  {chosen.name}")

        # -- Col 0: raw DAPI --
        dapi_img = load_dapi(chosen, crop_size)
        ax = axes[row_idx, 0]
        ax.imshow(dapi_img, cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_ylabel(
            CATEGORY_LABELS[cat],
            fontsize=8, fontweight="bold",
            rotation=0, ha="right", va="center",
            labelpad=6,
        )
        _clean_ax(ax)

        # -- Cols 1–7: model masks --
        for col_idx, model_id in enumerate(MODELS, start=1):
            mask_path = masks_dir / model_id / chosen.name
            ax = axes[row_idx, col_idx]

            if mask_path.exists():
                mask = load_mask(mask_path, crop_size)
                ax.imshow(render_mask(mask), interpolation="nearest")
            else:
                ax.set_facecolor("#111111")
                ax.text(
                    0.5, 0.5, "missing",
                    color="white", ha="center", va="center",
                    transform=ax.transAxes, fontsize=6,
                )
                print(f"    WARNING: mask not found: {mask_path}")

            _clean_ax(ax)

    return fig


def _clean_ax(ax: plt.Axes) -> None:
    """Remove ticks, tick labels, and spines from an image subplot."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nuclei mask gallery — Part A (raw inference, no normalisation)",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to NUCLEI-BENCHMARK_AE-CURATED-2026-02-19 directory",
    )
    parser.add_argument(
        "--masks-dir", default=None,
        help="Directory containing per-model mask subdirectories "
             "(default: <script_dir>/results/masks)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory for the figure "
             "(default: <script_dir>/results/figures)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for image selection (default: 42)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=512,
        help="Side length of the centre crop in pixels (default: 512)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output figure DPI (default: 300)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir   = Path(args.data_dir)
    masks_dir  = (
        Path(args.masks_dir) if args.masks_dir
        else script_dir / "results" / "masks"
    )
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else script_dir / "results" / "figures"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.size":   8,
        "axes.linewidth": 0.6,
    })

    fig = build_figure(data_dir, masks_dir, args.seed, args.crop_size)

    stem = "panel_F_masks_partA"
    for ext in ("pdf", "png"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
