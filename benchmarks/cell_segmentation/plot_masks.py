#!/usr/bin/env python
"""
Cell segmentation mask gallery — Part A (raw inference, no normalisation).

Grid layout (7 rows × 7 columns):
  col  0–6 = one difficulty category each (curated FarRed image)

  row  0   = FarRed intensity distribution (bar histogram, raw uint16)
  row  1   = raw FarRed (grayscale, 1–99.8th percentile stretch)
  rows 2–5 = predicted cell masks for the 4 single-channel models
             (uniform fill colour + white instance-boundary contours)
  row  6   = consensus heatmap (jet, 0–4 model votes; black = background)
             + vertical colourbar to the right

Each panel is a 1024×1024 centre crop of the original image.
Images per category are curated (hardcoded in SELECTED_IMAGES).

Output: <output-dir>/panel_F_masks_partA.{pdf,png}

Usage:
    conda activate nuclei-bench
    python plot_masks.py \\
        --data-dir "data/images" \\
        [--masks-dir results/masks] \\
        [--output-dir results/figures] \\
        [--crop-size 1024] \\
        [--dpi 300]
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tifffile
from skimage.segmentation import find_boundaries


# ---------------------------------------------------------------------------
# Curated image selection (one per category)
# ---------------------------------------------------------------------------

# Filenames use the FarRed channel (wv 631 - FarRed).
# These are the same FOVs used in the nuclei benchmark, just the cell channel.
SELECTED_IMAGES = {
    "low_confluency": ("01_low_confluency",         "HA1_rep1_G - 13(fld 6 wv 631 - FarRed).tif",   "Low confluency"),
    "clustered":      ("03_clustered_touching",      "HA6_rep1_P - 13(fld 9 wv 631 - FarRed).tif",   "Clustered"),
    "mitotic":        ("04_mitotic",                 "HA28_rep1_H - 05(fld 3 wv 631 - FarRed).tif",  "Mitotic"),
    "debris":         ("09_debris_artifacts",        "HA1_rep1_N - 24(fld 5 wv 631 - FarRed).tif",   "Debris"),
    "flatfield":      ("06_flatfield_inhomogeneity", "HA43_rep1_N - 05(fld 4 wv 631 - FarRed).tif",  "Flat-field"),
    "high_intensity": ("08_high_intensity",          "HA30_rep1_K - 05(fld 5 wv 631 - FarRed).tif",  "High intensity"),
    "defocused":      ("09_debris_artifacts",        "HA16_rep1_E - 13(fld 7 wv 631 - FarRed).tif",  "Defocused"),
}

CATEGORY_ORDER = list(SELECTED_IMAGES.keys())


# ---------------------------------------------------------------------------
# Model configuration (single-channel only for gallery)
# ---------------------------------------------------------------------------

MODELS = [
    "cellpose_cyto2",
    "cellpose_cyto3",
    "deepcell_mesmer",
    "instanseg_fluorescence",
]

MODEL_LABELS = {
    "cellpose_cyto2":        "Cellpose cyto2",
    "cellpose_cyto3":        "Cellpose cyto3",
    "deepcell_mesmer":       "Mesmer",
    "instanseg_fluorescence": "InstanSeg",
}

# Fill colour for each model (used for instance colouring)
MODEL_COLORS = {
    "cellpose_cyto2":        "#42A5F5",  # blue
    "cellpose_cyto3":        "#1565C0",  # navy
    "deepcell_mesmer":       "#E65100",  # orange
    "instanseg_fluorescence": "#7B1FA2",  # purple
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def centre_crop(arr, size):
    """Centre-crop a 2D array to (size, size)."""
    h, w = arr.shape[:2]
    r0 = max(0, (h - size) // 2)
    c0 = max(0, (w - size) // 2)
    return arr[r0:r0 + size, c0:c0 + size]


def stretch(img, plo=1, phi=99.8):
    """Percentile stretch to [0, 1]."""
    lo = np.percentile(img, plo)
    hi = np.percentile(img, phi)
    if hi == lo:
        return np.zeros_like(img, dtype=float)
    return np.clip((img.astype(float) - lo) / (hi - lo), 0, 1)


def colorise_labels(labels):
    """Map integer label mask to an RGBA image with distinct hue per instance."""
    h, w = labels.shape
    rgba = np.zeros((h, w, 4), dtype=float)
    ids = np.unique(labels)
    ids = ids[ids > 0]
    cmap = mpl.colormaps["tab20b"]
    for i, lid in enumerate(ids):
        mask = labels == lid
        color = cmap(i % 20)
        rgba[mask] = color
    return rgba


def overlay_with_boundaries(ax, img_grey, labels, fill_color):
    """Show grey image with colourised fill and white boundary contours."""
    ax.imshow(img_grey, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if labels.max() > 0:
        rgba = colorise_labels(labels)
        rgba[..., 3] = np.where(labels > 0, 0.45, 0.0)
        ax.imshow(rgba, interpolation="nearest")
        boundaries = find_boundaries(labels, mode="inner")
        boundary_rgba = np.zeros((*labels.shape, 4))
        boundary_rgba[boundaries] = [1, 1, 1, 0.9]
        ax.imshow(boundary_rgba, interpolation="nearest")
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cell segmentation mask gallery (Panel F)",
    )
    parser.add_argument(
        "--data-dir", default="data/images",
        help="Path to image directory (default: data/images)",
    )
    parser.add_argument(
        "--masks-dir", default=None,
        help="Directory containing masks/ subfolders (default: <script_dir>/results/masks)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <script_dir>/results/figures)",
    )
    parser.add_argument(
        "--crop-size", type=int, default=1024,
        help="Centre-crop size in pixels (default: 1024)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="Output DPI (default: 300)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    data_dir = Path(args.data_dir)
    masks_dir = Path(args.masks_dir) if args.masks_dir else script_dir / "results" / "masks"
    output_dir = Path(args.output_dir) if args.output_dir else script_dir / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    crop = args.crop_size
    n_cats = len(CATEGORY_ORDER)
    n_models = len(MODELS)
    n_rows = 2 + n_models + 1  # histogram + raw + models + consensus
    n_cols = n_cats

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8),
        gridspec_kw={"hspace": 0.05, "wspace": 0.05},
    )

    row_labels = (
        ["Histogram", "FarRed (raw)"]
        + [MODEL_LABELS[m] for m in MODELS]
        + ["Consensus"]
    )

    for col_idx, cat_key in enumerate(CATEGORY_ORDER):
        cat_folder, fname, cat_label = SELECTED_IMAGES[cat_key]
        img_path = data_dir / cat_folder / fname

        # --- Load and crop image ---
        if img_path.exists():
            img_raw = tifffile.imread(str(img_path))
        else:
            print(f"  WARNING: image not found: {img_path}")
            img_raw = np.zeros((crop, crop), dtype=np.uint16)

        img_crop = centre_crop(img_raw, crop)
        img_grey = stretch(img_crop)

        # --- Row 0: histogram ---
        ax = axes[0, col_idx]
        flat = img_crop.ravel()
        lo, hi = np.percentile(flat, 0.1), np.percentile(flat, 99.9)
        bins = np.linspace(lo, hi, 64)
        counts, edges = np.histogram(flat, bins=bins)
        ax.bar(edges[:-1], counts, width=np.diff(edges), color="#78909C", linewidth=0)
        ax.set_xlim(lo, hi)
        ax.set_ylim(bottom=0)
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("Histogram", fontsize=7, rotation=0, ha="right", va="center")
        ax.set_title(cat_label, fontsize=8, fontweight="bold", pad=4)

        # --- Row 1: raw FarRed ---
        ax = axes[1, col_idx]
        ax.imshow(img_grey, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("FarRed (raw)", fontsize=7, rotation=0, ha="right", va="center")

        # --- Rows 2 to 2+n_models: model masks ---
        all_binary = []
        for model_row, model_id in enumerate(MODELS):
            ax = axes[2 + model_row, col_idx]
            mask_path = masks_dir / model_id / fname
            if mask_path.exists():
                labels = tifffile.imread(str(mask_path)).astype(np.int32)
                labels_crop = centre_crop(labels, crop)
            else:
                labels_crop = np.zeros((crop, crop), dtype=np.int32)
                if col_idx == 0:
                    print(f"  WARNING: mask not found: {mask_path}")

            overlay_with_boundaries(ax, img_grey, labels_crop, MODEL_COLORS[model_id])
            n_cells = labels_crop.max()
            ax.text(0.02, 0.97, str(n_cells), transform=ax.transAxes,
                    color="white", fontsize=7, va="top", ha="left",
                    bbox=dict(facecolor="black", alpha=0.4, pad=1, boxstyle="round,pad=0.2"))
            if col_idx == 0:
                ax.set_ylabel(MODEL_LABELS[model_id], fontsize=7, rotation=0,
                              ha="right", va="center")

            all_binary.append((labels_crop > 0).astype(np.uint8))

        # --- Last row: consensus heatmap ---
        ax = axes[n_rows - 1, col_idx]
        if all_binary:
            votes = np.sum(all_binary, axis=0).astype(float)
            # Black background where all models agree on background (votes==0)
            display = np.zeros((*votes.shape, 3))
            fg_mask = votes > 0
            if fg_mask.any():
                cmap_jet = mpl.colormaps["jet"]
                normed = votes / n_models
                colored = cmap_jet(normed)
                display[fg_mask] = colored[fg_mask, :3]
            # Blend over grey image where background
            bg_grey = img_grey[..., np.newaxis] * np.array([0.3, 0.3, 0.3])
            final = np.where(fg_mask[..., np.newaxis], display, bg_grey)
            ax.imshow(final, interpolation="nearest")
        ax.axis("off")
        if col_idx == 0:
            ax.set_ylabel("Consensus", fontsize=7, rotation=0, ha="right", va="center")

    # Colourbar for consensus row (rightmost column)
    cbar_ax = fig.add_axes([0.92, 0.02, 0.012, 0.12])
    sm = mpl.cm.ScalarMappable(
        cmap="jet", norm=mpl.colors.Normalize(vmin=0, vmax=n_models)
    )
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("# models", fontsize=6)
    cb.set_ticks([0, n_models // 2, n_models])
    cb.ax.tick_params(labelsize=5)

    fig.suptitle(
        "F   Cell segmentation masks — single-channel models (FarRed)",
        fontsize=10, fontweight="bold", x=0.01, ha="left",
    )

    for ext in ("pdf", "png"):
        out = output_dir / f"panel_F_masks_partA.{ext}"
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
