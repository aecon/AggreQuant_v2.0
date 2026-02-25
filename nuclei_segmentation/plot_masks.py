#!/usr/bin/env python
"""
Nuclei segmentation mask gallery — Part A (raw inference, no normalisation).

Grid layout  (9 rows × 10 columns):
  col  0   = DAPI intensity distribution (vertical histogram, raw uint16, 0–65535)
  col  1   = raw DAPI (grayscale, 1–99.8th percentile stretch)
  cols 2–8 = predicted nuclei masks for the 7 single-channel models
             (uniform fill colour + white instance-boundary contours)
  col  9   = consensus heatmap (jet, 0–7 model votes; black = background)
             + horizontal colourbar above the column header

Each panel is a 1024×1024 centre crop of the original 2040×2040 image.
Images per category are curated (hardcoded in SELECTED_IMAGES).

Output: <output-dir>/panel_F_masks_partA.{pdf,png}

Usage:
    conda activate nuclei-bench
    python plot_masks.py \\
        --data-dir "/media/athena/SpeedDrive/.../NUCLEI-BENCHMARK_AE-CURATED-2026-02-19" \\
        [--masks-dir results/masks] \\
        [--output-dir results/figures] \\
        [--crop-size 512] \\
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

SELECTED_IMAGES = {
    "01_low_confluency":          "HA1_rep1_G - 13(fld 6 wv 390 - Blue).tif",
    "02_high_confluency":         "HA23_rep1_P - 13(fld 7 wv 390 - Blue).tif",
    "03_clustered_touching":      "HA6_rep1_P - 13(fld 9 wv 390 - Blue).tif",
    "04_mitotic":                 "HA28_rep1_H - 05(fld 3 wv 390 - Blue).tif",
    "05_defocused":               "HA24_rep1_E - 05(fld 1 wv 390 - Blue).tif",
    "06_flatfield_inhomogeneity": "HA43_rep1_N - 05(fld 4 wv 390 - Blue).tif",
    "08_high_intensity":          "HA30_rep1_K - 05(fld 5 wv 390 - Blue).tif",
    "09_debris_artifacts":        "HA11_rep1_I - 12(fld 5 wv 390 - Blue).tif",
}

CATEGORY_ORDER = list(SELECTED_IMAGES.keys())

CATEGORY_LABELS = {
    "01_low_confluency":          "Low confluency",
    "02_high_confluency":         "High confluency",
    "03_clustered_touching":      "Clustered",
    "04_mitotic":                 "Mitotic",
    "05_defocused":               "Defocused",
    "06_flatfield_inhomogeneity": "Flat-field",
    "08_high_intensity":          "High intensity",
    "09_debris_artifacts":        "Debris",
}


# ---------------------------------------------------------------------------
# Model configuration
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
    "stardist_2d_fluo":       "StarDist 2D",
    "cellpose_nuclei":        "Cellpose nuclei",
    "cellpose_cyto2_no_nuc":  "Cellpose cyto2",
    "cellpose_cyto3_no_nuc":  "Cellpose cyto3",
    "deepcell_nuclear":       "DeepCell Nuclear",
    "deepcell_mesmer":        "Mesmer",
    "instanseg_fluorescence": "InstanSeg",
}

N_MODELS = len(MODELS)

# Uniform fill colour for all nucleus instances (R, G, B) in [0, 1]
FILL_COLOR = (0.18, 0.53, 0.72)   # steel blue

# Column indices
COL_HIST       = 0
COL_RAW        = 1
COL_MASK_START = 2                      # cols 2–8  (7 models)
COL_CONS       = COL_MASK_START + N_MODELS  # col 9

# Fixed uint16 intensity axis limits for all histograms
UINT16_MAX = 65535


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def centre_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Return a centre crop of a 2-D array."""
    h, w = arr.shape
    r0 = (h - size) // 2
    c0 = (w - size) // 2
    return arr[r0:r0 + size, c0:c0 + size]


def load_dapi(path: Path, crop_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Load uint16 DAPI TIFF and centre-crop.

    Returns
    -------
    raw_u16  : (H, W) uint16 — raw pixel values (used for histogram)
    norm_f32 : (H, W) float32 in [0, 1] — 1–99.8th-percentile stretch (for display)
    """
    raw = tifffile.imread(str(path))
    raw = centre_crop(raw, crop_size)
    lo, hi = np.percentile(raw, [1.0, 99.8])
    norm = np.clip((raw.astype(np.float32) - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    return raw, norm


def load_mask(path: Path, crop_size: int) -> np.ndarray | None:
    """Load uint16 instance-label TIFF and centre-crop. Returns None if missing."""
    if not path.exists():
        return None
    return centre_crop(tifffile.imread(str(path)), crop_size)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_mask(mask: np.ndarray) -> np.ndarray:
    """Single-colour fill + white instance-boundary contours.

    Returns (H, W, 3) float32 on a black background.
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    rgb[mask > 0] = FILL_COLOR
    bounds = find_boundaries(mask, mode="inner", background=0)
    rgb[bounds] = (1.0, 1.0, 1.0)
    return rgb


def render_consensus(masks: list[np.ndarray | None]) -> np.ndarray:
    """Pixel-wise sum of binary masks mapped through the jet colormap.

    Vote counts are normalised to [0, 1] so that N_MODELS/N_MODELS = 1.0
    maps to the red end of jet.  Background pixels (0 votes) are black.

    Returns (H, W, 3) float32.
    """
    valid = [m for m in masks if m is not None]
    if not valid:
        return np.zeros((512, 512, 3), dtype=np.float32)

    h, w = valid[0].shape
    votes = sum((m > 0).astype(np.float32) for m in valid)

    rgb = mpl.colormaps["jet"](votes / N_MODELS)[:, :, :3].astype(np.float32)
    rgb[votes == 0] = 0.0   # black background
    return rgb


# ---------------------------------------------------------------------------
# Axes helpers
# ---------------------------------------------------------------------------

def _clean_ax(ax: plt.Axes) -> None:
    """Remove all ticks and spines."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def compute_hist_max(data_dir: Path, crop_size: int) -> int:
    """Return the 95th-percentile intensity across all selected cropped images.

    Used as the shared y-axis maximum for all histograms so that the range
    is meaningful (not wasted on the empty high end of the uint16 scale).
    """
    p99_values = []
    for cat, fname in SELECTED_IMAGES.items():
        raw = tifffile.imread(str(data_dir / cat / fname))
        raw = centre_crop(raw, crop_size)
        p99_values.append(np.percentile(raw, 95))
    return int(np.max(p99_values))


def _draw_intensity_hist(ax: plt.Axes, raw_u16: np.ndarray,
                         hist_max: int) -> None:
    """Vertical intensity histogram with y-axis clipped to hist_max.

    Intensity (uint16) on the y-axis; pixel count on the x-axis.
    All rows share the same y-axis limits (0 to hist_max) so
    distributions are directly comparable across categories.
    """
    counts, edges = np.histogram(raw_u16.ravel(), bins=64,
                                 range=(0, hist_max + 1))
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    bin_h = (edges[1] - edges[0]) * 0.95
    ax.barh(bin_centers, counts, height=bin_h, color="#909090", edgecolor="none")
    ax.set_xlim(left=0)
    ax.invert_xaxis()           # baseline (count=0) on the right, next to the image
    ax.set_ylim(0, hist_max)
    ax.set_facecolor("white")
    _clean_ax(ax)


def _add_consensus_colorbar(fig: plt.Figure, ax_top: plt.Axes) -> None:
    """Add a thin horizontal jet colourbar above the consensus column.

    Positioned in the top-margin area, directly above ax_top (the first-row
    consensus subplot).  Tick marks show integer model counts 0–N_MODELS.
    """
    pos = ax_top.get_position()   # Bbox in figure coordinates

    cbar_h   = 0.012   # colourbar height in figure-coordinate units
    cbar_gap = 0.004   # gap between colourbar and top of the subplot

    cbar_ax = fig.add_axes([
        pos.x0,
        pos.y1 + cbar_gap,
        pos.width,
        cbar_h,
    ])

    norm = mpl.colors.Normalize(vmin=0, vmax=N_MODELS)
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap="jet")
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_ticks(range(N_MODELS + 1))
    cbar.set_ticklabels([str(i) for i in range(N_MODELS + 1)])
    cbar.ax.tick_params(labelsize=5, length=2, pad=1)
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.xaxis.set_label_position("top")
    cbar.set_label("# models", fontsize=6, labelpad=2)


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    data_dir: Path,
    masks_dir: Path,
    crop_size: int,
) -> plt.Figure:
    hist_max = compute_hist_max(data_dir, crop_size)
    print(f"  Histogram y-axis max (global 99th percentile): {hist_max}")

    n_rows = len(CATEGORY_ORDER)
    n_cols = 1 + 1 + N_MODELS + 1   # hist | raw | 7 models | consensus = 10

    # Histogram column is narrower than the image columns (ratio 1.5 vs 4)
    width_ratios = [1.5, 4] + [4] * N_MODELS + [4]

    # Figure geometry (inches)
    cell_inch    = 1.4
    hist_inch    = cell_inch * 1.5 / 4.0   # 0.525"
    left_margin  = 1.25
    top_margin   = 0.65   # slightly taller to fit colourbar above consensus
    right_margin = 0.05
    bot_margin   = 0.05

    subplots_width = (N_MODELS + 2) * cell_inch + hist_inch
    fig_w = left_margin + subplots_width + right_margin
    fig_h = top_margin  + n_rows * cell_inch + bot_margin

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": width_ratios},
    )

    fig.subplots_adjust(
        left   = left_margin  / fig_w,
        right  = 1.0 - right_margin / fig_w,
        bottom = bot_margin   / fig_h,
        top    = 1.0 - top_margin   / fig_h,
        wspace = 0.01,
        hspace = 0.01,
    )

    # --- Column headers ---
    col_titles = (
        ["Intensity", "Raw DAPI"]
        + [MODEL_LABELS[m] for m in MODELS]
        + ["Consensus"]          # colourbar added separately below
    )
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=7.5, fontweight="bold", pad=4)

    # --- Row by row ---
    for row_idx, cat in enumerate(CATEGORY_ORDER):
        fname     = SELECTED_IMAGES[cat]
        dapi_path = data_dir / cat / fname
        print(f"  {CATEGORY_LABELS[cat]:<20s}  {fname}")

        raw_u16, norm_f32 = load_dapi(dapi_path, crop_size)

        # Col 0 — intensity histogram; category label as ylabel
        ax = axes[row_idx, COL_HIST]
        _draw_intensity_hist(ax, raw_u16, hist_max)
        ax.set_ylabel(
            CATEGORY_LABELS[cat],
            fontsize=8, fontweight="bold",
            rotation=0, ha="right", va="center", labelpad=6,
        )

        # Col 1 — raw DAPI
        ax = axes[row_idx, COL_RAW]
        ax.imshow(norm_f32, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        _clean_ax(ax)

        # Cols 2–8 — model masks; collect for consensus
        all_masks: list[np.ndarray | None] = []
        for offset, model_id in enumerate(MODELS):
            col_idx   = COL_MASK_START + offset
            mask_path = masks_dir / model_id / fname
            mask      = load_mask(mask_path, crop_size)
            all_masks.append(mask)

            ax = axes[row_idx, col_idx]
            if mask is not None:
                ax.imshow(render_mask(mask), interpolation="nearest")
            else:
                ax.set_facecolor("#111111")
                ax.text(0.5, 0.5, "missing", color="white",
                        ha="center", va="center",
                        transform=ax.transAxes, fontsize=6)
                print(f"    WARNING: mask not found: {mask_path}")
            _clean_ax(ax)

        # Col 9 — consensus heatmap
        ax = axes[row_idx, COL_CONS]
        ax.imshow(render_consensus(all_masks), interpolation="nearest")
        _clean_ax(ax)

    # Colourbar above the consensus column (after layout is set)
    _add_consensus_colorbar(fig, axes[0, COL_CONS])

    return fig


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
        "--crop-size", type=int, default=1024,
        help="Side length of the centre crop in pixels, subsampled by 2 for display (default: 1024)",
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

    print("Building mask gallery (Part A):")
    fig = build_figure(data_dir, masks_dir, args.crop_size)

    stem = "panel_F_masks_partA"
    for ext in ("pdf", "png"):
        out = output_dir / f"{stem}.{ext}"
        fig.savefig(str(out), dpi=args.dpi, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
