#!/usr/bin/env python
"""
Interactive Streamlit viewer for nuclei segmentation benchmark masks.

Loads DAPI images and predicted masks from all 13 model configurations.
Shows a grid of all images in a selected category with mask overlays.

Usage:
    streamlit run viewer.py
    streamlit run viewer.py -- --data-dir /path/to/curated/images
    streamlit run viewer.py -- --masks-dir /path/to/results/masks
"""

import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import tifffile
from skimage.color import label2rgb
from skimage.segmentation import find_boundaries
from skimage.morphology import dilation, square
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
# Default paths — edit here or override with CLI args after `--`
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent
_DEFAULT_DATA_DIR = _SCRIPT_DIR / "data" / "images"
_DEFAULT_MASKS_DIR = _SCRIPT_DIR / "results" / "masks"


# ---------------------------------------------------------------------------
# Model and category metadata
# ---------------------------------------------------------------------------

MODEL_ORDER = [
    "stardist_2d_fluo",
    "deepcell_nuclear",
    "deepcell_mesmer",
    # "deepcell_mesmer_with_cell",      # two-channel (+cell) — excluded
    "cellpose_nuclei",
    "cellpose_cyto2_no_nuc",
    "cellpose_cyto2_with_nuc",
    # "cellpose_cyto2_with_cell",       # two-channel (+cell) — excluded
    "cellpose_cyto3_no_nuc",
    "cellpose_cyto3_with_nuc",
    # "cellpose_cyto3_with_cell",       # two-channel (+cell) — excluded
    "instanseg_fluorescence",
    # "instanseg_fluorescence_with_cell",  # two-channel (+cell) — excluded
]

MODEL_LABELS = {
    "stardist_2d_fluo":                 "StarDist 2D",
    "deepcell_nuclear":                 "DeepCell Nuclear",
    "deepcell_mesmer":                  "Mesmer",
    "deepcell_mesmer_with_cell":        "Mesmer +cell",
    "cellpose_nuclei":                  "Cellpose nuclei",
    "cellpose_cyto2_no_nuc":            "Cellpose cyto2",
    "cellpose_cyto2_with_nuc":          "Cellpose cyto2 +nuc",
    "cellpose_cyto2_with_cell":         "Cellpose cyto2 +cell",
    "cellpose_cyto3_no_nuc":            "Cellpose cyto3",
    "cellpose_cyto3_with_nuc":          "Cellpose cyto3 +nuc",
    "cellpose_cyto3_with_cell":         "Cellpose cyto3 +cell",
    "instanseg_fluorescence":           "InstanSeg",
    "instanseg_fluorescence_with_cell": "InstanSeg +cell",
}

# Fixed RGB colors per model — used in contour overlay mode
MODEL_COLORS = {
    "stardist_2d_fluo":                 (46,  125,  50),   # green
    "deepcell_nuclear":                 (198,  40,  40),   # dark red
    "deepcell_mesmer":                  (230,  81,   0),   # orange
    "deepcell_mesmer_with_cell":        (255, 167,  38),   # amber
    "cellpose_nuclei":                  ( 21, 101, 192),   # blue
    "cellpose_cyto2_no_nuc":            ( 66, 165, 245),   # light blue
    "cellpose_cyto2_with_nuc":          (100, 181, 246),   # lighter blue
    "cellpose_cyto2_with_cell":         (  1,  87, 155),   # dark blue
    "cellpose_cyto3_no_nuc":            ( 13,  71, 161),   # navy
    "cellpose_cyto3_with_nuc":          ( 63,  81, 181),   # indigo
    "cellpose_cyto3_with_cell":         ( 74,  20, 140),   # deep purple
    "instanseg_fluorescence":           (123,  31, 162),   # purple
    "instanseg_fluorescence_with_cell": (194,  24,  91),   # pink
}

CATEGORY_LABELS = {
    "01_low_confluency":          "Low confluency",
    "02_high_confluency":         "High confluency",
    "03_clustered_touching":      "Clustered / touching",
    "04_mitotic":                 "Mitotic",
    "05_defocused":               "Defocused",
    "06_flatfield_inhomogeneity": "Flat-field inhomogeneity",
    "07_low_intensity":           "Low intensity",
    "08_high_intensity":          "High intensity",
    "09_debris_artifacts":        "Debris / artifacts",
}


# ---------------------------------------------------------------------------
# Path resolution from CLI args
# ---------------------------------------------------------------------------

def _get_paths() -> tuple[Path, Path]:
    """Parse optional --data-dir / --masks-dir from args passed after `--`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-dir",  default=str(_DEFAULT_DATA_DIR))
    parser.add_argument("--masks-dir", default=str(_DEFAULT_MASKS_DIR))
    try:
        idx = sys.argv.index("--")
        args, _ = parser.parse_known_args(sys.argv[idx + 1:])
    except ValueError:
        args, _ = parser.parse_known_args([])
    return Path(args.data_dir), Path(args.masks_dir)


# ---------------------------------------------------------------------------
# Cached data-scanning helpers
# ---------------------------------------------------------------------------

@st.cache_data
def scan_categories(data_dir: str) -> list[tuple[str, str]]:
    """Return sorted list of (folder_name, display_label) tuples."""
    d = Path(data_dir)
    folders = sorted(p.name for p in d.iterdir() if p.is_dir())
    return [(f, CATEGORY_LABELS.get(f, f)) for f in folders]


@st.cache_data
def scan_images(data_dir: str, category: str) -> list[str]:
    """Return sorted DAPI filenames for a given category folder."""
    return sorted(p.name for p in (Path(data_dir) / category).glob("*Blue*.tif"))


@st.cache_data
def scan_models(masks_dir: str) -> list[str]:
    """Return model_ids present in masks_dir, restricted to MODEL_ORDER only."""
    available = {p.name for p in Path(masks_dir).iterdir() if p.is_dir()}
    return [m for m in MODEL_ORDER if m in available]


# ---------------------------------------------------------------------------
# Cached image / mask loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_dapi(data_dir: str, category: str, filename: str, downsample: int) -> np.ndarray:
    """Load DAPI TIF, normalize to uint8, downsample. Returns (H, W, 3) uint8."""
    path = Path(data_dir) / category / filename
    img = tifffile.imread(str(path)).astype(np.float32)
    img = img[::downsample, ::downsample]
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    img = np.clip((img - p1) / max(p99 - p1, 1.0), 0.0, 1.0)
    u8 = (img * 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)  # (H, W, 3) grayscale-as-RGB


@st.cache_data
def load_mask(masks_dir: str, model_id: str, filename: str, downsample: int) -> np.ndarray | None:
    """Load instance mask TIF, downsample (nearest). Returns (H, W) uint16 or None."""
    path = Path(masks_dir) / model_id / filename
    if not path.exists():
        return None
    mask = tifffile.imread(str(path))
    return mask[::downsample, ::downsample]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_filled(
    dapi_rgb: np.ndarray,
    mask: np.ndarray | None,
    alpha: float,
) -> np.ndarray:
    """Filled label map blended over DAPI. Returns (H, W, 3) uint8."""
    if mask is None or mask.max() == 0:
        return dapi_rgb
    dapi_float = dapi_rgb.astype(np.float64) / 255.0
    overlay = label2rgb(mask, image=dapi_float, bg_label=0, alpha=alpha, kind="overlay")
    return (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)


def render_contours(
    dapi_rgb: np.ndarray,
    model_masks: list[tuple[str, np.ndarray | None]],
    thickness: int = 1,
) -> np.ndarray:
    """Draw boundary contours for each (model_id, mask) pair onto DAPI.

    Each model gets its fixed color from MODEL_COLORS. Returns (H, W, 3) uint8.
    """
    canvas = dapi_rgb.copy()
    sq = square(thickness) if thickness > 1 else None
    for model_id, mask in model_masks:
        if mask is None or mask.max() == 0:
            continue
        color = MODEL_COLORS.get(model_id, (255, 255, 255))
        boundaries = find_boundaries(mask, mode="outer")
        if sq is not None:
            boundaries = dilation(boundaries, sq)
        canvas[boundaries] = color
    return canvas


def render_consensus(
    dapi_rgb: np.ndarray,
    model_masks: list[tuple[str, np.ndarray | None]],
) -> np.ndarray:
    """Pixel-wise count of how many models call foreground. Returns (H, W, 3) uint8.

    Jet colormap: blue = 1 model, red = all models agree. Background (0 votes)
    shows as unmodified DAPI.
    """
    h, w = dapi_rgb.shape[:2]
    votes = np.zeros((h, w), dtype=np.float32)
    n_valid = 0
    for _, mask in model_masks:
        if mask is not None and mask.shape == (h, w):
            votes += (mask > 0).astype(np.float32)
            n_valid += 1
    if n_valid == 0:
        return dapi_rgb

    # Map raw vote counts through jet (normalized to 0–1 for colormap lookup)
    votes_norm = votes / n_valid
    heatmap = plt.cm.jet(votes_norm)[:, :, :3]  # (H, W, 3) float, drop alpha

    dapi_float = dapi_rgb.astype(np.float32) / 255.0
    fg_mask = votes > 0
    blended = dapi_float.copy()
    blended[fg_mask] = 0.4 * dapi_float[fg_mask] + 0.6 * heatmap[fg_mask]
    return (np.clip(blended, 0, 1) * 255).astype(np.uint8)


def _consensus_colorbar(n_models: int) -> plt.Figure:
    """Return a slim horizontal matplotlib figure showing the jet colorbar."""
    fig, ax = plt.subplots(figsize=(5, 0.45))
    fig.subplots_adjust(bottom=0.55)
    norm = plt.Normalize(vmin=0, vmax=n_models)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="jet")
    cb = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label("Models agreeing (foreground vote count)", fontsize=8)
    cb.set_ticks(range(n_models + 1))
    cb.set_ticklabels([str(i) for i in range(n_models + 1)], fontsize=7)
    fig.patch.set_alpha(0)
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _model_color_key(model_ids: list[str]) -> None:
    """Render a horizontal row of colored label chips for contour mode."""
    cols = st.columns(min(len(model_ids), 7))
    for i, mid in enumerate(model_ids):
        r, g, b = MODEL_COLORS.get(mid, (180, 180, 180))
        hex_col = f"#{r:02x}{g:02x}{b:02x}"
        cols[i % len(cols)].markdown(
            f'<span style="background:{hex_col};padding:2px 7px;'
            f'border-radius:4px;color:white;font-size:11px;font-weight:600">'
            f'{MODEL_LABELS.get(mid, mid)}</span>',
            unsafe_allow_html=True,
        )
    st.markdown("&nbsp;", unsafe_allow_html=True)


def _short_name(filename: str) -> str:
    """Extract a compact display name from the raw filename."""
    # e.g. "HA11_rep1_P - 13(fld 7 wv 390 - Blue).tif" → "HA11 P-13 fld7"
    stem = Path(filename).stem
    parts = stem.split("(")
    base = parts[0].strip()
    if len(parts) > 1:
        fld_part = parts[1].split("wv")[0].strip().rstrip(")")
        return f"{base} {fld_part}"
    return base


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="Nuclei Benchmark Viewer",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    data_dir, masks_dir = _get_paths()

    # ---- Sidebar ----
    with st.sidebar:
        st.title("🔬 Nuclei Benchmark")

        # Validate paths
        if not data_dir.exists():
            st.error(f"Data dir not found:\n`{data_dir}`")
            st.stop()
        if not masks_dir.exists():
            st.error(f"Masks dir not found:\n`{masks_dir}`")
            st.stop()

        # Category picker
        categories = scan_categories(str(data_dir))
        cat_labels = [lbl for _, lbl in categories]
        cat_idx = st.selectbox(
            "Category",
            range(len(categories)),
            format_func=lambda i: cat_labels[i],
        )
        cat_folder, cat_label = categories[cat_idx]

        available_models = scan_models(str(masks_dir))
        display_names = [MODEL_LABELS.get(m, m) for m in available_models]

        st.markdown("---")

        # Display mode
        display_mode = st.radio(
            "Overlay mode",
            ["Filled — single model", "Consensus heatmap"],
            help=(
                "**Filled**: colorized instance labels with contour outlines for one model.\n\n"
                "**Consensus**: pixel count of how many models agree on foreground."
            ),
        )

        # Mode-specific controls
        alpha = 0.4          # default (used by Filled)
        contour_thickness = 1  # default (used by Contours)
        show_raw_only = False
        selected_models: list[str] = []

        if display_mode.startswith("Filled"):
            model_idx = st.selectbox(
                "Model",
                range(len(available_models)),
                format_func=lambda i: display_names[i],
            )
            selected_models = [available_models[model_idx]]
            alpha = st.slider("Fill opacity", 0.1, 0.9, 0.4, 0.05)
            show_raw_only = st.checkbox("Raw image only", value=False)

        else:  # Consensus
            sel_indices = st.multiselect(
                "Models to include",
                range(len(available_models)),
                default=list(range(len(available_models))),
                format_func=lambda i: display_names[i],
            )
            selected_models = [available_models[i] for i in sel_indices]

        st.markdown("---")

        downsample = st.select_slider(
            "Downsample factor",
            options=[1, 2, 4, 8],
            value=4,
            help="Reduces image size before display. 4× → 510 px. Faster loading.",
        )
        n_cols = 5

        st.markdown("---")
        st.caption(f"Data: `{data_dir.name}`")
        st.caption(f"Masks: `{masks_dir}`")

    # ---- Main area ----
    st.header(cat_label)
    images = scan_images(str(data_dir), cat_folder)

    if not images:
        st.warning("No DAPI images found in this category folder.")
        return

    n_images = len(images)
    st.caption(f"{n_images} images · downsample {downsample}× · {len(available_models)} model dirs found")

    # Colorbar for consensus mode
    if display_mode.startswith("Consensus") and selected_models:
        fig_cb = _consensus_colorbar(len(selected_models))
        st.pyplot(fig_cb, use_container_width=False)
        plt.close(fig_cb)

    # ---- Image grid ----
    for row_start in range(0, n_images, n_cols):
        row_imgs = images[row_start: row_start + n_cols]
        cols = st.columns(n_cols)
        for col, fname in zip(cols, row_imgs):
            dapi = load_dapi(str(data_dir), cat_folder, fname, downsample)

            if display_mode.startswith("Filled"):
                if show_raw_only:
                    rendered = dapi
                    no_mask = False
                else:
                    mid = selected_models[0]
                    mask = load_mask(str(masks_dir), mid, fname, downsample)
                    rendered = render_filled(dapi, mask, alpha)
                    no_mask = mask is None

            else:  # Consensus
                model_masks = [
                    (mid, load_mask(str(masks_dir), mid, fname, downsample))
                    for mid in selected_models
                ]
                rendered = render_consensus(dapi, model_masks)
                no_mask = not any(m is not None for _, m in model_masks)

            caption = _short_name(fname) + (" ⚠️" if no_mask else "")
            col.image(rendered, caption=caption, use_container_width=True)

    # ---- Single image view ----
    st.markdown("---")
    st.subheader("Single image view")

    sel_fname = st.selectbox(
        "Select image",
        images,
        format_func=_short_name,
    )

    if sel_fname:
        dapi_large = load_dapi(str(data_dir), cat_folder, sel_fname, max(1, downsample // 2))

        fig_cons = None

        if display_mode.startswith("Filled"):
            if show_raw_only:
                img_out = dapi_large
                caption = "Raw DAPI"
            else:
                mid = selected_models[0]
                mask = load_mask(str(masks_dir), mid, sel_fname, max(1, downsample // 2))
                img_out = render_filled(dapi_large, mask, alpha)
                if mask is not None and mask.max() > 0:
                    boundaries = find_boundaries(mask, mode="outer")
                    img_out = img_out.copy()
                    img_out[boundaries] = (255, 255, 255)
                caption = MODEL_LABELS.get(mid, mid)

        else:  # Consensus
            model_masks = [
                (mid, load_mask(str(masks_dir), mid, sel_fname, max(1, downsample // 2)))
                for mid in selected_models
            ]
            img_out = render_consensus(dapi_large, model_masks)
            caption = f"Consensus — {len(selected_models)} models"

            # Per-pixel model-contribution data for hover tooltip
            h, w = dapi_large.shape[:2]
            cd_channels = []
            for _mid, _mask in model_masks:
                ch = ((_mask > 0).astype(np.uint8)
                      if (_mask is not None and _mask.shape == (h, w))
                      else np.zeros((h, w), dtype=np.uint8))
                cd_channels.append(ch)
            vote_counts = (np.sum(cd_channels, axis=0).astype(np.uint8)
                           if cd_channels else np.zeros((h, w), dtype=np.uint8))
            cd_channels.append(vote_counts)
            customdata = np.stack(cd_channels, axis=-1)  # (H, W, N_sel+1)

            n_sel = len(selected_models)
            hover_lines = [f"<b>Votes: %{{customdata[{n_sel}]}}</b>"]
            for i, _mid in enumerate(selected_models):
                lbl = MODEL_LABELS.get(_mid, _mid)
                hover_lines.append(f"{lbl}: %{{customdata[{i}]}}")
            hovertemplate = "<br>".join(hover_lines) + "<extra></extra>"

            fig_cons = go.Figure(go.Image(
                z=img_out,
                customdata=customdata,
                hovertemplate=hovertemplate,
            ))
            fig_cons.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            )

        zoom = st.slider("Image width (%)", 20, 100, 60, step=5, key="single_zoom")
        col_img, _ = st.columns([zoom, 100 - zoom])
        if fig_cons is not None:
            with col_img:
                st.caption(caption)
                st.plotly_chart(fig_cons, use_container_width=True)
        else:
            col_img.image(img_out, caption=caption, use_container_width=True)


if __name__ == "__main__":
    main()
