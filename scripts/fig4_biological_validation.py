"""Figure 4: Biological validation on CRISPR screen controls.

Runs segmentation in 3 passes (nuclei → cells → aggregates) on NT and RAB13
control wells, loading only one model at a time to avoid GPU OOM. Computes
per-cell morphological features and generates publication-ready figures.

Passes:
  1. StarDist nuclei segmentation → save masks → unload
  2. Cellpose cell segmentation (using nuclei masks) → save masks → unload
  3. UNet + filter aggregate segmentation → post-process → save masks → unload
  4. Load all masks, compute per-cell features, generate plots

Outputs (saved to --output-dir):
    masks/<plate>/                 Cached segmentation masks (.tif)
    cell_features_dl.csv           Per-cell features (UNet segmentation)
    cell_features_filter.csv       Per-cell features (filter segmentation)
    field_summary.csv              Per-field summary (cell counts, % positive)
    panel_a_representative.png     Representative FOVs: NT vs RAB13
    panel_b_quantification.png     Box/violin: per-cell features, NT vs RAB13
    panel_c_morphological.png      Radar chart: morphological profiles
    panel_d_dl_vs_filter.png       DL vs filter profiling comparison

Usage:
    conda run -n AggreQuant python scripts/fig4_biological_validation.py
    conda run -n AggreQuant python scripts/fig4_biological_validation.py --plates HA_32_rep_2
    conda run -n AggreQuant python scripts/fig4_biological_validation.py --max-fields 5
"""

import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

from aggrequant.common.image_utils import load_image
from aggrequant.common.logging import get_logger
from aggrequant.common.gpu_utils import configure_tensorflow_memory_growth
from aggrequant.loaders.images import build_field_triplets

logger = get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "whole_plates"
TRAINING_ROOT = PROJECT_ROOT / "training_output"
OUTPUT_ROOT = TRAINING_ROOT / "fig4_biological_validation"

DEFAULT_CHECKPOINT = (
    TRAINING_ROOT / "loss_function" / "dice03_bce07_pw3" / "checkpoints" / "best.pt"
)

# Well layout: 384-well plate, controls in columns 5 and 13
# Rows A-H: col 5 = NT, col 13 = RAB13
# Rows I-P: col 5 = RAB13, col 13 = NT
ROWS_UPPER = set("ABCDEFGH")

NT_WELLS = (
    [f"{r}05" for r in "ABCDEFGH"] + [f"{r}13" for r in "IJKLMNOP"]
)
RAB13_WELLS = (
    [f"{r}13" for r in "ABCDEFGH"] + [f"{r}05" for r in "IJKLMNOP"]
)

CHANNEL_PURPOSES = {
    "nuclei": "390",
    "aggregates": "473",
    "cells": "631",
}


def get_condition(well_id):
    """Return 'NT' or 'RAB13' for a well ID like 'A05'."""
    if well_id in NT_WELLS:
        return "NT"
    elif well_id in RAB13_WELLS:
        return "RAB13"
    return "unknown"


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def free_gpu():
    """Free GPU memory from both PyTorch and TensorFlow."""
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# StarDist subprocess (TensorFlow GPU memory is only freed on process exit)
# ---------------------------------------------------------------------------

def _stardist_worker(tasks, use_gpu):
    """Run StarDist nuclei segmentation on a list of (image_path, output_path) pairs.

    This function is meant to run in a separate process so that TensorFlow
    GPU memory is fully released when the process exits.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")

    if use_gpu:
        configure_tensorflow_memory_growth()

    from aggrequant.segmentation.stardist import StarDistSegmenter
    seg = StarDistSegmenter(verbose=True)

    for i, (img_path, out_path) in enumerate(tasks):
        logger.info(f"  [{i+1}/{len(tasks)}] {out_path.parent.parent.name}/{out_path.stem}")
        nuc_img = load_image(img_path)
        labels = seg.segment(nuc_img)
        tifffile.imwrite(out_path, labels.astype(np.uint16))


def _run_stardist_subprocess(plate_triplets, use_gpu):
    """Run StarDist in a subprocess to isolate TensorFlow GPU memory."""
    import multiprocessing as mp

    # Build task list: (image_path, output_path) pairs
    tasks = []
    for plate_dir, triplets in plate_triplets.items():
        mask_dir = plate_dir / "fig4_masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

        for t in triplets:
            nuc_path = mask_dir / f"{t.well_id}_f{t.field_id}_nuclei.tif"
            if nuc_path.exists():
                logger.info(f"  {plate_dir.name}/{t.well_id}/f{t.field_id} (cached)")
                continue
            tasks.append((t.paths["nuclei"], nuc_path))

    if not tasks:
        logger.info("  All nuclei masks cached, skipping")
        return

    logger.info(f"  {len(tasks)} fields to segment in subprocess")
    ctx = mp.get_context("spawn")  # spawn = clean process, no inherited TF state
    p = ctx.Process(target=_stardist_worker, args=(tasks, use_gpu))
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"StarDist subprocess failed with exit code {p.exitcode}")


# ---------------------------------------------------------------------------
# Three-pass segmentation
# ---------------------------------------------------------------------------

def run_segmentation(plate_dirs, output_dir, checkpoint, use_gpu, max_fields=None):
    """Run segmentation in 3 passes, one model loaded at a time.

    Pass 1: StarDist (nuclei) for all plates/fields → unload
    Pass 2: Cellpose (cells) for all plates/fields → unload
    Pass 3: UNet + filter (aggregates) for all plates/fields → post-process → unload

    Masks are saved as TIF in output_dir/masks/<plate>/
    """
    # Discover triplets for all plates
    plate_triplets = {}
    for plate_dir in plate_dirs:
        triplets = build_field_triplets(plate_dir, CHANNEL_PURPOSES)
        if max_fields:
            triplets = triplets[:max_fields]
        plate_triplets[plate_dir] = triplets
        logger.info(f"Plate {plate_dir.name}: {len(triplets)} fields")

    # -- Pass 1: Nuclei (StarDist) --
    # StarDist uses TensorFlow, which never releases GPU memory within a process.
    # We run it in a subprocess so the OS reclaims all GPU memory when it exits.
    logger.info("\n=== Pass 1/3: Nuclei segmentation (StarDist) ===")
    _run_stardist_subprocess(plate_triplets, use_gpu)
    logger.info("StarDist subprocess finished — GPU memory released")

    # -- Pass 2: Cells (Cellpose) --
    logger.info("\n=== Pass 2/3: Cell segmentation (Cellpose) ===")
    from aggrequant.segmentation.cellpose import CellposeSegmenter
    cell_seg = CellposeSegmenter(gpu=use_gpu, verbose=True)

    for plate_dir, triplets in plate_triplets.items():
        mask_dir = plate_dir / "fig4_masks"

        for i, t in enumerate(triplets):
            cell_path = mask_dir / f"{t.well_id}_f{t.field_id}_cells.tif"
            if cell_path.exists():
                logger.info(f"  [{i+1}/{len(triplets)}] {plate_dir.name}/{t.well_id}/f{t.field_id} (cached)")
                continue

            nuc_path = mask_dir / f"{t.well_id}_f{t.field_id}_nuclei.tif"
            if not nuc_path.exists():
                logger.warning(f"  Skipping {t.well_id}/f{t.field_id}: no nuclei mask")
                continue

            logger.info(f"  [{i+1}/{len(triplets)}] {plate_dir.name}/{t.well_id}/f{t.field_id}")
            cell_img = load_image(t.paths["cells"])
            nuclei_labels = tifffile.imread(nuc_path)
            cell_labels = cell_seg.segment(cell_img, nuclei_labels)

            # Save updated nuclei (unmatched nuclei zeroed out by Cellpose)
            tifffile.imwrite(nuc_path, nuclei_labels.astype(np.uint16))
            tifffile.imwrite(cell_path, cell_labels.astype(np.uint16))

    del cell_seg
    free_gpu()
    logger.info("Cellpose unloaded")

    # -- Pass 3: Aggregates (UNet + filter) + post-processing --
    logger.info("\n=== Pass 3/3: Aggregate segmentation + post-processing ===")
    from aggrequant.segmentation.aggregates.neural_network import NeuralNetworkSegmenter
    from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter
    from aggrequant.segmentation.postprocessing import (
        remove_border_objects,
        filter_aggregates_by_cells,
    )

    unet_seg = NeuralNetworkSegmenter(
        weights_path=checkpoint,
        device="cuda" if use_gpu else "cpu",
        verbose=True,
    )
    filter_seg = FilterBasedSegmenter(verbose=True)

    for plate_dir, triplets in plate_triplets.items():
        mask_dir = plate_dir / "fig4_masks"

        for i, t in enumerate(triplets):
            unet_path = mask_dir / f"{t.well_id}_f{t.field_id}_agg_unet.tif"
            filter_path = mask_dir / f"{t.well_id}_f{t.field_id}_agg_filter.tif"

            if unet_path.exists() and filter_path.exists():
                logger.info(f"  [{i+1}/{len(triplets)}] {plate_dir.name}/{t.well_id}/f{t.field_id} (cached)")
                continue

            nuc_path = mask_dir / f"{t.well_id}_f{t.field_id}_nuclei.tif"
            cell_path = mask_dir / f"{t.well_id}_f{t.field_id}_cells.tif"
            if not nuc_path.exists() or not cell_path.exists():
                logger.warning(f"  Skipping {t.well_id}/f{t.field_id}: missing nuclei/cell masks")
                continue

            logger.info(f"  [{i+1}/{len(triplets)}] {plate_dir.name}/{t.well_id}/f{t.field_id}")
            agg_img = load_image(t.paths["aggregates"])
            nuclei_labels = tifffile.imread(nuc_path)
            cell_labels = tifffile.imread(cell_path)

            # Post-process nuclei/cells (border removal)
            cell_labels, nuclei_labels = remove_border_objects(cell_labels, nuclei_labels)

            # UNet aggregates
            unet_labels = unet_seg.segment(agg_img)
            unet_labels = filter_aggregates_by_cells(unet_labels, cell_labels)
            tifffile.imwrite(unet_path, unet_labels.astype(np.uint32))

            # Filter aggregates
            filter_labels = filter_seg.segment(agg_img)
            filter_labels = filter_aggregates_by_cells(filter_labels, cell_labels)
            tifffile.imwrite(filter_path, filter_labels.astype(np.uint32))

            # Save post-processed nuclei/cells (border objects removed)
            tifffile.imwrite(nuc_path, nuclei_labels.astype(np.uint16))
            tifffile.imwrite(cell_path, cell_labels.astype(np.uint16))

    del unet_seg, filter_seg
    free_gpu()
    logger.info("Aggregate segmenters unloaded")


# ---------------------------------------------------------------------------
# Feature extraction from saved masks
# ---------------------------------------------------------------------------

def load_masks_and_compute_features(plate_dir, method, output_dir):
    """Load saved masks and compute per-cell morphological features.

    Arguments:
        plate_dir: Path to plate image folder
        method: 'unet' or 'filter'
        output_dir: Base output directory (masks in output_dir/masks/<plate>/)

    Returns:
        Tuple of (cell_features, field_summaries)
    """
    plate_name = plate_dir.name
    mask_dir = plate_dir / "fig4_masks"
    if not mask_dir.exists():
        logger.warning(f"Mask dir not found: {mask_dir}")
        return [], []

    triplets = build_field_triplets(plate_dir, CHANNEL_PURPOSES)

    cell_features = []
    field_summaries = []

    for triplet in triplets:
        well_id, field_id = triplet.well_id, triplet.field_id
        condition = get_condition(well_id)

        nuc_path = mask_dir / f"{well_id}_f{field_id}_nuclei.tif"
        cell_path = mask_dir / f"{well_id}_f{field_id}_cells.tif"
        agg_path = mask_dir / f"{well_id}_f{field_id}_agg_{method}.tif"

        if not all(p.exists() for p in [nuc_path, cell_path, agg_path]):
            continue

        nuclei_labels = tifffile.imread(nuc_path)
        cell_labels = tifffile.imread(cell_path)
        agg_labels = tifffile.imread(agg_path)
        agg_img = load_image(triplet.paths["aggregates"])

        cells = compute_cell_features(
            nuclei_labels, cell_labels, agg_labels, agg_img,
        )
        for c in cells:
            c["condition"] = condition
            c["well_id"] = well_id
            c["field"] = field_id
        cell_features.extend(cells)

        n_cells = len(cells)
        n_agg_pos = sum(1 for c in cells if c["agg_count"] > 0)
        pct_pos = (n_agg_pos / n_cells * 100) if n_cells > 0 else 0.0

        field_summaries.append({
            "well_id": well_id,
            "field": field_id,
            "condition": condition,
            "n_cells": n_cells,
            "n_agg_positive": n_agg_pos,
            "pct_positive": pct_pos,
            "n_aggregates": sum(c["agg_count"] for c in cells),
        })

        logger.info(
            f"  {well_id}/f{field_id} ({condition}): "
            f"{n_cells} cells, {n_agg_pos} agg+ ({pct_pos:.1f}%)"
        )

    return cell_features, field_summaries


def compute_cell_features(nuclei_labels, cell_labels, agg_labels, agg_image,
                          min_agg_area=9):
    """Compute per-cell morphological features from segmentation masks.

    Returns list of dicts, one per cell.
    """
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids > 0]

    if len(cell_ids) == 0:
        return []

    # Precompute aggregate regionprops
    agg_props = regionprops(agg_labels, intensity_image=agg_image)

    # Map each aggregate to the cell containing its centroid
    cell_to_aggs = defaultdict(list)
    for prop in agg_props:
        if prop.area < min_agg_area:
            continue
        cy, cx = int(prop.centroid[0]), int(prop.centroid[1])
        if 0 <= cy < cell_labels.shape[0] and 0 <= cx < cell_labels.shape[1]:
            cid = cell_labels[cy, cx]
            if cid > 0:
                cell_to_aggs[cid].append(prop)

    # Precompute nucleus centroids
    nuc_centroids = {}
    for prop in regionprops(nuclei_labels):
        nuc_centroids[prop.label] = np.array(prop.centroid)

    # Cell areas
    cell_props = {p.label: p for p in regionprops(cell_labels)}

    results = []
    for cid in cell_ids:
        cp = cell_props.get(cid)
        if cp is None:
            continue
        cell_area = cp.area

        aggs = cell_to_aggs.get(cid, [])
        n_aggs = len(aggs)

        if n_aggs == 0:
            results.append({
                "cell_id": int(cid),
                "cell_area": cell_area,
                "agg_count": 0,
                "agg_burden": 0.0,
                "mean_agg_area": 0.0,
                "mean_agg_intensity": 0.0,
                "mean_agg_eccentricity": 0.0,
                "mean_agg_solidity": 0.0,
                "mean_agg_nucleus_dist": 0.0,
            })
            continue

        areas = [a.area for a in aggs]
        intensities = [a.mean_intensity for a in aggs]
        eccentricities = [a.eccentricity for a in aggs]
        solidities = [a.solidity for a in aggs]

        # Distance to nucleus centroid
        nuc_c = nuc_centroids.get(cid)
        if nuc_c is not None:
            dists = [
                np.sqrt(
                    (a.centroid[0] - nuc_c[0]) ** 2
                    + (a.centroid[1] - nuc_c[1]) ** 2
                )
                for a in aggs
            ]
        else:
            dists = [0.0] * n_aggs

        results.append({
            "cell_id": int(cid),
            "cell_area": cell_area,
            "agg_count": n_aggs,
            "agg_burden": sum(areas) / cell_area,
            "mean_agg_area": float(np.mean(areas)),
            "mean_agg_intensity": float(np.mean(intensities)),
            "mean_agg_eccentricity": float(np.mean(eccentricities)),
            "mean_agg_solidity": float(np.mean(solidities)),
            "mean_agg_nucleus_dist": float(np.mean(dists)),
        })

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_panel_a(plate_dir, output_dir):
    """Panel A: Representative FOVs from NT and RAB13.

    Composite overlays (nuclei=blue, cells=dim red, aggregates=green)
    for 3 NT and 3 RAB13 fields.
    """
    import matplotlib.pyplot as plt

    mask_dir = plate_dir / "fig4_masks"
    triplets = build_field_triplets(plate_dir, CHANNEL_PURPOSES)

    # Group by condition
    by_condition = {"NT": [], "RAB13": []}
    for t in triplets:
        cond = get_condition(t.well_id)
        mask_path = mask_dir / f"{t.well_id}_f{t.field_id}_agg_unet.tif"
        if mask_path.exists() and cond in by_condition:
            by_condition[cond].append(t)

    def norm(img):
        p1, p99 = np.percentile(img, (1, 99))
        return np.clip((img.astype(np.float32) - p1) / max(p99 - p1, 1), 0, 1)

    def pick_spread(lst, n=3):
        if len(lst) <= n:
            return lst
        indices = np.linspace(0, len(lst) - 1, n, dtype=int)
        return [lst[i] for i in indices]

    n_cols = 3
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))

    for row_idx, (cond, label) in enumerate([("NT", "NT"), ("RAB13", "RAB13")]):
        picks = pick_spread(by_condition[cond], n_cols)
        for col_idx in range(n_cols):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(picks):
                ax.axis("off")
                continue

            t = picks[col_idx]
            nuc_img = load_image(t.paths["nuclei"])
            cell_img = load_image(t.paths["cells"])
            agg_img = load_image(t.paths["aggregates"])

            composite = np.zeros((*nuc_img.shape, 3), dtype=np.float32)
            composite[:, :, 2] = norm(nuc_img) * 0.5   # blue = nuclei
            composite[:, :, 1] = norm(agg_img)          # green = aggregates
            composite[:, :, 0] = norm(cell_img) * 0.3   # red = cells (dim)

            ax.imshow(composite)
            ax.set_title(f"{label} — {t.well_id}/f{t.field_id}", fontsize=10)
            ax.axis("off")

    fig.suptitle(f"Representative FOVs — {plate_dir.name}", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "panel_a_representative.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_panel_b(all_cell_features, output_dir):
    """Panel B: Box/violin plots of per-cell features, NT vs RAB13, per plate."""
    import matplotlib.pyplot as plt

    features_to_plot = [
        ("agg_count", "Aggregates per cell"),
        ("agg_burden", "Aggregate burden\n(fraction of cell area)"),
        ("mean_agg_area", "Mean aggregate area\n(pixels)"),
        ("mean_agg_intensity", "Mean aggregate\nintensity"),
    ]

    plates = sorted(all_cell_features.keys())
    n_plates = len(plates)
    n_features = len(features_to_plot)

    fig, axes = plt.subplots(n_features, n_plates, figsize=(6 * n_plates, 4 * n_features))
    if n_plates == 1:
        axes = axes[:, np.newaxis]

    for col, plate in enumerate(plates):
        data = all_cell_features[plate]
        nt_cells = [c for c in data if c["condition"] == "NT"]
        rab_cells = [c for c in data if c["condition"] == "RAB13"]

        for row, (feat_key, feat_label) in enumerate(features_to_plot):
            ax = axes[row, col]
            nt_vals = [c[feat_key] for c in nt_cells]
            rab_vals = [c[feat_key] for c in rab_cells]

            # Only agg-positive cells for morphological features
            if feat_key in ("mean_agg_area", "mean_agg_intensity"):
                nt_vals = [v for v in nt_vals if v > 0]
                rab_vals = [v for v in rab_vals if v > 0]

            if not nt_vals or not rab_vals:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            parts = ax.violinplot(
                [nt_vals, rab_vals], showmedians=True, showmeans=False,
            )
            colors = ["#4393c3", "#d6604d"]
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")

            # SSMD
            nt_arr, rab_arr = np.array(nt_vals), np.array(rab_vals)
            denom = np.sqrt(np.var(rab_arr) + np.var(nt_arr))
            ssmd = (np.mean(rab_arr) - np.mean(nt_arr)) / denom if denom > 0 else 0
            ax.text(
                0.98, 0.95, f"SSMD={ssmd:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

            ax.set_xticks([1, 2])
            ax.set_xticklabels(
                [f"NT\n(n={len(nt_vals)})", f"RAB13\n(n={len(rab_vals)})"]
            )
            ax.set_ylabel(feat_label)
            ax.grid(axis="y", alpha=0.3)

            if row == 0:
                ax.set_title(plate, fontsize=12, fontweight="bold")

    fig.suptitle("NT vs RAB13 — Per-Cell Features (UNet)", fontsize=14, y=1.01)
    fig.tight_layout()
    path = output_dir / "panel_b_quantification.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_panel_c(all_cell_features, output_dir):
    """Panel C: Radar chart of morphological profiles (z-scored, pooled)."""
    import matplotlib.pyplot as plt

    features = [
        ("agg_burden", "Burden"),
        ("agg_count", "Count"),
        ("mean_agg_area", "Area"),
        ("mean_agg_intensity", "Intensity"),
        ("mean_agg_eccentricity", "Eccentricity"),
        ("mean_agg_solidity", "Solidity"),
        ("mean_agg_nucleus_dist", "Nuc. dist"),
    ]
    feat_keys = [k for k, _ in features]
    feat_labels = [l for _, l in features]

    # Z-score within each plate, then pool
    pooled = {"NT": [], "RAB13": []}
    for plate, data in all_cell_features.items():
        pos_cells = [c for c in data if c["agg_count"] > 0]
        if len(pos_cells) < 10:
            continue

        all_vals = {k: np.array([c[k] for c in pos_cells]) for k in feat_keys}
        means = {k: np.mean(v) for k, v in all_vals.items()}
        stds = {k: np.std(v) for k, v in all_vals.items()}

        for c in pos_cells:
            z_cell = {
                k: (c[k] - means[k]) / stds[k] if stds[k] > 0 else 0.0
                for k in feat_keys
            }
            pooled[c["condition"]].append(z_cell)

    if not pooled["NT"] or not pooled["RAB13"]:
        logger.warning("Not enough data for radar chart")
        return

    # Mean z-scores per condition
    profiles = {}
    for cond in ["NT", "RAB13"]:
        profiles[cond] = [
            np.mean([c[k] for c in pooled[cond]]) for k in feat_keys
        ]

    # Radar plot
    n_feat = len(feat_keys)
    angles = np.linspace(0, 2 * np.pi, n_feat, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = {"NT": "#4393c3", "RAB13": "#d6604d"}
    for cond in ["NT", "RAB13"]:
        values = profiles[cond] + profiles[cond][:1]
        ax.plot(angles, values, "o-", linewidth=2, label=cond, color=colors[cond])
        ax.fill(angles, values, alpha=0.15, color=colors[cond])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feat_labels, fontsize=10)
    ax.set_title(
        "Morphological Profiles (z-scored, agg+ cells only)", fontsize=12, y=1.08
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    ax.grid(True)

    path = output_dir / "panel_c_morphological.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_panel_d(dl_features, filter_features, output_dir):
    """Panel D: DL vs filter morphological discrimination comparison."""
    import matplotlib.pyplot as plt

    features_to_compare = [
        ("agg_burden", "Aggregate burden"),
        ("mean_agg_area", "Mean agg. area"),
        ("mean_agg_eccentricity", "Mean eccentricity"),
        ("mean_agg_solidity", "Mean solidity"),
    ]

    n_feat = len(features_to_compare)
    fig, axes = plt.subplots(2, n_feat, figsize=(4 * n_feat, 8))

    method_data = [
        ("DL (UNet)", dl_features, axes[0, :]),
        ("Filter", filter_features, axes[1, :]),
    ]

    for method_label, feat_data, axrow in method_data:
        # Pool agg-positive cells across plates
        nt_cells = []
        rab_cells = []
        for plate_data in feat_data.values():
            for c in plate_data:
                if c["agg_count"] > 0:
                    if c["condition"] == "NT":
                        nt_cells.append(c)
                    else:
                        rab_cells.append(c)

        for col, (feat_key, feat_label) in enumerate(features_to_compare):
            ax = axrow[col]
            nt_vals = [c[feat_key] for c in nt_cells]
            rab_vals = [c[feat_key] for c in rab_cells]

            if not nt_vals or not rab_vals:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            parts = ax.violinplot(
                [nt_vals, rab_vals], showmedians=True, showmeans=False,
            )
            colors = ["#4393c3", "#d6604d"]
            for i, pc in enumerate(parts["bodies"]):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
            parts["cmedians"].set_color("black")

            # SSMD
            nt_arr, rab_arr = np.array(nt_vals), np.array(rab_vals)
            denom = np.sqrt(np.var(rab_arr) + np.var(nt_arr))
            ssmd = (np.mean(rab_arr) - np.mean(nt_arr)) / denom if denom > 0 else 0
            ax.text(
                0.98, 0.95, f"SSMD={ssmd:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

            ax.set_xticks([1, 2])
            ax.set_xticklabels(["NT", "RAB13"])
            if col == 0:
                ax.set_ylabel(method_label, fontsize=11, fontweight="bold")
            ax.set_title(feat_label if method_label == "DL (UNet)" else "", fontsize=10)
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("DL vs Filter: Morphological Discrimination", fontsize=14, y=1.02)
    fig.tight_layout()
    path = output_dir / "panel_d_dl_vs_filter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def save_cell_features_csv(all_cell_features, output_path):
    """Save per-cell features to CSV (all plates combined)."""
    rows = []
    for plate, data in all_cell_features.items():
        for c in data:
            rows.append({"plate": plate, **c})

    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {output_path} ({len(rows)} cells)")


def save_field_summary_csv(all_field_summaries, output_path):
    """Save per-field summary to CSV."""
    rows = []
    for plate, summaries in all_field_summaries.items():
        for s in summaries:
            rows.append({"plate": plate, **s})

    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Saved {output_path} ({len(rows)} fields)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 4: Biological validation on CRISPR screen controls"
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_ROOT),
                        help="Root directory with plate folders")
    parser.add_argument("--plates", nargs="+", default=None,
                        help="Plate folder names to process (default: all)")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="UNet checkpoint path")
    parser.add_argument("-o", "--output-dir", type=str, default=str(OUTPUT_ROOT),
                        help="Output directory")
    parser.add_argument("--max-fields", type=int, default=None,
                        help="Max fields per plate (for testing)")
    parser.add_argument("--skip-segmentation", action="store_true",
                        help="Skip segmentation, only compute features from existing masks")
    parser.add_argument("--no-gpu", dest="gpu", action="store_false", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    # Discover plates
    if args.plates:
        plate_dirs = [data_dir / p for p in args.plates]
    else:
        plate_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    plate_names = [d.name for d in plate_dirs]
    logger.info(f"Plates: {plate_names}")

    # Step 1: Run 3-pass segmentation
    if not args.skip_segmentation:
        run_segmentation(
            plate_dirs, output_dir, args.checkpoint, args.gpu,
            max_fields=args.max_fields,
        )

    # Step 2: Load masks and compute per-cell features
    logger.info("\n" + "=" * 60)
    logger.info("Computing per-cell features from saved masks")
    logger.info("=" * 60)

    all_dl_features = {}
    all_filter_features = {}
    all_field_summaries = {}

    for plate_dir in plate_dirs:
        plate_name = plate_dir.name
        logger.info(f"\nPlate: {plate_name}")

        logger.info("  Loading UNet masks...")
        dl_cells, dl_fields = load_masks_and_compute_features(
            plate_dir, "unet", output_dir,
        )
        all_dl_features[plate_name] = dl_cells
        all_field_summaries[plate_name] = dl_fields
        logger.info(f"  UNet: {len(dl_cells)} cells from {len(dl_fields)} fields")

        logger.info("  Loading filter masks...")
        filter_cells, _ = load_masks_and_compute_features(
            plate_dir, "filter", output_dir,
        )
        all_filter_features[plate_name] = filter_cells
        logger.info(f"  Filter: {len(filter_cells)} cells")

    # Step 3: Save CSVs
    save_cell_features_csv(all_dl_features, output_dir / "cell_features_dl.csv")
    save_cell_features_csv(all_filter_features, output_dir / "cell_features_filter.csv")
    save_field_summary_csv(all_field_summaries, output_dir / "field_summary.csv")

    # Step 4: Generate plots
    logger.info("\nGenerating plots...")

    if plate_dirs:
        plot_panel_a(plate_dirs[0], output_dir)

    plot_panel_b(all_dl_features, output_dir)
    plot_panel_c(all_dl_features, output_dir)
    plot_panel_d(all_dl_features, all_filter_features, output_dir)

    logger.info(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
