#!/usr/bin/env python
"""
Preprocessing Cellpose segmentation benchmark: 3 variants x 90 curated HCS images.

Standalone script (no AggreQuant imports). Compares how different preprocessing of
the nuclear channel affects Cellpose cyto3 cell segmentation:

  1. cellpose_cell_only    — single-channel FarRed, no nuclear info
  2. cellpose_raw_nuclei   — FarRed + raw DAPI image
  3. cellpose_nuclei_seeds — FarRed + binary nuclei mask from StarDist

Primary images: *wv 631 - FarRed*.tif
Nuclear channel: matching *wv 390 - Blue*.tif

Usage:
    conda activate cell-bench
    python run_benchmark.py \\
        --data-dir data/images \\
        [--output-dir results] \\
        [--no-gpu] \\
        [--no-masks]
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import skimage.filters
import skimage.measure
import skimage.morphology
import tifffile
from tqdm import tqdm

# TF: allocate GPU memory on demand (must precede any TF import)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------

VARIANT_IDS = [
    "cellpose_cell_only",
    "cellpose_raw_nuclei",
    "cellpose_nuclei_seeds",
]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cellpose(gpu):
    from cellpose import models
    return models.Cellpose(gpu=gpu, model_type="cyto3")


def load_stardist():
    from stardist.models import StarDist2D
    return StarDist2D.from_pretrained("2D_versatile_fluo")


# ---------------------------------------------------------------------------
# StarDist preprocessing — reproduces aggrequant/segmentation/stardist.py
# ---------------------------------------------------------------------------

SIGMA_DENOISE = 2
SIGMA_BACKGROUND = 50
MIN_NUCLEUS_AREA = 300
MAX_NUCLEUS_AREA = 15000


def stardist_to_binary_mask(img_nuc, stardist_model):
    """Reproduce AggreQuant's StarDist pipeline and return binary mask.

    Steps (matching StarDistSegmenter.segment()):
      1. Gaussian denoise (sigma=2)
      2. Background normalization: denoised / Gaussian(sigma=50)
      3. csbdeep normalize + StarDist predict_instances
      4. Size exclusion (300-15000 px)
      5. Border separation (Sobel + dilation)
      6. Convert to binary float32 mask
    """
    from csbdeep.utils import normalize

    # 1. Gaussian denoise
    img = img_nuc.astype(np.float32)
    denoised = skimage.filters.gaussian(
        img, sigma=SIGMA_DENOISE, mode="reflect", preserve_range=True,
    )

    # 2. Background normalization
    background = skimage.filters.gaussian(
        denoised, sigma=SIGMA_BACKGROUND, mode="reflect", preserve_range=True,
    )
    normalized = denoised / (background + 1e-8)

    # 3. StarDist inference
    labels, _ = stardist_model.predict_instances(
        normalize(normalized), predict_kwargs=dict(verbose=False),
    )

    # 4. Size exclusion
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label_id, area in zip(unique_labels[1:], counts[1:]):
        if area < MIN_NUCLEUS_AREA or area > MAX_NUCLEUS_AREA:
            labels[labels == label_id] = 0

    # 5. Border separation (Sobel + dilation)
    edges = skimage.filters.sobel(labels)
    fat_edges = skimage.morphology.dilation(edges > 0)
    labels[fat_edges] = 0

    # 6. Binary mask
    return (labels > 0).astype(np.float32)


# ---------------------------------------------------------------------------
# Preprocessing (excluded from Cellpose timing)
# ---------------------------------------------------------------------------

def preprocess_cell_only(img_cell):
    """Single-channel FarRed: (H, W), channels=[0, 0]."""
    return img_cell


def preprocess_raw_nuclei(img_cell, img_nuc):
    """FarRed + raw DAPI: (2, H, W), channels=[1, 2]."""
    h, w = img_cell.shape[:2]
    img_2ch = np.zeros((2, h, w))
    img_2ch[0, :, :] = img_cell
    img_2ch[1, :, :] = img_nuc
    return img_2ch


def preprocess_nuclei_seeds(img_cell, nuclei_mask):
    """FarRed + binary nuclei mask: (2, H, W), channels=[1, 2]."""
    h, w = img_cell.shape[:2]
    img_2ch = np.zeros((2, h, w))
    img_2ch[0, :, :] = img_cell
    img_2ch[1, :, :] = nuclei_mask
    return img_2ch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def infer_cellpose(model, img_pre, channels):
    """Run Cellpose eval (this is what gets timed)."""
    masks, _, _, _ = model.eval(
        img_pre, diameter=None, channels=channels,
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels):
    """Compute cell count, area, solidity, and eccentricity from label mask.

    Returns dict with: cell_count, mean/median_area_px, mean/median_solidity,
    mean/median_eccentricity.
    """
    if labels.max() == 0:
        return {
            "cell_count": 0,
            "mean_area_px": 0.0, "median_area_px": 0.0,
            "mean_solidity": 0.0, "median_solidity": 0.0,
            "mean_eccentricity": 0.0, "median_eccentricity": 0.0,
        }

    props = skimage.measure.regionprops(labels)
    areas = np.array([p.area for p in props])
    solidities = np.array([p.solidity for p in props])
    eccentricities = np.array([p.eccentricity for p in props])

    return {
        "cell_count": len(props),
        "mean_area_px": round(float(np.mean(areas)), 1),
        "median_area_px": round(float(np.median(areas)), 1),
        "mean_solidity": round(float(np.mean(solidities)), 4),
        "median_solidity": round(float(np.median(solidities)), 4),
        "mean_eccentricity": round(float(np.mean(eccentricities)), 4),
        "median_eccentricity": round(float(np.median(eccentricities)), 4),
    }


# ---------------------------------------------------------------------------
# Image discovery
# ---------------------------------------------------------------------------

def discover_images(data_dir):
    """Walk category subfolders; return FarRed images as list of dicts.

    Each dict: path, name, category, nuc_path.
    """
    images = []
    for cat_dir in sorted(Path(data_dir).iterdir()):
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*FarRed*.tif*")):
            nuc = p.parent / p.name.replace("wv 631 - FarRed", "wv 390 - Blue")
            images.append({
                "path": p,
                "name": p.name,
                "category": cat_dir.name,
                "nuc_path": nuc if nuc.exists() else None,
            })
    return images


# ---------------------------------------------------------------------------
# GPU utilities
# ---------------------------------------------------------------------------

def clear_tf_gpu():
    """Release TensorFlow GPU memory."""
    try:
        import tensorflow as tf
        tf.keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()


def clear_torch_gpu():
    """Release PyTorch GPU memory."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocessing Cellpose benchmark: 3 variants x 90 images",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to image directory (9 category subfolders with FarRed + Blue TIFFs)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <script_dir>/results)",
    )
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU-only")
    parser.add_argument("--no-masks", action="store_true", help="Skip saving masks")
    parser.add_argument(
        "--variants", nargs="+", default=None,
        help=f"Run only these variants (default: all). Choices: {VARIANT_IDS}",
    )
    args = parser.parse_args()

    gpu = not args.no_gpu
    save_masks = not args.no_masks
    data_dir = Path(args.data_dir)
    output_dir = (
        Path(args.output_dir) if args.output_dir
        else Path(__file__).parent / "results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Discover images ---
    images = discover_images(data_dir)
    if not images:
        parser.error(f"No FarRed .tif images found under {data_dir}")

    # Filter to images that have DAPI (needed for raw_nuclei and seeds variants)
    images = [im for im in images if im["nuc_path"] is not None]
    categories = sorted({im["category"] for im in images})
    print(f"Found {len(images)} FarRed images with matching DAPI in {len(categories)} categories")

    # --- Select variants ---
    variants = VARIANT_IDS
    if args.variants:
        for v in args.variants:
            if v not in VARIANT_IDS:
                parser.error(f"Unknown variant: {v}. Valid: {VARIANT_IDS}")
        variants = args.variants

    print(f"Running {len(variants)} variant(s), GPU={'on' if gpu else 'off'}")
    print(f"Output -> {output_dir}\n")

    # ------------------------------------------------------------------
    # Phase 1: Pre-compute StarDist binary masks (TensorFlow)
    # Must run BEFORE Cellpose to avoid TF/PyTorch GPU memory conflicts.
    # Masks are cached to disk in a scratch directory.
    # ------------------------------------------------------------------
    nuclei_mask_cache = {}  # image_name -> binary mask array
    stardist_timing = {}    # image_name -> seconds

    if "cellpose_nuclei_seeds" in variants:
        # Check if seeds variant already has all masks cached
        seeds_mask_dir = output_dir / "masks" / "cellpose_nuclei_seeds"
        seeds_existing = set()
        if seeds_mask_dir.is_dir():
            seeds_existing = {p.name for p in seeds_mask_dir.glob("*.tif*")}
        needs_stardist = any(im["name"] not in seeds_existing for im in images)

        if needs_stardist:
            print("Phase 1: Pre-computing StarDist nuclei masks (TensorFlow) ...")
            t0 = time.perf_counter()
            stardist_model = load_stardist()
            print(f"  StarDist loaded in {time.perf_counter() - t0:.1f}s")

            for img_info in tqdm(images, desc="StarDist", unit="img"):
                if img_info["name"] in seeds_existing:
                    continue  # will load from Cellpose mask cache later
                img_nuc = tifffile.imread(img_info["nuc_path"])
                t_sd = time.perf_counter()
                nuclei_mask_cache[img_info["name"]] = stardist_to_binary_mask(
                    img_nuc, stardist_model,
                )
                stardist_timing[img_info["name"]] = time.perf_counter() - t_sd

            del stardist_model
            clear_tf_gpu()
            print(f"  {len(nuclei_mask_cache)} masks computed, TF memory released\n")
        else:
            print("Phase 1: All seeds masks cached, skipping StarDist\n")

    # ------------------------------------------------------------------
    # Phase 2: Run Cellpose for all variants (PyTorch)
    # ------------------------------------------------------------------
    print("Phase 2: Loading Cellpose cyto3 (PyTorch) ...")
    t0 = time.perf_counter()
    cellpose_model = load_cellpose(gpu)
    print(f"  ready in {time.perf_counter() - t0:.1f}s\n")

    # Load previous timing data for checkpoint recovery
    prev_timing = {}
    timing_path = output_dir / "timing.csv"
    if timing_path.exists():
        try:
            prev_df = pd.read_csv(timing_path)
            for _, row in prev_df.iterrows():
                prev_timing[(row["variant_id"], row["image_name"])] = {
                    "inference_time_s": row["inference_time_s"],
                    "stardist_preprocess_time_s": row.get("stardist_preprocess_time_s", float("nan")),
                    "device": row["device"],
                }
        except Exception:
            pass

    count_rows = []
    timing_rows = []
    device_str = "gpu" if gpu else "cpu"

    for variant_id in variants:
        mask_dir = output_dir / "masks" / variant_id

        # Check cached masks
        existing = set()
        if mask_dir.is_dir():
            existing = {p.name for p in mask_dir.glob("*.tif*")}

        n_cached = sum(1 for im in images if im["name"] in existing)
        n_new = len(images) - n_cached

        if n_cached == len(images):
            print(f"{variant_id}: all {n_cached} masks cached, loading from disk")
        elif n_cached > 0:
            print(f"{variant_id}: {n_cached} cached, {n_new} to run")

        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        for img_info in tqdm(images, desc=variant_id, unit="img"):
            mask_path = mask_dir / img_info["name"]

            if img_info["name"] in existing:
                # --- Cached: load mask from disk ---
                labels = tifffile.imread(str(mask_path)).astype(np.int32)
                metrics = compute_metrics(labels)

                prev = prev_timing.get((variant_id, img_info["name"]))
                elapsed = prev["inference_time_s"] if prev else float("nan")
                sd_time = prev.get("stardist_preprocess_time_s", float("nan")) if prev else float("nan")
                dev = prev["device"] if prev else device_str
            else:
                # --- New: run inference ---
                img_cell = tifffile.imread(img_info["path"])

                sd_time = float("nan")

                try:
                    if variant_id == "cellpose_cell_only":
                        img_pre = preprocess_cell_only(img_cell)
                        channels = [0, 0]
                    elif variant_id == "cellpose_raw_nuclei":
                        img_nuc = tifffile.imread(img_info["nuc_path"])
                        img_pre = preprocess_raw_nuclei(img_cell, img_nuc)
                        channels = [1, 2]
                    elif variant_id == "cellpose_nuclei_seeds":
                        nuclei_mask = nuclei_mask_cache.get(img_info["name"])
                        if nuclei_mask is None:
                            raise RuntimeError("StarDist mask not in cache")
                        sd_time = stardist_timing.get(img_info["name"], float("nan"))
                        img_pre = preprocess_nuclei_seeds(img_cell, nuclei_mask)
                        channels = [1, 2]
                    else:
                        raise ValueError(f"Unknown variant: {variant_id}")

                    t_start = time.perf_counter()
                    labels = infer_cellpose(cellpose_model, img_pre, channels)
                    elapsed = time.perf_counter() - t_start

                except Exception as e:
                    tqdm.write(f"  ERROR {variant_id} on {img_info['name']}: {e}")
                    labels = np.zeros(img_cell.shape[:2], dtype=np.int32)
                    elapsed = float("nan")

                labels = np.asarray(labels, dtype=np.int32)
                metrics = compute_metrics(labels)
                dev = device_str

                if save_masks:
                    tifffile.imwrite(
                        str(mask_path),
                        labels.astype(np.uint16),
                        compression="zlib",
                    )

            count_rows.append({
                "image_name": img_info["name"],
                "category": img_info["category"],
                "variant_id": variant_id,
                **metrics,
            })
            timing_rows.append({
                "variant_id": variant_id,
                "image_name": img_info["name"],
                "inference_time_s": round(elapsed, 4) if not np.isnan(elapsed) else elapsed,
                "stardist_preprocess_time_s": round(sd_time, 4) if not np.isnan(sd_time) else sd_time,
                "device": dev,
            })

    # Cleanup
    del cellpose_model
    clear_torch_gpu()

    # --- Save CSVs (merge with existing data for variants not in this run) ---
    counts_df = pd.DataFrame(count_rows)
    timing_df = pd.DataFrame(timing_rows)

    run_variants = set(variants)

    counts_path = output_dir / "counts.csv"
    if counts_path.exists():
        prev_counts = pd.read_csv(counts_path)
        kept = prev_counts[~prev_counts["variant_id"].isin(run_variants)]
        counts_df = pd.concat([kept, counts_df], ignore_index=True)

    if timing_path.exists():
        prev_timing_df = pd.read_csv(timing_path)
        kept = prev_timing_df[~prev_timing_df["variant_id"].isin(run_variants)]
        timing_df = pd.concat([kept, timing_df], ignore_index=True)

    counts_df.to_csv(counts_path, index=False)
    timing_df.to_csv(timing_path, index=False)

    # --- Summary ---
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Saved: {counts_path} ({len(counts_df)} rows)")
    print(f"Saved: {timing_path} ({len(timing_df)} rows)")
    if save_masks:
        print(f"Masks: {output_dir / 'masks'}/  ({len(variants)} variant dirs)")

    if not counts_df.empty:
        print(f"\n{sep}")
        print("Mean cell count — category x variant")
        print(sep)
        pivot = counts_df.pivot_table(
            values="cell_count", index="category",
            columns="variant_id", aggfunc="mean",
        )
        print(pivot.round(1).to_string())

        print(f"\n{sep}")
        print("Mean solidity — category x variant")
        print(sep)
        pivot_sol = counts_df.pivot_table(
            values="mean_solidity", index="category",
            columns="variant_id", aggfunc="mean",
        )
        print(pivot_sol.round(4).to_string())

        print(f"\n{sep}")
        print("Cellpose inference time per variant (seconds / image)")
        print(sep)
        ts = timing_df.groupby("variant_id")["inference_time_s"].agg(
            ["mean", "std", "min", "max"]
        )
        print(ts.round(3).to_string())

        if "cellpose_nuclei_seeds" in timing_df["variant_id"].values:
            seeds_t = timing_df[timing_df["variant_id"] == "cellpose_nuclei_seeds"]
            sd_times = seeds_t["stardist_preprocess_time_s"].dropna()
            if len(sd_times) > 0:
                print(f"\nStarDist preprocessing: {sd_times.mean():.3f} ± {sd_times.std():.3f} s/image")

    print(f"\n{sep}")
    print("Done.")


if __name__ == "__main__":
    main()
