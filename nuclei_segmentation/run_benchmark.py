#!/usr/bin/env python
"""
Nuclei segmentation benchmark: 13 model configs x 100 curated HCS images.

Standalone script (no AggreQuant imports). Runs pretrained StarDist, Cellpose,
DeepCell, and InstanSeg models on challenging edge cases. Saves labeled masks, nuclei
counts with area statistics, and per-image inference timing.

Three configs use a real FarRed cell channel alongside the DAPI nuclei channel:
  - cellpose_cyto2_with_cell, cellpose_cyto3_with_cell, deepcell_mesmer_with_cell
Images without a matching FarRed file are skipped for those configs.

Usage:
    conda activate nuclei-bench
    python run_benchmark.py \
        --data-dir /path/to/curated/images \
        [--output-dir results] \
        [--image-mpp 0.325] \
        [--no-gpu] \
        [--models stardist_2d_fluo cellpose_nuclei ...]
"""

import argparse
import gc
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

# TF: allocate GPU memory on demand (must precede any TF import)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")  # suppress INFO logs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_stardist():
    from stardist.models import StarDist2D
    return StarDist2D.from_pretrained("2D_versatile_fluo")


def load_cellpose(model_type, gpu):
    from cellpose import models
    return models.Cellpose(gpu=gpu, model_type=model_type)


def load_deepcell_nuclear():
    from deepcell.applications import NuclearSegmentation
    return NuclearSegmentation.from_version("1.1")


def load_deepcell_mesmer():
    from deepcell.applications import Mesmer
    return Mesmer()


def load_instanseg():
    from instanseg import InstanSeg
    return InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)


# ---------------------------------------------------------------------------
# Preprocessing (everything *before* the model call)
# ---------------------------------------------------------------------------

def preprocess_stardist(img):
    from csbdeep.utils import normalize
    return normalize(img, 1, 99.8, axis=(0, 1))


def preprocess_cellpose(img, channels):
    if channels == [1, 2]:
        return np.stack([img, img], axis=-1)  # (H, W, 2) — DAPI as both
    return img  # (H, W)


def preprocess_cellpose_cell(img_nuc, img_cell):
    """Cellpose cyto with real cell channel: [cell, nucleus] → channels=[1,2]."""
    return np.stack([img_cell, img_nuc], axis=-1)  # (H, W, 2)


def preprocess_deepcell_1ch(img):
    return img[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)


def preprocess_deepcell_2ch(img):
    membrane = np.zeros_like(img)
    img_2ch = np.stack([img, membrane], axis=-1)  # (H, W, 2)
    return img_2ch[np.newaxis, ...]  # (1, H, W, 2)


def preprocess_deepcell_2ch_cell(img_nuc, img_cell):
    """Mesmer with real cell channel: [nuclear, membrane] order."""
    img_2ch = np.stack([img_nuc, img_cell], axis=-1)  # (H, W, 2)
    return img_2ch[np.newaxis, ...]  # (1, H, W, 2)


def preprocess_instanseg_1ch(img):
    """InstanSeg single channel: (H, W) float32."""
    return img.astype(np.float32)


def preprocess_instanseg_2ch(img_nuc, img_cell):
    """InstanSeg two channel: (H, W, 2) float32 → will be reshaped internally."""
    return np.stack([img_nuc, img_cell], axis=0).astype(np.float32)  # (2, H, W)


# ---------------------------------------------------------------------------
# Inference (only the model call — this is what gets timed)
# ---------------------------------------------------------------------------

def infer_stardist(model, img_pre):
    labels, _ = model.predict_instances(img_pre)
    return labels


def infer_cellpose(model, img_pre, channels):
    masks, _, _, _ = model.eval(
        img_pre, diameter=None, channels=channels,
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks


def infer_deepcell_nuclear(model, img_pre, image_mpp):
    labels = model.predict(img_pre, image_mpp=image_mpp)
    return labels[0, :, :, 0]


def infer_deepcell_mesmer(model, img_pre, image_mpp):
    labels = model.predict(img_pre, image_mpp=image_mpp, compartment="nuclear")
    return labels[0, :, :, 0]


def infer_instanseg(model, img_pre, image_mpp):
    """InstanSeg: eval_small_image with pixel_size, target='nuclei'."""
    result = model.eval_small_image(
        img_pre, pixel_size=image_mpp, target="nuclei", return_image_tensor=False,
    )
    return result.squeeze().cpu().numpy()  # (1,1,H,W) → (H,W)


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
# Ordered: TF models first (StarDist, DeepCell), then PyTorch (Cellpose).
# Configs sharing a model_key are adjacent so the model loads once.

MODEL_CONFIGS = [
    # --- Single-channel (DAPI only) ---
    {"id": "stardist_2d_fluo",       "model_key": "stardist",   "framework": "tensorflow"},
    {"id": "deepcell_nuclear",        "model_key": "dc_nuclear", "framework": "tensorflow"},
    {"id": "deepcell_mesmer",         "model_key": "dc_mesmer",  "framework": "tensorflow"},
    {"id": "cellpose_nuclei",         "model_key": "cp_nuclei",  "framework": "pytorch"},
    {"id": "cellpose_cyto2_no_nuc",   "model_key": "cp_cyto2",   "framework": "pytorch"},
    {"id": "cellpose_cyto2_with_nuc", "model_key": "cp_cyto2",   "framework": "pytorch"},
    {"id": "cellpose_cyto3_no_nuc",   "model_key": "cp_cyto3",   "framework": "pytorch"},
    {"id": "cellpose_cyto3_with_nuc", "model_key": "cp_cyto3",   "framework": "pytorch"},
    # --- InstanSeg (PyTorch) ---
    {"id": "instanseg_fluorescence",     "model_key": "instanseg", "framework": "pytorch"},
    # --- Two-channel (DAPI + FarRed cell channel) ---
    {"id": "deepcell_mesmer_with_cell",  "model_key": "dc_mesmer", "framework": "tensorflow", "needs_cell": True},
    {"id": "cellpose_cyto2_with_cell",   "model_key": "cp_cyto2",  "framework": "pytorch",    "needs_cell": True},
    {"id": "cellpose_cyto3_with_cell",   "model_key": "cp_cyto3",  "framework": "pytorch",    "needs_cell": True},
    {"id": "instanseg_fluorescence_with_cell", "model_key": "instanseg", "framework": "pytorch", "needs_cell": True},
]

ALL_MODEL_IDS = [c["id"] for c in MODEL_CONFIGS]

_LOADERS = {
    "stardist":   lambda gpu: load_stardist(),
    "dc_nuclear": lambda gpu: load_deepcell_nuclear(),
    "dc_mesmer":  lambda gpu: load_deepcell_mesmer(),
    "cp_nuclei":  lambda gpu: load_cellpose("nuclei", gpu),
    "cp_cyto2":   lambda gpu: load_cellpose("cyto2", gpu),
    "cp_cyto3":   lambda gpu: load_cellpose("cyto3", gpu),
    "instanseg":  lambda gpu: load_instanseg(),
}


def preprocess(config_id, img_nuc, img_cell=None):
    """Return model-ready input (excluded from timing).

    Args:
        config_id: Model configuration ID.
        img_nuc: DAPI nuclei image (H, W).
        img_cell: FarRed cell image (H, W), only for *_with_cell configs.
    """
    if config_id == "stardist_2d_fluo":
        return preprocess_stardist(img_nuc)
    if config_id == "deepcell_nuclear":
        return preprocess_deepcell_1ch(img_nuc)
    if config_id == "deepcell_mesmer":
        return preprocess_deepcell_2ch(img_nuc)
    if config_id == "deepcell_mesmer_with_cell":
        return preprocess_deepcell_2ch_cell(img_nuc, img_cell)
    if config_id == "instanseg_fluorescence":
        return preprocess_instanseg_1ch(img_nuc)
    if config_id == "instanseg_fluorescence_with_cell":
        return preprocess_instanseg_2ch(img_nuc, img_cell)
    if config_id.endswith("_with_cell"):
        return preprocess_cellpose_cell(img_nuc, img_cell)
    # Remaining Cellpose configs
    channels = [1, 2] if config_id.endswith("_with_nuc") else [0, 0]
    return preprocess_cellpose(img_nuc, channels)


def infer(config_id, model, img_pre, image_mpp):
    """Run the model call only (this is what gets timed)."""
    if config_id == "stardist_2d_fluo":
        return infer_stardist(model, img_pre)
    if config_id == "deepcell_nuclear":
        return infer_deepcell_nuclear(model, img_pre, image_mpp)
    if config_id in ("deepcell_mesmer", "deepcell_mesmer_with_cell"):
        return infer_deepcell_mesmer(model, img_pre, image_mpp)
    if config_id.startswith("instanseg_"):
        return infer_instanseg(model, img_pre, image_mpp)
    # All Cellpose configs: _with_cell and _with_nuc use channels=[1,2]
    channels = [1, 2] if ("_with_nuc" in config_id or "_with_cell" in config_id) else [0, 0]
    return infer_cellpose(model, img_pre, channels)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_images(data_dir):
    """Walk category subfolders; return DAPI-only images as list of dicts.

    Each dict has: path, name, category, cell_path (Path or None).
    FarRed files are expected alongside DAPI with 'wv 631 - FarRed' in name.
    """
    images = []
    for cat_dir in sorted(Path(data_dir).iterdir()):
        if not cat_dir.is_dir():
            continue
        for p in sorted(cat_dir.glob("*Blue*.tif*")):
            farred = p.parent / p.name.replace("wv 390 - Blue", "wv 631 - FarRed")
            images.append({
                "path": p,
                "name": p.name,
                "category": cat_dir.name,
                "cell_path": farred if farred.exists() else None,
            })
    return images


def count_and_areas(labels):
    """Return (nuclei_count, mean_area_px, median_area_px)."""
    if labels.max() == 0:
        return 0, 0.0, 0.0
    areas = np.bincount(labels.ravel())[1:]  # drop background
    areas = areas[areas > 0]                  # drop label gaps
    if len(areas) == 0:
        return 0, 0.0, 0.0
    return len(areas), float(np.mean(areas)), float(np.median(areas))


def clear_gpu(framework):
    """Release GPU memory for the given framework."""
    if framework == "tensorflow":
        import tensorflow as tf
        tf.keras.backend.clear_session()
    elif framework == "pytorch":
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    gc.collect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Nuclei segmentation benchmark: 13 models x 100 images",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to curated image directory (9 category subfolders)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--image-mpp", type=float, default=0.325,
        help="Microns per pixel for DeepCell (default: 0.325)",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="Force CPU-only inference",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help=f"Run only these model IDs (default: all). Choices: {ALL_MODEL_IDS}",
    )
    parser.add_argument(
        "--no-masks", action="store_true",
        help="Skip saving label masks to disk",
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
    categories = sorted({im["category"] for im in images})
    if not images:
        parser.error(f"No .tif images found under {data_dir}")
    n_with_cell = sum(1 for im in images if im["cell_path"] is not None)
    print(f"Found {len(images)} DAPI images in {len(categories)} categories")
    print(f"  {n_with_cell} have matching FarRed cell channel")

    # --- Select model configs ---
    configs = MODEL_CONFIGS
    if args.models:
        valid = set(ALL_MODEL_IDS)
        for m in args.models:
            if m not in valid:
                parser.error(f"Unknown model: {m}. Valid: {sorted(valid)}")
        configs = [c for c in MODEL_CONFIGS if c["id"] in args.models]

    print(f"Running {len(configs)} model config(s), GPU={'on' if gpu else 'off'}")
    print(f"Output -> {output_dir}\n")

    # --- Load previous timing data for checkpoint recovery ---
    prev_timing = {}
    timing_path = output_dir / "timing.csv"
    if timing_path.exists():
        try:
            prev_df = pd.read_csv(timing_path)
            for _, row in prev_df.iterrows():
                prev_timing[(row["model_id"], row["image_name"])] = {
                    "inference_time_s": row["inference_time_s"],
                    "device": row["device"],
                }
        except Exception:
            pass  # corrupt/empty file — ignore

    # --- Run benchmark ---
    count_rows = []
    timing_rows = []

    loaded_key = None
    model = None
    fw = None

    for cfg in configs:
        cid = cfg["id"]
        mkey = cfg["model_key"]
        fw_new = cfg["framework"]
        needs_cell = cfg.get("needs_cell", False)

        # Filter images for this config
        if needs_cell:
            cfg_images = [im for im in images if im["cell_path"] is not None]
        else:
            cfg_images = images

        mask_dir = output_dir / "masks" / cid

        # Check which images already have masks (checkpoint)
        existing = set()
        if mask_dir.is_dir():
            existing = {p.name for p in mask_dir.glob("*.tif*")}

        n_cached = sum(1 for im in cfg_images if im["name"] in existing)
        n_new = len(cfg_images) - n_cached

        if needs_cell and len(cfg_images) < len(images):
            print(f"{cid}: {len(cfg_images)}/{len(images)} images have cell channel")

        if n_cached == len(cfg_images):
            print(f"{cid}: all {n_cached} masks cached, loading counts from disk")
        elif n_cached > 0:
            print(f"{cid}: {n_cached} cached, {n_new} to run")

        # Only load model if there are new images to process
        need_model = n_new > 0
        if need_model:
            if mkey != loaded_key:
                if model is not None:
                    del model
                    clear_gpu(fw)
                print(f"Loading {mkey} ({fw_new}) ...")
                t_load = time.perf_counter()
                model = _LOADERS[mkey](gpu)
                print(f"  ready in {time.perf_counter() - t_load:.1f}s")
                loaded_key = mkey
                fw = fw_new

        if save_masks:
            mask_dir.mkdir(parents=True, exist_ok=True)

        device_str = "gpu" if gpu else "cpu"

        for img_info in tqdm(cfg_images, desc=cid, unit="img"):
            mask_path = mask_dir / img_info["name"]

            if img_info["name"] in existing:
                # --- Cached: load mask from disk ---
                labels = tifffile.imread(str(mask_path)).astype(np.int32)
                n, mean_a, med_a = count_and_areas(labels)

                # Recover timing from previous run if available
                prev = prev_timing.get((cid, img_info["name"]))
                elapsed = prev["inference_time_s"] if prev else float("nan")
                dev = prev["device"] if prev else device_str
            else:
                # --- New: run inference ---
                img_nuc = tifffile.imread(img_info["path"])
                img_cell = None
                if needs_cell:
                    img_cell = tifffile.imread(img_info["cell_path"])

                try:
                    img_pre = preprocess(cid, img_nuc, img_cell)
                    t0 = time.perf_counter()
                    labels = infer(cid, model, img_pre, args.image_mpp)
                    elapsed = time.perf_counter() - t0
                except Exception as e:
                    tqdm.write(f"  ERROR {cid} on {img_info['name']}: {e}")
                    labels = np.zeros(img_nuc.shape[:2], dtype=np.int32)
                    elapsed = float("nan")

                labels = np.asarray(labels, dtype=np.int32)
                n, mean_a, med_a = count_and_areas(labels)
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
                "model_id": cid,
                "nuclei_count": n,
                "mean_area_px": round(mean_a, 1),
                "median_area_px": round(med_a, 1),
            })
            timing_rows.append({
                "model_id": cid,
                "image_name": img_info["name"],
                "inference_time_s": round(elapsed, 4) if not np.isnan(elapsed) else elapsed,
                "device": dev,
            })

    # Cleanup last model
    if model is not None:
        del model
        clear_gpu(fw)

    # --- Save CSVs ---
    counts_df = pd.DataFrame(count_rows)
    timing_df = pd.DataFrame(timing_rows)
    counts_df.to_csv(output_dir / "counts.csv", index=False)
    timing_df.to_csv(output_dir / "timing.csv", index=False)

    # --- Summary ---
    sep = "=" * 70
    print(f"\n{sep}")
    print(f"Saved: {output_dir / 'counts.csv'} ({len(counts_df)} rows)")
    print(f"Saved: {output_dir / 'timing.csv'} ({len(timing_df)} rows)")
    if save_masks:
        print(f"Masks: {output_dir / 'masks'}/  ({len(configs)} model dirs)")

    if not counts_df.empty:
        print(f"\n{sep}")
        print("Mean nuclei count — category x model")
        print(sep)
        pivot = counts_df.pivot_table(
            values="nuclei_count", index="category",
            columns="model_id", aggfunc="mean",
        )
        print(pivot.round(1).to_string())

        print(f"\n{sep}")
        print("Inference time per model (seconds / image)")
        print(sep)
        ts = timing_df.groupby("model_id")["inference_time_s"].agg(
            ["mean", "std", "min", "max"]
        )
        print(ts.round(3).to_string())

    print(f"\n{sep}")
    print("Done.")


if __name__ == "__main__":
    main()
