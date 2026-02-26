#!/usr/bin/env python
"""
Cell segmentation benchmark: 10 model configs x 90 curated HCS images.

Standalone script (no AggreQuant imports). Runs pretrained Cellpose, DeepCell
Mesmer, InstanSeg, and CellSAM models on FarRed (cytoplasmic) channel images.
Five configs also use the paired DAPI channel as a nuclear hint.

Primary images: *wv 631 - FarRed*.tif
Nuclear hint  : matching *wv 390 - Blue*.tif (always present in this dataset)

Usage:
    conda activate nuclei-bench
    python run_benchmark.py \\
        --data-dir data/images \\
        [--output-dir results] \\
        [--image-mpp 0.325] \\
        [--no-gpu] \\
        [--models cellpose_cyto2 deepcell_mesmer ...]
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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_cellpose(model_type, gpu):
    from cellpose import models
    return models.Cellpose(gpu=gpu, model_type=model_type)


def load_deepcell_mesmer():
    from deepcell.applications import Mesmer
    return Mesmer()


def load_instanseg():
    from instanseg import InstanSeg
    return InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)


def load_cellsam():
    from cellSAM import get_model
    return get_model()


# ---------------------------------------------------------------------------
# Preprocessing (excluded from timing)
# ---------------------------------------------------------------------------

def preprocess_cellpose(img_cell):
    """Single-channel FarRed: (H, W), channels=[0, 0]."""
    return img_cell


def preprocess_cellpose_with_nuc(img_cell, img_nuc):
    """FarRed + DAPI: (H, W, 2). channels=[1, 2] → cytoplasm=FarRed, nuc=DAPI."""
    return np.stack([img_cell, img_nuc], axis=-1)


def preprocess_deepcell_cell_only(img_cell):
    """Mesmer with FarRed only: [zeros, FarRed] → (1, H, W, 2)."""
    nuclear = np.zeros_like(img_cell)
    img_2ch = np.stack([nuclear, img_cell], axis=-1)
    return img_2ch[np.newaxis, ...]


def preprocess_deepcell_with_nuc(img_cell, img_nuc):
    """Mesmer with nuclear hint: [DAPI, FarRed] → (1, H, W, 2)."""
    img_2ch = np.stack([img_nuc, img_cell], axis=-1)
    return img_2ch[np.newaxis, ...]


def preprocess_instanseg_1ch(img_cell):
    """InstanSeg single-channel: (H, W) float32."""
    return img_cell.astype(np.float32)


def preprocess_instanseg_2ch(img_cell, img_nuc):
    """InstanSeg two-channel: (2, H, W) float32 with [DAPI, FarRed] order."""
    return np.stack([img_nuc, img_cell], axis=0).astype(np.float32)


def preprocess_cellsam_1ch(img_cell):
    """CellSAM single-channel: (H, W) — format_image_shape maps to blue."""
    return img_cell


def preprocess_cellsam_2ch(img_cell, img_nuc):
    """CellSAM two-channel: (H, W, 2) — maps to green=DAPI, blue=FarRed."""
    return np.stack([img_nuc, img_cell], axis=-1)


# ---------------------------------------------------------------------------
# Inference (only the model call — this is what gets timed)
# ---------------------------------------------------------------------------

def infer_cellpose(model, img_pre, channels):
    masks, _, _, _ = model.eval(
        img_pre, diameter=None, channels=channels,
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks


def infer_deepcell_mesmer(model, img_pre, image_mpp):
    labels = model.predict(img_pre, image_mpp=image_mpp, compartment="whole-cell")
    return labels[0, :, :, 0]


def infer_instanseg(model, img_pre, image_mpp):
    result = model.eval_small_image(
        img_pre, pixel_size=image_mpp, target="cells", return_image_tensor=False,
    )
    return result.squeeze().cpu().numpy()


def infer_cellsam(model, img_pre, gpu):
    from cellSAM import segment_cellular_image
    device = "cuda" if gpu else "cpu"
    mask, _, _ = segment_cellular_image(img_pre, model, device=device)
    return mask


# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------
# TF models first (DeepCell), then PyTorch (Cellpose, InstanSeg).
# Configs sharing a model_key are adjacent so the model loads once.

MODEL_CONFIGS = [
    # --- Single-channel (FarRed only) ---
    {"id": "deepcell_mesmer",         "model_key": "dc_mesmer", "framework": "tensorflow"},
    {"id": "cellpose_cyto2",          "model_key": "cp_cyto2",  "framework": "pytorch"},
    {"id": "cellpose_cyto3",          "model_key": "cp_cyto3",  "framework": "pytorch"},
    {"id": "instanseg_fluorescence",  "model_key": "instanseg", "framework": "pytorch"},
    # --- Two-channel (FarRed + DAPI nuclear hint) ---
    {"id": "deepcell_mesmer_with_nuc",        "model_key": "dc_mesmer", "framework": "tensorflow", "needs_nuc": True},
    {"id": "cellpose_cyto2_with_nuc",         "model_key": "cp_cyto2",  "framework": "pytorch",    "needs_nuc": True},
    {"id": "cellpose_cyto3_with_nuc",         "model_key": "cp_cyto3",  "framework": "pytorch",    "needs_nuc": True},
    {"id": "instanseg_fluorescence_with_nuc", "model_key": "instanseg", "framework": "pytorch",    "needs_nuc": True},
    # --- CellSAM (grouped for single model load) ---
    {"id": "cellsam",                 "model_key": "cellsam", "framework": "pytorch"},
    {"id": "cellsam_with_nuc",        "model_key": "cellsam", "framework": "pytorch", "needs_nuc": True},
]

ALL_MODEL_IDS = [c["id"] for c in MODEL_CONFIGS]

_LOADERS = {
    "dc_mesmer": lambda gpu: load_deepcell_mesmer(),
    "cp_cyto2":  lambda gpu: load_cellpose("cyto2", gpu),
    "cp_cyto3":  lambda gpu: load_cellpose("cyto3", gpu),
    "instanseg": lambda gpu: load_instanseg(),
    "cellsam":   lambda gpu: load_cellsam(),
}


def preprocess(config_id, img_cell, img_nuc=None):
    """Return model-ready input (excluded from timing).

    Args:
        config_id: Model configuration ID.
        img_cell: FarRed cell image (H, W) — primary channel.
        img_nuc: DAPI nuclei image (H, W) — optional nuclear hint.
    """
    if config_id == "deepcell_mesmer":
        return preprocess_deepcell_cell_only(img_cell)
    if config_id == "deepcell_mesmer_with_nuc":
        return preprocess_deepcell_with_nuc(img_cell, img_nuc)
    if config_id == "instanseg_fluorescence":
        return preprocess_instanseg_1ch(img_cell)
    if config_id == "instanseg_fluorescence_with_nuc":
        return preprocess_instanseg_2ch(img_cell, img_nuc)
    if config_id == "cellsam":
        return preprocess_cellsam_1ch(img_cell)
    if config_id == "cellsam_with_nuc":
        return preprocess_cellsam_2ch(img_cell, img_nuc)
    if config_id.endswith("_with_nuc"):
        return preprocess_cellpose_with_nuc(img_cell, img_nuc)
    # Remaining single-channel Cellpose configs
    return preprocess_cellpose(img_cell)


def infer(config_id, model, img_pre, image_mpp, gpu=True):
    """Run the model call only (this is what gets timed)."""
    if config_id.startswith("deepcell_mesmer"):
        return infer_deepcell_mesmer(model, img_pre, image_mpp)
    if config_id.startswith("instanseg_"):
        return infer_instanseg(model, img_pre, image_mpp)
    if config_id.startswith("cellsam"):
        return infer_cellsam(model, img_pre, gpu)
    # All Cellpose configs: +nuc uses channels=[1, 2], others use [0, 0]
    channels = [1, 2] if config_id.endswith("_with_nuc") else [0, 0]
    return infer_cellpose(model, img_pre, channels)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def discover_images(data_dir):
    """Walk category subfolders; return FarRed images as list of dicts.

    Each dict: path, name, category, nuc_path (Path or None).
    DAPI files are expected alongside FarRed with 'wv 390 - Blue' in name.
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


def count_and_areas(labels):
    """Return (cell_count, mean_area_px, median_area_px)."""
    if labels.max() == 0:
        return 0, 0.0, 0.0
    areas = np.bincount(labels.ravel())[1:]
    areas = areas[areas > 0]
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
        description="Cell segmentation benchmark: 10 models x 90 FarRed images",
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Path to image directory (9 category subfolders with FarRed + Blue TIFFs)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: <script_dir>/results)",
    )
    parser.add_argument(
        "--image-mpp", type=float, default=0.325,
        help="Microns per pixel for DeepCell/InstanSeg (default: 0.325)",
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
        parser.error(f"No FarRed .tif images found under {data_dir}")
    n_with_nuc = sum(1 for im in images if im["nuc_path"] is not None)
    print(f"Found {len(images)} FarRed images in {len(categories)} categories")
    print(f"  {n_with_nuc} have matching DAPI nuclear channel")

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
            pass

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
        needs_nuc = cfg.get("needs_nuc", False)

        # Skip images without DAPI if this config needs it
        if needs_nuc:
            cfg_images = [im for im in images if im["nuc_path"] is not None]
            if len(cfg_images) < len(images):
                print(f"{cid}: {len(cfg_images)}/{len(images)} images have DAPI channel")
        else:
            cfg_images = images

        mask_dir = output_dir / "masks" / cid

        # Check which images already have masks (checkpoint)
        existing = set()
        if mask_dir.is_dir():
            existing = {p.name for p in mask_dir.glob("*.tif*")}

        n_cached = sum(1 for im in cfg_images if im["name"] in existing)
        n_new = len(cfg_images) - n_cached

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

                prev = prev_timing.get((cid, img_info["name"]))
                elapsed = prev["inference_time_s"] if prev else float("nan")
                dev = prev["device"] if prev else device_str
            else:
                # --- New: run inference ---
                img_cell = tifffile.imread(img_info["path"])
                img_nuc = None
                if needs_nuc:
                    img_nuc = tifffile.imread(img_info["nuc_path"])

                try:
                    img_pre = preprocess(cid, img_cell, img_nuc)
                    t0 = time.perf_counter()
                    labels = infer(cid, model, img_pre, args.image_mpp, gpu)
                    elapsed = time.perf_counter() - t0
                except Exception as e:
                    tqdm.write(f"  ERROR {cid} on {img_info['name']}: {e}")
                    labels = np.zeros(img_cell.shape[:2], dtype=np.int32)
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
                "cell_count": n,
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
        print("Mean cell count — category x model")
        print(sep)
        pivot = counts_df.pivot_table(
            values="cell_count", index="category",
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
