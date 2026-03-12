"""Inference script: run trained mini Cellpose on images.

Usage:
    conda run -n mini_cellpose_AE python predict.py \
        --checkpoint checkpoints/best.pt \
        --image-dir data/images \
        --output-dir results/masks \
        [--cellprob-threshold 0.0] [--niter 200] [--no-gpu]
"""

import argparse
from pathlib import Path

import numpy as np
import tifffile
import torch
from tqdm import tqdm

from model import MiniCellposeUNet
from dynamics import compute_masks


def normalize_image(img):
    """Percentile normalization matching training preprocessing."""
    p_low = np.percentile(img, 1)
    p_high = np.percentile(img, 99.8)
    if p_high - p_low < 1e-6:
        return np.zeros_like(img, dtype=np.float32)
    img = (img.astype(np.float32) - p_low) / (p_high - p_low)
    return np.clip(img, 0, 1)


@torch.no_grad()
def predict_image(model, img, device, cellprob_threshold=0.0, niter=200,
                  min_size=15):
    """Run full inference on a single image.

    Args:
        model: Trained MiniCellposeUNet.
        img: (H, W) uint16 raw image.
        device: Torch device.
        cellprob_threshold: Foreground threshold.
        niter: Euler integration steps.
        min_size: Minimum mask size in pixels.

    Returns:
        masks: (H, W) uint16 label mask.
    """
    # Normalize and convert to tensor
    img_norm = normalize_image(img)
    x = torch.from_numpy(img_norm[None, None]).to(device)  # (1, 1, H, W)

    # Forward pass
    pred, style = model(x)
    pred = pred[0].cpu().numpy()  # (3, H, W)

    # Extract flows and cell probability
    flows = pred[:2]  # (2, H, W)
    cellprob = 1.0 / (1.0 + np.exp(-pred[2]))  # sigmoid

    # Compute masks from flows
    masks = compute_masks(
        flows, cellprob, niter=niter,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size, device=device,
    )

    return masks


def discover_images(data_dir):
    """Walk category subfolders for FarRed images."""
    data_dir = Path(data_dir)
    images = []
    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        for img_path in sorted(category_dir.glob("*wv 631 - FarRed*.tif")):
            images.append({
                "path": img_path,
                "name": img_path.name,
                "category": category_dir.name,
            })
    return images


def main():
    parser = argparse.ArgumentParser(description="Mini Cellpose inference")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="results/masks")
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--niter", type=int, default=200)
    parser.add_argument("--min-size", type=int, default=15)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    # Device
    if args.no_gpu or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")

    # Load model
    model = MiniCellposeUNet(in_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    # Discover images
    images = discover_images(args.image_dir)
    print(f"Found {len(images)} images")

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run inference
    for img_info in tqdm(images, desc="Predicting"):
        out_path = out_dir / img_info["name"]

        # Skip if already computed
        if out_path.exists():
            continue

        img = tifffile.imread(str(img_info["path"]))
        masks = predict_image(
            model, img, device,
            cellprob_threshold=args.cellprob_threshold,
            niter=args.niter,
            min_size=args.min_size,
        )

        tifffile.imwrite(
            str(out_path), masks.astype(np.uint16),
            compression="zlib",
        )

    print(f"Masks saved to {out_dir}/")


if __name__ == "__main__":
    main()
