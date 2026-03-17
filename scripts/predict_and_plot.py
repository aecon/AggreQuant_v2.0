"""Visualize predictions of a trained model on an image.

Shows a 3-panel figure: raw image, probability map, and overlay of
predicted labels on the raw image. Optionally includes ground truth mask.

Usage:
    # Basic — checkpoint + image:
    python scripts/predict_and_plot.py checkpoint.pt image.tif

    # With ground truth mask:
    python scripts/predict_and_plot.py checkpoint.pt image.tif --mask mask.tif

    # Adjust threshold and save:
    python scripts/predict_and_plot.py checkpoint.pt image.tif --threshold 0.4 -o output.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggrequant.common.image_utils import load_image
from aggrequant.nn.inference import load_model, predict, postprocess_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on an image"
    )
    parser.add_argument("checkpoint", type=str, help="Path to best.pt checkpoint")
    parser.add_argument("image", type=str, help="Path to input image (.tif)")
    parser.add_argument("--mask", type=str, default=None,
                        help="Path to ground truth mask (optional)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (default: 0.5)")
    parser.add_argument("--min-area", type=int, default=9,
                        help="Remove objects smaller than this (default: 9)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Save figure to this path (default: show interactively)")
    return parser.parse_args()


def make_label_overlay(image, labels, alpha=0.4):
    """Create an RGBA overlay of colored labels on a grayscale image.

    Arguments:
        image: 2D grayscale image (H, W), will be normalized to [0, 1]
        labels: Instance label map (H, W), 0 = background
        alpha: Overlay opacity for labeled regions

    Returns:
        RGB array (H, W, 3) with labels colored over the grayscale image
    """
    # Normalize image to [0, 1] for display
    img_min, img_max = image.min(), image.max()
    if img_max > img_min:
        img_norm = (image - img_min) / (img_max - img_min)
    else:
        img_norm = np.zeros_like(image, dtype=np.float32)

    # Grayscale to RGB
    rgb = np.stack([img_norm] * 3, axis=-1)

    # Color the labels
    if labels.max() > 0:
        rng = np.random.RandomState(0)
        n_labels = int(labels.max())
        colors = rng.rand(n_labels + 1, 3).astype(np.float32)
        colors[0] = 0  # background stays dark

        mask = labels > 0
        label_rgb = colors[labels]
        rgb[mask] = (1 - alpha) * rgb[mask] + alpha * label_rgb[mask]

    return np.clip(rgb, 0, 1)


def plot_predictions(image, prob_map, labels, mask=None, threshold=0.5,
                     save_path=None):
    """Plot raw image, probability map, and prediction overlay."""
    has_mask = mask is not None
    n_panels = 4 if has_mask else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 1: Raw image
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Input image")
    axes[0].axis("off")

    # Panel 2: Probability map
    im = axes[1].imshow(prob_map, cmap="magma", vmin=0, vmax=1)
    axes[1].set_title("Probability map")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    # Panel 3: Prediction overlay
    overlay = make_label_overlay(image, labels)
    n_objects = int(labels.max())
    axes[2].imshow(overlay)
    axes[2].set_title(f"Predictions ({n_objects} objects, t={threshold})")
    axes[2].axis("off")

    # Panel 4: Ground truth (if provided)
    if has_mask:
        gt_binary = (mask > 0).astype(np.uint32)
        gt_overlay = make_label_overlay(image, gt_binary)
        axes[3].imshow(gt_overlay)
        axes[3].set_title("Ground truth")
        axes[3].axis("off")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    args = parse_args()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = load_model(args.checkpoint)

    # Load image
    print(f"Loading image: {args.image}")
    image = load_image(args.image)
    if image.ndim == 3:
        image = image[:, :, 0]

    # Load mask if provided
    mask = None
    if args.mask:
        print(f"Loading mask: {args.mask}")
        mask = load_image(args.mask)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

    # Predict
    print("Running inference...")
    prob_map = predict(model, image)

    # Postprocess
    labels = postprocess_predictions(
        prob_map, threshold=args.threshold, remove_objects_below=args.min_area,
    )
    n_objects = int(labels.max())
    print(f"Detected {n_objects} objects (threshold={args.threshold})")

    # Plot
    save_path = args.output
    plot_predictions(image, prob_map, labels, mask=mask,
                     threshold=args.threshold, save_path=save_path)


if __name__ == "__main__":
    main()
