"""Visualize predictions of a trained model on an image.

Produces a 2-panel figure:
1. Contrast-enhanced raw image (percentile-scaled)
2. Prediction vs ground truth overlay:
   - Yellow  = overlap (TP)
   - Magenta = prediction only (FP)
   - Cyan    = GT only (FN)

Without a ground truth mask, panel 2 shows prediction in green on black.

In interactive mode (--no-save), the panels have linked axes and a
crosshair cursor that appears on both panels simultaneously.

Paths are auto-resolved from the checkpoint location:
- Images from training_output/symlinks/images/
- Masks from training_output/symlinks/masks/ (same filename)
- Output saved next to checkpoint as prediction_<stem>.png

Usage:
    # Minimal — just checkpoint (uses first image, auto-finds mask, auto-saves):
    python scripts/predict_and_plot.py training_output/dice03_bce07_pw3/checkpoints/best.pt

    # Pick a specific image by index or name:
    python scripts/predict_and_plot.py training_output/dice03_bce07_pw3/checkpoints/best.pt --image 3
    python scripts/predict_and_plot.py training_output/dice03_bce07_pw3/checkpoints/best.pt --image image_0003.tif

    # Override any auto-resolved path:
    python scripts/predict_and_plot.py checkpoint.pt --image /path/to/image.tif --mask /path/to/mask.tif -o out.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from aggrequant.common.image_utils import load_image
from aggrequant.nn.inference import load_model, predict

TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training_output"
SYMLINK_DIR = TRAINING_ROOT / "symlinks"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize model predictions on an image"
    )
    parser.add_argument("checkpoint", type=str, help="Path to best.pt checkpoint")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path, filename, or index into symlinks/images/ "
                             "(default: first image)")
    parser.add_argument("--mask", type=str, default=None,
                        help="Mask path (default: auto-resolve from symlinks/masks/)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (default: 0.5)")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path (default: next to checkpoint)")
    parser.add_argument("--no-save", action="store_true",
                        help="Show interactively instead of saving")
    return parser.parse_args()


def resolve_image(image_arg):
    """Resolve image argument to a Path.

    Accepts:
    - None → first image in symlinks/images/
    - Integer string → index into sorted symlinks/images/
    - Filename (no /) → resolve from symlinks/images/
    - Full path → use as-is
    """
    image_dir = SYMLINK_DIR / "images"

    if image_arg is None:
        files = sorted(image_dir.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No .tif files in {image_dir}")
        return files[0]

    # Full/relative path
    p = Path(image_arg)
    if p.exists():
        return p

    # Integer index
    try:
        idx = int(image_arg)
        files = sorted(image_dir.glob("*.tif"))
        if idx < 0 or idx >= len(files):
            raise IndexError(f"Image index {idx} out of range (0-{len(files)-1})")
        return files[idx]
    except ValueError:
        pass

    # Filename in symlinks
    resolved = image_dir / image_arg
    if resolved.exists():
        return resolved

    raise FileNotFoundError(
        f"Cannot resolve image '{image_arg}'. "
        f"Tried: direct path, index into {image_dir}, filename in {image_dir}"
    )


def resolve_mask(mask_arg, image_path):
    """Resolve mask path. Auto-discovers from symlinks/masks/ if not given."""
    if mask_arg is not None:
        p = Path(mask_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Mask not found: {mask_arg}")

    # Auto-resolve: same filename in symlinks/masks/
    auto_mask = SYMLINK_DIR / "masks" / image_path.name
    if auto_mask.exists():
        return auto_mask

    return None


def resolve_output(output_arg, checkpoint_path, image_path):
    """Resolve output path. Defaults to next to the checkpoint."""
    if output_arg is not None:
        return Path(output_arg)

    checkpoint_dir = Path(checkpoint_path).resolve().parent
    run_dir = checkpoint_dir.parent
    return run_dir / f"prediction_{image_path.stem}.png"


def enhance_contrast(image, plow=1.0, phigh=99.0):
    """Scale image to [0, 1] using percentile clipping."""
    vmin = np.percentile(image, plow)
    vmax = np.percentile(image, phigh)
    if vmax <= vmin:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image.astype(np.float32) - vmin) / (vmax - vmin), 0, 1)


def make_comparison_overlay(pred_binary, gt_binary=None):
    """Create RGB overlay of prediction vs ground truth on black background.

    Colors:
        Yellow  = overlap (TP)
        Magenta = prediction only (FP)
        Cyan    = GT only (FN)

    If gt_binary is None, prediction is shown in green.
    """
    h, w = pred_binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    pred = pred_binary.astype(bool)

    if gt_binary is not None:
        gt = gt_binary.astype(bool)
        overlap = pred & gt
        pred_only = pred & ~gt
        gt_only = ~pred & gt

        yellow = np.array([1.0, 1.0, 0.0])
        magenta = np.array([1.0, 0.0, 1.0])
        cyan = np.array([0.0, 1.0, 1.0])

        for mask, color in [(overlap, yellow), (pred_only, magenta), (gt_only, cyan)]:
            if mask.any():
                rgb[mask] = color
    else:
        if pred.any():
            rgb[pred] = np.array([0.0, 1.0, 0.0])

    return rgb


class LinkedCrosshair:
    """Crosshair cursor that appears on all linked axes simultaneously."""

    def __init__(self, axes):
        self.axes = axes
        self.lines = []
        for ax in axes:
            hline = ax.axhline(color="white", linewidth=0.5, alpha=0.6, visible=False)
            vline = ax.axvline(color="white", linewidth=0.5, alpha=0.6, visible=False)
            self.lines.append((hline, vline))

    def on_move(self, event):
        if event.inaxes not in self.axes:
            for hline, vline in self.lines:
                hline.set_visible(False)
                vline.set_visible(False)
            event.canvas.draw_idle()
            return

        for hline, vline in self.lines:
            hline.set_ydata([event.ydata])
            vline.set_xdata([event.xdata])
            hline.set_visible(True)
            vline.set_visible(True)

        event.canvas.draw_idle()


def plot_predictions(image, pred_binary, gt_binary=None,
                     threshold=0.5, save_path=None):
    """Plot contrast-enhanced image and comparison overlay."""
    image_norm = enhance_contrast(image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             sharex=True, sharey=True)

    # Panel 1: Contrast-enhanced image
    axes[0].imshow(image_norm, cmap="gray")
    axes[0].set_title("Input image (contrast-enhanced)")
    axes[0].axis("off")

    # Panel 2: Prediction vs GT overlay
    overlay = make_comparison_overlay(pred_binary, gt_binary)
    axes[1].imshow(overlay)
    axes[1].axis("off")

    if gt_binary is not None:
        axes[1].set_title(
            f"Overlay (p={threshold}) — "
            "yellow=TP, magenta=FP, cyan=FN"
        )
    else:
        axes[1].set_title(f"Prediction (p={threshold})")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    # Interactive: add linked crosshair cursor and show
    crosshair = LinkedCrosshair(axes)
    fig.canvas.mpl_connect("motion_notify_event", crosshair.on_move)
    plt.show()

    plt.close(fig)


def main():
    args = parse_args()

    # Resolve paths
    image_path = resolve_image(args.image)
    mask_path = resolve_mask(args.mask, image_path)
    save_path = None if args.no_save else resolve_output(
        args.output, args.checkpoint, image_path,
    )

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image:      {image_path}")
    print(f"Mask:       {mask_path or '(none)'}")
    print(f"Output:     {save_path or '(interactive)'}")

    # Load model
    model = load_model(args.checkpoint)

    # Load image
    image = load_image(image_path)
    if image.ndim == 3:
        image = image[:, :, 0]

    # Load mask
    gt_binary = None
    if mask_path:
        mask = load_image(mask_path)
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        gt_binary = (mask > 0).astype(np.uint8)

    # Predict
    print("Running inference...")
    prob_map = predict(model, image)

    # Threshold
    pred_binary = (prob_map > args.threshold).astype(np.uint8)
    print(f"Foreground pixels: {pred_binary.sum()} (threshold={args.threshold})")

    # Plot
    plot_predictions(image, pred_binary, gt_binary,
                     threshold=args.threshold, save_path=save_path)


if __name__ == "__main__":
    main()
