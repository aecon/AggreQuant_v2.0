"""Figure 2: DL vs filter-based aggregate segmentation on 19 annotated images.

Runs both the baseline UNet (DiceBCE loss) and the filter-based segmenter on
all 19 annotated images, computes per-image and aggregate metrics, and
generates publication-ready comparison figures.

Outputs (saved to --output-dir):
    per_image_metrics.csv      Per-image metrics for both methods
    summary_metrics.csv        Mean +/- std across all images
    metrics_comparison.png     Bar chart with error bars (Panel B)
    size_distributions.png     Aggregate area distributions (Panel C)
    core_edge_errors.png       FP/FN edge vs core breakdown
    overlay_grid.png           TP/FP/FN overlays for selected images

Usage:
    conda run -n AggreQuant python scripts/fig2_dl_vs_filter.py
    conda run -n AggreQuant python scripts/fig2_dl_vs_filter.py -o figures/fig2/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

from aggrequant.common.image_utils import load_image
from aggrequant.common.logging import get_logger
from aggrequant.nn.inference import load_model, predict
from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter

logger = get_logger(__name__)

TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training_output"
SYMLINK_DIR = TRAINING_ROOT / "symlinks"

# Best model: baseline UNet trained with DiceBCE (alpha=0.3, beta=0.7, pw=3.0)
DEFAULT_CHECKPOINT = (
    TRAINING_ROOT / "loss_function" / "dice03_bce07_pw3" / "checkpoints" / "best.pt"
)


# ---------------------------------------------------------------------------
# Metrics (numpy, operate on binary masks)
# ---------------------------------------------------------------------------


def pixel_metrics(pred, gt):
    """Dice, IoU, precision, recall from binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int((pred_b & gt_b).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & gt_b).sum())

    denom_dice = 2 * tp + fp + fn
    denom_iou = tp + fp + fn

    return {
        "dice": 2 * tp / denom_dice if denom_dice > 0 else 0.0,
        "iou": tp / denom_iou if denom_iou > 0 else 0.0,
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def eroded_core_dice(pred, gt, edge_width=3):
    """Dice on GT cores only (each object eroded individually)."""
    gt_b = gt.astype(bool)
    gt_labels, n_objects = ndimage.label(gt_b)

    core_mask = np.zeros_like(gt_b)
    for i in range(1, n_objects + 1):
        obj = gt_labels == i
        eroded = ndimage.binary_erosion(obj, iterations=edge_width)
        core_mask |= eroded if eroded.any() else obj

    if core_mask.sum() == 0:
        return 0.0

    pred_b = pred.astype(bool)
    tp = int((pred_b & core_mask).sum())
    fp = int((pred_b & ~gt_b).sum())
    fn = int((~pred_b & core_mask).sum())
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def object_metrics(pred, gt, match_radius=10.0):
    """Object-level precision/recall via centroid matching."""
    pred_labels, n_pred = ndimage.label(pred.astype(bool))
    gt_labels, n_gt = ndimage.label(gt.astype(bool))

    if n_pred == 0 and n_gt == 0:
        return {"obj_precision": 1.0, "obj_recall": 1.0,
                "n_pred_objects": 0, "n_gt_objects": 0}
    if n_pred == 0:
        return {"obj_precision": 0.0, "obj_recall": 0.0,
                "n_pred_objects": 0, "n_gt_objects": n_gt}
    if n_gt == 0:
        return {"obj_precision": 0.0, "obj_recall": 0.0,
                "n_pred_objects": n_pred, "n_gt_objects": 0}

    pred_centroids = np.array(
        ndimage.center_of_mass(pred, pred_labels, range(1, n_pred + 1))
    )
    gt_centroids = np.array(
        ndimage.center_of_mass(gt, gt_labels, range(1, n_gt + 1))
    )

    gt_matched = set()
    pred_tp = 0
    for pc in pred_centroids:
        dists = np.sqrt(((gt_centroids - pc) ** 2).sum(axis=1))
        nearest = dists.argmin()
        if dists[nearest] <= match_radius and nearest not in gt_matched:
            pred_tp += 1
            gt_matched.add(nearest)

    return {
        "obj_precision": pred_tp / n_pred if n_pred > 0 else 0.0,
        "obj_recall": len(gt_matched) / n_gt if n_gt > 0 else 0.0,
        "n_pred_objects": n_pred,
        "n_gt_objects": n_gt,
    }


def core_edge_errors(pred, gt, edge_width=3):
    """Classify FP/FN pixels as edge (boundary) or core (structural) errors."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    fp_mask = pred_b & ~gt_b
    fn_mask = ~pred_b & gt_b

    dist_to_gt = ndimage.distance_transform_edt(~gt_b)
    dist_to_boundary = ndimage.distance_transform_edt(gt_b)

    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())

    fp_edge = int((fp_mask & (dist_to_gt <= edge_width)).sum())
    fp_core = int((fp_mask & (dist_to_gt > edge_width)).sum())
    fn_edge = int((fn_mask & (dist_to_boundary < edge_width)).sum())
    fn_core = int((fn_mask & (dist_to_boundary >= edge_width)).sum())

    return {
        "n_fp": n_fp, "n_fn": n_fn,
        "fp_edge": fp_edge, "fp_core": fp_core,
        "fn_edge": fn_edge, "fn_core": fn_core,
        "fp_edge_frac": fp_edge / n_fp if n_fp > 0 else 0.0,
        "fp_core_frac": fp_core / n_fp if n_fp > 0 else 0.0,
        "fn_edge_frac": fn_edge / n_fn if n_fn > 0 else 0.0,
        "fn_core_frac": fn_core / n_fn if n_fn > 0 else 0.0,
    }


def aggregate_sizes(binary_mask):
    """Return array of per-object areas (in pixels) from a binary mask."""
    labels, n = ndimage.label(binary_mask.astype(bool))
    if n == 0:
        return np.array([])
    return np.array(ndimage.sum(binary_mask.astype(bool), labels, range(1, n + 1)))


# ---------------------------------------------------------------------------
# Evaluate one image
# ---------------------------------------------------------------------------


def evaluate_image(image, gt_binary, unet_binary, filter_binary,
                   edge_width=3, match_radius=10.0):
    """Compute all metrics for both methods on a single image."""
    results = {}
    for method_name, pred in [("unet", unet_binary), ("filter", filter_binary)]:
        m = {}
        m.update(pixel_metrics(pred, gt_binary))
        m["core_dice"] = eroded_core_dice(pred, gt_binary, edge_width)
        m.update(object_metrics(pred, gt_binary, match_radius))
        m.update(core_edge_errors(pred, gt_binary, edge_width))
        results[method_name] = m
    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_metrics_comparison(summary, output_dir):
    """Bar chart: mean metrics with std error bars, UNet vs Filter."""
    import matplotlib.pyplot as plt

    metrics = [
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("core_dice", "Core Dice"),
        ("obj_precision", "Obj Prec"),
        ("obj_recall", "Obj Recall"),
    ]

    methods = ["unet", "filter"]
    labels = ["UNet", "Filter"]
    colors = ["#2166ac", "#b2182b"]

    n_metrics = len(metrics)
    x = np.arange(n_metrics)
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (method, label, color) in enumerate(zip(methods, labels, colors)):
        means = [summary[method][key]["mean"] for key, _ in metrics]
        stds = [summary[method][key]["std"] for key, _ in metrics]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds,
                      label=label, color=color, alpha=0.85,
                      capsize=3, error_kw={"linewidth": 1})
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("DL vs Filter-Based Segmentation (n=19 images)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_precision_recall_scatter(all_per_image, summary, output_dir):
    """Precision-Recall scatter: per-image points + mean marker.

    Inspired by Jurczenko thesis Fig 18. Each image is a small point,
    the mean is a large marker with size proportional to F1 and color
    encoding IoU.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, ax = plt.subplots(figsize=(7, 6))

    methods = [
        ("unet", "UNet", "o", "#2166ac"),
        ("filter", "Filter", "s", "#b2182b"),
    ]

    for method, label, marker, color in methods:
        recalls = [e[method]["recall"] for e in all_per_image]
        precisions = [e[method]["precision"] for e in all_per_image]

        # Per-image points (small, semi-transparent)
        ax.scatter(recalls, precisions, marker=marker, color=color,
                   alpha=0.35, s=30, edgecolors="none")

        # Mean point (large, with IoU as color and F1 as size)
        mean_rec = summary[method]["recall"]["mean"]
        mean_prec = summary[method]["precision"]["mean"]
        mean_iou = summary[method]["iou"]["mean"]
        mean_f1 = summary[method]["dice"]["mean"]  # Dice = F1

        # Size proportional to F1 (scaled for visibility)
        size = mean_f1 * 500

        sc = ax.scatter([mean_rec], [mean_prec], marker=marker, s=size,
                        c=[mean_iou], cmap="viridis", vmin=0.2, vmax=0.8,
                        edgecolors="black", linewidths=1.5, zorder=5)
        ax.annotate(f"{label}\nF1={mean_f1:.2f}  IoU={mean_iou:.2f}",
                    (mean_rec, mean_prec),
                    textcoords="offset points", xytext=(12, 8),
                    fontsize=8, fontweight="bold")

    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("IoU", fontsize=9)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Precision–Recall Trade-off (n=19 images)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Add reference lines
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
    ax.axvline(0.5, color="gray", linestyle=":", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "precision_recall_scatter.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_size_distributions(all_sizes, output_dir):
    """Violin/box plots of aggregate areas: GT vs UNet vs Filter."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))

    data = []
    labels = []
    colors = []
    for name, color in [("GT", "#4daf4a"), ("UNet", "#2166ac"),
                         ("Filter", "#b2182b")]:
        sizes = all_sizes[name.lower()]
        if len(sizes) > 0:
            data.append(sizes)
            labels.append(f"{name}\n(n={len(sizes)})")
            colors.append(color)

    if not data:
        plt.close(fig)
        return

    parts = ax.violinplot(data, showmedians=True, showmeans=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("black")

    # Add box plot inside
    bp = ax.boxplot(data, widths=0.15, showfliers=False,
                    medianprops={"color": "black", "linewidth": 1.5},
                    boxprops={"linewidth": 0})
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Aggregate area (pixels)")
    ax.set_title("Aggregate size distributions")
    ax.grid(axis="y", alpha=0.3)

    # Add median annotations
    for i, d in enumerate(data):
        median = np.median(d)
        ax.text(i + 1, median, f"  {median:.0f}", va="center", fontsize=8)

    fig.tight_layout()
    path = output_dir / "size_distributions.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_core_edge_errors(totals, output_dir):
    """Stacked bar chart: edge vs core FP/FN breakdown for both methods."""
    import matplotlib.pyplot as plt

    methods = ["unet", "filter"]
    labels = ["UNet", "Filter"]
    x = np.arange(len(methods))
    bar_width = 0.6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # FP breakdown
    fp_edge = [totals[m]["fp_edge"] for m in methods]
    fp_core = [totals[m]["fp_core"] for m in methods]
    bars_edge = ax1.bar(x, fp_edge, bar_width, label="Edge (boundary overshoot)",
                        color="#f4a582")
    bars_core = ax1.bar(x, fp_core, bar_width, bottom=fp_edge,
                        label="Core (hallucinated)", color="#b2182b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Total pixel count (19 images)")
    ax1.set_title("False Positives")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    # Annotate percentages
    for i in range(len(methods)):
        total = fp_edge[i] + fp_core[i]
        if total > 0:
            pct = fp_edge[i] / total * 100
            ax1.text(i, total + total * 0.02,
                     f"{pct:.0f}% edge", ha="center", fontsize=9)

    # FN breakdown
    fn_edge = [totals[m]["fn_edge"] for m in methods]
    fn_core = [totals[m]["fn_core"] for m in methods]
    ax2.bar(x, fn_edge, bar_width, label="Edge (boundary disagreement)",
            color="#92c5de")
    ax2.bar(x, fn_core, bar_width, bottom=fn_edge,
            label="Core (missed aggregate)", color="#2166ac")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Total pixel count (19 images)")
    ax2.set_title("False Negatives")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)
    for i in range(len(methods)):
        total = fn_edge[i] + fn_core[i]
        if total > 0:
            pct = fn_edge[i] / total * 100
            ax2.text(i, total + total * 0.02,
                     f"{pct:.0f}% edge", ha="center", fontsize=9)

    fig.tight_layout()
    path = output_dir / "core_edge_errors.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def plot_overlay_grid(images, gt_masks, unet_preds, filter_preds,
                      image_names, output_dir, n_examples=4):
    """Side-by-side TP/FP/FN overlays for selected images.

    Shows: input | GT | UNet overlay | Filter overlay
    """
    import matplotlib.pyplot as plt

    # Select images spread across the dataset
    n_images = len(images)
    indices = np.linspace(0, n_images - 1, min(n_examples, n_images), dtype=int)

    n_rows = len(indices)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        img = images[idx]
        gt = gt_masks[idx]
        unet = unet_preds[idx]
        filt = filter_preds[idx]

        # Contrast-enhanced input
        p1, p99 = np.percentile(img, (1, 99))
        enhanced = np.clip((img.astype(np.float32) - p1) / max(p99 - p1, 1), 0, 1)

        axes[row, 0].imshow(enhanced, cmap="gray")
        axes[row, 0].set_title(image_names[idx], fontsize=9)

        # GT mask
        axes[row, 1].imshow(gt, cmap="gray")
        axes[row, 1].set_title("Ground truth", fontsize=9)

        # UNet overlay
        axes[row, 2].imshow(_make_overlay(unet, gt))
        axes[row, 2].set_title("UNet", fontsize=9)

        # Filter overlay
        axes[row, 3].imshow(_make_overlay(filt, gt))
        axes[row, 3].set_title("Filter", fontsize=9)

        for c in range(4):
            axes[row, c].axis("off")

    fig.text(0.5, 0.01,
             "Yellow = TP    Magenta = FP    Cyan = FN",
             ha="center", fontsize=10, style="italic")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    path = output_dir / "overlay_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


def _make_overlay(pred_binary, gt_binary):
    """RGB overlay: yellow=TP, magenta=FP, cyan=FN."""
    h, w = pred_binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    rgb[pred & gt] = [1.0, 1.0, 0.0]
    rgb[pred & ~gt] = [1.0, 0.0, 1.0]
    rgb[~pred & gt] = [0.0, 1.0, 1.0]
    return rgb


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def save_per_image_csv(all_per_image, output_path):
    """Save per-image metrics for both methods."""
    if not all_per_image:
        return

    # Collect all metric keys from first entry
    first = all_per_image[0]
    method_keys = sorted(first["unet"].keys())

    fieldnames = ["image"]
    for method in ["unet", "filter"]:
        for key in method_keys:
            fieldnames.append(f"{method}_{key}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in all_per_image:
            row = {"image": entry["image"]}
            for method in ["unet", "filter"]:
                for key in method_keys:
                    row[f"{method}_{key}"] = entry[method].get(key, 0)
            writer.writerow(row)
    logger.info(f"Saved {output_path}")


def save_summary_csv(summary, output_path):
    """Save mean +/- std summary."""
    keys = sorted(summary["unet"].keys())
    fieldnames = ["method"]
    for key in keys:
        fieldnames.extend([f"{key}_mean", f"{key}_std"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method in ["unet", "filter"]:
            row = {"method": method}
            for key in keys:
                row[f"{key}_mean"] = summary[method][key]["mean"]
                row[f"{key}_std"] = summary[method][key]["std"]
            writer.writerow(row)
    logger.info(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Figure 2: DL vs filter-based aggregate segmentation"
    )
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT),
                        help="UNet checkpoint path")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="UNet probability threshold (default: 0.5)")
    parser.add_argument("--edge-width", type=int, default=3,
                        help="Edge band width for core/edge classification")
    parser.add_argument("--match-radius", type=float, default=10.0,
                        help="Max centroid distance for object matching")
    parser.add_argument("-o", "--output-dir", type=str,
                        default="training_output/fig2_dl_vs_filter",
                        help="Output directory")
    parser.add_argument("-n", "--n-overlay", type=int, default=4,
                        help="Number of images in overlay grid")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover images and masks
    image_dir = SYMLINK_DIR / "images"
    mask_dir = SYMLINK_DIR / "masks"
    image_paths = sorted(image_dir.glob("*.tif"))
    logger.info(f"Found {len(image_paths)} images in {image_dir}")

    # Model and segmenter loaded lazily (only if cache misses)
    unet_model = None
    filter_seg = None

    # Process all images
    all_per_image = []
    all_images = []
    all_gt = []
    all_unet = []
    all_filter = []
    all_names = []

    # Accumulated sizes for distribution plot
    all_sizes = {"gt": [], "unet": [], "filter": []}

    # Accumulated core/edge totals
    totals = {
        "unet": {"fp_edge": 0, "fp_core": 0, "fn_edge": 0, "fn_core": 0},
        "filter": {"fp_edge": 0, "fp_core": 0, "fn_edge": 0, "fn_core": 0},
    }

    for image_path in image_paths:
        name = image_path.stem
        mask_path = mask_dir / image_path.name
        if not mask_path.exists():
            logger.warning(f"No mask for {name}, skipping")
            continue

        logger.info(f"Processing {name}...")

        # Load
        image = load_image(image_path)
        if image.ndim == 3:
            image = image[:, :, 0]
        gt = load_image(mask_path)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt_binary = (gt > 0).astype(np.uint8)

        # Check for cached predictions
        cache_dir = output_dir / "masks"
        unet_cache = cache_dir / f"{name}_unet.npy"
        filter_cache = cache_dir / f"{name}_filter.npy"

        if unet_cache.exists() and filter_cache.exists():
            logger.info(f"  Loading cached masks for {name}")
            unet_binary = np.load(unet_cache)
            filter_binary = np.load(filter_cache)
        else:
            # Load models on first cache miss
            if unet_model is None:
                logger.info(f"Loading UNet from {args.checkpoint}")
                unet_model = load_model(args.checkpoint)
            if filter_seg is None:
                filter_seg = FilterBasedSegmenter()

            # UNet prediction
            prob_map = predict(unet_model, image)
            unet_binary = (prob_map > args.threshold).astype(np.uint8)

            # Filter prediction
            filter_labels = filter_seg.segment(image)
            filter_binary = (filter_labels > 0).astype(np.uint8)

            # Cache masks
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(unet_cache, unet_binary)
            np.save(filter_cache, filter_binary)

        # Evaluate
        results = evaluate_image(
            image, gt_binary, unet_binary, filter_binary,
            edge_width=args.edge_width, match_radius=args.match_radius,
        )
        results_entry = {"image": name}
        results_entry.update(results)
        all_per_image.append(results_entry)

        # Accumulate core/edge totals
        for method in ["unet", "filter"]:
            for key in ["fp_edge", "fp_core", "fn_edge", "fn_core"]:
                totals[method][key] += results[method][key]

        # Accumulate sizes
        all_sizes["gt"].append(aggregate_sizes(gt_binary))
        all_sizes["unet"].append(aggregate_sizes(unet_binary))
        all_sizes["filter"].append(aggregate_sizes(filter_binary))

        # Store for overlays
        all_images.append(image)
        all_gt.append(gt_binary)
        all_unet.append(unet_binary)
        all_filter.append(filter_binary)
        all_names.append(name)

        # Log per-image summary
        u = results["unet"]
        f = results["filter"]
        logger.info(
            f"  UNet:   Dice={u['dice']:.3f}  Prec={u['precision']:.3f}  "
            f"Rec={u['recall']:.3f}  ObjP={u['obj_precision']:.3f}"
        )
        logger.info(
            f"  Filter: Dice={f['dice']:.3f}  Prec={f['precision']:.3f}  "
            f"Rec={f['recall']:.3f}  ObjP={f['obj_precision']:.3f}"
        )

    # Concatenate sizes
    for key in all_sizes:
        all_sizes[key] = np.concatenate(all_sizes[key]) if all_sizes[key] else np.array([])

    # Compute summary statistics
    metric_keys = sorted(all_per_image[0]["unet"].keys())
    summary = {}
    for method in ["unet", "filter"]:
        summary[method] = {}
        for key in metric_keys:
            values = [entry[method][key] for entry in all_per_image]
            summary[method][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY (mean +/- std across %d images)", len(all_per_image))
    logger.info("=" * 70)
    header_metrics = ["dice", "iou", "precision", "recall", "core_dice",
                      "obj_precision", "obj_recall"]
    header = f"{'Method':<10}"
    for m in header_metrics:
        header += f"{m:>14}"
    logger.info(header)
    logger.info("-" * len(header))
    for method, label in [("unet", "UNet"), ("filter", "Filter")]:
        line = f"{label:<10}"
        for m in header_metrics:
            mean = summary[method][m]["mean"]
            std = summary[method][m]["std"]
            line += f"{mean:>8.3f}±{std:<4.3f}"
        logger.info(line)
    logger.info("=" * 70)

    # Print core/edge totals
    logger.info("\nCore/Edge error totals:")
    for method, label in [("unet", "UNet"), ("filter", "Filter")]:
        t = totals[method]
        fp_total = t["fp_edge"] + t["fp_core"]
        fn_total = t["fn_edge"] + t["fn_core"]
        logger.info(
            f"  {label}: FP={fp_total:,} ({t['fp_edge']:,} edge, {t['fp_core']:,} core)  "
            f"FN={fn_total:,} ({t['fn_edge']:,} edge, {t['fn_core']:,} core)"
        )

    # Save CSVs
    save_per_image_csv(all_per_image, output_dir / "per_image_metrics.csv")
    save_summary_csv(summary, output_dir / "summary_metrics.csv")

    # Generate plots
    plot_metrics_comparison(summary, output_dir)
    plot_precision_recall_scatter(all_per_image, summary, output_dir)
    plot_size_distributions(all_sizes, output_dir)
    plot_core_edge_errors(totals, output_dir)
    plot_overlay_grid(all_images, all_gt, all_unet, all_filter,
                      all_names, output_dir, n_examples=args.n_overlay)

    logger.info(f"\nAll outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
