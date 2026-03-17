"""Quantitative comparison of predictions across trained models.

Loads all available checkpoints from training_output/, runs inference on
an image, and computes quantitative metrics comparing each model's
predictions against the ground truth.

Metrics computed:
- Standard pixel-level: Dice, IoU, precision, recall
- Eroded-core Dice: agreement on aggregate centers only (edge-independent)
- Object-level precision/recall: centroid matching (edge-independent)
- FP/FN confidence analysis: probability distributions of errors
- Core vs edge error classification: fraction of errors at boundaries vs cores
- Local intensity analysis: whether errors correlate with dim image regions

Usage:
    # Compare all loss_function runs on first image:
    python scripts/compare_predictions.py

    # Specific image:
    python scripts/compare_predictions.py --image 3

    # Specific runs within the group:
    python scripts/compare_predictions.py --runs baseline bce_pw3 dice03_bce07_pw3

    # Compare ablation runs instead:
    python scripts/compare_predictions.py --group ablation

    # Custom threshold, save results:
    python scripts/compare_predictions.py --threshold 0.4 -o results/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage

from aggrequant.common.image_utils import load_image
from aggrequant.common.logging import get_logger
from aggrequant.nn.inference import load_model, predict

logger = get_logger(__name__)

TRAINING_ROOT = Path(__file__).resolve().parent.parent / "training_output"
SYMLINK_DIR = TRAINING_ROOT / "symlinks"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantitative comparison of model predictions"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Image path, filename, or index (default: first)")
    parser.add_argument("--mask", type=str, default=None,
                        help="GT mask path (default: auto-resolve)")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run names to compare (default: all with best.pt)")
    parser.add_argument("--group", type=str, default="loss_function",
                        help="Subfolder under training_output/ (default: loss_function)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Probability threshold (default: 0.5)")
    parser.add_argument("--edge-width", type=int, default=3,
                        help="Edge band width in pixels for core/edge "
                             "classification (default: 3)")
    parser.add_argument("--match-radius", type=float, default=10.0,
                        help="Max centroid distance for object matching "
                             "(default: 10 pixels)")
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Directory to save CSV and plots (default: print only)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Path resolution (same logic as predict_and_plot.py)
# ---------------------------------------------------------------------------


def resolve_image(image_arg):
    image_dir = SYMLINK_DIR / "images"
    if image_arg is None:
        files = sorted(image_dir.glob("*.tif"))
        if not files:
            raise FileNotFoundError(f"No .tif files in {image_dir}")
        return files[0]
    p = Path(image_arg)
    if p.exists():
        return p
    try:
        idx = int(image_arg)
        files = sorted(image_dir.glob("*.tif"))
        if idx < 0 or idx >= len(files):
            raise IndexError(f"Index {idx} out of range (0-{len(files)-1})")
        return files[idx]
    except ValueError:
        pass
    resolved = image_dir / image_arg
    if resolved.exists():
        return resolved
    raise FileNotFoundError(f"Cannot resolve image '{image_arg}'")


def resolve_mask(mask_arg, image_path):
    if mask_arg is not None:
        p = Path(mask_arg)
        if p.exists():
            return p
        raise FileNotFoundError(f"Mask not found: {mask_arg}")
    auto_mask = SYMLINK_DIR / "masks" / image_path.name
    if auto_mask.exists():
        return auto_mask
    return None


def discover_runs(group, run_names=None):
    """Find all runs with a best.pt checkpoint under a group subfolder."""
    group_dir = TRAINING_ROOT / group

    if run_names:
        runs = {}
        for name in run_names:
            cp = group_dir / name / "checkpoints" / "best.pt"
            if not cp.exists():
                raise FileNotFoundError(f"No checkpoint for run '{name}': {cp}")
            runs[name] = cp
        return runs

    runs = {}
    for cp in sorted(group_dir.glob("*/checkpoints/best.pt")):
        name = cp.parent.parent.name
        runs[name] = cp
    if not runs:
        raise FileNotFoundError(f"No checkpoints found in {group_dir}")
    return runs


# ---------------------------------------------------------------------------
# Pixel-level metrics
# ---------------------------------------------------------------------------


def pixel_metrics(pred, gt):
    """Compute Dice, IoU, precision, recall from binary masks."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    tp = (pred_b & gt_b).sum()
    fp = (pred_b & ~gt_b).sum()
    fn = (~pred_b & gt_b).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def eroded_core_dice(pred, gt, edge_width=3):
    """Dice computed only on GT cores (excluding edges).

    Each GT object is eroded individually. Objects too small to survive
    erosion are included whole (they are entirely "core").
    """
    gt_b = gt.astype(bool)
    gt_labels, n_objects = ndimage.label(gt_b)

    core_mask = np.zeros_like(gt_b)
    for label_id in range(1, n_objects + 1):
        obj_mask = gt_labels == label_id
        eroded = ndimage.binary_erosion(obj_mask, iterations=edge_width)
        if eroded.any():
            core_mask |= eroded
        else:
            core_mask |= obj_mask

    if core_mask.sum() == 0:
        return 0.0

    pred_b = pred.astype(bool)
    tp = (pred_b & core_mask).sum()
    fp_core = (pred_b & ~gt_b).sum()  # FPs still counted normally
    fn_core = (~pred_b & core_mask).sum()

    denom = 2 * tp + fp_core + fn_core
    return 2 * tp / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Object-level centroid matching
# ---------------------------------------------------------------------------


def object_metrics(pred, gt, match_radius=10.0):
    """Object-level precision/recall via centroid matching.

    Each predicted blob is matched to the nearest GT blob within
    match_radius pixels. Unmatched predictions are FP objects,
    unmatched GT blobs are FN objects.
    """
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

    pred_centroids = np.array(ndimage.center_of_mass(
        pred, pred_labels, range(1, n_pred + 1)))
    gt_centroids = np.array(ndimage.center_of_mass(
        gt, gt_labels, range(1, n_gt + 1)))

    # For each predicted centroid, find nearest GT centroid
    gt_matched = set()
    pred_tp = 0
    for pc in pred_centroids:
        dists = np.sqrt(((gt_centroids - pc) ** 2).sum(axis=1))
        nearest_idx = dists.argmin()
        if dists[nearest_idx] <= match_radius and nearest_idx not in gt_matched:
            pred_tp += 1
            gt_matched.add(nearest_idx)

    obj_precision = pred_tp / n_pred if n_pred > 0 else 0.0
    obj_recall = len(gt_matched) / n_gt if n_gt > 0 else 0.0

    return {
        "obj_precision": obj_precision,
        "obj_recall": obj_recall,
        "n_pred_objects": n_pred,
        "n_gt_objects": n_gt,
    }


# ---------------------------------------------------------------------------
# FP/FN confidence analysis
# ---------------------------------------------------------------------------


def confidence_analysis(prob_map, pred, gt):
    """Analyze probability values at FP and FN pixels."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    fp_mask = pred_b & ~gt_b
    fn_mask = ~pred_b & gt_b

    fp_probs = prob_map[fp_mask] if fp_mask.any() else np.array([])
    fn_probs = prob_map[fn_mask] if fn_mask.any() else np.array([])

    return {
        "mean_fp_prob": float(fp_probs.mean()) if len(fp_probs) > 0 else 0.0,
        "median_fp_prob": float(np.median(fp_probs)) if len(fp_probs) > 0 else 0.0,
        "mean_fn_prob": float(fn_probs.mean()) if len(fn_probs) > 0 else 0.0,
        "median_fn_prob": float(np.median(fn_probs)) if len(fn_probs) > 0 else 0.0,
        "fp_high_conf_frac": float((fp_probs > 0.8).mean()) if len(fp_probs) > 0 else 0.0,
        "fn_low_conf_frac": float((fn_probs < 0.2).mean()) if len(fn_probs) > 0 else 0.0,
        "fp_probs": fp_probs,
        "fn_probs": fn_probs,
    }


# ---------------------------------------------------------------------------
# Core vs edge error classification
# ---------------------------------------------------------------------------


def core_edge_errors(pred, gt, edge_width=3):
    """Classify FP/FN pixels as core or edge errors.

    Edge pixels: within edge_width of the GT boundary.
    Core pixels: everything else.

    For FPs:
    - "edge FP": within edge_width of GT foreground (boundary overshoot)
    - "core FP": far from any GT foreground (hallucinated aggregate)

    For FNs:
    - "edge FN": within edge_width of GT boundary (boundary disagreement)
    - "core FN": deep inside GT foreground (missed aggregate core)
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)

    fp_mask = pred_b & ~gt_b
    fn_mask = ~pred_b & gt_b

    # Distance from each pixel to nearest GT foreground pixel
    dist_to_gt = ndimage.distance_transform_edt(~gt_b)

    # Distance from each GT foreground pixel to nearest boundary
    # (= nearest background pixel)
    dist_to_boundary = ndimage.distance_transform_edt(gt_b)

    # FP classification: distance to nearest GT foreground
    fp_near_gt = fp_mask & (dist_to_gt <= edge_width)
    fp_far_from_gt = fp_mask & (dist_to_gt > edge_width)

    # FN classification: distance to GT boundary
    fn_at_edge = fn_mask & (dist_to_boundary < edge_width)
    fn_at_core = fn_mask & (dist_to_boundary >= edge_width)

    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()

    return {
        "fp_edge_frac": float(fp_near_gt.sum() / n_fp) if n_fp > 0 else 0.0,
        "fp_core_frac": float(fp_far_from_gt.sum() / n_fp) if n_fp > 0 else 0.0,
        "fn_edge_frac": float(fn_at_edge.sum() / n_fn) if n_fn > 0 else 0.0,
        "fn_core_frac": float(fn_at_core.sum() / n_fn) if n_fn > 0 else 0.0,
        "n_fp_pixels": int(n_fp),
        "n_fn_pixels": int(n_fn),
        "n_fp_edge": int(fp_near_gt.sum()),
        "n_fp_core": int(fp_far_from_gt.sum()),
        "n_fn_edge": int(fn_at_edge.sum()),
        "n_fn_core": int(fn_at_core.sum()),
    }


# ---------------------------------------------------------------------------
# Local intensity analysis at error components
# ---------------------------------------------------------------------------


def intensity_analysis(image, pred, gt, window=32):
    """Check whether FP/FN objects occur in dim image regions.

    For each FP and FN connected component, measures the local mean
    intensity around its centroid and compares against the intensity
    distribution of GT aggregate regions.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    image_f = image.astype(np.float32)

    # Reference: intensity at GT aggregate pixels
    gt_intensities = image_f[gt_b]
    if len(gt_intensities) == 0:
        gt_p10 = 0.0
    else:
        gt_p10 = np.percentile(gt_intensities, 10)

    h, w = image.shape
    half = window // 2

    def local_mean_intensity(centroids):
        """Mean intensity in a window around each centroid."""
        intensities = []
        for cy, cx in centroids:
            y0 = max(0, int(cy) - half)
            y1 = min(h, int(cy) + half)
            x0 = max(0, int(cx) - half)
            x1 = min(w, int(cx) + half)
            intensities.append(image_f[y0:y1, x0:x1].mean())
        return np.array(intensities)

    # FP components
    fp_mask = pred_b & ~gt_b
    fp_labels, n_fp = ndimage.label(fp_mask)
    fp_in_dim = 0
    if n_fp > 0:
        fp_centroids = ndimage.center_of_mass(fp_mask, fp_labels, range(1, n_fp + 1))
        fp_intensities = local_mean_intensity(fp_centroids)
        fp_in_dim = int((fp_intensities < gt_p10).sum())

    # FN components
    fn_mask = ~pred_b & gt_b
    fn_labels, n_fn = ndimage.label(fn_mask)
    fn_in_dim = 0
    if n_fn > 0:
        fn_centroids = ndimage.center_of_mass(fn_mask, fn_labels, range(1, n_fn + 1))
        fn_intensities = local_mean_intensity(fn_centroids)
        fn_in_dim = int((fn_intensities < gt_p10).sum())

    return {
        "n_fp_components": n_fp,
        "n_fn_components": n_fn,
        "fp_in_dim_region": fp_in_dim,
        "fn_in_dim_region": fn_in_dim,
        "fp_dim_frac": fp_in_dim / n_fp if n_fp > 0 else 0.0,
        "fn_dim_frac": fn_in_dim / n_fn if n_fn > 0 else 0.0,
        "gt_intensity_p10": float(gt_p10),
    }


# ---------------------------------------------------------------------------
# Per-component feature extraction (intensity + sharpness)
# ---------------------------------------------------------------------------


def _component_features(mask, image, window=32):
    """Compute mean intensity and local sharpness for each connected component.

    Returns arrays of shape (n_components,) for intensity and sharpness.
    Sharpness is measured as Laplacian variance in a window around each
    component's centroid.
    """
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.array([]), np.array([])

    image_f = image.astype(np.float64)
    h, w = image.shape
    half = window // 2

    intensities = np.zeros(n)
    sharpnesses = np.zeros(n)

    centroids = ndimage.center_of_mass(mask, labels, range(1, n + 1))

    for i, (cy, cx) in enumerate(centroids):
        # Mean intensity of the component itself
        comp_mask = labels == (i + 1)
        intensities[i] = image_f[comp_mask].mean()

        # Laplacian variance in a window around centroid
        y0 = max(0, int(cy) - half)
        y1 = min(h, int(cy) + half)
        x0 = max(0, int(cx) - half)
        x1 = min(w, int(cx) + half)
        patch = image_f[y0:y1, x0:x1]
        lap = ndimage.laplace(patch)
        sharpnesses[i] = lap.var()

    return intensities, sharpnesses


# ---------------------------------------------------------------------------
# Aggregate all metrics for one model
# ---------------------------------------------------------------------------


def evaluate_model(prob_map, image, gt, threshold=0.5, edge_width=3,
                   match_radius=10.0):
    """Compute all metrics for a single model's prediction."""
    pred_binary = (prob_map > threshold).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    results = {}
    results.update(pixel_metrics(pred_binary, gt_binary))
    results["eroded_core_dice"] = eroded_core_dice(
        pred_binary, gt_binary, edge_width)
    results.update(object_metrics(pred_binary, gt_binary, match_radius))

    conf = confidence_analysis(prob_map, pred_binary, gt_binary)
    # Store arrays separately (not for CSV)
    fp_probs = conf.pop("fp_probs")
    fn_probs = conf.pop("fn_probs")
    results.update(conf)

    results.update(core_edge_errors(pred_binary, gt_binary, edge_width))
    results.update(intensity_analysis(image, pred_binary, gt_binary))

    return results, fp_probs, fn_probs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_summary_table(all_results):
    """Print a formatted comparison table."""
    metrics_order = [
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("precision", "Prec"),
        ("recall", "Recall"),
        ("eroded_core_dice", "Core Dice"),
        ("obj_precision", "Obj Prec"),
        ("obj_recall", "Obj Rec"),
        ("mean_fp_prob", "FP prob"),
        ("mean_fn_prob", "FN prob"),
        ("fp_high_conf_frac", "FP hi-cf"),
        ("fp_core_frac", "FP core%"),
        ("fn_core_frac", "FN core%"),
        ("fp_dim_frac", "FP dim%"),
        ("fn_dim_frac", "FN dim%"),
    ]

    name_width = max(len(name) for name in all_results) + 2
    col_width = 9

    header = f"{'Model':<{name_width}}"
    for _, label in metrics_order:
        header += f"{label:>{col_width}}"
    logger.info("\n" + "=" * len(header))
    logger.info("PREDICTION COMPARISON")
    logger.info("=" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for name, results in all_results.items():
        line = f"{name:<{name_width}}"
        for key, _ in metrics_order:
            val = results.get(key, 0)
            line += f"{val:>{col_width}.4f}"
        logger.info(line)

    logger.info("=" * len(header))


def print_detailed_counts(all_results):
    """Print per-model FP/FN pixel and object counts."""
    counts_order = [
        ("n_gt_objects", "GT obj"),
        ("n_pred_objects", "Pred obj"),
        ("n_fp_pixels", "FP px"),
        ("n_fn_pixels", "FN px"),
        ("n_fp_edge", "FP edge"),
        ("n_fp_core", "FP core"),
        ("n_fn_edge", "FN edge"),
        ("n_fn_core", "FN core"),
        ("n_fp_components", "FP comp"),
        ("n_fn_components", "FN comp"),
        ("fp_in_dim_region", "FP dim"),
        ("fn_in_dim_region", "FN dim"),
    ]

    name_width = max(len(name) for name in all_results) + 2
    col_width = 9

    header = f"{'Model':<{name_width}}"
    for _, label in counts_order:
        header += f"{label:>{col_width}}"
    logger.info("\n" + "-" * len(header))
    logger.info("DETAILED COUNTS")
    logger.info("-" * len(header))
    logger.info(header)
    logger.info("-" * len(header))

    for name, results in all_results.items():
        line = f"{name:<{name_width}}"
        for key, _ in counts_order:
            val = results.get(key, 0)
            line += f"{val:>{col_width}d}"
        logger.info(line)

    logger.info("-" * len(header))


def save_csv(all_results, output_path):
    """Save results to CSV."""
    if not all_results:
        return

    first = next(iter(all_results.values()))
    # Exclude non-numeric fields
    fieldnames = ["model"] + [k for k in first if isinstance(first[k], (int, float))]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for name, results in all_results.items():
            row = {"model": name}
            row.update({k: results[k] for k in fieldnames if k != "model"})
            writer.writerow(row)

    logger.info(f"Results saved to {output_path}")


def save_confidence_histograms(all_fp_probs, all_fn_probs, output_dir):
    """Save FP and FN probability histograms per model."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bins = np.linspace(0, 1, 30)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # FP histogram
    for name, probs in all_fp_probs.items():
        if len(probs) > 0:
            counts, _ = np.histogram(probs, bins=bins, density=True)
            axes[0].bar(bin_centers, counts, width=bins[1] - bins[0],
                        alpha=0.2)
            axes[0].plot(bin_centers, counts, linewidth=1.5, label=name)
    axes[0].set_xlabel("Predicted probability")
    axes[0].set_ylabel("Density")
    axes[0].set_title("False positive confidence")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # FN histogram
    for name, probs in all_fn_probs.items():
        if len(probs) > 0:
            counts, _ = np.histogram(probs, bins=bins, density=True)
            axes[1].bar(bin_centers, counts, width=bins[1] - bins[0],
                        alpha=0.2)
            axes[1].plot(bin_centers, counts, linewidth=1.5, label=name)
    axes[1].set_xlabel("Predicted probability")
    axes[1].set_ylabel("Density")
    axes[1].set_title("False negative confidence")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "confidence_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confidence histograms saved to {path}")


def save_intensity_distributions(image, gt_binary, all_predictions, output_dir):
    """Plot intensity distributions at TP, FP, FN, TN pixels per model.

    Shows where errors fall on the image intensity spectrum relative to
    correct predictions. Each model gets a 4-class density plot.
    """
    import matplotlib.pyplot as plt

    names = list(all_predictions.keys())
    image_f = image.astype(np.float32)
    gt_b = gt_binary.astype(bool)
    n_models = len(names)

    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             sharey=True)
    if n_models == 1:
        axes = [axes]

    # Use image intensity range for bins
    bins = np.linspace(image_f.min(), image_f.max(), 50)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bar_width = bins[1] - bins[0]

    for ax, name in zip(axes, names):
        pred_b = all_predictions[name].astype(bool)

        tp_mask = pred_b & gt_b
        fp_mask = pred_b & ~gt_b
        fn_mask = ~pred_b & gt_b
        tn_mask = ~pred_b & ~gt_b

        categories = [
            ("TP", tp_mask, "#2ca02c"),   # green
            ("FP", fp_mask, "#d62728"),   # red
            ("FN", fn_mask, "#1f77b4"),   # blue
            ("TN", tn_mask, "#999999"),   # gray
        ]

        for label, mask, color in categories:
            pixels = image_f[mask]
            if len(pixels) == 0:
                continue
            counts, _ = np.histogram(pixels, bins=bins, density=True)
            ax.plot(bin_centers, counts, linewidth=1.5, label=label,
                    color=color)

        ax.set_xlabel("Pixel intensity")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Density")
    fig.suptitle("Intensity distributions at TP / FP / FN / TN pixels",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    path = output_dir / "intensity_distributions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Intensity distributions saved to {path}")


def annotation_quality_analysis(image, gt_binary, all_predictions, output_dir):
    """Assess whether FPs/FNs are model errors or annotation errors.

    For each FP, FN, and TP component, computes (intensity, sharpness).
    Components are classified as "aggregate-like" or not based on the TP
    distribution. Produces:
    - Scatter plot of all components in (intensity, sharpness) space
    - Histograms of intensity and sharpness per error class
    - Summary counts table
    """
    import matplotlib.pyplot as plt

    gt_b = gt_binary.astype(bool)
    names = list(all_predictions.keys())

    # Collect per-component features for all models
    all_features = {}
    for name in names:
        pred_b = all_predictions[name].astype(bool)
        tp_mask = pred_b & gt_b
        fp_mask = pred_b & ~gt_b
        fn_mask = ~pred_b & gt_b

        tp_int, tp_sharp = _component_features(tp_mask, image)
        fp_int, fp_sharp = _component_features(fp_mask, image)
        fn_int, fn_sharp = _component_features(fn_mask, image)

        all_features[name] = {
            "tp": (tp_int, tp_sharp),
            "fp": (fp_int, fp_sharp),
            "fn": (fn_int, fn_sharp),
        }

    # --- Scatter plot (one panel per model) ---
    n_models = len(names)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5),
                             sharex=True, sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        tp_int, tp_sharp = all_features[name]["tp"]
        fp_int, fp_sharp = all_features[name]["fp"]
        fn_int, fn_sharp = all_features[name]["fn"]

        if len(tp_int) > 0:
            ax.scatter(tp_int, tp_sharp, c="#2ca02c", alpha=0.3, s=10,
                       label="TP", zorder=1)
        if len(fp_int) > 0:
            ax.scatter(fp_int, fp_sharp, c="#d62728", alpha=0.5, s=15,
                       label="FP", zorder=2)
        if len(fn_int) > 0:
            ax.scatter(fn_int, fn_sharp, c="#1f77b4", alpha=0.5, s=15,
                       label="FN", zorder=2)

        # Draw TP 25th percentile thresholds
        if len(tp_int) > 0:
            int_thresh = np.percentile(tp_int, 25)
            sharp_thresh = np.percentile(tp_sharp, 25)
            ax.axvline(int_thresh, color="gray", linestyle="--", alpha=0.5,
                       linewidth=0.8)
            ax.axhline(sharp_thresh, color="gray", linestyle="--", alpha=0.5,
                       linewidth=0.8)

        ax.set_xlabel("Mean intensity")
        ax.set_title(name, fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    axes[0].set_ylabel("Sharpness (Laplacian var)")
    fig.suptitle(
        "Component features: intensity vs sharpness\n"
        "Dashed lines = TP 25th percentile thresholds",
        fontsize=11, y=1.04)
    fig.tight_layout()
    path = output_dir / "annotation_quality_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Annotation quality scatter saved to {path}")

    # --- Classification + histograms ---
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models),
                             squeeze=False)

    for row, name in enumerate(names):
        tp_int, tp_sharp = all_features[name]["tp"]
        fp_int, fp_sharp = all_features[name]["fp"]
        fn_int, fn_sharp = all_features[name]["fn"]

        if len(tp_int) == 0:
            continue

        int_thresh = np.percentile(tp_int, 25)
        sharp_thresh = np.percentile(tp_sharp, 25)

        # Classify FPs
        fp_agg_like = np.zeros(len(fp_int), dtype=bool)
        if len(fp_int) > 0:
            fp_agg_like = (fp_int >= int_thresh) & (fp_sharp >= sharp_thresh)

        # Classify FNs
        fn_agg_like = np.zeros(len(fn_int), dtype=bool)
        if len(fn_int) > 0:
            fn_agg_like = (fn_int >= int_thresh) & (fn_sharp >= sharp_thresh)

        n_fp_annot_miss = fp_agg_like.sum()
        n_fp_model_err = len(fp_int) - n_fp_annot_miss
        n_fn_annot_err = len(fn_int) - fn_agg_like.sum()
        n_fn_model_miss = fn_agg_like.sum()

        logger.info(
            f"{name}: FP → {n_fp_annot_miss} annotation misses, "
            f"{n_fp_model_err} model hallucinations | "
            f"FN → {n_fn_annot_err} annotation errors, "
            f"{n_fn_model_miss} model misses"
        )

        # Intensity histogram
        ax_int = axes[row, 0]
        bins_int = 40
        if len(tp_int) > 0:
            ax_int.hist(tp_int, bins=bins_int, density=True, alpha=0.3,
                        color="#2ca02c", label="TP")
        if len(fp_int) > 0:
            ax_int.hist(fp_int[fp_agg_like], bins=bins_int, density=True,
                        alpha=0.5, color="#ff7f0e",
                        label=f"FP annot. miss ({n_fp_annot_miss})")
            ax_int.hist(fp_int[~fp_agg_like], bins=bins_int, density=True,
                        alpha=0.5, color="#d62728",
                        label=f"FP hallucination ({n_fp_model_err})")
        if len(fn_int) > 0:
            ax_int.hist(fn_int[~fn_agg_like], bins=bins_int, density=True,
                        alpha=0.5, color="#9467bd",
                        label=f"FN annot. error ({n_fn_annot_err})")
            ax_int.hist(fn_int[fn_agg_like], bins=bins_int, density=True,
                        alpha=0.5, color="#1f77b4",
                        label=f"FN model miss ({n_fn_model_miss})")
        ax_int.axvline(int_thresh, color="gray", linestyle="--", alpha=0.7)
        ax_int.set_xlabel("Mean intensity")
        ax_int.set_ylabel("Density")
        ax_int.set_title(f"{name} — intensity", fontsize=10)
        ax_int.legend(fontsize=7)
        ax_int.grid(True, alpha=0.2)

        # Sharpness histogram
        ax_sh = axes[row, 1]
        bins_sh = 40
        if len(tp_sharp) > 0:
            ax_sh.hist(tp_sharp, bins=bins_sh, density=True, alpha=0.3,
                       color="#2ca02c", label="TP")
        if len(fp_sharp) > 0:
            ax_sh.hist(fp_sharp[fp_agg_like], bins=bins_sh, density=True,
                       alpha=0.5, color="#ff7f0e",
                       label=f"FP annot. miss ({n_fp_annot_miss})")
            ax_sh.hist(fp_sharp[~fp_agg_like], bins=bins_sh, density=True,
                       alpha=0.5, color="#d62728",
                       label=f"FP hallucination ({n_fp_model_err})")
        if len(fn_sharp) > 0:
            ax_sh.hist(fn_sharp[~fn_agg_like], bins=bins_sh, density=True,
                       alpha=0.5, color="#9467bd",
                       label=f"FN annot. error ({n_fn_annot_err})")
            ax_sh.hist(fn_sharp[fn_agg_like], bins=bins_sh, density=True,
                       alpha=0.5, color="#1f77b4",
                       label=f"FN model miss ({n_fn_model_miss})")
        ax_sh.axvline(sharp_thresh, color="gray", linestyle="--", alpha=0.7)
        ax_sh.set_xlabel("Sharpness (Laplacian var)")
        ax_sh.set_ylabel("Density")
        ax_sh.set_title(f"{name} — sharpness", fontsize=10)
        ax_sh.legend(fontsize=7)
        ax_sh.grid(True, alpha=0.2)

    fig.suptitle(
        "Error classification: annotation errors vs model errors\n"
        "Threshold = TP 25th percentile (dashed line)",
        fontsize=11, y=1.02)
    fig.tight_layout()
    path = output_dir / "annotation_quality_histograms.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Annotation quality histograms saved to {path}")


def _make_overlay(pred_binary, gt_binary):
    """RGB overlay: yellow=TP, magenta=FP, cyan=FN."""
    h, w = pred_binary.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)
    rgb[pred & gt] = [1.0, 1.0, 0.0]       # yellow = TP
    rgb[pred & ~gt] = [1.0, 0.0, 1.0]      # magenta = FP
    rgb[~pred & gt] = [0.0, 1.0, 1.0]      # cyan = FN
    return rgb


def plot_metrics_bars(all_results, output_dir):
    """Bar chart comparing key metrics across models."""
    import matplotlib.pyplot as plt

    metrics = [
        ("dice", "Dice"),
        ("iou", "IoU"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("eroded_core_dice", "Core Dice"),
        ("obj_precision", "Obj Prec"),
        ("obj_recall", "Obj Recall"),
    ]
    names = list(all_results.keys())
    n_models = len(names)
    n_metrics = len(metrics)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 3), 6))
    x = np.arange(n_metrics)
    bar_width = 0.8 / n_models

    for i, name in enumerate(names):
        values = [all_results[name].get(key, 0) for key, _ in metrics]
        offset = (i - n_models / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=name, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in metrics])
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "metrics_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Metrics bar chart saved to {path}")


def plot_overlay_grid(image, gt_binary, all_predictions, output_dir):
    """Grid of TP/FP/FN overlay maps, one column per model."""
    import matplotlib.pyplot as plt

    names = list(all_predictions.keys())
    n_cols = len(names) + 1  # first column = input image

    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = [axes]

    # Contrast-enhanced input image
    p1, p99 = np.percentile(image, (1, 99))
    enhanced = np.clip((image.astype(np.float32) - p1) / (p99 - p1), 0, 1)
    axes[0].imshow(enhanced, cmap="gray")
    axes[0].set_title("Input", fontsize=10)
    axes[0].axis("off")

    # Overlay per model
    for i, name in enumerate(names):
        pred = all_predictions[name]
        overlay = _make_overlay(pred, gt_binary)
        axes[i + 1].imshow(overlay)
        axes[i + 1].set_title(name, fontsize=10)
        axes[i + 1].axis("off")

    fig.text(0.5, 0.02, "Yellow = TP    Magenta = FP    Cyan = FN",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = output_dir / "overlay_grid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Overlay grid saved to {path}")


def plot_core_edge_errors(all_results, output_dir):
    """Stacked bar chart: core vs edge breakdown of FP and FN errors."""
    import matplotlib.pyplot as plt

    names = list(all_results.keys())
    n_models = len(names)
    x = np.arange(n_models)
    bar_width = 0.6

    fp_edge = [all_results[n]["n_fp_edge"] for n in names]
    fp_core = [all_results[n]["n_fp_core"] for n in names]
    fn_edge = [all_results[n]["n_fn_edge"] for n in names]
    fn_core = [all_results[n]["n_fn_core"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, n_models * 2), 5))

    # FP breakdown
    ax1.bar(x, fp_edge, bar_width, label="Edge", color="#f4a582")
    ax1.bar(x, fp_core, bar_width, bottom=fp_edge, label="Core", color="#b2182b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right")
    ax1.set_ylabel("Pixel count")
    ax1.set_title("False Positives: core vs edge")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # FN breakdown
    ax2.bar(x, fn_edge, bar_width, label="Edge", color="#92c5de")
    ax2.bar(x, fn_core, bar_width, bottom=fn_edge, label="Core", color="#2166ac")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right")
    ax2.set_ylabel("Pixel count")
    ax2.set_title("False Negatives: core vs edge")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "core_edge_errors.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Core/edge error chart saved to {path}")


def plot_intensity_correlation(all_results, output_dir):
    """Bar chart: fraction of FP/FN errors in dim image regions."""
    import matplotlib.pyplot as plt

    names = list(all_results.keys())
    n_models = len(names)
    x = np.arange(n_models)
    bar_width = 0.35

    fp_dim = [all_results[n]["fp_dim_frac"] for n in names]
    fn_dim = [all_results[n]["fn_dim_frac"] for n in names]

    fig, ax = plt.subplots(figsize=(max(8, n_models * 2), 5))

    bars_fp = ax.bar(x - bar_width / 2, fp_dim, bar_width,
                     label="FP in dim", color="#e08080")
    bars_fn = ax.bar(x + bar_width / 2, fn_dim, bar_width,
                     label="FN in dim", color="#6090c0")

    # Annotate with counts (e.g. "3/12")
    for i, name in enumerate(names):
        r = all_results[name]
        fp_label = f"{r['fp_in_dim_region']}/{r['n_fp_components']}"
        fn_label = f"{r['fn_in_dim_region']}/{r['n_fn_components']}"
        ax.text(bars_fp[i].get_x() + bars_fp[i].get_width() / 2,
                bars_fp[i].get_height() + 0.02,
                fp_label, ha="center", va="bottom", fontsize=7)
        ax.text(bars_fn[i].get_x() + bars_fn[i].get_width() / 2,
                bars_fn[i].get_height() + 0.02,
                fn_label, ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction in dim regions")

    gt_p10 = all_results[names[0]]["gt_intensity_p10"]
    ax.set_title(f"Errors in dim regions (threshold: GT intensity p10 = {gt_p10:.1f})")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = output_dir / "intensity_correlation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Intensity correlation chart saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()

    # Resolve paths
    image_path = resolve_image(args.image)
    mask_path = resolve_mask(args.mask, image_path)
    runs = discover_runs(args.group, args.runs)

    if mask_path is None:
        raise FileNotFoundError(
            "Ground truth mask is required for comparison. "
            "Provide --mask or ensure symlinks/masks/ has a matching file."
        )

    logger.info(f"Image:      {image_path}")
    logger.info(f"Mask:       {mask_path}")
    logger.info(f"Runs:       {list(runs.keys())}")
    logger.info(f"Threshold:  {args.threshold}")
    logger.info(f"Edge width: {args.edge_width}")

    # Load image and GT
    image = load_image(image_path)
    if image.ndim == 3:
        image = image[:, :, 0]

    gt = load_image(mask_path)
    if gt.ndim == 3:
        gt = gt[:, :, 0]

    # Evaluate each model
    gt_binary = (gt > 0).astype(np.uint8)
    all_results = {}
    all_fp_probs = {}
    all_fn_probs = {}
    all_predictions = {}

    for name, checkpoint_path in runs.items():
        logger.info(f"Evaluating {name}...")
        model = load_model(checkpoint_path)
        prob_map = predict(model, image)
        pred_binary = (prob_map > args.threshold).astype(np.uint8)

        results, fp_probs, fn_probs = evaluate_model(
            prob_map, image, gt,
            threshold=args.threshold,
            edge_width=args.edge_width,
            match_radius=args.match_radius,
        )
        all_results[name] = results
        all_fp_probs[name] = fp_probs
        all_fn_probs[name] = fn_probs
        all_predictions[name] = pred_binary

    # Print summary
    print_summary_table(all_results)
    print_detailed_counts(all_results)

    # Save if output dir specified
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_csv(all_results, output_dir / "comparison_results.csv")
        save_confidence_histograms(all_fp_probs, all_fn_probs, output_dir)
        save_intensity_distributions(image, gt_binary, all_predictions,
                                     output_dir)
        plot_metrics_bars(all_results, output_dir)
        plot_overlay_grid(image, gt_binary, all_predictions, output_dir)
        plot_core_edge_errors(all_results, output_dir)
        plot_intensity_correlation(all_results, output_dir)
        annotation_quality_analysis(image, gt_binary, all_predictions,
                                    output_dir)


if __name__ == "__main__":
    main()
