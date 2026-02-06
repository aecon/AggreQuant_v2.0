"""
Focus/blur quality assessment for microscopy images.

This module provides patch-based focus metrics to detect blurry regions
in microscopy images. Blurry regions can be excluded from downstream
analysis to improve data quality.

Author: Athena Economides, 2026, UZH
"""

import cv2
import numpy as np
from typing import Tuple, Dict

from aggrequant.common.logging import get_logger

logger = get_logger(__name__)

# Default parameters
DEFAULT_PATCH_SIZE = (40, 40)
DEFAULT_BLUR_THRESHOLD = 15.0  # for Variance of Laplacian


# =============================================================================
# Focus Metric Functions
# =============================================================================

def variance_of_laplacian(patch: np.ndarray) -> float:
    """
    Compute variance of Laplacian - primary metric for blur detection.

    Higher values indicate sharper images with more edges.
    This is the most reliable single metric for blur detection.

    We use OpenCV's Laplacian instead of scikit-image because:
    - cv2.Laplacian uses optimized separable filters
    - Better numerical precision with CV_64F output
    - Industry standard for blur detection (Pech-Pacheco et al., 2000)

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Variance of the Laplacian response
    """
    lap = cv2.Laplacian(patch.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def laplace_energy(patch: np.ndarray) -> float:
    """
    Compute Laplacian energy - mean of squared Laplacian values.

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Mean of squared Laplacian values
    """
    lap = cv2.Laplacian(patch.astype(np.float64), cv2.CV_64F)
    return float(np.mean(lap * lap))


def sobel_metric(patch: np.ndarray) -> float:
    """
    Compute Sobel gradient magnitude metric.

    We use OpenCV's Sobel instead of scikit-image because:
    - cv2.Sobel provides separate x/y gradients for proper magnitude calculation
    - Optimized SIMD implementation for better performance
    - Consistent with cv2.Laplacian usage in this module

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Mean gradient magnitude
    """
    patch_f = patch.astype(np.float64)
    sobelx = cv2.Sobel(patch_f, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch_f, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx**2 + sobely**2)
    return float(np.mean(mag))


def brenner_metric(patch: np.ndarray) -> float:
    """
    Compute Brenner's focus measure.

    Sum of squared differences between pixels 2 apart.
    Fast to compute and effective for focus detection.

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Brenner focus measure (sum of squared differences)
    """
    patch_f = patch.astype(np.float64)
    diff = patch_f[2:, :] - patch_f[:-2, :]
    return float(np.sum(diff * diff))


def focus_score(patch: np.ndarray) -> float:
    """
    Compute focus score - variance to mean ratio.

    Also known as coefficient of variation squared.

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Variance divided by mean (with epsilon for stability)
    """
    patch_f = patch.astype(np.float64)
    V = np.nanvar(patch_f)
    M = np.nanmean(patch_f)
    eps = np.finfo(np.float32).eps
    return float(V / (M + eps))


# =============================================================================
# Patch-based Analysis
# =============================================================================

def compute_patch_focus_maps(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
) -> Tuple[Dict[str, np.ndarray], list, list]:
    """
    Compute focus metric maps using non-overlapping patches.

    Divides the image into a grid of non-overlapping patches and computes
    all focus metrics for each patch.

    Arguments:
        image: 2D grayscale image
        patch_size: tuple (height, width) for patches

    Returns:
        maps: dict mapping metric name to 2D score array
        ys: list of patch y coordinates (top-left)
        xs: list of patch x coordinates (top-left)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    h, w = image.shape
    ph, pw = patch_size

    # Non-overlapping grid
    ys = list(range(0, h - ph + 1, ph))
    xs = list(range(0, w - pw + 1, pw))

    n_y = len(ys)
    n_x = len(xs)

    if n_y == 0 or n_x == 0:
        raise ValueError(f"Image size {h}x{w} too small for patch size {patch_size}")

    # Initialize maps
    maps = {
        "VarianceLaplacian": np.zeros((n_y, n_x)),
        "LaplaceEnergy": np.zeros((n_y, n_x)),
        "Sobel": np.zeros((n_y, n_x)),
        "Brenner": np.zeros((n_y, n_x)),
        "FocusScore": np.zeros((n_y, n_x)),
    }

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = image[y:y+ph, x:x+pw]
            maps["VarianceLaplacian"][i, j] = variance_of_laplacian(patch)
            maps["LaplaceEnergy"][i, j] = laplace_energy(patch)
            maps["Sobel"][i, j] = sobel_metric(patch)
            maps["Brenner"][i, j] = brenner_metric(patch)
            maps["FocusScore"][i, j] = focus_score(patch)

    return maps, ys, xs


def generate_blur_mask(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
    threshold: float = DEFAULT_BLUR_THRESHOLD,
    metric: str = "VarianceLaplacian",
) -> np.ndarray:
    """
    Generate a binary mask indicating blurry regions.

    Arguments:
        image: 2D grayscale image
        patch_size: tuple (height, width) for patches
        threshold: patches with metric value below this are marked as blurry
        metric: which metric to use for thresholding
                ("VarianceLaplacian", "LaplaceEnergy", "Sobel", "Brenner", "FocusScore")

    Returns:
        blur_mask: boolean array, True where image is blurry
    """
    maps, ys, xs = compute_patch_focus_maps(image, patch_size)

    if metric not in maps:
        raise ValueError(f"Unknown metric '{metric}'. Available: {list(maps.keys())}")

    score_map = maps[metric]
    blur_flags = score_map < threshold

    h, w = image.shape
    ph, pw = patch_size
    mask = np.zeros((h, w), dtype=bool)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            if blur_flags[i, j]:
                mask[y:y+ph, x:x+pw] = True

    n_blurry = int(blur_flags.sum())
    n_total = blur_flags.size
    logger.debug(f"Masked {n_blurry}/{n_total} patches as blurry ({100*n_blurry/n_total:.1f}%)")

    return mask
