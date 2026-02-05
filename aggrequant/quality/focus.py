"""
Focus/blur quality assessment for microscopy images.

This module provides tools to detect blurry regions in microscopy images
using patch-based focus metrics. Blurry regions can be excluded from
downstream analysis to improve data quality.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
Ported from: /home/athena/1_CODES/Vangelis_aSyn_aggregate_detection/bluriness.py
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

from aggrequant.common.logging import get_logger
from aggrequant.common.image_utils import normalize_image, to_uint8

logger = get_logger(__name__)

# Default parameters
DEFAULT_PATCH_SIZE = (40, 40)
DEFAULT_BLUR_THRESHOLD = 15  # for Variance of Laplacian


def _prepare_image_for_cv2(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 for OpenCV operations.

    Uses percentile normalization to handle outliers and avoid
    division by zero for constant or zero images.

    Arguments:
        image: 2D image array (any dtype)

    Returns:
        uint8 image suitable for cv2 functions
    """
    if image.dtype == np.uint8:
        return image

    # Use percentile normalization for robustness against outliers
    normalized = normalize_image(image, method="percentile", percentile_low=1.0, percentile_high=99.0)
    return to_uint8(normalized)


@dataclass
class FocusMetrics:
    """
    Container for focus quality metrics of a single image.

    Attributes:
        variance_laplacian_mean: Mean variance of Laplacian across patches (higher = sharper)
        variance_laplacian_std: Std of variance of Laplacian across patches
        variance_laplacian_min: Minimum variance of Laplacian (worst patch)
        laplace_energy_mean: Mean Laplacian energy across patches
        sobel_mean: Mean Sobel gradient magnitude
        brenner_mean: Mean Brenner focus measure
        focus_score_mean: Mean focus score (variance/mean ratio)
        n_patches_total: Total number of patches analyzed
        n_patches_blurry: Number of patches below blur threshold
        pct_patches_blurry: Percentage of patches that are blurry
        pct_area_blurry: Percentage of image area that is blurry
        is_likely_blurry: True if >50% of image is blurry
    """
    variance_laplacian_mean: float
    variance_laplacian_std: float
    variance_laplacian_min: float
    laplace_energy_mean: float
    sobel_mean: float
    brenner_mean: float
    focus_score_mean: float
    n_patches_total: int
    n_patches_blurry: int
    pct_patches_blurry: float
    pct_area_blurry: float
    is_likely_blurry: bool


# =============================================================================
# Focus Metric Functions
# =============================================================================

def variance_of_laplacian(patch: np.ndarray) -> float:
    """
    Compute variance of Laplacian - primary metric for blur detection.

    Higher values indicate sharper images with more edges.
    This is the most reliable single metric for blur detection.

    Arguments:
        patch: 2D grayscale image patch (uint8 or float)

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

    Arguments:
        patch: 2D grayscale image patch

    Returns:
        Mean gradient magnitude
    """
    patch_float = patch.astype(np.float64)
    sobelx = cv2.Sobel(patch_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch_float, cv2.CV_64F, 0, 1, ksize=3)
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
    patch_float = patch.astype(np.float64)
    diff = patch_float[2:, :] - patch_float[:-2, :]
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
    V = np.nanvar(patch.astype(np.float64))
    M = np.nanmean(patch.astype(np.float64))
    eps = np.finfo(np.float32).eps
    return float(V / (M + eps))


# =============================================================================
# Main Functions
# =============================================================================

def compute_focus_metrics(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    verbose: bool = False,
    debug: bool = False
) -> FocusMetrics:
    """
    Compute focus quality metrics for an image using patch-based analysis.

    Divides the image into non-overlapping patches and computes multiple
    focus metrics for each patch. Returns aggregated statistics.

    Arguments:
        image: 2D grayscale image (uint8 or uint16)
        patch_size: tuple (height, width) for non-overlapping patches
        blur_threshold: threshold for Variance of Laplacian (patches below are blurry)
        verbose: print progress messages
        debug: print detailed debug information

    Returns:
        FocusMetrics dataclass with all computed metrics
    """
    # Ensure image is suitable for processing
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Convert to uint8 for cv2 functions
    img = _prepare_image_for_cv2(image)

    h, w = img.shape
    ph, pw = patch_size

    # Compute grid positions (non-overlapping)
    ys = list(range(0, h - ph + 1, ph))
    xs = list(range(0, w - pw + 1, pw))

    n_patches = len(ys) * len(xs)

    if n_patches == 0:
        raise ValueError(f"Image size {h}x{w} too small for patch size {patch_size}")

    if verbose:
        logger.info(f"Image size: {h}x{w}, patch grid: {len(ys)}x{len(xs)} = {n_patches} patches")

    # Storage for per-patch metrics
    var_lap_values = []
    lap_energy_values = []
    sobel_values = []
    brenner_values = []
    focus_values = []

    # Compute metrics for each patch
    for y in ys:
        for x in xs:
            patch = img[y:y+ph, x:x+pw]

            var_lap_values.append(variance_of_laplacian(patch))
            lap_energy_values.append(laplace_energy(patch))
            sobel_values.append(sobel_metric(patch))
            brenner_values.append(brenner_metric(patch))
            focus_values.append(focus_score(patch))

    # Convert to arrays
    var_lap = np.array(var_lap_values)

    # Count blurry patches (below threshold)
    n_blurry = int(np.sum(var_lap < blur_threshold))
    pct_blurry = n_blurry / n_patches * 100

    # Compute area percentage (accounting for edge pixels not covered by patches)
    total_patch_area = n_patches * ph * pw
    blurry_patch_area = n_blurry * ph * pw
    pct_area_blurry = blurry_patch_area / (h * w) * 100

    if debug:
        logger.debug(f"Variance Laplacian: mean={np.mean(var_lap):.2f}, min={np.min(var_lap):.2f}")
        logger.debug(f"Blurry patches: {n_blurry}/{n_patches} ({pct_blurry:.1f}%)")

    return FocusMetrics(
        variance_laplacian_mean=float(np.mean(var_lap)),
        variance_laplacian_std=float(np.std(var_lap)),
        variance_laplacian_min=float(np.min(var_lap)),
        laplace_energy_mean=float(np.mean(lap_energy_values)),
        sobel_mean=float(np.mean(sobel_values)),
        brenner_mean=float(np.mean(brenner_values)),
        focus_score_mean=float(np.mean(focus_values)),
        n_patches_total=n_patches,
        n_patches_blurry=n_blurry,
        pct_patches_blurry=float(pct_blurry),
        pct_area_blurry=float(pct_area_blurry),
        is_likely_blurry=(pct_blurry > 50),
    )


def generate_blur_mask(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    verbose: bool = False,
    debug: bool = False
) -> np.ndarray:
    """
    Generate a binary mask indicating blurry regions.

    Arguments:
        image: 2D grayscale image
        patch_size: tuple (height, width)
        blur_threshold: threshold for Variance of Laplacian

    Returns:
        blur_mask: boolean array, True where image is blurry
    """
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Convert to uint8 for cv2 functions
    img = _prepare_image_for_cv2(image)

    h, w = img.shape
    ph, pw = patch_size

    blur_mask = np.zeros((h, w), dtype=bool)

    ys = list(range(0, h - ph + 1, ph))
    xs = list(range(0, w - pw + 1, pw))

    n_blurry = 0
    for y in ys:
        for x in xs:
            patch = img[y:y+ph, x:x+pw]
            var_lap = variance_of_laplacian(patch)

            if var_lap < blur_threshold:
                blur_mask[y:y+ph, x:x+pw] = True
                n_blurry += 1

    if verbose:
        n_patches = len(ys) * len(xs)
        logger.info(f"Masked {n_blurry}/{n_patches} patches as blurry")

    return blur_mask


def compute_patch_focus_maps(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
    verbose: bool = False
) -> Tuple[Dict[str, np.ndarray], list, list]:
    """
    Compute focus metric maps for visualization.

    Returns 2D arrays where each element corresponds to a patch.

    Arguments:
        image: 2D grayscale image
        patch_size: tuple (height, width) for patches
        verbose: print progress messages

    Returns:
        maps: dict mapping metric name to 2D score array
        ys: list of patch y coordinates
        xs: list of patch x coordinates
    """
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Convert to uint8 for cv2 functions
    img = _prepare_image_for_cv2(image)

    h, w = img.shape
    ph, pw = patch_size

    ys = list(range(0, h - ph + 1, ph))
    xs = list(range(0, w - pw + 1, pw))

    n_y = len(ys)
    n_x = len(xs)

    if verbose:
        logger.info(f"Computing focus maps: {n_y}x{n_x} patches")

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
            patch = img[y:y+ph, x:x+pw]
            maps["VarianceLaplacian"][i, j] = variance_of_laplacian(patch)
            maps["LaplaceEnergy"][i, j] = laplace_energy(patch)
            maps["Sobel"][i, j] = sobel_metric(patch)
            maps["Brenner"][i, j] = brenner_metric(patch)
            maps["FocusScore"][i, j] = focus_score(patch)

    return maps, ys, xs
