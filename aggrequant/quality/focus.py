"""
Focus/blur quality assessment for microscopy images.

This module provides patch-based focus metrics to detect blurry regions
in microscopy images. Blurry regions can be excluded from downstream
analysis to improve data quality.

All images are internally normalized to 8-bit scale [0, 255] using percentile
normalization (similar to csbdeep.utils.normalize) before computing focus metrics.
This ensures that thresholds (like DEFAULT_BLUR_THRESHOLD) are portable across
different bit depths (8-bit, 12-bit, 16-bit, float, etc.).

Author: Athena Economides, 2026, UZH
"""

import cv2
import numpy as np
from scipy import ndimage
from typing import Tuple, Dict

from aggrequant.common.logging import get_logger

logger = get_logger(__name__)

# Default parameters
DEFAULT_PATCH_SIZE = (80, 80)

# All available patch-level focus metrics (name -> function)
# Populated after function definitions below.
ALL_PATCH_METRICS: Dict[str, callable] = {}
DEFAULT_BLUR_THRESHOLD = 15.0  # for Variance of Laplacian (calibrated for 8-bit scale)

# Percentile normalization parameters (similar to csbdeep defaults)
DEFAULT_PMIN = 1.0
DEFAULT_PMAX = 99.8


# =============================================================================
# Internal Utilities
# =============================================================================

def _normalize_to_8bit(
    image: np.ndarray,
    pmin: float = DEFAULT_PMIN,
    pmax: float = DEFAULT_PMAX,
) -> np.ndarray:
    """
    Normalize image to 8-bit scale [0, 255] using percentile normalization.

    Similar to csbdeep.utils.normalize but scales to [0, 255] instead of [0, 1].
    This ensures focus metric thresholds are portable across different bit depths.

    Arguments:
        image: Input image of any dtype
        pmin: Lower percentile for clipping (default: 1.0)
        pmax: Upper percentile for clipping (default: 99.8)

    Returns:
        Image as float64 in range [0, 255]
    """
    img = image.astype(np.float64)

    p_low = np.percentile(img, pmin)
    p_high = np.percentile(img, pmax)

    if p_high - p_low < 1e-10:
        # Constant or near-constant image - return zeros (no edges/features)
        return np.zeros_like(img, dtype=np.float64)

    # Clip and scale to [0, 255]
    img_clipped = np.clip(img, p_low, p_high)
    return (img_clipped - p_low) / (p_high - p_low) * 255.0


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


# Dispatch table: metric name -> function
ALL_PATCH_METRICS.update({
    "VarianceLaplacian": variance_of_laplacian,
    "LaplaceEnergy": laplace_energy,
    "Sobel": sobel_metric,
    "Brenner": brenner_metric,
    "FocusScore": focus_score,
})


# =============================================================================
# Global Image Quality Metrics (Frequency Domain)
# =============================================================================

def power_log_log_slope(image: np.ndarray) -> float:
    """
    Compute Power Log-Log Slope (PLLS) - frequency domain metric for global defocus.

    This metric computes the slope of the power spectrum on a log-log scale.
    More negative values indicate more blur (high frequencies are lost in blurry images).

    This was identified as the best single metric for focus quality by Bray et al. (2012)
    and is used in CellProfiler's MeasureImageQuality module.

    Reference:
        Bray et al., "Workflow and metrics for image quality control in
        large-scale high-content screens", J Biomol Screen, 2012

    Arguments:
        image: 2D grayscale image (any dtype)

    Returns:
        Slope of log-log power spectrum (typically -1 to -4, more negative = blurrier)
    """
    img = image.astype(np.float64)

    # Compute 2D FFT
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift) ** 2

    # Compute radial average (azimuthal integration)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    # Create radius map
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    # Maximum radius to consider (avoid corners)
    max_r = min(cx, cy)

    # Compute radial mean of power spectrum
    radial_sum = ndimage.sum(magnitude, r, index=np.arange(1, max_r))
    radial_count = ndimage.sum(np.ones_like(magnitude), r, index=np.arange(1, max_r))
    radial_mean = radial_sum / (radial_count + 1e-10)

    # Log-log linear fit (skip DC component at index 0)
    freqs = np.arange(1, len(radial_mean) + 1)
    valid = radial_mean > 0

    if valid.sum() < 10:
        logger.warning("Insufficient valid frequency bins for PLLS computation")
        return 0.0

    log_f = np.log(freqs[valid])
    log_p = np.log(radial_mean[valid])

    # Linear regression: slope of log(power) vs log(frequency)
    slope, _ = np.polyfit(log_f, log_p, 1)

    return float(slope)


def compute_global_focus_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Compute global (whole-image) focus quality metrics.

    These metrics assess the overall focus quality of the entire image,
    complementing patch-based metrics which detect local blur regions.

    Arguments:
        image: 2D grayscale image (any dtype)

    Returns:
        Dictionary with global focus metrics:
            - power_log_log_slope: PLLS value (more negative = blurrier)
            - global_variance_laplacian: VoL computed on full image
            - high_freq_ratio: Ratio of high to low frequency energy
    """
    # Normalize for consistent VoL
    img_norm = _normalize_to_8bit(image)

    # PLLS (on original image, not normalized)
    plls = power_log_log_slope(image)

    # Global Variance of Laplacian
    global_vol = variance_of_laplacian(img_norm)

    # High frequency ratio (simple FFT-based measure)
    f = np.fft.fft2(image.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 4

    y, x = np.ogrid[:h, :w]
    low_mask = ((y - cy) ** 2 + (x - cx) ** 2) < radius ** 2

    low_energy = magnitude[low_mask].sum()
    high_energy = magnitude[~low_mask].sum()
    high_freq_ratio = high_energy / (low_energy + 1e-10)

    return {
        "power_log_log_slope": plls,
        "global_variance_laplacian": global_vol,
        "high_freq_ratio": high_freq_ratio,
    }


# =============================================================================
# Patch-based Analysis
# =============================================================================

def compute_patch_focus_maps(
    image: np.ndarray,
    patch_size: Tuple[int, int] = DEFAULT_PATCH_SIZE,
    metrics: list = None,
) -> Tuple[Dict[str, np.ndarray], list, list]:
    """
    Compute focus metric maps using non-overlapping patches.

    Divides the image into a grid of non-overlapping patches and computes
    the requested focus metrics for each patch.

    The image is internally normalized to 8-bit scale [0, 255] using percentile
    normalization before computing metrics. This ensures thresholds are portable
    across different bit depths.

    Arguments:
        image: 2D grayscale image (any dtype/bit-depth)
        patch_size: tuple (height, width) for patches
        metrics: list of metric names to compute (default: all).
                 Valid names: "VarianceLaplacian", "LaplaceEnergy",
                 "Sobel", "Brenner", "FocusScore".

    Returns:
        maps: dict mapping metric name to 2D score array
        ys: list of patch y coordinates (top-left)
        xs: list of patch x coordinates (top-left)
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")

    # Resolve which metrics to compute
    if metrics is None:
        selected = ALL_PATCH_METRICS
    else:
        unknown = set(metrics) - set(ALL_PATCH_METRICS)
        if unknown:
            raise ValueError(
                f"Unknown metric(s): {unknown}. "
                f"Available: {list(ALL_PATCH_METRICS)}"
            )
        selected = {m: ALL_PATCH_METRICS[m] for m in metrics}

    h, w = image.shape
    ph, pw = patch_size

    # Non-overlapping grid
    ys = list(range(0, h - ph + 1, ph))
    xs = list(range(0, w - pw + 1, pw))

    n_y = len(ys)
    n_x = len(xs)

    if n_y == 0 or n_x == 0:
        raise ValueError(f"Image size {h}x{w} too small for patch size {patch_size}")

    # Normalize entire image to 8-bit scale for consistent metrics
    image_norm = _normalize_to_8bit(image)

    # Initialize maps only for selected metrics
    maps = {name: np.zeros((n_y, n_x)) for name in selected}

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            patch = image_norm[y:y+ph, x:x+pw]
            for name, func in selected.items():
                maps[name][i, j] = func(patch)

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
    maps, ys, xs = compute_patch_focus_maps(image, patch_size, metrics=[metric])

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
