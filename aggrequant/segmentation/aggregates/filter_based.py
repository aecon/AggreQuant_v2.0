"""
Filter-based aggregate segmentation.

Classical segmentation using normalized intensity thresholding
and morphological operations.

Author: Athena Economides, 2026, UZH
"""

import numpy as np
import scipy.ndimage
import skimage.morphology
from typing import Optional

from aggrequant.segmentation.base import BaseSegmenter
from aggrequant.common.image_utils import (
    remove_small_holes_compat,
    remove_small_objects_compat,
)


# Default parameters
MEDIAN_FILTER_SIZE = 4
SIGMA_NOISE_REDUCTION = 1
SIGMA_BACKGROUND = 20
INTENSITY_CAP = 3500
NORMALIZED_INTENSITY_THRESHOLD = 1.60
SMALL_HOLE_AREA_THRESHOLD = 6000
MIN_AGGREGATE_AREA = 9


class FilterBasedSegmenter(BaseSegmenter):
    """
    Filter-based aggregate segmentation.

    This classical method uses:
    1. Background normalization (capped intensity / Gaussian blur)
    2. Intensity thresholding on normalized image
    3. Median filtering for noise reduction
    4. Morphological cleanup (hole filling, small object removal)

    Suitable for well-defined, high-contrast aggregates.
    """

    def __init__(
        self,
        median_filter_size: int = MEDIAN_FILTER_SIZE,
        sigma_noise_reduction: float = SIGMA_NOISE_REDUCTION,
        sigma_background: float = SIGMA_BACKGROUND,
        intensity_cap: int = INTENSITY_CAP,
        normalized_threshold: float = NORMALIZED_INTENSITY_THRESHOLD,
        small_hole_area: int = SMALL_HOLE_AREA_THRESHOLD,
        min_aggregate_area: int = MIN_AGGREGATE_AREA,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize filter-based segmenter.

        Arguments:
            median_filter_size: Size of median filter for regularization
            sigma_noise_reduction: Gaussian sigma for noise reduction
            sigma_background: Gaussian sigma for background estimation
            intensity_cap: Maximum intensity for background estimation
            normalized_threshold: Threshold on normalized intensity
            small_hole_area: Area threshold for hole filling
            min_aggregate_area: Minimum aggregate area in pixels
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        super().__init__(verbose=verbose, debug=debug)

        self.median_filter_size = median_filter_size
        self.sigma_noise_reduction = sigma_noise_reduction
        self.sigma_background = sigma_background
        self.intensity_cap = intensity_cap
        self.normalized_threshold = normalized_threshold
        self.small_hole_area = small_hole_area
        self.min_aggregate_area = min_aggregate_area

    @property
    def name(self) -> str:
        return "FilterBasedSegmenter"

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment aggregates in the input image.

        Arguments:
            image: Input grayscale image (2D array)

        Returns:
            labels: Instance segmentation labels (uint32)
                    0 = background, 1+ = individual aggregates
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")
        self._debug(f"Input range: [{image.min()}, {image.max()}], median: {np.median(image):.1f}")

        # Step 1: Background normalization
        normalized = self._normalize_background(image)

        # Step 2: Threshold
        segmented = self._threshold(normalized)

        # Step 3: Median filter for regularization
        segmented = scipy.ndimage.median_filter(segmented, size=self.median_filter_size)

        # Step 4: Connected components
        labels = skimage.morphology.label(segmented, connectivity=2)
        self._debug(f"Initial connected components: {labels.max()}")

        # Step 5: Remove small holes
        no_holes = remove_small_holes_compat(
            segmented, area_threshold=self.small_hole_area, connectivity=2
        )
        labels = skimage.morphology.label(no_holes, connectivity=2)
        self._debug(f"After removing small holes: {labels.max()}")

        # Step 6: Remove small objects
        no_small = remove_small_objects_compat(
            labels, min_size=self.min_aggregate_area, connectivity=2
        )
        labels = skimage.morphology.label(no_small, connectivity=2)
        self._debug(f"After removing small objects: {labels.max()}")

        self._log(f"Detected {labels.max()} aggregates")
        return labels.astype(np.uint32)

    def segment_probability(self, image: np.ndarray) -> np.ndarray:
        """
        Return normalized intensity as a "probability" map.

        Arguments:
            image: Input grayscale image

        Returns:
            probability: Normalized intensity scaled to [0, 1]
        """
        normalized = self._normalize_background(image)

        # Scale to approximate probability
        # Values above threshold -> high probability
        prob = np.clip(normalized / (self.normalized_threshold * 1.5), 0, 1)

        return prob.astype(np.float32)

    def _normalize_background(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image by dividing by estimated background.

        The background is estimated using a large Gaussian blur
        on the capped intensity image.
        """
        img = image.astype(np.float32)

        # Cap intensity for background estimation
        capped = np.clip(img, 0, self.intensity_cap)

        # Estimate background
        background = scipy.ndimage.gaussian_filter(
            capped, sigma=self.sigma_background, mode='reflect'
        )

        # Normalize
        normalized = img / (background + 1e-8)
        assert np.min(normalized) >= 0

        # Noise reduction
        normalized = scipy.ndimage.gaussian_filter(
            normalized, sigma=self.sigma_noise_reduction, mode='reflect'
        )

        self._debug(f"Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
        return normalized

    def _threshold(self, normalized: np.ndarray) -> np.ndarray:
        """Apply threshold to normalized image."""
        segmented = np.zeros(normalized.shape, dtype=np.uint8)
        segmented[normalized > self.normalized_threshold] = 1

        self._debug(f"Threshold {self.normalized_threshold}: {segmented.sum()} pixels")
        return segmented
