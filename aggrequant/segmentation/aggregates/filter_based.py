"""Filter-based aggregate segmentation."""

import numpy as np
import scipy.ndimage
import skimage.filters
import skimage.morphology

from aggrequant.segmentation.base import BaseSegmenter
from aggrequant.segmentation.postprocessing import (
    remove_small_holes,
    remove_small_objects,
)


# Default parameters: Manually tuned based on visual inspection on the validation dataset
MEDIAN_FILTER_SIZE = 4  # applied after segmentation to regularize shape
SIGMA_NOISE_REDUCTION = 1  # applied on normalized data to reduce digitization noise
SIGMA_BACKGROUND = 20  # used to generate a model of background illumination
INTENSITY_CAP = 3500  # used to cap intensity for background estimation
NORMALIZED_INTENSITY_THRESHOLD = 1.60  # threshold on normalized intensity to generate segmentation
SMALL_HOLE_AREA_THRESHOLD = 6000  # fill holes in segmented data that are smaller than this threshold (pixels^2)
MIN_AGGREGATE_AREA = 9  # ignore segmented objects smaller than this threshold


class FilterBasedSegmenter(BaseSegmenter):
    """
    Filter-based aggregate segmentation.

    This classical method uses:
    1. Background normalization (original image / background estimation with capped foreground intensity)
    2. Noise reduction of normalized image
    3. Intensity thresholding of normalized image
    4. Median filtering for noise reduction
    5. Morphological cleanup (hole filling, small object removal)

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
        """
        super().__init__(verbose=verbose)

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
        # Step 1: Background normalization
        normalized = self._normalize_background(image)

        # Step 2: Threshold
        segmented = self._threshold(normalized)

        # Step 3: Median filter for regularization
        segmented = scipy.ndimage.median_filter(segmented, size=self.median_filter_size)

        # Step 4: Remove small holes
        no_holes = remove_small_holes(
            segmented, max_size=self.small_hole_area, connectivity=2
        )

        # Step 5: Connected components - must be done after small hole removal - see recommendation in scikit-image:
        # https://github.com/scikit-image/scikit-image/blob/main/src/skimage/morphology/misc.py#L257
        labels = skimage.morphology.label(no_holes, connectivity=2)

        # Step 6: Remove small objects
        no_small = remove_small_objects(
            labels, max_size=self.min_aggregate_area, connectivity=2
        )

        # Make labeling consecutive using a LUT (fast O(N), no re-labeling)
        unique = np.unique(no_small)  # sorted unique values including 0 (background)
        lut = np.zeros(int(unique.max()) + 1, dtype=np.uint32)
        lut[unique] = np.arange(len(unique), dtype=np.uint32)
        labels = lut[no_small]

        self._log(f"Detected {labels.max()} aggregates")
        return labels.astype(np.uint32)


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
        background = skimage.filters.gaussian(
            capped, sigma=self.sigma_background, mode='reflect', preserve_range=True
        )

        # Normalize
        normalized = img / (background + 1e-8)
        assert np.min(normalized) >= 0

        # Noise reduction
        normalized = skimage.filters.gaussian(
            normalized, sigma=self.sigma_noise_reduction, mode='reflect', preserve_range=True
        )

        return normalized


    def _threshold(self, normalized: np.ndarray) -> np.ndarray:
        """Apply threshold to normalized image."""
        segmented = np.zeros(normalized.shape, dtype=np.uint8)
        segmented[normalized > self.normalized_threshold] = 1

        return segmented



