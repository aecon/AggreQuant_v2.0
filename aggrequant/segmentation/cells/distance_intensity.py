"""
Distance-intensity watershed cell segmentation.

Classical cell segmentation using a combination of distance transform
and intensity information for watershed segmentation.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np
import skimage.filters
import skimage.morphology
import skimage.exposure
from scipy import ndimage
from skimage.segmentation import watershed
from typing import Tuple

from ..base import BaseSegmenter


# Default parameters
THRESHOLD_NORMALIZED = 1.04
MIN_CELL_INTENSITY = 200
FOREGROUND_SIGMA = 2
BACKGROUND_SIGMA = 50
MAX_INTENSITY_PERCENTILE = 99.8
MAX_DISTANCE_FROM_NUCLEI = 100
CLAHE_FIELD_BLUR_SIGMA = 6
HOLE_AREA_THRESHOLD = 400
MIN_NUCLEUS_AREA = 300


class DistanceIntensitySegmenter(BaseSegmenter):
    """
    Cell segmentation using distance-intensity watershed.

    This method combines:
    1. Intensity-based cell area detection (background normalization + CLAHE)
    2. Distance transform from nuclei
    3. Watershed segmentation using combined field

    Suitable when Cellpose is not available or for comparison.
    """

    def __init__(
        self,
        threshold_normalized: float = THRESHOLD_NORMALIZED,
        min_cell_intensity: int = MIN_CELL_INTENSITY,
        foreground_sigma: float = FOREGROUND_SIGMA,
        background_sigma: float = BACKGROUND_SIGMA,
        max_intensity_percentile: float = MAX_INTENSITY_PERCENTILE,
        max_distance_from_nuclei: int = MAX_DISTANCE_FROM_NUCLEI,
        clahe_blur_sigma: float = CLAHE_FIELD_BLUR_SIGMA,
        hole_area_threshold: int = HOLE_AREA_THRESHOLD,
        min_nucleus_area: int = MIN_NUCLEUS_AREA,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize distance-intensity segmenter.

        Arguments:
            threshold_normalized: Threshold for normalized intensity
            min_cell_intensity: Minimum intensity to consider as cell
            foreground_sigma: Gaussian sigma for foreground denoising
            background_sigma: Gaussian sigma for background estimation
            max_intensity_percentile: Percentile for intensity capping
            max_distance_from_nuclei: Maximum distance from nuclei
            clahe_blur_sigma: Gaussian sigma for CLAHE smoothing
            hole_area_threshold: Minimum hole area to fill
            min_nucleus_area: Minimum nucleus area for validation
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        super().__init__(verbose=verbose, debug=debug)

        self.threshold_normalized = threshold_normalized
        self.min_cell_intensity = min_cell_intensity
        self.foreground_sigma = foreground_sigma
        self.background_sigma = background_sigma
        self.max_intensity_percentile = max_intensity_percentile
        self.max_distance_from_nuclei = max_distance_from_nuclei
        self.clahe_blur_sigma = clahe_blur_sigma
        self.hole_area_threshold = hole_area_threshold
        self.min_nucleus_area = min_nucleus_area

    @property
    def name(self) -> str:
        return "DistanceIntensitySegmenter"

    def segment(
        self,
        image: np.ndarray,
        nuclei_labels: np.ndarray,
        seeds: np.ndarray
    ) -> np.ndarray:
        """
        Segment cells using distance-intensity watershed.

        Arguments:
            image: Input cell image (2D grayscale)
            nuclei_labels: All nuclei labels
            seeds: Binary mask of non-border nuclei

        Returns:
            labels: Instance segmentation labels (uint16)
                    0 = background, 1+ = individual cells
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Convert to float
        img = image.astype(np.float32)

        # Step 1: Generate cell area mask
        cell_mask, scaled_clahe = self._generate_cell_mask(img)

        # Create nuclei mask
        nuclei_mask = (nuclei_labels > 0).astype(np.uint8)

        # Extend cell mask to include nuclei seeds
        cell_mask[seeds == 1] = True

        # Step 2: Combine intensity and distance fields
        field = self._create_watershed_field(scaled_clahe, nuclei_mask)

        # Step 3: Watershed segmentation
        labels = watershed(field, mask=cell_mask, watershed_line=True)
        labels = labels.astype(np.uint16)

        self._debug(f"Watershed detected {labels.max()} cells")

        # Step 4: Remove cells without nuclei
        labels = self._exclude_cells_without_nucleus(labels, seeds)

        self._log(f"Final count: {labels.max()} cells")
        return labels

    def _generate_cell_mask(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate binary mask of cell areas.

        Returns:
            cell_mask: Binary mask of cell regions
            scaled_clahe: CLAHE-enhanced image for watershed field
        """
        # Calculate cap threshold
        cap_threshold = np.percentile(image, self.max_intensity_percentile)
        self._debug(f"Signal max intensity: {cap_threshold:.1f}")

        # Denoise
        denoised = skimage.filters.gaussian(image, sigma=self.foreground_sigma)

        # Cap foreground for background estimation
        capped = image.copy()
        capped[image > cap_threshold] = cap_threshold

        # Estimate background
        background = skimage.filters.gaussian(capped, sigma=self.background_sigma)

        # Normalize
        normalized = denoised / (background + 1e-8)

        # Scale to [0, 1]
        scaled, scale = self._scale_values_01(normalized)

        # Apply CLAHE
        scaled_clahe = skimage.exposure.equalize_adapthist(scaled, kernel_size=150)

        # Ignore low intensity pixels
        scaled_clahe[denoised < self.min_cell_intensity] = 0

        # Rescale CLAHE output
        imin, imax = scale
        cells_area = scaled_clahe * (imax - imin) + imin

        # Threshold
        cell_mask = cells_area > self.threshold_normalized

        # Remove small holes
        cell_mask = skimage.morphology.remove_small_holes(
            cell_mask, area_threshold=self.hole_area_threshold
        )

        return cell_mask, scaled_clahe

    def _create_watershed_field(
        self,
        scaled_clahe: np.ndarray,
        nuclei_mask: np.ndarray
    ) -> np.ndarray:
        """
        Create combined intensity-distance field for watershed.

        The field combines:
        - Inverted intensity (cells are valleys)
        - Distance from nuclei (farther = higher)
        """
        # Intensity field (smoothed)
        intensity_field = skimage.filters.gaussian(scaled_clahe, sigma=self.clahe_blur_sigma)

        # Invert and scale intensity
        inv_intensity, _ = self._scale_values_01(intensity_field)
        inv_intensity = 1.0 - inv_intensity
        inv_intensity[nuclei_mask == 1] = 0

        # Distance transform
        distances = ndimage.distance_transform_edt(1 - nuclei_mask)
        distances[distances >= self.max_distance_from_nuclei] = 0

        # Combined field: inverted intensity * distance
        field = inv_intensity * distances

        return field

    def _scale_values_01(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Scale image values to [0, 1] range."""
        imin = np.min(image)
        imax = np.max(image)

        if imax - imin < 1e-8:
            return np.zeros_like(image), (imin, imax)

        scaled = (image - imin) / (imax - imin)
        return scaled.astype(np.float32), (imin, imax)

    def _exclude_cells_without_nucleus(
        self,
        labels: np.ndarray,
        seeds: np.ndarray
    ) -> np.ndarray:
        """Remove cells that don't contain a nucleus."""
        all_labels = np.unique(labels[labels > 0])
        removed = 0

        for label_id in all_labels:
            idx = labels == label_id
            nucleus_area = np.sum(seeds[idx])

            if nucleus_area < 0.8 * self.min_nucleus_area:
                labels[idx] = 0
                removed += 1
                self._debug(f"Removed cell {label_id}: nucleus area {nucleus_area}")

        if removed > 0:
            self._log(f"Removed {removed} cells without nuclei")

        return labels
