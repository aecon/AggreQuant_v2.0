"""
Cellpose-based cell segmentation.

Wraps the pre-trained Cellpose cyto2 model for cell segmentation
using both cell and nuclei channels.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np
from typing import Optional

from ..base import BaseSegmenter


# Default parameters
MIN_NUCLEUS_AREA = 300
FLOW_THRESHOLD = 0.4
CELLPROB_THRESHOLD = 0.0


class CellposeSegmenter(BaseSegmenter):
    """
    Cell segmentation using Cellpose.

    Cellpose is a deep learning model for cell segmentation that uses
    both the cell body channel and nuclei information for better results.

    The pipeline:
    1. Create 2-channel input [cells, nuclei_mask]
    2. Run Cellpose eval
    3. Post-processing: Ensure cells contain nuclei
    """

    def __init__(
        self,
        model_type: str = "cyto2",
        gpu: bool = True,
        min_nucleus_area: int = MIN_NUCLEUS_AREA,
        flow_threshold: float = FLOW_THRESHOLD,
        cellprob_threshold: float = CELLPROB_THRESHOLD,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize Cellpose segmenter.

        Arguments:
            model_type: Cellpose model type (default: 'cyto2')
            gpu: Whether to use GPU
            min_nucleus_area: Minimum nucleus area for validation
            flow_threshold: Flow threshold for Cellpose
            cellprob_threshold: Cell probability threshold
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        super().__init__(verbose=verbose, debug=debug)

        self.model_type = model_type
        self.gpu = gpu
        self.min_nucleus_area = min_nucleus_area
        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold

        self._model = None

    @property
    def name(self) -> str:
        return "CellposeSegmenter"

    @property
    def model(self):
        """Lazy load the Cellpose model."""
        if self._model is None:
            self._log(f"Loading Cellpose model: {self.model_type}")
            from cellpose.models import Cellpose
            self._model = Cellpose(gpu=self.gpu, model_type=self.model_type)
        return self._model

    def segment(self, image: np.ndarray, nuclei_labels: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Segment cells in the input image.

        Arguments:
            image: Input cell image (2D grayscale)
            nuclei_labels: Optional nuclei labels for better segmentation

        Returns:
            labels: Instance segmentation labels (uint16)
                    0 = background, 1+ = individual cells
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        if nuclei_labels is None:
            # Segment without nuclei information
            return self._segment_single_channel(image)

        return self._segment_with_nuclei(image, nuclei_labels)

    def segment_with_seeds(
        self,
        image: np.ndarray,
        nuclei_labels: np.ndarray,
        seeds: np.ndarray
    ) -> np.ndarray:
        """
        Segment cells ensuring each cell contains a nucleus.

        Arguments:
            image: Input cell image (2D grayscale)
            nuclei_labels: All nuclei labels
            seeds: Binary mask of non-border nuclei

        Returns:
            labels: Cell labels where each cell contains a nucleus
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Get cell segmentation
        labels = self._segment_with_nuclei(image, nuclei_labels)

        # Add cells where there was a nucleus but no cell detected
        idx = (seeds == 1) & (labels == 0)
        labels[idx] = 1

        # Remove cells without nuclei
        labels = self._exclude_cells_without_nucleus(labels, seeds)

        self._log(f"Final count: {labels.max()} cells")
        return labels.astype(np.uint16)

    def _segment_single_channel(self, image: np.ndarray) -> np.ndarray:
        """Segment using only the cell channel."""
        masks, _, _, _ = self.model.eval(
            image,
            diameter=None,
            channels=[0, 0],  # Grayscale
            resample=True,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            do_3D=False,
        )
        return masks.astype(np.uint16)

    def _segment_with_nuclei(self, image: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
        """Segment using both cell and nuclei channels."""
        # Create nuclei mask
        nuclei_mask = (nuclei_labels > 0).astype(np.float32)

        # Create 2-channel input: [cells, nuclei_mask]
        n = image.shape[0]
        input_image = np.zeros((2, n, n))
        input_image[0, :, :] = image
        input_image[1, :, :] = nuclei_mask

        # Run Cellpose
        # channels=[1,2] means: channel 1 is cytoplasm, channel 2 is nuclei
        masks, _, _, _ = self.model.eval(
            input_image,
            diameter=None,
            channels=[1, 2],
            resample=True,
            flow_threshold=self.flow_threshold,
            cellprob_threshold=self.cellprob_threshold,
            do_3D=False,
        )

        self._debug(f"Cellpose detected {masks.max()} cells")
        return masks

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

            # Cell must have sufficient nucleus area
            if nucleus_area < 0.8 * self.min_nucleus_area:
                labels[idx] = 0
                removed += 1
                self._debug(f"Removed cell {label_id}: nucleus area {nucleus_area}")

        if removed > 0:
            self._log(f"Removed {removed} cells without nuclei")

        return labels
