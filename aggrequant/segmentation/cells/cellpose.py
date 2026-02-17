"""Cellpose-based cell segmentation."""

import numpy as np

from aggrequant.segmentation.base import BaseSegmenter


class CellposeSegmenter(BaseSegmenter):
    """
    Cell segmentation using Cellpose.

    Cellpose is a deep learning model for cell segmentation that uses
    both the cell body channel and nuclei information for better results.

    The pipeline:
    1. Create 2-channel input [cells, nuclei_mask]
    2. Run Cellpose eval
    3. Match cells to nuclei (relabel cells to match nucleus IDs)
    """

    def __init__(
        self,
        gpu: bool = True,
        model_type: str = "cyto3",
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize Cellpose segmenter.

        Arguments:
            gpu: Whether to use GPU
            model_type: Cellpose model type (e.g., "cyto3", "nuclei")
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        super().__init__(verbose=verbose, debug=debug)
        self.gpu = gpu
        self.model_type = model_type
        self._model = None

    @property
    def name(self) -> str:
        return "CellposeSegmenter"

    @property
    def model(self):
        """Lazy load the Cellpose model."""
        if self._model is None:
            self._log(f"Loading Cellpose model ({self.model_type})")
            from cellpose import models
            self._model = models.Cellpose(gpu=self.gpu, model_type=self.model_type)
        return self._model

    def segment(self, image: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
        """
        Segment cells in the input image.

        Arguments:
            image: Input cell image (2D grayscale)
            nuclei_labels: Nuclei labels from nuclei segmentation

        Returns:
            labels: Instance segmentation labels (uint16)
                    0 = background, 1+ = individual cells
                    Cell labels match their corresponding nucleus labels
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Run Cellpose with nuclei information
        cell_labels = self._segment_with_nuclei(image, nuclei_labels)

        # Match cells to nuclei (relabel cells to match nucleus IDs)
        labels = self._match_cells_to_nuclei(cell_labels, nuclei_labels)

        self._log(f"Final count: {labels.max()} cells")
        return labels.astype(np.uint16)

    def _segment_with_nuclei(self, image: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
        """Segment using both cell and nuclei channels."""
        # Create nuclei mask
        nuclei_mask = (nuclei_labels > 0).astype(np.float32)

        # Create 2-channel input: [cells, nuclei_mask]
        h, w = image.shape[:2]
        input_image = np.zeros((2, h, w))
        input_image[0, :, :] = image
        input_image[1, :, :] = nuclei_mask

        # Run Cellpose (v3 returns 4 values: masks, flows, styles, diams)
        # channels=[1,2] means: channel 1 is cytoplasm, channel 2 is nuclei
        masks, _, _, _ = self.model.eval(
            input_image,
            channels=[1, 2]
        )

        self._debug(f"Cellpose detected {masks.max()} cells")
        return masks

    def _match_cells_to_nuclei(
        self,
        cell_labels: np.ndarray,
        nuclei_labels: np.ndarray
    ) -> np.ndarray:
        """
        Relabel cells to match their corresponding nucleus IDs.

        For each nucleus:
        - Find the cell with most overlap and assign it the nucleus ID
        - If no cell overlaps, use nucleus pixels as the cell
        """
        output = np.zeros_like(cell_labels)

        for nuc_id in np.unique(nuclei_labels):
            if nuc_id == 0:
                continue

            nuc_mask = nuclei_labels == nuc_id

            # Find cells overlapping with this nucleus
            overlapping = cell_labels[nuc_mask]
            overlapping = overlapping[overlapping > 0]

            if len(overlapping) > 0:
                # Use the cell with most overlap
                best_cell = np.bincount(overlapping).argmax()
                output[cell_labels == best_cell] = nuc_id
            else:
                # No cell detected - use nucleus pixels as cell
                output[nuc_mask] = nuc_id
                self._debug(f"No cell for nucleus {nuc_id}, using nucleus mask")

        return output
