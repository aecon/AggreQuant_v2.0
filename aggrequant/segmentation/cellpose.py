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
            nuclei_labels: Nuclei labels from nuclei segmentation.
                           Modified in-place: unmatched nuclei (no corresponding
                           cell) are zeroed out to keep nuclei and cells in sync.

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

        # Zero out nuclei with no matched cell, in-place (includes background 0)
        matched_ids = np.unique(labels)
        nuclei_labels[~np.isin(nuclei_labels, matched_ids)] = 0

        self._log(f"Final count: {labels.max()} cells")
        return labels.astype(np.uint16)

    def _segment_with_nuclei(self, image: np.ndarray, nuclei_labels: np.ndarray) -> np.ndarray:
        """Segment using both cell and nuclei channels."""
        # Create nuclei mask
        nuclei_mask = (nuclei_labels > 0).astype(np.float32)

        # Create 2-channel input: [cells, nuclei_mask]
        # https://cellpose.readthedocs.io/en/v3.1.1.1/settings.html#channels
        # "The first channel is the channel you want to segment.
        # The second channel is an optional channel that is helpful
        # in models trained with images with a nucleus channel"
        # https://cellpose.readthedocs.io/en/v3.1.1.1/models.html#cytoplasm-model-cyto3-cyto2-cyto
        # "The cytoplasm models in cellpose are trained on two-channel images,
        # where the first channel is the channel to segment,
        # and the second channel is an optional nuclear channel"
        h, w = image.shape[:2]
        input_image = np.zeros((2, h, w))
        input_image[0, :, :] = image
        input_image[1, :, :] = nuclei_mask

        # Run Cellpose (v3 returns 4 values: masks, flows, styles, diams)
        # channels=[1,2] means: channel 1 is cytoplasm, channel 2 is nuclei
        masks, _, _, _ = self.model.eval(
            input_image,
            channels=[1, 2],
            diameter=None   # for automated diameter estimation
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

        Builds all (cell, nucleus) overlap pairs, scored by the fraction of
        nucleus pixels inside the cell. Pairs are sorted by score descending
        and greedily assigned — each cell and nucleus can only be matched once.
        Unmatched cells (no nucleus) and unmatched nuclei (no cell) are dropped.
        This guarantees an identical number of cells and nuclei in the output.
        """
        output = np.zeros_like(cell_labels)

        # Precompute nucleus areas
        nuc_ids, nuc_areas = np.unique(nuclei_labels, return_counts=True)
        nucleus_area = dict(zip(nuc_ids, nuc_areas))
        nucleus_area.pop(0, None)

        # Build all (score, cell_id, nucleus_id) triples
        triples = []
        for cell_id in np.unique(cell_labels):
            if cell_id == 0:
                continue
            nuclei_in_cell = nuclei_labels[cell_labels == cell_id]
            nuclei_in_cell = nuclei_in_cell[nuclei_in_cell > 0]
            if len(nuclei_in_cell) == 0:
                continue
            for nuc_id, overlap in zip(*np.unique(nuclei_in_cell, return_counts=True)):
                score = overlap / nucleus_area[nuc_id]
                triples.append((score, int(cell_id), int(nuc_id)))

        # Sort by score descending; greedily assign
        triples.sort(key=lambda t: t[0], reverse=True)

        assigned_cells = set()
        assigned_nuclei = set()
        for score, cell_id, nuc_id in triples:
            if cell_id not in assigned_cells and nuc_id not in assigned_nuclei:
                output[cell_labels == cell_id] = nuc_id
                assigned_cells.add(cell_id)
                assigned_nuclei.add(nuc_id)

        n_cells = int(np.sum(np.unique(cell_labels) > 0))
        n_matched = len(assigned_cells)
        if n_cells > n_matched:
            self._debug(f"Dropped {n_cells - n_matched} unmatched cells/nuclei")

        return output



