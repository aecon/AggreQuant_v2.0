"""
Simple segmentation pipeline for AggreQuant.

Segments nuclei, cells, and aggregates from microscopy images.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import skimage.morphology
import tifffile

from .loaders.config import PipelineConfig
from .loaders.images import ImageLoader, group_files_by_field, load_tiff
from .segmentation.nuclei.stardist import StarDistSegmenter
from .segmentation.cells.cellpose import CellposeSegmenter
from .segmentation.aggregates.filter_based import FilterBasedSegmenter


class SegmentationPipeline:
    """Pipeline for nuclei, cell, and aggregate segmentation."""

    def __init__(self, config_path: Path, verbose: bool = False):
        """
        Initialize pipeline from config file.

        Arguments:
            config_path: Path to YAML configuration file
            verbose: Print progress messages
        """
        self.config = PipelineConfig.from_yaml(config_path)
        self.verbose = verbose

        # Build channel pattern mapping
        self._channel_patterns = {}
        self._channel_by_purpose = {}
        for ch in self.config.channels:
            self._channel_patterns[ch.name] = ch.pattern
            self._channel_by_purpose[ch.purpose] = ch.pattern

        # Initialize segmenters
        seg = self.config.segmentation
        self._nuclei_segmenter = StarDistSegmenter(
            sigma_denoise=seg.nuclei_sigma_denoise,
            sigma_background=seg.nuclei_sigma_background,
            min_area=seg.nuclei_min_area,
            max_area=seg.nuclei_max_area,
            verbose=verbose,
        )
        self._cell_segmenter = CellposeSegmenter(
            gpu=self.config.use_gpu,
            verbose=verbose,
        )
        self._aggregate_segmenter = FilterBasedSegmenter(
            normalized_threshold=seg.aggregate_intensity_threshold,
            min_aggregate_area=seg.aggregate_min_size,
            verbose=verbose,
        )

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def run(self):
        """Run the segmentation pipeline on all images."""
        # Configure TensorFlow memory growth before any model loading
        import tensorflow as tf
        gpus = tf.config.experimental.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self._log(f"Loading images from {self.config.input_dir}")

        loader = ImageLoader(
            directory=self.config.input_dir,
            channel_patterns=self._channel_patterns,
            verbose=self.verbose,
        )

        self._log(f"Found {loader.n_wells} wells")

        for well_id in loader.wells:
            well_files = loader.get_well_files(well_id)
            if not well_files:
                self._log(f"Skipping {well_id}: no files")
                continue

            fields = group_files_by_field(well_files)
            if not fields:
                self._log(f"Skipping {well_id}: no valid fields")
                continue

            self._log(f"Processing well {well_id} ({len(fields)} fields)")

            for field_id, field_files in sorted(fields.items()):
                self._process_field(well_id, field_id, field_files)

        self._log("Pipeline complete")

    def _process_field(self, well_id: str, field_id: str, field_files: list):
        """Process a single field of view."""
        # Load images by matching channel patterns
        nuclei_img = self._load_channel(field_files, "nuclei")
        cell_img = self._load_channel(field_files, "cells")
        aggregate_img = self._load_channel(field_files, "aggregates")

        if nuclei_img is None or cell_img is None or aggregate_img is None:
            self._log(f"  Skipping {well_id}/f{field_id}: missing channel(s)")
            return

        self._log(f"  Processing {well_id}/f{field_id}")

        # Segment
        nuclei_labels = self._nuclei_segmenter.segment(nuclei_img)
        cell_labels = self._cell_segmenter.segment(cell_img, nuclei_labels)
        aggregate_labels = self._aggregate_segmenter.segment(aggregate_img)

        # Post-processing
        cell_labels, nuclei_labels = self._remove_border_objects(cell_labels, nuclei_labels)
        aggregate_labels = self._filter_aggregates_by_cells(aggregate_labels, cell_labels)
        nuclei_labels, cell_labels = self._relabel_consecutive(nuclei_labels, cell_labels)

        # Save masks
        if self.config.output.save_masks:
            self._save_masks(well_id, field_id, nuclei_labels, cell_labels, aggregate_labels)

    def _load_channel(self, field_files: list, purpose: str) -> Optional[np.ndarray]:
        """Load image for a specific channel purpose."""
        pattern = self._channel_by_purpose.get(purpose)
        if pattern is None:
            return None

        for f in field_files:
            if pattern.lower() in f.name.lower():
                return load_tiff(f)
        return None

    def _remove_border_objects(
        self, cell_labels: np.ndarray, nuclei_labels: np.ndarray
    ) -> tuple:
        """Remove cells touching image border and their corresponding nuclei."""
        # Find labels touching any border
        border_ids = set()
        border_ids.update(np.unique(cell_labels[0, :]))       # top
        border_ids.update(np.unique(cell_labels[-1, :]))      # bottom
        border_ids.update(np.unique(cell_labels[:, 0]))       # left
        border_ids.update(np.unique(cell_labels[:, -1]))      # right
        border_ids.discard(0)

        # Remove from both cell and nuclei labels (they share IDs)
        for label_id in border_ids:
            cell_labels[cell_labels == label_id] = 0
            nuclei_labels[nuclei_labels == label_id] = 0

        return cell_labels, nuclei_labels

    def _filter_aggregates_by_cells(
        self, aggregate_labels: np.ndarray, cell_labels: np.ndarray
    ) -> np.ndarray:
        """Remove aggregates that are outside detected cells."""
        # Create mask of cell regions
        cell_mask = cell_labels > 0

        # Zero out aggregates outside cells
        aggregate_labels = aggregate_labels.copy()
        aggregate_labels[~cell_mask] = 0

        # Relabel to remove gaps from deleted aggregates
        aggregate_labels = skimage.morphology.label(aggregate_labels > 0)

        return aggregate_labels.astype(np.uint32)

    def _relabel_consecutive(
        self, nuclei_labels: np.ndarray, cell_labels: np.ndarray
    ) -> tuple:
        """Relabel nuclei and cells to consecutive IDs, maintaining correspondence."""
        # Get unique non-zero labels (shared between nuclei and cells)
        unique_ids = np.unique(nuclei_labels[nuclei_labels > 0])

        # Create mapping: old_id -> new_id
        id_map = {0: 0}
        for new_id, old_id in enumerate(sorted(unique_ids), start=1):
            id_map[old_id] = new_id

        # Apply mapping
        new_nuclei = np.zeros_like(nuclei_labels)
        new_cells = np.zeros_like(cell_labels)

        for old_id, new_id in id_map.items():
            if old_id == 0:
                continue
            new_nuclei[nuclei_labels == old_id] = new_id
            new_cells[cell_labels == old_id] = new_id

        return new_nuclei, new_cells

    def _save_masks(
        self,
        well_id: str,
        field_id: str,
        nuclei_labels: np.ndarray,
        cell_labels: np.ndarray,
        aggregate_labels: np.ndarray,
    ):
        """Save segmentation masks to output directory."""
        mask_dir = self.config.output.output_dir / well_id
        mask_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(mask_dir / f"f{field_id}_nuclei.tif", nuclei_labels.astype(np.uint16))
        tifffile.imwrite(mask_dir / f"f{field_id}_cells.tif", cell_labels.astype(np.uint16))
        tifffile.imwrite(mask_dir / f"f{field_id}_aggregates.tif", aggregate_labels.astype(np.uint32))
