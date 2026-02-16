"""
Simple segmentation pipeline for AggreQuant.

Segments nuclei, cells, and aggregates from microscopy images.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import skimage.morphology
import tifffile

from .loaders.config import PipelineConfig
from .loaders.images import ImageLoader, group_files_by_field
from .common.image_utils import load_image
from .quality.focus import compute_patch_focus_maps, compute_global_focus_metrics
from .quantification.measurements import compute_field_measurements
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

        # Build channel pattern mappings
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

        # Accumulated results (saved to CSV at end of run)
        self._focus_results: List[Dict] = []
        self._field_results: List[Dict] = []

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def run(self, max_fields: Optional[int] = None):
        """
        Run the segmentation pipeline on all images.

        Arguments:
            max_fields: If set, stop after processing this many fields.
        """
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

        self._focus_results = []
        fields_processed = 0
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
                fields_processed += 1
                if max_fields is not None and fields_processed >= max_fields:
                    self._log(f"Reached max_fields={max_fields}, stopping")
                    self._save_focus_metrics()
                    self._save_field_measurements()
                    return

        self._save_focus_metrics()
        self._save_field_measurements()
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

        # Focus quality metrics
        quality = self.config.quality
        if "nuclei" in quality.compute_on:
            metrics = self._compute_focus_metrics(nuclei_img, "nuclei")
            self._focus_results.append({"well_id": well_id, "field_id": field_id, "channel": "nuclei", **metrics})
        if "cells" in quality.compute_on:
            metrics = self._compute_focus_metrics(cell_img, "cells")
            self._focus_results.append({"well_id": well_id, "field_id": field_id, "channel": "cells", **metrics})

        # Segment
        nuclei_labels = self._nuclei_segmenter.segment(nuclei_img)
        cell_labels = self._cell_segmenter.segment(cell_img, nuclei_labels)
        aggregate_labels = self._aggregate_segmenter.segment(aggregate_img)

        # Post-processing
        cell_labels, nuclei_labels = self._remove_border_objects(cell_labels, nuclei_labels)
        aggregate_labels = self._filter_aggregates_by_cells(aggregate_labels, cell_labels)
        nuclei_labels, cell_labels = self._relabel_consecutive(nuclei_labels, cell_labels)

        # Quantification
        result, _ = compute_field_measurements(
            cell_labels, aggregate_labels, nuclei_labels,
            min_aggregate_area=self.config.segmentation.aggregate_min_size,
            verbose=self.verbose,
        )
        self._field_results.append({
            "well_id": well_id,
            "field_id": field_id,
            "n_cells": result.n_cells,
            "total_nuclei_area_px": result.total_nuclei_area_px,
            "total_cell_area_px": result.total_cell_area_px,
            "total_aggregate_area_px": result.total_aggregate_area_px,
            "n_aggregates": result.n_aggregates,
            "n_aggregate_positive_cells": result.n_aggregate_positive_cells,
            "pct_aggregate_positive_cells": result.pct_aggregate_positive_cells,
        })
        self._log(f"    Quantification: {result.n_cells} cells, {result.n_aggregates} aggregates, "
                  f"{result.pct_aggregate_positive_cells:.1f}% agg-positive")

        # Save masks
        if self.config.output.save_masks:
            self._save_masks(well_id, field_id, nuclei_labels, cell_labels, aggregate_labels)

    def _compute_focus_metrics(self, image: np.ndarray, channel: str) -> Dict:
        """
        Compute focus quality metrics for a single image.

        Arguments:
            image: 2D grayscale image
            channel: Channel label (for logging)

        Returns:
            Flat dict of metric results with prefixed keys.
        """
        quality = self.config.quality
        results = {}

        # Patch-based metrics
        if quality.compute_patch_metrics and quality.patch_metrics:
            maps, _, _ = compute_patch_focus_maps(image, patch_size=quality.patch_size)
            for metric_name in quality.patch_metrics:
                score_map = maps[metric_name]
                results[f"patch_{metric_name}_mean"] = float(score_map.mean())
                results[f"patch_{metric_name}_min"] = float(score_map.min())
                results[f"patch_{metric_name}_max"] = float(score_map.max())

        # Global metrics
        if quality.compute_global_metrics and quality.global_metrics:
            global_results = compute_global_focus_metrics(image)
            for metric_name in quality.global_metrics:
                results[f"global_{metric_name}"] = global_results[metric_name]

        n_metrics = len(results)
        self._log(f"    Focus ({channel}): {n_metrics} metrics computed")
        return results

    def _save_focus_metrics(self):
        """Save accumulated focus metrics to CSV."""
        if not self._focus_results:
            return

        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "focus_metrics.csv"

        df = pd.DataFrame(self._focus_results)
        df.to_csv(path, index=False)
        self._log(f"Focus metrics saved to {path} ({len(df)} rows)")

    def _save_field_measurements(self):
        """Save accumulated field measurements to CSV."""
        if not self._field_results:
            return

        output_dir = self.config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "field_measurements.csv"

        df = pd.DataFrame(self._field_results)
        df.to_csv(path, index=False)
        self._log(f"Field measurements saved to {path} ({len(df)} rows)")

    def _load_channel(self, field_files: list, purpose: str) -> Optional[np.ndarray]:
        """Load image for a specific channel purpose."""
        pattern = self._channel_by_purpose.get(purpose)
        if pattern is None:
            return None

        for f in field_files:
            if pattern.lower() in f.name.lower():
                return load_image(f)
        return None

    def _remove_border_objects(
        self, cell_labels: np.ndarray, nuclei_labels: np.ndarray
    ) -> tuple:
        """Remove cells touching image border and their corresponding nuclei."""
        cell_labels = cell_labels.copy()
        nuclei_labels = nuclei_labels.copy()

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
        unique_ids = np.unique(nuclei_labels[nuclei_labels > 0])

        # Build lookup table: old_id -> new_id
        max_id = max(nuclei_labels.max(), cell_labels.max())
        lookup = np.zeros(max_id + 1, dtype=nuclei_labels.dtype)
        for new_id, old_id in enumerate(sorted(unique_ids), start=1):
            lookup[old_id] = new_id

        return lookup[nuclei_labels], lookup[cell_labels]

    def _save_masks(
        self,
        well_id: str,
        field_id: str,
        nuclei_labels: np.ndarray,
        cell_labels: np.ndarray,
        aggregate_labels: np.ndarray,
    ):
        """Save segmentation masks to output directory."""
        labels_dir = self.config.output.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_nuclei.tif", nuclei_labels.astype(np.uint16))
        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_cells.tif", cell_labels.astype(np.uint16))
        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_aggregates.tif", aggregate_labels.astype(np.uint32))
