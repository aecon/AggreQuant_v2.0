"""Segmentation pipeline for nuclei, cells, and aggregates."""

from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import tifffile

from aggrequant.loaders.config import PipelineConfig
from aggrequant.loaders.images import ImageLoader, group_files_by_field
from aggrequant.common.image_utils import load_image
from aggrequant.common.logging import get_logger
from aggrequant.common.gpu_utils import configure_tensorflow_memory_growth
from aggrequant.focus import compute_focus_metrics
from aggrequant.colocalization import quantify_field
from aggrequant.segmentation.nuclei.stardist import StarDistSegmenter
from aggrequant.segmentation.cells.cellpose import CellposeSegmenter
from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter
from aggrequant.segmentation.postprocessing import (
    remove_border_objects,
    filter_aggregates_by_cells,
    relabel_consecutive,
)


logger = get_logger(__name__)


class SegmentationPipeline:
    """Pipeline for nuclei, cell, and aggregate segmentation."""

    def __init__(self, config_path: Path, verbose: bool = False):
        """
        Initialize pipeline from config file.

        Arguments:
            config_path: Path to YAML configuration file
            verbose: Enable info-level logging
        """
        self.config = PipelineConfig.from_yaml(config_path)
        self.verbose = verbose

        if verbose:
            import logging
            logging.getLogger("aggrequant").setLevel(logging.INFO)

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
        # Relabeling assumes cells use nuclei as seeds (cell IDs match nucleus IDs).
        # Currently only CellposeSegmenter guarantees this.
        if seg.cell_model not in ("cyto3",):
            raise ValueError(
                f"Unsupported cell model '{seg.cell_model}'. "
                "Only Cellpose models (cyto3) are currently supported."
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

        # Configure GPU before any model loading
        if self.config.use_gpu:
            configure_tensorflow_memory_growth()

        self._plate_name = self.config.plate_name

        # Accumulated results (saved to CSV at end of run)
        self._field_results: List[dict] = []

    def run(self, max_fields: Optional[int] = None):
        """
        Run the segmentation pipeline on all images.

        Arguments:
            max_fields: If set, stop after processing this many fields.
        """
        logger.info(f"Loading images from {self.config.input_dir}")

        loader = ImageLoader(
            directory=self.config.input_dir,
            channel_patterns=self._channel_patterns,
            verbose=self.verbose,
        )

        logger.info(f"Found {loader.n_wells} wells")

        self._field_results = []
        fields_processed = 0
        for well_id in loader.wells:
            well_files = loader.get_well_files(well_id)
            if not well_files:
                logger.info(f"Skipping {well_id}: no files")
                continue

            fields = group_files_by_field(well_files)
            if not fields:
                logger.info(f"Skipping {well_id}: no valid fields")
                continue

            logger.info(f"Processing well {well_id} ({len(fields)} fields)")

            for field_id, field_files in sorted(fields.items()):
                self._process_field(well_id, field_id, field_files)
                fields_processed += 1
                if max_fields is not None and fields_processed >= max_fields:
                    logger.info(f"Reached max_fields={max_fields}, stopping")
                    self._save_results()
                    return

        self._save_results()
        logger.info("Pipeline complete")

    def _process_field(self, well_id: str, field_id: str, field_files: list):
        """Process a single field of view."""
        # Load images by matching channel patterns
        nuclei_img = self._load_channel(field_files, "nuclei")
        cell_img = self._load_channel(field_files, "cells")
        aggregate_img = self._load_channel(field_files, "aggregates")

        if nuclei_img is None or cell_img is None or aggregate_img is None:
            logger.info(f"  Skipping {well_id}/f{field_id}: missing channel(s)")
            return

        logger.info(f"  Processing {well_id}/f{field_id}")

        # Focus quality metrics
        focus_metrics = {}
        quality = self.config.quality
        patch = quality.patch_metrics if quality.compute_patch_metrics else None
        glbl = quality.global_metrics if quality.compute_global_metrics else None
        if "nuclei" in quality.compute_on:
            metrics = compute_focus_metrics(
                nuclei_img, patch_metrics=patch,
                global_metrics=glbl, patch_size=quality.patch_size,
            )
            focus_metrics.update({f"nuclei_{k}": v for k, v in metrics.items()})
        if "cells" in quality.compute_on:
            metrics = compute_focus_metrics(
                cell_img, patch_metrics=patch,
                global_metrics=glbl, patch_size=quality.patch_size,
            )
            focus_metrics.update({f"cells_{k}": v for k, v in metrics.items()})

        # Segment
        nuclei_labels = self._nuclei_segmenter.segment(nuclei_img)
        cell_labels = self._cell_segmenter.segment(cell_img, nuclei_labels)
        aggregate_labels = self._aggregate_segmenter.segment(aggregate_img)

        # Post-processing
        cell_labels, nuclei_labels = remove_border_objects(cell_labels, nuclei_labels)
        aggregate_labels = filter_aggregates_by_cells(aggregate_labels, cell_labels)
        nuclei_labels, cell_labels = relabel_consecutive(nuclei_labels, cell_labels)

        # Quantification
        measurements = quantify_field(
            cell_labels, aggregate_labels, nuclei_labels,
            min_aggregate_area=self.config.segmentation.aggregate_min_size,
        )
        row = {
            "plate_name": self._plate_name,
            "well_id": well_id,
            "field": int(field_id),
            **measurements,
            **focus_metrics,
        }
        self._field_results.append(row)
        logger.info(f"    Quantification: {measurements['n_cells']} cells, "
                    f"{measurements['n_aggregates']} aggregates, "
                    f"{measurements['pct_aggregate_positive_cells']:.1f}% agg-positive")

        # Save masks
        if self.config.output.save_masks:
            self._save_masks(well_id, field_id, nuclei_labels, cell_labels, aggregate_labels)

    def _save_results(self):
        """Save accumulated field measurements (incl. focus metrics) to CSV."""
        if not self._field_results:
            return

        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "field_measurements.csv"

        df = pd.DataFrame(self._field_results)
        df.to_csv(path, index=False)
        logger.info(f"Field measurements saved to {path} ({len(df)} rows)")

    def _load_channel(self, field_files: list, purpose: str) -> Optional[np.ndarray]:
        """Load image for a specific channel purpose."""
        pattern = self._channel_by_purpose.get(purpose)
        if pattern is None:
            return None

        for f in field_files:
            if pattern.lower() in f.name.lower():
                return load_image(f)
        return None

    def _save_masks(
        self,
        well_id: str,
        field_id: str,
        nuclei_labels: np.ndarray,
        cell_labels: np.ndarray,
        aggregate_labels: np.ndarray,
    ):
        """Save segmentation masks to output directory."""
        labels_dir = self.config.output_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_nuclei.tif", nuclei_labels.astype(np.uint16))
        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_cells.tif", cell_labels.astype(np.uint16))
        tifffile.imwrite(labels_dir / f"{well_id}_f{field_id}_aggregates.tif", aggregate_labels.astype(np.uint32))
