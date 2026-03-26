"""Segmentation pipeline for nuclei, cells, and aggregates."""

from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd
import tifffile

from aggrequant.loaders.config import PipelineConfig
from aggrequant.loaders.images import FieldTriplet, build_field_triplets
from aggrequant.common.image_utils import load_image
from aggrequant.common.logging import get_logger
from aggrequant.common.gpu_utils import configure_tensorflow_memory_growth
from aggrequant.focus import compute_focus_metrics
from aggrequant.colocalization import quantify_field
from aggrequant.segmentation.stardist import StarDistSegmenter
from aggrequant.segmentation.cellpose import CellposeSegmenter
from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter
from aggrequant.segmentation.aggregates.neural_network import NeuralNetworkSegmenter
from aggrequant.segmentation.postprocessing import (
    remove_border_objects,
    filter_aggregates_by_cells
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

        # Build purpose -> pattern mapping for triplet discovery
        self._channel_by_purpose = {}
        for ch in self.config.channels:
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
        if seg.aggregate_method == "filter":
            self._aggregate_segmenter = FilterBasedSegmenter(
                normalized_threshold=seg.aggregate_intensity_threshold,
                min_aggregate_area=seg.aggregate_min_size,
                verbose=verbose,
            )
        elif seg.aggregate_method == "unet":
            self._aggregate_segmenter = NeuralNetworkSegmenter(
                weights_path=seg.aggregate_model_path,
                remove_objects_below=seg.aggregate_min_size,
                device="cuda" if self.config.use_gpu else "cpu",
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown aggregate_method '{seg.aggregate_method}'. "
                f"Must be 'filter' or 'unet'."
            )

        # Configure GPU before any model loading
        if self.config.use_gpu:
            configure_tensorflow_memory_growth()

        self._plate_name = self.config.plate_name

        # Accumulated results (saved to CSV at end of run)
        self._field_results: List[dict] = []

    def run(self, max_fields: Optional[int] = None, segmentation_only: bool = False):
        """
        Run the segmentation pipeline on all images.

        Arguments:
            max_fields: If set, stop after processing this many fields.
            segmentation_only: If True, skip quantification, CSV output, and plots.
        """
        self._segmentation_only = segmentation_only
        logger.info(f"Loading images from {self.config.input_dir}")

        triplets = build_field_triplets(
            self.config.input_dir, self._channel_by_purpose,
        )
        logger.info(f"Found {len(triplets)} complete fields")

        # Load existing results when resuming (so skipped fields keep their data)
        if not segmentation_only:
            csv_path = self.config.output_dir / "field_measurements.csv"
            if not self.config.output.overwrite_masks and csv_path.exists():
                self._field_results = pd.read_csv(csv_path).to_dict("records")
                logger.info(f"Loaded {len(self._field_results)} existing results")
            else:
                self._field_results = []

        for i, triplet in enumerate(triplets):
            self._process_field(triplet)
            if max_fields is not None and (i + 1) >= max_fields:
                logger.info(f"Reached max_fields={max_fields}, stopping")
                break

        if not segmentation_only:
            self._save_results()
            self._generate_plots()
        logger.info("Pipeline complete")

    def _mask_paths(self, well_id: str, field_id: str) -> dict:
        """Return dict of {type: Path} for the 3 label TIFs."""
        labels_dir = self.config.output_dir / "labels"
        return {
            "nuclei": labels_dir / f"{well_id}_f{field_id}_nuclei.tif",
            "cells": labels_dir / f"{well_id}_f{field_id}_cells.tif",
            "aggregates": labels_dir / f"{well_id}_f{field_id}_aggregates.tif",
        }

    def _masks_exist(self, well_id: str, field_id: str) -> bool:
        """Return True if all 3 label TIFs exist on disk."""
        return all(p.exists() for p in self._mask_paths(well_id, field_id).values())

    def _field_has_results(self, well_id: str, field_id: str) -> bool:
        """Return True if this field already has a row in _field_results."""
        fid = int(field_id)
        return any(
            r["well_id"] == well_id and r["field"] == fid
            for r in self._field_results
        )

    def _recompute_from_masks(self, triplet: FieldTriplet):
        """Load cached masks from disk and recompute quantification."""
        well_id, field_id = triplet.well_id, triplet.field_id
        paths = self._mask_paths(well_id, field_id)

        nuclei_labels = tifffile.imread(paths["nuclei"])
        cell_labels = tifffile.imread(paths["cells"])
        aggregate_labels = tifffile.imread(paths["aggregates"])

        measurements = quantify_field(
            cell_labels, aggregate_labels, nuclei_labels,
            min_aggregate_area=self.config.segmentation.aggregate_min_size,
        )
        row = {
            "plate_name": self._plate_name,
            "well_id": well_id,
            "field": int(field_id),
            **measurements,
        }
        self._field_results.append(row)
        logger.info(f"    Recomputed from masks: {measurements['n_cells']} cells, "
                    f"{measurements['n_aggregates']} aggregates, "
                    f"{measurements['pct_aggregate_positive_cells']:.1f}% agg-positive")

    def _process_field(self, triplet: FieldTriplet):
        """Process a single field of view."""
        well_id, field_id = triplet.well_id, triplet.field_id

        # Skip if masks already exist (previous run completed this field)
        if not self.config.output.overwrite_masks and self._masks_exist(well_id, field_id):
            # Recompute quantification if missing from CSV
            if not self._segmentation_only and not self._field_has_results(well_id, field_id):
                logger.info(f"  Recomputing {well_id}/f{field_id} (masks cached, CSV missing)")
                self._recompute_from_masks(triplet)
            else:
                logger.info(f"  Skipping {well_id}/f{field_id} (cached)")
            return

        try:
            self._segment_field(triplet)
        except Exception as e:
            logger.error(f"  Failed {well_id}/f{field_id}: {e}")

    def _segment_field(self, triplet: FieldTriplet):
        """Run segmentation, quantification, and mask saving for one field."""
        well_id, field_id = triplet.well_id, triplet.field_id
        logger.info(f"  Processing {well_id}/f{field_id}")

        # Load images directly from triplet paths
        nuclei_img = load_image(triplet.paths["nuclei"])
        cell_img = load_image(triplet.paths["cells"])
        aggregate_img = load_image(triplet.paths["aggregates"])

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

        # Quantification
        if not self._segmentation_only:
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

    def _generate_plots(self):
        """Generate plate heatmaps and QC plots from field measurements."""
        csv_path = self.config.output_dir / "field_measurements.csv"
        if not csv_path.exists():
            return
        try:
            from aggrequant.visualization.heatmaps import generate_all_heatmaps
        except ImportError:
            logger.warning("plotly not installed — skipping heatmap generation")
            return
        plots_dir = generate_all_heatmaps(
            csv_path, plate_format=self.config.plate_format,
        )
        logger.info(f"Heatmaps saved to {plots_dir}")

        if self.config.control_wells:
            from aggrequant.visualization.qc_plots import plot_control_strip
            qc_path = self.config.output_dir / "plots" / "qc_control_strip.png"
            plot_control_strip(csv_path, self.config.control_wells, output_path=qc_path)
            logger.info(f"QC strip plot saved to {qc_path}")

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
