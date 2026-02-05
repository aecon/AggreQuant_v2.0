#!/usr/bin/env python
"""
Simple segmentation-only pipeline.

Usage:
    python scripts/run_segmentation.py

Edit CONFIG_PATH below to point to your YAML config file.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import tifffile

from aggrequant.loaders.config import PipelineConfig
from aggrequant.loaders.images import ImageLoader, group_files_by_field
from aggrequant.segmentation.nuclei.stardist import StarDistSegmenter
from aggrequant.segmentation.cells.cellpose import CellposeSegmenter
from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter

# ============================================
# CONFIGURATION - Edit this path as needed
# ============================================
CONFIG_PATH = Path("configs/plate1_384well.yaml")


def main():
    # Load config
    config = PipelineConfig.from_yaml(CONFIG_PATH)
    print(f"Loaded config from: {CONFIG_PATH}")
    print(f"Input directory: {config.input_dir}")

    # Create output directory for masks
    masks_dir = config.output.output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {masks_dir}")

    # Build channel pattern dict from config
    channel_patterns = {ch.name: ch.pattern for ch in config.channels}

    # Initialize image loader
    loader = ImageLoader(
        directory=config.input_dir,
        channel_patterns=channel_patterns,
        verbose=config.verbose,
    )
    print(f"Found {loader.n_wells} wells")

    # Initialize segmenters
    nuclei_seg = StarDistSegmenter(
        sigma_denoise=config.segmentation.nuclei_sigma_denoise,
        sigma_background=config.segmentation.nuclei_sigma_background,
        min_area=config.segmentation.nuclei_min_area,
        max_area=config.segmentation.nuclei_max_area,
        verbose=config.verbose,
    )
    cell_seg = CellposeSegmenter(
        model_type=config.segmentation.cell_model,
        gpu=config.use_gpu,
        flow_threshold=config.segmentation.cell_flow_threshold,
        cellprob_threshold=config.segmentation.cell_cellprob_threshold,
        verbose=config.verbose,
    )
    agg_seg = FilterBasedSegmenter(
        normalized_threshold=config.segmentation.aggregate_intensity_threshold,
        min_aggregate_area=config.segmentation.aggregate_min_size,
        verbose=config.verbose,
    )

    # Find channel names by purpose
    nuclei_channel = next(ch.name for ch in config.channels if ch.purpose == "nuclei")
    cell_channel = next(ch.name for ch in config.channels if ch.purpose == "cells")
    agg_channel = next(ch.name for ch in config.channels if ch.purpose == "aggregates")

    # Process each well
    for well in loader.wells:
        print(f"\nProcessing well {well}...")
        well_files = loader.get_well_files(well)
        fields = group_files_by_field(well_files)

        for field_id, field_files in fields.items():
            # Load images for this field
            images = {}
            for ch_name, pattern in channel_patterns.items():
                matching = [f for f in field_files if pattern.lower() in f.name.lower()]
                if matching:
                    images[ch_name] = tifffile.imread(matching[0])

            if nuclei_channel not in images:
                continue

            # Segment nuclei
            nuclei_labels = nuclei_seg.segment(images[nuclei_channel])

            # Segment cells (if channel available)
            if cell_channel in images:
                cell_labels = cell_seg.segment(images[cell_channel], nuclei_labels)
            else:
                cell_labels = np.zeros_like(nuclei_labels)

            # Segment aggregates (if channel available)
            if agg_channel in images:
                agg_labels = agg_seg.segment(images[agg_channel])
            else:
                agg_labels = np.zeros_like(nuclei_labels)

            # Save masks
            base_name = f"{well}_f{field_id}"
            tifffile.imwrite(masks_dir / f"{base_name}_nuclei.tif", nuclei_labels)
            tifffile.imwrite(masks_dir / f"{base_name}_cells.tif", cell_labels.astype(np.uint16))
            tifffile.imwrite(masks_dir / f"{base_name}_aggregates.tif", agg_labels)

            print(f"  Field {field_id}: {nuclei_labels.max()} nuclei, "
                  f"{cell_labels.max()} cells, {agg_labels.max()} aggregates")

    print(f"\nDone! Masks saved to: {masks_dir}")


if __name__ == "__main__":
    main()
