#!/usr/bin/env python
"""Test script for the segmentation pipeline."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from pathlib import Path
import yaml

from aggrequant.pipeline import SegmentationPipeline


def create_test_config(output_dir: Path) -> Path:
    """Create a test config file pointing to test data."""
    config = {
        "input_dir": str(Path(__file__).parent.parent / "data" / "test"),
        "plate_format": "384",
        "channels": [
            {"name": "DAPI", "pattern": "390", "purpose": "nuclei"},
            {"name": "GFP", "pattern": "473", "purpose": "aggregates"},
            {"name": "CellMask", "pattern": "631", "purpose": "cells"},
        ],
        "segmentation": {
            "nuclei_sigma_denoise": 2.0,
            "nuclei_sigma_background": 50.0,
            "nuclei_min_area": 300,
            "nuclei_max_area": 15000,
            "aggregate_method": "filter",
            "aggregate_min_size": 9,
            "aggregate_intensity_threshold": 1.6,
        },
        "quality": {
            "focus_patch_size": [40, 40],
            "focus_blur_threshold": 15.0,
        },
        "output": {
            "output_dir": str(output_dir),
            "save_masks": True,
            "save_overlays": False,
            "save_statistics": False,
        },
        "use_gpu": True,
        "verbose": True,
    }

    config_path = output_dir / "test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


def main():
    output_dir = Path(__file__).parent.parent / "data" / "test" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    config_path = create_test_config(output_dir)
    print(f"Config file: {config_path}")

    print("\n" + "=" * 50)
    print("Running segmentation pipeline...")
    print("=" * 50 + "\n")

    pipeline = SegmentationPipeline(config_path, verbose=True)
    pipeline.run()

    print("\n" + "=" * 50)
    print("Checking outputs...")
    print("=" * 50)


if __name__ == "__main__":
    main()
