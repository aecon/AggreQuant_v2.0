#!/usr/bin/env python
"""
Test image detection step of the AggreQuant pipeline.

This script ONLY runs the image discovery/detection step without performing
segmentation or any other analysis. It is useful for validating that your
configuration correctly detects images from the input directory.

Usage:
    conda activate AggreQuant
    python scripts/test_image_detection.py configs/test_384well.yaml

    # With verbose output:
    python scripts/test_image_detection.py configs/test_384well.yaml --verbose

    # Show sample filenames:
    python scripts/test_image_detection.py configs/test_384well.yaml --show-files
"""

import sys
import argparse
from pathlib import Path
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from aggrequant.common import print_config_summary, print_section_header
from aggrequant.loaders.config import PipelineConfig
from aggrequant.loaders.images import build_field_triplets


def print_image_detection_results(
    triplets,
    config,
    show_files: bool = False,
    max_wells_to_show: int = 10,
):
    """
    Print detailed results of the image detection step.

    Arguments:
        triplets: List of FieldTriplet (complete fields only)
        config: PipelineConfig object
        show_files: Whether to show sample filenames
        max_wells_to_show: Maximum number of wells to show in detail
    """
    print_section_header("IMAGE DETECTION RESULTS")

    # Group triplets by well for statistics
    wells_fields = defaultdict(list)
    for t in triplets:
        wells_fields[t.well_id].append(t)

    n_channels = len(config.channels)
    print(f"  Total wells detected:   {len(wells_fields)}")
    print(f"  Total fields of view:   {len(triplets)}")
    print(f"  Total image files:      {len(triplets) * n_channels}")
    print()

    # Channel statistics (all triplets are complete, so each has all channels)
    print("  Channel breakdown:")
    for ch in config.channels:
        print(f"    - {ch.name} (pattern='{ch.pattern}'): {len(triplets)} images")
    print()

    # Fields per well statistics
    fields_per_well = [len(ts) for ts in wells_fields.values()]
    if fields_per_well:
        avg_fields = sum(fields_per_well) / len(fields_per_well)
        print(f"  Fields per well:")
        print(f"    - Average: {avg_fields:.1f}")
        print(f"    - Min:     {min(fields_per_well)}")
        print(f"    - Max:     {max(fields_per_well)}")
        print()

    # Well list
    well_ids = sorted(wells_fields.keys())
    print(f"  Detected wells ({len(well_ids)} total):")

    # Group by row for nicer display
    rows = defaultdict(list)
    for well_id in well_ids:
        rows[well_id[0]].append(well_id)

    for row_letter in sorted(rows.keys()):
        row_wells = rows[row_letter]
        if len(row_wells) <= 12:
            print(f"    Row {row_letter}: {', '.join(row_wells)}")
        else:
            shown = row_wells[:6] + ["..."] + row_wells[-3:]
            print(f"    Row {row_letter}: {', '.join(shown)} ({len(row_wells)} wells)")
    print()

    # Control well detection
    print("  Control wells (from config):")
    for control_type, wells in config.control_wells.items():
        detected = [w for w in wells if w in wells_fields]
        missing = [w for w in wells if w not in wells_fields]
        print(f"    - {control_type}: {len(detected)}/{len(wells)} found")
        if missing:
            print(f"      Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")
    print()

    # Detailed well breakdown (limited)
    if max_wells_to_show > 0:
        print(f"  Detailed well breakdown (first {min(max_wells_to_show, len(well_ids))} wells):")
        for well_id in well_ids[:max_wells_to_show]:
            well_triplets = wells_fields[well_id]
            n_files = len(well_triplets) * n_channels
            print(f"    {well_id}: {len(well_triplets)} fields, {n_files} files")

            if show_files:
                first = sorted(well_triplets, key=lambda t: t.field_id)[0]
                for purpose, path in sorted(first.paths.items()):
                    print(f"        - [{purpose}] {path.name}")

        if len(well_ids) > max_wells_to_show:
            print(f"    ... and {len(well_ids) - max_wells_to_show} more wells")
    print()

    print("  All detected fields have complete channel sets (incomplete fields were skipped).")
    print("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--show-files",
        action="store_true",
        help="Show sample filenames for each well"
    )
    parser.add_argument(
        "--max-wells",
        type=int,
        default=10,
        help="Maximum number of wells to show in detail (default: 10)"
    )

    args = parser.parse_args()

    # Validate config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")

    try:
        config = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print configuration summary
    print_config_summary(config)

    # Validate input directory exists
    if not config.input_dir.exists():
        print(f"Error: Input directory not found: {config.input_dir}")
        print(f"       Please verify the path in your config file.")
        return 1

    # Run image detection
    print_section_header("RUNNING IMAGE DETECTION")
    print(f"  Scanning directory: {config.input_dir}")
    print()

    try:
        # Build purpose -> pattern mapping from config
        channel_purposes = {ch.purpose: ch.pattern for ch in config.channels}

        if args.verbose:
            print(f"  Channel purposes: {channel_purposes}")
            print()

        # Discover image triplets (scans directory once)
        triplets = build_field_triplets(config.input_dir, channel_purposes)

        # Print results
        print_image_detection_results(
            triplets=triplets,
            config=config,
            show_files=args.show_files,
            max_wells_to_show=args.max_wells,
        )

        print("Image detection test completed successfully.")
        print("NOTE: This script did NOT run segmentation or analysis.")
        print("      Use scripts/run_plate_analysis.py to run the full pipeline.")

        return 0

    except Exception as e:
        print(f"\nError during image detection: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
