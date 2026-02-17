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
from aggrequant.loaders.images import ImageLoader, group_files_by_field


def print_image_detection_results(
    loader,
    plate_structure: dict,
    config,
    show_files: bool = False,
    max_wells_to_show: int = 10,
):
    """
    Print detailed results of the image detection step.

    Arguments:
        loader: ImageLoader instance
        plate_structure: Dict mapping well_id -> field_id -> file_list
        config: PipelineConfig object
        show_files: Whether to show sample filenames
        max_wells_to_show: Maximum number of wells to show in detail
    """
    print_section_header("IMAGE DETECTION RESULTS")

    # Basic counts
    total_wells = len(plate_structure)
    total_fields = sum(len(fields) for fields in plate_structure.values())
    total_files = sum(
        len(files)
        for fields in plate_structure.values()
        for files in fields.values()
    )

    print(f"  Total wells detected:   {total_wells}")
    print(f"  Total fields of view:   {total_fields}")
    print(f"  Total image files:      {total_files}")
    print()

    # Channel statistics
    print("  Channel breakdown:")
    channel_counts = defaultdict(int)
    for ch in config.channels:
        pattern = ch.pattern.lower()
        for well_id, fields in plate_structure.items():
            for field_id, files in fields.items():
                for f in files:
                    if pattern in f.name.lower():
                        channel_counts[ch.name] += 1

    for ch in config.channels:
        count = channel_counts[ch.name]
        print(f"    - {ch.name} (pattern='{ch.pattern}'): {count} images")

    print()

    # Fields per well statistics
    fields_per_well = [len(fields) for fields in plate_structure.values()]
    if fields_per_well:
        avg_fields = sum(fields_per_well) / len(fields_per_well)
        min_fields = min(fields_per_well)
        max_fields = max(fields_per_well)
        print(f"  Fields per well:")
        print(f"    - Average: {avg_fields:.1f}")
        print(f"    - Min:     {min_fields}")
        print(f"    - Max:     {max_fields}")
        print()

    # Files per field statistics
    files_per_field = []
    for fields in plate_structure.values():
        for file_list in fields.values():
            files_per_field.append(len(file_list))

    if files_per_field:
        avg_files = sum(files_per_field) / len(files_per_field)
        min_files = min(files_per_field)
        max_files = max(files_per_field)
        print(f"  Files per field:")
        print(f"    - Average: {avg_files:.1f}")
        print(f"    - Min:     {min_files}")
        print(f"    - Max:     {max_files}")
        print()

    # Well list
    well_ids = sorted(plate_structure.keys())
    print(f"  Detected wells ({len(well_ids)} total):")

    # Group by row for nicer display
    rows = defaultdict(list)
    for well_id in well_ids:
        row_letter = well_id[0]
        rows[row_letter].append(well_id)

    for row_letter in sorted(rows.keys()):
        row_wells = rows[row_letter]
        if len(row_wells) <= 12:
            print(f"    Row {row_letter}: {', '.join(row_wells)}")
        else:
            # Truncate for 384-well plates
            shown = row_wells[:6] + ["..."] + row_wells[-3:]
            print(f"    Row {row_letter}: {', '.join(shown)} ({len(row_wells)} wells)")

    print()

    # Control well detection
    print("  Control wells (from config):")
    for control_type, wells in config.control_wells.items():
        detected = [w for w in wells if w in plate_structure]
        missing = [w for w in wells if w not in plate_structure]
        print(f"    - {control_type}: {len(detected)}/{len(wells)} found")
        if missing:
            print(f"      Missing: {', '.join(missing[:5])}{'...' if len(missing) > 5 else ''}")

    print()

    # Detailed well breakdown (limited)
    if max_wells_to_show > 0:
        print(f"  Detailed well breakdown (first {min(max_wells_to_show, len(well_ids))} wells):")
        for well_id in well_ids[:max_wells_to_show]:
            fields = plate_structure[well_id]
            n_fields = len(fields)
            n_files = sum(len(f) for f in fields.values())
            print(f"    {well_id}: {n_fields} fields, {n_files} files")

            if show_files:
                # Show sample files from first field
                first_field = sorted(fields.keys())[0]
                sample_files = fields[first_field][:3]
                for f in sample_files:
                    print(f"        - {f.name}")
                if len(fields[first_field]) > 3:
                    print(f"        ... and {len(fields[first_field]) - 3} more in field {first_field}")

        if len(well_ids) > max_wells_to_show:
            print(f"    ... and {len(well_ids) - max_wells_to_show} more wells")

    print()

    # Warnings / issues
    issues = []

    # Check for wells with missing channels
    for well_id, fields in plate_structure.items():
        for field_id, files in fields.items():
            for ch in config.channels:
                pattern = ch.pattern.lower()
                matching = [f for f in files if pattern in f.name.lower()]
                if not matching:
                    issues.append(f"{well_id}/field{field_id}: missing {ch.name} channel")

    if issues:
        print("  WARNINGS:")
        for issue in issues[:10]:
            print(f"    - {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more issues")
    else:
        print("  No issues detected - all fields have all expected channels.")

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
        # Build channel patterns from config
        channel_patterns = {}
        for ch in config.channels:
            channel_patterns[ch.name] = ch.pattern

        if args.verbose:
            print(f"  Channel patterns: {channel_patterns}")
            print()

        # Create image loader (this triggers file discovery)
        loader = ImageLoader(
            directory=config.input_dir,
            channel_patterns=channel_patterns,
            verbose=args.verbose,
        )

        # Build field structure for each well
        plate_structure = {}
        for well_id in loader.wells:
            well_files = loader.get_well_files(well_id)
            fields = group_files_by_field(well_files)
            plate_structure[well_id] = fields

        # Print results
        print_image_detection_results(
            loader=loader,
            plate_structure=plate_structure,
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
