#!/usr/bin/env python
"""
Run AggreQuant plate analysis from a YAML configuration file.

This script provides a reproducible way to run the complete analysis pipeline
for High Content Screening data. All parameters are specified in the config file.

Usage:
    conda activate AggreQuant
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml

    # With verbose output:
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml --verbose

    # Process specific wells only:
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml --wells A01 A02 B01

    # Dry run (validate config without processing):
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml --dry-run

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_progress_bar(width: int = 50):
    """Create a progress bar callback function."""
    def progress_callback(progress: float, message: str):
        filled = int(width * progress)
        bar = "=" * filled + "-" * (width - filled)
        # Truncate message if too long
        msg = message[:40].ljust(40)
        print(f"\r[{bar}] {progress*100:5.1f}% {msg}", end="", flush=True)
        if progress >= 1.0:
            print()  # Newline at completion
    return progress_callback


def print_config_summary(config):
    """Print a summary of the configuration."""
    print("\n" + "=" * 70)
    print("  CONFIGURATION SUMMARY")
    print("=" * 70)
    print(f"  Input directory:    {config.input_dir}")
    print(f"  Output directory:   {config.output.output_dir}")
    print(f"  Plate format:       {config.plate_format}-well")
    print(f"  GPU enabled:        {config.use_gpu}")
    print()
    print("  Channels:")
    for ch in config.channels:
        print(f"    - {ch.name}: pattern='{ch.pattern}', purpose={ch.purpose}")
    print()
    print("  Segmentation:")
    print(f"    - Nuclei:     {config.segmentation.nuclei_model}")
    print(f"    - Cells:      {config.segmentation.cell_model}")
    print(f"    - Aggregates: {config.segmentation.aggregate_method}")
    print()
    print("  Control wells:")
    for control_type, wells in config.control_wells.items():
        print(f"    - {control_type}: {', '.join(wells[:4])}{'...' if len(wells) > 4 else ''}")
    print("=" * 70 + "\n")


def print_results_summary(result):
    """Print a summary of the analysis results."""
    print("\n" + "=" * 70)
    print("  ANALYSIS RESULTS")
    print("=" * 70)
    print(f"  Plate:              {result.plate_name}")
    print(f"  Plate format:       {result.plate_format}-well")
    print(f"  Timestamp:          {result.timestamp}")
    print()
    print("  Processing:")
    print(f"    - Wells processed:    {result.total_n_wells_processed}")
    print(f"    - Fields processed:   {result.total_n_fields_processed}")
    print(f"    - Total cells:        {result.total_n_cells:,}")
    print(f"    - Avg cells/well:     {result.avg_cells_per_well:.1f}")
    print()

    if result.ssmd is not None:
        print("  Quality Metrics:")
        print(f"    - SSMD:               {result.ssmd:.3f}")
        if result.ssmd_control_pair:
            print(f"    - Control pair:       {result.ssmd_control_pair[0]} vs {result.ssmd_control_pair[1]}")
        print()

    if result.processing_time_seconds:
        mins = result.processing_time_seconds / 60
        print(f"  Processing time:    {mins:.1f} minutes")

    print("=" * 70)

    # Print well-level summary statistics
    if result.well_results:
        print("\n  Well Statistics (aggregate-positive %):")

        # Group by control type
        controls = {}
        samples = []
        for well_id, well_result in sorted(result.well_results.items()):
            if well_result.control_type:
                if well_result.control_type not in controls:
                    controls[well_result.control_type] = []
                controls[well_result.control_type].append(well_result.pct_aggregate_positive_cells)
            else:
                samples.append(well_result.pct_aggregate_positive_cells)

        import numpy as np

        for control_type, values in controls.items():
            arr = np.array(values)
            print(f"    {control_type:12s}: mean={arr.mean():.2f}%, std={arr.std():.2f}%, n={len(arr)}")

        if samples:
            arr = np.array(samples)
            print(f"    {'samples':12s}: mean={arr.mean():.2f}%, std={arr.std():.2f}%, n={len(arr)}")

        print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run AggreQuant plate analysis from YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run analysis with default settings
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml

    # Process specific wells only
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml --wells A01 A02 B01 B02

    # Validate configuration without processing
    python scripts/run_plate_analysis.py configs/plate1_384well.yaml --dry-run
        """
    )

    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--wells",
        nargs="+",
        help="Specific wells to process (e.g., --wells A01 A02 B01)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running analysis"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )

    args = parser.parse_args()

    # Validate config file exists
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        return 1

    # Load configuration
    print(f"\nLoading configuration from: {args.config}")

    try:
        from aggrequant.loaders.config import PipelineConfig
        config = PipelineConfig.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Apply command-line overrides
    if args.verbose:
        config.verbose = True
    if args.debug:
        config.debug = True
    if args.no_gpu:
        config.use_gpu = False

    # Print configuration summary
    print_config_summary(config)

    # Validate input directory exists
    if not config.input_dir.exists():
        print(f"Error: Input directory not found: {config.input_dir}")
        print(f"       Please place your plate images in: {config.input_dir.absolute()}")
        return 1

    # Check for images in input directory
    image_count = len(list(config.input_dir.glob("**/*.tif*")))
    if image_count == 0:
        print(f"Warning: No TIFF images found in {config.input_dir}")
        print(f"         Looking for: **/*.tif, **/*.tiff")
        if not args.dry_run:
            return 1
    else:
        print(f"Found {image_count} TIFF images in input directory")

    # Dry run - just validate config
    if args.dry_run:
        print("\n[DRY RUN] Configuration validated successfully.")
        print(f"          Would process {image_count} images")
        return 0

    # Run the pipeline
    print("\n" + "=" * 70)
    print("  STARTING ANALYSIS")
    print("=" * 70 + "\n")

    try:
        from aggrequant import AggreQuantPipeline

        # Create pipeline
        pipeline = AggreQuantPipeline(
            config,
            verbose=config.verbose,
            debug=config.debug,
        )

        # Create progress callback
        progress_callback = create_progress_bar(width=50)

        # Run analysis
        result = pipeline.run(
            progress_callback=progress_callback,
            wells_to_process=args.wells,
        )

        # Print results summary
        print_results_summary(result)

        # Print output location
        print(f"\nResults saved to: {config.output.output_dir.absolute()}")
        print("\nOutput files:")
        for f in sorted(config.output.output_dir.glob("*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")

        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70 + "\n")

        return 0

    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        return 130

    except ImportError as e:
        print(f"\nError: Missing dependency: {e}")
        print("\nMake sure all required packages are installed:")
        print("  pip install stardist cellpose torch")
        return 1

    except Exception as e:
        print(f"\nError during analysis: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
