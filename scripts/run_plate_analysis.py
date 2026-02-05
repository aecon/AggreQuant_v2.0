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

Author: Athena Economides, 2026, UZH
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import CLI utilities from the aggrequant module
from aggrequant.common import (
    create_progress_bar,
    print_config_summary,
    print_results_summary,
)


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
