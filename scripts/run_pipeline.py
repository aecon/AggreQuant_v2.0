#!/usr/bin/env python
"""
Run the AggreQuant analysis pipeline from command line.

Usage:
    python scripts/run_pipeline.py config.yaml
    python scripts/run_pipeline.py --input /path/to/images --output /path/to/output

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run AggreQuant analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run from YAML config file
    python scripts/run_pipeline.py config.yaml

    # Run with command-line options
    python scripts/run_pipeline.py --input /data/plate1 --output /results/plate1

    # Specify plate format and method
    python scripts/run_pipeline.py --input /data/plate1 --output /results \\
        --plate-format 384 --method filter
        """
    )

    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output directory for results"
    )
    parser.add_argument(
        "--plate-format",
        choices=["96", "384"],
        default="96",
        help="Plate format (default: 96)"
    )
    parser.add_argument(
        "--method",
        choices=["unet", "filter"],
        default="unet",
        help="Aggregate segmentation method (default: unet)"
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Path to UNet model weights (for unet method)"
    )
    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=15.0,
        help="Blur detection threshold (default: 15.0)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug output"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.config is None and args.input is None:
        parser.error("Either a config file or --input directory is required")

    from aggrequant import run_pipeline_from_config, run_pipeline_from_dict
    from aggrequant.loaders.config import PipelineConfig

    def progress_callback(progress: float, message: str):
        """Print progress to console."""
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% {message:<50}", end="", flush=True)
        if progress >= 1.0:
            print()  # Newline at end

    try:
        if args.config:
            # Run from config file
            print(f"Loading configuration from {args.config}")
            result = run_pipeline_from_config(
                args.config,
                progress_callback=progress_callback,
                verbose=args.verbose,
            )
        else:
            # Run from command-line arguments
            print(f"Processing images from {args.input}")

            config_dict = {
                "input_dir": str(args.input),
                "output_dir": str(args.output or args.input / "output"),
                "plate_format": args.plate_format,
                "aggregate_method": args.method,
                "model_path": str(args.model) if args.model else None,
                "blur_threshold": args.blur_threshold,
                "save_masks": True,
                "save_overlays": True,
                "use_gpu": not args.no_gpu,
                "control_wells": {},  # No controls from CLI
            }

            result = run_pipeline_from_dict(
                config_dict,
                progress_callback=progress_callback,
                verbose=args.verbose,
            )

        # Print summary
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Plate: {result.plate_name}")
        print(f"Wells processed: {result.total_n_wells_processed}")
        print(f"Fields processed: {result.total_n_fields_processed}")
        print(f"Total cells: {result.total_n_cells}")
        print(f"Average cells per well: {result.avg_cells_per_well:.1f}")

        if result.ssmd is not None:
            print(f"SSMD: {result.ssmd:.3f}")
            if result.ssmd_control_pair:
                print(f"  Controls: {result.ssmd_control_pair[0]} vs {result.ssmd_control_pair[1]}")

        if result.processing_time_seconds:
            print(f"Processing time: {result.processing_time_seconds:.1f} seconds")

        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
