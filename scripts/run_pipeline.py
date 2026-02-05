#!/usr/bin/env python
"""
Run the AggreQuant analysis pipeline from a YAML configuration file.

Usage:
    python scripts/run_pipeline.py config.yaml
    python scripts/run_pipeline.py config.yaml --verbose
    python scripts/run_pipeline.py config.yaml --debug

Author: Athena Economides, 2026, UZH
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run AggreQuant analysis pipeline from YAML config",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to YAML configuration file"
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

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    from aggrequant import run_pipeline_from_config

    def progress_callback(progress: float, message: str):
        """Print progress to console."""
        bar_width = 40
        filled = int(bar_width * progress)
        bar = "=" * filled + "-" * (bar_width - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% {message:<50}", end="", flush=True)
        if progress >= 1.0:
            print()

    try:
        print(f"Loading configuration from {args.config}")
        result = run_pipeline_from_config(
            args.config,
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
