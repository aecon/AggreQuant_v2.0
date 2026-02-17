#!/usr/bin/env python
"""
Run the AggreQuant segmentation pipeline from a YAML configuration file.

Usage:
    python scripts/run_pipeline.py configs/test_384well.yaml
    python scripts/run_pipeline.py configs/test_384well.yaml --verbose
"""

import sys
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Run AggreQuant segmentation pipeline from YAML config",
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
        "--max-fields",
        type=int,
        default=None,
        help="Stop after processing this many fields (for quick testing)"
    )

    args = parser.parse_args()

    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    from aggrequant import SegmentationPipeline

    try:
        print(f"Loading configuration from {args.config}")
        pipeline = SegmentationPipeline(
            config_path=args.config,
            verbose=args.verbose,
        )
        pipeline.run(max_fields=args.max_fields)

    except KeyboardInterrupt:
        print("\n\nAnalysis cancelled by user.")
        sys.exit(1)

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
