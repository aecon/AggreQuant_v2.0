"""
CLI utility functions for AggreQuant scripts.

Provides progress bars, configuration summaries, and result formatting
for command-line tools.

Author: Athena Economides, 2026, UZH
"""

from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aggrequant.loaders.config import PipelineConfig
    from aggrequant.quantification.results import PlateResult


# Type alias for progress callback
ProgressCallback = Callable[[float, str], None]


def create_progress_bar(width: int = 50) -> ProgressCallback:
    """
    Create a progress bar callback function.

    Arguments:
        width: Width of the progress bar in characters

    Returns:
        A callback function that accepts (progress: float, message: str)
        where progress is in [0.0, 1.0]
    """
    def progress_callback(progress: float, message: str):
        filled = int(width * progress)
        bar = "=" * filled + "-" * (width - filled)
        # Truncate message if too long
        msg = message[:40].ljust(40)
        print(f"\r[{bar}] {progress*100:5.1f}% {msg}", end="", flush=True)
        if progress >= 1.0:
            print()  # Newline at completion
    return progress_callback


def print_config_summary(config: "PipelineConfig") -> None:
    """
    Print a formatted summary of the pipeline configuration.

    Arguments:
        config: PipelineConfig object to summarize
    """
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
    print("    - Nuclei:     StarDist (2D_versatile_fluo)")
    print(f"    - Cells:      {config.segmentation.cell_model}")
    print(f"    - Aggregates: {config.segmentation.aggregate_method}")
    print()
    print("  Control wells:")
    for control_type, wells in config.control_wells.items():
        print(f"    - {control_type}: {', '.join(wells[:4])}{'...' if len(wells) > 4 else ''}")
    print("=" * 70 + "\n")


def print_results_summary(result: "PlateResult") -> None:
    """
    Print a formatted summary of the analysis results.

    Arguments:
        result: PlateResult object to summarize
    """
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


def print_section_header(title: str, width: int = 70) -> None:
    """
    Print a formatted section header.

    Arguments:
        title: Title text for the header
        width: Total width of the header line
    """
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_key_value(key: str, value: str, indent: int = 2) -> None:
    """
    Print a key-value pair with consistent formatting.

    Arguments:
        key: Label/key text
        value: Value text
        indent: Number of spaces to indent
    """
    spaces = " " * indent
    print(f"{spaces}{key:20s} {value}")
