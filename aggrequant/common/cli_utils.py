"""CLI utility functions for summaries and result formatting."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aggrequant.loaders.config import PipelineConfig


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
    print(f"  Output directory:   {config.output_dir}")
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
