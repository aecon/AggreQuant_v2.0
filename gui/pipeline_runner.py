"""
GUI-specific pipeline runner that builds config from user selections.

This module is internal to the GUI and not part of the public API.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable

from aggrequant.pipeline import AggreQuantPipeline
from aggrequant.loaders.config import (
    PipelineConfig,
    ChannelConfig,
    SegmentationConfig,
    QualityConfig,
    OutputConfig,
)
from aggrequant.quantification.results import PlateResult


ProgressCallback = Callable[[float, str], None]


def run_pipeline_from_dict(
    config_dict: Dict[str, Any],
    progress_callback: Optional[ProgressCallback] = None,
    verbose: bool = True,
) -> PlateResult:
    """
    Run pipeline from a configuration dictionary (for GUI).

    Arguments:
        config_dict: Configuration as dictionary
        progress_callback: Optional progress callback
        verbose: Print progress messages

    Returns:
        PlateResult with all analysis results
    """
    # Build channel configs
    channels = []
    for ch_data in config_dict.get("channels", []):
        channels.append(ChannelConfig(**ch_data))

    # If no channels specified, use defaults
    if not channels:
        channels = [
            ChannelConfig(name="DAPI", pattern="C01", purpose="nuclei"),
            ChannelConfig(name="GFP", pattern="C02", purpose="aggregates"),
            ChannelConfig(name="CellMask", pattern="C03", purpose="cells"),
        ]

    seg_config = SegmentationConfig(
        aggregate_method=config_dict.get("aggregate_method", "unet"),
        aggregate_model_path=config_dict.get("model_path"),
    )

    quality_config = QualityConfig(
        focus_blur_threshold=config_dict.get("blur_threshold", 15.0),
        focus_reject_threshold=config_dict.get("blur_reject_pct", 50.0),
    )

    output_config = OutputConfig(
        output_subdir=config_dict.get("output_subdir", "aggrequant_output"),
        save_masks=config_dict.get("save_masks", True),
        save_overlays=config_dict.get("save_overlays", True),
        save_statistics=True,
    )

    # Convert control_wells from {well: type} to {type: [wells]}
    control_wells_input = config_dict.get("control_wells", {})
    control_wells = {}
    for well, ctrl_type in control_wells_input.items():
        if ctrl_type not in control_wells:
            control_wells[ctrl_type] = []
        control_wells[ctrl_type].append(well)

    config = PipelineConfig(
        input_dir=Path(config_dict.get("input_dir", ".")),
        plate_format=config_dict.get("plate_format", "96"),
        channels=channels,
        segmentation=seg_config,
        quality=quality_config,
        output=output_config,
        control_wells=control_wells,
        use_gpu=config_dict.get("use_gpu", True),
        verbose=verbose,
    )

    pipeline = AggreQuantPipeline(config, verbose=verbose)
    return pipeline.run(progress_callback=progress_callback)
