"""
Data loading and configuration for HCS image analysis.

Author: Athena Economides, 2026, UZH
"""

from .config import (
    ChannelConfig,
    SegmentationConfig,
    QualityConfig,
    OutputConfig,
    PipelineConfig,
    create_default_config,
)
from .images import (
    load_image,
    load_image_stack,
    parse_operetta_filename,
    parse_imageexpress_filename,
    find_channel_files,
    group_files_by_well,
    group_files_by_field,
    ImageLoader,
)
from .plate import (
    PLATE_LAYOUTS,
    well_id_to_indices,
    indices_to_well_id,
    FieldOfView,
    Well,
    Plate,
    create_plate_from_wells,
    generate_all_well_ids,
)

__all__ = [
    # Config
    "ChannelConfig",
    "SegmentationConfig",
    "QualityConfig",
    "OutputConfig",
    "PipelineConfig",
    "create_default_config",
    # Images
    "load_image",
    "load_image_stack",
    "parse_operetta_filename",
    "parse_imageexpress_filename",
    "find_channel_files",
    "group_files_by_well",
    "group_files_by_field",
    "ImageLoader",
    # Plate
    "PLATE_LAYOUTS",
    "well_id_to_indices",
    "indices_to_well_id",
    "FieldOfView",
    "Well",
    "Plate",
    "create_plate_from_wells",
    "generate_all_well_ids",
]
