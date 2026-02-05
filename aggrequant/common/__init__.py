"""
Common utilities shared across the aggrequant package.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .image_utils import (
    SUPPORTED_IMAGE_EXTENSIONS,
    find_image_files,
    load_image,
    load_image_stack,
    normalize_image,
    to_uint8,
    to_float32,
    pad_to_multiple,
    unpad,
    remove_small_holes_compat,
    remove_small_objects_compat,
)
from .logging import (
    setup_logging,
    get_logger,
)
from .cli_utils import (
    create_progress_bar,
    print_config_summary,
    print_results_summary,
    print_section_header,
    print_key_value,
    ProgressCallback,
)

__all__ = [
    # Image utilities
    "SUPPORTED_IMAGE_EXTENSIONS",
    "find_image_files",
    "load_image",
    "load_image_stack",
    "normalize_image",
    "to_uint8",
    "to_float32",
    "pad_to_multiple",
    "unpad",
    "remove_small_holes_compat",
    "remove_small_objects_compat",
    # Logging
    "setup_logging",
    "get_logger",
    # CLI utilities
    "create_progress_bar",
    "print_config_summary",
    "print_results_summary",
    "print_section_header",
    "print_key_value",
    "ProgressCallback",
]
