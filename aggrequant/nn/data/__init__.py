"""Data loading and augmentation for neural network training.

This module provides PyTorch datasets and data augmentation pipelines
for aggregate segmentation training.

Author: Athena Economides, 2026, UZH

Example:
    >>> from aggrequant.nn.data import AggregateDataset, get_training_augmentation
    >>> transform = get_training_augmentation()
    >>> dataset = AggregateDataset(
    ...     image_dir="/path/to/images",
    ...     mask_dir="/path/to/masks",
    ...     transform=transform,
    ... )
"""

from aggrequant.nn.data.dataset import (
    AggregateDataset,
    PatchDataset,
    InferenceDataset,
    create_dataloaders,
    load_image,
    normalize_image,
)
from aggrequant.nn.data.augmentation import (
    get_training_augmentation,
    get_validation_augmentation,
    get_test_time_augmentation,
    get_light_augmentation,
    get_heavy_augmentation,
    get_augmentation_by_name,
    list_augmentation_presets,
    AUGMENTATION_PRESETS,
)

__all__ = [
    # Dataset classes
    "AggregateDataset",
    "PatchDataset",
    "InferenceDataset",
    "create_dataloaders",
    "load_image",
    "normalize_image",
    # Augmentation functions
    "get_training_augmentation",
    "get_validation_augmentation",
    "get_test_time_augmentation",
    "get_light_augmentation",
    "get_heavy_augmentation",
    "get_augmentation_by_name",
    "list_augmentation_presets",
    "AUGMENTATION_PRESETS",
]
