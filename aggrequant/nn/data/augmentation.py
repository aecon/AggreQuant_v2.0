"""Albumentations augmentation pipelines for aggregate segmentation.

This module provides data augmentation pipelines following nnU-Net best
practices for microscopy image segmentation.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04

Example:
    >>> from aggrequant.nn.data.augmentation import get_training_augmentation
    >>> transform = get_training_augmentation()
    >>> augmented = transform(image=image, mask=mask)
    >>> aug_image, aug_mask = augmented['image'], augmented['mask']
"""

from typing import Optional, Tuple, Dict, Any
import numpy as np

try:
    import albumentations as A
    from albumentations.core.transforms_interface import ImageOnlyTransform
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    A = None


def _check_albumentations() -> None:
    """Check if albumentations is available."""
    if not HAS_ALBUMENTATIONS:
        raise ImportError(
            "albumentations is required for data augmentation. "
            "Install with: pip install albumentations"
        )


def get_training_augmentation(
    image_size: Optional[Tuple[int, int]] = None,
    p_spatial: float = 0.5,
    p_intensity: float = 0.5,
    p_noise: float = 0.3,
    p_blur: float = 0.2,
    p_elastic: float = 0.2,
    rotation_limit: int = 90,
    scale_limit: Tuple[float, float] = (-0.2, 0.2),
    shift_limit: float = 0.1,
) -> "A.Compose":
    """Get training augmentation pipeline.

    Follows nnU-Net best practices for microscopy segmentation:
    - Spatial: rotation, flip, scale, shift, elastic deformation
    - Intensity: brightness, contrast, gamma
    - Noise: Gaussian noise, multiplicative noise
    - Blur: Gaussian blur, motion blur

    Arguments:
        image_size: Target size for resize (None to keep original size)
        p_spatial: Probability for spatial augmentations
        p_intensity: Probability for intensity augmentations
        p_noise: Probability for noise augmentations
        p_blur: Probability for blur augmentations
        p_elastic: Probability for elastic deformation
        rotation_limit: Maximum rotation angle in degrees
        scale_limit: Scale range (min, max) relative to 1.0
        shift_limit: Maximum shift as fraction of image size

    Returns:
        Albumentations Compose object

    Example:
        >>> transform = get_training_augmentation(image_size=(128, 128))
        >>> result = transform(image=image, mask=mask)
    """
    _check_albumentations()

    transforms = []

    # Optional resize
    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    # Spatial augmentations
    spatial_transforms = [
        # Flips - very important for microscopy
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),

        # Rotation (90 degree multiples are common in microscopy)
        A.RandomRotate90(p=0.5),

        # Affine transforms
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotation_limit,
            border_mode=0,  # cv2.BORDER_CONSTANT
            p=p_spatial,
        ),
    ]
    transforms.extend(spatial_transforms)

    # Elastic deformation - simulates tissue deformation
    if p_elastic > 0:
        transforms.append(
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                alpha_affine=120 * 0.03,
                border_mode=0,
                p=p_elastic,
            )
        )

    # Intensity augmentations
    intensity_transforms = A.OneOf([
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=1.0,
        ),
        A.RandomGamma(gamma_limit=(80, 120), p=1.0),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
    ], p=p_intensity)
    transforms.append(intensity_transforms)

    # Noise augmentations
    if p_noise > 0:
        noise_transforms = A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
        ], p=p_noise)
        transforms.append(noise_transforms)

    # Blur augmentations (simulates focus variations)
    if p_blur > 0:
        blur_transforms = A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
        ], p=p_blur)
        transforms.append(blur_transforms)

    return A.Compose(transforms)


def get_validation_augmentation(
    image_size: Optional[Tuple[int, int]] = None,
) -> "A.Compose":
    """Get validation augmentation pipeline.

    Minimal transforms - only resize if needed, no augmentation.

    Arguments:
        image_size: Target size for resize (None to keep original size)

    Returns:
        Albumentations Compose object
    """
    _check_albumentations()

    transforms = []
    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    return A.Compose(transforms)


def get_test_time_augmentation() -> "A.Compose":
    """Get test-time augmentation (TTA) pipeline.

    Returns transforms for TTA inference (flip and rotate).
    Use with TTA inference function to get multiple predictions.

    Returns:
        Albumentations Compose object
    """
    _check_albumentations()

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def get_light_augmentation(
    image_size: Optional[Tuple[int, int]] = None,
) -> "A.Compose":
    """Get lightweight augmentation pipeline.

    Minimal augmentation for quick training or when data is already diverse.

    Arguments:
        image_size: Target size for resize

    Returns:
        Albumentations Compose object
    """
    _check_albumentations()

    transforms = []
    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    transforms.extend([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])

    return A.Compose(transforms)


def get_heavy_augmentation(
    image_size: Optional[Tuple[int, int]] = None,
) -> "A.Compose":
    """Get heavy augmentation pipeline.

    Strong augmentation for small datasets or difficult tasks.

    Arguments:
        image_size: Target size for resize

    Returns:
        Albumentations Compose object
    """
    _check_albumentations()

    transforms = []
    if image_size is not None:
        transforms.append(A.Resize(height=image_size[0], width=image_size[1]))

    transforms.extend([
        # Spatial - aggressive
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.15,
            scale_limit=(-0.3, 0.3),
            rotate_limit=180,
            border_mode=0,
            p=0.7,
        ),

        # Elastic - more aggressive
        A.ElasticTransform(
            alpha=200,
            sigma=200 * 0.05,
            alpha_affine=200 * 0.03,
            border_mode=0,
            p=0.4,
        ),

        # Grid distortion
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),

        # Intensity - more variation
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=1.0,
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8), p=1.0),
        ], p=0.7),

        # Noise - more types
        A.OneOf([
            A.GaussNoise(var_limit=(20.0, 80.0), p=1.0),
            A.MultiplicativeNoise(multiplier=(0.8, 1.2), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
        ], p=0.5),

        # Blur
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),

        # Dropout (simulates imaging artifacts)
        A.OneOf([
            A.CoarseDropout(
                max_holes=8,
                max_height=16,
                max_width=16,
                fill_value=0,
                p=1.0,
            ),
            A.PixelDropout(dropout_prob=0.01, p=1.0),
        ], p=0.2),
    ])

    return A.Compose(transforms)


# Preset augmentation configurations
AUGMENTATION_PRESETS: Dict[str, Dict[str, Any]] = {
    "none": {
        "description": "No augmentation",
        "factory": get_validation_augmentation,
    },
    "light": {
        "description": "Light augmentation (flips and rotations only)",
        "factory": get_light_augmentation,
    },
    "standard": {
        "description": "Standard nnU-Net-style augmentation",
        "factory": get_training_augmentation,
    },
    "heavy": {
        "description": "Heavy augmentation for small datasets",
        "factory": get_heavy_augmentation,
    },
}


def get_augmentation_by_name(
    name: str,
    image_size: Optional[Tuple[int, int]] = None,
    **kwargs,
) -> "A.Compose":
    """Get augmentation pipeline by preset name.

    Arguments:
        name: Preset name ("none", "light", "standard", "heavy")
        image_size: Target image size
        **kwargs: Additional arguments for the augmentation factory

    Returns:
        Albumentations Compose object

    Example:
        >>> transform = get_augmentation_by_name("standard", image_size=(128, 128))
    """
    if name not in AUGMENTATION_PRESETS:
        available = list(AUGMENTATION_PRESETS.keys())
        raise ValueError(
            f"Unknown augmentation preset: '{name}'. "
            f"Available: {available}"
        )

    factory = AUGMENTATION_PRESETS[name]["factory"]
    return factory(image_size=image_size, **kwargs)


def list_augmentation_presets() -> Dict[str, str]:
    """List available augmentation presets.

    Returns:
        Dictionary mapping preset names to descriptions
    """
    return {name: info["description"] for name, info in AUGMENTATION_PRESETS.items()}


__all__ = [
    "get_training_augmentation",
    "get_validation_augmentation",
    "get_test_time_augmentation",
    "get_light_augmentation",
    "get_heavy_augmentation",
    "get_augmentation_by_name",
    "list_augmentation_presets",
    "AUGMENTATION_PRESETS",
]
