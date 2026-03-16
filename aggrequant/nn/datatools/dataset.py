"""
PyTorch Dataset and utilities for aggregate segmentation training.

Training data workflow:
1. Use ``extract_patches`` to grid-cut full images into non-overlapping patches
   and save them to disk (images/ and masks/ subdirectories).
2. Use ``create_dataloaders`` to load patches, split into train/val by
   shuffling the patch file list, and wrap in DataLoaders.

This approach ensures every image contributes patches to both train and val,
while no single patch appears in both — preventing data leakage.

Example:
    >>> from aggrequant.nn.datatools.dataset import extract_patches, create_dataloaders
    >>> from aggrequant.nn.datatools.augmentation import get_training_augmentation
    >>>
    >>> extract_patches("raw/images", "raw/masks", "patches", patch_size=128)
    >>> train_loader, val_loader = create_dataloaders(
    ...     patch_dir="patches",
    ...     train_transform=get_training_augmentation(),
    ... )
"""

from pathlib import Path
from typing import List, Tuple, Optional, Callable, Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from aggrequant.common.image_utils import (
    load_image,
    find_image_files,
    normalize_image as _normalize_image_common,
)
from aggrequant.common.logging import get_logger
from aggrequant.nn.datatools.augmentation import apply_transform

logger = get_logger(__name__)


def normalize_image(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Normalize image to [0, 1] range using percentile scaling.

    This is a convenience wrapper around the common normalize_image function
    that uses percentile-based normalization by default, which is preferred
    for neural network training.

    Arguments:
        image: Input image
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        Normalized image in [0, 1] range as float32
    """
    return _normalize_image_common(
        image,
        method="percentile",
        percentile_low=percentile_low,
        percentile_high=percentile_high,
    )


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------


def extract_patches(
    image_dir: Union[str, Path],
    mask_dir: Union[str, Path],
    output_dir: Union[str, Path],
    patch_size: int = 128,
    image_pattern: str = "*.tif",
    mask_suffix: str = "_mask",
) -> int:
    """Grid-cut images and masks into non-overlapping patches and save to disk.

    Creates ``output_dir/images/`` and ``output_dir/masks/`` subdirectories.
    Patch filenames encode their source image and position:
    ``{original_stem}_y{row:03d}_x{col:03d}.tif``

    Incomplete edge patches (where the grid doesn't fit exactly) are skipped.

    Arguments:
        image_dir: Directory containing full-size images
        mask_dir: Directory containing corresponding masks
        output_dir: Directory to save patches (images/ and masks/ subdirs)
        patch_size: Patch size in pixels (default: 128)
        image_pattern: Glob pattern for finding images (default: "*.tif")
        mask_suffix: Suffix appended to image stem to find mask file
            (e.g., "img_001.tif" -> "img_001_mask.tif")

    Returns:
        Total number of patches extracted

    Example:
        >>> n = extract_patches("raw/images", "raw/masks", "patches", patch_size=128)
        >>> print(f"Extracted {n} patches")
    """
    import tifffile

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    output_dir = Path(output_dir)

    out_images = output_dir / "images"
    out_masks = output_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    # Find image-mask pairs
    image_files = sorted(image_dir.glob(image_pattern))
    if len(image_files) == 0:
        raise ValueError(
            f"No images found in {image_dir} matching '{image_pattern}'"
        )

    total_patches = 0

    for img_path in image_files:
        # Find corresponding mask
        mask_name = img_path.stem + mask_suffix + img_path.suffix
        mask_path = mask_dir / mask_name

        if not mask_path.exists():
            # Try same filename
            mask_path = mask_dir / img_path.name

        if not mask_path.exists():
            logger.warning(f"No mask found for {img_path.name}, skipping")
            continue

        # Load
        image = load_image(img_path)
        mask = load_image(mask_path)

        if image.ndim == 3:
            image = image[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        h, w = image.shape
        n_rows = h // patch_size
        n_cols = w // patch_size

        for row in range(n_rows):
            for col in range(n_cols):
                y = row * patch_size
                x = col * patch_size
                img_patch = image[y:y + patch_size, x:x + patch_size]
                mask_patch = mask[y:y + patch_size, x:x + patch_size]

                patch_name = f"{img_path.stem}_y{row:03d}_x{col:03d}.tif"
                tifffile.imwrite(out_images / patch_name, img_patch)
                tifffile.imwrite(out_masks / patch_name, mask_patch)
                total_patches += 1

        logger.info(
            f"{img_path.name}: {n_rows}x{n_cols} = {n_rows * n_cols} patches "
            f"(skipped {h - n_rows * patch_size}px bottom, "
            f"{w - n_cols * patch_size}px right)"
        )

    logger.info(f"Total: {total_patches} patches saved to {output_dir}")
    return total_patches


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    """Dataset for pre-extracted patches.

    Loads image and mask patches from parallel directory structures where
    each image patch has a corresponding mask patch with the same filename.

    Arguments:
        image_files: List of image patch file paths
        mask_dir: Directory containing mask patches (matched by filename)
        transform: torchvision v2 transform pipeline
        normalize: Whether to normalize images to [0, 1]
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization

    Example:
        >>> dataset = PatchDataset(
        ...     image_files=list(Path("patches/images").glob("*.tif")),
        ...     mask_dir="patches/masks",
        ... )
    """

    def __init__(
        self,
        image_files: List[Path],
        mask_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        normalize: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> None:
        self.image_files = sorted(image_files)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.normalize = normalize
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        if len(self.image_files) == 0:
            raise ValueError("No image files provided")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name

        image = load_image(img_path)
        mask = load_image(mask_path)

        # Ensure 2D
        if image.ndim == 3:
            image = image[:, :, 0]
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize
        if self.normalize:
            image = normalize_image(image, self.percentile_low, self.percentile_high)
        else:
            image = image.astype(np.float32)

        mask = (mask > 0).astype(np.float32)

        # Apply augmentation and convert to tensors
        if self.transform is not None:
            image, mask = apply_transform(self.transform, image, mask)
        else:
            image = torch.from_numpy(image).unsqueeze(0).float()
            mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


# ---------------------------------------------------------------------------
# DataLoader creation
# ---------------------------------------------------------------------------


def create_dataloaders(
    patch_dir: Union[str, Path],
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    val_split: float = 0.2,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from a patch directory.

    Discovers all patches in ``patch_dir/images/``, shuffles the file list
    with a fixed seed, and splits into train/val. This ensures patches from
    every source image appear in both sets, with no patch overlap.

    Arguments:
        patch_dir: Directory with images/ and masks/ subdirectories
            (as created by ``extract_patches``)
        train_transform: Augmentation for training (None for no augmentation)
        val_transform: Augmentation for validation (None for no augmentation)
        val_split: Fraction of patches for validation (default: 0.2)
        batch_size: Batch size for both loaders
        num_workers: Number of data loading workers
        seed: Random seed for reproducible splits

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     "patches",
        ...     train_transform=get_training_augmentation(),
        ...     batch_size=32,
        ... )
    """
    patch_dir = Path(patch_dir)
    image_dir = patch_dir / "images"
    mask_dir = patch_dir / "masks"

    # Find all patch files
    all_files = find_image_files(image_dir)
    if len(all_files) == 0:
        raise ValueError(f"No image files found in {image_dir}")

    # Shuffle and split
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(all_files))
    val_count = int(len(all_files) * val_split)

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]

    train_files = [all_files[i] for i in train_indices]
    val_files = [all_files[i] for i in val_indices]

    logger.info(
        f"Split {len(all_files)} patches: "
        f"{len(train_files)} train, {len(val_files)} val"
    )

    # Create datasets
    train_dataset = PatchDataset(
        image_files=train_files,
        mask_dir=mask_dir,
        transform=train_transform,
    )

    val_dataset = PatchDataset(
        image_files=val_files,
        mask_dir=mask_dir,
        transform=val_transform,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
