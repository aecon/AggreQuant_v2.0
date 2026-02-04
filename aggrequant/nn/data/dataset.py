"""PyTorch Dataset for aggregate segmentation training.

This module provides dataset classes for loading image/mask pairs with
support for patch extraction and integration with albumentations.

Author: Athena Economides

Example:
    >>> from aggrequant.nn.data.dataset import AggregateDataset
    >>> dataset = AggregateDataset(
    ...     image_dir="/path/to/images",
    ...     mask_dir="/path/to/masks",
    ...     patch_size=128,
    ... )
    >>> image, mask = dataset[0]
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import skimage.io

# Try to import tifffile for better TIFF support
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Load an image from disk.

    Arguments:
        path: Path to image file (TIFF, PNG, etc.)

    Returns:
        Image as numpy array (H, W) or (H, W, C)
    """
    path = str(path)
    if HAS_TIFFFILE and path.lower().endswith(('.tif', '.tiff')):
        return tifffile.imread(path)
    return skimage.io.imread(path)


def normalize_image(
    image: np.ndarray,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Normalize image to [0, 1] range using percentile scaling.

    Arguments:
        image: Input image
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping

    Returns:
        Normalized image in [0, 1] range as float32
    """
    image = image.astype(np.float32)
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)

    if p_high - p_low < 1e-6:
        return np.zeros_like(image, dtype=np.float32)

    image = (image - p_low) / (p_high - p_low)
    return np.clip(image, 0, 1).astype(np.float32)


class AggregateDataset(Dataset):
    """PyTorch Dataset for aggregate segmentation.

    Loads image/mask pairs with optional patch extraction and augmentation.

    Arguments:
        image_dir: Directory containing input images
        mask_dir: Directory containing mask images
        image_pattern: Glob pattern for finding images (default: "*.tif")
        mask_suffix: Suffix to replace in image filename to get mask filename
            (e.g., if images are "img_001.tif", masks might be "img_001_mask.tif")
        patch_size: Size of patches to extract (None for full images)
        patches_per_image: Number of patches to extract per image (if patch_size is set)
        transform: Albumentations transform pipeline
        normalize: Whether to normalize images to [0, 1]
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        verbose: Print loading information
        debug: Print detailed debug information

    Example:
        >>> from aggrequant.nn.data.augmentation import get_training_augmentation
        >>> dataset = AggregateDataset(
        ...     image_dir="/path/to/images",
        ...     mask_dir="/path/to/masks",
        ...     patch_size=128,
        ...     patches_per_image=16,
        ...     transform=get_training_augmentation(),
        ... )
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        image_pattern: str = "*.tif",
        mask_suffix: str = "_mask",
        patch_size: Optional[int] = None,
        patches_per_image: int = 16,
        transform: Optional[Callable] = None,
        normalize: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        verbose: bool = False,
        debug: bool = False,
    ) -> None:
        me = "AggregateDataset.__init__"

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.transform = transform
        self.normalize = normalize
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.verbose = verbose
        self.debug = debug

        # Find all image files
        self.image_files = sorted(self.image_dir.glob(image_pattern))

        if len(self.image_files) == 0:
            raise ValueError(
                f"({me}) No images found in {self.image_dir} "
                f"matching pattern '{image_pattern}'"
            )

        # Build image-mask pairs
        self.pairs: List[Tuple[Path, Path]] = []
        for img_path in self.image_files:
            # Try to find corresponding mask
            mask_name = img_path.stem + mask_suffix + img_path.suffix
            mask_path = self.mask_dir / mask_name

            if not mask_path.exists():
                # Try without suffix replacement
                mask_path = self.mask_dir / img_path.name

            if mask_path.exists():
                self.pairs.append((img_path, mask_path))
            elif verbose:
                print(f"({me}) Warning: No mask found for {img_path.name}")

        if len(self.pairs) == 0:
            raise ValueError(
                f"({me}) No image-mask pairs found. "
                f"Check mask_dir and mask_suffix settings."
            )

        if verbose:
            print(f"({me}) Found {len(self.pairs)} image-mask pairs")

        # Calculate total length
        if patch_size is not None:
            self._length = len(self.pairs) * patches_per_image
        else:
            self._length = len(self.pairs)

    def __len__(self) -> int:
        """Return total number of samples."""
        return self._length

    def _extract_random_patch(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a random patch from image and mask.

        Arguments:
            image: Input image (H, W)
            mask: Input mask (H, W)

        Returns:
            Tuple of (image_patch, mask_patch)
        """
        h, w = image.shape[:2]
        ps = self.patch_size

        # Ensure image is large enough
        if h < ps or w < ps:
            # Pad if necessary
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = image.shape[:2]

        # Random top-left corner
        y = np.random.randint(0, h - ps + 1)
        x = np.random.randint(0, w - ps + 1)

        return image[y:y+ps, x:x+ps], mask[y:y+ps, x:x+ps]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample.

        Arguments:
            idx: Sample index

        Returns:
            Tuple of (image_tensor, mask_tensor)
        """
        me = "AggregateDataset.__getitem__"

        # Determine which image-mask pair to load
        if self.patch_size is not None:
            pair_idx = idx // self.patches_per_image
        else:
            pair_idx = idx

        img_path, mask_path = self.pairs[pair_idx]

        # Load image and mask
        image = load_image(img_path)
        mask = load_image(mask_path)

        if self.debug:
            print(f"({me}) Loaded {img_path.name}: shape={image.shape}, dtype={image.dtype}")

        # Ensure 2D
        if image.ndim == 3:
            image = image[:, :, 0]  # Take first channel
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Normalize image
        if self.normalize:
            image = normalize_image(image, self.percentile_low, self.percentile_high)
        else:
            image = image.astype(np.float32)

        # Binarize mask
        mask = (mask > 0).astype(np.float32)

        # Extract patch if configured
        if self.patch_size is not None:
            image, mask = self._extract_random_patch(image, mask)

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors (C, H, W)
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


class PatchDataset(Dataset):
    """Dataset for pre-extracted patches.

    Loads patches from directories containing pre-extracted image and mask patches.

    Arguments:
        image_dir: Directory containing image patches
        mask_dir: Directory containing mask patches
        transform: Albumentations transform pipeline
        normalize: Whether to normalize images

    Example:
        >>> dataset = PatchDataset(
        ...     image_dir="/path/to/patches/images",
        ...     mask_dir="/path/to/patches/masks",
        ... )
    """

    def __init__(
        self,
        image_dir: Union[str, Path],
        mask_dir: Union[str, Path],
        transform: Optional[Callable] = None,
        normalize: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
    ) -> None:
        me = "PatchDataset.__init__"

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.normalize = normalize
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high

        # Find all patches (support multiple formats)
        patterns = ["*.tif", "*.tiff", "*.png", "*.npy"]
        self.image_files = []
        for pattern in patterns:
            self.image_files.extend(self.image_dir.glob(pattern))
        self.image_files = sorted(self.image_files)

        if len(self.image_files) == 0:
            raise ValueError(f"({me}) No patch files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name

        # Handle numpy files
        if img_path.suffix == '.npy':
            image = np.load(img_path)
            mask = np.load(mask_path)
        else:
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

        # Apply augmentation
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        # Convert to tensors
        image = torch.from_numpy(image).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return image, mask


class InferenceDataset(Dataset):
    """Dataset for inference (no masks required).

    Arguments:
        image_paths: List of image paths or directory
        normalize: Whether to normalize images
        patch_size: Size of patches for sliding window inference
        stride: Stride for sliding window (default: patch_size)

    Returns:
        For full images: (image_tensor, image_path)
        For patches: (patch_tensor, patch_info_dict)
    """

    def __init__(
        self,
        image_paths: Union[str, Path, List[Union[str, Path]]],
        normalize: bool = True,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0,
        patch_size: Optional[int] = None,
        stride: Optional[int] = None,
    ) -> None:
        me = "InferenceDataset.__init__"

        # Handle directory input
        if isinstance(image_paths, (str, Path)):
            path = Path(image_paths)
            if path.is_dir():
                patterns = ["*.tif", "*.tiff", "*.png"]
                self.image_paths = []
                for pattern in patterns:
                    self.image_paths.extend(path.glob(pattern))
                self.image_paths = sorted(self.image_paths)
            else:
                self.image_paths = [path]
        else:
            self.image_paths = [Path(p) for p in image_paths]

        if len(self.image_paths) == 0:
            raise ValueError(f"({me}) No images found")

        self.normalize = normalize
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size

        # If using patches, pre-compute patch positions for all images
        self.patches_info = []
        if patch_size is not None:
            for img_idx, img_path in enumerate(self.image_paths):
                img = load_image(img_path)
                h, w = img.shape[:2]
                for y in range(0, h - patch_size + 1, self.stride):
                    for x in range(0, w - patch_size + 1, self.stride):
                        self.patches_info.append({
                            'img_idx': img_idx,
                            'y': y,
                            'x': x,
                            'img_path': img_path,
                        })

    def __len__(self) -> int:
        if self.patch_size is not None:
            return len(self.patches_info)
        return len(self.image_paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, Union[Path, Dict[str, Any]]]:
        if self.patch_size is not None:
            # Return patch
            info = self.patches_info[idx]
            image = load_image(info['img_path'])

            if image.ndim == 3:
                image = image[:, :, 0]

            y, x = info['y'], info['x']
            patch = image[y:y+self.patch_size, x:x+self.patch_size]

            if self.normalize:
                patch = normalize_image(patch, self.percentile_low, self.percentile_high)
            else:
                patch = patch.astype(np.float32)

            patch = torch.from_numpy(patch).unsqueeze(0).float()
            return patch, info
        else:
            # Return full image
            img_path = self.image_paths[idx]
            image = load_image(img_path)

            if image.ndim == 3:
                image = image[:, :, 0]

            if self.normalize:
                image = normalize_image(image, self.percentile_low, self.percentile_high)
            else:
                image = image.astype(np.float32)

            image = torch.from_numpy(image).unsqueeze(0).float()
            return image, img_path


def create_dataloaders(
    train_image_dir: Union[str, Path],
    train_mask_dir: Union[str, Path],
    val_image_dir: Optional[Union[str, Path]] = None,
    val_mask_dir: Optional[Union[str, Path]] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    patch_size: int = 128,
    patches_per_image: int = 16,
    batch_size: int = 16,
    num_workers: int = 4,
    val_split: float = 0.2,
    seed: int = 42,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Create training and validation dataloaders.

    Arguments:
        train_image_dir: Directory with training images
        train_mask_dir: Directory with training masks
        val_image_dir: Directory with validation images (optional)
        val_mask_dir: Directory with validation masks (optional)
        train_transform: Augmentation for training
        val_transform: Augmentation for validation
        patch_size: Patch size for extraction
        patches_per_image: Patches per image
        batch_size: Batch size
        num_workers: Number of data loading workers
        val_split: Validation split ratio (if val dirs not provided)
        seed: Random seed for split

    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader, random_split

    # Create training dataset
    train_dataset = AggregateDataset(
        image_dir=train_image_dir,
        mask_dir=train_mask_dir,
        patch_size=patch_size,
        patches_per_image=patches_per_image,
        transform=train_transform,
    )

    # Create validation dataset
    if val_image_dir is not None and val_mask_dir is not None:
        val_dataset = AggregateDataset(
            image_dir=val_image_dir,
            mask_dir=val_mask_dir,
            patch_size=patch_size,
            patches_per_image=patches_per_image // 2,
            transform=val_transform,
        )
    else:
        # Split training dataset
        total_len = len(train_dataset)
        val_len = int(total_len * val_split)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset = random_split(
            train_dataset, [train_len, val_len], generator=generator
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


__all__ = [
    "AggregateDataset",
    "PatchDataset",
    "InferenceDataset",
    "create_dataloaders",
    "load_image",
    "normalize_image",
]
