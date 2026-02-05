"""
Image utility functions for loading, normalization and type conversion.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path
from typing import Union, List
import numpy as np

# Optional imports for image loading
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import skimage.io
    import skimage.morphology
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = (".tif",)


def find_image_files(
    directory: Union[str, Path],
    recursive: bool = False,
) -> List[Path]:
    """
    Find all supported image files in a directory.

    Arguments:
        directory: Directory to search
        recursive: If True, search subdirectories recursively

    Returns:
        Sorted list of image file paths
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    image_files = []
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        pattern = f"**/*{ext}" if recursive else f"*{ext}"
        image_files.extend(directory.glob(pattern))

    return sorted(image_files)


def load_image(path: Union[str, Path]) -> np.ndarray:
    """
    Load an image from disk.

    Supports TIFF files (via tifffile) and other common formats (via skimage).
    For TIFF files, tifffile is preferred as it handles microscopy metadata better.

    Arguments:
        path: Path to image file (TIFF, PNG, JPEG, etc.)

    Returns:
        Image as numpy array (H, W) or (H, W, C)

    Raises:
        ImportError: If no suitable image loading library is available
        FileNotFoundError: If the file doesn't exist
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    path_str = str(path)
    is_tiff = path_str.lower().endswith(('.tif', '.tiff'))

    # Prefer tifffile for TIFF files
    if is_tiff and HAS_TIFFFILE:
        return tifffile.imread(path_str)

    # Fall back to skimage for other formats or if tifffile unavailable
    if HAS_SKIMAGE:
        return skimage.io.imread(path_str)

    # If only tifffile is available and it's a TIFF
    if is_tiff and HAS_TIFFFILE:
        return tifffile.imread(path_str)

    raise ImportError(
        "No image loading library available. "
        "Install tifffile (pip install tifffile) or "
        "scikit-image (pip install scikit-image)"
    )


def load_image_stack(
    paths: List[Union[str, Path]],
    dtype: type = None
) -> np.ndarray:
    """
    Load multiple images into a stack.

    Arguments:
        paths: List of paths to load
        dtype: Output dtype (None = keep original)

    Returns:
        3D array of shape (n_images, height, width)
    """
    images = []
    for p in paths:
        img = load_image(p)
        if dtype is not None:
            img = img.astype(dtype)
        images.append(img)

    return np.stack(images, axis=0)


def normalize_image(
    image: np.ndarray,
    method: str = "minmax",
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Arguments:
        image: Input image (any dtype)
        method: Normalization method
            - "minmax": Scale to [0, 1] using min/max
            - "percentile": Use percentile clipping for robustness
            - "zscore": Zero mean, unit std, then scale to [0, 1]
        percentile_low: Lower percentile for "percentile" method (default: 1.0)
        percentile_high: Upper percentile for "percentile" method (default: 99.0)

    Returns:
        Normalized image as float32 in [0, 1]
    """
    img = image.astype(np.float32)

    if method == "minmax":
        img_min = img.min()
        img_max = img.max()
        if img_max - img_min > 0:
            img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

    elif method == "percentile":
        p_low = np.percentile(img, percentile_low)
        p_high = np.percentile(img, percentile_high)
        if p_high - p_low > 1e-6:
            img = np.clip((img - p_low) / (p_high - p_low), 0, 1)
        else:
            img = np.zeros_like(img)

    elif method == "zscore":
        mean = np.mean(img)
        std = np.std(img)
        if std > 0:
            img = (img - mean) / std
            # Scale to [0, 1]
            img_min = img.min()
            img_max = img.max()
            if img_max - img_min > 0:
                img = (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return img.astype(np.float32)


def to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convert image to uint8 (0-255).

    Arguments:
        image: Input image (any dtype, assumes [0,1] if float)

    Returns:
        Image as uint8
    """
    if image.dtype == np.uint8:
        return image

    if image.dtype in [np.float32, np.float64]:
        # Assume [0, 1] range
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)

    if image.dtype == np.uint16:
        # Scale from 16-bit to 8-bit
        return (image / 256).astype(np.uint8)

    # Generic fallback: normalize then convert
    img_norm = normalize_image(image, method="minmax")
    return (img_norm * 255).astype(np.uint8)


def to_float32(image: np.ndarray, normalize: bool = True) -> np.ndarray:
    """
    Convert image to float32.

    Arguments:
        image: Input image (any dtype)
        normalize: If True, normalize to [0, 1]

    Returns:
        Image as float32
    """
    if normalize:
        return normalize_image(image, method="minmax")

    return image.astype(np.float32)


def pad_to_multiple(image: np.ndarray, multiple: int = 32, mode: str = "reflect") -> np.ndarray:
    """
    Pad image so dimensions are multiples of a given number.

    Useful for neural networks that require specific input sizes.

    Arguments:
        image: Input image (2D or 3D)
        multiple: Pad to nearest multiple of this number
        mode: Padding mode for np.pad

    Returns:
        Padded image
    """
    if len(image.shape) == 2:
        h, w = image.shape
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        pad_h = new_h - h
        pad_w = new_w - w
        return np.pad(image, ((0, pad_h), (0, pad_w)), mode=mode)

    elif len(image.shape) == 3:
        h, w, c = image.shape
        new_h = ((h + multiple - 1) // multiple) * multiple
        new_w = ((w + multiple - 1) // multiple) * multiple
        pad_h = new_h - h
        pad_w = new_w - w
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)

    else:
        raise ValueError(f"Expected 2D or 3D image, got shape {image.shape}")


def unpad(image: np.ndarray, original_shape: tuple) -> np.ndarray:
    """
    Remove padding to restore original image size.

    Arguments:
        image: Padded image
        original_shape: Original (h, w) or (h, w, c) shape

    Returns:
        Unpadded image
    """
    if len(original_shape) == 2:
        h, w = original_shape
        return image[:h, :w]
    elif len(original_shape) == 3:
        h, w, c = original_shape
        return image[:h, :w, :c]
    else:
        raise ValueError(f"Expected 2D or 3D shape, got {original_shape}")


def remove_small_holes_compat(
    mask: np.ndarray,
    area_threshold: int,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove small holes from a binary mask with scikit-image version compatibility.

    Handles the API change in scikit-image where the parameter name changed
    from `area_threshold` to `max_size` in version 0.26.

    Arguments:
        mask: Binary mask (will be converted to bool)
        area_threshold: Maximum area of holes to remove
        connectivity: Connectivity for hole detection (1 or 2)

    Returns:
        Binary mask with small holes filled
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for morphology operations")

    bool_mask = mask.astype(bool)
    try:
        # New API (skimage >= 0.26)
        return skimage.morphology.remove_small_holes(
            bool_mask, max_size=area_threshold, connectivity=connectivity
        )
    except TypeError:
        # Old API (skimage < 0.26)
        return skimage.morphology.remove_small_holes(
            bool_mask, area_threshold=area_threshold, connectivity=connectivity
        )


def remove_small_objects_compat(
    labels: np.ndarray,
    min_size: int,
    connectivity: int = 2,
) -> np.ndarray:
    """
    Remove small objects from a label image with scikit-image version compatibility.

    Handles the API change in scikit-image where the parameter name changed
    from `min_size` to `max_size` in version 0.26.

    Arguments:
        labels: Label image (integer array)
        min_size: Minimum size of objects to keep
        connectivity: Connectivity for object detection (1 or 2)

    Returns:
        Label image with small objects removed
    """
    if not HAS_SKIMAGE:
        raise ImportError("scikit-image is required for morphology operations")

    try:
        # New API (skimage >= 0.26)
        return skimage.morphology.remove_small_objects(
            labels, max_size=min_size, connectivity=connectivity
        )
    except TypeError:
        # Old API (skimage < 0.26)
        return skimage.morphology.remove_small_objects(
            labels, min_size=min_size, connectivity=connectivity
        )
