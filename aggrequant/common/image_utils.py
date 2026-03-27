"""Image loading, normalization, and type conversion utilities."""

from pathlib import Path
from typing import List, Union
import numpy as np

# Optional imports for image loading
try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

try:
    import skimage.io
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS = (".tif", ".tiff")


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

    raise ImportError(
        "No image loading library available. "
        "Install tifffile (pip install tifffile) or "
        "scikit-image (pip install scikit-image)"
    )



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


