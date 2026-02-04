"""
Image utility functions for normalization and type conversion.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import numpy as np


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize image to [0, 1] range.

    Arguments:
        image: Input image (any dtype)
        method: Normalization method
            - "minmax": Scale to [0, 1] using min/max
            - "percentile": Use 1st and 99th percentile for robustness
            - "zscore": Zero mean, unit std, then scale to [0, 1]

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
        p1 = np.percentile(img, 1)
        p99 = np.percentile(img, 99)
        if p99 - p1 > 0:
            img = np.clip((img - p1) / (p99 - p1), 0, 1)
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
