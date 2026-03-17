"""Inference utilities for aggregate segmentation models.

Provides two inference modes for large microscopy images:
1. Full-resolution: pad to multiple of 16, single forward pass (fast, needs VRAM)
2. Tiled with Gaussian blending: overlapping tiles with smooth blending (fallback)

Auto-detect mode tries full-resolution first and falls back to tiled on OOM.

Example:
    >>> from aggrequant.nn.inference import load_model, predict
    >>>
    >>> model = load_model("checkpoints/best.pt")
    >>> prob_map = predict(model, image)
    >>>
    >>> # Force tiled inference
    >>> prob_map = predict_tiled(model, image, tile_size=256, stride=128)
"""

from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from aggrequant.common.image_utils import normalize_image
from aggrequant.common.logging import get_logger
from aggrequant.nn.utils import get_device

logger = get_logger(__name__)


def load_model(
    checkpoint_path: Union[str, Path],
    device: torch.device = None,
) -> nn.Module:
    """Load a trained model from a self-contained checkpoint.

    Checkpoints saved by the Trainer contain both the architecture config
    (from ``UNet.get_config()``) and the trained weights, so no external
    information is needed to reconstruct the model.

    Arguments:
        checkpoint_path: Path to a ``.pt`` checkpoint file
        device: Device to load the model onto (default: auto-detect)

    Returns:
        Model in eval mode, ready for inference

    Example:
        >>> model = load_model("training_output/baseline/checkpoints/best.pt")
        >>> prob_map = predict(model, image)
    """
    from aggrequant.nn.architectures.unet import UNet

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False,
    )

    if "model_config" not in checkpoint:
        raise ValueError(
            f"Checkpoint {checkpoint_path} does not contain 'model_config'. "
            "Was it saved by the Trainer?"
        )

    model = UNet(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if device is not None:
        model = model.to(get_device(device))

    return model


def _to_tensor(image: np.ndarray) -> torch.Tensor:
    """Convert a normalized 2D numpy image to a (1, 1, H, W) tensor."""
    return torch.from_numpy(image).unsqueeze(0).unsqueeze(0).float()


def _pad_to_multiple(image: np.ndarray, multiple: int = 16) -> np.ndarray:
    """Pad image to nearest multiple of `multiple` using reflection.

    Arguments:
        image: 2D array (H, W)
        multiple: Pad dimensions to be divisible by this (default: 16)

    Returns:
        Padded image (H_pad, W_pad) where both dims are multiples of `multiple`
    """
    h, w = image.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")


def _gaussian_kernel_2d(size: int, sigma: float = None) -> np.ndarray:
    """Create a 2D Gaussian weight map for tile blending.

    Center pixels get weight ~1, edges fade to ~0. This eliminates
    hard seams when stitching overlapping tile predictions.

    Arguments:
        size: Tile size (square)
        sigma: Gaussian sigma (default: size / 4)

    Returns:
        2D weight map of shape (size, size), values in (0, 1]
    """
    if sigma is None:
        sigma = size / 4.0

    coords = np.arange(size) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = np.outer(g, g)

    # Normalize so center is 1.0
    kernel = kernel / kernel.max()
    return kernel.astype(np.float32)


@torch.no_grad()
def predict_full(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device = None,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Run inference on a full image in one forward pass.

    Pads image to nearest multiple of 16 (not power-of-2), runs the model,
    applies sigmoid, and crops back to original size.

    Arguments:
        model: Trained segmentation model
        image: Input grayscale image (H, W), any dtype
        device: Device to run on (default: auto-detect)
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization

    Returns:
        Probability map as float32 array (H, W), values in [0, 1]
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()

    h, w = image.shape

    # Normalize and pad
    normalized = normalize_image(
        image, method="percentile",
        percentile_low=percentile_low, percentile_high=percentile_high,
    )
    padded = _pad_to_multiple(normalized, multiple=16)

    # Forward pass
    tensor = _to_tensor(padded).to(device)
    output = model(tensor)

    # Handle deep supervision output
    if isinstance(output, tuple):
        output = output[0]

    prob = torch.sigmoid(output).squeeze().cpu().numpy()

    # Crop back to original size
    return prob[:h, :w]


@torch.no_grad()
def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int = 256,
    stride: int = 128,
    device: torch.device = None,
    batch_size: int = 16,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Run inference using overlapping tiles with Gaussian blending.

    Splits the image into overlapping tiles, predicts each, and blends
    predictions using Gaussian weights (center-heavy, edges fade) to
    avoid hard seams at tile boundaries.

    Arguments:
        model: Trained segmentation model
        image: Input grayscale image (H, W), any dtype
        tile_size: Size of each square tile in pixels (default: 256)
        stride: Step between tiles (default: 128, i.e. 50% overlap)
        device: Device to run on (default: auto-detect)
        batch_size: Number of tiles per batch (default: 16)
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization

    Returns:
        Probability map as float32 array (H, W), values in [0, 1]
    """
    device = get_device(device)
    model = model.to(device)
    model.eval()

    h, w = image.shape

    # Normalize the full image (not per-tile — consistent normalization)
    normalized = normalize_image(
        image, method="percentile",
        percentile_low=percentile_low, percentile_high=percentile_high,
    )

    # Pad so tiles cover the entire image
    pad_h = (tile_size - h % stride) % stride
    pad_w = (tile_size - w % stride) % stride
    padded = np.pad(normalized, ((0, pad_h), (0, pad_w)), mode="reflect")
    ph, pw = padded.shape

    # Compute tile positions
    positions = []
    for y in range(0, ph - tile_size + 1, stride):
        for x in range(0, pw - tile_size + 1, stride):
            positions.append((y, x))

    # Gaussian blending weight
    weight = _gaussian_kernel_2d(tile_size)

    # Accumulate weighted predictions
    prediction_sum = np.zeros((ph, pw), dtype=np.float32)
    weight_sum = np.zeros((ph, pw), dtype=np.float32)

    # Process tiles in batches
    for batch_start in range(0, len(positions), batch_size):
        batch_positions = positions[batch_start:batch_start + batch_size]

        # Extract tiles
        tiles = []
        for y, x in batch_positions:
            tiles.append(padded[y:y + tile_size, x:x + tile_size])

        # Stack into batch tensor (B, 1, tile_size, tile_size)
        batch_tensor = torch.from_numpy(
            np.stack(tiles)
        ).unsqueeze(1).float().to(device)

        # Forward pass
        output = model(batch_tensor)
        if isinstance(output, tuple):
            output = output[0]

        probs = torch.sigmoid(output).squeeze(1).cpu().numpy()

        # Accumulate with Gaussian weights
        for i, (y, x) in enumerate(batch_positions):
            prediction_sum[y:y + tile_size, x:x + tile_size] += probs[i] * weight
            weight_sum[y:y + tile_size, x:x + tile_size] += weight

    # Normalize by total weight
    prediction = prediction_sum / (weight_sum + 1e-8)

    # Crop back to original size
    return prediction[:h, :w]


@torch.no_grad()
def predict(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device = None,
    tile_size: int = 256,
    stride: int = 128,
    batch_size: int = 16,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """Run inference with automatic mode selection.

    Tries full-resolution first. Falls back to tiled inference on CUDA OOM.

    Arguments:
        model: Trained segmentation model
        image: Input grayscale image (H, W), any dtype
        device: Device to run on (default: auto-detect)
        tile_size: Tile size for fallback tiled mode (default: 256)
        stride: Stride for fallback tiled mode (default: 128)
        batch_size: Batch size for tiled mode (default: 16)
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization

    Returns:
        Probability map as float32 array (H, W), values in [0, 1]
    """
    try:
        return predict_full(
            model, image, device=device,
            percentile_low=percentile_low, percentile_high=percentile_high,
        )
    except torch.cuda.OutOfMemoryError:
        logger.warning(
            "Full-resolution inference failed (OOM). "
            "Falling back to tiled inference."
        )
        torch.cuda.empty_cache()
        return predict_tiled(
            model, image, tile_size=tile_size, stride=stride,
            device=device, batch_size=batch_size,
            percentile_low=percentile_low, percentile_high=percentile_high,
        )


def postprocess_predictions(
    probability_map: np.ndarray,
    threshold: float = 0.5,
    remove_objects_below: int = 9,
    fill_holes_below: int = 6000,
) -> np.ndarray:
    """Convert probability map to instance segmentation labels.

    Arguments:
        probability_map: Float32 array (H, W) with values in [0, 1]
        threshold: Probability threshold for binarization (default: 0.5)
        remove_objects_below: Remove objects smaller than this many pixels (default: 9)
        fill_holes_below: Fill holes smaller than this many pixels (default: 6000)

    Returns:
        Instance label array (uint32), 0 = background, 1+ = individual objects
    """
    import skimage.morphology

    from aggrequant.segmentation.postprocessing import (
        remove_small_holes,
        remove_small_objects,
    )

    # Binarize
    binary = (probability_map > threshold).astype(np.uint8)

    # Fill small holes
    no_holes = remove_small_holes(binary, max_size=fill_holes_below, connectivity=2)

    # Connected components
    labels = skimage.morphology.label(no_holes, connectivity=2)

    # Remove small objects
    no_small = remove_small_objects(labels, max_size=remove_objects_below, connectivity=2)

    # Relabel consecutively
    unique = np.unique(no_small)
    lut = np.zeros(int(unique.max()) + 1, dtype=np.uint32)
    lut[unique] = np.arange(len(unique), dtype=np.uint32)
    labels = lut[no_small]

    return labels.astype(np.uint32)
