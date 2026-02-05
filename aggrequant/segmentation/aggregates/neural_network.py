"""
Neural network-based aggregate segmentation.

Uses PyTorch UNet models for semantic segmentation with
sliding window inference and patch stitching.
"""

import numpy as np
import skimage.morphology
from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
import torch.nn as nn

from ..base import BaseSegmenter
from aggrequant.common.image_utils import (
    remove_small_holes_compat,
    remove_small_objects_compat,
)
from aggrequant.nn.utils import get_device


# Default parameters
PATCH_SIZE = 128
STRIDE = 32
BATCH_SIZE = 64
PROBABILITY_THRESHOLD = 0.7
SMALL_HOLE_AREA_THRESHOLD = 6000
MIN_AGGREGATE_AREA = 9


class NeuralNetworkSegmenter(BaseSegmenter):
    """
    Neural network-based aggregate segmentation.

    Uses sliding window inference with overlapping patches
    for seamless segmentation of large images.

    The pipeline:
    1. Pad image to power of 2
    2. Extract overlapping patches
    3. Preprocess patches (normalize to [0, 1])
    4. Batch inference with PyTorch model
    5. Stitch patches with overlap averaging
    6. Threshold probability map
    7. Morphological cleanup
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        weights_path: Optional[Union[str, Path]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        patch_size: int = PATCH_SIZE,
        stride: int = STRIDE,
        batch_size: int = BATCH_SIZE,
        probability_threshold: float = PROBABILITY_THRESHOLD,
        small_hole_area: int = SMALL_HOLE_AREA_THRESHOLD,
        min_aggregate_area: int = MIN_AGGREGATE_AREA,
        device: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """
        Initialize neural network segmenter.

        Arguments:
            model: Pre-loaded PyTorch model (if None, will load from weights_path)
            weights_path: Path to model weights file
            unet_config: UNet configuration dict (e.g., {"encoder_block": "residual"})
            patch_size: Size of input patches
            stride: Stride between patches (smaller = more overlap)
            batch_size: Batch size for inference
            probability_threshold: Threshold for binary segmentation
            small_hole_area: Area threshold for hole filling
            min_aggregate_area: Minimum aggregate area in pixels
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Print progress messages
            debug: Print detailed debug information

        Example:
            >>> segmenter = NeuralNetworkSegmenter(
            ...     weights_path="model.pt",
            ...     unet_config={"encoder_block": "residual", "use_attention_gates": True},
            ... )
        """
        super().__init__(verbose=verbose, debug=debug)

        self._model = model
        self.weights_path = Path(weights_path) if weights_path else None
        self.unet_config = unet_config or {}
        self.patch_size = patch_size
        self.stride = stride
        self.batch_size = batch_size
        self.probability_threshold = probability_threshold
        self.small_hole_area = small_hole_area
        self.min_aggregate_area = min_aggregate_area

        # Auto-select device
        self.device = get_device(device)

        self._debug(f"Using device: {self.device}")

    @property
    def name(self) -> str:
        return "NeuralNetworkSegmenter"

    @property
    def model(self) -> nn.Module:
        """Lazy load the model."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> nn.Module:
        """Load model from weights."""
        from aggrequant.nn.architectures import UNet

        if self.weights_path and self.weights_path.exists():
            self._log(f"Loading model from {self.weights_path}")

            # Create model architecture with config
            model = UNet(in_channels=1, out_channels=1, **self.unet_config)

            # Load weights
            checkpoint = torch.load(self.weights_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)

            model.to(self.device)
            model.eval()
            return model

        else:
            raise ValueError(
                f"No model provided and weights_path not found: {self.weights_path}"
            )

    def set_model(self, model: nn.Module):
        """Set the model directly."""
        self._model = model.to(self.device)
        self._model.eval()

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment aggregates in the input image.

        Arguments:
            image: Input grayscale image (2D array)

        Returns:
            labels: Instance segmentation labels (uint32)
                    0 = background, 1+ = individual aggregates
        """
        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        # Get probability map
        probability = self.predict_probability(image)

        # Threshold
        segmented = (probability > self.probability_threshold).astype(np.uint8)

        # Connected components
        labels = skimage.morphology.label(segmented, connectivity=2)
        self._debug(f"Initial connected components: {labels.max()}")

        # Remove small holes
        no_holes = remove_small_holes_compat(
            segmented, area_threshold=self.small_hole_area, connectivity=2
        )
        labels = skimage.morphology.label(no_holes, connectivity=2)
        self._debug(f"After removing small holes: {labels.max()}")

        # Remove small objects
        no_small = remove_small_objects_compat(
            labels, min_size=self.min_aggregate_area, connectivity=2
        )
        labels = skimage.morphology.label(no_small, connectivity=2)
        self._debug(f"After removing small objects: {labels.max()}")

        self._log(f"Detected {labels.max()} aggregates")
        return labels.astype(np.uint32)

    def predict_probability(self, image: np.ndarray) -> np.ndarray:
        """
        Predict probability map using sliding window inference.

        Arguments:
            image: Input grayscale image

        Returns:
            probability: Probability map in [0, 1]
        """
        original_shape = image.shape

        # Pad to power of 2
        padded, target_size = self._pad_to_power2(image)
        self._debug(f"Padded from {original_shape} to {padded.shape}")

        # Extract patches
        patches = self._extract_patches(padded)
        self._debug(f"Extracted {len(patches)} patches")

        # Preprocess patches
        patches = np.array([self._preprocess(p) for p in patches], dtype=np.float32)

        # Batch inference
        predictions = self._batch_inference(patches)

        # Stitch patches
        stitched = self._stitch_patches(predictions, target_size)

        # Crop to original size
        result = stitched[:original_shape[0], :original_shape[1]]

        return result

    def _pad_to_power2(self, image: np.ndarray) -> tuple:
        """Pad image to next power of 2."""
        n = image.shape[0]
        target = self._next_power2(n)

        padded = np.zeros((target, target), dtype=image.dtype)
        padded[:n, :n] = image

        return padded, target

    def _next_power2(self, x: int) -> int:
        """Find smallest power of 2 >= x."""
        if x == 0:
            return 1
        return 2 ** (x - 1).bit_length()

    def _extract_patches(self, image: np.ndarray) -> list:
        """Extract overlapping patches using sliding window."""
        n = image.shape[0]
        patches = []

        # Number of patches per dimension
        n_patches = (n - self.patch_size) // self.stride + 1

        for i in range(n_patches):
            for j in range(n_patches):
                x0 = i * self.stride
                y0 = j * self.stride
                x1 = x0 + self.patch_size
                y1 = y0 + self.patch_size

                # Handle edge cases
                if x1 > n:
                    x0 = n - self.patch_size
                    x1 = n
                if y1 > n:
                    y0 = n - self.patch_size
                    y1 = n

                patch = image[x0:x1, y0:y1].astype(np.uint16)
                patches.append(patch)

        return patches

    def _preprocess(self, patch: np.ndarray) -> np.ndarray:
        """Preprocess patch: normalize to [0, 1]."""
        patch = patch.astype(np.float32)
        max_val = patch.max()
        if max_val > 0:
            patch = patch / max_val
        return patch

    def _batch_inference(self, patches: np.ndarray) -> np.ndarray:
        """Run batch inference on patches."""
        n_patches = len(patches)
        predictions = []

        # Add channel dimension
        patches = patches[:, np.newaxis, :, :]  # (N, 1, H, W)

        with torch.no_grad():
            for i in range(0, n_patches, self.batch_size):
                batch = patches[i:i + self.batch_size]
                batch_tensor = torch.from_numpy(batch).to(self.device)

                # Forward pass
                output = self.model(batch_tensor)

                # Handle deep supervision output
                if isinstance(output, tuple):
                    output = output[0]

                # Apply sigmoid if needed
                if output.min() < 0 or output.max() > 1:
                    output = torch.sigmoid(output)

                # Move to CPU and convert
                pred = output.cpu().numpy()
                predictions.append(pred)

        # Concatenate all predictions
        predictions = np.concatenate(predictions, axis=0)

        # Remove channel dimension: (N, 1, H, W) -> (N, H, W)
        predictions = predictions[:, 0, :, :]

        return predictions

    def _stitch_patches(self, predictions: np.ndarray, target_size: int) -> np.ndarray:
        """Stitch patches back together with overlap averaging."""
        result = np.zeros((target_size, target_size), dtype=np.float32)
        counter = np.zeros((target_size, target_size), dtype=np.float32)

        n_patches_per_dim = (target_size - self.patch_size) // self.stride + 1
        ds = self.stride

        for i in range(n_patches_per_dim):
            for j in range(n_patches_per_dim):
                k = i * n_patches_per_dim + j

                # Calculate coordinates
                x0 = i * ds
                y0 = j * ds
                x1 = x0 + self.patch_size
                y1 = y0 + self.patch_size

                # Handle edge cases
                if x1 > target_size:
                    x0 = target_size - self.patch_size
                    x1 = target_size
                if y1 > target_size:
                    y0 = target_size - self.patch_size
                    y1 = target_size

                # Determine which portion of the patch to use
                # For non-edge patches, use only the central region
                z0, z1 = 0, self.patch_size
                h0, h1 = 0, self.patch_size

                if i > 0:
                    x0 = x0 + ds
                    z0 = ds
                if i < n_patches_per_dim - 1:
                    x1 = x1 - ds
                    z1 = self.patch_size - ds
                if j > 0:
                    y0 = y0 + ds
                    h0 = ds
                if j < n_patches_per_dim - 1:
                    y1 = y1 - ds
                    h1 = self.patch_size - ds

                # Add prediction
                result[x0:x1, y0:y1] += predictions[k, z0:z1, h0:h1]
                counter[x0:x1, y0:y1] += 1

        # Average overlapping regions
        counter[counter == 0] = 1  # Avoid division by zero
        result = result / counter

        return result
