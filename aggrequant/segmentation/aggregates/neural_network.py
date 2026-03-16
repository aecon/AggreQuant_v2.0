"""Neural network-based aggregate segmentation."""

import numpy as np
from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from aggrequant.segmentation.base import BaseSegmenter
from aggrequant.nn.utils import get_device


class NeuralNetworkSegmenter(BaseSegmenter):
    """Neural network-based aggregate segmentation.

    Thin wrapper around nn.inference that conforms to the BaseSegmenter
    interface. Model architecture is read from the checkpoint (saved by
    Trainer.save_checkpoint), so only a weights path is needed.

    The pipeline:
    1. Load model from checkpoint (lazy, on first call)
    2. Run inference (full-resolution with OOM fallback to tiled)
    3. Post-process probability map into instance labels
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        weights_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.5,
        remove_objects_below: int = 9,
        fill_holes_below: int = 6000,
        device: Optional[str] = None,
        verbose: bool = False,
        debug: bool = False,
    ):
        """Initialize neural network segmenter.

        Arguments:
            model: Pre-loaded PyTorch model (skips checkpoint loading)
            weights_path: Path to checkpoint file (must contain model_config)
            threshold: Probability threshold for binarization
            remove_objects_below: Remove objects smaller than this (pixels)
            fill_holes_below: Fill holes smaller than this (pixels)
            device: Device to use ('cuda', 'cpu', or None for auto)
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        super().__init__(verbose=verbose, debug=debug)

        self._model = model
        self.weights_path = Path(weights_path) if weights_path else None
        self.threshold = threshold
        self.remove_objects_below = remove_objects_below
        self.fill_holes_below = fill_holes_below
        self.device = get_device(device)

        self._debug(f"Using device: {self.device}")

    @property
    def name(self) -> str:
        return "NeuralNetworkSegmenter"

    @property
    def model(self) -> nn.Module:
        """Lazy-load model on first access."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> nn.Module:
        """Load model architecture + weights from checkpoint."""
        from aggrequant.nn.architectures.unet import UNet

        if self.weights_path is None or not self.weights_path.exists():
            raise ValueError(
                f"Weights not found: {self.weights_path}"
            )

        self._log(f"Loading model from {self.weights_path}")
        checkpoint = torch.load(
            self.weights_path, map_location=self.device, weights_only=True,
        )

        # Read architecture config from checkpoint
        if 'model_config' not in checkpoint:
            raise ValueError(
                f"Checkpoint missing 'model_config'. "
                f"Re-train with the current Trainer to embed architecture config."
            )

        model = UNet(**checkpoint['model_config'])

        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.to(self.device)
        model.eval()
        return model

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment aggregates in the input image.

        Arguments:
            image: Input grayscale image (2D array)

        Returns:
            Instance segmentation labels (uint32).
            0 = background, 1+ = individual aggregates.
        """
        from aggrequant.nn.inference import predict, postprocess_predictions

        self._debug(f"Input image shape: {image.shape}, dtype: {image.dtype}")

        prob_map = predict(self.model, image, device=self.device)

        labels = postprocess_predictions(
            prob_map,
            threshold=self.threshold,
            remove_objects_below=self.remove_objects_below,
            fill_holes_below=self.fill_holes_below,
        )

        self._log(f"Detected {labels.max()} aggregates")
        return labels
