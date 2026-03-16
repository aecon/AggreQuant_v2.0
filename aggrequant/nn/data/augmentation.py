"""
Torchvision v2 augmentation pipelines for aggregate segmentation.

This module provides data augmentation pipelines following nnU-Net best
practices for microscopy image segmentation, using torchvision.transforms.v2
with tv_tensors for joint image/mask handling.

Spatial transforms (flip, rotate, affine) are applied to both image and mask.
Intensity transforms (brightness, contrast, noise, blur) are applied to the
image only. This is handled automatically by tv_tensors types.

Note: Elastic deformation, shear, and random erasing are intentionally excluded
because they distort small aggregate structures unnaturally.

Example:
    >>> from aggrequant.nn.data.augmentation import get_training_augmentation
    >>> transform = get_training_augmentation()
    >>> image, mask = apply_transform(transform, image, mask)
"""

from typing import Tuple

import numpy as np
import torch
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


# ---------------------------------------------------------------------------
# Custom transforms (not available in torchvision v2)
# ---------------------------------------------------------------------------


class RandomRotate90(v2.Transform):
    """Randomly rotate the input by a multiple of 90 degrees."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1).item() > self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        k = torch.randint(1, 4, (1,)).item()  # 90, 180, or 270
        angle = k * 90
        outputs = []
        for inpt in inputs:
            outputs.append(F.rotate(inpt, angle))
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class RandomGamma(v2.Transform):
    """Random gamma correction (image-only).

    Arguments:
        gamma_range: Tuple of (min, max) gamma values. Values <1 brighten,
            >1 darken.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        gamma_range: Tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
    ):
        super().__init__()
        self.gamma_range = gamma_range
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1).item() > self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        gamma = torch.empty(1).uniform_(*self.gamma_range).item()
        outputs = []
        for inpt in inputs:
            if isinstance(inpt, tv_tensors.Mask):
                outputs.append(inpt)  # don't modify masks
            else:
                outputs.append(torch.clamp(inpt.float().pow(gamma), 0.0, 1.0))
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


class MultiplicativeNoise(v2.Transform):
    """Apply multiplicative noise (image-only).

    Multiplies each pixel by a random value drawn from
    Uniform(multiplier_range[0], multiplier_range[1]).

    Arguments:
        multiplier_range: Range of multipliers.
        p: Probability of applying the transform.
    """

    def __init__(
        self,
        multiplier_range: Tuple[float, float] = (0.9, 1.1),
        p: float = 0.5,
    ):
        super().__init__()
        self.multiplier_range = multiplier_range
        self.p = p

    def forward(self, *inputs):
        if torch.rand(1).item() > self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        outputs = []
        for inpt in inputs:
            if isinstance(inpt, tv_tensors.Mask):
                outputs.append(inpt)
            else:
                noise = torch.empty_like(inpt).uniform_(*self.multiplier_range)
                outputs.append(torch.clamp(inpt * noise, 0.0, 1.0))
        return tuple(outputs) if len(outputs) > 1 else outputs[0]


# ---------------------------------------------------------------------------
# Helper: wrap numpy arrays as tv_tensors, call transform, unwrap
# ---------------------------------------------------------------------------


def _wrap_inputs(
    image: np.ndarray, mask: np.ndarray
) -> Tuple[tv_tensors.Image, tv_tensors.Mask]:
    """Convert numpy HW arrays to tv_tensors (C, H, W) tensors.

    Arguments:
        image: Grayscale image as numpy array (H, W), float32 in [0, 1].
        mask: Binary mask as numpy array (H, W), float32.

    Returns:
        Tuple of (tv_tensors.Image, tv_tensors.Mask) with shape (1, H, W).
    """
    img_t = tv_tensors.Image(
        torch.from_numpy(image).unsqueeze(0).float()
    )
    mask_t = tv_tensors.Mask(
        torch.from_numpy(mask).unsqueeze(0).float()
    )
    return img_t, mask_t


def apply_transform(
    transform: v2.Compose,
    image: np.ndarray,
    mask: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply a transform pipeline to an image/mask pair.

    Wraps numpy arrays as tv_tensors, applies the transform, and returns
    plain tensors.

    Arguments:
        transform: A torchvision v2 Compose transform.
        image: Grayscale image (H, W), float32 in [0, 1].
        mask: Binary mask (H, W), float32.

    Returns:
        Tuple of (image_tensor, mask_tensor) each with shape (1, H, W).
    """
    img_tv, mask_tv = _wrap_inputs(image, mask)
    img_out, mask_out = transform(img_tv, mask_tv)
    return img_out, mask_out


# ---------------------------------------------------------------------------
# Training augmentation
# ---------------------------------------------------------------------------


def get_training_augmentation(
    p_spatial: float = 0.5,
    p_intensity: float = 0.5,
    p_noise: float = 0.3,
    p_blur: float = 0.2,
    rotation_limit: int = 90,
    scale_limit: Tuple[float, float] = (0.8, 1.2),
    shift_limit: float = 0.1,
) -> v2.Compose:
    """Get training augmentation pipeline.

    Designed for aggregate segmentation in microscopy images:
    - Spatial: rotation, flip, scale, shift (no elastic/shear — distorts
      small aggregate structures unnaturally)
    - Intensity: brightness, contrast, gamma
    - Noise: Gaussian noise, multiplicative noise
    - Blur: Gaussian blur

    Arguments:
        p_spatial: Probability for affine augmentations.
        p_intensity: Probability for intensity augmentations.
        p_noise: Probability for noise augmentations.
        p_blur: Probability for blur augmentations.
        rotation_limit: Maximum rotation angle in degrees.
        scale_limit: Scale range as (min_scale, max_scale).
        shift_limit: Maximum shift as fraction of image size.

    Returns:
        torchvision v2 Compose pipeline.

    Example:
        >>> transform = get_training_augmentation()
        >>> img_t, mask_t = apply_transform(transform, image, mask)
    """
    transforms = []

    # Spatial augmentations — applied to both image and mask
    transforms.extend([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
    ])

    if p_spatial > 0:
        transforms.append(
            v2.RandomAffine(
                degrees=rotation_limit,
                translate=(shift_limit, shift_limit),
                scale=scale_limit,
                fill=0,
            )
        )

    # Intensity augmentations — image-only (tv_tensors.Mask is untouched)
    if p_intensity > 0:
        transforms.append(
            v2.RandomChoice([
                v2.ColorJitter(brightness=0.2, contrast=0.2),
                RandomGamma(gamma_range=(0.8, 1.2)),
            ])
        )

    # Noise
    if p_noise > 0:
        transforms.append(
            v2.RandomApply(
                [v2.RandomChoice([
                    v2.GaussianNoise(sigma=0.05),
                    MultiplicativeNoise(multiplier_range=(0.9, 1.1)),
                ])],
                p=p_noise,
            )
        )

    # Blur
    if p_blur > 0:
        transforms.append(
            v2.RandomApply(
                [v2.GaussianBlur(kernel_size=(3, 5))],
                p=p_blur,
            )
        )

    return v2.Compose(transforms)
