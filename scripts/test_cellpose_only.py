#!/usr/bin/env python
"""Minimal test: Cellpose only, no TensorFlow/StarDist."""

import numpy as np
import tifffile

# Load a test image
image_path = "data/test/B - 02(fld 1 wv 631 - FarRed).tif"
print(f"Loading image: {image_path}")
img = tifffile.imread(image_path)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# Load Cellpose model
print("Loading Cellpose model...")
from cellpose import models
model = models.Cellpose(gpu=True, model_type='cyto3')
print("Model loaded successfully!")

# Create dummy nuclei mask (just for testing)
nuclei_mask = np.zeros_like(img, dtype=np.float32)

# Create 2-channel input
n = img.shape[0]
input_image = np.zeros((2, n, n))
input_image[0, :, :] = img
input_image[1, :, :] = nuclei_mask

# Run segmentation
print("Running segmentation...")
masks, flows, styles, diams = model.eval(
    input_image,
    channels=[1, 2],
)

print(f"Segmentation complete! Found {masks.max()} cells")
