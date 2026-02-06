#!/usr/bin/env python
"""
Benchmark script comparing StarDist vs Cellpose for nuclei segmentation.

Generates side-by-side overlay figures for qualitative evaluation across
different microscopy conditions (blur, contrast variations, etc.).

For methods paper supplementary material.

Author: Athena Economides, 2026, UZH
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.segmentation

from aggrequant.common.image_utils import load_image, normalize_image
from aggrequant.segmentation.nuclei.stardist import StarDistSegmenter


# Directories
SCRIPT_DIR = Path(__file__).parent
INPUT_DIR = SCRIPT_DIR / "benchmark_images"
OUTPUT_DIR = SCRIPT_DIR / "benchmark_output"


def load_cellpose_nuclei_model(gpu: bool = True):
    """Load Cellpose with nuclei model."""
    from cellpose import models
    return models.Cellpose(gpu=gpu, model_type="nuclei")


def segment_cellpose(model, image: np.ndarray) -> np.ndarray:
    """Run Cellpose nuclei segmentation."""
    masks, _, _, _ = model.eval(image, channels=[0, 0])
    return masks


def create_overlay(image: np.ndarray, labels: np.ndarray, color=(1, 0, 0)) -> np.ndarray:
    """Create RGB overlay with label contours."""
    # Normalize image to [0, 1]
    img_norm = normalize_image(image, method="percentile")

    # Create RGB image
    rgb = np.stack([img_norm] * 3, axis=-1)

    # Find boundaries
    if labels.max() > 0:
        boundaries = skimage.segmentation.find_boundaries(labels, mode="outer")
        for i, c in enumerate(color):
            rgb[:, :, i][boundaries] = c

    return rgb


def create_comparison_figure(
    image: np.ndarray,
    stardist_labels: np.ndarray,
    cellpose_labels: np.ndarray,
    title: str,
) -> plt.Figure:
    """Create side-by-side comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Raw image
    img_norm = normalize_image(image, method="percentile")
    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Raw Image")
    axes[0].axis("off")

    # StarDist overlay
    stardist_overlay = create_overlay(image, stardist_labels, color=(1, 0, 0))
    axes[1].imshow(stardist_overlay)
    axes[1].set_title(f"StarDist (n={stardist_labels.max()})")
    axes[1].axis("off")

    # Cellpose overlay
    cellpose_overlay = create_overlay(image, cellpose_labels, color=(0, 1, 0))
    axes[2].imshow(cellpose_overlay)
    axes[2].set_title(f"Cellpose (n={cellpose_labels.max()})")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    return fig


def main():
    # Check input directory
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please create the directory and add benchmark images.")
        return

    # Find images
    image_files = sorted(INPUT_DIR.glob("*.tif"))
    if not image_files:
        print(f"No .tif files found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize models
    print("Loading StarDist model...")
    stardist = StarDistSegmenter(verbose=False)

    print("Loading Cellpose model...")
    cellpose_model = load_cellpose_nuclei_model(gpu=True)

    # Process each image
    print("\nProcessing images...")
    print("=" * 50)

    for img_path in image_files:
        print(f"  {img_path.name}")

        # Load image
        image = load_image(img_path)

        # Segment with both models
        stardist_labels = stardist.segment(image)
        cellpose_labels = segment_cellpose(cellpose_model, image)

        # Create comparison figure
        fig = create_comparison_figure(
            image,
            stardist_labels,
            cellpose_labels,
            title=img_path.stem,
        )

        # Save figure
        output_path = OUTPUT_DIR / f"{img_path.stem}_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("=" * 50)
    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
