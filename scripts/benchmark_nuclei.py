#!/usr/bin/env python
"""Benchmark script comparing StarDist vs Cellpose for nuclei segmentation."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import skimage.filters
import skimage.morphology
import skimage.segmentation
import tifffile
from csbdeep.utils import normalize  # percentile normalization to range [0,1]


# Directories
INPUT_DIR = Path("/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/benchmark_nuclei")
OUTPUT_DIR = INPUT_DIR / "output"

# StarDist preprocessing parameters
SIGMA_DENOISE = 2.0
SIGMA_BACKGROUND = 50.0

MIN_NUCLEUS_AREA = 300
MAX_NUCLEUS_AREA = 15000


def load_image(path: Path) -> np.ndarray:
    """Load a TIFF image."""
    return tifffile.imread(str(path))


def preprocess_background_normalization(image: np.ndarray) -> np.ndarray:
    """Preprocess image for StarDist (denoise + background normalization)."""
    img = image.astype(np.float32)
    denoised = skimage.filters.gaussian(img, sigma=SIGMA_DENOISE)
    background = skimage.filters.gaussian(denoised, sigma=SIGMA_BACKGROUND, mode='nearest', preserve_range=True)
    normalized = denoised / (background + 1e-8)
    return normalized


def segment_stardist(image: np.ndarray, preprocess: bool = False) -> np.ndarray:
    """Run StarDist nuclei segmentation."""
    from stardist.models import StarDist2D
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    if preprocess:
        input_image = preprocess_background_normalization(image)
    else:
        input_image = image
    # Stardist expects normalized data in [0,1] range
    labels, _ = model.predict_instances(normalize(input_image), predict_kwargs=dict(verbose=False))
    return labels


def segment_cellpose(image: np.ndarray, gpu: bool = True, preprocess: bool = False) -> np.ndarray:
    """Run Cellpose nuclei segmentation.
    models in v3: 'cyto3', 'cyto2', 'nuclei'"""
    from cellpose import models
    model = models.Cellpose(gpu=gpu, model_type="nuclei")
    if preprocess:
        input_image = preprocess_background_normalization(image)
    else:
        input_image = image
    # if NUCLEUS channel does not exist, set the second channel to 0
    channels = [[0,0]]
    masks, _, _, _ = model.eval(input_image, channels=channels)
    return masks


def create_overlay(image: np.ndarray, labels: np.ndarray, color=(1, 0, 0)) -> np.ndarray:
    """Create RGB overlay with label contours."""
    img_norm = normalize(image)
    rgb = np.stack([img_norm] * 3, axis=-1)

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
    img_norm = normalize(image)
    axes[0].imshow(img_norm, cmap="gray")
    axes[0].set_title("Raw Image")
    axes[0].axis("off")

    # StarDist overlay
    stardist_overlay = create_overlay(image, stardist_labels, color=(1, 0, 1))
    axes[1].imshow(stardist_overlay)
    axes[1].set_title(f"StarDist (n={stardist_labels.max()})")
    axes[1].axis("off")

    # Cellpose overlay
    cellpose_overlay = create_overlay(image, cellpose_labels, color=(1, 0, 1))
    axes[2].imshow(cellpose_overlay)
    axes[2].set_title(f"Cellpose (n={cellpose_labels.max()})")
    axes[2].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    return fig


def main():
    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please create the directory and add benchmark images.")
        return

    image_files = sorted(INPUT_DIR.glob("*.tif"))
    if not image_files:
        print(f"No .tif files found in {INPUT_DIR}")
        return

    print(f"Found {len(image_files)} images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nProcessing images...")
    print("=" * 50)

    for img_path in image_files:
        print(f"  {img_path.name}")

        image = load_image(img_path)

        stardist_labels = segment_stardist(image, preprocess=False)
        cellpose_labels = segment_cellpose(image, gpu=True, preprocess=False)

        fig = create_comparison_figure(
            image,
            stardist_labels,
            cellpose_labels,
            title=img_path.stem,
        )

        output_path = OUTPUT_DIR / f"{img_path.stem}_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("=" * 50)
    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
