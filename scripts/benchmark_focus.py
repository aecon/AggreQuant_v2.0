#!/usr/bin/env python
"""
Benchmark script for focus/blur metrics visualization.

Author: Athena Economides, 2026, UZH
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from aggrequant.quality.focus import compute_patch_focus_maps


# Directories
INPUT_DIR = Path("/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/benchmark_nuclei")
OUTPUT_DIR = INPUT_DIR / "output_focus"

# Focus metrics parameters
PATCH_SIZE = (40, 40)


def load_image(path: Path) -> np.ndarray:
    """Load a TIFF image."""
    return tifffile.imread(str(path))


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range using percentiles."""
    p_low, p_high = np.percentile(image, (1, 99.8))
    if p_high - p_low < 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - p_low) / (p_high - p_low), 0, 1).astype(np.float32)


def create_focus_figure(
    image: np.ndarray,
    title: str,
    patch_size: tuple = PATCH_SIZE,
) -> plt.Figure:
    """Create figure showing image and focus metric maps."""
    maps, ys, xs = compute_patch_focus_maps(image, patch_size=patch_size)

    # Fixed display limits for each metric
    limits = {
        "VarianceLaplacian": (0, 2000),
        "LaplaceEnergy": (0, 2000),
        "Sobel": (0, 200),
        "Brenner": (0, 2e6),
        "FocusScore": (0, 50),
    }

    n_maps = len(maps)
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(4 * (n_maps + 1), 4))

    # Original image
    im = axes[0].imshow(normalize(image), cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Focus metric maps
    for k, (name, m) in enumerate(maps.items()):
        vmin, vmax = limits[name]
        im = axes[k + 1].imshow(m, cmap="viridis") #, vmin=vmin, vmax=vmax)
        axes[k + 1].set_title(name)
        axes[k + 1].axis("off")
        fig.colorbar(im, ax=axes[k + 1], fraction=0.046, pad=0.04)

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

        fig = create_focus_figure(image, title=img_path.stem)
        output_path = OUTPUT_DIR / f"{img_path.stem}_focus.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("=" * 50)
    print(f"Done. Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
