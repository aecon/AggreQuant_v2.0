"""
Select 10 images per difficulty category from a pool of DAPI (Blue) images.

Computes image-level metrics and ranks images to fill 9 categories:
  01_low_confluency, 02_high_confluency, 03_clustered_touching,
  04_mitotic, 05_defocused, 06_flatfield_inhomogeneity,
  07_low_intensity, 08_high_intensity, 09_debris

Usage:
    conda activate AggreQuant
    python select_images.py --input-dir /path/to/SAMPLES --output-dir /path/to/NUCLEI-BENCHMARK

The script copies (not moves) selected images into the category subfolders
and generates:
  - selection_metrics.csv  (all images with all computed metrics)
  - selection_summary.csv  (selected images with assigned category)
  - thumbnails/            (contact sheets per category for visual QC)
"""

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage import filters, morphology, measure
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(img: np.ndarray) -> dict:
    """Compute all selection metrics for a single DAPI image."""
    img_f = img.astype(np.float64)
    h, w = img_f.shape

    # --- Basic intensity ---
    mean_intensity = img_f.mean()
    median_intensity = np.median(img_f)
    max_intensity = img_f.max()
    std_intensity = img_f.std()

    # --- Foreground area fraction (proxy for confluency) ---
    # Otsu threshold on the image
    thresh = filters.threshold_otsu(img)
    binary = img_f > thresh
    # Clean up small noise
    binary_clean = morphology.remove_small_objects(binary, min_size=100)
    area_fraction = binary_clean.sum() / binary_clean.size

    # --- Focus metric: variance of Laplacian ---
    laplacian = ndimage.laplace(img_f)
    var_laplacian = laplacian.var()

    # --- Foreground intensity (decorrelated from confluency) ---
    fg_pixels = img_f[binary_clean]
    if len(fg_pixels) > 0:
        foreground_mean_intensity = fg_pixels.mean()
        foreground_median_intensity = np.median(fg_pixels)
    else:
        foreground_mean_intensity = 0
        foreground_median_intensity = 0

    # --- Flat-field: background illumination gradient ---
    # Smooth heavily to estimate illumination field, ignoring individual cells
    from scipy.ndimage import gaussian_filter
    bg_field = gaussian_filter(img_f, sigma=150)
    bg_min = bg_field.min()
    bg_max = bg_field.max()
    flatfield_ratio = bg_max / (bg_min + 1e-8)
    # Also compute gradient magnitude of the smoothed field
    bg_gy, bg_gx = np.gradient(bg_field)
    flatfield_gradient = np.sqrt(bg_gy**2 + bg_gx**2).mean()

    # --- Intensity skewness (high skew = bright outliers / debris) ---
    centered = img_f - mean_intensity
    skewness = (centered ** 3).mean() / (std_intensity ** 3 + 1e-8)

    # --- Connected component analysis on foreground ---
    labeled, n_objects = ndimage.label(binary_clean)
    if n_objects > 0:
        props = measure.regionprops(labeled)
        areas = np.array([p.area for p in props])
        largest_cc_area = areas.max()
        mean_cc_area = areas.mean()
        median_cc_area = np.median(areas)
        # Clustering metric: how much bigger is the largest CC vs typical nucleus
        # High ratio = merged/touching nuclei, independent of overall confluency
        cc_size_ratio = largest_cc_area / (median_cc_area + 1e-8)
        # Fraction of foreground in oversized CCs (>2x median)
        oversized_thresh = 2.0 * median_cc_area
        oversized_fraction = areas[areas > oversized_thresh].sum() / (areas.sum() + 1e-8)
        # Eccentricity stats
        eccentricities = np.array([p.eccentricity for p in props])
        mean_eccentricity = eccentricities.mean()
        high_eccent_fraction = (eccentricities > 0.85).sum() / len(eccentricities)
    else:
        largest_cc_area = 0
        mean_cc_area = 0
        median_cc_area = 0
        cc_size_ratio = 0
        oversized_fraction = 0
        mean_eccentricity = 0
        high_eccent_fraction = 0

    # --- Debris: bright outlier objects relative to typical foreground ---
    if foreground_median_intensity > 0 and n_objects > 0:
        # Objects with peak intensity > 3x the median foreground
        debris_thresh = 3.0 * foreground_median_intensity
        debris_mask = img_f > debris_thresh
        debris_labeled, n_debris = ndimage.label(debris_mask)
        if n_debris > 0:
            debris_props = measure.regionprops(debris_labeled)
            # Small bright objects = debris; large bright = saturated nuclei
            n_debris_small = sum(1 for p in debris_props if p.area < 500)
            # Intensity ratio of brightest object to median foreground
            peak_ratio = max_intensity / foreground_median_intensity
        else:
            n_debris_small = 0
            peak_ratio = 1.0
    else:
        n_debris_small = 0
        peak_ratio = 1.0

    return {
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "max_intensity": max_intensity,
        "std_intensity": std_intensity,
        "area_fraction": area_fraction,
        "var_laplacian": var_laplacian,
        "foreground_mean_intensity": foreground_mean_intensity,
        "foreground_median_intensity": foreground_median_intensity,
        "flatfield_ratio": flatfield_ratio,
        "flatfield_gradient": flatfield_gradient,
        "skewness": skewness,
        "n_objects": n_objects,
        "largest_cc_area": largest_cc_area,
        "mean_cc_area": mean_cc_area,
        "median_cc_area": median_cc_area,
        "cc_size_ratio": cc_size_ratio,
        "oversized_fraction": oversized_fraction,
        "mean_eccentricity": mean_eccentricity,
        "high_eccent_fraction": high_eccent_fraction,
        "n_debris_small": n_debris_small,
        "peak_ratio": peak_ratio,
    }


# ---------------------------------------------------------------------------
# Category selection logic
# ---------------------------------------------------------------------------

def select_categories(df: pd.DataFrame, n_per_cat: int = 10) -> pd.DataFrame:
    """
    Assign images to categories based on metric rankings.

    Each image is assigned to at most one category. Categories are filled
    in a specific order to handle overlaps (e.g., a defocused image might
    also be low-intensity — we assign it to defocused first).
    """
    df = df.copy()
    df["category"] = None
    assigned = set()

    def assign(mask, category):
        """Assign top n_per_cat unassigned images matching mask."""
        candidates = mask & ~df.index.isin(assigned)
        indices = df.loc[candidates].index[:n_per_cat]
        for idx in indices:
            df.loc[idx, "category"] = category
            assigned.add(idx)

    def assign_top(sort_by, ascending, category, filter_fn=None):
        """Pick top n_per_cat unassigned images when sorted by given columns.

        filter_fn: optional callable(df) -> boolean Series to pre-filter candidates.
        """
        subset = df if filter_fn is None else df[filter_fn(df)]
        sorted_df = subset.sort_values(sort_by, ascending=ascending)
        count = 0
        for idx in sorted_df.index:
            if idx in assigned:
                continue
            df.loc[idx, "category"] = category
            assigned.add(idx)
            count += 1
            if count >= n_per_cat:
                break

    # --- Priority order ---

    # 1. Defocused — lowest variance of Laplacian
    assign_top("var_laplacian", True, "05_defocused")

    # 2. Low intensity — lowest mean intensity
    assign_top("mean_intensity", True, "07_low_intensity")

    # 3. Debris — most bright small outlier objects relative to foreground
    assign_top(["n_debris_small", "peak_ratio"], [False, False], "09_debris")

    # 4. Low confluency — lowest area fraction
    assign_top("area_fraction", True, "01_low_confluency")

    # 5. High confluency — highest area fraction, but NOT dominated by
    #    merged blobs (filter out images with extreme clustering)
    assign_top("area_fraction", False, "02_high_confluency",
               filter_fn=lambda d: d["cc_size_ratio"] < d["cc_size_ratio"].quantile(0.8))

    # 6. Clustered/touching — highest ratio of largest CC to median CC size
    #    This captures merging independent of overall confluency
    assign_top("cc_size_ratio", False, "03_clustered_touching")

    # 7. Flat-field inhomogeneity — highest background illumination gradient
    #    Uses heavily smoothed (sigma=150) background field, not raw quadrant means
    assign_top("flatfield_ratio", False, "06_flatfield_inhomogeneity")

    # 8. High intensity — brightest foreground per-nucleus, NOT overall image mean
    #    This isolates staining brightness from cell density
    assign_top("foreground_median_intensity", False, "08_high_intensity")

    # 9. Mitotic — placeholder (will be manually curated)
    assign_top(["high_eccent_fraction", "mean_eccentricity"], [False, False], "04_mitotic")

    return df


# ---------------------------------------------------------------------------
# Thumbnail generation
# ---------------------------------------------------------------------------

def make_contact_sheet(df_cat: pd.DataFrame, category: str, output_dir: Path):
    """Generate a contact sheet of thumbnails for one category."""
    files = df_cat["filepath"].tolist()
    n = len(files)
    if n == 0:
        return

    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for i, (_, row) in enumerate(df_cat.iterrows()):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        img = tifffile.imread(row["filepath"])
        vmin, vmax = np.percentile(img, [1, 99.5])
        ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
        # Show the most relevant metric for this category
        cat_metrics = {
            "01_low_confluency": f"area_frac={row['area_fraction']:.3f}",
            "02_high_confluency": f"area_frac={row['area_fraction']:.3f}  cc_ratio={row.get('cc_size_ratio', 0):.1f}",
            "03_clustered_touching": f"cc_ratio={row.get('cc_size_ratio', 0):.1f}  oversized={row.get('oversized_fraction', 0):.2f}",
            "04_mitotic": f"hi_eccent={row.get('high_eccent_fraction', 0):.2f}",
            "05_defocused": f"VoL={row['var_laplacian']:.0f}",
            "06_flatfield_inhomogeneity": f"ff_ratio={row.get('flatfield_ratio', 0):.2f}  ff_grad={row.get('flatfield_gradient', 0):.1f}",
            "07_low_intensity": f"mean={row['mean_intensity']:.0f}  fg_med={row.get('foreground_median_intensity', 0):.0f}",
            "08_high_intensity": f"fg_med={row.get('foreground_median_intensity', 0):.0f}  mean={row['mean_intensity']:.0f}",
            "09_debris": f"n_debris={row.get('n_debris_small', 0):.0f}  peak_ratio={row.get('peak_ratio', 0):.1f}",
        }
        detail = cat_metrics.get(category, f"mean={row['mean_intensity']:.0f}")
        ax.set_title(
            f"{Path(row['filepath']).name[:40]}...\n{detail}",
            fontsize=7,
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(n, rows * cols):
        r, c = divmod(i, cols)
        axes[r, c].axis("off")

    fig.suptitle(f"{category} (n={n})", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / f"{category}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _compute_metrics_for_file(fpath: Path) -> dict:
    """Worker function for parallel metric computation."""
    img = tifffile.imread(str(fpath))
    metrics = compute_metrics(img)
    metrics["filepath"] = str(fpath)
    metrics["filename"] = fpath.name
    metrics["condition"] = fpath.parent.name  # NT or RAB13
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Select images for nuclei benchmark")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to SAMPLES folder containing NT/ and RAB13/ subfolders",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to NUCLEI-BENCHMARK folder with 9 category subfolders",
    )
    parser.add_argument(
        "--n-per-cat",
        type=int,
        default=10,
        help="Number of images per category (default: 10)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumb_dir = output_dir / "thumbnails"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    # Create category subfolders
    categories = [
        "01_low_confluency", "02_high_confluency", "03_clustered_touching",
        "04_mitotic", "05_defocused", "06_flatfield_inhomogeneity",
        "07_low_intensity", "08_high_intensity", "09_debris",
    ]
    for cat in categories:
        (output_dir / cat).mkdir(parents=True, exist_ok=True)

    # Discover all Blue images
    blue_files = sorted(
        list(input_dir.rglob("*Blue*.tif"))
    )
    # Exclude any output* folders
    blue_files = [f for f in blue_files if not any(p.startswith("output") for p in f.parts)]

    print(f"Found {len(blue_files)} Blue images")

    # Compute metrics (or load from cache)
    metrics_csv = output_dir / "selection_metrics.csv"
    if metrics_csv.exists():
        print(f"Loading cached metrics from {metrics_csv}")
        df = pd.read_csv(metrics_csv)
        print(f"Loaded metrics for {len(df)} images")
    else:
        import os
        n_workers = os.cpu_count() #or 4, 12)
        print(f"Computing metrics using {n_workers} workers...")

        records = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_compute_metrics_for_file, fpath): fpath
                for fpath in blue_files
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Computing metrics"
            ):
                records.append(future.result())

        df = pd.DataFrame(records)
        df.to_csv(metrics_csv, index=False)
        print(f"Saved metrics for {len(df)} images to {metrics_csv}")

    # Select categories
    df = select_categories(df, n_per_cat=args.n_per_cat)

    # Summary of selected images
    selected = df[df["category"].notna()].copy()
    summary_csv = output_dir / "selection_summary.csv"
    selected.to_csv(summary_csv, index=False)
    print(f"\nSelected {len(selected)} images across {selected['category'].nunique()} categories:")
    print(selected["category"].value_counts().sort_index().to_string())

    # Copy images to category folders and generate contact sheets
    for cat in sorted(selected["category"].unique()):
        cat_dir = output_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        df_cat = selected[selected["category"] == cat]
        for _, row in df_cat.iterrows():
            src = Path(row["filepath"])
            dst = cat_dir / src.name
            if not dst.exists():
                shutil.copy2(str(src), str(dst))

        make_contact_sheet(df_cat, cat, thumb_dir)
        print(f"  {cat}: {len(df_cat)} images copied, contact sheet saved")

    print(f"\nDone. Review thumbnails in: {thumb_dir}")
    print("Visually verify selections before running the benchmark.")


if __name__ == "__main__":
    main()
