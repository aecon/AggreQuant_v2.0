# Preprocessing Cellpose Segmentation Benchmark — Implementation Plan

**Date**: 2026-03-10
**Location**: `benchmarks/preprocessing_cellpose_segmentation/`
**Conda environment**: `cell-bench` (has both Cellpose and StarDist)

---

## 1. Objective

Quantify how different preprocessing of the nuclear channel affects Cellpose cell
segmentation. The AggreQuant pipeline feeds Cellpose a **binary nuclei mask** derived
from StarDist segmentation — not the raw DAPI image. This benchmark tests whether that
design choice actually improves cell segmentation compared to using raw DAPI or no
nuclear channel at all.

Produces a **supplementary figure** for the AggreQuant methods paper
(Cell Reports Methods), complementing the nuclei and cell segmentation benchmarks.

---

## 2. Relationship to Other Benchmarks

Same **90 FOVs** from the nuclei and cell segmentation benchmarks, from the same
SpeedDrive directory:

```
/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/
  2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19/
```

- Nuclei benchmark: validated StarDist `2D_versatile_fluo` as the best nuclei model.
- Cell benchmark: validated Cellpose `cyto3` as the best cell model, showed that
  adding a DAPI channel gives modest improvement for Cellpose (unlike Mesmer/InstanSeg
  which depend on it).
- **This benchmark**: isolates the question of *how* the nuclear information is
  provided to Cellpose — raw DAPI vs pre-segmented binary mask vs nothing.

`data/images` is a symlink to the same SpeedDrive directory.

---

## 3. Variants (3 total)

All variants use the same Cellpose `cyto3` model. The only variable is the input
preprocessing of the nuclear channel.

| Variant ID | Description | Cellpose input | channels |
|---|---|---|---|
| `cellpose_cell_only` | Single-channel: FarRed cytoplasm only | `(H, W)` | `[0, 0]` |
| `cellpose_raw_nuclei` | Two-channel: FarRed + raw DAPI image | `(2, H, W)` — `[cell, DAPI]` | `[1, 2]` |
| `cellpose_nuclei_seeds` | Two-channel: FarRed + binary nuclei mask from StarDist | `(2, H, W)` — `[cell, mask]` | `[1, 2]` |

### How nuclei seeds are generated

Reproduces exactly the pipeline from `aggrequant/segmentation/stardist.py` and
`aggrequant/segmentation/cellpose.py`:

**StarDist preprocessing** (`StarDistSegmenter.segment()`):

1. Gaussian denoise (sigma=2)
2. Background normalization: `denoised / (Gaussian(sigma=50) + eps)`
3. `csbdeep.utils.normalize` (percentile normalization to [0, 1])
4. `StarDist2D.predict_instances`
5. Size exclusion (300–15,000 px)
6. Border separation (Sobel + dilation)
7. Consecutive relabeling

**Binary mask creation** (`CellposeSegmenter._segment_with_nuclei()`):

```python
nuclei_mask = (nuclei_labels > 0).astype(np.float32)
```

**Cellpose input assembly**:

```python
input_image = np.zeros((2, h, w))
input_image[0, :, :] = cell_image      # FarRed cytoplasm
input_image[1, :, :] = nuclei_mask     # binary mask (0.0 or 1.0)
masks, _, _, _ = model.eval(input_image, channels=[1, 2], diameter=None)
```

The benchmark reproduces this logic inline (no AggreQuant imports) to ensure it tests
exactly the same pipeline behavior.

### Channel conventions

**Cellpose `channels=[0, 0]`**: single-channel grayscale input, no nuclear hint.

**Cellpose `channels=[1, 2]`**: channel 1 is cytoplasm (to segment), channel 2 is
nuclear hint. Input shape `(2, H, W)` where `[0]` = cytoplasm, `[1]` = nuclear.

---

## 4. Input Data

### Directory structure

```
preprocessing_cellpose_segmentation/data/images -> SpeedDrive/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19/
  ├── 01_low_confluency/
  │   ├── *wv 390 - Blue*.tif    (DAPI — nuclear channel)
  │   └── *wv 631 - FarRed*.tif  (primary — cell channel)
  ├── 02_high_confluency/
  ├── 03_clustered_touching/
  ├── 04_mitotic/
  ├── 05_defocused/
  ├── 06_flatfield_inhomogeneity/
  ├── 07_low_intensity/
  ├── 08_high_intensity/
  └── 09_debris_artifacts/
```

- 90 FarRed images total; every image has a matching DAPI file.
- Primary: `*wv 631 - FarRed*.tif` (uint16, single-channel).
- Nuclear: derived by replacing `wv 631 - FarRed` → `wv 390 - Blue` in filename.

---

## 5. Code Architecture

```
preprocessing_cellpose_segmentation/
├── BENCHMARK_PLAN.md        # This file
├── run_benchmark.py         # Main script — runs 3 variants on 90 images
├── plot_results.py          # Figure panels A–E
├── data/
│   └── images -> <SpeedDrive symlink>
└── results/
    ├── masks/               # Per-variant label masks
    │   ├── cellpose_cell_only/
    │   ├── cellpose_raw_nuclei/
    │   └── cellpose_nuclei_seeds/
    ├── counts.csv           # cell_count, areas, shape metrics
    └── timing.csv           # Inference times
```

---

## 6. Metrics

### Per-image, per-variant

| Metric | Description |
|---|---|
| `cell_count` | Number of detected cells |
| `mean_area_px` | Mean cell area in pixels |
| `median_area_px` | Median cell area in pixels |
| `mean_solidity` | Mean solidity (area / convex_hull_area) — shape regularity |
| `median_solidity` | Median solidity |
| `mean_eccentricity` | Mean eccentricity — elongation measure |
| `median_eccentricity` | Median eccentricity |

**Solidity** captures how "clean" the segmentation boundaries are. A perfect circle or
convex polygon has solidity=1.0; irregular/fragmented shapes have lower values. If
nuclei seeds produce cleaner cell boundaries, this will show up as higher solidity.

**Eccentricity** ranges from 0 (circle) to 1 (line). Captures whether different
preprocessing leads to more elongated cell detections.

### Timing

Only the Cellpose `model.eval()` call is timed. StarDist preprocessing for the seeds
variant is excluded from timing since it is a pipeline cost, not a Cellpose cost.
StarDist time is recorded separately as `stardist_preprocess_time_s`.

---

## 7. Output Specification

### `counts.csv`

| Column | Type | Description |
|---|---|---|
| `image_name` | str | FarRed filename |
| `category` | str | Folder name (e.g. `01_low_confluency`) |
| `variant_id` | str | Variant ID (e.g. `cellpose_cell_only`) |
| `cell_count` | int | Number of detected cells |
| `mean_area_px` | float | Mean cell area in pixels |
| `median_area_px` | float | Median cell area in pixels |
| `mean_solidity` | float | Mean solidity across cells |
| `median_solidity` | float | Median solidity |
| `mean_eccentricity` | float | Mean eccentricity |
| `median_eccentricity` | float | Median eccentricity |

### `timing.csv`

| Column | Type | Description |
|---|---|---|
| `variant_id` | str | Variant ID |
| `image_name` | str | FarRed filename |
| `inference_time_s` | float | Cellpose eval wall-clock time (seconds) |
| `stardist_preprocess_time_s` | float | StarDist preprocessing time (seeds variant only, NaN otherwise) |
| `device` | str | `gpu` or `cpu` |

---

## 8. `run_benchmark.py` — Design

```
1. Parse arguments (data_dir, output_dir, gpu flag)
2. Discover images: walk category subfolders for FarRed + matching DAPI
3. Load models once:
   - Cellpose cyto3 (shared across all 3 variants)
   - StarDist 2D_versatile_fluo (for the seeds variant only)
4. For each image:
   a. Load FarRed (cell) and DAPI (nuclei) with tifffile
   b. For each variant:
      - Preprocess input according to variant
      - Time Cellpose inference (wall-clock)
      - Compute cell count, areas, solidity, eccentricity via regionprops
      - Save label mask as uint16 TIFF
      - Append row to counts and timing lists
5. Save counts.csv and timing.csv
6. Print summary table
```

### Checkpoint/resume

Same pattern as the cell benchmark: if mask files already exist on disk, load them
instead of re-running inference. Timing data is recovered from the previous
`timing.csv` if available.

### StarDist preprocessing (inline)

The StarDist pipeline is reproduced inline (no AggreQuant imports) with the exact same
parameters:

```python
import skimage.filters
import skimage.morphology
from csbdeep.utils import normalize
from stardist.models import StarDist2D

def stardist_to_binary_mask(img_nuc, stardist_model):
    """Reproduce AggreQuant's StarDist pipeline and return binary mask."""
    # 1. Gaussian denoise
    img = img_nuc.astype(np.float32)
    denoised = skimage.filters.gaussian(img, sigma=2, mode='reflect', preserve_range=True)

    # 2. Background normalization
    background = skimage.filters.gaussian(denoised, sigma=50, mode='reflect', preserve_range=True)
    normalized = denoised / (background + 1e-8)

    # 3. StarDist inference
    labels, _ = stardist_model.predict_instances(
        normalize(normalized), predict_kwargs=dict(verbose=False)
    )

    # 4. Size exclusion (300–15000 px)
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label_id, area in zip(unique_labels[1:], counts[1:]):
        if area < 300 or area > 15000:
            labels[labels == label_id] = 0

    # 5. Border separation (Sobel + dilation)
    edges = skimage.filters.sobel(labels)
    fat_edges = skimage.morphology.dilation(edges > 0)
    labels[fat_edges] = 0

    # 6. Binary mask
    return (labels > 0).astype(np.float32)
```

---

## 9. `plot_results.py` — Design

### Panel A: Cell count comparison across categories

Line plot with error bars (mean ± SD). X-axis: 9 categories (sorted by median count).
3 lines, one per variant. Same style as cell benchmark Panel A.

### Panel B: Pairwise count difference

Per-image count difference between variants:
- `seeds − cell_only` (how much does adding seeds help vs no nuclear info?)
- `seeds − raw_nuclei` (how much does pre-segmenting help vs raw DAPI?)
- `raw_nuclei − cell_only` (how much does raw DAPI help vs nothing?)

Box plots grouped by category. Positive values = first variant detects more cells.

### Panel C: Cell area distribution

Box plots of mean cell area per variant per category. Reveals whether different
preprocessing leads to systematically larger or smaller cell detections.

### Panel D: Solidity comparison

Box plots of mean solidity per variant per category. Tests whether nuclei seeds
produce cleaner, more regular cell boundaries.

### Panel E: Per-image scatter (seeds vs raw DAPI)

Scatter plot: x-axis = `cellpose_raw_nuclei` cell count, y-axis =
`cellpose_nuclei_seeds` cell count. One dot per image, colored by category.
Diagonal line = perfect agreement. Reveals whether the two approaches agree or
diverge systematically.

---

## 10. Expected Findings

1. **Does the nuclear channel help at all?** The cell benchmark already showed modest
   improvement for Cellpose with +nuc. This benchmark quantifies the effect more
   precisely with a single model (cyto3).

2. **Does pre-segmenting nuclei help more than raw DAPI?** This is the key question.
   The binary mask from StarDist provides clean, denoised nuclear locations — no
   intensity gradients, no background noise, no out-of-focus blur. If this helps,
   it validates the pipeline's design choice.

3. **Do seeds produce cleaner cell shapes?** The solidity metric will reveal whether
   the binary mask helps Cellpose produce more regular cell boundaries, especially
   in challenging categories (defocused, low intensity, debris).

4. **Are there category-specific effects?** Seeds might help most in categories where
   the raw DAPI is degraded (defocused, low intensity, flat-field inhomogeneity),
   since StarDist's preprocessing (background normalization) already handles these
   artifacts.

---

## 11. Key Differences from Cell Segmentation Benchmark

| Aspect | Cell Benchmark | This Benchmark |
|---|---|---|
| Question | Which model is best? | Which preprocessing is best? |
| Models | 5 models × 2 channel configs = 10 | 1 model (cyto3) × 3 preprocessing variants |
| Nuclear input | Raw DAPI or nothing | Raw DAPI, binary StarDist mask, or nothing |
| Extra metrics | — | Solidity, eccentricity (shape quality) |
| Extra dependency | — | StarDist (for seeds variant) |
| Standalone | Yes (no AggreQuant imports) | Yes (reproduces StarDist pipeline inline) |

---

## 12. Execution Checklist

- [ ] Create symlink `data/images` → SpeedDrive
- [ ] Verify `cell-bench` env has both Cellpose and StarDist
- [ ] Run `run_benchmark.py --data-dir data/images`
- [ ] Inspect `results/counts.csv` and `results/timing.csv`
- [ ] Run `plot_results.py` to generate panels A–E
- [ ] Interpret results and update `model_selection_rationale.md` if needed
