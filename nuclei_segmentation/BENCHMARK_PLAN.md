# Nuclei Segmentation Benchmark — Implementation Plan

**Date**: 2026-02-19
**Location**: `AggreQuant_paper/benchmarks/nuclei_segmentation/`
**Conda environment**: `nuclei-bench`

---

## 1. Objective

Compare pretrained nuclei segmentation models on challenging HCS edge cases.
No manual ground truth annotations are available — evaluation is based on
inter-model agreement, per-category visual comparison, nuclei count concordance,
and inference speed.

This produces a **supplementary figure** for the AggreQuant methods paper
(Cell Reports Methods).

---

## 2. Environment Setup

### Conda Environment

```bash
conda create -n nuclei-bench python=3.10 -y
conda activate nuclei-bench
```

**Why Python 3.10**: DeepCell 0.12.10 requires `python >=3.7, <3.11` and pins
`tensorflow~=2.8.0`. TensorFlow 2.8.x supports Python 3.7–3.10. Cellpose and
StarDist both support 3.10. Python 3.10 is the highest common denominator.

### Install Order (order matters due to TF pinning)

```bash
# 1. TensorFlow (pinned by DeepCell)
pip install tensorflow==2.8.4

# 2. DeepCell (brings in tensorflow-addons, spektral, etc.)
pip install DeepCell==0.12.10

# 3. StarDist + csbdeep
pip install stardist==0.9.2

# 4. Cellpose (brings in PyTorch)
pip install cellpose>=3.0,<4.0

# 5. Utilities
pip install tifffile pandas matplotlib seaborn scikit-image tqdm
```

### Known Constraints

| Package | TF Requirement | PyTorch | Python |
|---------|---------------|---------|--------|
| DeepCell 0.12.10 | ~=2.8.0 (hard pin) | — | >=3.7, <3.11 |
| StarDist 0.9.2 | >=2.6.0 (flexible) | — | >=3.6 |
| Cellpose >=3.0,<4.0 | — | >=1.6 | ~3.9–3.11 |

- `tensorflow-addons~=0.16.1` (DeepCell dep) is deprecated but installs fine.
- TensorFlow and PyTorch coexist without conflict.
- NumPy must be >=1.22, <1.24 to satisfy all packages.

---

## 3. Models and Configurations (8 total)

### 3.1 StarDist 2D

| ID | Model | Notes |
|----|-------|-------|
| `stardist_2d_fluo` | `2D_versatile_fluo` | Default prob/nms thresholds |

```python
from stardist.models import StarDist2D
from csbdeep.utils import normalize

model = StarDist2D.from_pretrained('2D_versatile_fluo')
img_norm = normalize(img, 1, 99.8, axis=(0, 1))
labels, details = model.predict_instances(img_norm)
```

- Input: `(H, W)` float, percentile-normalized to [0, 1].
- Output: `labels` (H, W) int labeled mask; `details` dict with centroids, probabilities, polygon coords.

### 3.2 Cellpose

| ID | Model type | channels | Description |
|----|-----------|----------|-------------|
| `cellpose_nuclei` | `nuclei` | `[0, 0]` | Purpose-built nuclei model, single-channel |
| `cellpose_cyto2_no_nuc` | `cyto2` | `[0, 0]` | Cyto2 on DAPI only, no nuclear hint |
| `cellpose_cyto2_with_nuc` | `cyto2` | `[1, 2]` | Cyto2 with DAPI provided as both channels (see note) |
| `cellpose_cyto3_no_nuc` | `cyto3` | `[0, 0]` | Cyto3 on DAPI only, no nuclear hint |
| `cellpose_cyto3_with_nuc` | `cyto3` | `[1, 2]` | Cyto3 with DAPI provided as both channels (see note) |

```python
from cellpose import models

model = models.Cellpose(gpu=True, model_type=model_type)
masks, flows, styles, diams = model.eval(
    img,               # (H, W) for channels=[0,0]
    diameter=None,     # auto-estimate
    channels=channels,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
)
```

**Note on "with nuclear channel"**: Since benchmark images are single-channel
DAPI, providing a nuclear channel means constructing a 2-channel image where
both channels contain the same DAPI image:

```python
import numpy as np
img_2ch = np.stack([img, img], axis=-1)  # (H, W, 2)
# channels=[1, 2] means: segment channel 1 (red), guided by channel 2 (green)
masks, flows, styles, diams = model.eval(img_2ch, channels=[1, 2], diameter=None)
```

This tests whether providing the nuclear hint helps cyto models on nuclei-only data.

- Input: `(H, W)` or `(H, W, 2)` depending on configuration.
- Cellpose applies its own internal percentile normalization.
- Output: `masks` (H, W) int labeled mask; `flows` list; `styles` vector; `diams` float.

### 3.3 DeepCell

| ID | Application | Input channels | Notes |
|----|-------------|---------------|-------|
| `deepcell_nuclear` | `NuclearSegmentation` | 1 | Purpose-built for fluorescent nuclei |
| `deepcell_mesmer` | `Mesmer` (compartment='nuclear') | 2 (nuclear + zeros) | Trained on TissueNet (1.2M+ cells) |

```python
import numpy as np
from deepcell.applications import NuclearSegmentation, Mesmer

# NuclearSegmentation — single channel
app = NuclearSegmentation()
img_4d = img[np.newaxis, :, :, np.newaxis]  # (1, H, W, 1)
labels = app.predict(img_4d, image_mpp=0.325)
labels = labels[0, :, :, 0]  # back to (H, W)

# Mesmer — two channels (nuclear + zeros for membrane)
app = Mesmer()
membrane = np.zeros_like(img)
img_2ch = np.stack([img, membrane], axis=-1)       # (H, W, 2)
img_4d = img_2ch[np.newaxis, ...]                   # (1, H, W, 2)
labels = app.predict(img_4d, image_mpp=0.325, compartment='nuclear')
labels = labels[0, :, :, 0]
```

- `image_mpp`: microns per pixel. Model rescales internally to match training resolution.
  NuclearSegmentation trained at 0.65 mpp; Mesmer at ~0.5 mpp.
  **Our images**: 0.325 µm/pixel (20x objective, GE InCell Analyzer). DeepCell will
  upscale internally (~2x for NuclearSegmentation, ~1.5x for Mesmer) to match training resolution.
- DeepCell applies its own internal preprocessing (histogram normalization).
- Output: labeled mask `(H, W)` int.

---

## 4. Input Data

### Directory Structure

```
benchmarks/nuclei_segmentation/
└── data/
    ├── 01_low_confluency/        # ~10 TIFF images
    ├── 02_high_confluency/
    ├── 03_clustered_touching/
    ├── 04_mitotic/
    ├── 05_defocused/
    ├── 06_flatfield_inhomogeneity/
    ├── 07_low_intensity/
    ├── 08_high_intensity/
    └── 09_debris/
```

- Each folder contains ~10 single-channel DAPI TIFF images (grayscale, uint16).
- ~90 images total.
- All images are from NT and RAB13 control wells, sampled across multiple plates.

### Image Loading

```python
import tifffile

img = tifffile.imread(path)  # returns numpy array, typically uint16
```

No AggreQuant imports — the benchmark is fully standalone.

---

## 5. Code Architecture

### Files

```
benchmarks/nuclei_segmentation/
├── BENCHMARK_PLAN.md          # This file
├── run_benchmark.py           # Main script — runs all models on all images
├── plot_results.py            # Generates the 3-panel supplementary figure
├── data/                      # Input images (9 category folders)
├── results/                   # Output
│   ├── masks/                 # Saved label masks per model per image
│   │   ├── stardist_2d_fluo/
│   │   ├── cellpose_nuclei/
│   │   ├── ...
│   │   └── deepcell_mesmer/
│   ├── counts.csv             # Nuclei counts: image, category, model, count
│   └── timing.csv             # Inference times: model, image, time_seconds
└── figures/                   # Output figure panels
    ├── panel_A_gallery.png    # or .pdf
    ├── panel_B_counts.png
    └── panel_C_speed.png
```

### `run_benchmark.py` — Design

```
1. Parse arguments (data_dir, output_dir, gpu flag, image_mpp)
2. Discover images: walk data/ subfolders, sort by category
3. Load all models once (expensive — do this upfront):
   - StarDist2D.from_pretrained('2D_versatile_fluo')
   - Cellpose(model_type='nuclei'), Cellpose(model_type='cyto2'), Cellpose(model_type='cyto3')
   - NuclearSegmentation()
   - Mesmer()
4. For each image:
   a. Load with tifffile.imread()
   b. For each model configuration:
      - Preprocess as required by model (normalize, reshape, etc.)
      - Time the inference (wall-clock, excluding preprocessing)
      - Extract labeled mask
      - Count nuclei (number of unique labels, excluding 0)
      - Save mask as uint16 TIFF to results/masks/{model_id}/{image_name}.tif
      - Append row to counts list and timing list
5. Save counts.csv and timing.csv
6. Print summary table to stdout
```

**Design considerations**:
- Load each model **once** and reuse across all images.
- Time **only inference** (not model loading, not preprocessing, not I/O).
  Use `time.perf_counter()` for wall-clock timing.
- Save masks so `plot_results.py` can generate overlays without re-running models.
- Use `tqdm` for progress bars.
- Handle GPU memory: if running on GPU, call appropriate cleanup between models
  if needed (TF `clear_session`, torch `empty_cache`).

### `plot_results.py` — Design

**Panel A: Category Gallery**

- Layout: 9 rows (categories) × 4+ columns (raw DAPI | model overlays).
- For each category, pick the single most illustrative FOV.
  Selection strategy: choose the FOV with the highest count disagreement
  among models (max count - min count), as these are the most informative.
- Overlay: colored contours of segmentation masks on the DAPI image.
  Use distinct colors per model (e.g., StarDist=cyan, Cellpose=yellow, DeepCell=magenta).
- Annotate nuclei count per model in each panel.
- Add arrows/circles highlighting failure modes (merged nuclei, missed objects,
  false positives). This may require some manual curation after initial generation.

**Panel B: Per-Category Nuclei Count Comparison**

- Grouped bar chart or dot plot with error bars.
- X-axis: 9 difficulty categories.
- Y-axis: nuclei count (or count relative to consensus mean).
- One color per model configuration.
- Error bars: SD or IQR across the ~10 images per category.
- Optionally: a secondary plot of pairwise count correlation between models.

**Panel C: Inference Speed**

- Bar chart: mean time per FOV for each model configuration.
- Error bars: SD across all images.
- Annotate with estimated time per 384-well plate (mean_time × 384 × 9 FOVs).
- Separate bars for CPU vs GPU if both are benchmarked.

---

## 6. Output Specification

### `counts.csv`

| Column | Type | Description |
|--------|------|-------------|
| image_name | str | Filename (without path) |
| category | str | Folder name (e.g., `01_low_confluency`) |
| model_id | str | Model configuration ID (e.g., `stardist_2d_fluo`) |
| nuclei_count | int | Number of detected nuclei (unique labels excluding 0) |
| mean_area_px | float | Mean area of detected nuclei in pixels |
| median_area_px | float | Median area of detected nuclei in pixels |

### `timing.csv`

| Column | Type | Description |
|--------|------|-------------|
| model_id | str | Model configuration ID |
| image_name | str | Filename |
| inference_time_s | float | Wall-clock inference time in seconds (excluding preprocessing) |
| device | str | `cpu` or `gpu` |

---

## 7. Important Notes

- **Standalone code**: No imports from the `aggrequant` package. Each model
  is called through its own public API with default/recommended preprocessing.
  This ensures the benchmark evaluates the pretrained models as their authors
  intended, not as AggreQuant wraps them.

- **Fairness**: Each model uses its own recommended normalization:
  - StarDist: `csbdeep.utils.normalize(img, 1, 99.8)`
  - Cellpose: internal percentile normalization (automatic)
  - DeepCell: internal histogram normalization (automatic)

- **`image_mpp` for DeepCell**: Both NuclearSegmentation and Mesmer accept
  a microns-per-pixel parameter and rescale internally. The correct value
  depends on the microscope objective and camera. **This must be determined
  from the microscope metadata before running.**

- **GPU considerations**: StarDist uses TensorFlow GPU, Cellpose uses PyTorch
  GPU, DeepCell uses TensorFlow GPU. If running sequentially on the same GPU,
  ensure proper memory cleanup between TF and PyTorch models.

- **Reproducibility**: Pin random seeds where applicable. Save the exact
  package versions (`pip freeze > requirements_frozen.txt`) after environment
  setup.

---

## 8. Execution Checklist

- [ ] Create conda environment `nuclei-bench` with Python 3.10
- [ ] Install dependencies in correct order
- [ ] Verify all three packages import without error
- [ ] Determine `image_mpp` from microscope metadata
- [ ] Place curated images in `data/` subfolders (9 categories × ~10 images)
- [ ] Run `run_benchmark.py`
- [ ] Inspect `counts.csv` and `timing.csv` for sanity
- [ ] Run `plot_results.py` to generate figure panels
- [ ] Manually curate Panel A (select best representative FOVs, add annotations)
- [ ] Assemble final supplementary figure
