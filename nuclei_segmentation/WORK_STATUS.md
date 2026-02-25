# Nuclei Segmentation Benchmark — Current Work Status

**Last updated**: 2026-02-25 (plot_masks.py implemented; viewer hover + raw-only features added; viewer consensus UX improvements)
**Purpose**: Handoff document so any Claude instance can continue this work.

---

## What This Is

We are building a **supplementary figure** for the AggreQuant methods paper
(target journal: Cell Reports Methods). The figure compares pretrained nuclei
segmentation models on challenging HCS edge cases. It is NOT part of the
AggreQuant package — it is a standalone benchmark that lives in
`AggreQuant_paper/benchmarks/nuclei_segmentation/`.

The main paper manuscript is at `AggreQuant_paper/manuscript_CRM/`.
The AggreQuant package codebase is at `../Aggrequant/` (sibling directory).
The overall paper figure plan is in `AggreQuant_paper/PROGRESS.md`.

---

## Key Decisions Made

### 1. No ground truth annotations
We do not have manual nuclei annotations. Evaluation uses:
- Inter-model agreement (count concordance, pairwise detection overlap)
- Per-category visual comparison (qualitative gallery)
- Inference speed

### 2. Stress-test, not overall comparison
StarDist, Cellpose, and DeepCell are all well-calibrated and community-validated.
We do NOT expect large systematic differences on standard images. The benchmark
is designed to reveal how each model handles **specific challenging cases**, not
to declare an overall "winner." The framing is: "here is where the models
differ, and here is why we chose X for our pipeline."

### 3. Standalone code
The benchmark does NOT import from the AggreQuant package. Each model is called
through its own public API with default/recommended preprocessing. This ensures
fair comparison (each model used as its authors intended).

### 4. Conda environment — `nuclei-bench`
- **Name**: `nuclei-bench`
- **Python**: 3.10 (forced by DeepCell's TF 2.8.x pin + Python <3.11 requirement)
- **Status**: Created and verified. All imports work, both TF and PyTorch GPU confirmed (RTX 3090).

**Installed packages** (key versions):

| Package | Version | GPU Backend |
|---------|---------|-------------|
| tensorflow | 2.8.4 | CUDA (via cuDNN 8) |
| deepcell | 0.12.10 | (via TF) |
| stardist | 0.9.2 | (via TF) |
| cellpose | 3.1.1.3 | PyTorch CUDA |
| torch | 2.10.0+cu128 | CUDA 12.8 |
| numpy | 1.26.4 | — |

Full frozen requirements: `benchmarks/nuclei_segmentation/requirements_frozen.txt`

**Custom conda activate.d scripts** (auto-set on `conda activate nuclei-bench`):

Located at: `/home/athena/miniconda3/envs/nuclei-bench/etc/conda/activate.d/cudnn_env.sh`
This script does two things:
1. **cuDNN 8 fix**: Adds `nvidia/cudnn/lib` to `LD_LIBRARY_PATH` so TF 2.8 finds `libcudnn.so.8`
   (PyTorch installed cuDNN 9, but TF 2.8 needs cuDNN 8; both are in the same pip package)
2. **DeepCell access token**: Sets `DEEPCELL_ACCESS_TOKEN` env var (required by DeepCell 0.12.10
   to download model weights via `fetch_data()`). Token was created at https://users.deepcell.org

Deactivation script at: `/home/athena/miniconda3/envs/nuclei-bench/etc/conda/deactivate.d/cudnn_env.sh`
(unsets the above variables)

### 5. Image pixel size
- **0.325 µm/pixel** (20x objective, GE InCell Analyzer)
- Used for DeepCell's `image_mpp` parameter

### 6. DeepCell API changes in 0.12.10
- `NuclearSegmentation()` no longer auto-downloads models. Must use
  `NuclearSegmentation.from_version("1.1")` classmethod instead.
- `Mesmer(model=None)` still auto-downloads (different code path).
- Both require `DEEPCELL_ACCESS_TOKEN` env var to be set.

### 7. Two-channel (cell channel) support
Three models can use a secondary FarRed cell/membrane channel to aid nuclei segmentation:

| Model | Supports cell channel? | Input format with cell channel |
|-------|----------------------|-------------------------------|
| StarDist | No | Single DAPI only |
| DeepCell Nuclear | No | Single DAPI only |
| DeepCell Mesmer | **Yes** | `[1, H, W, 2]`: [DAPI, FarRed] (nuclear first, membrane second) |
| Cellpose nuclei | No | Single DAPI only |
| Cellpose cyto2 | **Yes** | `(H, W, 2)`: [FarRed, DAPI], `channels=[1,2]` (cyto=ch1, nuc=ch2) |
| Cellpose cyto3 | **Yes** | `(H, W, 2)`: [FarRed, DAPI], `channels=[1,2]` (cyto=ch1, nuc=ch2) |
| InstanSeg | **Yes** | `(2, H, W)` float32: [DAPI, FarRed], channel-invariant |

Note on existing `_with_nuc` configs: These duplicate DAPI as both channels to test whether
the cyto model's nuclear-hint pathway changes behavior even when both channels are identical.
The `_with_cell` configs test the **intended** use: real FarRed as cytoplasm + DAPI as nucleus.

---

## 13 Model Configurations

### Single-channel configs (DAPI only) — 9 configs

| ID | Package | Model | Input | Notes |
|----|---------|-------|-------|-------|
| `stardist_2d_fluo` | StarDist | `2D_versatile_fluo` | (H,W) normalized | `csbdeep.utils.normalize(img, 1, 99.8)` |
| `cellpose_nuclei` | Cellpose | `nuclei` | (H,W), channels=[0,0] | Purpose-built for nuclei |
| `cellpose_cyto2_no_nuc` | Cellpose | `cyto2` | (H,W), channels=[0,0] | Cyto model, no nuclear hint |
| `cellpose_cyto2_with_nuc` | Cellpose | `cyto2` | (H,W,2), channels=[1,2] | DAPI duplicated as both channels |
| `cellpose_cyto3_no_nuc` | Cellpose | `cyto3` | (H,W), channels=[0,0] | Super-generalist, no nuclear hint |
| `cellpose_cyto3_with_nuc` | Cellpose | `cyto3` | (H,W,2), channels=[1,2] | DAPI duplicated as both channels |
| `deepcell_nuclear` | DeepCell | `NuclearSegmentation.from_version("1.1")` | (1,H,W,1), mpp=0.325 | Single-channel, trained on cell lines |
| `deepcell_mesmer` | DeepCell | `Mesmer` (compartment='nuclear') | (1,H,W,2), mpp=0.325 | Zeros for membrane channel |
| `instanseg_fluorescence` | InstanSeg | `fluorescence_nuclei_and_cells` | (H,W) float32, mpp=0.325 | Embedding-based, channel-invariant, target='nuclei' |

**Note**: `instanseg-torch==0.1.1` — PyTorch-based, embedding approach. Uses `eval_small_image()`
with `pixel_size=0.325`, `target="nuclei"`, `return_image_tensor=False`. Native resolution 0.5 µm/px
(rescales internally). Internal percentile normalization.

### Two-channel configs (DAPI + FarRed cell channel) — 4 configs

| ID | Package | Model | Input | Notes |
|----|---------|-------|-------|-------|
| `cellpose_cyto2_with_cell` | Cellpose | `cyto2` | (H,W,2), channels=[1,2] | FarRed=cytoplasm, DAPI=nucleus |
| `cellpose_cyto3_with_cell` | Cellpose | `cyto3` | (H,W,2), channels=[1,2] | FarRed=cytoplasm, DAPI=nucleus |
| `deepcell_mesmer_with_cell` | DeepCell | `Mesmer` (compartment='nuclear') | (1,H,W,2), mpp=0.325 | DAPI=nuclear ch, FarRed=membrane ch |
| `instanseg_fluorescence_with_cell` | InstanSeg | `fluorescence_nuclei_and_cells` | (2,H,W) float32, mpp=0.325 | [DAPI, FarRed], target='nuclei' |

Full API code snippets for StarDist/Cellpose/DeepCell in `BENCHMARK_PLAN.md` sections 3.1–3.3.

### InstanSeg API details

```python
from instanseg import InstanSeg
model = InstanSeg("fluorescence_nuclei_and_cells", verbosity=0)

# Single channel
result = model.eval_small_image(img_float32, pixel_size=0.325, target="nuclei", return_image_tensor=False)
labels = result.squeeze().cpu().numpy()  # (1,1,H,W) → (H,W)

# Two channels
img_2ch = np.stack([dapi, farred], axis=0).astype(np.float32)  # (2,H,W)
result = model.eval_small_image(img_2ch, pixel_size=0.325, target="nuclei", return_image_tensor=False)
```

- Model native pixel size: 0.5 µm/px (rescales internally from our 0.325 µm/px)
- `target="nuclei"` returns only nuclei labels (model also supports cell segmentation)
- Internal percentile normalization (no manual preprocessing needed)
- pip package: `instanseg-torch==0.1.1`

---

## 9 Difficulty Categories (FINALIZED — manually curated by user)

**Curated dataset location**:
`/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19`

| Folder | Category | DAPI images | FarRed images |
|--------|----------|-------------|---------------|
| `01_low_confluency` | Low confluency | 10 | 10 |
| `02_high_confluency` | High confluency | 10 | 10 |
| `03_clustered_touching` | Clustered/touching nuclei | 10 | 10 |
| `04_mitotic` | Mitotic figures | 10 | 10 |
| `05_defocused` | Defocused / blurry | 10 | 10 |
| `06_flatfield_inhomogeneity` | Flat-field illumination gradient | 10 | 10 |
| `07_low_intensity` | Low overall DAPI intensity | 10 | 10 |
| `08_high_intensity` | High overall DAPI intensity | 10 | 10 |
| `09_debris_artifacts` | Debris and artifacts | **20** | **20** |

**Total: 100 DAPI images + 100 FarRed images** (2040×2040, uint16, from GE InCell Analyzer).

Each category folder contains paired files:
- DAPI: `*wv 390 - Blue*.tif`
- FarRed: `*wv 631 - FarRed*.tif` (same name with channel swapped)

The script discovers DAPI images by matching `*Blue*.tif*` and derives the FarRed path
by replacing `wv 390 - Blue` → `wv 631 - FarRed` in the filename.

Images sourced from NT and RAB13 control wells across ~55 384-well plates.
Initial automated selection used metric-based ranking (see `select_images.py`),
followed by manual curation by the user (especially mitotic category, which
cannot be reliably auto-detected from image-level statistics). The `09_debris_artifacts`
folder was intentionally kept larger to provide more test cases for model robustness.

### Image selection history
- v1: Automated selection had correlated categories (high confluency ≈ clustered ≈ high intensity).
- v2: Fixed with decorrelated metrics (foreground_median_intensity, cc_size_ratio, smoothed background ratio).
- Final: User manually curated all categories, replacing auto-selected images as needed.
  The `select_images.py` script and `selection_metrics.csv` were tools in this process,
  not the final arbiter.

### FarRed cell channel image sourcing
FarRed images were matched to curated DAPI images by extracting core identifiers
(HA plate, well row, column, field number) using regex. The matching had to handle
3 naming patterns because the user renamed/simplified filenames during curation:

- **94/100** matched from `SAMPLES_10_perCondition_perPLATE/{NT,RAB13}/`
  (source filenames had extra `Plate1_`, `Plate3_`, `Plate5_` or `HA_XX_RepN__` prefixes)
- **6/100** matched from `weird_stuff/{HA_11_rep_1,HA1_rep1,HA2_rep2,HA7_rep1}/`
  (non-standard well positions not in the main sample set)

All FarRed images were copied with standardized names matching the DAPI naming convention.

---

## Data Locations

| What | Path |
|------|------|
| **Paper repo** | `/home/athena/1_CODES/AggreQuant_paper/` |
| **AggreQuant package** | `/home/athena/1_CODES/Aggrequant/` |
| **Raw HCS data** | `/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/` |
| **Curated benchmark images** | `/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19` |
| **FarRed source (main)** | `/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/SAMPLES_10_perCondition_perPLATE` |
| **FarRed source (6 extras)** | `/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/weird_stuff` |
| **Benchmark code** | `/home/athena/1_CODES/AggreQuant_paper/benchmarks/nuclei_segmentation/` |
| **Benchmark results** | `benchmarks/nuclei_segmentation/results/` |
| **Conda env** | `/home/athena/miniconda3/envs/nuclei-bench/` |
| **Conda activate.d scripts** | `/home/athena/miniconda3/envs/nuclei-bench/etc/conda/activate.d/cudnn_env.sh` |

---

## Current State of Work

### Completed
- [x] Designed supplementary figure plan (`docs/supplementary_nuclei_segmentation_benchmark.md`)
- [x] Created benchmark implementation plan (`benchmarks/nuclei_segmentation/BENCHMARK_PLAN.md`)
- [x] Wrote image selection script (`benchmarks/nuclei_segmentation/select_images.py`)
- [x] Computed image metrics on all 1,120 candidate images
- [x] Iteratively fixed category selection (v1 → v2 decorrelated metrics)
- [x] **User manually curated all 9 category folders (100 images total)**
- [x] Draft paper text for image selection (`benchmarks/nuclei_segmentation/paper_text_image_selection.md`)
- [x] **Created `nuclei-bench` conda environment** (Python 3.10, all packages verified)
- [x] **Wrote `run_benchmark.py`** with checkpoint/resume support (13 model configs)
- [x] **Sourced FarRed cell channel images** for all 100 DAPI images (94 from main source, 6 from `weird_stuff`)
- [x] **Added InstanSeg** (`instanseg-torch==0.1.1`) — 2 configs (single-channel + with cell)
- [x] **All 13 model configs ran successfully** on all 100 images
  - 95 unique mask files per model (5 images appear in 2 categories — intentional cross-listing)
  - `results/counts.csv`: 1,300 rows (13 models × 100 entries)
  - `results/timing.csv`: 1,200 rows with valid timing (StarDist has NaN — see note below)
- [x] **Wrote `plot_results.py`** — generates 5 separate figure panels (A–E, PDF + PNG each)
- [x] **Panel A redesigned**: magma colormap, SD error bars with x-offsets, uniform line thickness
- [x] **Panel D added**: per-image rank grid (2×5), one subplot per rank position within categories
- [x] **Panel E added**: per-category grid (3×3), per-image counts with tab20 colormap
- [x] **`plot_masks.py` written** — Panel F Part A: 10 rows × 7 columns (intensity histogram + raw DAPI + 7 models + consensus), 1024×1024 crops, steelblue fill + white contours, hardcoded curated images per category (High confluency excluded)
- [x] **Viewer: raw-only mode** — "Raw image only" checkbox in Filled mode skips mask overlay
- [x] **Viewer: stale-transition removed** — CSS override disables Streamlit's opacity fade between re-runs
- [x] **Viewer: per-pixel model hover** — Consensus single-image view has a toggleable "Per-pixel model hover" checkbox (default OFF). When ON: uses Plotly `go.Image` with per-pixel `text` array (bitmask LUT, vectorised); hovering shows only the active models at that pixel (no false 0-vote entries). When OFF: uses fast `st.image` (PNG transfer, ~10–30× faster than Plotly JSON serialisation). Zoom slider available in both Filled mode and Consensus hover-OFF mode; Plotly's built-in pan/zoom used in hover-ON mode.

### Known data notes
- **5 images appear in 2 categories each** (intentional — one image can exhibit multiple difficulties):
  - HA18 → Low confluency + Mitotic
  - HA29 → High confluency + High intensity
  - HA43 → Flat-field + Low intensity
  - HA6 → Clustered + Flat-field
  - HA9 → Low confluency + Defocused
- **StarDist timing is missing**: StarDist masks were cached from the very first run (before
  timing.csv existed). To get StarDist timing: `rm -rf results/masks/stardist_2d_fluo/` then
  re-run with `--models stardist_2d_fluo`.

### Next steps
- [ ] **Run `plot_masks.py`** to generate Panel F Part A (run on nuclei-bench env; check output at `results/figures/panel_F_masks_partA.{pdf,png}`)
- [ ] Part B of mask gallery: add `--normalise` support to `plot_masks.py` and re-run inference on the 7 selected images
- [ ] **Re-run StarDist** to get timing data (optional, see note above)
- [ ] Integrate Panel F reference into manuscript
- [ ] Write supplementary methods text describing the benchmark

### How to resume / run the benchmark

```bash
conda activate nuclei-bench
cd /home/athena/1_CODES/AggreQuant_paper/benchmarks/nuclei_segmentation
python run_benchmark.py \
    --data-dir "/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19"
```

The script will:
1. Detect existing masks in `results/masks/<model_id>/` and skip those images
2. Load previous timing from `results/timing.csv` for cached entries
3. Only load a model if there are un-segmented images for it
4. Write complete `counts.csv` and `timing.csv` covering all models (cached + new)

To run specific models only: `--models deepcell_nuclear deepcell_mesmer`
To run only the new cell-channel configs: `--models deepcell_mesmer_with_cell cellpose_cyto2_with_cell cellpose_cyto3_with_cell`
To skip mask saving: `--no-masks`
To force CPU: `--no-gpu`

---

## Bugs Fixed During Development

1. **DeepCell `NuclearSegmentation()` TypeError**: DeepCell 0.12.10 changed the API.
   `NuclearSegmentation()` no longer works — must use `NuclearSegmentation.from_version("1.1")`.
   `Mesmer(model=None)` still works as before.

2. **DeepCell access token required**: DeepCell 0.12.10 requires `DEEPCELL_ACCESS_TOKEN`
   env var to download model weights. Token is set automatically in the conda activate.d script.

3. **cuDNN version mismatch**: TF 2.8 needs `libcudnn.so.8` but pip-installed PyTorch brings
   cuDNN 9. Fixed by installing `nvidia-cudnn-cu11==8.6.0.163` (provides both .so.8 and .so.9)
   and adding the lib path via conda activate.d script.

4. **Cellpose `__version__`**: Cellpose 3.x removed `cellpose.__version__`. Use
   `importlib.metadata.version("cellpose")` instead.

---

## Benchmark Results Summary

### Inference speed (GPU, RTX 3090)

| Model | Mean time/image (s) |
|-------|-------------------|
| InstanSeg | 0.20 |
| InstanSeg +cell | 0.21 |
| DeepCell Nuclear | 2.10 |
| Mesmer | 2.44 |
| Mesmer +cell | 2.53 |
| Cellpose nuclei | 3.48 |
| Cellpose cyto2 +nuc | 4.17 |
| Cellpose cyto3 | 4.66 |
| Cellpose cyto2 +cell | 4.76 |
| Cellpose cyto3 +nuc | 4.93 |
| Cellpose cyto3 +cell | 5.27 |
| Cellpose cyto2 | 5.44 |
| StarDist 2D | **N/A** (timing not captured) |

### Category median count (across all models)

| Category | Median count |
|----------|-------------|
| Low confluency | 411 |
| Debris | 660 |
| Mitotic | 724 |
| Low intensity | 828 |
| Defocused | 925 |
| Flat-field | 1420 |
| High confluency | 1612 |
| High intensity | 1757 |
| Clustered | 2257 |

### Key findings
- **InstanSeg is ~17× faster** than the slowest model (Cellpose cyto2) at 0.20 vs 5.44 s/image.
- **Cellpose cyto2 +cell dramatically under-counts** in dense categories (e.g., 806 vs 2490 in
  Clustered). Adding the FarRed cell channel causes over-merging of nuclei into cell regions.
- **InstanSeg consistently under-counts** relative to the consensus, especially in Low intensity
  (324 vs median 828) and Defocused (537 vs median 925).
- **DeepCell Mesmer and StarDist tend to count the most** across categories.
- **+nuc variants** (DAPI duplicated as both channels) barely deviate from base models.
- **Debris category** shows the highest inter-model disagreement (CV up to 200%+).

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `BENCHMARK_PLAN.md` | Full technical plan: env setup, model APIs, code architecture, output specs |
| `WORK_STATUS.md` | This file — current state and handoff context |
| `run_benchmark.py` | Main benchmark script — runs all 13 models on curated images (with checkpoint) |
| `plot_results.py` | Generates 5 supplementary figure panels (A–E) |
| `select_images.py` | Image selection script (used during curation, not needed for benchmark) |
| `paper_text_image_selection.md` | Draft supplementary methods text for image selection |
| `requirements_frozen.txt` | Exact pip freeze of the nuclei-bench environment |
| `results/counts.csv` | Nuclei counts + area stats per image × model (1,300 rows) |
| `results/timing.csv` | Inference timing per image × model (1,200 with valid times) |
| `results/masks/<model_id>/*.tif` | Saved label masks (uint16, zlib compressed), 95 per model |
| `results/figures/panel_A_counts.{pdf,png}` | Line plot: mean count ± SD per model across categories |
| `results/figures/panel_B_agreement.{pdf,png}` | Box plot: inter-model count CV per category |
| `results/figures/panel_C_timing.{pdf,png}` | Bar chart: GPU inference time per model |
| `results/figures/panel_D_per_image.{pdf,png}` | 2×5 grid: per-rank-position counts across categories |
| `results/figures/panel_E_per_category.{pdf,png}` | 3×3 grid: per-image counts within each category |
| `plot_masks.py` | Mask gallery figure (Part A) — 10×7 grid, 1024×1024 crops, steelblue fill + white contours + consensus heatmap |
| `results/figures/panel_F_masks_partA.{pdf,png}` | Gallery: 10 rows (hist+raw+7 models+consensus) × 7 category columns |
| `docs/supplementary_nuclei_segmentation_benchmark.md` | High-level figure design (panels, narrative) |

---

## plot_results.py Design

Generates 5 separate figures (each saved as PDF + PNG in `results/figures/`):

### Panel A — Count line plot (`panel_A_counts`)
- One line per model, x = categories sorted by median count (ascending), y = mean nuclei count
- **SD error bars** on each point (alpha=0.2, capsize=1) showing within-category spread
- Small per-model x-offset (±0.15) to prevent error bar overlap
- **Magma colormap** (sequential, trimmed: n_models+2 slots to avoid lightest yellows)
- Uniform line thickness (1.5pt) for all models
- Different marker symbol per model (circle, square, diamond, triangles, etc.)
- Legend ordered: StarDist → Cellpose (nuclei, cyto2, cyto3) → DeepCell → InstanSeg
- Legend in single column, outside plot on the right

### Panel B — Agreement box plot (`panel_B_agreement`)
- Per-image coefficient of variation (CV%) of counts across the 9 single-channel models
- One box per category; shows where models disagree most

### Panel C — Timing bar chart (`panel_C_timing`)
- Horizontal bars: mean inference time per model, sorted fastest on top
- Error bars = SD across images
- Colored by framework: purple = TensorFlow, blue = PyTorch
- Models with no timing data (StarDist) are excluded with a console warning

### Panel D — Per-image rank grid (`panel_D_per_image`)
- 2×5 grid of subplots, one per within-category image rank (1–10)
- Within each category, images ranked by median count across all models
- Subplot k shows each model's count for the k-th ranked image in every category
- Same category x-axis order as Panel A; magma colormap
- No error bars (single image per point)

### Panel E — Per-category image grid (`panel_E_per_category`)
- 3×3 grid, one subplot per difficulty category
- x = individual images sorted by median count (ascending), y = nuclei count
- One line per model; **tab20 colormap** (categorical, maximally distinct colors)
- Horizontal grid lines only; marker size 5.5

### Running
```bash
conda activate nuclei-bench
python plot_results.py [--results-dir results] [--output-dir results/figures] [--dpi 300]
```

---

## User Preferences (from CLAUDE.md and conversation)

- Do NOT append attribution lines to git commits
- Always read `PROGRESS.md` at session start
- The benchmark code must be standalone (no AggreQuant imports)
- User wants parallelized computation where possible
- Image pixel size: 0.325 µm/pixel (20x objective, GE InCell Analyzer)
- User prefers to run the benchmark themselves (not via Claude's Bash tool) to see live output
- Always update `WORK_STATUS.md` with new info; never delete previous info unless it is now incorrect
