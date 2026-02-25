# Cell Segmentation Benchmark — Implementation Plan

**Date**: 2026-02-25
**Location**: `AggreQuant_paper/benchmarks/cell_segmentation/`
**Conda environment**: `cell-bench`

---

## 1. Objective

Compare pretrained whole-cell segmentation models on the same HCS images used
in the nuclei benchmark (FarRed channel), with DAPI as an optional nuclear hint.
No manual ground truth — evaluation is based on inter-model agreement, per-category
count comparison, and inference speed.

Produces a **supplementary figure** for the AggreQuant methods paper
(Cell Reports Methods), paired with the nuclei segmentation figure.

---

## 2. Relationship to the Nuclei Benchmark

The images are the **same 90 FOVs** used in the nuclei benchmark, from the same
SpeedDrive directory:

```
/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/
  2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19/
```

- Nuclei benchmark: uses `*wv 390 - Blue*` (DAPI) as primary channel.
- Cell benchmark: uses `*wv 631 - FarRed*` (cell membrane/cytoplasm) as primary
  channel, with DAPI available as a nuclear hint for two-channel configurations.

`data/images` is a symlink to the same SpeedDrive directory.
`cell-bench` is a separate conda env cloned from `nuclei-bench` (`pip install --no-deps -r requirements_frozen.txt`).

**Environment note (2026-02-25):** During setup, `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`
was temporarily added to the conda activate scripts to work around a DeepCell/protobuf incompatibility.
The definitive fix was downgrading protobuf to 3.20.3 (`pip install "protobuf>=3.9.2,<4"`),
after which the env var was removed. The same downgrade was applied to `nuclei-bench`.

---

## 3. Models and Configurations (8 total)

Models that are nuclei-specific (StarDist `2D_versatile_fluo`, DeepCell
`NuclearSegmentation`, Cellpose `nuclei`) are excluded — they are not meaningful
on a cytoplasmic channel.

### Single-channel (FarRed only)

| ID | Model | Channels | Notes |
|----|-------|----------|-------|
| `cellpose_cyto2` | Cellpose cyto2 | `[0, 0]` | FarRed only, auto-diameter |
| `cellpose_cyto3` | Cellpose cyto3 | `[0, 0]` | FarRed only, auto-diameter |
| `deepcell_mesmer` | Mesmer | `[zeros, FarRed]` | Membrane only; compartment=`whole-cell` |
| `instanseg_fluorescence` | InstanSeg | `(H, W)` | Single-channel FarRed; target=`cells` |

### Two-channel (FarRed + DAPI nuclear hint)

| ID | Model | Channels | Notes |
|----|-------|----------|-------|
| `cellpose_cyto2_with_nuc` | Cellpose cyto2 | `[1, 2]` → `[FarRed, DAPI]` | Cytoplasm + nuclear hint |
| `cellpose_cyto3_with_nuc` | Cellpose cyto3 | `[1, 2]` → `[FarRed, DAPI]` | Cytoplasm + nuclear hint |
| `deepcell_mesmer_with_nuc` | Mesmer | `[DAPI, FarRed]` | Nuclear + membrane; compartment=`whole-cell` |
| `instanseg_fluorescence_with_nuc` | InstanSeg | `(2, H, W)` → `[DAPI, FarRed]` | Two-channel; target=`cells` |

### Channel conventions

**Cellpose** `channels=[1, 2]`: index-1 in the (H, W, 2) array is cytoplasm;
index-2 is the nuclear hint.
```python
img_2ch = np.stack([img_cell, img_nuc], axis=-1)  # img[:,:,0]=FarRed, img[:,:,1]=DAPI
masks = model.eval(img_2ch, channels=[1, 2], ...)
```

**Mesmer without nuclear**: provide zeros for the nuclear channel (channel 0).
```python
img_2ch = np.stack([np.zeros_like(img_cell), img_cell], axis=-1)  # (H, W, 2)
```

**Mesmer with nuclear**: standard Mesmer convention is `[nuclear, membrane]`.
```python
img_2ch = np.stack([img_nuc, img_cell], axis=-1)  # (H, W, 2)
labels = app.predict(img_4d, image_mpp=0.325, compartment="whole-cell")
```

**InstanSeg** two-channel: `(2, H, W)` with `[DAPI, FarRed]` order.
```python
img_2ch = np.stack([img_nuc, img_cell], axis=0).astype(np.float32)  # (2, H, W)
result = model.eval_small_image(img_2ch, pixel_size=image_mpp, target="cells")
```

---

## 4. Input Data

### Directory Structure

```
cell_segmentation/data/images -> SpeedDrive/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19/
  ├── 01_low_confluency/
  │   ├── *wv 390 - Blue*.tif    (DAPI — nuclear hint)
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
- Nuclear hint: derived by replacing `wv 631 - FarRed` → `wv 390 - Blue` in filename.

---

## 5. Code Architecture

```
cell_segmentation/
├── BENCHMARK_PLAN.md
├── run_benchmark.py      # Main script (adapted from nuclei benchmark)
├── plot_results.py       # Figure panels A–E (adapted)
├── plot_masks.py         # Mask gallery (adapted)
├── data/
│   └── images -> <SpeedDrive symlink>
└── results/
    ├── masks/            # Per-model label masks
    ├── counts.csv        # cell_count, mean_area_px, median_area_px
    └── timing.csv
```

---

## 6. Output Specification

### `counts.csv`

| Column | Type | Description |
|--------|------|-------------|
| image_name | str | FarRed filename |
| category | str | Folder name (e.g. `01_low_confluency`) |
| model_id | str | Config ID (e.g. `cellpose_cyto2`) |
| cell_count | int | Number of detected cells |
| mean_area_px | float | Mean area of detected cells in pixels |
| median_area_px | float | Median area of detected cells in pixels |

### `timing.csv`

Same structure as nuclei benchmark.

---

## 7. Key Differences from Nuclei Benchmark

| Aspect | Nuclei | Cell |
|--------|--------|------|
| Primary channel | DAPI (Blue) | FarRed |
| Optional hint | FarRed as membrane | DAPI as nuclear hint |
| Models excluded | — | StarDist, DeepCell Nuclear, Cellpose nuclei |
| `needs_*` flag | `needs_cell` | `needs_nuc` |
| Mesmer compartment | `nuclear` | `whole-cell` |
| InstanSeg target | `nuclei` | `cells` |
| Count column | `nuclei_count` | `cell_count` |

---

## 8. Execution Checklist

- [ ] Verify `nuclei-bench` env is active and all imports work
- [ ] Confirm symlink `data/images` resolves to SpeedDrive
- [ ] Run `run_benchmark.py --data-dir data/images`
- [ ] Inspect `results/counts.csv` and `results/timing.csv`
- [ ] Run `plot_results.py` to generate panels A–E
- [ ] Run `plot_masks.py` to generate panel F mask gallery
- [ ] Manually curate/annotate panel F if needed
