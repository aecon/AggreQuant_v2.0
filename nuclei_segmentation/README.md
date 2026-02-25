# Nuclei Segmentation Benchmark

Compares 13 pretrained nuclei segmentation model configurations across 9 difficulty categories (100 curated HCS images). See `WORK_STATUS.md` for full context.

---

## Run the benchmark

```bash
conda activate nuclei-bench
python run_benchmark.py --data-dir data/
```

Outputs saved to `results/masks/`, `results/counts.csv`, `results/timing.csv`. Supports checkpoint/resume.

Useful options:
- `--models stardist_2d_fluo deepcell_nuclear` — run specific models only
- `--no-masks` — skip saving mask files
- `--no-gpu` — force CPU

---

## Re-generate figures

```bash
conda activate nuclei-bench
python plot_results.py          # excludes +cell models by default → results/figures/no_cell/
python plot_results.py --include-cell   # all 13 models → results/figures/
```

---

## Generate mask gallery (Panel F)

```bash
conda activate nuclei-bench
python plot_masks.py \
    --data-dir "/media/athena/SpeedDrive/.../NUCLEI-BENCHMARK_AE-CURATED-2026-02-19" \
    [--masks-dir results/masks] \
    [--output-dir results/figures] \
    [--crop-size 1024] \
    [--dpi 300]
```

Produces `results/figures/panel_F_masks_partA.{pdf,png}`: a 10-row × 7-column grid showing
one curated image per difficulty category (columns) for each of 7 single-channel models (rows),
plus an intensity histogram row, a raw DAPI row, and a consensus heatmap row.
The 7 categories shown are: Low confluency, Clustered, Mitotic, Debris, Flat-field,
High intensity, Defocused.

---

## Interactive viewer

Streamlit app for browsing DAPI images and mask overlays across all models.

### Setup (one-time)

```bash
conda activate nuclei-bench
pip install -r requirements_viewer.txt
```

### Run

```bash
streamlit run viewer.py
```

The viewer expects images at `data/images/` and masks at `results/masks/`. Override with CLI args:

```bash
streamlit run viewer.py -- --data-dir /path/to/images --masks-dir /path/to/masks
```

### Viewer features

- **Filled — single model**: colorized instance labels blended over DAPI with white contour boundaries.
  - "Raw image only" checkbox shows just the DAPI with no overlay.
- **Consensus heatmap**: pixel-wise sum of binary foreground masks from all selected models,
  mapped through the jet colormap. Background (0 votes) shows the unmodified DAPI.
  - Single image view uses a Plotly interactive chart — hover over any pixel to see which models
    contribute at that location and the total vote count.

---

> **Note:** `data/images` is a symlink to the actual dataset location on the SpeedDrive:
```
/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19
```
