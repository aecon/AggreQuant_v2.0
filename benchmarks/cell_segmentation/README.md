# Cell Segmentation Benchmark

Compares 8 pretrained whole-cell segmentation model configurations across 9 difficulty
categories (90 curated HCS FarRed images). Sister benchmark to `nuclei_segmentation/`.

Same image set, different channel: primary input is `wv 631 - FarRed` (cytoplasm),
with `wv 390 - Blue` (DAPI) available as an optional nuclear hint for the `_with_nuc` configs.

---

## Run the benchmark

```bash
conda activate cell-bench
python run_benchmark.py --data-dir data/images
```

Outputs saved to `results/masks/`, `results/counts.csv`, `results/timing.csv`.
Supports checkpoint/resume — re-run the same command to continue after interruption.

Useful options:
- `--models cellpose_cyto2 deepcell_mesmer` — run specific models only
- `--no-masks` — skip saving mask files (faster, saves disk)
- `--no-gpu` — force CPU

> **Important:** use `conda activate` (not `conda run`) so the activate scripts fire and
> set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which is required for DeepCell.

---

## Re-generate figures

```bash
conda activate cell-bench
python plot_results.py          # all 8 models → results/figures/
```

---

## Generate mask gallery (Panel F)

```bash
conda activate cell-bench
python plot_masks.py \
    --data-dir data/images \
    [--masks-dir results/masks] \
    [--output-dir results/figures] \
    [--crop-size 1024] \
    [--dpi 300]
```

Produces `results/figures/panel_F_masks_partA.{pdf,png}`: a 7-row × 7-column grid
showing one curated FarRed image per difficulty category for each of the 4
single-channel models, plus a histogram row, a raw FarRed row, and a consensus heatmap row.

---

> **Note:** `data/images` is a symlink to the same SpeedDrive directory used by the
> nuclei benchmark:
```
/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19
```
