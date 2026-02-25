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

## Interactive viewer

Streamlit app for browsing DAPI images and mask overlays side-by-side across all models.

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

---

> **Note:** `data/images` is a symlink to the actual dataset location on the SpeedDrive:  
```
/media/athena/SpeedDrive/ATHENA/PROJECT_AggreQuant/2026_02_18_BENCHMARCKS/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19
```
