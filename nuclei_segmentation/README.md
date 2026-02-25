# Nuclei Segmentation Benchmark

Compares 13 pretrained nuclei segmentation model configurations across 9 difficulty categories (100 curated HCS images). See `WORK_STATUS.md` for full context.

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
conda activate nuclei-bench
streamlit run viewer.py
```

The viewer expects images at `data/images/` and masks at `results/masks/`. Override with CLI args:

```bash
streamlit run viewer.py -- \
    --data-dir /path/to/curated/images \
    --masks-dir /path/to/results/masks
```

---

## Re-generate figures

```bash
conda activate nuclei-bench
python plot_results.py          # excludes +cell models by default → results/figures/no_cell/
python plot_results.py --include-cell   # all 13 models → results/figures/
```
