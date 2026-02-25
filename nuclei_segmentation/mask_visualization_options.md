# Mask Visualization Options

## Option 1 — Category spotlight gallery (Panel F, best for paper)

A grid where **rows = selected images** (1–2 per category, chosen by highest inter-model count CV) and **columns = 7 representative single-channel models** (StarDist, Cellpose nuclei, Cellpose cyto2, Cellpose cyto3, DeepCell Nuclear, Mesmer, InstanSeg), plus one column for the raw DAPI.

Each cell: colorized instance label map (each nucleus a distinct random color). For dense fields, crop to a 512×512 ROI to make individual nuclei legible.

**Pros**: tells the narrative clearly (here's what each model sees under each challenge), paper-ready, selection is data-driven (high-CV images are genuinely interesting).
**Cons**: requires picking good ROIs; 2040×2040 images need cropping.

---

## Option 2 — Pixel-level consensus heatmap

For a selected image, binarize all single-channel model masks and sum them pixel-wise → 0–7 foreground votes. Overlay as a heatmap on the DAPI image. Shows spatially *where* models agree and where they diverge (edges, touching nuclei, debris).

**Pros**: visually striking, quantitative spatial interpretation, works at full image resolution.
**Cons**: single number per pixel loses instance information; best used as a complement to Option 1, not a standalone.

---

## Option 3 — Contour overlay on DAPI

Draw only the **outlines** of each model's predicted nuclei on top of the DAPI image. One color per model (5–6 models → distinct colors).

**Pros**: all models visible in one image; agreements show up as overlapping lines, disagreements as isolated colored outlines.
**Cons**: very crowded for dense images (clustered category); works best for low-confluency or mitotic.

---

## Option 4 — Interactive viewer (exploration only)

Load DAPI + all 13 mask layers in **napari** (it's already in the conda env or easy to add). Toggle model layers on/off interactively.

**Pros**: best for your own exploration before deciding which images/crops to highlight.
**Cons**: not for the paper directly.

---

## Recommendation

**For the paper**: do Option 1 (gallery grid) as Panel F, with Option 2 (consensus heatmap) as an inset or supplementary figure for 1–2 particularly interesting cases (e.g., a clustered field or a debris field).

**For image selection**: write a small script that computes per-image CV from `counts.csv`, ranks images within each category by CV descending, then pulls up the masks for the top 1–2 per category. That automates the "interesting image selection."

The implementation would be a new `plot_masks.py` script in `benchmarks/nuclei_segmentation/` that:
1. Reads `counts.csv` to identify high-disagreement images per category
2. Loads DAPI raws from the SpeedDrive benchmark dir + corresponding model masks from `results/masks/`
3. Produces a matplotlib figure with `label2rgb`-colorized masks + optional crop ROI

---

## Chosen Design: Two-Part Supplementary Figure (Option A — Confirmed)

### Part A — Raw inference (no normalization)

**Layout**: 9 rows × 8 columns (1 raw + 7 models).

- **Rows**: One per difficulty category (9 categories). One image randomly selected per category (fixed seed for reproducibility).
- **Columns**: Column 0 = raw DAPI image (percentile-stretched to [0,1]). Columns 1–N = one per model.
- **Models shown** (7 family representatives, `_with_nuc` and `_with_cell` variants excluded):
  `stardist_2d_fluo`, `cellpose_nuclei`, `cellpose_cyto2_no_nuc`, `cellpose_cyto3_no_nuc`, `deepcell_nuclear`, `deepcell_mesmer`, `instanseg_fluorescence`.
- **Crop**: 512×512 center crop of each 2040×2040 image.
- **Label rendering**: All nuclei filled in a single solid color (e.g., steelblue). White contour lines overlaid on instance boundaries (via `skimage.segmentation.find_boundaries`, mode='inner') to separate touching nuclei.
- **Row labels**: Category names on the left. Column headers: model names on top.
- **Image selection**: random with fixed seed (no data-driven CV selection — random is sufficient for a visual gallery).

### Part B — Normalized inference

**Identical layout to Part A**, but each DAPI image is passed through a **percentile normalization** (e.g., `csbdeep.utils.normalize(img, 1, 99.8)` or equivalent) before being fed to every model. This tests whether pre-normalization improves segmentation for edge cases (low intensity, high intensity, defocused).

- Same randomly selected images as Part A (same seed), same 512×512 center crops.
- Same visual rendering (single fill color + white contours).
- Same 7 models.
- The normalization step is applied uniformly to all models before inference. Models that already do internal normalization (e.g., InstanSeg) will effectively receive double-normalization — this is intentional (tests robustness to over-normalization).

### Implementation

Script: `plot_masks.py` in `benchmarks/nuclei_segmentation/`

```
python plot_masks.py \
    --data-dir <curated benchmark dir> \
    --masks-dir results/masks \
    --output-dir results/figures \
    --seed 42 \
    --part A   # or B (B adds percentile normalization before inference)
```

Part A reads existing saved masks directly (no model re-run needed).
Part B requires re-running inference with normalization applied — either re-use `run_benchmark.py` with a `--normalize` flag, or run inference inline within `plot_masks.py` (inline preferred to keep the script self-contained).
