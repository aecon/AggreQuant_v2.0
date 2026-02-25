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

### Part A — Raw inference (no normalization) — IMPLEMENTED

**Layout**: 10 rows × 7 columns.

- **Rows**: intensity histogram | raw DAPI | 7 single-channel models | consensus heatmap.
- **Columns**: 7 curated difficulty categories (one hardcoded image per category):
  Low confluency, Clustered, Mitotic, Debris, Flat-field, High intensity, Defocused.
  (High confluency excluded — too visually similar to Clustered at the chosen crop.)
- **Models shown** (7 family representatives, `_with_nuc` and `_with_cell` variants excluded):
  `stardist_2d_fluo`, `cellpose_nuclei`, `cellpose_cyto2_no_nuc`, `cellpose_cyto3_no_nuc`,
  `deepcell_nuclear`, `deepcell_mesmer`, `instanseg_fluorescence`.
- **Crop**: 1024×1024 center crop of each 2040×2040 image.
- **Label rendering**: all nuclei filled in a single solid steelblue color. White contour lines
  overlaid on instance boundaries (`skimage.segmentation.find_boundaries`, mode='inner').
- **Intensity histogram**: raw uint16 intensity distribution (64 bins, grey bars on white).
  x-axis max = 95th percentile across all 7 selected images (shared axis for comparability).
- **Consensus heatmap**: pixel-wise sum of binary masks from all 7 models → jet colormap
  (blue = 1 model, red = all 7 agree; black background = 0 votes). Vertical colorbar on the right.
- **Column headers**: category names. **Row labels**: on the left.
- **Image selection**: hardcoded per category (see `SELECTED_IMAGES` dict in `plot_masks.py`).

Output: `results/figures/panel_F_masks_partA.{pdf,png}`

```bash
python plot_masks.py \
    --data-dir <curated benchmark dir> \
    [--masks-dir results/masks] \
    [--output-dir results/figures] \
    [--crop-size 1024] \
    [--dpi 300]
```

Part A reads existing saved masks directly (no model re-run needed).

### Part B — Normalized inference

**Identical layout to Part A**, but each DAPI image is passed through a **percentile normalization**
(`csbdeep.utils.normalize(img, 1, 99.8)` or equivalent) before being fed to every model.
This tests whether pre-normalization improves segmentation for edge cases.

- Same hardcoded images as Part A, same 1024×1024 center crops.
- Same visual rendering (single fill color + white contours, consensus heatmap).
- Same 7 models.
- Not yet implemented — requires adding a `--normalise` flag to `plot_masks.py` and re-running
  inference on the 7 selected images.
