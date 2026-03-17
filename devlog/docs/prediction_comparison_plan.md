# Prediction Map Comparison Plan

Strategies for comparing predictions across loss configurations and
evaluating model quality beyond standard pixel-level metrics.

The 4 loss runs: `baseline_no_scheduler`, `baseline`, `bce_pw3`, `dice03_bce07_pw3`.

---

## 1. Visual comparisons

### 1.1 Side-by-side prediction panels

Multi-panel figure showing the same image region across all 4 models + GT.
Columns: raw image, model 1 prob map, model 2 prob map, ..., GT mask.
Reveals where each model over/under-predicts relative to others.

### 1.2 Probability map comparison (not just binary)

Show raw probability maps side by side instead of thresholded masks.
Reveals:
- **Confidence differences** — does Dice+BCE produce sharper (closer to 0/1)
  probabilities than pure weighted BCE?
- **False positive confidence** — are FPs high-confidence (model is certain
  but wrong) or borderline (near threshold)?
- Threshold-independent: the probability map is the true model output.

### 1.3 Zoomed inset panels

Full images (2040x2040) hide differences. Pick 3-4 representative crops
(~200x200) showing:
- A dense aggregate cluster
- An isolated small aggregate
- A region where models disagree
- A region with annotation edge noise

---

## 2. Edge-aware evaluation

Standard pixel-level TP/FP/FN is misleading at boundaries because GT edges
are unreliable (annotators are confident about cores, uncertain about edges).

### 2.1 Eroded GT comparison

Erode GT mask by K pixels (e.g. 2-3). Evaluate only on the eroded cores.
This answers: "Do models agree on aggregate centers?" independently of
boundary placement. Compute Dice/IoU on eroded GT to isolate core detection
from edge disagreements.

### 2.2 Object-level centroid matching

For each connected component in the prediction and GT, compute centroids.
Match predicted centroids to GT centroids within a distance threshold (e.g.
10 pixels). This gives:
- **Object-level precision**: fraction of predicted objects with a matching GT
- **Object-level recall**: fraction of GT objects with a matching prediction
- Independent of exact boundary shape — a correctly detected aggregate with
  slightly wrong edges still counts as TP.

### 2.3 Edge exclusion band

Dilate GT boundaries by K pixels to create an exclusion band. Pixels in this
band are excluded from FP/FN counting. Only FPs clearly away from any GT
object are counted. Reduces the impact of boundary annotation noise on metrics.

---

## 3. Quantitative analysis of errors

### 3.1 FP/FN confidence distribution

For each model, collect the predicted probability at every pixel classified
as FP or FN (at a given threshold). Plot as histograms:

- **FP probability histogram**: Are false positives high-confidence (p > 0.8,
  model is confident but wrong — concerning) or borderline (0.5 < p < 0.6,
  just above threshold — less concerning, threshold-sensitive)?
- **FN probability histogram**: Are false negatives low-confidence (p < 0.2,
  model didn't see anything — missed detection) or borderline (0.4 < p < 0.5,
  just below threshold — threshold-sensitive)?

**Method**: After thresholding at p, extract `prob_map[FP_mask]` and
`prob_map[FN_mask]`, plot distributions. Compare across loss configurations.

### 3.2 Core vs edge error classification

Distinguish whether each error pixel is a core error (bad) or an edge error
(expected given annotation uncertainty).

**Method**:
1. Compute the **distance transform** of the GT mask: for each foreground
   pixel, its distance to the nearest background pixel (= distance to edge).
2. Define edge pixels as those with distance < K (e.g. K=3 pixels).
3. Define core pixels as those with distance >= K.
4. Classify each FP/FN pixel:
   - **FN at core** (distance to GT edge >= K): model missed the center of an
     annotated aggregate. This is a genuine miss.
   - **FN at edge** (distance to GT edge < K): model disagrees on boundary
     placement. Expected and not penalized.
   - **FP near GT edge** (FP pixel within K pixels of any GT foreground):
     model extends slightly beyond annotation boundary. Expected.
   - **FP far from GT** (FP pixel > K pixels from any GT foreground): model
     hallucinated an aggregate. Genuine false positive.

**Quantification**: Report the fraction of FPs and FNs that are edge vs core
errors. A good model has mostly edge errors and few core errors. Compare
across loss configurations.

**Implementation**: Use `scipy.ndimage.distance_transform_edt` on the GT
mask and on the inverse GT mask.

### 3.3 Error analysis in low-quality image regions

Check whether errors correlate with image quality — specifically whether
missed/hallucinated aggregates occur in smudged, unfocused, or dim areas.

**Method A — Local intensity analysis**:
1. Compute the local mean intensity in a window (e.g. 32x32) around each
   FP/FN connected component's centroid.
2. Compare against the global intensity distribution of GT aggregate regions.
3. If FPs cluster in dim regions (below e.g. 10th percentile of GT aggregate
   intensity), the model is detecting noise in underexposed areas.
4. If FNs cluster in dim regions, the model fails on faint aggregates (which
   may also be uncertain annotations).

**Method B — Local focus/sharpness analysis**:
1. Compute a local sharpness metric (e.g. Laplacian variance in a 32x32
   window) at each FP/FN centroid.
2. Low sharpness = unfocused region. If FPs/FNs cluster in low-sharpness
   areas, the model is sensitive to focus quality.
3. Can use the existing `aggrequant.focus` module's patch metrics
   (Brenner gradient, normalized variance) for this.

**Method C — Spatial error map**:
1. Accumulate FP and FN pixels across all images to build spatial error
   frequency maps.
2. Overlay on mean image intensity to check if errors concentrate in specific
   plate regions (e.g. well edges where focus drops).

**Quantification**: For each FP/FN connected component, report:
- Centroid coordinates
- Mean local intensity (percentile relative to GT aggregates)
- Local sharpness score
- Classification: core error or edge error (from 3.2)
- Predicted probability at centroid

This gives a per-error table that can be filtered and aggregated.

---

## 4. Summary metrics per model

For each of the 4 loss configurations, report:

| Metric | What it measures |
|---|---|
| Dice, IoU, precision, recall | Standard pixel-level (from training) |
| Eroded-core Dice | Agreement on aggregate centers only |
| Object-level precision/recall | Detection accuracy independent of edges |
| FP core fraction | % of FP pixels that are far from GT (genuine FPs) |
| FN core fraction | % of FN pixels that are at GT cores (genuine misses) |
| Mean FP probability | Average confidence of false positives |
| Mean FN probability | Average confidence of false negatives |
| FP in dim regions (%) | % of FP components in low-intensity areas |
| FN in dim regions (%) | % of FN components in low-intensity areas |

This table directly answers which loss produces the most clinically relevant
errors (core misses) vs benign errors (edge disagreements).

---

## Implementation priority

1. **FP/FN confidence histograms** (3.1) — simplest, most informative
2. **Core vs edge error classification** (3.2) — directly addresses the
   annotation uncertainty problem
3. **Object-level centroid matching** (2.2) — edge-independent detection metric
4. **Zoomed side-by-side panels** (1.3) — visual sanity check
5. **Image quality correlation** (3.3) — explains remaining errors

---

## Running the comparison (`scripts/compare_predictions.py`)

Auto-discovers all checkpoints in `training_output/loss_function/*/checkpoints/best.pt`,
runs inference, and computes all metrics from sections 2-4 above. Use `--group` to
select a different subfolder (e.g., `--group ablation`).

```bash
# All available runs, first image, print summary table:
conda run -n AggreQuant python scripts/compare_predictions.py

# Save CSV + confidence histograms to a directory:
conda run -n AggreQuant python scripts/compare_predictions.py -o training_output/comparison/

# Specific runs only:
conda run -n AggreQuant python scripts/compare_predictions.py --runs baseline dice03_bce07_pw3

# Specific image (by index or filename):
conda run -n AggreQuant python scripts/compare_predictions.py --image 5
conda run -n AggreQuant python scripts/compare_predictions.py --image image_0005.tif

# Custom threshold and edge width:
conda run -n AggreQuant python scripts/compare_predictions.py --threshold 0.4 --edge-width 5
```

### Metrics computed

| Category | Metric | Description |
|---|---|---|
| **Pixel-level** | Dice, IoU, precision, recall | Standard overlap metrics |
| **Pixel-level** | Eroded-core Dice | Dice on GT eroded by `edge_width` — measures agreement on aggregate centers only |
| **Object-level** | Object precision | Fraction of predicted blobs with a GT centroid within `match_radius` |
| **Object-level** | Object recall | Fraction of GT blobs with a matched prediction |
| **Confidence** | Mean/median FP probability | Average predicted probability at false positive pixels |
| **Confidence** | Mean/median FN probability | Average predicted probability at false negative pixels |
| **Confidence** | FP high-confidence fraction | % of FP pixels with p > 0.8 (model is certain but wrong) |
| **Confidence** | FN low-confidence fraction | % of FN pixels with p < 0.2 (model saw nothing) |
| **Core vs edge** | FP edge/core fraction | % of FP pixels near GT boundary (overshoot) vs far from GT (hallucinated) |
| **Core vs edge** | FN edge/core fraction | % of FN pixels at GT boundary (edge disagreement) vs deep inside GT (missed core) |
| **Intensity** | FP/FN in dim regions | % of FP/FN components whose local intensity is below the 10th percentile of GT aggregate intensity |

### Options

| Argument | Default | Description |
|---|---|---|
| `--image` | First image in `symlinks/images/` | Image path, filename, or index |
| `--mask` | Auto-resolved from `symlinks/masks/` | Ground truth mask path |
| `--runs` | All runs with `best.pt` | Run names to compare |
| `--threshold` | `0.5` | Probability threshold |
| `--edge-width` | `3` | Edge band width (pixels) for core/edge classification |
| `--match-radius` | `10.0` | Max centroid distance (pixels) for object matching |
| `-o` / `--output-dir` | Print only | Directory to save CSV and plots |

### Output

- **Summary table** + **detailed counts table** (always printed)
- With `-o`, saves the following files:

#### `comparison_results.csv`
Full metrics table (all computed values per model).

#### `metrics_comparison.png` — How good is each model?
Grouped bar chart with 7 metrics, one bar per model per metric:
- **Dice** — overlap between prediction and GT (harmonic mean of precision+recall). Higher = better overall segmentation.
- **IoU** — same idea as Dice but stricter (intersection / union). Always lower than Dice.
- **Precision** — of all pixels the model predicted as aggregate, what fraction actually is aggregate? Low precision = too many false positives (model is over-predicting).
- **Recall** — of all GT aggregate pixels, what fraction did the model find? Low recall = missed aggregates.
- **Core Dice** — same as Dice but ignoring edge pixels. Tests whether models agree on aggregate centers, removing boundary disagreement noise.
- **Obj Prec** — of all predicted blobs, what fraction matched a real GT blob (by centroid distance)? Low = hallucinated objects.
- **Obj Recall** — of all GT blobs, what fraction was detected? Low = missed objects entirely.

#### `overlay_grid.png` — Where does each model get it right/wrong?
Side-by-side spatial maps:
- Column 1: contrast-enhanced input image
- Columns 2–N: per-model overlay where each pixel is colored:
  - Yellow = TP (both model and GT agree it's aggregate)
  - Magenta = FP (model says aggregate, GT says no)
  - Cyan = FN (GT says aggregate, model missed it)
  - Black = TN (both agree it's background)

Look for: magenta clusters (hallucinated aggregates), cyan clusters (missed aggregates), and how much yellow vs cyan/magenta there is.

#### `core_edge_errors.png` — Are errors at boundaries or deep inside?
Two stacked bar charts (one for FP, one for FN). Each bar is split into:
- **Edge** (light color, bottom) — error pixels within 3px of a GT boundary. These are "boundary disagreement" — model and GT just drew the border differently.
- **Core** (dark color, top) — error pixels far from any GT boundary. These are the real errors — hallucinated aggregates (FP core) or completely missed aggregate centers (FN core).

What to look for: If most errors are edge errors, the model is actually good — it just disagrees on exact boundaries. If core errors dominate, the model is fundamentally wrong (missing or inventing objects).

#### `intensity_correlation.png` — Do errors happen in dim image regions?
Grouped bars per model showing what fraction of error components (connected blobs) fall in dim image regions:
- **FP in dim** (red) — fraction of false positive blobs located where the image is dark. High = model is hallucinating aggregates in low-signal areas.
- **FN in dim** (blue) — fraction of missed GT blobs in dark areas. High = model struggles with faint aggregates.

The annotations (e.g. "2769/3450") show absolute counts: errors_in_dim / total_error_components. The "dim" threshold is the 10th percentile of intensity at GT aggregate pixels.

#### `confidence_histograms.png` — How confident is the model when it's wrong?
Two plots showing the probability distribution at error pixels (line + bar):
- **Left (FP confidence)** — probability values where the model predicted aggregate but GT disagrees. If peaked near 1.0, the model is confidently wrong on its false positives (hard to fix with threshold tuning).
- **Right (FN confidence)** — probability values where the model missed GT aggregate. If peaked near 0.0, the model was confidently ignoring those pixels. If peaked near 0.5, a lower threshold might recover them.

Practical use: If FP confidence is mostly 0.5–0.7, raising the threshold would eliminate many FPs. If FN confidence clusters near 0.4–0.5, lowering the threshold would recover them.

#### `intensity_distributions.png` — What intensities do errors occur at?
One panel per model showing density curves for pixel intensity at TP, FP, FN, and TN locations:
- **TP** (green) — intensity of correctly predicted aggregate pixels
- **FP** (red) — intensity where model predicted aggregate but GT disagrees
- **FN** (blue) — intensity where model missed GT aggregate
- **TN** (gray) — background intensity

Shows whether errors cluster at specific intensity ranges — e.g. FPs in dim regions or FNs where aggregates are faint.
