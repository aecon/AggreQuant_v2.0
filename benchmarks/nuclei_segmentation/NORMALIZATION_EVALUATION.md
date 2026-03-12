# Evaluating the Effect of Background Normalization on Nuclei Segmentation

## Context

AggreQuant's `StarDistSegmenter._preprocess` applies a Gaussian-based background
normalization before nuclei segmentation:

```python
denoised = gaussian(img, sigma=2)        # denoise
background = gaussian(denoised, sigma=50) # estimate background
normalized = denoised / (background + 1e-8)
```

The nuclei benchmark reproduces this exact preprocessing via `normalize_background()`
in `run_benchmark.py` (activated with `--normalize`). Results are saved to
`results_normalized/` alongside the raw results in `results/`.

The question: **does this background normalization actually improve nuclei
segmentation, and if so, under which conditions?**

## Why internal consistency (not ground truth)

There are no manual annotations for these 100 images. Instead, all evaluation
strategies below use **inter-model agreement** as a proxy for correctness:

- More models agreeing on the same pixels → more likely correct.
- Lower sensitivity to illumination artifacts → more robust pipeline.
- Stable morphometric distributions → normalization is not distorting biology.


## Evaluation Strategies

### 1. Paired Count Comparison

Compare nuclei counts per image, per model between raw and normalized runs.

- **Delta count**: `count_normalized - count_raw` per (image, model) pair.
  Normalization should reduce extreme outliers, not systematically push counts
  up or down.
- **Category-stratified view**: The effect should be strongest for categories
  where background is the root problem — **flat_field** (illumination gradient)
  and **low_intensity** (dim signal on noisy background). Expect minimal effect
  on **defocused** or **debris** (different failure modes).

### 2. Inter-Model Agreement (CV) — Raw vs. Normalized

The coefficient of variation (CV) of nuclei counts across the 7 single-channel
models is already computed per image in `plot_results.py`. The key test:

> Does normalization reduce inter-model disagreement?

- Compute `CV_raw` and `CV_normalized` per image.
- **Paired Wilcoxon signed-rank test** (or paired t-test) per category.
- **Scatter plot** of `CV_raw` vs `CV_normalized` with the diagonal — points
  below the line mean normalization helped.

This is the strongest available metric.

### 3. Consensus Stability

Use the pixel-wise consensus heatmap (model vote count per pixel) to compute a
**consensus score** per image:

```
consensus_score = mean(vote_map[vote_map > 0]) / N_models
```

A value near 1.0 means all models agree on every foreground pixel. Compare this
score between raw and normalized — normalization should improve consensus in
illumination-challenged images.

### 4. Area Distribution Consistency

Background normalization should not change *what* is a nucleus, but it may
change how models delineate boundaries.

- Compare **median area** distributions (raw vs normalized) per model.
- Large shifts suggest normalization is altering boundary detection, not just
  improving detection reliability.
- A good outcome: counts change (better detection) but median areas stay stable
  (same objects, found more reliably).

### 5. Per-Category Effect Size Summary Table

Summarize all metrics in a single table:

| Category       | Median ΔCV | Median Δcount | p-value | Interpretation     |
|----------------|-----------|---------------|---------|--------------------|
| flat_field     | -12%      | +15           | 0.003   | Strong benefit     |
| low_intensity  | -8%       | +10           | 0.01    | Moderate benefit   |
| defocused      | +1%       | -2            | 0.8     | No effect          |
| ...            |           |               |         |                    |

*(Values above are illustrative; actual numbers come from the analysis.)*

This tells you *where* normalization matters and where it does not.

### 6. Visual Spot-Check (Panel F Comparison)

Regenerate `plot_masks.py` for the same 7 curated images using
`results_normalized/` masks. A side-by-side comparison of raw vs. normalized
mask galleries — especially for the flat-field and low-intensity examples — is
the most convincing visual evidence.


## Expected Outcomes by Category

| Category         | Expected normalization effect | Rationale                                    |
|------------------|------------------------------|----------------------------------------------|
| flat_field       | Strong benefit               | Corrects illumination gradient directly       |
| low_intensity    | Moderate benefit             | Lifts dim signal relative to local background |
| high_intensity   | Minimal / neutral            | Signal already strong                         |
| low_confluency   | Minimal                      | Few nuclei, background not the issue          |
| high_confluency  | Minimal                      | Dense signal, background less relevant        |
| clustered        | Minimal                      | Separation problem, not background problem    |
| mitotic          | Minimal                      | Shape problem, not background problem         |
| debris           | Uncertain                    | May amplify debris if debris is dim           |
| defocused        | Minimal / negative           | Out-of-focus blur ≠ background; may amplify   |


## Implementation

All outputs are saved to `results_comparison/`.

### Generated outputs

| File | Script | Description | Data considered |
|------|--------|-------------|-----------------|
| `cv_scatter.{pdf,png}` | `compare_normalization.py` | Scatter of CV_raw vs CV_norm per image, colored by category | 7 single-channel models; 1 point per image (CV across models); all 9 categories |
| `delta_count_boxes.{pdf,png}` | `compare_normalization.py` | Box plots of (count_norm − count_raw) by category | All 13 models × all images; 1 data point per (image, model) pair |
| `delta_count_per_model.{pdf,png}` | `compare_normalization.py` | Box plots of (count_norm − count_raw) by model | 7 single-channel models × all images across all categories |
| `area_shift.{pdf,png}` | `compare_normalization.py` | Paired bar chart of median nucleus area per model | 7 single-channel models; grand median across all images and categories |
| `heatmap_pct_count.{pdf,png}` | `compare_normalization.py` | Model × category heatmap of median % count change: (norm − raw) / raw | 7 single-channel models × 9 categories; median over 10 images per cell |
| `heatmap_pct_area.{pdf,png}` | `compare_normalization.py` | Model × category heatmap of median % area change: (norm − raw) / raw | 7 single-channel models × 9 categories; median over 10 images per cell |
| `normalization_summary.csv` | `compare_normalization.py` | Per-category statistics: median Δcount, median ΔCV, Wilcoxon p-values | CV: 7 single-channel models; Δcount: all 13 models; per category |
| `intensity_hist_<cat>.{pdf,png}` | `analyze_intensity_detection.py` | Overlaid count histograms (log y-axis) with smooth overlay + endpoint markers | 7 single-channel models; raw masks only; all nuclei in the category |
| `intensity_cdf_<cat>.{pdf,png}` | `analyze_intensity_detection.py` | Cumulative distribution of nucleus intensity per model | Same data; CDF per model |
| `intensity_per_nucleus_<cat>.csv` | `analyze_intensity_detection.py` | Per-nucleus raw data (median intensity, area, model) | All nuclei from all models for the category |
| `intensity_detection_summary.txt` | `analyze_intensity_detection.py` | Summary table: N nuclei, median/Q10/Q25 intensity per model | Same data as histograms |


## Quantitative Results

Summary from `compare_normalization.py` (95 images × 13 models):

| Category       | Median Δcount | Median ΔCV | p-value (CV) | p-value (count) |
|----------------|--------------|-----------|-------------|----------------|
| Low confluency |        -18.5 |    0.1119 |      0.0020 |         0.0039 |
| High confluency|        -72.0 |    0.1118 |      0.0020 |         0.0020 |
| Clustered      |       -246.5 |    0.1956 |      0.0020 |         0.0020 |
| Mitotic        |        -21.0 |    0.1203 |      0.0020 |         0.0273 |
| Defocused      |        -40.5 |    0.1143 |      0.0195 |         0.1055 |
| Flat-field     |        -67.0 |    0.1037 |      0.0020 |         0.0098 |
| Low intensity  |         +7.0 |    0.0506 |      0.0039 |         0.0020 |
| High intensity |        -84.0 |    0.1219 |      0.0020 |         0.0020 |
| Debris         |        -12.5 |    0.0754 |      0.6742 |         0.3884 |

**Key observations from the quantitative analysis:**

- Normalization **increases** inter-model disagreement (positive ΔCV) across all
  categories. Nearly all CV scatter points fall above the diagonal.
- Counts systematically **decrease** after normalization (negative Δcount) for 7/9
  categories. Only low intensity shows a marginal increase (+7).
- The CV increase is statistically significant (p < 0.05) for all categories
  except debris (p = 0.67).
- InstanSeg is the most affected model (median Δ ≈ -300 counts). StarDist and
  DeepCell Nuclear are relatively stable near zero.
- Median nucleus area increases modestly for most models after normalization,
  suggesting the background division amplifies dim edges and expands boundaries.
- These pretrained models already have internal normalization (percentile scaling,
  histogram equalization). The extra background division distorts the intensity
  distributions they were trained on.


### Per-model, per-category breakdown

Median % change in nuclei count (from `heatmap_pct_count.png`):

| Model            | LowConf | HighConf | Clustered | Mitotic | Defocused | Flat-field | LowInt | HighInt | Debris |
|------------------|---------|----------|-----------|---------|-----------|------------|--------|---------|--------|
| StarDist 2D      |   -3.0  |   -3.2   |    -0.3   |  -1.9   |   -1.1    |     0.7    |  -2.7  |   -7.0  |  +2.2  |
| DeepCell Nuclear |   +1.7  |   -0.9   |    -1.5   |   0.0   |   +0.4    |    +0.1    |  +1.5  |   -1.3  |  +4.5  |
| Mesmer           |   +1.4  |   -2.0   |    -7.3   |  +1.1   |   +2.9    |   +19.2    |  -2.9  |   ---   | +16.1  |
| Cellpose nuclei  |   -1.6  |   -3.1   |    -6.0   |  -0.9   |   -4.2    |    +3.2    | +24.4  |   -4.3  |  -0.5  |
| Cellpose cyto2   |  -11.4  |   -5.8   |   -37.6   | -10.3   |  -11.4    |    -8.8    |  -4.6  |   -6.9  |  -7.7  |
| Cellpose cyto3   |   -6.1  |   -4.4   |   -12.2   |  -4.9   |   -1.6    |    ---     |  ---   |   -4.0  |  -2.2  |
| InstanSeg        |  -50.5  |  -41.3   |   -71.3   | -48.6   |  -54.7    |   -43.5    | -38.4  |  -38.4  |   ---  |

Median % change in nucleus area (from `heatmap_pct_area.png`):

| Model            | LowConf | HighConf | Clustered | Mitotic | Defocused | Flat-field | LowInt | HighInt | Debris |
|------------------|---------|----------|-----------|---------|-----------|------------|--------|---------|--------|
| StarDist 2D      |   +3.3  |   +2.2   |    -0.6   |  +0.7   |   +1.0    |    -0.1    |  -2.0  |   +1.5  |  +1.8  |
| DeepCell Nuclear |   +8.8  |   +5.6   |    ---    |  +7.0   |   +6.5    |    +5.7    |  -1.6  |   +5.4  |  +8.1  |
| Mesmer           |  +12.4  |   +7.1   |    +5.3   |  +5.7   |   +5.8    |    -1.7    | -15.4  |   +4.6  |  +7.8  |
| Cellpose nuclei  |   +2.3  |   -1.1   |    -5.6   |  -1.2   |   +1.6    |    -1.1    |  +2.3  |   -2.4  |  +0.2  |
| Cellpose cyto2   |  +21.2  |  +12.3   |   +20.7   | +20.8   |  +13.5    |   +22.2    | +20.2  |  +12.4  | +23.6  |
| Cellpose cyto3   |  +12.1  |   -7.6   |   -13.7   |  +2.4   |   -1.5    |   -12.4    | -10.0  |   -8.6  | +22.2  |
| InstanSeg        |  +14.6  |   +9.9   |   +15.7   |  +8.8   |  +11.2    |   +12.0    |  +6.6  |   +4.0  | +12.9  |


## Visual Inspection Notes

### Low confluency

Easiest category — nuclei are well-separated with clear boundaries.

Quantitatively, count changes are small for most models (StarDist -3%, DeepCell
Nuclear +1.7%, Cellpose nuclei -1.6%). The main outliers are Cellpose cyto2
(-11.4% count) and InstanSeg (-50.5% count). Area expands modestly for most
models (+2 to +12%), with Cellpose cyto2 showing the largest area inflation
(+21.2%).

| Model            | Observation                                                    |
|------------------|----------------------------------------------------------------|
| StarDist 2D      | No visible difference other than slight area expansion (+3.3%) |
| Cellpose nuclei  | Misses some nuclei in both variants; count stable (-1.6%)      |
| Cellpose cyto2   | Mixed: loses 11.4% of counts but areas expand +21.2%          |
| Cellpose cyto3   | Counts drop slightly (-6.1%), areas expand +12.1%             |
| InstanSeg        | Severe count loss (-50.5%) — normalization breaks this model  |
