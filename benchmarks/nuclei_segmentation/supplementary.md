# Draft Text: Image Selection for Nuclei Segmentation Benchmark

## For Supplementary Methods

To evaluate nuclei segmentation models under conditions representative of
real-world HCS data variability, we curated a benchmark set of 90 DAPI
fluorescence images (2040 × 2040 pixels, 0.325 µm/pixel) spanning nine
challenging imaging categories: low confluency, high confluency,
clustered/touching nuclei, mitotic figures, defocused fields, flat-field
illumination inhomogeneity, low overall intensity, high overall intensity,
and debris/artifacts (10 images per category). Images were drawn from NT
(non-targeting) and RAB13 (positive control) wells across 55 independent
384-well plates to capture plate-to-plate variability in staining, seeding
density, and imaging conditions. For each candidate image, we computed a
panel of image-level descriptors including mean and foreground-median
intensity, Otsu-thresholded area fraction, variance of Laplacian (focus
quality), background illumination gradient (Gaussian-smoothed at σ = 150
pixels), connected-component size ratios, and relative debris counts.
Categories were filled by greedy ranked assignment: images were sorted by
the defining metric for each category and assigned to the first category for
which they qualified, ensuring that each image appeared in exactly one
category and that confounding overlaps (e.g., defocused images appearing in
the low-confluency set) were resolved by priority. To decorrelate
categories that share correlated proxies — notably high confluency, high
intensity, and clustered nuclei, which are driven by cell density in DAPI
images — we used foreground-median intensity (per-nucleus brightness rather
than whole-image mean) for the high-intensity category and the ratio of
largest to median connected-component area for the clustered/touching
category. The mitotic-figures category was curated by visual inspection, as
whole-image summary statistics cannot reliably detect rare morphological
events at the single-cell level. Automated selections were verified by
visual review of thumbnail contact sheets prior to benchmarking.

## Benchmark Results

Inter-model agreement (Panel B) — quantified as the coefficient of variation (CV) of nuclei counts across the seven single-channel models — varies markedly across imaging categories. Models converge most tightly on high-confluency and flat-field images (median CV ~9% and ~15%, respectively), where nuclei are abundant and signal quality is high. By contrast, the debris/artifacts category yields a median CV of ~42% with individual images exceeding 100%, and elevated spread is also observed for mitotic and defocused fields, reflecting the fundamental ambiguity these conditions introduce. Panel C reveals a 27-fold range in GPU inference speed: InstanSeg is the fastest at 0.20 s per image, followed by StarDist (1.23 s), the two DeepCell models (2.1–2.4 s), and finally the Cellpose variants (3.5–5.4 s). TensorFlow-based models cluster in the intermediate range, while PyTorch-based models span the entire spectrum from fastest to slowest, demonstrating that framework alone does not determine throughput. Panels E and F reveal consistent model-level tendencies that hold across all categories and image ranks. StarDist returns nuclei counts at or near the high end of the range across virtually every category subplot, while InstanSeg returns counts consistently at the lower end, with correspondingly sparser mask coverage visible in Panel F. DeepCell Nuclear displays a qualitatively different segmentation style: its predicted labels tend to cover larger spatial extents than those of other models, resulting in broader nuclear boundaries that in some cases encompass adjacent nuclei within a single label, as is visible in the mask panel. The consensus heatmap row in Panel F reinforces these observations — well-focused, high-signal categories (high intensity, flat-field, clustered) show predominantly warm colors indicating broad model agreement on foreground regions, while the debris and defocused categories are dominated by cool colors and patchy patterns, reflecting that each model responds to these challenging conditions differently.

## Interactive Benchmark Viewer

To facilitate visual inspection of segmentation results across the benchmark dataset, we developed an interactive browser-based viewer (`viewer.py`). The viewer organizes images by their quality category (e.g., low confluency, defocused, high intensity) and displays the full set of field-of-view images for each category in a gallery layout. Users can examine the output of any single model by overlaying its predicted instance labels as semi-transparent filled regions onto the raw DAPI image, with adjustable opacity. Alternatively, a consensus heatmap mode aggregates predictions from all models simultaneously, coloring each pixel according to how many models agreed on it being foreground, which provides an immediate visual summary of where models converge or diverge. A dedicated single-image view offers a larger, higher-resolution rendering of any selected field for closer examination. The viewer is intended as a qualitative complement to the quantitative metrics, enabling rapid identification of failure modes, edge cases, and category-specific differences in model behavior.
