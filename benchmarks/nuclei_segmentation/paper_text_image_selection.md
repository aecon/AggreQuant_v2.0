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
