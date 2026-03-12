# Model Selection Rationale for AggreQuant

**Date:** 2026-03-04
**Based on:** Nuclei segmentation benchmark (13 configs, 100 images, 9 difficulty categories)
and cell segmentation benchmark (10 configs, 90 images, same categories).

---

## Nuclei Segmentation

### Selected: StarDist `2D_versatile_fluo` (already implemented)

StarDist was designed specifically for star-convex nuclei detection in fluorescence microscopy.
It handles the clustered/touching nuclei category by design and tends to count among the highest
across all difficulty categories — consistent with its purpose-built architecture. The background
normalization preprocessing added to the AggreQuant implementation further improves robustness
across the illumination variation categories (flat-field, high/low intensity).

### Rejected alternatives

| Model | Reason |
|---|---|
| **InstanSeg** | 17× faster than Cellpose, but **systematically under-counts** in low intensity and defocused categories — exactly the challenging HCS cases this pipeline encounters. Wrong failure mode for a batch pipeline tool. |
| **DeepCell NuclearSegmentation** | Requires TensorFlow 2.8 (separate environment), a user account and access token, and has documented API instability (breaking change in v0.12.10). No clear performance advantage over StarDist. |
| **DeepCell Mesmer (nuclear compartment)** | Same TF 2.8 / access token issues. Tends to over-count relative to consensus; no advantage for the pipeline. |
| **Cellpose `nuclei`** | No extra dependency cost (Cellpose already required for cells), but StarDist is more purpose-built. Could be added as a user-selectable `model_type` alternative if needed, but not as the default. |

---

## Cell Segmentation

### Selected: Cellpose `cyto3` (primary) and `cyto2` (alternative, already supported via `model_type`)

Cellpose cyto2 and cyto3 are the **only benchmarked models explicitly trained and validated on
single-channel cytoplasm fluorescence from cell culture**. cyto3 additionally incorporates
HCS-style training data (Pachitariu & Stringer 2022, *Nature Methods*), making it the strongest
default for FarRed channel HCS images. Both operate natively in single-channel mode without
requiring a nuclear stain.

### Rejected alternatives

| Model | Reason |
|---|---|
| **DeepCell Mesmer** | Trained on TissueNet: FFPE/fresh tissue sections with paired DAPI + membrane antibody channels (E-cadherin, pan-CK, Na/K-ATPase). Cell culture segmentation is explicitly out of scope per the developers (Greenwald et al. 2022, *Nat Biotechnol*). With a zeroed nuclear channel, it detects very few cells in all difficulty categories. Only competitive when given a DAPI channel, but that is an out-of-distribution workaround. |
| **InstanSeg (fluorescence)** | Trained on CPDMI_2023: 8–32 channel multiplexed tissue images, all including DAPI. With a single cytoplasm channel only, the nucleus-detection component has nothing to detect and cell boundary estimation becomes unreliable. Developers explicitly state: *"If you only have one channel, you should probably only choose nuclei as the output."* Sparse detections in single-channel configuration; only competitive with +nuc. |
| **CellSAM** | Pre-release (GitHub-only, no PyPI). Heavy ViT-B backbone. Undersegments in single-channel relative to Cellpose, especially in dense categories (roughly half the cell count). Improves with +nuc but still not consistently competitive. Worth revisiting when a stable release is available. |

### Key finding: nuclear channel dependency

The +nuc benchmark configurations reveal a clear divide. Adding the DAPI channel:

- **Cellpose cyto2/cyto3:** modest improvement — consistent with training on single-channel
  cytoplasm images. The nuclear hint is supplementary, not required.
- **Mesmer, InstanSeg, CellSAM:** dramatic improvement — consistent with architectures that
  fundamentally depend on a nuclear stain to seed cell detection. Their single-channel
  performance is poor because they are operating outside their training distribution.

This confirms that for workflows where a DAPI channel may not be available (or to keep the
pipeline modular), only the Cellpose family is architecturally appropriate.

---

## Summary

| Task | Model | Justification |
|---|---|---|
| Nuclei | StarDist `2D_versatile_fluo` | Purpose-built for fluorescence nuclei; handles touching/clustered nuclei; benchmark-validated |
| Cells | Cellpose `cyto3` | Only model trained on single-channel HCS cytoplasm fluorescence; natively single-channel |
| Cells (alt) | Cellpose `cyto2` | Same training domain; useful legacy/comparison option; no extra dependency |

The current implementation is the benchmark-validated choice. The benchmarks exist to justify
this selection, not to find a replacement.

---

## Future Candidate

**Cellpose 3** (Stringer & Pachitariu 2025, *Nature Methods*) adds learned image restoration
(denoising, deblurring) before segmentation using the same cyto2/cyto3 weights. Same training
domain, drop-in API replacement. Most likely to help on the defocused and low-intensity
difficulty categories. Not yet benchmarked — worth evaluating when stable.
