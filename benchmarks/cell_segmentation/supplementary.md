# Draft Text: Cell Segmentation Benchmark

## Model Scope

Mesmer (DeepCell) was trained on TissueNet (Greenwald et al. 2022, *Nature Biotechnology*), a dataset
drawn from multiplexed tissue imaging modalities (MIBI, CyCIF, CODEX, mIF). Its two-channel design
expects a nuclear channel (DNA/DAPI) paired with a membrane-targeted marker (E-cadherin, pan-CK,
Na/K-ATPase) — not a diffuse cytoplasmic fill. InstanSeg (fluorescence; Goldsborough et al. 2024)
was trained on a broader curated collection of fluorescence microscopy images and supports variable
input channels, but was not specifically validated on single-channel FarRed cytoplasm HCS images.
**Mesmer and InstanSeg are included for completeness and cross-benchmark comparison, but they are not
operating in their validated domain. The Cellpose results are the primary comparison.**
Cellpose cyto2 and cyto3 were explicitly designed and trained for cytoplasm-based cell segmentation
in fluorescence microscopy across a wide range of cell types and imaging conditions; cyto3 additionally
incorporates HCS-style training data, making both the most defensible choice for FarRed channel data.
