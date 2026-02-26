# Draft Text: Cell Segmentation Benchmark

## Model Scope

Mesmer (DeepCell) was trained on TissueNet (Greenwald et al. 2022, *Nature Biotechnology*), a dataset
drawn from multiplexed tissue imaging modalities (MIBI, CyCIF, CODEX, mIF). Its two-channel design
expects a nuclear channel (DNA/DAPI) paired with a membrane-targeted marker (E-cadherin, pan-CK,
Na/K-ATPase) — not a diffuse cytoplasmic fill. InstanSeg (fluorescence; Goldsborough et al. 2024)
was trained on a broader curated collection of fluorescence microscopy images and supports variable
input channels, but was not specifically validated on single-channel FarRed cytoplasm HCS images.
CellSAM (Israel, Marks et al. 2025, *Nature Methods*) pairs a CellFinder object detector with
Meta's Segment Anything Model (SAM) for fully automatic instance segmentation across microscopy
modalities; it handles single-channel grayscale input natively but was trained on a diverse collection
that includes multi-channel data.
**Mesmer, InstanSeg, and CellSAM are included for completeness and cross-benchmark comparison, but
they are not operating in their validated domain. The Cellpose results are the primary comparison.**
Cellpose cyto2 and cyto3 were explicitly designed and trained for cytoplasm-based cell segmentation
in fluorescence microscopy across a wide range of cell types and imaging conditions; cyto3 additionally
incorporates HCS-style training data, making both the most defensible choice for FarRed channel data.

## Effect of Nuclear Channel (+nuc Configurations)

Each model was run in two configurations: single-channel (FarRed cytoplasm only) and two-channel
(FarRed + DAPI nuclear hint). The comparison reveals a striking difference in nuclear-channel
dependency across model families.

**Cellpose cyto2 and cyto3** show only modest improvement with the DAPI channel. This is expected:
both models were trained on single-channel cytoplasm fluorescence images and do not require a nuclear
signal to segment cells. The nuclear hint provides supplementary information but is not architecturally
necessary.

**Mesmer** shows a dramatic improvement with +nuc. Single-channel Mesmer (zeroed nuclear input)
detects very few cells across all difficulty categories. With the DAPI channel restored, detection
improves substantially. This is consistent with Mesmer's design: it was trained on TissueNet where
every image contains a paired DAPI or Histone H3 nuclear stain. The nuclear channel is not optional —
it is the primary signal Mesmer uses to seed cell detection. The single-channel configuration (zeroed
nuclear input) is an undocumented workaround that places the model entirely outside its training
distribution. The developers explicitly stated that cell culture segmentation is out of scope
(Greenwald et al. 2022).

**InstanSeg** shows a similarly dramatic improvement with +nuc. The single-channel cytoplasm
configuration yields sparse detections, while the two-channel configuration performs comparably to
Cellpose. This aligns with the training data (CPDMI: 8–32 channel tissue images, all including DAPI)
and the developers' guidance on image.sc: *"If you only have one channel, you should probably only
choose nuclei as the output."* Without a nuclear signal, InstanSeg's nucleus-detection component has
nothing to detect, and cell boundary estimation becomes unreliable.

**CellSAM** also improves substantially with +nuc, though less dramatically than Mesmer or InstanSeg.
Single-channel CellSAM undersegments (detecting roughly half the cells that Cellpose finds in dense
categories), while +nuc CellSAM approaches Cellpose-level counts. CellSAM handles grayscale input
natively, but its CellFinder detector benefits from having two information-rich channels rather than
one.

**Key finding:** The single-channel vs. +nuc comparison confirms that Mesmer, InstanSeg, and CellSAM
fundamentally depend on a nuclear stain for reliable whole-cell segmentation. Only the Cellpose family
is designed and validated for single-channel cytoplasm fluorescence — making it the appropriate
baseline for this imaging modality. The +nuc configurations demonstrate that when given the input they
were designed for, Mesmer and InstanSeg can perform competitively, but this requires a DAPI channel
that may not be available in all experimental designs.
