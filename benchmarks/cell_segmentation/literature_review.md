# Literature Review: Cell Segmentation Models for Fluorescence Microscopy

**Date:** 2026-02-25
**Context:** Cell segmentation benchmark using single-channel FarRed (Cy5) cytoplasm
fluorescence from HCS cell culture images (no DAPI). This review documents the training
data, validated use cases, and known limitations of Mesmer and InstanSeg, and assesses
whether they are appropriate for this imaging context.

---

## 1. Mesmer (DeepCell)

### Primary Paper

**Greenwald, N.F., Miller, G., Moen, E., et al. (2022).** Whole-cell segmentation of tissue
images with human-level performance using large-scale data annotation and deep learning.
*Nature Biotechnology*, 40, 555–565.
DOI: [10.1038/s41587-021-01094-0](https://doi.org/10.1038/s41587-021-01094-0)
PubMed: [34795433](https://pubmed.ncbi.nlm.nih.gov/34795433/)

**What this paper tells us:**

Mesmer is trained on **TissueNet**, a dataset of >1 million manually labeled cells from
FFPE and fresh tissue sections. Key facts:

- **Imaging modalities:** Six platforms — MIBI-TOF, CODEX, CyCIF, IMC (Imaging Mass
  Cytometry), Vectra multiplexed immunofluorescence, MxIF. All are multiplexed antibody
  imaging platforms used on tissue sections.
- **Tissue types:** Nine organs (breast, colon, lung, pancreas, skin, immune tissue, and
  others). Human, mouse, and macaque. Normal tissue and tumor resections.
- **Nuclear channel markers:** DAPI or Histone H3 (HH3).
- **"Cytoplasm"/membrane channel markers:** E-cadherin, CD45, Pan-Keratin, CD3, CD14,
  CD56, HLA-G, vimentin — all antibody-based membrane markers, not diffuse cytoplasmic
  dyes.
- **Architecture:** Hard two-channel requirement (nuclear + membrane). There is no
  pretrained single-channel cytoplasm mode.
- **Explicit scope limitation:** The paper states that segmentation of cell culture images
  is out of scope — *"analyses of hematoxylin and eosin images and images of cells in cell
  culture have been achieved by prior work... making these functionalities available through
  DeepCell will be the focus of future work."*
- **Annotation effort:** >4,000 person-hours. ~2,600 training images (512×512 px each).
  Validated against five expert annotators and four board-certified pathologists
  (human-level F1 = 0.82 vs. prior methods 0.41–0.63).

**Relevance to this benchmark:** Mesmer was not trained on cytoplasm-only fluorescence or
HCS cell culture data. Running it with a zeroed nuclear channel (as done here for the
single-channel configs) is an undocumented, unvalidated workaround. The "membrane" channel
it expects is an antibody marker; a diffuse FarRed cytoplasmic dye is a different imaging
regime entirely.

---

### Notable Use Cases of Mesmer

All documented uses of Mesmer in the published literature are on tissue imaging with paired
nuclear + membrane channels. None involve single-channel cytoplasm fluorescence or cell culture.

---

**Risom, T., et al. (2022).** Transition to invasive breast cancer is associated with
progressive changes in the structure and composition of tumor stroma.
*Cell*, 185(2), 299–310.e18.
DOI: [10.1016/j.cell.2021.12.023](https://doi.org/10.1016/j.cell.2021.12.023)
PubMed: [35063072](https://pubmed.ncbi.nlm.nih.gov/35063072/)

Used Mesmer to segment cells in MIBI-TOF images of 79 breast tissue resections (DCIS and
invasive breast cancer) with a 37-plex antibody panel. Nuclear channel: HH3 +
phosphorous (P). Membrane channel: combined signal of E-cadherin, Pan-Keratin, CD45, CD44,
and GLUT1. Demonstrates standard, intended use of Mesmer on tissue with a full multi-marker
panel.

---

**Greenbaum, S., et al. (2023).** A spatially resolved timeline of the human
maternal–fetal interface.
*Nature*, 619(7970), 595–605.
DOI: [10.1038/s41586-023-06298-9](https://doi.org/10.1038/s41586-023-06298-9)
PMC: [PMC10356615](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356615/)

Used Mesmer retrained on 93,000 manually annotated cells from the specific cohort (first-
trimester human decidua) for cell segmentation in MIBI-TOF images with a 37-plex panel.
Nuclear channel: Histone H3. Membrane channels: VIM, HLA-G, CD3, CD14, CD56. Illustrates
that Mesmer may need cohort-specific fine-tuning even for tissue data — further reducing
confidence in out-of-domain application.

---

**Amitay, Y., et al. (2023).** CellSighter: a neural network to classify cells in highly
multiplexed images.
*Nature Communications*, 14, 4302.
DOI: [10.1038/s41467-023-40066-7](https://doi.org/10.1038/s41467-023-40066-7)

Used Mesmer to re-segment cells from the CODEX dataset of 35 colorectal cancer patients
(56-protein panel, FFPE tissue). CellSighter was then trained on the 85,179 resulting cell
masks. Relevant because it shows Mesmer being used as a ground-truth generator for
downstream tools — confirming its role as a tissue-imaging reference standard, not a
general-purpose cytoplasm segmentor.

---

**Tong, A., et al. (2024).** Deep cell phenotyping and spatial analysis of multiplexed
imaging with TRACERx-PHLEX.
*Nature Communications*, 15, 5344.
DOI: [10.1038/s41467-024-48870-5](https://doi.org/10.1038/s41467-024-48870-5)

The PHLEX pipeline (deep-imcyto) was validated and benchmarked against Mesmer on IMC images
of 236 TMA cores from 83 NSCLC patients. Mesmer served as the state-of-the-art reference.
Imaging: FFPE lung cancer, paired nuclear + membrane IMC channels. Confirms Mesmer's role
as the dominant reference method for tissue multiplexed imaging benchmarks.

---

**Hollandi, R., et al. (2025).** Quantitative benchmarking of nuclear segmentation algorithms
in multiplexed immunofluorescence imaging for translational studies.
*Communications Biology*.
DOI: [10.1038/s42003-025-08184-8](https://doi.org/10.1038/s42003-025-08184-8)

Benchmarked multiple segmentation methods on multiplexed immunofluorescence tissue images.
Mesmer achieved the highest nuclear segmentation accuracy (F1 = 0.67 at IoU = 0.5) and was
recommended as the top candidate for multiplexed immunofluorescence workflows. All tested
images were tissue sections with paired nuclear/membrane channels. Further confirms Mesmer's
strength specifically in tissue multiplexed imaging.

---

## 2. InstanSeg (Fluorescence Variant)

### Primary Papers

**Goldsborough, T., Philps, B., O'Callaghan, A., et al. (2024).** InstanSeg: an
embedding-based instance segmentation algorithm optimized for accurate, efficient and
portable cell segmentation.
*arXiv*: 2408.15954.
URL: [https://arxiv.org/abs/2408.15954](https://arxiv.org/abs/2408.15954)

**What this paper tells us:**

The core InstanSeg algorithm — trained on six public nucleus segmentation datasets
(CoNSeP, TNBC_2018, MoNuSeg, LyNSeC, NuInsSeg, IHC TMA). All are H&E brightfield
histopathology datasets. This paper defines the embedding-based instance segmentation
backbone. No fluorescence or cytoplasm data in this version.

---

**Goldsborough, T., O'Callaghan, A., Inglis, F., et al. (2024).** A novel channel
invariant architecture for the segmentation of cells and nuclei in multiplexed images
using InstanSeg.
*bioRxiv*, 2024.09.04.611150.
DOI: [10.1101/2024.09.04.611150](https://doi.org/10.1101/2024.09.04.611150)

**What this paper tells us:**

This paper introduces **ChannelNet** — a module that compresses any number of input
channels (in any order) into a fixed three-channel representation (nuclear, cytoplasmic,
membranous), which is then passed to the InstanSeg backbone. This is the architecture
behind the `instanseg_fluorescence_nuclei_and_cells` pretrained model.

- **Training data:** CPDMI_2023 (see below) — FFPE tissue sections with 8–32 antibody
  channels per image, always including DAPI. Three platforms: Vectra, Zeiss InSituPlex,
  CODEX. No HCS, no cell culture, no single-channel data.
- **Validation:** Held-out CODEX test set (28–32 channel images). Whole-cell segmentation
  accuracy increases monotonically with number of input channels, confirming the model's
  dependence on multi-channel input.
- **Single-channel limitation (explicitly stated by developers on image.sc):** *"If you
  only have one channel, you should probably only choose nuclei as the output."* With a
  cytoplasm-only channel and no nuclear signal, there is nothing for the nucleus-detection
  component to detect, and cell boundary detection becomes unreliable.
- **Comparison to Mesmer:** InstanSeg + ChannelNet outperforms Mesmer on the full CPDMI
  CODEX test set (28–32 channels). Both are tissue-imaging tools.

---

### Training Dataset Paper

**Aleynick, N., Li, Y., Xie, Y., et al. (2023).** CPDMI: A Cross-Platform Dataset of
Multiplex fluorescent cellular object image annotations.
*Scientific Data*, 10, 193.
DOI: [10.1038/s41597-023-02108-z](https://doi.org/10.1038/s41597-023-02108-z)
PMC: [PMC10082189](https://pmc.ncbi.nlm.nih.gov/articles/PMC10082189/)

**What this paper tells us:**

CPDMI_2023 is the primary training dataset for `instanseg_fluorescence`. Key facts:

- **Scale:** 105,774 cellular object annotations (82,058 whole-cell + 23,716 nuclei-only)
  across 170 annotated image sets.
- **Platforms:** Akoya Vectra 3.0 (sequential IF); Ultivue InSituPlex + Zeiss Axioscan
  (sequential IF); Akoya CODEX (cyclical IF).
- **Tissue:** FFPE clinical specimens — lung adenocarcinoma, pancreatic ductal
  adenocarcinoma, breast cancer, ovarian cancer, melanoma, lymphoma, normal lymph node
  and tonsil.
- **Channels:** >40 antibody markers (CD3, CD4, CD8, CD20, PD-1, PD-L1, Ki67, CD68,
  PanCK, FoxP3, and others) + DAPI nuclear counterstain in all images. Images have 8–32
  channels each.
- **No HCS, no cell culture, no single-channel data.**

This paper confirms that InstanSeg's fluorescence model has never seen cytoplasm-only
or HCS cell culture images during training.

---

### Notable Use Cases of InstanSeg

InstanSeg is a recent tool (preprints September 2024) with limited peer-reviewed downstream
citations as of February 2026.

---

**Bankhead, P., et al. (2017).** QuPath: Open source software for digital pathology image
analysis.
*Scientific Reports*, 7, 16878.
DOI: [10.1038/s41598-017-17204-5](https://doi.org/10.1038/s41598-017-17204-5)

Not an InstanSeg paper per se, but relevant because InstanSeg is integrated as the default
segmentation backend in **QuPath 0.6**. QuPath itself is very widely used in digital
pathology (>10,000 citations). The adoption of InstanSeg into QuPath means its user base
is primarily digital pathologists working with tissue sections — not cell culture HCS.

---

## 3. Reference: Cellpose (the appropriate tool for this use case)

**Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021).** Cellpose: a generalist
algorithm for cellular segmentation.
*Nature Methods*, 18, 100–106.
DOI: [10.1038/s41592-020-01018-x](https://doi.org/10.1038/s41592-020-01018-x)

**What this paper tells us:**

Cellpose `cyto` and `cyto2` were explicitly trained to segment cells from **cytoplasm
fluorescence images without a nuclear channel**. Training data includes a wide range of
fluorescence cell culture images across cell types and imaging conditions. The paper
demonstrates segmentation of single-channel cytoplasm images (phase contrast, DIC, and
fluorescent cytoplasmic dyes).

---

**Pachitariu, M., & Stringer, C. (2022).** Cellpose 2.0: how to train your own model.
*Nature Methods*, 19, 1523–1525.
DOI: [10.1038/s41592-022-01663-4](https://doi.org/10.1038/s41592-022-01663-4)

Introduces `cyto3` and human-in-the-loop retraining. cyto3 extends the training data
further and specifically incorporates HCS-style fluorescence images, making it the most
directly applicable model to this benchmark's data.

---

## 4. Additional Pre-trained Models (Not Yet Benchmarked)

The following models were identified as potentially suitable for cell segmentation in
fluorescently labeled 2D images. They are not yet included in this benchmark but represent
the current landscape of available pre-trained tools.

---

### 4.1 Cellpose 3 (2024–2025)

**Stringer, C., & Pachitariu, M. (2025).** Cellpose3: one-click image restoration for
improved cellular segmentation.
*Nature Methods*.
DOI: [10.1038/s41592-025-02595-5](https://doi.org/10.1038/s41592-025-02595-5)

Extends Cellpose with built-in image restoration (denoising, deblurring, upsampling)
applied before segmentation. Uses the same `cyto2`/`cyto3` segmentation weights but
trains a separate restoration network to produce images that are well-segmented by the
generalist model, rather than optimizing for pixel-level fidelity. Drop-in replacement
via the Cellpose Python API — adds "one-click" buttons in the GUI.

**Relevance:** Directly applicable. Same training domain as cyto2/cyto3 (fluorescence
cell culture). Most useful when images are noisy, blurry, or undersampled — could improve
results on the harder difficulty categories in this benchmark.

---

### 4.2 Omnipose

**Cutler, K.J., Stringer, C., Lo, T.W., et al. (2022).** Omnipose: a high-precision
morphology-independent solution for bacterial cell segmentation.
*Nature Methods*, 19, 1438–1448.
DOI: [10.1038/s41592-022-01639-4](https://doi.org/10.1038/s41592-022-01639-4)
GitHub: [https://github.com/kevinjohncutler/omnipose](https://github.com/kevinjohncutler/omnipose)

Builds on Cellpose with improved distance-field representations that handle non-convex,
elongated, and irregular cell morphologies better than Cellpose's original gradient flows.
Pre-trained models include `bact_fluor` (bacterial fluorescence) and a general `cyto2`
variant compatible with the Cellpose API.

**Relevance:** Potentially useful if cells in our images have irregular or elongated
morphologies. However, the pre-trained fluorescence model was optimized for bacterial
cells, which differ substantially from mammalian cells. Would require testing to assess
out-of-the-box performance on HCS cell culture.

---

### 4.3 Micro-SAM (Segment Anything for Microscopy)

**Archit, A., Nair, S., Khalid, N., et al. (2024).** Segment Anything for Microscopy.
*Nature Methods*.
DOI: [10.1038/s41592-024-02580-4](https://doi.org/10.1038/s41592-024-02580-4)
GitHub: [https://github.com/computational-cell-analytics/micro-sam](https://github.com/computational-cell-analytics/micro-sam)

Fine-tunes Meta's Segment Anything Model (SAM) foundation model specifically for light
and electron microscopy. Supports both interactive (point/box prompts) and automatic
instance segmentation modes. The fine-tuned models improve segmentation quality across
a wide range of microscopy imaging conditions compared to vanilla SAM.

**Relevance:** Foundation-model approach with broad generalization. Could work on
single-channel fluorescence, but automatic mode requires an object-detector front-end
(AMG or AIS). Not specifically trained on cytoplasm-only fluorescence — performance on
this modality would need empirical evaluation. Heavier compute requirements than
Cellpose.

---

### 4.4 CellSAM (2025)

**Israel, U., Marks, M., Dilip, R., et al. (2025).** CellSAM: a foundation model for
cell segmentation.
*Nature Methods*.
DOI: [10.1038/s41592-025-02879-w](https://doi.org/10.1038/s41592-025-02879-w)

Pairs a trained cell-finding object detector ("CellFinder") with SAM to produce fully
automatic instance segmentation. CellFinder generates bounding-box prompts, which are
then passed to SAM for mask generation. Trained on a diverse collection of microscopy
datasets across modalities.

**Relevance:** Designed for fully automatic operation across imaging modalities. Promising
generalist approach, but very recent — limited independent benchmarking available.
Would need to verify performance on single-channel cytoplasm fluorescence specifically.

---

### 4.5 SAMCell (2025)

**SAMCell: Generalized label-free biological cell segmentation with Segment Anything.**
*bioRxiv*, 2025.02.06.636835.
DOI: [10.1101/2025.02.06.636835](https://doi.org/10.1101/2025.02.06.636835)

Another SAM-based approach focused on generalized biological cell segmentation, including
label-free (transmitted light) modalities. Very recent preprint (February 2025), not yet
peer-reviewed.

**Relevance:** Primarily targets label-free imaging (phase contrast, DIC). May generalize
to fluorescence, but this is not its primary design target.

---

### 4.6 MEDIAR (NeurIPS 2022 Challenge Winner)

Winner of the NeurIPS 2022 Cell Segmentation Challenge. Transformer-based architecture
with strong out-of-the-box generalization across diverse cell types and imaging modalities.

**Relevance:** Demonstrated broad generalization in a competition setting, but less
widely adopted than Cellpose or StarDist in practice. Worth considering if available
pre-trained weights generalize to single-channel fluorescence.

---

### 4.7 Models Considered but Not Suitable

| Model | Why not suitable for this benchmark |
|---|---|
| CellViT / HoverNet | Designed for H&E histopathology, not fluorescence microscopy |
| MedSAM | Fine-tuned on radiology images (CT, X-ray, ultrasound), not microscopy |
| CellSeg3D | 3D-specific; our images are 2D |

---

### 4.8 Key Findings from Systematic Benchmarks

**Li, H., et al. (2024).** A systematic evaluation of computational methods for cell
segmentation.
*Briefings in Bioinformatics*, 25(5), bbae407.
DOI: [10.1093/bib/bbae407](https://doi.org/10.1093/bib/bbae407)

Key takeaways relevant to this benchmark:

- No single model dominates across all tissue/cell types — the best model varies by
  imaging context.
- Dual-channel input (cytoplasm + nucleus) dramatically improves whole-cell segmentation,
  with accuracy gains up to 3× for Cellpose (F1 from 0.17 to 0.7).
- Cellpose and StarDist achieved equivalent performance to Mesmer when trained on
  TissueNet, while FeatureNet, RetinaMask, and Ilastik did not.
- For single-channel workflows, Cellpose remains the strongest general-purpose option.

---

## 5. Summary and Verdict

| Model | Trained on cytoplasm fluorescence HCS? | Trained on cell culture? | Single-channel cytoplasm supported? | Status in benchmark |
|---|---|---|---|---|
| Mesmer | No | No | No (undocumented workaround only) | Included |
| InstanSeg fluorescence | No | No | No (developers advise against it) | Included |
| Cellpose cyto2/cyto3 | Yes | Yes | Yes (primary use case) | Included |
| Cellpose 3 | Yes | Yes | Yes (+ image restoration) | Not yet tested |
| Omnipose | Bacterial only | No | Yes (but bacterial focus) | Not yet tested |
| Micro-SAM | Broad microscopy | Mixed | Untested on cytoplasm-only | Not yet tested |
| CellSAM | Broad microscopy | Mixed | Untested on cytoplasm-only | Not yet tested |

**Mesmer** is a tissue-imaging model requiring paired nuclear + membrane antibody channels.
Using it on a single FarRed cytoplasm channel from cell culture is outside its training
distribution, architecturally non-standard (zeroed nuclear channel), and explicitly
flagged as out of scope by the developers.

**InstanSeg (fluorescence)** has broader training data than Mesmer (multi-channel
multiplexed tissue imaging vs. MIBI-only tissue imaging), but still depends on multi-channel
multiplexed input including a nuclear stain. With only a cytoplasm channel and no nuclear
signal, the model lacks the input it was designed for. Developers explicitly advise against
using the whole-cell output mode with a single channel.

**Both models are included in this benchmark for completeness and cross-benchmark
comparison, but they are not operating in their validated domain. The Cellpose results
are the primary comparison.**

Cellpose cyto2 and cyto3 are the only models in this benchmark with documented training
and validation on single-channel cytoplasm fluorescence images from cell culture.

**For future benchmarking,** Cellpose 3 is the most promising addition — it shares the
same training domain and adds image restoration that could help on degraded images.
The SAM-based models (Micro-SAM, CellSAM) are interesting foundation-model approaches
but require empirical validation on single-channel cytoplasm fluorescence before drawing
conclusions.
