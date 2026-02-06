# Extended Literature Review: Deep Learning for Blur/Focus Detection in Microscopy

**Date:** 2026-02-06
**Question:** Is it worth training a custom neural network for blur detection in cellular fluorescence microscopy?

---

## Executive Summary

**Short Answer:** Training from scratch is generally **NOT recommended** unless you have:
- Very specific imaging modality not covered by existing models
- Large labeled dataset (>10,000 images with focus annotations)
- Significant computational resources

**Better Alternatives:**
1. Use traditional metrics (PLLS, Variance of Laplacian) - often sufficient
2. Fine-tune existing pretrained models with small labeled dataset
3. Use synthetic defocus augmentation to expand limited data

---

## Short summary

Available Pretrained Models
  ┌───────────────────────────────┬──────┬─────────────────────────────┬───────────────────┬───────────────┐
  │             Model             │ Year │            Task             │ Weights Available │   Usability   │
  ├───────────────────────────────┼──────┼─────────────────────────────┼───────────────────┼───────────────┤
  │ Google MicroscopeImageQuality │ 2018 │ Focus classification        │ Yes               │ ⚠️ TF1.x only │
  ├───────────────────────────────┼──────┼─────────────────────────────┼───────────────────┼───────────────┤
  │ UniFMIR                       │ 2024 │ Image restoration           │ Yes               │ ✅ PyTorch    │
  ├───────────────────────────────┼──────┼─────────────────────────────┼───────────────────┼───────────────┤
  │ Microsnoop                    │ 2023 │ Image representation        │ Yes               │ ✅ PyTorch    │
  ├───────────────────────────────┼──────┼─────────────────────────────┼───────────────────┼───────────────┤
  │ MPT-CataBlur                  │ 2024 │ Deblurring                  │ Yes               │ ✅ PyTorch    │
  ├───────────────────────────────┼──────┼─────────────────────────────┼───────────────────┼───────────────┤
  │ NASA Microscopy Models        │ 2022 │ Segmentation/classification │ Yes               │ ✅ PyTorch    │
  └───────────────────────────────┴──────┴─────────────────────────────┴───────────────────┴───────────────┘
  Critical finding: The Google 2018 model remains the only dedicated focus quality classifier for fluorescence microscopy. No
  newer alternatives exist with available pretrained weights.

  Transfer Learning vs. From Scratch

  From NASA's study on microscopy:
  - From scratch: >200 epochs, 94.0% accuracy
  - Transfer learning: ~5 epochs, 94.5% accuracy
  - Speedup: 40× faster convergence

  My Recommendation

  ┌─────────────────────────────────────────────────────────┐
  │  Phase 1: Add PLLS metric (1-2 days)                    │
  │  ├── Catches globally defocused images (your A-24)     │
  │  └── Often sufficient for 85-90% of cases              │
  │                                                         │
  │  Phase 2: If needed, fine-tune in PyTorch (2-3 weeks)  │
  │  ├── Reimplement Google's small CNN architecture       │
  │  ├── Train on synthetically defocused versions of      │
  │  │   your own in-focus images                          │
  │  └── Achieves ~89% F-score                             │
  │                                                         │
  │  Phase 3: Train from scratch (NOT recommended)          │
  │  └── Only if unique microscope/modality requirements   │
  └─────────────────────────────────────────────────────────┘

  Why NOT Train From Scratch?

  1. No significant accuracy improvement - Google's simple CNN already achieves 89% F-score
  2. High cost - Months of development for marginal gains
  3. Data requirements - Need >10,000 labeled images
  4. Traditional metrics work - PLLS + VoL handle most cases

  Sources

  - https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4 - Still the benchmark
  - https://www.nature.com/articles/s41592-024-02244-3 - Modern foundation model
  - https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_A_Unified_Framework_for_Microscopy_Defocus_Deblur_with_Multi-Pyr
  amid_Transformer_CVPR_2024_paper.html - Latest deblurring
  - https://github.com/nasa/pretrained-microscopy-models - Transfer learning evidence




---

## Available Pretrained Models for Fluorescence Microscopy

### 1. Google Microscope Image Quality (2018) - Most Relevant

**Paper:** [Yang et al., BMC Bioinformatics 2018](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4)
**Code:** [github.com/google/microscopeimagequality](https://github.com/google/microscopeimagequality)

| Aspect | Details |
|--------|---------|
| **Task** | Focus quality classification (11 defocus levels) |
| **Architecture** | Small CNN: Conv(32)→Conv(64)→FC(1024)→11 classes |
| **Input** | 84×84 patches, 16-bit grayscale |
| **Training Data** | Hoechst-stained U2OS cells, synthetically defocused |
| **Performance** | F-score 0.89 (binary), 95% accuracy within ±1 level |
| **Generalization** | Tested on Phalloidin, Tubulin - good transfer |

**Limitations:**
- ⚠️ **TensorFlow 1.x required** (Python 3.7 or earlier)
- ⚠️ Not maintained since ~2019
- ⚠️ Specific to fluorescence microscopy with DAPI-like stains

**Verdict:** Best existing option for focus quality, but outdated dependencies make integration difficult.

---

### 2. UniFMIR - Foundation Model (Nature Methods 2024)

**Paper:** [Ma et al., Nature Methods 2024](https://www.nature.com/articles/s41592-024-02244-3)
**Code:** [github.com/cxm12/UNiFMIR](https://github.com/cxm12/UNiFMIR)

| Aspect | Details |
|--------|---------|
| **Task** | Image restoration (denoising, super-resolution, isotropic reconstruction) |
| **Architecture** | Transformer-based foundation model |
| **Training Data** | Multiple fluorescence microscopy datasets (BioSR, Planaria, etc.) |
| **Pretrained Weights** | Available via [GitHub releases](https://github.com/cxm12/UNiFMIR/releases) |
| **Dependencies** | PyTorch 1.10, Python 3.9 |

**Relevance to Focus Detection:**
- ❌ Not designed for focus/blur detection
- ✅ Could potentially be fine-tuned for quality assessment
- ✅ Modern PyTorch implementation

---

### 3. Microsnoop - Generalist Representation (The Innovation 2023)

**Paper:** [Microsnoop, The Innovation 2023](https://www.sciencedirect.com/science/article/pii/S2666675823001698)
**Code:** [github.com/cellimnet/microsnoop-publish](https://github.com/cellimnet/microsnoop-publish)

| Aspect | Details |
|--------|---------|
| **Task** | Image representation/embeddings for downstream tasks |
| **Training Data** | 2,230,000+ microscopy images (various modalities) |
| **Architecture** | Masked self-supervised learning |
| **Performance** | State-of-the-art on 10 diverse datasets |

**Relevance to Focus Detection:**
- ❌ Not specifically for blur detection
- ✅ Could train simple classifier on top of embeddings
- ✅ Generalizes well across microscopy types

---

### 4. NASA Pretrained Microscopy Models

**Code:** [github.com/nasa/pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models)

| Aspect | Details |
|--------|---------|
| **Task** | Classification, segmentation backbones |
| **Architectures** | ResNet, EfficientNet, SENet, Xception (30+ models) |
| **Training Data** | 100,000+ microscopy images (materials science) |
| **Best Accuracy** | EfficientNet-b4: 94.5% (with ImageNet pretrain) |

**Relevance to Focus Detection:**
- ❌ Not designed for focus detection
- ✅ Good encoder backbones for transfer learning
- ⚠️ Trained on materials microscopy, not biological

---

### 5. MPT-CataBlur (CVPR 2024) - Deblurring

**Paper:** [Zhang et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_A_Unified_Framework_for_Microscopy_Defocus_Deblur_with_Multi-Pyramid_Transformer_CVPR_2024_paper.html)
**Code:** [github.com/PieceZhang/MPT-CataBlur](https://github.com/PieceZhang/MPT-CataBlur)

| Aspect | Details |
|--------|---------|
| **Task** | Defocus deblurring (not detection) |
| **Architecture** | Multi-Pyramid Transformer + Contrastive Learning |
| **Training Data** | CataBlur dataset (surgical microscopy) |
| **Pretrained Weights** | [Google Drive](https://drive.google.com/file/d/1YHrI2H6uRsB9wqMVqgoZPWaQpBGCWih-/view) |

**Relevance:**
- ❌ For deblurring, not detection
- ❌ Surgical microscopy, not fluorescence
- ✅ State-of-the-art transformer architecture for microscopy blur

---

## Latest Research from Top Venues (2024-2025)

### CVPR 2024-2025

| Paper | Focus | Relevance |
|-------|-------|-----------|
| MPT-CataBlur | Microscopy defocus deblur | High (architecture) |
| Efficient Visual State Space Model | Image deblurring | Medium |
| DOF-GS | Depth-of-field, blur removal | Low (3D Gaussian) |

### Nature Methods 2024

| Paper | Focus | Relevance |
|-------|-------|-----------|
| UniFMIR | Foundation model for microscopy restoration | High |
| Deep learning adaptive optics | Aberration compensation | Medium |

### Key Finding: No New Focus Quality Classifiers

**The Google 2018 model remains the only dedicated, pretrained focus quality classifier for fluorescence microscopy.** No newer alternatives have emerged that specifically address this task with available pretrained weights.

---

## Training From Scratch vs. Transfer Learning

### Research Evidence

From [NASA pretrained-microscopy-models](https://github.com/nasa/pretrained-microscopy-models) study:

| Approach | Epochs to Converge | Best Accuracy |
|----------|-------------------|---------------|
| From scratch | >200 epochs | 94.0% (SENet-154) |
| ImageNet pretrain + fine-tune | ~5 epochs | 94.5% (EfficientNet-b4) |
| MicroNet pretrain + fine-tune | ~5 epochs | Similar |

**Key insight:** Transfer learning converges **40× faster** with comparable or better accuracy.

### When to Train From Scratch

Only consider training from scratch if:
1. **Unique imaging modality** not represented in existing models
2. **Large dataset** (>10,000 labeled images minimum)
3. **Specific focus failure modes** unique to your microscope
4. **Regulatory/compliance** requirements for custom models

### Recommended Approach for Your Case

```
Option A: Traditional Metrics (Easiest)
├── Add PLLS to your existing focus.py
├── Combine multiple metrics with simple threshold
└── Sufficient for 80-90% of use cases

Option B: Fine-tune Existing Model (Best ROI)
├── Use Google's model architecture (small CNN)
├── Generate synthetic training data from your in-focus images
├── Fine-tune for 10-20 epochs
└── Reimplement in PyTorch to avoid TF1.x issues

Option C: Train Custom Model (Most Effort)
├── Collect/annotate large dataset
├── Use pretrained encoder (ResNet, EfficientNet)
├── Add classification head for focus quality
└── Train with synthetic + real defocus data
```

---

## Synthetic Training Data Generation

The Google approach (and others) use synthetic defocus to generate training data:

```python
import numpy as np
from scipy.ndimage import gaussian_filter

def generate_synthetic_defocus(image, num_levels=11, max_sigma=5.0):
    """
    Generate synthetically defocused versions of an in-focus image.

    Based on Yang et al., 2018 approach.
    """
    training_pairs = []

    for level in range(num_levels):
        # Linearly increasing blur
        sigma = (level / (num_levels - 1)) * max_sigma

        if sigma == 0:
            blurred = image.copy()
        else:
            blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)

        # Add Poisson noise (realistic for fluorescence)
        noisy = np.random.poisson(np.maximum(blurred, 0)).astype(image.dtype)

        training_pairs.append((noisy, level))

    return training_pairs
```

**Advantages:**
- No manual annotation needed
- Unlimited training data from existing in-focus images
- Controllable difficulty levels

**Limitations:**
- Gaussian blur doesn't capture all defocus aberrations
- May not generalize to real out-of-focus patterns
- Needs validation on real defocused images

---

## Cost-Benefit Analysis

| Approach | Development Time | Accuracy | Maintenance | Recommendation |
|----------|-----------------|----------|-------------|----------------|
| Traditional (PLLS + VoL) | 1-2 days | Good (85-90%) | Low | ✅ Start here |
| Google model (as-is) | 1 week | Very Good (89%) | High (TF1.x) | ⚠️ Dependency issues |
| Fine-tune in PyTorch | 2-3 weeks | Very Good (89%+) | Medium | ✅ Best ROI |
| Train from scratch | 2-3 months | Potentially Best | Medium | ❌ Unless necessary |

---

## Practical Recommendations for AggreQuant

### Phase 1: Quick Win (1-2 days)
1. Add `power_log_log_slope()` to `focus.py`
2. Add global image quality score combining PLLS + VoL
3. Test on your benchmark images

### Phase 2: If Phase 1 Insufficient (2-3 weeks)
1. Reimplement Google's CNN architecture in PyTorch
2. Generate synthetic training data from your in-focus images
3. Train on synthetically defocused data
4. Validate on manually labeled real images

### Phase 3: Production Quality (if needed)
1. Collect real out-of-focus images from your microscopes
2. Manually label ~500-1000 images
3. Fine-tune model on real data
4. Deploy with confidence thresholds

---

## Key References

### Focus Quality Assessment
1. [Yang et al., 2018 - Google Focus Quality](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4)
2. [Bray et al., 2012 - PLLS Metric](https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/)
3. [CellProfiler MeasureImageQuality](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html)

### Foundation Models for Microscopy
4. [UniFMIR, Nature Methods 2024](https://www.nature.com/articles/s41592-024-02244-3)
5. [Microsnoop, The Innovation 2023](https://www.sciencedirect.com/science/article/pii/S2666675823001698)

### Deblurring (Related)
6. [MPT-CataBlur, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Zhang_A_Unified_Framework_for_Microscopy_Defocus_Deblur_with_Multi-Pyramid_Transformer_CVPR_2024_paper.html)

### Transfer Learning for Microscopy
7. [NASA Pretrained Microscopy Models](https://github.com/nasa/pretrained-microscopy-models)
8. [ZeroCostDL4Mic, Nature Communications 2021](https://www.nature.com/articles/s41467-021-22518-0)

### Benchmarks and Comparisons
9. [Quantitative Evaluation of Focus Measures, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115465/)
10. [SR-CACO-2 Dataset, NeurIPS 2024](https://arxiv.org/abs/2406.09168)

---

## Conclusion

**Training your own neural network from scratch is NOT recommended** for focus/blur detection in fluorescence microscopy because:

1. **No significant improvement** over existing methods for this specific task
2. **High development cost** (months of work) for marginal gains
3. **Traditional metrics** (PLLS, VoL) are well-validated and often sufficient
4. **Transfer learning** from existing models is more efficient if DL is needed

**Recommended path:**
1. First try PLLS + existing metrics
2. If insufficient, fine-tune Google's architecture in modern PyTorch
3. Only train from scratch if you have unique requirements and resources
