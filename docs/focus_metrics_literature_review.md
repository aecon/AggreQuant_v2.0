# Focus/Blur Detection: Literature Review and Comparison

**Date:** 2026-02-06
**Context:** Evaluation of AggreQuant's focus.py implementation against state-of-the-art methods

---

## Current Implementation Summary (focus.py)

| Metric | Category | Notes |
|--------|----------|-------|
| Variance of Laplacian | Spatial/Gradient | Industry standard, well-chosen |
| Laplacian Energy | Spatial/Gradient | Redundant with VoL |
| Sobel | Spatial/Gradient | Good complementary metric |
| Brenner | Spatial/Gradient | Fast, effective |
| FocusScore (var/mean) | Statistics | Basic normalized variance |

---

## State-of-the-Art Methods from Literature

### 1. Google/Verily Deep Learning Model (Yang et al., 2018) - Highly Cited

- **Paper:** https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4
- **GitHub:** https://github.com/google/microscopeimagequality

**Architecture:**
- Conv(32, 5×5) → MaxPool → Conv(64, 5×5) → MaxPool → FC(1024) → Dropout → 11 classes
- Operates on 84×84 patches
- Outputs probability distribution over 11 defocus levels

**Performance:**
- Binary in/out-of-focus: **F-score 0.89** (vs 0.84 for PLLS)
- 95% accuracy within ±1 defocus level
- Generalizes across stains (Hoechst → Phalloidin)

**Limitation:** Requires TensorFlow 1.x, Python 3.7 - outdated dependencies

---

### 2. Power Log-Log Slope (PLLS) - Previous State-of-the-Art

- **Reference:** https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/

**Algorithm:**
```python
def power_log_log_slope(image):
    # Radial power spectrum
    f = np.fft.fft2(image)
    power = np.abs(np.fft.fftshift(f))**2
    radial_power = azimuthal_average(power)

    # Log-log linear fit
    freqs = np.arange(1, len(radial_power))
    slope, _ = np.polyfit(np.log(freqs), np.log(radial_power[1:]), 1)
    return slope  # More negative = more blur
```

**Why it works:** Blur removes high frequencies; defocused images have steeper negative slopes.

**Used by:** CellProfiler's `MeasureImageQuality` module

---

### 3. CellProfiler Suite - Industry Standard

- **Documentation:** https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html
- **Code:** https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/measureimagequality.py

**Metrics implemented:**

| Metric | AggreQuant Implementation | Notes |
|--------|:------------------------:|-------|
| FocusScore (normalized variance) | ✅ Similar | `var / mean` |
| LocalFocusScore | ❌ Missing | Tile-based, more robust |
| PowerLogLogSlope | ❌ Missing | Frequency domain, previous SOTA |
| Correlation (Haralick H3) | ❌ Missing | Texture-based |
| Saturation metrics | ❌ Missing | Over/under-exposure |

---

### 4. Recent Quantitative Evaluation (2025)

- **Paper:** https://pmc.ncbi.nlm.nih.gov/articles/PMC12115465/

**Best performers by criterion:**

| Criterion | Best Operator |
|-----------|---------------|
| Sensitivity | FDWT (Wavelet) |
| Noise robustness | Variance |
| Balanced | **Tenengrad** (similar to Sobel) |

**Key finding:** "Tenengrad exhibits the most superior noise robustness among spatial operators"

---

## Gap Analysis: What AggreQuant is Missing

### Critical Gaps

| Feature | Priority | Rationale |
|---------|----------|-----------|
| **PowerLogLogSlope (PLLS)** | High | Detects global defocus (frequency-domain) |
| **Deep learning classifier** | Medium | Higher accuracy but adds dependency |
| **LocalFocusScore** | Medium | Already have patch-based, but tile averaging helps |

### Minor Gaps

| Feature | Priority | Rationale |
|---------|----------|-----------|
| Tenengrad | Low | Very similar to Sobel metric |
| Haralick correlation | Low | Texture-based, complementary |
| Wavelet-based (FDWT) | Low | Best sensitivity but complex |

---

## Recommendations

### 1. Add PLLS (High Priority)

This directly addresses globally defocused images:

```python
def power_log_log_slope(image: np.ndarray) -> float:
    """
    Compute Power Log-Log Slope - detects global defocus.

    More negative values indicate more blur (high frequencies lost).
    Reference: Bray et al., J Biomol Screen, 2012
    """
    from scipy import ndimage

    # Compute 2D FFT
    f = np.fft.fft2(image.astype(np.float64))
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift) ** 2

    # Radial average (azimuthal integration)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)

    max_r = min(cx, cy)
    radial_sum = ndimage.sum(magnitude, r, index=np.arange(1, max_r))
    radial_count = ndimage.sum(np.ones_like(magnitude), r, index=np.arange(1, max_r))
    radial_mean = radial_sum / (radial_count + 1e-10)

    # Log-log linear fit
    freqs = np.arange(1, len(radial_mean) + 1)
    valid = radial_mean > 0
    if valid.sum() < 10:
        return 0.0

    log_f = np.log(freqs[valid])
    log_p = np.log(radial_mean[valid])
    slope, _ = np.polyfit(log_f, log_p, 1)

    return float(slope)  # Typically -1 to -4; more negative = blurrier
```

### 2. Add Global Quality Score (High Priority)

Complements patch-based analysis:

```python
def image_focus_quality(image: np.ndarray) -> dict:
    """Compute global image focus quality metrics."""
    plls = power_log_log_slope(image)

    # Global variance of Laplacian
    img_norm = _normalize_to_8bit(image)
    global_vol = variance_of_laplacian(img_norm)

    return {
        "power_log_log_slope": plls,
        "global_variance_laplacian": global_vol,
        "is_globally_defocused": plls < -3.0,  # Threshold needs calibration
    }
```

### 3. Consider Pre-trained Deep Learning (Medium Priority)

For highest accuracy, integrate Google's model or train your own:

```python
# Option A: Use Google's pre-trained model (if dependencies work)
# pip install microscopeimagequality  # TF1.x required

# Option B: Train simple CNN on your data
# - Synthetically defocus known good images
# - Train small network similar to Yang et al.
# - More portable than TF1.x dependency
```

---

## Summary Table

| Method | AggreQuant | CellProfiler | Google DL | Literature Best |
|--------|:---------:|:------------:|:---------:|:---------------:|
| Variance of Laplacian | ✅ | ❌ | ❌ | Good baseline |
| Normalized Variance | ✅ | ✅ | ❌ | Good for brightfield |
| Sobel/Tenengrad | ✅ | ❌ | ❌ | Noise robust |
| Brenner | ✅ | ❌ | ❌ | Fast |
| **PLLS** | ❌ | ✅ | ❌ | **Previous SOTA** |
| **Deep Learning** | ❌ | ❌ | ✅ | **Current SOTA** |
| Haralick Correlation | ❌ | ✅ | ❌ | Texture-based |
| Wavelet (FDWT) | ❌ | ❌ | ❌ | Best sensitivity |

---

## Bottom Line

AggreQuant's implementation covers the **essential spatial-domain metrics** well. The main gaps are:

1. **PLLS** - Would catch globally defocused images
2. **Deep learning** - 5-6% F-score improvement over PLLS

For a quick win, add PLLS. For production-grade quality control, consider the deep learning approach (potentially retraining with modern PyTorch rather than the outdated TF1.x codebase).

---

## Sources

- [Yang et al., 2018 - Assessing microscope image focus quality with deep learning](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/s12859-018-2087-4)
- [Google Microscope Image Quality GitHub](https://github.com/google/microscopeimagequality)
- [Bray et al., 2012 - Workflow and metrics for image quality control](https://pmc.ncbi.nlm.nih.gov/articles/PMC3593271/)
- [CellProfiler MeasureImageQuality Documentation](https://cellprofiler-manual.s3.amazonaws.com/CPmanual/MeasureImageQuality.html)
- [CellProfiler measureimagequality.py Source](https://github.com/CellProfiler/CellProfiler/blob/master/cellprofiler/modules/measureimagequality.py)
- [2025 Quantitative Evaluation of Focus Measure Operators](https://pmc.ncbi.nlm.nih.gov/articles/PMC12115465/)
- [OpenCV Autofocus Comparative Study](https://opencv.org/blog/autofocus-using-opencv-a-comparative-study-of-focus-measures-for-sharpness-assessment/)
- [Sun et al., 2004 - Autofocusing Algorithm Selection in Computer Microscopy](https://amnl.mie.utoronto.ca/data/J7.pdf)
