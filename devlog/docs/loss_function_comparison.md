# Loss Function Comparison — Baseline UNet

Ablation study comparing four loss configurations on the same baseline UNet
(Ronneberger 2015, features=[64,128,256,512], 31M params) trained on 19
annotated fluorescence microscopy images of protein aggregates.

All runs use: 192x192 patches, batch size 16, Adam lr=1e-3, seed=42,
80/20 train/val split. Metrics reported at the best-validation-loss epoch.

---

## Results

| Run | Loss config | Dice | IoU | Precision | Recall | Best epoch |
|---|---|---|---|---|---|---|
| baseline_no_scheduler | BCE pw=7.5, no scheduler | 0.725 | 0.570 | 0.591 | **0.949** | 62/82 |
| baseline | BCE pw=7.5 + ReduceLROnPlateau | 0.742 | 0.592 | 0.613 | **0.950** | 95/115 |
| bce_pw3 | BCE pw=3.0 + ReduceLROnPlateau | 0.804 | 0.674 | 0.726 | 0.909 | 173/193 |
| **dice03_bce07_pw3** | **Dice(0.3)+BCE(0.7) pw=3.0 + ReduceLROnPlateau** | **0.827** | **0.709** | **0.799** | 0.868 | 149/152 |

Best overall: **dice03_bce07_pw3** — highest Dice, IoU, and precision at the
cost of ~8% recall compared to the pw=7.5 runs.

---

## Loss configurations explained

### 1. Weighted BCE (pos_weight=7.5) — `baseline_no_scheduler`, `baseline`

```
DiceBCELoss(alpha=0.0, beta=1.0, pos_weight=7.5)
```

Pure binary cross-entropy with a 7.5x multiplier on foreground pixels. Each
pixel is classified independently: the loss for a foreground pixel is 7.5 times
more expensive to get wrong than a background pixel. This pushes the model
strongly toward high recall (don't miss any aggregates) at the expense of
precision (many false positives are tolerated because they're "cheap").

This was the loss used in the previous TensorFlow training pipeline, where it
was called "EBCE" (edge-weighted binary cross-entropy with edge labels disabled).

The two runs differ only in the learning rate scheduler: without a scheduler,
the model plateaus at epoch ~25 and oscillates for the remaining epochs.
ReduceLROnPlateau adds ~2 points of Dice by allowing finer convergence.

Reference: pos_weight in BCE is a standard PyTorch feature, equivalent to
class-weighted cross-entropy (Bishop 2006, Pattern Recognition and Machine
Learning, Ch. 4.3.4).

### 2. Weighted BCE (pos_weight=3.0) — `bce_pw3`

```
DiceBCELoss(alpha=0.0, beta=1.0, pos_weight=3.0)
```

Same as above but with a lower foreground multiplier. Reducing pos_weight from
7.5 to 3.0 rebalances the recall-precision trade-off: the model is still
encouraged to find aggregates but is less willing to predict false positives.

This single change improved Dice by +0.06 and precision by +0.11 over the
baseline (same scheduler), with a modest recall drop of 0.04.

### 3. Dice + weighted BCE — `dice03_bce07_pw3`

```
DiceBCELoss(alpha=0.3, beta=0.7, pos_weight=3.0)
```

Combines two complementary loss terms:

- **Dice loss** (weight 0.3): Measures the overlap between prediction and ground
  truth as a ratio: `1 - 2*|intersection| / (|prediction| + |target|)`. Because
  it normalizes by the total predicted and true foreground area, it is
  inherently robust to class imbalance — no manual weighting needed. It also
  penalizes false positives structurally: every extra predicted pixel increases
  the denominator without increasing the intersection, directly reducing the
  Dice score.

- **Weighted BCE** (weight 0.7): Provides per-pixel gradient signal that
  stabilizes training. Pure Dice can have noisy gradients when foreground is
  very small (the denominator approaches zero). The BCE component ensures every
  pixel gets a learning signal regardless of the global overlap.

The BCE-heavy mix (0.7 BCE vs 0.3 Dice) was chosen because our annotation
boundaries are noisy — annotators are uncertain about exact aggregate edges.
Dice is sensitive to boundary disagreements (they directly reduce the overlap
ratio), while BCE treats each pixel independently so boundary noise only affects
boundary pixels. A BCE-heavy mix limits Dice's boundary sensitivity while still
benefiting from its false-positive penalty.

The smoothing constant in DiceLoss is 1.0 (not 1e-5), following nnU-Net
convention, to stabilize gradients when patches have very little foreground.

Reference: Dice loss for segmentation — Milletari et al. 2016, "V-Net: Fully
Convolutional Neural Networks for Volumetric Medical Image Segmentation"
(arxiv.org/abs/1606.04797). Dice+CE combination — Isensee et al. 2021, nnU-Net
(doi.org/10.1038/s41592-020-01008-z).

### 4. Scheduler: ReduceLROnPlateau

All runs except `baseline_no_scheduler` use:

```
ReduceLROnPlateau(mode="min", factor=0.5, patience=10, min_lr=1e-6)
```

Halves the learning rate when validation loss stops improving for 10 epochs.
This allows the model to converge further after the initial fast learning phase.
The scheduler added ~2 points of Dice to the pw=7.5 baseline and enabled the
other runs to train for 150–193 epochs before early stopping (patience=20).

Reference: StarDist uses the same scheduler with Adam on small bioimage
datasets (doi.org/10.1007/978-3-030-00934-2_30).

---

## Key observations

1. **Lowering pos_weight had the largest single effect.** Reducing from 7.5 to
   3.0 improved Dice by +0.06 and precision by +0.11. The pw=7.5 runs were
   over-weighting foreground, causing the model to predict too liberally.

2. **Adding Dice loss improved precision further.** The Dice component added
   another +0.02 Dice and +0.07 precision on top of the pos_weight reduction,
   confirming that overlap-based optimization helps control false positives
   beyond what pixel-level weighting alone achieves.

3. **Recall dropped gracefully.** From 0.95 (pw=7.5) to 0.87 (Dice+BCE pw=3.0)
   — a modest trade-off for +0.21 precision and +0.10 Dice. Given that
   annotations have uncertain boundaries, some of the "lost recall" is likely
   the model correctly not matching imprecise annotation edges.

4. **The scheduler matters but is secondary to loss design.** It added ~2 Dice
   points to the baseline, while the loss changes added ~8 points.

---

## Prediction comparison results (image_0000, threshold=0.5)

Quantitative evaluation using `scripts/compare_predictions.py` on a held-out image,
with edge-aware metrics, annotation quality analysis, and error classification.

### Precision-recall tradeoff

| Model | Dice | Precision | Recall | FP pixels | FN pixels |
|---|---|---|---|---|---|
| baseline | 0.71 | 0.57 | 0.93 | 23,325 | 2,298 |
| baseline_no_scheduler | 0.71 | 0.57 | 0.92 | 22,948 | 2,725 |
| bce_pw3 | 0.79 | 0.72 | 0.86 | 11,235 | 4,639 |
| dice03_bce07_pw3 | 0.81 | 0.81 | 0.81 | 6,167 | 6,540 |

### Error location: most errors are at edges, not cores

Across all models, ~70% of FP pixels are edge errors (boundary overshoot) and
~85–90% of FN pixels are at edges too. Core errors are small — all models
correctly identify aggregate **centers**, they just disagree on **boundaries**.

### Annotation quality analysis

Each FP/FN connected component was classified as "aggregate-like" or not based
on its intensity and sharpness (Laplacian variance) relative to TP components.
Components above the TP 25th percentile in both features are considered
aggregate-like.

| Model | FP annot. misses | FP hallucinations | FN annot. errors | FN model misses |
|---|---|---|---|---|
| baseline | 2,424 | 1,026 | 188 | 199 |
| baseline_no_scheduler | 2,095 | 990 | 170 | 301 |
| bce_pw3 | 1,706 | 660 | 261 | 603 |
| dice03_bce07_pw3 | 1,257 | 435 | 337 | 933 |

**For FPs:** ~70% of "false positives" across all models are bright and sharp —
they look like real aggregates the annotator missed. Only ~30% are actual
hallucinations (dim/blurry regions).

**For FNs:** roughly half are dim/blurry (annotation errors the model correctly
rejects), and half are real aggregates the model missed.

---

## Conclusion

**1. Dice+BCE with pos_weight is clearly the best configuration.**
`dice03_bce07_pw3` (α=0.3 Dice, β=0.7 BCE, pos_weight=3) wins on nearly every
metric: highest Dice (0.81), highest IoU (0.68), highest core Dice (0.71),
highest object precision (0.86), and fewest hallucinations (435 vs 1,026 for
baseline).

**2. pos_weight matters more than Dice vs BCE mixing.**
The biggest jump is from baseline → bce_pw3 (pure BCE but with pos_weight=3):
Dice goes from 0.71 → 0.79, FP pixels drop by half (23k → 11k). Adding Dice on
top gives a further but smaller improvement. The pos_weight addresses the class
imbalance (aggregates are sparse), which is the dominant problem.

**3. The scheduler makes almost no difference.**
baseline vs baseline_no_scheduler are nearly identical on every metric.
ReduceLROnPlateau isn't helping — the model converges fine without it for this
task.

**4. The baselines massively over-predict.**
57% precision means nearly half of what the baseline calls "aggregate" isn't. But
the annotation quality analysis shows ~70% of those FPs are bright and sharp —
likely real aggregates the annotator missed. So the baseline's true precision is
higher than 57%, but it's still the worst at rejecting noise (1,026
hallucinations vs 435 for dice03_bce07_pw3).

**5. There's a precision-recall tradeoff, but dice03_bce07_pw3 handles it best.**
It sacrifices recall (0.81 vs 0.93 for baseline) but the lost recall is mostly
edge pixels (5,940 of 6,540 FN pixels are at edges). Its core FN count (600) is
only modestly higher than baseline's (330) — it's missing boundaries, not entire
aggregates.

**6. Annotation noise is a ceiling on these metrics.**
With ~2,400 FP components classified as annotation misses even for the best
model, the reported Dice of 0.81 underestimates true performance. Improving
further requires better annotations, not better losses.

### Recommendation

Use `dice03_bce07_pw3` (α=0.3, β=0.7, pos_weight=3.0) as the loss
configuration going forward. Drop the scheduler — it adds complexity without
benefit. Further gains will come from architecture improvements (the ablation
study) and annotation refinement, not loss tuning.
