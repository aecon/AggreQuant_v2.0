# Architecture Ablation Study

Benchmark of 7 UNet architecture variants on the same annotated aggregate
dataset, using the best loss configuration from the loss comparison study.

---

## Fixed hyperparameters

All variants share:

| Parameter | Value |
|---|---|
| Loss | DiceBCE (alpha=0.3, beta=0.7, pos_weight=3.0) |
| Optimizer | Adam, lr=1e-3 |
| Scheduler | ReduceLROnPlateau (factor=0.5, patience=10, min_lr=1e-6) |
| Patch size | 192x192 |
| Batch size | 16 |
| Val split | 20% (seed=42) |
| Early stopping | patience=20 |
| Max epochs | 200 |
| Augmentation | Flip, rotate90, affine (no zoom) |

---

## Variants

| # | Name | What it adds | Reference |
|---|---|---|---|
| 1 | `baseline` | Baseline UNet (DoubleConv encoder/decoder) | Ronneberger 2015 |
| 2 | `resunet` | +residual blocks | He 2016 |
| 3 | `attention_resunet` | +attention gates on skip connections | Oktay 2018 |
| 4 | `se_attention_resunet` | +SE channel attention | Hu 2018 |
| 5 | `aspp_se_attention_resunet` | +ASPP multi-scale bridge | Chen 2017 |
| 6 | `convnext_unet` | ConvNeXt encoder + attention gates | Liu 2022 |
| 7 | `eca_attention_resunet` | ECA replaces SE | Wang 2020 |

---

## Running the ablation

```bash
# All 7 variants, single seed:
conda run -n AggreQuant python scripts/train_ablation.py

# Specific variants:
conda run -n AggreQuant python scripts/train_ablation.py --models baseline resunet attention_resunet

# Single variant:
conda run -n AggreQuant python scripts/train_ablation.py --models baseline

# Multi-seed for confidence intervals:
conda run -n AggreQuant python scripts/train_ablation.py --seeds 42 123 456

# Re-run everything (ignore existing checkpoints):
conda run -n AggreQuant python scripts/train_ablation.py --no-skip-existing
```

Output structure:
```
training_output/ablation/
├── ablation_results.csv
├── baseline/checkpoints/
├── resunet/checkpoints/
├── attention_resunet/checkpoints/
├── se_attention_resunet/checkpoints/
├── aspp_se_attention_resunet/checkpoints/
├── convnext_unet/checkpoints/
└── eca_attention_resunet/checkpoints/
```

Existing patches in `training_output/patches/` are reused automatically.
Existing checkpoints are skipped by default (`--no-skip-existing` to retrain).

---

## Results

5 of 7 variants completed training. `convnext_unet` and `eca_attention_resunet`
have not been trained yet. Metrics reported at the epoch with the lowest
validation loss.

| Model | Val Dice | Val IoU | Val Precision | Val Recall | Best epoch | Total epochs |
|---|---|---|---|---|---|---|
| **resunet** | **0.829** | **0.711** | **0.807** | 0.863 | 146 | 166 |
| attention_resunet | 0.828 | 0.710 | 0.799 | 0.869 | 138 | 158 |
| baseline | 0.828 | 0.709 | 0.796 | 0.872 | 118 | 138 |
| se_attention_resunet | 0.824 | 0.703 | 0.784 | 0.876 | 84 | 104 |
| aspp_se_attention_resunet | 0.816 | 0.692 | 0.775 | 0.871 | 69 | 77 |

---

## Analysis

### No meaningful improvement from architecture changes

The top 3 models (resunet, attention_resunet, baseline) are within **0.1% Dice**
of each other. The training curves confirm this — all plateau at the same
validation loss floor (~0.077).

### More complexity actually hurts

- **se_attention_resunet** is slightly worse and converges earlier (stopped at
  epoch 84). The SE channel attention adds parameters without improving
  generalization on this small dataset.
- **aspp_se_attention_resunet** is the worst performer and stopped earliest
  (epoch 69 of only 77 total). The increased parameter count from ASPP +
  SE + attention gates causes faster overfitting on 19 training images.

### Training curve observations

- All models reach similar validation loss floors (~0.077–0.082).
- Simpler models train longer before early stopping, suggesting better
  generalization.
- No model shows underfitting — the train-val gap is modest for all variants.
- The more complex models show steeper overfitting: train loss keeps dropping
  while val loss stagnates.

---

## Conclusion

The architecture is **not the bottleneck**. The baseline UNet is already
sufficient for this task. The limiting factors are:

1. **Dataset size** (19 images) — more complex models overfit faster without
   enough data to learn the additional parameters.
2. **Annotation quality** — as shown by the loss comparison study, ~70% of FPs
   are annotation misses, not model errors. Better architectures cannot fix
   noisy labels.

Further investment should go toward **more/better training data**, not
architecture complexity. The simple baseline UNet with
`DiceBCELoss(alpha=0.3, beta=0.7, pos_weight=3.0)` remains the best practical
choice.

The remaining two variants (`convnext_unet`, `eca_attention_resunet`) are
unlikely to change this conclusion given the clear trend that added complexity
provides no benefit on this dataset size.
