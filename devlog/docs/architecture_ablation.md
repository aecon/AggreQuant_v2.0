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

*To be filled after training runs complete.*
