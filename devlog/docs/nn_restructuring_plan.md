# NN Module Restructuring Plan

Detailed plan for the `aggrequant/nn/` module redesign. Written so that any
agent can continue the work from where it was left off.

---

## Current State (2026-03-12)

### Completed

- [x] **Step 1: Clean up `__init__.py` files** — All `nn/` subpackage `__init__.py`
  files converted to package markers only (docstring, no re-exports). Files changed:
  - `aggrequant/nn/__init__.py`
  - `aggrequant/nn/architectures/__init__.py`
  - `aggrequant/nn/architectures/blocks/__init__.py`
  - `aggrequant/nn/training/__init__.py`
  - `aggrequant/nn/evaluation/__init__.py`
  - `aggrequant/nn/data/__init__.py`

- [x] **Step 2: Add new blocks** — Two new block files created:
  - `aggrequant/nn/architectures/blocks/convnext.py` — `ConvNeXtBlock`, `LayerNorm2d`
  - `aggrequant/nn/architectures/blocks/eca.py` — `ECABlock`

- [x] **Step 2b: Update unet.py** — `ModularUNet`, `EncoderBlock`, `DecoderBlock`
  updated to support:
  - `encoder_block="convnext"` / `decoder_block="convnext"` option
  - `use_eca=True` option (mutually exclusive with `use_se` and `use_cbam`)
  - Helper functions `_make_conv_block()` and `_make_channel_attention()` extracted
    to reduce duplication
  - Imports changed from re-exported `blocks` package to direct file imports

- [x] **Step 2c: Fix ASPP BatchNorm bug** — `ASPPPooling` had BatchNorm on a 1×1
  tensor (after global avg pool), which fails with batch_size=1. Fixed by removing
  BatchNorm from the pooling branch and using `bias=True` on the conv instead.

- [x] **Smoke test** — All 9 configurations verified (batch_size=2, input 128×128):
  1. Baseline UNet (31.0M params)
  2. ResUNet (31.9M params)
  3. Attention ResUNet (32.3M params)
  4. SE Attention ResUNet (32.3M params)
  5. ASPP SE Attention ResUNet (31.7M params)
  6. ConvNeXt UNet (29.7M params)
  7. ECA Attention ResUNet (32.3M params)
  8. CBAM UNet (31.3M params)
  9. Deep Supervision ResUNet (31.2M params)

- [x] **Documentation** — `docs/architecture_modules.md` created with mathematical
  descriptions of all 9 modules and the recommended 7-variant ablation study.

- [x] **Step 4: Add model registry** — `aggrequant/nn/architectures/registry.py`
  with 7 named presets and `create_model()` / `list_models()` / `describe_models()`.
  All 7 models verified.

- [x] **Step 5: Add inference module** — `aggrequant/nn/inference.py` with
  `predict_full()`, `predict_tiled()` (Gaussian blending), `predict()` (auto OOM
  fallback), and `postprocess_predictions()`. Tested on synthetic 2040×2040 image.

- [x] **Step 6: Run full test suite** — 37/37 tests passed, no regressions.

### NOT YET DONE

- [ ] **Step 3: Add UNet++ architecture** *(deferred — implement in a future session)*

---

## Step 3: Add UNet++ Architecture

**File:** `aggrequant/nn/architectures/unet_plusplus.py`

UNet++ requires a fundamentally different skip connection structure (nested dense
blocks), so it should be a separate class, not a flag on ModularUNet.

**Design:**
- Class `UNetPlusPlus(nn.Module)` with the same constructor signature as
  `ModularUNet` (in_channels, out_channels, features, etc.) for consistency
- Internally builds the dense skip grid X_{i,j} where each node receives all
  previous nodes at the same level + upsampled node from below
- Uses the same `_make_conv_block()` helper for the dense nodes
- Supports deep supervision (output from each column of the top row)
- Same `count_parameters()`, `__repr__()`, `get_config()` interface

**Key implementation detail:**
The dense grid for features=[64, 128, 256, 512] looks like:

```
Level 0:  X₀₀ → X₀₁ → X₀₂ → X₀₃ → X₀₄  (64 ch)
Level 1:  X₁₀ → X₁₁ → X₁₂ → X₁₃         (128 ch)
Level 2:  X₂₀ → X₂₁ → X₂₂               (256 ch)
Level 3:  X₃₀ → X₃₁                      (512 ch)
Bridge:   X₄₀                             (1024 ch)
```

Each X_{i,j} (j>0) receives:
- All previous outputs at same level: X_{i,0}, X_{i,1}, ..., X_{i,j-1}
- Upsampled output from below: Upsample(X_{i+1,j-1})
- These are concatenated along channels, then passed through a conv block

The first column X_{i,0} is the encoder output (same as standard UNet).

**Testing:** Smoke test with same input (2, 1, 128, 128), verify output shape and
parameter count.

---

## Step 4: Add Model Registry

**File:** `aggrequant/nn/architectures/registry.py`

A dictionary mapping variant names to their constructor kwargs. This makes it
easy to instantiate any variant by name (for benchmarking scripts, configs, etc.).

**Design:**

```python
MODEL_REGISTRY = {
    "baseline": {
        "class": "UNet",
        "kwargs": {"encoder_block": "double_conv", "decoder_block": "double_conv"},
    },
    "resunet": {
        "class": "UNet",
        "kwargs": {"encoder_block": "residual", "decoder_block": "residual"},
    },
    "attention_resunet": {
        "class": "UNet",
        "kwargs": {
            "encoder_block": "residual", "decoder_block": "residual",
            "use_attention_gates": True,
        },
    },
    "se_attention_resunet": {
        "class": "UNet",
        "kwargs": {
            "encoder_block": "residual", "decoder_block": "residual",
            "use_attention_gates": True, "use_se": True,
        },
    },
    "aspp_se_attention_resunet": {
        "class": "UNet",
        "kwargs": {
            "encoder_block": "residual", "decoder_block": "residual",
            "use_attention_gates": True, "use_se": True, "bridge_type": "aspp",
        },
    },
    "unet_plusplus": {
        "class": "UNetPlusPlus",
        "kwargs": {"encoder_block": "residual"},
    },
    "convnext_unet": {
        "class": "UNet",
        "kwargs": {"encoder_block": "convnext", "use_attention_gates": True},
    },
}
```

Plus a `create_model(name, **override_kwargs)` function that looks up the registry,
merges overrides, and returns the instantiated model.

**Side comparison variants** (not in main registry, but documented):
- `"eca_attention_resunet"` — same as `se_attention_resunet` but `use_eca=True`
  instead of `use_se=True`
- Any variant + `use_deep_supervision=True`

---

## Step 5: Add Inference Module

**File:** `aggrequant/nn/inference.py`

Ports the TF backup's sliding-window patch/stitch inference to PyTorch, with
improvements.

**Design — two inference modes:**

### Mode 1: Full-resolution (primary)

```python
def predict_full(model, image, device=None):
    """Run inference on a full image in one forward pass.

    Pads image to nearest multiple of 16 (not power-of-2), runs model,
    crops back to original size.
    """
```

- Input: numpy array (H, W), typically 2040×2040
- Pad to 2048×2048 (nearest multiple of 16) using reflection padding
- Normalize using percentile normalization (from `aggrequant.common.image_utils`)
- Convert to tensor (1, 1, H, W), run model, sigmoid, crop back
- Return probability map as numpy array (H, W)

### Mode 2: Tiled with Gaussian blending (fallback)

```python
def predict_tiled(model, image, tile_size=256, stride=128, device=None):
    """Run inference using overlapping tiles with Gaussian blending.

    For GPUs with insufficient VRAM for full-resolution.
    """
```

- Split image into overlapping tiles (default 256×256, stride 128)
- Create Gaussian weight map for blending (center-weighted, edges fade)
- Run model on each tile
- Accumulate weighted predictions and weight sums
- Divide to get blended probability map

### Auto-detect wrapper

```python
def predict(model, image, device=None, tile_size=256, stride=128):
    """Run inference with automatic mode selection.

    Tries full-resolution first. Falls back to tiled if CUDA OOM.
    """
```

### Post-processing

```python
def postprocess_predictions(probability_map, threshold=0.5,
                            min_area=9, max_hole_area=6000):
    """Convert probability map to instance labels.

    Threshold → morphological cleanup → connected components.
    Reuses existing postprocessing functions from
    aggrequant.segmentation.postprocessing.
    """
```

**Testing:** Test with a synthetic 2040×2040 image, verify output shape and
that tiled output approximately matches full-resolution output.

---

## Step 6: Run Full Test Suite

After all changes, run:

```bash
conda run -n AggreQuant python -m pytest tests/ -v
```

If tests fail, fix. The existing tests are in `tests/unit/` and test
aggregate/nuclei/cell segmentation — they should not be affected by nn/ changes
since nn/ is not imported by the main pipeline segmenters.

Additionally, create a quick integration test:

```python
# Test that all registry models can do a forward pass
def test_all_registry_models():
    from aggrequant.nn.architectures.registry import create_model, MODEL_REGISTRY
    x = torch.randn(2, 1, 128, 128)
    for name in MODEL_REGISTRY:
        model = create_model(name)
        out = model(x)
        assert out.shape == (2, 1, 128, 128)
```

---

## File Map

After completion, the `nn/` module structure will be:

```
aggrequant/nn/
├── __init__.py                          # Package marker only
├── utils.py                             # get_device()
├── inference.py                         # NEW: predict_full, predict_tiled, predict
├── architectures/
│   ├── __init__.py                      # Package marker only
│   ├── unet.py                          # ModularUNet (variants 1-5, 7)
│   ├── unet_plusplus.py                 # NEW: UNetPlusPlus (variant 6)
│   ├── registry.py                      # NEW: MODEL_REGISTRY, create_model()
│   └── blocks/
│       ├── __init__.py                  # Package marker only
│       ├── conv.py                      # SingleConv, DoubleConv
│       ├── residual.py                  # ResidualBlock, BottleneckResidualBlock
│       ├── attention.py                 # AttentionGate, MultiHeadAttentionGate
│       ├── se.py                        # SEBlock, SEConvBlock, SEResidualBlock
│       ├── cbam.py                      # ChannelAttention, SpatialAttention, CBAM, etc.
│       ├── eca.py                       # NEW: ECABlock
│       ├── convnext.py                  # NEW: ConvNeXtBlock, LayerNorm2d
│       └── aspp.py                      # ASPP, ASPPBridge, LightASPP (bug fixed)
├── data/
│   ├── __init__.py                      # Package marker only
│   ├── dataset.py                       # AggregateDataset, PatchDataset, InferenceDataset
│   └── augmentation.py                  # Albumentations pipelines
├── training/
│   ├── __init__.py                      # Package marker only
│   ├── losses.py                        # DiceLoss, DiceBCELoss, FocalLoss, etc.
│   └── trainer.py                       # Trainer, TrainingHistory, train_model
└── evaluation/
    ├── __init__.py                      # Package marker only
    └── metrics.py                       # dice_score, iou_score, evaluate_model, etc.
```

---

## Coding Conventions (for any continuing agent)

- **Imports**: Always use full absolute imports (`from aggrequant.nn.architectures.blocks.conv import DoubleConv`), never relative or re-exported
- **`__init__.py`**: Package markers only — docstring, no imports, no `__all__`
- **Docstrings**: Google/NumPy style with Arguments/Returns/Example sections
- **Type hints**: On all function signatures
- **No author lines** in code files
- **Line length**: 100 characters (Black formatter)
- **Execution**: Always use `conda run -n AggreQuant` for running code
- **Commits**: One-liner messages, no attribution lines
