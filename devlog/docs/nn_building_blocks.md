# Neural Network Building Blocks: Conceptual Guide

This document explains the building blocks in `aggrequant/nn/` in plain language.
For mathematical formulations, see `architecture_modules.md`.

Tensor shape convention: **(B, C, H, W)** where B = batch size (number of images
processed at once), C = channels (feature maps), H = height, W = width.

---

## Basic Blocks (`blocks/conv.py`)

### SingleConv

The atomic unit. Three operations in sequence:

```
Conv2d (3x3) -> BatchNorm -> ReLU
```

- **Conv2d**: Slides a 3x3 kernel across the image, computing weighted sums.
  `in_channels -> out_channels`. With `padding=1`, the output H x W stays the
  same as input. `bias=False` because BatchNorm already has a learnable bias.
- **BatchNorm**: For each channel, normalizes values to mean ~0 and std ~1 across
  the batch, then applies learnable scale and shift. Stabilizes training.
- **ReLU**: Replaces all negative values with 0. Adds non-linearity — without it,
  stacking convolutions would just be one big linear operation. `inplace=True`
  saves memory.

### DoubleConv

Two `SingleConv` blocks stacked back to back. This is the standard UNet building
block from Ronneberger 2015 — every level of the encoder and decoder applies two
3x3 convolutions. The `mid_channels` parameter controls the channel count between
the two convs (default: same as `out_channels`).

Exists as a class (rather than inline code) because `unet.py` needs it as a
pluggable block with a consistent `(in_channels, out_channels)` interface, same
as `ResidualBlock` and `ConvNeXtBlock`.

---

## Residual Blocks (`blocks/residual.py`)

### ResidualBlock

Same as `DoubleConv` but with a **skip connection** (shortcut) that adds the input
directly to the output:

```
output = ReLU(input + DoubleConv(input))
```

If `in_channels != out_channels`, the skip uses a 1x1 convolution to match
dimensions.

**Why it matters:** In deep networks, gradients can vanish as they flow backward
through many layers, making training difficult. The skip connection provides a
direct highway for gradients — they can flow through the addition without being
attenuated. The network only needs to learn the *residual* (the difference between
input and desired output), which is easier than learning the full transformation
from scratch. This is what enabled training networks with 100+ layers (He et al.,
2016).

### BottleneckResidualBlock

A more parameter-efficient variant using the pattern:
```
1x1 conv (reduce channels by 4x) -> 3x3 conv -> 1x1 conv (expand back)
```

The 1x1 convolutions are cheap (no spatial mixing), so the expensive 3x3 conv
operates on fewer channels. Same skip connection as `ResidualBlock`. Useful for
deeper networks where parameter count matters.

---

## Attention Gate (`blocks/attention.py`)

### AttentionGate

Helps the decoder decide which parts of the encoder's skip connection are useful.

**The problem:** In a standard UNet, skip connections pass *all* encoder features
to the decoder — relevant and irrelevant alike. The encoder encodes background
texture, cell membranes, aggregates, and everything else. The decoder gets all of
it and has to figure out what matters.

**How it works:** Takes two inputs:

- **g** (gate) — decoder features. Low-resolution but semantically rich. Through
  training, the bottleneck and decoder learn to represent task-relevant information.
  The decoder doesn't literally "see" the ground truth — but backpropagation shapes
  its features to encode what the task needs.
- **x** (skip) — encoder features. High-resolution but contains everything.

Steps:
1. Both `g` and `x` are projected to a shared intermediate space via 1x1
   convolutions (`W_g` and `W_x`)
2. `g` is upsampled to match `x`'s spatial size
3. They're added, passed through ReLU -> 1x1 conv -> Sigmoid, producing an
   **attention map** (values 0 to 1 for each spatial location)
4. The attention map multiplies the original skip features: `x * attention`

```
g -> W_g --+
           +-- add -> ReLU -> psi -> attention map (0..1) -> multiply with x
x -> W_x --+
```

Locations where the decoder "agrees" with the encoder get attention ~ 1 (kept).
Irrelevant locations get attention ~ 0 (suppressed).

**For aggregates:** The gate signal says "I'm looking for aggregate-like patterns
here." The attention map dims skip features in background regions and brightens
them where aggregates likely are.

### MultiHeadAttentionGate

Same idea as `AttentionGate`, but splits the skip features into multiple groups
("heads") and computes a separate attention map for each. With 4 heads, you get
4 independent attention maps, each potentially focusing on different aspects
(e.g., brightness, edges, texture).

Inspired by multi-head attention in transformers. Adds parameters and complexity
over a single gate. Not used by any of the 7 registry presets currently.

---

## Squeeze-and-Excitation (`blocks/se.py`)

### SEBlock

Learns which **channels** (feature maps) are important and scales them accordingly.

The problem: a convolution treats all its output channels equally. But for a given
input, some channels might detect relevant patterns (e.g., bright punctate spots)
while others detect irrelevant ones (e.g., smooth background). SE learns to
amplify the useful channels and suppress the rest.

Three steps:
1. **Squeeze**: Global average pooling — collapse each channel's H x W spatial map
   into a single number. This gives a channel descriptor vector of length C.
2. **Excitation**: Two fully-connected layers (C -> C/16 -> C) with ReLU in between
   and Sigmoid at the end. Learns a weight between 0 and 1 for each channel.
3. **Scale**: Multiply each channel by its learned weight.

No spatial information — it treats the entire spatial extent equally. It only asks
"which channels matter?" not "where in the image?"

### SEConvBlock / SEResidualBlock

Pre-composed blocks: `DoubleConv + SEBlock` and `ResidualBlock + SEBlock`
respectively. Convenience classes that apply SE after the convolutions. The `UNet`
class composes these separately (via `EncoderBlock`), so these composite blocks
exist as standalone alternatives.

---

## CBAM (`blocks/cbam.py`)

### ChannelAttention

Like SE, but uses **both** average pooling and max pooling, fed through a shared
MLP. The max pooling path captures the strongest activation per channel (SE only
uses the average, which can wash out strong local signals). The two paths are
added and passed through Sigmoid.

Output: a weight per channel, shape (B, C, 1, 1).

### SpatialAttention

Learns **where** in the image to focus. Takes a feature map and computes:
1. Average and max across all channels at each spatial position -> two (B, 1, H, W)
   maps
2. Concatenates them and applies a 7x7 convolution -> Sigmoid

Output: a weight per spatial location, shape (B, 1, H, W).

### CBAM

Sequential application: ChannelAttention first (which channels?), then
SpatialAttention (where in space?). Each multiplies its weights element-wise with
the features.

**SE vs CBAM:** CBAM is a strict superset of SE — its channel attention is SE plus
max-pooling, and it adds spatial attention on top. More expressive, but the spatial
attention overlaps with AttentionGate (both learn per-pixel spatial weights). When
AttentionGate is already present, CBAM's spatial component may be redundant.

### CBAMConvBlock / CBAMResidualBlock

Pre-composed: `DoubleConv + CBAM` and `ResidualBlock + CBAM`. Same rationale as
SEConvBlock/SEResidualBlock.

---

## ECA (`blocks/eca.py`)

### ECABlock

A lightweight alternative to SE for channel attention. Instead of two FC layers
with a bottleneck, uses a single **1D convolution** across channels:

```
SE:  pool -> FC(C -> C/16) -> ReLU -> FC(C/16 -> C) -> Sigmoid    (2C^2/16 params)
ECA: pool -> Conv1D(kernel=k) -> Sigmoid                           (k params, k ~ 3-5)
```

SE squeezes channels through a bottleneck (C -> C/16 -> C), which loses
information. ECA avoids this — each channel interacts with its k nearest neighbors
via the 1D convolution, with no reduction. The kernel size k is automatically
computed from the channel count (more channels -> slightly larger kernel).

**Result:** Same goal as SE (learn which channels to amplify), simpler mechanism,
fewer parameters, and the paper reports better performance.

---

## ConvNeXt (`blocks/convnext.py`)

### LayerNorm2d

Standard `nn.LayerNorm` works on the last dimension. For image tensors (B, C, H, W),
we need to normalize across channels. This adapter permutes to (B, H, W, C),
applies LayerNorm, and permutes back.

### ConvNeXtBlock

A "modernized" convolution block that borrows design ideas from Vision Transformers
but stays purely convolutional. Drop-in replacement for `ResidualBlock`.

```
output = input + gamma * pointwise_up(GELU(pointwise_down(LayerNorm(depthwise_7x7(input)))))
```

Key differences vs `ResidualBlock`:

| | ResidualBlock | ConvNeXtBlock |
|---|---|---|
| **Convolution** | Two 3x3 standard convs | One 7x7 depthwise conv (each channel independently) |
| **Receptive field** | 5x5 (from two 3x3) | 7x7 per layer |
| **Normalization** | BatchNorm | LayerNorm (more stable for small batches) |
| **Activation** | ReLU | GELU (smoother gradients) |
| **Bottleneck** | Standard (squeeze then expand) | Inverted (expand 4x then squeeze) |
| **Residual scaling** | Fixed (weight 1.0) | Learnable per-channel gamma (LayerScale) |

The depthwise conv is cheap (C x 7 x 7 params vs C^2 x 3 x 3 for standard conv)
so the larger kernel doesn't cost much. The inverted bottleneck puts more capacity
in the non-linear part. Matches Vision Transformer performance on ImageNet without
any attention mechanism (Liu et al., 2022).

---

## ASPP (`blocks/aspp.py`)

### ASPP (Atrous Spatial Pyramid Pooling)

Captures context at **multiple spatial scales** simultaneously using dilated
(atrous) convolutions.

A standard 3x3 convolution sees a 3x3 neighborhood. A **dilated** 3x3 convolution
with dilation rate *d* samples every d-th pixel in the 3x3 pattern, giving an
effective receptive field of (2d+1) x (2d+1) with only 9 parameters. It's like
looking at the image through a wider lens without increasing computation.

ASPP runs several branches in parallel:
- 1x1 convolution (local features)
- 3x3 convolution with dilation 6 (13x13 receptive field)
- 3x3 convolution with dilation 12 (25x25 receptive field)
- 3x3 convolution with dilation 18 (37x37 receptive field)
- Global average pooling (entire image context)

All branch outputs are concatenated and projected to the desired output channels.

**For aggregates:** Helps distinguish small punctate aggregates from large diffuse
ones, and helps understand the cellular context around each aggregate.

### ASPPBridge

Wraps ASPP for use as the UNet bottleneck (bridge). Adds a refinement convolution
after the ASPP output. Drop-in replacement for `DoubleConv` at the bridge position.

### LightASPP

Lightweight variant using depthwise separable convolutions instead of standard
convolutions in each branch. Fewer parameters, similar multi-scale capability.

---

## How the UNet Assembles These Blocks

The `UNet` class in `unet.py` uses these blocks as pluggable components:

| UNet component | Block options |
|---|---|
| **Encoder blocks** | `DoubleConv`, `ResidualBlock`, or `ConvNeXtBlock` |
| **Decoder blocks** | `DoubleConv`, `ResidualBlock`, or `ConvNeXtBlock` |
| **Bridge (bottleneck)** | `DoubleConv`, `ResidualBlock`, or `ASPPBridge` |
| **Skip connections** | Plain concatenation or `AttentionGate` |
| **Channel attention** | None, `SEBlock`, `CBAM`, or `ECABlock` |

Each encoder level: conv block -> optional channel attention -> max pool (downsample).
Each decoder level: upsample -> concatenate skip -> optional attention gate -> conv block -> optional channel attention.

The `registry.py` defines 7 named presets that combine these options into specific
architectures for the ablation study (see `architecture_modules.md` for the full
table and rationale).

---

## Training Data

Source data for the aggregate segmentation model:

- **Raw images**: `../AggreQuant_training_data/data/2024-10-25_Annotations_19images_AE_DV_EDC_LJ/raw/`
- **Annotation masks**: `../AggreQuant_training_data/data/2024-10-25_Annotations_19images_AE_DV_EDC_LJ/annotated/`

19 annotated images from 4 annotators (AE, DV, EDC, LJ). TIFF format, naming
convention: `image_raw####.tif` / `image_ann####.tif`.

---

## Training Data Pipeline (`datatools/`)

### Patch Extraction (`extract_patches`)

Full-size microscopy images (e.g., 2040×2040) are too large for direct training.
Instead, they are grid-cut into non-overlapping patches (e.g., 128×128) and saved
to disk. For a 2040×2040 image with 128×128 patches: 15×15 = 225 patches per image
(the last 120 pixels on each edge are skipped).

Patches are saved as individual TIFF files with names encoding their source image
and grid position: `{image_stem}_y{row}_x{col}.tif`.

### Train/Val Split

The patch file list is **shuffled and split** into train and val sets (default
80/20). This ensures:
- Every source image contributes patches to both train and val
- No single patch appears in both sets — zero data leakage
- The NN sees the full heterogeneity of all images during training

This is better than splitting at the image level (which would exclude some images
from training entirely) or random-cropping on the fly (which can produce
overlapping patches across train/val).

### PatchDataset

A PyTorch `Dataset` that loads pre-extracted patches from disk. Each
`__getitem__` call loads one image/mask pair, normalizes the image (percentile
scaling to [0, 1]), binarizes the mask, and optionally applies augmentation.

### Augmentation (`augmentation.py`)

Uses `torchvision.transforms.v2` with `tv_tensors` for joint image/mask handling.
Spatial transforms (flip, rotate, affine) are applied to both image and mask
automatically. Intensity transforms (brightness, contrast, gamma, noise, blur)
are applied to the image only.

Intentionally excluded: elastic deformation, shear, and random erasing — these
distort small aggregate structures unnaturally.

---

## Loss Functions (`training/losses.py`)

### Available losses

| Loss | What it does | When to use |
|---|---|---|
| **DiceLoss** | 1 - Dice coefficient. Optimizes overlap directly. | Good for imbalanced segmentation |
| **DiceBCELoss** | alpha × Dice + beta × BCE. Combines pixel-wise and overlap-based supervision. | **Recommended default** (nnU-Net uses Dice + CE) |
| **FocalLoss** | Down-weights easy examples via (1-p)^gamma. | Severe class imbalance |
| **TverskyLoss** | Generalizes Dice with separate FP/FN weights. alpha > beta = better recall. | When FP/FN trade-off matters |
| **FocalTverskyLoss** | Focal mechanism on top of Tversky. | Hard examples + class imbalance |
| **EdgeWeightedLoss** | Adds Laplacian edge-weighted BCE on top of a base loss. | See note below |
| **DeepSupervisionLoss** | Wraps any loss for multi-scale deep supervision outputs. | When UNet has `use_deep_supervision=True` |

### Factory function

`get_loss_function(name)` maps string names to classes:
`"dice"`, `"bce"`, `"dice_bce"`, `"focal"`, `"tversky"`, `"focal_tversky"`.

### Loss selection for aggregate segmentation

Our data has two properties that constrain loss selection:

1. **Severe class imbalance (~1% foreground)**. Pure unweighted BCE is dominated
   by background pixels and under-trains on aggregates.

2. **Noisy annotation boundaries**. Annotators are confident about aggregate
   cores but uncertain about exact boundaries. This means ground truth edges are
   unreliable — the model should not be heavily penalized for boundary
   disagreements.

These two properties pull in different directions. Dice loss handles imbalance
well (it normalizes by prediction + target size) but is sensitive to boundary
noise (boundary disagreements directly reduce the overlap ratio). Weighted BCE
handles noisy boundaries better (each pixel is independent, so boundary errors
only affect boundary pixels) but needs manual class weighting.

**Losses considered and rejected:**

- **Focal Loss**: Designed for easy-example dominance in classification (Lin et
  al., 2017), not segmentation. Focuses training on hard/ambiguous pixels, which
  in our case are noisy boundaries and background regions resembling aggregates
  — this can increase false positives rather than reduce them. Multiple medical
  imaging studies (Ma et al. 2021, Yeung et al. 2022) found Focal Loss
  underperforms Dice-based losses for small object segmentation.

- **Focal Tversky Loss**: Adds two extra hyperparameters (alpha/beta + gamma)
  on top of Tversky. Abraham & Khan (2019) showed gains on small lesions, but
  follow-up studies found results were inconsistent and sensitive to gamma. With
  only 19 images, the hyperparameter search space is a concern.

- **Pure Dice Loss**: Noisy gradients when foreground is very small (denominator
  approaches zero). nnU-Net always combines Dice with CE for gradient stability.

**Losses under evaluation (baseline UNet ablation):**

| Config | Loss | Rationale |
|---|---|---|
| `alpha=0.0, beta=1.0, pw=7.5` | Pure weighted BCE | Previous approach. High recall (~0.93) but low precision (~0.64). The heavy pos_weight penalizes missed aggregates but not false positives. |
| `alpha=0.0, beta=1.0, pw=3.0` | Pure weighted BCE, lower weight | Reduces recall bias to improve precision. Simplest change from baseline. |
| `alpha=0.3, beta=0.7, pw=3.0` | BCE-heavy Dice+BCE mix | Dice adds overlap pressure (penalizes FPs structurally). BCE dominates so boundary noise hurts less. Moderate pos_weight keeps some recall pressure. |
| `alpha=0.5, beta=0.5` | nnU-Net default (Dice+BCE) | Zero hyperparameters, most battle-tested. Risk: recall may drop with noisy boundaries. |

The smoothing constant in DiceLoss is set to 1.0 (not 1e-5), which stabilizes
gradients when patches have very little foreground. This follows nnU-Net
convention.

References:
- Isensee et al. 2021 (nnU-Net): doi.org/10.1038/s41592-020-01008-z
- Ma et al. 2021 (Loss Odyssey): arxiv.org/abs/2103.02626
- Yeung et al. 2022: doi.org/10.1016/j.media.2022.102399

### Note on EdgeWeightedLoss — needs inspection

The current `EdgeWeightedLoss` implementation (formerly `BoundaryLoss`) has
known issues that should be addressed before using it in training:

1. **Uses Laplacian edge detection** instead of distance transforms. The
   literature (Kervadec et al. 2019, "Boundary loss for highly unbalanced
   segmentation") defines boundary loss using signed distance maps from the
   ground truth boundary, not Laplacian filters. Distance transforms give
   continuous weighting (pixels further from boundaries contribute less
   gradually), while Laplacian gives binary edge/non-edge.

2. **Hardcoded mixing weight** of 0.5 (line 451). Research shows ~10% boundary
   weight works best — larger weights (20-30%) consistently reduced performance
   (Nature Scientific Reports, 2025). The weight should be configurable and
   much lower than 0.5.

3. **Double BCE when wrapping DiceBCELoss**: `BoundaryLoss` always adds its own
   weighted BCE term on top of the base loss. If `base_loss = DiceBCELoss()`,
   you get two separate BCE terms — one from DiceBCELoss and one from
   BoundaryLoss.

4. **Scheduling**: The literature recommends mixing boundary loss *gradually* —
   regional loss dominates early training, boundary term increases over time.
   A fixed weight from epoch 1 is not best practice.

**Recommendation**: Use `DiceBCELoss` as the default. If ablation results show
boundary errors are a problem, implement a proper distance-transform-based
boundary loss with scheduled weighting. References:
- Kervadec et al. 2019: arxiv.org/abs/1812.07032
- Jadon 2020, survey of segmentation losses: arxiv.org/pdf/2006.14822
- Loss survey 2023: arxiv.org/html/2312.05391v1

---

## Learning Rate Scheduler (`training/trainer.py`)

### Scheduler selection

With Adam optimizer on a small dataset (19 images, 1900 patches), the scheduler
must be reactive to actual training dynamics rather than assuming a fixed
schedule. Key constraints:

1. **Early stopping** makes the total number of epochs unpredictable. Schedulers
   that back-load their useful behavior (OneCycleLR's final annealing,
   CosineAnnealing near T_max) may never reach their effective phase.

2. **Small, noisy validation set** (~380 patches). Validation loss oscillates
   epoch-to-epoch, so the scheduler must tolerate noise without triggering
   premature LR reductions.

3. **Class imbalance + weighted BCE**. Schedulers that allow LR to spike
   (OneCycleLR warmup, WarmRestarts) risk destabilizing the fragile foreground
   predictions — the 1% aggregate class can be "forgotten" during a high-LR
   restart.

**Schedulers considered and rejected:**

- **OneCycleLR** (Smith 2019): Incompatible with early stopping — designed for
  a fixed number of steps. The warmup to high LR is risky with severe class
  imbalance and weighted BCE (amplifies gradient noise from class weights).

- **CosineAnnealingWarmRestarts** (Loshchilov & Hutter 2017): Warm restarts to
  full LR can cause catastrophic forgetting of rare-class features. Benefits
  shown mainly on larger datasets.

- **CosineAnnealingLR**: Smooth and predictable, but T_max must match actual
  training length. With early stopping, T_max is unknown. Setting it too high
  means the useful low-LR phase is never reached.

- **PolynomialLR** (nnU-Net default, power=0.9): Designed for SGD + fixed 1000
  epochs without early stopping. With Adam's adaptive per-parameter LR, the
  scheduler's effect is dampened. Not well suited to our Adam + early stopping
  regime.

**Selected: ReduceLROnPlateau** (`factor=0.5, patience=10, min_lr=1e-6`).

This is what StarDist uses (Adam + small bioimage datasets). It reacts to actual
validation loss plateaus rather than following a fixed schedule. With our
baseline plateauing around epoch 25–30 at constant lr=1e-3, patience=10 triggers
the first reduction (1e-3 → 5e-4) around epoch 35–40, giving the model a second
phase of fine-tuning. Subsequent reductions at each new plateau, down to 1e-6.

References:
- StarDist: doi.org/10.1007/978-3-030-00934-2_30
- Smith 2019 (1cycle): arxiv.org/abs/1708.07120
- Loshchilov & Hutter 2017 (SGDR): arxiv.org/abs/1608.03983
