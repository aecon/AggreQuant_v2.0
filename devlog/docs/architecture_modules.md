# UNet Architecture Modules: Mathematical Summary and Selection

This document describes all candidate modules for the UNet architecture benchmark,
explains what each does mathematically, and recommends which to include in the
ablation study.

Notation: **x** is the input feature map of shape (B, C, H, W).

---

## Module Descriptions

### 1. DoubleConv (Baseline UNet block, Ronneberger 2015)

Two consecutive convolution-batchnorm-ReLU layers:

```
DoubleConv(x) = ReLU(BN(Conv₂(ReLU(BN(Conv₁(x))))))
```

Where Conv₁ and Conv₂ are 3×3 convolutions. This is the atomic building block of
the original UNet. Each convolution extracts local spatial features in a 3×3
neighborhood. Stacking two gives an effective receptive field of 5×5.

**What it does:** Learns local spatial patterns (edges, textures, small shapes).
No mechanism to look beyond the 5×5 window or to weight channels differently.

---

### 2. ResidualBlock (He et al. 2016)

Adds a skip connection around the double convolution:

```
ResidualBlock(x) = ReLU(x + BN(Conv₂(ReLU(BN(Conv₁(x))))))
```

If input and output channels differ, the skip uses a 1×1 conv to match dimensions:

```
ResidualBlock(x) = ReLU(Conv₁ₓ₁(x) + BN(Conv₂(ReLU(BN(Conv₁(x))))))
```

**What it does:** The identity shortcut lets gradients flow directly backward,
preventing vanishing gradients in deeper networks. The network only needs to learn
the *residual* F(x) = output - input, which is easier to optimize. Practically:
enables training deeper UNets without degradation.

---

### 3. AttentionGate (Oktay et al. 2018)

Applied to skip connections between encoder and decoder. Given gating signal **g**
(from decoder, coarse/semantic) and skip features **x** (from encoder, fine/spatial):

```
g₁ = BN(Conv₁ₓ₁(g))                          # project gate to intermediate space
x₁ = BN(Conv₁ₓ₁(x))                          # project skip to intermediate space
α  = σ(BN(Conv₁ₓ₁(ReLU(g₁ + x₁))))          # attention coefficients ∈ [0,1]
output = x ⊙ α                                # element-wise multiply
```

Where σ is sigmoid and ⊙ is element-wise multiplication. The attention map α has
shape (B, 1, H, W) — one scalar weight per spatial location.

**What it does:** The decoder "tells" the skip connection which spatial regions are
relevant. Regions where encoder features agree with decoder semantics get α ≈ 1
(passed through); irrelevant regions get α ≈ 0 (suppressed). This helps when the
encoder skip carries a lot of background noise that the decoder doesn't need —
which is exactly the case for aggregate segmentation where most of the image is
empty background.

---

### 4. SEBlock — Squeeze-and-Excitation (Hu et al. 2018)

Channel-wise recalibration. Three steps:

```
Step 1 — Squeeze:    z_c = (1/HW) Σᵢⱼ x_c(i,j)       # global avg pool per channel → (B, C)
Step 2 — Excitation: s = σ(W₂ · ReLU(W₁ · z))          # two FC layers: C → C/r → C, then sigmoid
Step 3 — Scale:      output_c = s_c · x_c                # multiply each channel by learned weight
```

Where W₁ ∈ ℝ^(C/r × C), W₂ ∈ ℝ^(C × C/r), r is the reduction ratio (default 16).

**What it does:** Learns which channels (feature maps) are important for the
current input and scales them accordingly. If channel 42 detects "bright punctate
spots" and that's present in the input, its weight goes up. If channel 17 detects
"large diffuse blobs" and there are none, its weight goes down. Essentially:
adaptive feature selection along the channel dimension. No spatial information — it
treats the entire spatial extent equally.

---

### 5. CBAM — Convolutional Block Attention Module (Woo et al. 2018)

Sequential application of channel attention (like SE but with max-pool too) then
spatial attention:

**Channel attention (enhanced SE):**

```
z_avg = AvgPool(x)                      # (B, C)
z_max = MaxPool(x)                      # (B, C)
M_c = σ(MLP(z_avg) + MLP(z_max))       # shared MLP, combine both → (B, C, 1, 1)
x' = M_c ⊙ x                           # channel-reweighted features
```

**Spatial attention:**

```
f_avg = mean(x', dim=channel)           # (B, 1, H, W) — average across channels
f_max = max(x', dim=channel)            # (B, 1, H, W) — max across channels
M_s = σ(Conv₇ₓ₇([f_avg; f_max]))       # 7×7 conv on concatenated maps → (B, 1, H, W)
output = M_s ⊙ x'                       # spatially-reweighted features
```

**What it does:** SE asks "which channels matter?" — CBAM asks that AND "which
spatial locations matter?" The channel attention is a strict improvement over SE
(adds max-pooling path). The spatial attention then highlights where in the image
to focus. For aggregates: channel attention selects the right feature detectors,
spatial attention highlights aggregate locations and suppresses background.

**SE vs CBAM:** CBAM ⊃ SE (CBAM's channel attention is SE + max-pool, plus spatial
attention). CBAM has more parameters and computation. If spatial attention doesn't
help (e.g., when attention gates already handle spatial selection), SE alone may
suffice.

---

### 6. ASPP — Atrous Spatial Pyramid Pooling (Chen et al. 2017)

Parallel dilated convolutions with different dilation rates:

```
branch₁ = BN(Conv₁ₓ₁(x))                            # standard 1×1
branch₂ = BN(Conv₃ₓ₃(x, dilation=6))                 # receptive field: 13×13
branch₃ = BN(Conv₃ₓ₃(x, dilation=12))                # receptive field: 25×25
branch₄ = BN(Conv₃ₓ₃(x, dilation=18))                # receptive field: 37×37
branch₅ = Upsample(AvgPool(x, output_size=1×1))       # global context

output = Conv₁ₓ₁([branch₁; branch₂; branch₃; branch₄; branch₅])
```

A dilated convolution with dilation rate *d* samples every *d*-th pixel in a 3×3
pattern, giving a receptive field of (2d+1)×(2d+1) with only 9 parameters.

**What it does:** Captures context at multiple scales simultaneously. A standard
3×3 conv sees 3×3 pixels. With dilation=12, the same 3×3 conv sees a 25×25 region.
By running multiple dilations in parallel, the network can reason about local
detail AND wide context in a single layer. Used at the bridge/bottleneck of the
UNet. For aggregates: helps distinguish a small bright aggregate from a large
diffuse one, and helps the network understand the local cellular context around
each aggregate.

---

### 7. UNet++ — Nested Dense Skip Connections (Zhou et al. 2020)

Standard UNet has one skip connection per level: encoder level *i* → decoder level
*i*. UNet++ fills in the gaps with intermediate dense blocks:

```
Standard UNet skip:     X₀,₀ ──────────────────────────────→ X₀,₄
                        X₁,₀ ──────────────────→ X₁,₃
                        X₂,₀ ──────→ X₂,₂
                        X₃,₀ → X₃,₁

UNet++ fills the grid:  X₀,₀ → X₀,₁ → X₀,₂ → X₀,₃ → X₀,₄
                        X₁,₀ → X₁,₁ → X₁,₂ → X₁,₃
                        X₂,₀ → X₂,₁ → X₂,₂
                        X₃,₀ → X₃,₁
```

Each intermediate node X_{i,j} receives:

```
X_{i,j} = Conv([X_{i,0}, X_{i,1}, ..., X_{i,j-1}, Upsample(X_{i+1,j-1})])
```

i.e., the upsampled output from the level below, concatenated with ALL previous
nodes at the same level.

**What it does:** The standard UNet skip directly connects encoder features (which
haven't seen the decoded context) to the decoder. This semantic gap can hurt
performance. UNet++ bridges this gap gradually through intermediate convolution
nodes, producing features that are progressively refined before reaching the
decoder. The dense connections also provide multiple paths for gradient flow. For
aggregates: helps when you have objects at multiple scales (tiny punctate + large
diffuse aggregates) because multi-scale features are fused more thoroughly.

**Trade-off:** Significantly more parameters and memory than standard UNet (roughly
2-3× for the skip connections). This is a structural change to the UNet, not a
drop-in module.

---

### 8. ConvNeXt Block (Liu et al. 2022)

A "modernized" convolution block that borrows design choices from Vision
Transformers but stays purely convolutional:

```
ConvNeXt(x) = x + γ · Linear_up(GELU(Linear_down(LayerNorm(DepthwiseConv₇ₓ₇(x)))))
```

Broken down:

1. **Depthwise 7×7 conv**: Conv with groups=C (each channel convolved
   independently). Large kernel (7×7) for wide receptive field, but cheap
   (C×7×7 params vs C²×3×3).
2. **LayerNorm**: Instead of BatchNorm. More stable for small batches.
3. **Linear_down**: 1×1 conv, C → 4C (inverted bottleneck — expand, not squeeze)
4. **GELU**: Smooth activation (instead of ReLU)
5. **Linear_up**: 1×1 conv, 4C → C (project back)
6. **Residual + LayerScale γ**: Learnable per-channel scaling of the residual

**What it does:** Drop-in replacement for ResidualBlock. The key differences vs
ResidualBlock:

- 7×7 depthwise conv gives much larger receptive field per layer (7×7 vs 5×5 for
  two 3×3 convs) at lower compute
- Inverted bottleneck (expand then squeeze) vs standard bottleneck (squeeze then
  expand) — more capacity in the nonlinear part
- LayerNorm + GELU — more stable training, smoother gradients
- Matches ViT performance on ImageNet without any attention mechanism

**For aggregates:** Larger receptive field per block means fewer pooling levels
needed to "see" the context. The inverted bottleneck gives more representational
capacity. Whether this actually helps for 2040×2040 microscopy on a small dataset
is an empirical question.

---

### 9. ECA — Efficient Channel Attention (Wang et al. 2020)

Lightweight alternative to SE. Instead of two FC layers, uses a single 1D
convolution across channels:

```
z = AvgPool(x)                      # global average pool → (B, C, 1, 1)
z = z.squeeze()                     # → (B, C)
s = σ(Conv1D(z, kernel_size=k))     # 1D conv across channel dimension → (B, C)
output = s · x                      # channel-wise scaling
```

Where k (kernel size) is adaptively set based on channel count:
k = ψ(C) = |log₂(C)/γ + b|_odd (typically k=3 or 5).

**SE vs ECA:**

- SE: z → FC(C→C/r) → ReLU → FC(C/r→C) → σ. Parameters: 2C²/r
- ECA: z → Conv1D(kernel=k) → σ. Parameters: k (typically 3-5)

SE models channel interactions through a bottleneck (C→C/16→C), which loses
information. ECA models them via local 1D convolution — each channel interacts
with its k nearest neighbors in the channel ordering. Much fewer parameters, and
the paper shows it performs better.

**What it does:** Same goal as SE (learn which channels to amplify/suppress), but
with a simpler mechanism that avoids the information bottleneck. For aggregates:
if SE helps, ECA should help at least as much with far fewer parameters.

---

## Summary Table

| Module | What it learns | Dimension | Params added | Year |
|--------|---------------|-----------|--------------|------|
| **DoubleConv** | Local spatial patterns | Spatial (5×5) | Baseline | 2015 |
| **ResidualBlock** | Residual spatial + gradient highway | Spatial (5×5) | ~same + 1×1 skip | 2016 |
| **AttentionGate** | Where in space to pass skip features | Spatial (per-pixel) | Small (~2 convs) | 2018 |
| **SE** | Which channels matter | Channel-only | 2C²/r (small) | 2018 |
| **CBAM** | Which channels + where spatially | Channel + Spatial | SE + one 7×7 conv | 2018 |
| **ASPP** | Multi-scale spatial context | Spatial (multi-scale) | ~4× one conv layer | 2017 |
| **UNet++** | Gradual semantic bridging of skips | Structural | ~2-3× skip path | 2020 |
| **ConvNeXt** | Local spatial (wider RF, modern design) | Spatial (7×7) | ~same as Residual | 2022 |
| **ECA** | Which channels matter (no bottleneck) | Channel-only | k ≈ 3-5 params | 2020 |

---

## Redundancy Analysis

### SE vs CBAM vs ECA — three channel attention mechanisms

These three solve the same problem (channel recalibration) with increasing
sophistication:

- **SE** (2018): FC bottleneck. Simple, well-understood, widely cited.
- **CBAM** (2018): SE + max-pool + spatial attention. Strictly more expressive
  than SE, but the spatial attention component overlaps with AttentionGate (both
  learn per-pixel spatial weights).
- **ECA** (2020): 1D conv, no bottleneck. Fewer parameters than SE, reportedly
  better performance. Newer and cleaner.

**Redundancy:** Testing all three in an ablation study would be noisy — the
differences are small and hard to attribute clearly. SE is the simplest reference;
ECA is the modern improvement. CBAM's spatial attention overlaps with
AttentionGate's role.

**Recommendation:** Keep **SE** as the classical reference and **ECA** as its
modern replacement. Drop **CBAM** from the ablation study — its channel attention
is covered by SE/ECA, and its spatial attention is covered by AttentionGate.
Keep the CBAM implementation in the codebase (it works, it's tested) but don't
include it as a separate ablation variant.

### AttentionGate vs CBAM spatial attention

Both learn spatial weights, but at different points:

- **AttentionGate**: On skip connections, guided by decoder context.
- **CBAM spatial**: On feature maps within each block, self-guided.

AttentionGate is more targeted (it uses decoder semantics to gate encoder
features) and adds minimal parameters. CBAM spatial is more general but redundant
when AttentionGate is already present.

### DoubleConv vs ResidualBlock vs ConvNeXt — three encoder block designs

These are mutually exclusive (you pick one for the encoder/decoder):

- **DoubleConv**: Baseline, no skip connection.
- **ResidualBlock**: + identity shortcut. Well-proven.
- **ConvNeXt**: Modern design (7×7 depthwise, LayerNorm, GELU, inverted
  bottleneck). Represents current state-of-the-art in pure conv design.

All three should stay — they form a clear progression for the ablation study.

---

## Recommended Ablation Variants (7 models)

A clean incremental story with one module added per step:

| # | Name | Change vs previous | Modules active |
|---|------|--------------------|----------------|
| 1 | **Baseline UNet** | — | DoubleConv encoder/decoder |
| 2 | **ResUNet** | +Residual blocks | ResidualBlock encoder/decoder |
| 3 | **Attention ResUNet** | +Attention gates on skips | Residual + AttentionGate |
| 4 | **SE Attention ResUNet** | +Channel attention | Residual + AttentionGate + SE |
| 5 | **ASPP SE Attention ResUNet** | +Multi-scale bridge | Residual + AttentionGate + SE + ASPP bridge |
| 6 | **UNet++** | Dense nested skips (structural change) | ResidualBlock + dense skips |
| 7 | **ConvNeXt UNet** | Modern encoder blocks | ConvNeXt encoder + AttentionGate |

**Rationale:**

- **Variants 1-5** form a strict incremental ablation: each adds exactly one
  module. The ablation table in the paper shows the delta from each addition.
- **SE replaces CBAM** in the incremental chain — simpler, and CBAM's spatial
  attention overlaps with AttentionGate.
- **ECA** is tested as a swap for SE in variant 4 (same position, different
  mechanism) — reported separately as "SE vs ECA comparison" rather than a
  separate ablation row. This avoids bloating the main table while still
  providing the comparison.
- **Variants 6-7** are structural alternatives (not incremental additions) — they
  represent different architectural philosophies (dense connectivity vs modern
  convolutions). They are compared against the best of variants 1-5.
- **Deep supervision** is orthogonal (training technique). Test it on the best
  variant from 1-5 and report as a training ablation, not an architecture
  variant.

This gives 7 main rows in the benchmark table, plus 2 side comparisons
(SE vs ECA, ± deep supervision) — enough for a thorough paper without
combinatorial explosion.
