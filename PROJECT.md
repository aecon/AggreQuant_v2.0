# AggreQuant v2 - Project Documentation

> **Purpose**: This document contains all information needed to continue the AggreQuant v2 refactoring project. It includes analysis of existing codebases, literature review, architectural decisions, and detailed implementation tasks.

> **Author**: Athena Economides

> **Working Directory**: `/home/athena/1_CODES/AggreQuant`
> All development and refactoring should be done in this directory.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Existing Codebase Analysis](#2-existing-codebase-analysis)
3. [Literature Review](#3-literature-review)
4. [Architecture Design](#4-architecture-design)
5. [Implementation Phases](#5-implementation-phases)
6. [Technical Specifications](#6-technical-specifications)
7. [Code Style Guidelines](#7-code-style-guidelines)
8. [Dependencies](#8-dependencies)
9. [Testing Strategy](#9-testing-strategy)
10. [References](#10-references)

---

## 1. Project Overview

### 1.1 What is AggreQuant?

AggreQuant is an automated image analysis software for High Content Screening (HCS) plates that quantifies aggregate-positive cells. It processes multi-well microplates (96 or 384 wells) with multiple fields of view per well and 3 imaging channels:
- **Nuclei** (Blue, 390nm)
- **Cells** (FarRed, 640nm)
- **Aggregates** (Green, 473nm) - α-synuclein protein aggregates

### 1.2 Dual Goals

The refactored codebase has **two distinct goals**:

| Goal | Description | Location |
|------|-------------|----------|
| **Goal 1: Image Analysis Pipeline** | Segment CRISPR screen images and quantify aggregates | `aggrequant/` (main package) |
| **Goal 2: NN Development** | Implement and benchmark neural networks for aggregate segmentation | `aggrequant/nn/` (nested subpackage) |

### 1.3 Refactoring Goals

1. **PyTorch Migration**: Replace TensorFlow with PyTorch for all neural network components
2. **Modular Design**: Create extensible architecture for easy addition of new segmentation methods
3. **State-of-the-Art Models**: Implement and benchmark modern UNet variants
4. **Biologist-Friendly GUI**: Interactive plate well selector for control assignment
5. **Focus Quality Metrics**: Integrate blur detection for data quality control
6. **Comprehensive Export**: Statistics in formats suitable for post-processing

### 1.4 Scientific Contribution

**Novel contribution**: First systematic benchmark of deep learning architectures for α-synuclein aggregate segmentation in live-cell fluorescence HCS microscopy data.

### 1.5 Target Users

- Biology academic researchers with bioinformatics background
- Users should be able to run analysis via GUI without coding
- Code should be readable and understandable for those who want to customize

---

## 2. Existing Codebase Analysis

### 2.1 AggreQuant (Main Repository)

**Location**: `/home/athena/1_CODES/AggreQuant`

**Structure**:
```
AggreQuant/
├── main.py                     # Single plate processing entry point
├── main_multiplate.py          # Multi-plate batch processing
├── applications/
│   └── setup.yml               # User configuration (YAML)
├── processing/
│   ├── pipeline.py             # Main processing workflow
│   ├── nuclei.py               # StarDist nuclei segmentation
│   ├── cells.py                # Cellpose or distance-intensity cell segmentation
│   ├── aggregates.py           # UNet or filter-based aggregate segmentation
│   ├── quantification.py       # QoI computation
│   └── weights_best.keras      # Pre-trained UNet weights (TensorFlow)
├── statistics/
│   ├── statistics.py           # Per-well and plate statistics
│   ├── plate.py                # Plate data structure
│   └── diagnostics.py          # Montage generation
└── utils/
    ├── dataset.py              # Dataset class for configuration
    ├── yaml_reader.py          # YAML config parsing
    └── printer.py              # Logging utilities
```

**Key Files to Understand**:

1. **`processing/aggregates.py`** (lines 1-349):
   - `segment_aggregates_filters()`: Filter-based segmentation using normalized intensity
   - `segment_aggregates_UNet()`: UNet inference with sliding window patches
   - Parameters: `patch_size=128`, `stride=32`, `probability_threshold=0.7`
   - Patch stitching with overlap averaging

2. **`processing/pipeline.py`** (lines 1-79):
   - Loads StarDist, Cellpose, and UNet models
   - Processes image triplets sequentially
   - GPU memory management for TensorFlow

3. **`utils/dataset.py`** (lines 1-330):
   - Parses YAML configuration
   - Manages file paths and output directories
   - Control well handling

**Current Segmentation Methods**:

| Structure | Method | Implementation |
|-----------|--------|----------------|
| Nuclei | StarDist 2D | Pre-trained `2D_versatile_fluo` model |
| Cells | Cellpose | Pre-trained `cyto2` model |
| Cells | Distance-Intensity | Custom algorithmic (CellProfiler-inspired) |
| Aggregates | UNet | Custom trained, TensorFlow/Keras |
| Aggregates | Filter-based | Normalized intensity thresholding |

**Current QoI Metrics** (from `processing/quantification.py`):
- Percentage_Of_AggregatePositive_Cells
- Number_Of_Cells_Per_Image
- Percentage_Area_Aggregates (over cell area)
- AreaOfCells
- Percentage_Ambiguous_Aggregates
- Number_Aggregates_Per_Image
- Avg_Number_Aggregates_Per_AggPositive_Cell

---

### 2.2 Vangelis_aSyn_aggregate_detection

**Location**: `/home/athena/1_CODES/Vangelis_aSyn_aggregate_detection`

**Key Contribution**: Focus/blur quality detection

**Structure**:
```
Vangelis_aSyn_aggregate_detection/
├── bluriness.py                # Focus metrics implementation
└── process.py                  # Processing pipeline with blur masking
```

**Focus Metrics** (from `bluriness.py`):

```python
# 5 focus metrics implemented:

def variance_of_laplacian(patch):
    """PRIMARY METRIC - most reliable for blur detection"""
    lap = cv2.Laplacian(patch, cv2.CV_64F)
    return lap.var()

def laplace_energy(patch):
    """Mean of squared Laplacian"""
    lap = cv2.Laplacian(patch, cv2.CV_64F)
    return np.mean(lap * lap)

def sobel_metric(patch):
    """Gradient magnitude"""
    sobelx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(sobelx**2 + sobely**2))

def brenner_metric(patch):
    """Sum of squared pixel differences (2 pixels apart)"""
    diff = patch[2:, :] - patch[:-2, :]
    return np.sum(diff * diff)

def focus_score(patch):
    """Variance-to-mean ratio"""
    return np.nanvar(patch) / (np.nanmean(patch) + eps)
```

**Blur Detection Parameters**:
- `patch_size = (40, 40)` - non-overlapping patches
- `blur_threshold = 15` - for Variance of Laplacian
- Patches below threshold are marked as blurry

**Key Functions**:
- `compute_patch_focus_maps()`: Returns dict of 5 metric maps
- `mask_blurry_patches()`: Creates binary mask of blurry regions
- `compute_image_blur()`: Wrapper for full workflow

**Exported Metrics from process.py**:
- NumberCellsTotal / NumberCellsMasked
- PercPosCellsTotal / PercPosCellsMasked
- AreaCellsMasked, AreaAggMasked
- IntensityAggMasked
- **PercAreaMaskedOut** - key metric for filtering

---

### 2.3 HCS-AggreQuant_private

**Location**: `/home/athena/1_CODES/HCS-AggreQuant_private`

**Key Contributions**: Extended features, GUI prototype, UNet training infrastructure

**Structure** (relevant parts):
```
HCS-AggreQuant_private/
├── prototype/
│   ├── gui/
│   │   └── simple_gui.py           # Tkinter GUI prototype
│   └── quality_control/            # Additional quality metrics
├── unet_aggregates/
│   ├── python/
│   │   └── src/
│   │       ├── unet.py             # UNet architectures
│   │       ├── losses.py           # Loss functions
│   │       └── data.py             # Data loading
│   └── python_NN_w_generators_2024_08_22/
│       └── src/
│           ├── unet.py             # More architectures
│           ├── blocks.py           # NN building blocks
│           ├── augmentation.py     # Data augmentation
│           └── losses.py           # Loss functions
```

**UNet Architectures Available** (from `unet.py`):
1. `build_unet_Ronneberger2015_BatchNorm` - Original UNet with BatchNorm
2. `build_ResUNet_Zhang2018` - Residual UNet
3. `build_unet_TrailMap2020` - TrailMap-adapted UNet
4. `build_unet_TrailMap2020_same` - Same padding variant
5. `build_unet_TrailMap2020_same_SpatialDropout` - With dropout
6. `mitoUNet`, `Attention_UNet`, `Attention_ResUNet` - From blocks.py

**Current Data Augmentation** (from `augmentation.py`):
```python
# Using Keras ImageDataGenerator:
augmentation_types = {
    "None": {},
    "translate": {
        "width_shift_range": 0.1,
        "height_shift_range": 0.1,
        "horizontal_flip": True,
        "vertical_flip": True,
        "zoom_range": 0.2,
        "fill_mode": 'reflect',
    },
    "rotate": {... + "rotation_range": 40},
    "shear": {... + "shear_range": 30},
    "elastic": {... + elasticdeform library},
}

# Additional transforms:
# - adjust_contrast(): random contrast scaling
# - rescale_img(): zero mean, unit std, then [0,1]
# - elastic_transformation(): using elasticdeform library
```

**Loss Functions** (from `losses.py`):
```python
def weighted_binary_crossentropy_vector(y_true, y_pred):
    """
    BCE weighted by inverse class area.
    Handles class imbalance (aggregates are sparse).
    """
    # Computes: background_weight=1, aggregate_weight=area_bg/area_agg
```

**GUI Prototype** (from `simple_gui.py`):
- Basic Tkinter interface
- Directory browser
- Channel label text entries
- Very minimal - needs complete redesign

---

## 3. Literature Review

### 3.1 α-Synuclein Aggregate Segmentation

**Key Finding**: No published work on DL-based α-synuclein aggregate segmentation in fluorescence HCS microscopy.

**Related Work**:
- [Weakly Supervised Segmentation of Alpha-Synuclein Aggregates (Dec 2025)](https://arxiv.org/html/2511.16268)
  - Uses Vision Transformer + CRF
  - **Brightfield histopathology** (NOT fluorescence HCS)
  - Different imaging modality, different approach

**Conclusion**: Our work will be **novel** - first systematic benchmark for this specific application.

### 3.2 State-of-the-Art UNet Architectures (2024-2025)

Based on comprehensive literature review:

| Architecture | Key Innovation | Parameters | Performance | Priority |
|--------------|----------------|------------|-------------|----------|
| **UNet** | Skip connections (baseline) | ~7.8M | Reference | **Must have** |
| **UNet++** | Nested skip connections | ~9M | +3.9 IoU over UNet | **Must have** |
| **ResUNet** | Residual connections | ~8M | Better gradient flow | **Must have** |
| **Attention UNet** | Attention gates | ~8.7M | Focus on relevant features | **Must have** |
| **ResUNet + Deep Supervision** | Multi-scale loss | ~8M | 0.9498 Dice | **Recommended** |
| **TransUNet** | Transformer encoder | ~105M | +1-4% Dice over nnUNet | **If GPU allows** |
| **UltraLightUNet** | Multi-kernel lightweight | **0.316M** | SOTA on 12 benchmarks | **Recommended** |

**Sources**:
- [TransUNet (2024)](https://www.sciencedirect.com/science/article/pii/S1361841524002056)
- [UltraLightUNet (2024)](https://openreview.net/forum?id=BefqqrgdZ1)
- [UNet++ (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/)
- [Comprehensive UNet Review (2025)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70019)
- [ResUNet with Deep Supervision (2025)](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1593016/full)

**For Microscopy/Cell Segmentation Specifically**:
- [CellSegUNet](https://link.springer.com/article/10.1007/s00521-023-09374-3) - Combines UNet++ and ResUNet with attention, achieved 0.970 Dice on DSB
- [State-of-the-Art Deep Learning for Microscopy (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679639/)

### 3.3 Modular Architecture Approach

Instead of implementing separate full architectures, we use a **modular/compositional approach** where:
1. A **base UNet** provides the encoder-decoder structure
2. **Pluggable modules** can be enabled/disabled via configuration
3. This allows systematic A/B testing of individual improvements

```
┌─────────────────────────────────────────────────────────────────┐
│                      Configurable UNet                          │
├─────────────────────────────────────────────────────────────────┤
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│   │   Encoder   │────▶│   Bridge    │────▶│   Decoder   │      │
│   └─────────────┘     └─────────────┘     └─────────────┘      │
│         │              Skip Connections          ▲              │
│         └────────────────────────────────────────┘              │
│                                                                 │
│   Pluggable Modules:                                            │
│   [Residual] [Attention] [SE] [CBAM] [DeepSup] [Transformer]   │
└─────────────────────────────────────────────────────────────────┘
```

**Directory Structure**:
```
aggrequant/nn/architectures/
├── __init__.py
├── factory.py              # Model creation from config
│
├── blocks/                 # Pluggable building blocks
│   ├── __init__.py
│   ├── conv.py             # Basic conv blocks (single, double)
│   ├── residual.py         # Residual blocks (ResNet-style)
│   ├── attention.py        # Attention gates (Attention U-Net)
│   ├── se.py               # Squeeze-and-Excitation blocks
│   ├── cbam.py             # CBAM (Channel + Spatial attention)
│   ├── aspp.py             # Atrous Spatial Pyramid Pooling
│   └── transformer.py      # Transformer encoder (for TransUNet)
│
├── unet.py                 # Modular UNet class
│
└── configs/                # Pre-defined configurations
    ├── __init__.py
    └── presets.py          # All benchmark configurations
```

### 3.4 Pluggable Modules to Implement

| Module | File | Purpose | Reference |
|--------|------|---------|-----------|
| **Double Conv** | `conv.py` | Basic UNet block (2x Conv-BN-ReLU) | Ronneberger 2015 |
| **Residual Block** | `residual.py` | Skip connection within block | ResNet, ResUNet |
| **Attention Gate** | `attention.py` | Focus on relevant skip features | Attention U-Net |
| **SE Block** | `se.py` | Channel-wise recalibration | SENet |
| **CBAM** | `cbam.py` | Channel + Spatial attention | CBAM paper |
| **ASPP** | `aspp.py` | Multi-scale context in bridge | DeepLab |
| **Transformer** | `transformer.py` | Global context encoder | TransUNet |
| **Deep Supervision** | Built into UNet | Multi-scale loss | UNet++ |

### 3.5 Benchmark Configurations

```python
# Pre-defined configurations for systematic benchmarking
BENCHMARK_CONFIGS = {
    # ═══════════════════════════════════════════════════════════
    # BASELINE
    # ═══════════════════════════════════════════════════════════
    "unet_baseline": {
        "encoder_block": "double_conv",
        "decoder_block": "double_conv",
        "bridge": "double_conv",
        "use_residual": False,
        "use_attention_gates": False,
        "use_se": False,
        "use_cbam": False,
        "use_deep_supervision": False,
        "features": [64, 128, 256, 512],
    },

    # ═══════════════════════════════════════════════════════════
    # SINGLE MODULE ADDITIONS (isolate effect of each)
    # ═══════════════════════════════════════════════════════════
    "unet_residual": {
        "encoder_block": "residual",
        "decoder_block": "residual",
        "use_residual": True,
        # ... rest same as baseline
    },

    "unet_attention": {
        "use_attention_gates": True,
        # ... rest same as baseline
    },

    "unet_se": {
        "use_se": True,  # SE block after each conv block
        # ... rest same as baseline
    },

    "unet_cbam": {
        "use_cbam": True,  # CBAM instead of SE
        # ... rest same as baseline
    },

    "unet_deep_supervision": {
        "use_deep_supervision": True,
        # ... rest same as baseline
    },

    # ═══════════════════════════════════════════════════════════
    # COMBINATIONS (best modules together)
    # ═══════════════════════════════════════════════════════════
    "unet_res_attention": {
        "encoder_block": "residual",
        "decoder_block": "residual",
        "use_residual": True,
        "use_attention_gates": True,
    },

    "unet_res_se_attention": {
        "encoder_block": "residual",
        "decoder_block": "residual",
        "use_residual": True,
        "use_se": True,
        "use_attention_gates": True,
    },

    "unet_full": {
        "encoder_block": "residual",
        "decoder_block": "residual",
        "use_residual": True,
        "use_se": True,
        "use_attention_gates": True,
        "use_deep_supervision": True,
    },

    # ═══════════════════════════════════════════════════════════
    # TRANSFORMER-BASED (if GPU allows)
    # ═══════════════════════════════════════════════════════════
    "transunet": {
        "bridge": "transformer",
        "transformer_layers": 6,
        "transformer_heads": 8,
        "use_attention_gates": True,
    },

    # ═══════════════════════════════════════════════════════════
    # LIGHTWEIGHT (for deployment)
    # ═══════════════════════════════════════════════════════════
    "unet_light": {
        "features": [32, 64, 128, 256],  # Reduced features
        "encoder_block": "double_conv",
    },
}
```

### 3.6 Modular UNet Implementation Sketch

```python
# aggrequant/nn/architectures/unet.py
"""Modular UNet with pluggable components."""

import torch
import torch.nn as nn
from .blocks import (
    DoubleConv, ResidualBlock, AttentionGate,
    SEBlock, CBAM, TransformerEncoder
)

class ModularUNet(nn.Module):
    """
    Configurable UNet with pluggable modules.

    Arguments:
        in_channels: Number of input channels (1 for grayscale)
        out_channels: Number of output channels (1 for binary seg)
        features: List of feature counts per level [64, 128, 256, 512]
        encoder_block: "double_conv" or "residual"
        decoder_block: "double_conv" or "residual"
        bridge: "double_conv", "aspp", or "transformer"
        use_attention_gates: Add attention gates to skip connections
        use_se: Add SE blocks after conv blocks
        use_cbam: Add CBAM blocks (mutually exclusive with SE)
        use_deep_supervision: Return multi-scale outputs for deep supervision
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        features: list = [64, 128, 256, 512],
        encoder_block: str = "double_conv",
        decoder_block: str = "double_conv",
        bridge: str = "double_conv",
        use_attention_gates: bool = False,
        use_se: bool = False,
        use_cbam: bool = False,
        use_deep_supervision: bool = False,
        **kwargs
    ):
        super().__init__()

        self.use_deep_supervision = use_deep_supervision
        self.use_attention_gates = use_attention_gates

        # Select block types
        EncoderBlock = ResidualBlock if encoder_block == "residual" else DoubleConv
        DecoderBlock = ResidualBlock if decoder_block == "residual" else DoubleConv

        # Build encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for feature in features:
            self.encoders.append(self._make_block(EncoderBlock, in_ch, feature, use_se, use_cbam))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = feature

        # Build bridge
        self.bridge = self._make_bridge(bridge, features[-1], features[-1] * 2, **kwargs)

        # Build decoder
        self.decoders = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        if use_attention_gates:
            self.attention_gates = nn.ModuleList()

        reversed_features = list(reversed(features))
        in_ch = features[-1] * 2
        for feature in reversed_features:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, feature, kernel_size=2, stride=2))
            self.decoders.append(self._make_block(DecoderBlock, feature * 2, feature, use_se, use_cbam))
            if use_attention_gates:
                self.attention_gates.append(AttentionGate(feature, feature, feature // 2))
            in_ch = feature

        # Output
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

        # Deep supervision outputs
        if use_deep_supervision:
            self.deep_supervision_heads = nn.ModuleList([
                nn.Conv2d(f, out_channels, kernel_size=1) for f in reversed_features[:-1]
            ])

    def forward(self, x):
        # Encoder path
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        # Bridge
        x = self.bridge(x)

        # Decoder path
        skip_connections = skip_connections[::-1]
        deep_outputs = []

        for i, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoders)):
            x = upconv(x)
            skip = skip_connections[i]

            # Apply attention gate if enabled
            if self.use_attention_gates:
                skip = self.attention_gates[i](x, skip)

            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

            # Collect deep supervision outputs
            if self.use_deep_supervision and i < len(self.deep_supervision_heads):
                deep_outputs.append(self.deep_supervision_heads[i](x))

        output = self.final_conv(x)

        if self.use_deep_supervision and self.training:
            return output, deep_outputs
        return output
```

### 3.7 Modern Data Augmentation

**Best Practices** (from nnU-Net and microscopy literature):
- Spatial: rotation, flip, scale, shift, elastic deformation
- Intensity: brightness, contrast, gamma, CLAHE
- Noise: Gaussian, multiplicative (simulates camera noise)
- Blur: Gaussian, motion (simulates focus variations)

**Library**: Albumentations (more comprehensive than Keras ImageDataGenerator)

### 3.8 Modern Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| BCE | -[y·log(p) + (1-y)·log(1-p)] | Standard |
| Dice | 1 - 2·|X∩Y|/(|X|+|Y|) | Class imbalance |
| Dice+BCE | α·Dice + β·BCE | Combined benefits |
| Focal | -α·(1-p)^γ·log(p) | Hard examples |
| Tversky | 1 - TP/(TP+α·FN+β·FP) | FP/FN trade-off |

---

## 4. Architecture Design

### 4.1 Software Engineering Review

The architecture was reviewed by a software engineering expert. Key recommendations incorporated:

| Issue | Problem | Solution |
|-------|---------|----------|
| Two top-level packages | Packaging ambiguity | Nest `nn/` under `aggrequant/` |
| `io/` module name | Shadows Python built-in | Rename to `loaders/` |
| Scripts inside package | Hard to invoke | Top-level `scripts/` or CLI entry points |
| Missing utilities | Code duplication | Add `common/` module |
| Missing data management | No train/val/test tracking | Add `nn/data/` module |
| No results container | Scattered outputs | Add `quantification/results.py` |

### 4.2 Final Directory Structure

```
AggreQuant/
│
├── aggrequant/                     # Single installable package
│   ├── __init__.py
│   │
│   ├── common/                     # Shared utilities
│   │   ├── __init__.py
│   │   ├── image_utils.py          # Normalization, dtype conversion
│   │   ├── geometry.py             # Coordinate transforms, ROI handling
│   │   └── logging.py              # Consistent logging setup
│   │
│   ├── segmentation/               # Segmentation backends
│   │   ├── __init__.py
│   │   ├── base.py                 # Segmenter Protocol/ABC
│   │   ├── nuclei/
│   │   │   ├── __init__.py
│   │   │   └── stardist.py         # StarDist wrapper
│   │   ├── cells/
│   │   │   ├── __init__.py
│   │   │   ├── cellpose.py         # Cellpose wrapper
│   │   │   └── distance_intensity.py
│   │   └── aggregates/
│   │       ├── __init__.py
│   │       ├── filter_based.py     # Filter-based method
│   │       └── neural_network.py   # NN segmenter (uses aggrequant.nn)
│   │
│   ├── quality/                    # Image quality assessment
│   │   ├── __init__.py
│   │   ├── focus.py                # Blur/focus detection
│   │   └── visualization.py        # Focus map plotting
│   │
│   ├── quantification/             # Analysis & metrics
│   │   ├── __init__.py
│   │   ├── measurements.py         # QoI calculations
│   │   ├── colocalization.py       # Aggregate-cell colocalization
│   │   └── results.py              # Results container dataclass
│   │
│   ├── statistics/                 # Statistical analysis
│   │   ├── __init__.py
│   │   ├── well_stats.py           # Field → Well aggregation
│   │   ├── controls.py             # SSMD, control comparisons
│   │   └── export.py               # CSV, Parquet, Excel export
│   │
│   ├── loaders/                    # I/O utilities (renamed from io/)
│   │   ├── __init__.py
│   │   ├── images.py               # TIFF loading
│   │   ├── config.py               # YAML config + Pydantic validation
│   │   └── plate.py                # Plate/Well data structures
│   │
│   ├── nn/                         # NN development (nested subpackage)
│   │   ├── __init__.py
│   │   │
│   │   ├── architectures/          # Modular architecture system
│   │   │   ├── __init__.py
│   │   │   ├── unet.py             # ModularUNet class
│   │   │   ├── factory.py          # Model creation from config
│   │   │   │
│   │   │   ├── blocks/             # Pluggable building blocks
│   │   │   │   ├── __init__.py
│   │   │   │   ├── conv.py         # DoubleConv, SingleConv
│   │   │   │   ├── residual.py     # ResidualBlock
│   │   │   │   ├── attention.py    # AttentionGate
│   │   │   │   ├── se.py           # Squeeze-and-Excitation
│   │   │   │   ├── cbam.py         # CBAM block
│   │   │   │   ├── aspp.py         # ASPP bridge
│   │   │   │   └── transformer.py  # TransformerEncoder
│   │   │   │
│   │   │   └── configs/            # Benchmark configurations
│   │   │       ├── __init__.py
│   │   │       └── presets.py      # BENCHMARK_CONFIGS dict
│   │   │
│   │   ├── data/                   # Dataset management
│   │   │   ├── __init__.py
│   │   │   ├── dataset.py          # PyTorch Dataset
│   │   │   ├── augmentation.py     # Albumentations pipelines
│   │   │   └── splits.py           # Train/val/test management
│   │   │
│   │   ├── training/               # Training infrastructure
│   │   │   ├── __init__.py
│   │   │   ├── losses.py           # Loss functions
│   │   │   ├── trainer.py          # Training loop
│   │   │   └── experiment.py       # Experiment tracking
│   │   │
│   │   ├── evaluation/             # Model evaluation
│   │   │   ├── __init__.py
│   │   │   ├── metrics.py          # Dice, IoU, etc.
│   │   │   └── benchmark.py        # Architecture comparison
│   │   │
│   │   └── cli.py                  # NN-specific CLI commands
│   │
│   ├── pipeline.py                 # Main processing orchestrator
│   └── cli.py                      # Main CLI entry point
│
├── gui/                            # GUI application (optional install)
│   ├── __init__.py
│   ├── app.py                      # Main window
│   └── widgets/
│       ├── __init__.py
│       ├── plate_selector.py       # Interactive plate grid
│       ├── control_panel.py        # Control type assignment
│       ├── segmentation_preview.py # Live preview
│       ├── quality_settings.py     # Blur threshold controls
│       └── progress_panel.py       # Progress tracking
│
├── scripts/                        # Development/utility scripts
│   ├── prepare_dataset.py          # Prepare training data
│   ├── train_model.py              # Train a model
│   ├── run_benchmark.py            # Compare architectures
│   └── visualize_predictions.py    # Visualize results
│
├── configs/                        # Example YAML configs
│   ├── pipeline_default.yaml       # Default pipeline config
│   └── training/
│       ├── unet.yaml
│       ├── unet_plusplus.yaml
│       └── attention_unet.yaml
│
├── weights/                        # Pre-trained model weights
│   └── .gitkeep
│
├── data/                           # Training data (gitignored)
│   └── .gitkeep
│
├── experiments/                    # Training outputs (gitignored)
│   └── .gitkeep
│
├── tests/
│   ├── unit/
│   │   ├── test_focus_metrics.py
│   │   ├── test_architectures.py
│   │   ├── test_losses.py
│   │   └── test_augmentation.py
│   └── integration/
│       ├── test_pipeline.py
│       └── test_training.py
│
├── main.py                         # CLI entry point
├── main_gui.py                     # GUI entry point
├── pyproject.toml                  # Package configuration
├── PROJECT.md                      # This documentation
└── README.md                       # User-facing documentation
```

### 4.3 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Single package (`aggrequant/`) | Simpler installation, clear namespace |
| `nn/` nested under main package | Can use `pip install aggrequant[nn-training]` for optional deps |
| `loaders/` instead of `io/` | Avoids shadowing Python built-in |
| `common/` module | Prevents code duplication |
| `nn/data/` module | Proper dataset and split management |
| `results.py` dataclass | Structured output for post-processing |
| Top-level `scripts/` | Easy to run development scripts |
| `experiments/` directory | Track training runs locally |

### 4.4 Model Registry Pattern

```python
# aggrequant/nn/architectures/factory.py
"""Simple registry for model selection."""

from typing import Dict, Type, Callable
import torch.nn as nn

ARCHITECTURES: Dict[str, Callable[..., nn.Module]] = {}

def register(name: str):
    """Decorator to register an architecture."""
    def decorator(cls):
        ARCHITECTURES[name] = cls
        return cls
    return decorator

def create_model(name: str, **kwargs) -> nn.Module:
    """Create model by name."""
    if name not in ARCHITECTURES:
        available = list(ARCHITECTURES.keys())
        raise ValueError(f"Unknown architecture: {name}. Available: {available}")
    return ARCHITECTURES[name](**kwargs)

def list_architectures() -> list:
    """List available architectures."""
    return list(ARCHITECTURES.keys())
```

### 4.5 Export Data Schema

```python
# Complete export columns
EXPORT_COLUMNS = {
    # Identifiers
    'plate_name': str,
    'well_id': str,               # e.g., "A-01"
    'row': str,                   # e.g., "A"
    'column': int,                # e.g., 1
    'field': int,                 # e.g., 1-9

    # Control info
    'control_type': str,          # "NT", "RAB13", or None

    # Focus Quality Metrics
    'focus_variance_laplacian_mean': float,
    'focus_variance_laplacian_min': float,
    'focus_pct_patches_blurry': float,
    'focus_pct_area_blurry': float,
    'focus_is_likely_blurry': bool,

    # Cell Metrics
    'n_cells': int,
    'n_nuclei': int,
    'total_cell_area_px': float,

    # Aggregate Metrics (TOTAL)
    'n_aggregate_positive_cells': int,
    'pct_aggregate_positive_cells': float,
    'n_aggregates': int,
    'total_aggregate_area_px': float,
    'pct_aggregate_area_over_cell': float,
    'avg_aggregates_per_positive_cell': float,
    'pct_ambiguous_aggregates': float,

    # Aggregate Metrics (MASKED - excluding blur)
    'n_cells_masked': int,
    'n_aggregate_positive_cells_masked': int,
    'pct_aggregate_positive_cells_masked': float,
    'total_cell_area_masked_px': float,
    'total_aggregate_area_masked_px': float,

    # Metadata
    'segmentation_method': str,
    'model_weights': str,
    'blur_threshold_used': float,
    'timestamp': 'datetime64[ns]',
}
```

---

## 5. Implementation Phases

### Phase 1: Core Package Structure + Quality Module

**Priority**: HIGH | **Status**: IN PROGRESS

#### Task 1.1: Create Package Structure
- [x] Create directory structure as specified in Section 4.2
- [x] Create all `__init__.py` files with appropriate exports
- [x] Create `pyproject.toml` with dependencies (Section 8)
- [x] Set up `common/` module with basic utilities

#### Task 1.2: Implement Focus Quality Module
- [x] Port `bluriness.py` to `aggrequant/quality/focus.py`
- [x] Create `FocusMetrics` dataclass for results
- [x] Implement `compute_focus_metrics()` function
- [x] Implement `generate_blur_mask()` function
- [ ] Add visualization utilities (`aggrequant/quality/visualization.py`)
- [x] Write unit tests (`tests/unit/test_focus_metrics.py`) - 47 tests passing

#### Task 1.3: Implement Loaders Module
- [x] Port `utils/dataset.py` to `aggrequant/loaders/config.py`
- [x] Use Pydantic-style dataclasses for configuration validation
- [x] Create `aggrequant/loaders/plate.py` with Plate class
- [x] Create `aggrequant/loaders/images.py` for TIFF loading

#### Task 1.4: Implement Base Segmenter
- [x] Create `aggrequant/segmentation/base.py` with Protocol/ABC
- [x] Define interface: `segment(image) -> labels`

---

### Phase 2: PyTorch NN Module

**Priority**: HIGH | **Status**: COMPLETE

#### Task 2.1: Implement Building Blocks
Files in `aggrequant/nn/architectures/blocks/`:
- [x] `conv.py` - DoubleConv, SingleConv (basic UNet blocks)
- [x] `residual.py` - ResidualBlock with skip connection
- [x] `attention.py` - AttentionGate for skip connections
- [x] `se.py` - Squeeze-and-Excitation block
- [x] `cbam.py` - CBAM (Channel + Spatial attention)
- [x] `aspp.py` - ASPP bridge for multi-scale context

#### Task 2.2: Implement Modular UNet
Files in `aggrequant/nn/architectures/`:
- [x] `unet.py` - ModularUNet class with pluggable components
- [x] `factory.py` - Create model from config dict (18 presets)
- [x] `configs/presets.py` - Pre-defined benchmark configurations

The ModularUNet supports these configuration options:
```python
{
    "encoder_block": "double_conv" | "residual",
    "decoder_block": "double_conv" | "residual",
    "use_attention_gates": bool,
    "use_se": bool,
    "use_cbam": bool,
    "use_deep_supervision": bool,
    "features": [64, 128, 256, 512],
}
```

#### Task 2.3: Implement Data Module
Files in `aggrequant/nn/data/`:
- [x] `dataset.py` - AggregateDataset, PatchDataset, InferenceDataset
- [x] `augmentation.py` - Albumentations pipelines (nnU-Net style)

#### Task 2.4: Implement Training Module
Files in `aggrequant/nn/training/`:
- [x] `losses.py` - DiceLoss, DiceBCELoss, FocalLoss, TverskyLoss, DeepSupervisionLoss
- [x] `trainer.py` - Training loop with checkpointing, early stopping

#### Task 2.5: Implement Evaluation Module
Files in `aggrequant/nn/evaluation/`:
- [x] `metrics.py` - Dice, IoU, Precision, Recall, F1, SegmentationMetrics

---

### Phase 3: GUI Development

**Priority**: HIGH | **Status**: COMPLETE

#### Task 3.1: Set Up GUI Framework
- [x] Choose framework: customtkinter (recommended)
- [x] Create `gui/app.py` main window

#### Task 3.2: Implement Widgets
- [x] `plate_selector.py` - Interactive 96/384-well grid with drag selection
- [x] `control_panel.py` - Add/remove control types, color coding
- [x] `settings_panel.py` - Settings including blur threshold slider
- [x] `progress_panel.py` - Progress bar, log output, run/cancel buttons

#### Task 3.3: Integrate with Pipeline
- [x] Connect GUI widgets with callbacks
- [x] Implement save/load YAML configuration
- [x] Implement background analysis thread with cancel functionality

---

### Phase 4: Segmentation Backends

**Priority**: MEDIUM | **Status**: COMPLETE

#### Task 4.1: Port Nuclei Segmentation
- [x] Create `aggrequant/segmentation/nuclei/stardist.py`
- [x] Wrap StarDist model loading and inference
- [x] Pre-processing (Gaussian denoise, background normalization)
- [x] Post-processing (size exclusion, border separation, border exclusion)

#### Task 4.2: Port Cell Segmentation
- [x] Create `aggrequant/segmentation/cells/cellpose.py`
- [x] Create `aggrequant/segmentation/cells/distance_intensity.py`
- [x] Two-channel input support (cell image + nuclei mask)
- [x] Watershed-based cell splitting

#### Task 4.3: Port Aggregate Segmentation
- [x] Create `aggrequant/segmentation/aggregates/filter_based.py`
- [x] Create `aggrequant/segmentation/aggregates/neural_network.py`
- [x] Sliding window inference with patch stitching
- [x] PyTorch model integration

---

### Phase 5: Statistics and Export

**Priority**: MEDIUM | **Status**: COMPLETE

#### Task 5.1: Implement Quantification
- [x] Port `processing/quantification.py` to `aggrequant/quantification/measurements.py`
- [x] Add blur-masked QoI computation
- [x] Create `results.py` dataclass container (FieldResult, WellResult, PlateResult)

#### Task 5.2: Implement Statistics
- [x] Create `aggrequant/statistics/well_stats.py` - Field to well aggregation
- [x] Create `aggrequant/statistics/controls.py` - SSMD, Z-factor, control comparison
- [x] Create `aggrequant/statistics/export.py` - CSV, Parquet, Excel export

---

### Phase 6: Integration and Polish

**Priority**: LOW

#### Task 6.1: Create Pipeline Orchestrator
- [ ] Implement `aggrequant/pipeline.py`
- [ ] Connect all components
- [ ] Add error handling

#### Task 6.2: Create CLI
- [ ] Implement `aggrequant/cli.py`
- [ ] Implement `aggrequant/nn/cli.py`
- [ ] Add entry points in `pyproject.toml`

#### Task 6.3: Documentation
- [ ] Update README.md
- [ ] Add docstrings to all public functions
- [ ] Create user guide

#### Task 6.4: Testing
- [ ] Unit tests for each module
- [ ] Integration tests
- [ ] Test with real data

---

## 6. Technical Specifications

### 6.1 Image Specifications

| Property | Value |
|----------|-------|
| Format | TIFF (16-bit) |
| Size | 2040 × 2040 pixels (typical) |
| Channels | 3 (separate files) |
| Fields per well | 9-22 |
| Plate formats | 96-well, 384-well |

### 6.2 Neural Network Specifications

| Property | Value |
|----------|-------|
| Framework | PyTorch 2.0+ |
| Input size | 128 × 128 patches |
| Stride | 32 pixels (overlapping) |
| Output | Probability map [0, 1] |
| Threshold | 0.7 (default) |
| Batch size | 64 (adjustable) |

### 6.3 Focus Quality Specifications

| Property | Value |
|----------|-------|
| Patch size | 40 × 40 pixels |
| Overlap | 0 (non-overlapping) |
| Primary metric | Variance of Laplacian |
| Blur threshold | 15 (default) |
| Blurry image flag | >50% patches below threshold |

### 6.4 Configurations to Benchmark

All configurations use `ModularUNet` with different settings:

| Config Name | Key Settings | Purpose |
|-------------|--------------|---------|
| `unet_baseline` | Default double_conv blocks | Baseline reference |
| `unet_residual` | `encoder_block="residual"` | Test residual connections |
| `unet_attention` | `use_attention_gates=True` | Test attention gates |
| `unet_se` | `use_se=True` | Test SE blocks |
| `unet_cbam` | `use_cbam=True` | Test CBAM blocks |
| `unet_deep_supervision` | `use_deep_supervision=True` | Test deep supervision |
| `unet_res_attention` | residual + attention | Combined modules |
| `unet_res_se_attention` | residual + SE + attention | Triple combination |
| `unet_full` | All modules enabled | Maximum features |
| `transunet` | `bridge="transformer"` | Transformer encoder |
| `unet_light` | `features=[32,64,128,256]` | Lightweight variant |

**Benchmarking Strategy**:
1. First test individual modules vs baseline (isolate each improvement)
2. Then test best-performing combinations
3. Finally test transformer-based if GPU allows

---

## 7. Code Style Guidelines

Based on analysis of existing codebase (`/home/athena/1_CODES/AggreQuant`):

### 7.1 File Header

All `.py` files should include the following header in the docstring:

```python
"""
Module description here.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""
```

### 7.2 General Style

```python
# Module-level parameters at top of file (after imports)
PATCH_SIZE = 128
PROBABILITY_THRESHOLD = 0.7

verbose = False
debug = False


def segment_aggregates(image_file, output_files, model, verbose, debug):
    """
    Short description of function.

    Arguments:
        image_file: path to input TIFF image
        output_files: dict with output file paths
        model: loaded PyTorch model
        verbose: print progress messages
        debug: print detailed debug information

    Returns:
        labels: numpy array with instance labels (uint32)
    """
    me = "segment_aggregates"  # Function identifier for logging

    # Load image
    img = skimage.io.imread(image_file, plugin='tifffile')
    if debug:
        print(f"({me}) Image shape: {img.shape}, dtype: {img.dtype}")

    # Processing steps...

    # Assertions for validation
    assert np.min(result) >= 0

    # Save output
    skimage.io.imsave(output_files["labels"], result, plugin='tifffile')

    return result
```

### 7.3 Key Conventions

1. **Naming**: `snake_case` for functions and variables
2. **Logging**: Use `me = "function_name"` pattern with `print(f"({me}) message")`
3. **Flags**: Pass `verbose` and `debug` through function calls
4. **Assertions**: Use `assert` for validation
5. **Docstrings**: Include Arguments and Returns sections
6. **Parameters**: Module-level constants in UPPER_CASE
7. **No over-engineering**: Prefer simple functions over complex class hierarchies
8. **Direct imports**: `import numpy as np`, `import skimage.io`

### 7.4 File Organization

```python
# Standard import order:
import os
import sys

import numpy as np
import torch
import skimage.io

from .module import function

# Module-level parameters
PARAM_NAME = value

# Verbose/debug flags
verbose = False
debug = False


# Functions
def public_function():
    pass


def _private_function():
    pass
```

---

## 8. Dependencies

### 8.1 pyproject.toml

```toml
[project]
name = "aggrequant"
version = "2.0.0"
description = "Automated aggregate quantification for High Content Screening"
requires-python = ">=3.10"
dependencies = [
    # Image processing
    "numpy>=1.24",
    "scikit-image>=0.21",
    "opencv-python-headless>=4.8",
    "tifffile>=2023.7",

    # Data handling
    "pandas>=2.0",
    "pyarrow>=14.0",

    # Configuration
    "pyyaml>=6.0",
    "pydantic>=2.0",

    # CLI
    "click>=8.1",
    "tqdm>=4.65",

    # Visualization
    "matplotlib>=3.7",
]

[project.optional-dependencies]
# For running the segmentation pipeline
pipeline = [
    "torch>=2.0",
    "torchvision>=0.15",
    "stardist>=0.8",
    "cellpose>=3.0",
]

# For training neural networks
nn-training = [
    "torch>=2.0",
    "torchvision>=0.15",
    "segmentation-models-pytorch>=0.3",
    "albumentations>=1.3",
    "tensorboard>=2.15",
]

# For the GUI
gui = [
    "customtkinter>=5.2",
]

# All dependencies
all = [
    "aggrequant[pipeline,nn-training,gui]",
]

# Development
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=23.0",
    "ruff>=0.1",
]

[project.scripts]
aggrequant = "aggrequant.cli:main"
aggrequant-train = "aggrequant.nn.cli:train"
aggrequant-benchmark = "aggrequant.nn.cli:benchmark"
aggrequant-gui = "gui.app:main"
```

### 8.2 Key Library Versions

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | >=2.0 | Deep learning framework |
| segmentation-models-pytorch | >=0.3 | Pre-built architectures |
| StarDist | >=0.8 | Nuclei segmentation |
| Cellpose | >=3.0 | Cell segmentation |
| Albumentations | >=1.3 | Data augmentation |
| customtkinter | >=5.2 | Modern GUI |
| Pydantic | >=2.0 | Configuration validation |

---

## 9. Testing Strategy

### 9.1 Test Structure

```
tests/
├── conftest.py                # Shared fixtures
├── unit/
│   ├── test_focus_metrics.py  # Focus quality module (47 tests)
│   ├── test_plate.py          # Plate/Well data structures
│   ├── test_config.py         # Configuration management
│   ├── test_architectures.py  # NN architectures (30 tests)
│   ├── test_losses.py         # Loss functions (17 tests)
│   ├── test_metrics.py        # Evaluation metrics (27 tests)
│   └── test_gui.py            # GUI widgets (50 tests)
└── integration/
    └── __init__.py
```

**Total: 207+ tests**

### 9.2 Example Test

```python
# tests/unit/test_focus_metrics.py
import numpy as np
import pytest
from aggrequant.quality.focus import (
    variance_of_laplacian,
    compute_focus_metrics,
    FocusMetrics,
)


def test_variance_of_laplacian_sharp():
    """Sharp edges should have high variance of Laplacian."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 255
    vol = variance_of_laplacian(img)
    assert vol > 100


def test_variance_of_laplacian_blurry():
    """Blurry image should have low variance of Laplacian."""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    vol = variance_of_laplacian(img)
    assert vol < 1


def test_compute_focus_metrics_returns_dataclass():
    """compute_focus_metrics should return FocusMetrics dataclass."""
    img = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    result = compute_focus_metrics(img)
    assert isinstance(result, FocusMetrics)
    assert result.n_patches_total > 0
```

---

## 10. References

### 10.1 Code References

| File | Purpose | Key Functions |
|------|---------|---------------|
| `/home/athena/1_CODES/AggreQuant/processing/aggregates.py` | Aggregate segmentation | `segment_aggregates_UNet()`, `segment_aggregates_filters()` |
| `/home/athena/1_CODES/AggreQuant/processing/pipeline.py` | Processing orchestration | `process()`, `_image_triplet()` |
| `/home/athena/1_CODES/AggreQuant/utils/dataset.py` | Configuration | `Dataset` class |
| `/home/athena/1_CODES/Vangelis_aSyn_aggregate_detection/bluriness.py` | Focus metrics | `variance_of_laplacian()`, `compute_patch_focus_maps()` |
| `/home/athena/1_CODES/HCS-AggreQuant_private/unet_aggregates/python_NN_w_generators_2024_08_22/src/unet.py` | UNet architectures | `build_unet()`, `build_ResUNet_Zhang2018()` |
| `/home/athena/1_CODES/HCS-AggreQuant_private/unet_aggregates/python_NN_w_generators_2024_08_22/src/augmentation.py` | Data augmentation | `get_generator()`, `transform()` |

### 10.2 Literature References

**α-Synuclein Aggregate Segmentation**:
- [Weakly Supervised Segmentation of Alpha-Synuclein Aggregates (2025)](https://arxiv.org/html/2511.16268)

**UNet Architectures**:
- [A Comprehensive Review of U-Net and Its Variants (2025)](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/ipr2.70019)
- [TransUNet (2024)](https://www.sciencedirect.com/science/article/pii/S1361841524002056)
- [UltraLightUNet (2024)](https://openreview.net/forum?id=BefqqrgdZ1)
- [UNet++ (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7329239/)
- [ResUNet with Deep Supervision (2025)](https://www.frontiersin.org/journals/medicine/articles/10.3389/fmed.2025.1593016/full)

**Microscopy Segmentation**:
- [CellSegUNet (2023)](https://link.springer.com/article/10.1007/s00521-023-09374-3)
- [State-of-the-Art Deep Learning for Microscopy (2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11679639/)

---

## Quick Start for Continuing Development

**Working Directory**: `/home/athena/1_CODES/AggreQuant`

All development should be done in this directory. The refactored code will replace/extend the existing codebase here.

### Development Environment

**Conda Environment**: `AggreQuant`

```bash
# Activate the environment before any development work
conda activate AggreQuant

# The environment uses Python 3.11 and has core dependencies installed:
# numpy, scikit-image, opencv-python-headless, tifffile, pandas,
# pyarrow, pyyaml, pydantic, click, tqdm, matplotlib

# To install additional dependencies (e.g., for NN training):
pip install torch torchvision albumentations segmentation-models-pytorch

# To install the package in development mode:
pip install -e .
```

**Specialized Agent Available**: Use the `cv-ml-engineer` agent for:
- Implementing neural network architectures (modular UNet, blocks)
- Training infrastructure (losses, trainer, data augmentation)
- Model evaluation and benchmarking
- Performance optimization

1. **Read this document** thoroughly
2. **Explore existing code**:
   ```bash
   # Key files to understand first:
   cat /home/athena/1_CODES/AggreQuant/processing/aggregates.py
   cat /home/athena/1_CODES/Vangelis_aSyn_aggregate_detection/bluriness.py
   ```
3. **Start with Phase 1**: Create package structure and port focus metrics
4. **Follow code style**: Match existing conventions (Section 7)
5. **Test incrementally**: Write tests as you implement

---

---

## Current Progress & Next Steps

### Completed (2026-02-04)

**Environment Setup**:
- [x] Conda environment `AggreQuant` created with Python 3.11
- [x] Core dependencies installed (numpy, scikit-image, opencv-python-headless, tifffile, pandas, pyarrow, pyyaml, pydantic, click, tqdm, matplotlib)

**Phase 1 - Core Package (mostly complete)**:
- [x] `aggrequant/__init__.py` - Package init with version and author
- [x] `aggrequant/common/__init__.py` - Exports for common utilities
- [x] `aggrequant/common/image_utils.py` - normalize_image, to_uint8, to_float32, pad_to_multiple, unpad
- [x] `aggrequant/common/logging.py` - setup_logging, get_logger, SimpleLogger
- [x] `aggrequant/quality/__init__.py` - Exports for quality module
- [x] `aggrequant/quality/focus.py` - FocusMetrics, variance_of_laplacian, compute_focus_metrics, generate_blur_mask
- [x] `aggrequant/segmentation/base.py` - Segmenter Protocol, BaseSegmenter ABC
- [x] `aggrequant/loaders/__init__.py` - Exports for loaders module
- [x] `aggrequant/loaders/config.py` - PipelineConfig, ChannelConfig, SegmentationConfig, QualityConfig, OutputConfig
- [x] `aggrequant/loaders/images.py` - load_tiff, ImageLoader, parse_operetta_filename, parse_imageexpress_filename
- [x] `aggrequant/loaders/plate.py` - Plate, Well, FieldOfView, well_id_to_indices, generate_all_well_ids

### Completed (Phase 1 Bug Fixes & Tests)

**Bug Fixes Applied** (from code-reviewer agent):
- [x] Fixed division by zero in `focus.py` - use percentile normalization
- [x] Extracted `_prepare_image_for_cv2()` helper to eliminate code duplication
- [x] Added well ID validation with bounds checking in `plate.py`
- [x] Added warnings for unparseable filenames in `images.py`
- [x] Fixed tuple/list inconsistency in `QualityConfig`

**Unit Tests Created** (83 tests, all passing):
- [x] `tests/unit/test_focus_metrics.py` - Focus module tests
- [x] `tests/unit/test_plate.py` - Plate/Well data structure tests
- [x] `tests/unit/test_config.py` - Configuration tests including YAML round-trip

### Completed (Phase 2 - NN Module)

**Architectures** (cv-ml-engineer agent):
- [x] `aggrequant/nn/architectures/blocks/` - conv, residual, attention, se, cbam, aspp
- [x] `aggrequant/nn/architectures/unet.py` - ModularUNet with pluggable blocks
- [x] `aggrequant/nn/architectures/factory.py` - Model registry with 18 presets
- [x] `aggrequant/nn/architectures/configs/presets.py` - Benchmark configurations

**Data Pipeline**:
- [x] `aggrequant/nn/data/dataset.py` - AggregateDataset, PatchDataset, InferenceDataset
- [x] `aggrequant/nn/data/augmentation.py` - Albumentations pipelines (nnU-Net style)

**Training**:
- [x] `aggrequant/nn/training/losses.py` - Dice, BCE, Focal, Tversky, DeepSupervision
- [x] `aggrequant/nn/training/trainer.py` - Training loop with checkpointing

**Evaluation**:
- [x] `aggrequant/nn/evaluation/metrics.py` - dice, iou, precision, recall, f1

**Unit Tests** (74 new tests):
- [x] `tests/unit/test_architectures.py` - 30 tests
- [x] `tests/unit/test_losses.py` - 17 tests
- [x] `tests/unit/test_metrics.py` - 27 tests

**GUI Unit Tests** (50 tests):
- [x] `tests/unit/test_gui.py` - 50 tests for GUI components
- [x] `tests/conftest.py` - Shared fixtures for all tests

**Total unit tests: 207+**

### Completed (Phase 3 - GUI Development)

**GUI Framework** (customtkinter):
- [x] `gui/__init__.py` - Package exports (AggreQuantApp, main)
- [x] `gui/app.py` - Main application window with menu, layout, callbacks
- [x] `gui/widgets/__init__.py` - Widget exports
- [x] `gui/widgets/plate_selector.py` - Interactive 96/384-well grid with drag selection
- [x] `gui/widgets/control_panel.py` - Control type assignment (predefined + custom)
- [x] `gui/widgets/settings_panel.py` - Input/output paths, plate format, segmentation method, blur threshold
- [x] `gui/widgets/progress_panel.py` - Progress bar, log output, run/cancel buttons

**GUI Features**:
- Load/save YAML configuration
- Light mode only (white background) for clean appearance
- Background thread for analysis with cancel support
- Color-coded control types: negative (blue), NT (green), custom (orange)
- Rectangular buttons throughout for consistent visual style
- Segmented buttons for plate format (96/384-well) and segmentation method (UNet/Filter-based)

**GUI Launcher**:
- [x] `scripts/run_gui.py` - Launch GUI with `python scripts/run_gui.py`

**GUI Unit Tests** (50 tests):
- [x] `tests/unit/test_gui.py` - Comprehensive tests for all widgets
  - TestPlateSelector: 17 tests (initialization, selection, assignments, callbacks)
  - TestControlPanel: 8 tests (callbacks, custom types, color handling)
  - TestSettingsPanel: 9 tests (settings get/set, segmented buttons)
  - TestProgressPanel: 12 tests (state, progress, logging)
  - TestControlColors: 2 tests (color definitions)
  - TestWellConstants: 1 test (well state colors)
  - TestIntegration: 2 tests (component interactions)
- [x] `tests/conftest.py` - Shared pytest fixtures

### Completed (Phase 4 - Segmentation Backends)

**Nuclei Segmentation**:
- [x] `aggrequant/segmentation/nuclei/stardist.py` - StarDistSegmenter
  - Lazy model loading, pre-processing (Gaussian denoise, background normalization)
  - Post-processing (size exclusion, border separation, border exclusion)
  - `segment()` and `segment_with_seeds()` methods

**Cell Segmentation**:
- [x] `aggrequant/segmentation/cells/cellpose.py` - CellposeSegmenter
  - Two-channel input support (cell image + nuclei mask)
  - GPU support, excludes cells without nuclei
- [x] `aggrequant/segmentation/cells/distance_intensity.py` - DistanceIntensitySegmenter
  - Classical watershed-based method combining intensity and distance

**Aggregate Segmentation**:
- [x] `aggrequant/segmentation/aggregates/filter_based.py` - FilterBasedSegmenter
  - Background normalization, intensity thresholding, morphological cleanup
- [x] `aggrequant/segmentation/aggregates/neural_network.py` - NeuralNetworkSegmenter
  - PyTorch model integration, sliding window inference, patch stitching

### Completed (Phase 5 - Statistics & Export)

**Quantification Module** (`aggrequant/quantification/`):
- [x] `results.py` - FieldResult, WellResult, PlateResult dataclasses
- [x] `measurements.py` - compute_field_measurements(), compute_masked_measurements()
  - QoI: % aggregate-positive cells, aggregate count, area ratios, ambiguous aggregates

**Statistics Module** (`aggrequant/statistics/`):
- [x] `well_stats.py` - aggregate_field_to_well(), weighted aggregation
- [x] `controls.py` - SSMD, Z-factor, control comparison with interpretation
- [x] `export.py` - CSV, Parquet, Excel export, plate summary generation

### Next TODO Steps

**Remaining Phase 1**:
1. [ ] Create `aggrequant/quality/visualization.py` - Focus map visualization utilities

**Phase 6 - Integration & CLI**:
2. [ ] `aggrequant/pipeline.py` - Main processing orchestrator
3. [ ] `aggrequant/cli.py` - CLI entry point
4. [ ] Documentation and user guide

**Agent Usage**:
- Use **code-reviewer agent** for all refactoring tasks and code review
- Use **cv-ml-engineer agent** for NN-related work

### Files Verified Working

```bash
# Test imports in AggreQuant conda environment:
conda activate AggreQuant
python -c "
from aggrequant import __version__, __author__
from aggrequant.common import normalize_image, SimpleLogger
from aggrequant.quality import FocusMetrics, compute_focus_metrics
from aggrequant.loaders import PipelineConfig, Plate, Well, ImageLoader
from aggrequant.nn.architectures import create_model, list_architectures
from aggrequant.segmentation import (
    StarDistSegmenter, CellposeSegmenter, DistanceIntensitySegmenter,
    FilterBasedSegmenter, NeuralNetworkSegmenter
)
from aggrequant.quantification import (
    FieldResult, WellResult, PlateResult, compute_field_measurements
)
from aggrequant.statistics import (
    aggregate_field_to_well, compute_ssmd, export_to_csv
)
from gui import AggreQuantApp, main
from gui.widgets import PlateSelector, ControlPanel, SettingsPanel, ProgressPanel
print('Available models:', list_architectures())
print('All imports successful!')
"

# Launch GUI (requires display):
# python -c "from gui import main; main()"
```

---

*Last updated: 2026-02-04*
*Document version: 2.5*
