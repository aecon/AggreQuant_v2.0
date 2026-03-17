# Migration Report: aSynAggreCount → AggreQuant

**Date:** 2026-03-16
**Old version:** `aSynAggreCount_DV_2024-07-09` (archived at `/home/athena/1_CODES/0_FROM_HESTIA_2026-Feb-05/`)
**New version:** `AggreQuant v0.1.0` (`/home/athena/1_CODES/AggreQuant/`)

---

## 1. Executive Summary

AggreQuant is a ground-up rewrite of aSynAggreCount. The original was a working but monolithic script-based pipeline for quantifying alpha-synuclein aggregates in 384-well HCS plates. The rewrite restructures it into a proper Python package with modular architecture, adds deep learning infrastructure, focus quality assessment, a GUI, comprehensive tests, and extensive benchmarking. The core scientific workflow (nuclei → cells → aggregates → quantify) is preserved, but nearly every component has been redesigned.

| Aspect | aSynAggreCount (old) | AggreQuant (new) |
|--------|---------------------|------------------|
| Structure | Flat scripts + `processing/`, `utils/`, `statistics/` | Installable package (`pyproject.toml`) with 12 sub-packages |
| Python files | ~15 | ~90 |
| Test coverage | 1 test file | 131+ unit tests across 8 test modules |
| Entry points | `main.py`, `main_multiplate.py` | `scripts/run_pipeline.py`, `scripts/run_gui.py`, training scripts |
| Config | Basic YAML (manual parsing) | Pydantic-validated `PipelineConfig` dataclass |
| Cell segmentation | Cellpose v2 + custom distance-intensity | Cellpose v3 with greedy nucleus matching |
| Aggregate segmentation | Filter-based only | Filter-based + UNet neural network |
| Focus metrics | None | 5 patch metrics + 3 global metrics |
| Visualization | Montages only | Plate heatmaps, QC strip plots, interactive viewers |
| GUI | None | CustomTkinter application |
| Packaging | `requirements.txt` | `pyproject.toml` (pip-installable) |
| DL framework | TensorFlow only (StarDist) | TensorFlow (StarDist) + PyTorch (UNet, Cellpose v3) |

---

## 2. Project Rename and Repackaging

**Old:** `aSynAggreCount` — a collection of scripts run via `main.py` / `main_multiplate.py`. No packaging metadata. Dependencies listed in `requirements.txt`. Required `sys.path` manipulation or manual `PYTHONPATH` setup.

**New:** `aggrequant` — a proper pip-installable Python package.
- `pyproject.toml` with full metadata, dependencies, and optional dependency groups (`[nn]`, `[gui]`, `[dev]`)
- No `sys.path` hacks; all imports are absolute (`from aggrequant.segmentation.stardist import StarDistSegmenter`)
- `__init__.py` files are package markers only (no re-exports)
- Versioned (`v0.1.0`)

---

## 3. Complete File Listings

### 3.1 Old Codebase: `aSynAggreCount_DV_2024-07-09`

Every file in the repository, with a one-line description.

```
aSynAggreCount/
│
├── main.py                             # Entry point: process a single plate from one YAML setup file
├── main_multiplate.py                  # Entry point: batch-process multiple plates from a list of YAML files
├── run.sh                              # Shell launcher for main.py
├── requirements.txt                    # pip dependencies (TF 2.15, Cellpose 2.2, StarDist 0.8.5, etc.)
├── README.md                           # Project documentation with installation and usage guide
├── LICENSE                             # License file
│
├── processing/                         # ── Core image processing pipeline ──
│   ├── __init__.py                     # Package marker
│   ├── pipeline.py                     # Main orchestrator: loads images, calls segmenters, saves results
│   ├── nuclei.py                       # Nuclei segmentation: BEQ preprocessing → StarDist → size/border filtering
│   ├── cells.py                        # Cell segmentation: Cellpose (cyto2) + distance-intensity algorithm
│   ├── aggregates.py                   # Aggregate segmentation: cap → BEQ → threshold → median → conn. components
│   ├── quantification.py              # Colocalization: per-aggregate cell overlap → 7 Quantities of Interest
│   ├── image_functions.py              # Image loading utilities (TIFF, generic formats)
│   ├── montage.py                      # Montage visualization: random sample grids, overlay views, control columns
│   └── statistics.py                   # Wrapper calling statistics/statistics.py from pipeline
│
├── statistics/                         # ── Plate-level statistical analysis ──
│   ├── __init__.py                     # Package marker
│   ├── statistics.py                   # Well-level aggregation, SSMD, density maps, volcano plots
│   ├── plate.py                        # 384-well plate model: 16 rows × 24 cols × 9 fields of view
│   └── diagnostics.py                  # QC montage generation for control wells
│
├── utils/                              # ── Utility modules ──
│   ├── __init__.py                     # Package marker
│   ├── dataset.py                      # Dataset class: YAML config parsing, file discovery, output path management
│   ├── yaml_reader.py                  # Thin wrapper around yaml.safe_load()
│   ├── printer.py                      # print/stderr logging helpers: msg() and err()
│   └── data.py                         # Simple data container class
│
├── applications/                       # ── Configuration files ──
│   ├── setup.yml                       # Template YAML configuration
│   └── Dalila/                         # 77 experimental setup files (Jan–Jul 2024)
│       ├── setup_20240118_HA38_rep1.yml
│       ├── setup_20240118_HA38_rep2.yml
│       ├── ... (75 more files)
│       └── setup_test_2024-07-09.yml   # Test configuration
│
├── tests/                              # ── Testing ──
│   ├── test_nuclei.py                  # Unit tests for nuclei segmentation
│   ├── run.sh                          # Test runner script
│   ├── README.md                       # Test documentation
│   └── data/                           # Sample 3-channel test images (Blue, Green2, FarRed)
│
├── imagej/                             # ── ImageJ preprocessing macros ──
│   ├── split_No_CE.ijm                 # Channel splitting without contrast enhancement
│   └── split_With_CE.ijm              # Channel splitting with contrast enhancement
│
└── graphics/                           # ── Documentation images ──
    ├── pipeline.jpg                    # Pipeline diagram
    ├── raw_and_segmentation.jpg        # Raw vs segmented comparison
    └── segmentation.jpg                # Segmentation result example
```

**Totals:** 15 Python files, 78 YAML configs, 2 ImageJ macros, 1 test file.

---

### 3.2 New Codebase: `AggreQuant v0.1.0`

Every file in the repository, with a one-line description.

```
AggreQuant/
│
├── pyproject.toml                      # Package metadata, dependencies, optional groups [nn], [gui], [dev]
├── README.md                           # Project documentation
├── PROJECT.md                          # Development roadmap and phase planning
│
├── aggrequant/                         # ══ Main package ══
│   ├── __init__.py                     # Package marker (v0.1.0, __author__)
│   ├── pipeline.py                     # SegmentationPipeline: orchestrates load → segment → quantify → save
│   ├── focus.py                        # Focus quality metrics: 5 patch-based + 3 global (PLLS, Laplacian, etc.)
│   ├── colocalization.py               # Cell-aggregate overlap via sparse cross-tabulation; quantify_field()
│   │
│   ├── common/                         # ── Shared utilities ──
│   │   ├── __init__.py                 # Package marker
│   │   ├── image_utils.py              # load_image(), normalize_image(), find_image_files()
│   │   ├── logging.py                  # setup_logging(), get_logger() — Python logging configuration
│   │   ├── gpu_utils.py                # configure_tensorflow_memory_growth() — idempotent TF GPU setup
│   │   └── cli_utils.py               # print_config_summary(), print_section_header()
│   │
│   ├── loaders/                        # ── Configuration and data loading ──
│   │   ├── __init__.py                 # Package marker
│   │   ├── config.py                   # PipelineConfig dataclass hierarchy with YAML I/O and validation
│   │   ├── images.py                   # InCell filename parser; build_field_triplets() for image discovery
│   │   └── plate.py                    # Well ID ↔ indices conversion; 96/384-well plate layouts
│   │
│   ├── segmentation/                   # ── Segmentation backends ──
│   │   ├── __init__.py                 # Package marker
│   │   ├── base.py                     # BaseSegmenter ABC: enforces segment() interface, provides _log/_debug
│   │   ├── stardist.py                 # StarDistSegmenter: BEQ → StarDist 2D_versatile_fluo → size/border filter
│   │   ├── cellpose.py                 # CellposeSegmenter: Cellpose cyto3 + greedy nucleus-cell ID matching
│   │   ├── postprocessing.py           # Shared: remove_border_objects, filter_aggregates_by_cells, relabel_consecutive
│   │   └── aggregates/                 # ── Aggregate segmentation sub-package ──
│   │       ├── __init__.py             # Package marker
│   │       ├── filter_based.py         # FilterBasedSegmenter: BEQ → threshold → median → connected components
│   │       └── neural_network.py       # NeuralNetworkSegmenter: lazy model load → predict → postprocess
│   │
│   ├── nn/                             # ── Deep learning module (PyTorch) ──
│   │   ├── __init__.py                 # Package marker
│   │   ├── utils.py                    # get_device(): auto-detect CUDA/CPU
│   │   ├── inference.py                # predict_full(), predict_tiled() with Gaussian blending, predict() auto-detect
│   │   │
│   │   ├── architectures/              # ── UNet architecture and building blocks ──
│   │   │   ├── __init__.py             # Package marker
│   │   │   ├── unet.py                 # Modular UNet: pluggable encoder/decoder/bridge, attention, deep supervision
│   │   │   ├── registry.py             # 7 named model presets for ablation; create_model(), list_models()
│   │   │   └── blocks/                 # ── Pluggable building blocks ──
│   │   │       ├── __init__.py         # Package marker
│   │   │       ├── conv.py             # SingleConv, DoubleConv — baseline UNet blocks (Ronneberger 2015)
│   │   │       ├── residual.py         # ResidualBlock, BottleneckResidualBlock — skip connections (He 2016)
│   │   │       ├── attention.py        # AttentionGate, MultiHeadAttentionGate — skip filtering (Oktay 2018)
│   │   │       ├── se.py               # SEBlock, SEConvBlock, SEResidualBlock — channel recalibration (Hu 2018)
│   │   │       ├── cbam.py             # ChannelAttention, SpatialAttention, CBAM, CBAMConvBlock, CBAMResidualBlock (Woo 2018)
│   │   │       ├── eca.py              # ECABlock — efficient 1D-conv channel attention (Wang 2020)
│   │   │       ├── convnext.py         # LayerNorm2d, ConvNeXtBlock — modern depthwise+inverted bottleneck (Liu 2022)
│   │   │       └── aspp.py             # ASPPConv, ASPPPooling, ASPP, ASPPBridge, LightASPP — multi-scale context (Chen 2017)
│   │   │
│   │   ├── datatools/                  # ── Training data utilities ──
│   │   │   ├── __init__.py             # Package marker
│   │   │   ├── dataset.py              # extract_patches(), PatchDataset, create_dataloaders() with train/val split
│   │   │   └── augmentation.py         # torchvision v2 pipeline: spatial, intensity, noise, blur transforms
│   │   │
│   │   ├── training/                   # ── Training loop and losses ──
│   │   │   ├── __init__.py             # Package marker
│   │   │   ├── trainer.py              # Trainer class: train/val loop, checkpointing, early stopping, LR scheduling
│   │   │   └── losses.py              # DiceLoss, DiceBCELoss, FocalLoss, TverskyLoss, FocalTverskyLoss, EdgeWeightedLoss, DeepSupervisionLoss
│   │   │
│   │   └── evaluation/                 # ── Evaluation metrics ──
│   │       ├── __init__.py             # Package marker
│   │       └── metrics.py              # Dice, IoU, precision, recall, specificity, soft_dice; SegmentationMetrics; find_optimal_threshold
│   │
│   └── visualization/                  # ── Output visualization ──
│       ├── __init__.py                 # Package marker
│       ├── heatmaps.py                 # Plotly plate heatmaps: cell count, aggregate %, area, focus metrics per well
│       └── qc_plots.py                # Matplotlib strip plots for control well QC validation
│
├── gui/                                # ══ GUI application (CustomTkinter) ══
│   ├── __init__.py                     # Package marker
│   ├── app.py                          # Main window: well selection, config, analysis launch, progress monitoring
│   ├── pipeline_runner.py              # Background thread: builds PipelineConfig from GUI selections, runs pipeline
│   └── widgets/                        # ── Reusable UI components ──
│       ├── __init__.py                 # Package marker
│       ├── plate_selector.py           # Interactive 96/384-well plate grid with click/drag well selection
│       ├── control_panel.py            # Buttons for assigning control types (positive, negative, custom)
│       ├── settings_panel.py           # Segmentation method, quality threshold, and output option controls
│       └── progress_panel.py           # Progress bar, status messages, and scrolling log output
│
├── scripts/                            # ══ CLI entry points and utilities ══
│   ├── run_pipeline.py                 # Main CLI: run pipeline from YAML config with --verbose, --max-fields flags
│   ├── run_gui.py                      # Launch the CustomTkinter GUI application
│   ├── train_baseline.py              # Train a UNet model on extracted patches (WIP)
│   ├── plot_training.py               # Plot training history (loss curves, metric curves) from JSON (WIP)
│   ├── benchmark_focus.py              # Benchmark and visualize focus/blur metrics on test images
│   ├── benchmark_nuclei.py             # Compare StarDist vs Cellpose for nuclei segmentation
│   └── test_cellpose_only.py           # Minimal Cellpose test without TensorFlow/StarDist dependency
│
├── configs/                            # ══ Example YAML configurations ══
│   ├── README.md                       # Configuration documentation
│   └── test_384well.yaml               # Example 384-well plate configuration
│
├── tests/                              # ══ Unit tests (pytest) ══
│   ├── __init__.py                     # Package marker
│   ├── conftest.py                     # Shared pytest fixtures (synthetic images, temp dirs)
│   └── unit/                           # ── Per-module test files ──
│       ├── __init__.py                 # Package marker
│       ├── test_nuclei_segmentation.py # StarDist end-to-end: preprocessing, segmentation, relabeling
│       ├── test_cell_segmentation.py   # Cellpose: segmentation, nucleus matching, ID correspondence
│       ├── test_aggregate_segmentation.py # Filter-based: threshold, hole filling, size filtering
│       ├── test_nn_blocks.py           # All 8 architectural blocks: shape, gradient flow, attention weights
│       ├── test_nn_unet.py             # UNet configs, get_config() roundtrip, registry, deep supervision
│       ├── test_nn_inference.py        # Full-res padding, tiled blending, postprocessing, auto-detect
│       ├── test_nn_losses.py           # All loss functions: gradient flow, edge cases, deep supervision wrapper
│       └── test_nn_metrics.py          # Dice, IoU, precision, recall, soft_dice, SegmentationMetrics, threshold search
│
├── benchmarks/                         # ══ Segmentation benchmark suites ══
│   ├── model_selection_rationale.md    # Summary of model selection decisions across all benchmarks
│   │
│   ├── nuclei_segmentation/            # ── Nuclei: 13 models × 9 categories × 100 images ──
│   │   ├── README.md                   # Overview, model list, run instructions
│   │   ├── BENCHMARK_PLAN.md           # Detailed experimental design and implementation spec
│   │   ├── supplementary.md            # Draft supplementary text for the AggreQuant paper
│   │   ├── NORMALIZATION_EVALUATION.md # BEQ normalization effect on all 13 models
│   │   ├── NORMALIZATION_REEVAL_STARDIST_CELLPOSE.md  # Focused BEQ analysis for StarDist and Cellpose
│   │   ├── WORK_STATUS.md              # Task tracking for benchmark implementation
│   │   ├── mask_visualization_options.md # Discussion of visualization approaches
│   │   ├── run_benchmark.py            # Run all 13 models on curated image set, save masks + timing
│   │   ├── plot_results.py             # Generate comparison figures (counts, CV, timing, panels A–F)
│   │   ├── plot_masks.py               # Mask gallery: curated images × models, consensus heatmap
│   │   ├── select_images.py            # Curate 100 images into 9 difficulty categories
│   │   ├── viewer.py                   # Streamlit interactive viewer for mask comparison
│   │   ├── compare_normalization.py    # Compute and plot BEQ normalization effects
│   │   └── analyze_intensity_detection.py # Analyze detection rates vs image intensity
│   │
│   ├── cell_segmentation/              # ── Cells: 8 models, nuclear channel dependency ──
│   │   ├── README.md                   # Overview, model list, conclusions
│   │   ├── BENCHMARK_PLAN.md           # Experimental design
│   │   ├── supplementary.md            # Draft supplementary text on nuclear channel effects
│   │   ├── literature_review.md        # Mesmer/InstanSeg/CellSAM training data and validation analysis
│   │   ├── dependency_check.md         # Package compatibility notes
│   │   ├── run_benchmark.py            # Run 8 cell models (single + dual channel)
│   │   ├── plot_results.py             # Cell count and timing comparison figures
│   │   └── plot_masks.py               # Mask gallery for cell segmentation models
│   │
│   ├── preprocessing_cellpose_segmentation/  # ── Preprocessing: binary mask vs raw DAPI vs none ──
│   │   ├── BENCHMARK_PLAN.md           # Experimental design for 3 input variants
│   │   ├── run_benchmark.py            # Run Cellpose cyto3 with 3 nuclear input configurations
│   │   └── plot_results.py             # Count, solidity, and timing comparison figures
│   │
│   └── mini_cellpose_from_scratch/     # ── Educational: Cellpose reimplementation ──
│       ├── README.md                   # Algorithm explanation and usage
│       ├── environment.yml             # Conda environment for standalone use
│       ├── model.py                    # Small UNet (~2.5M params) with style vector
│       ├── dynamics.py                 # Flow target computation and Euler integration for instance masks
│       ├── dataset.py                  # PyTorch dataset with flow caching
│       ├── train.py                    # Training loop with flow-based loss
│       └── predict.py                  # Inference: forward pass → flow integration → instance labels
│
├── devlog/                             # ══ Development documentation ══
│   ├── README.md                       # Progress overview: milestones, future improvements, changelog
│   ├── restructure.md                  # Notes on package restructuring decisions
│   ├── Plan_2026-02-17.md              # Daily plan and task tracking
│   ├── Plan_2026-02-18.md
│   ├── Plan_2026-03-04.md
│   ├── Plan_2026-03-10.md
│   ├── Plan_2026-03-12.md
│   ├── Plan_2026-03-16.md
│   └── docs/                           # ── Technical documentation ──
│       ├── architecture_modules.md     # UNet block explanations for collaborators
│       ├── nn_building_blocks.md       # Neural network module summary table
│       ├── nn_restructuring_plan.md    # nn/ refactoring roadmap
│       └── migration_report.md         # This report
│
└── data/                               # ══ Test data ══
    └── test/                           # Sample images and configs for unit tests
        └── output/
            └── test_config.yaml        # Test configuration file
```

**Totals:** 47 Python files in main package, 8 test files, 7 GUI files, 7 scripts, 19 benchmark Python files, 1 YAML config + `pyproject.toml`.

---

## 4. Module-by-Module Comparison

### 4.1 Configuration & Data Loading

| Feature | Old | New |
|---------|-----|-----|
| Config format | YAML with manual key access | YAML validated by Pydantic `PipelineConfig` dataclass |
| Parser | `yaml_reader.py` (bare `yaml.safe_load`) | `config.py` with type checking, defaults, validation |
| Plate formats | 384-well only (hard-coded) | 384 and 96-well plates |
| File discovery | Channel color strings + glob in `Dataset` class | InCell filename parser with `FieldTriplet` NamedTuples |
| Image parsers | InCell assumed (not explicit) | InCell format explicitly supported; Operetta/ImageXpress removed after evaluation |
| Output paths | `Dataset.get_output_file_names()` | Pipeline generates flat `labels/` directory with well/field naming |

**Key changes:**
- `PipelineConfig` validates all fields at construction (channel patterns, metric names, plate format, segmentation params). Invalid configs fail fast with clear error messages.
- `build_field_triplets()` groups images by (well_id, field_id) and only returns complete triplets (all 3 channels present). Incomplete fields are skipped with a warning.

### 4.2 Nuclei Segmentation (StarDist)

| Feature | Old (`processing/nuclei.py`) | New (`segmentation/stardist.py`) |
|---------|------|------|
| Model | `2D_versatile_fluo` | `2D_versatile_fluo` (same) |
| Preprocessing | `img / G(σ=50)` with `G(σ=2)` denoise | Same algorithm, params exposed in YAML config |
| Post-processing | Size exclusion (300–15000 px), border dilation, border exclusion | Same filters, plus LUT-based consecutive relabeling (O(N)) |
| Relabeling | Not done (raw StarDist labels) | LUT-based relabeling to consecutive IDs (uint16) |
| Interface | Functions (`_pre_process`, `_segment_stardist`, etc.) | `StarDistSegmenter` class inheriting `BaseSegmenter` |
| Lazy loading | StarDist model loaded once in `pipeline.py` | Model loaded on first `segment()` call |

**Key changes:**
- Segmentation parameters (`sigma_denoise`, `sigma_background`, `min_area`, `max_area`) are now configurable via YAML instead of hard-coded constants.
- LUT-based relabeling replaces no relabeling — ensures consecutive label IDs for downstream processing.
- scikit-image API updated: `binary_dilation` → `dilation`, `preserve_range` and `reflect` mode used consistently.

### 4.3 Cell Segmentation (Cellpose)

| Feature | Old (`processing/cells.py`) | New (`segmentation/cellpose.py`) |
|---------|------|------|
| Models | Cellpose v2 (`cyto2`) + custom distance-intensity | Cellpose v3 (`cyto3`) only |
| Distance-intensity algo | Full implementation (CLAHE, watershed, distance+intensity fields) | **Removed** — Cellpose v3 renders it unnecessary |
| Nucleus matching | Remove cells without nuclei (simple filter) | **Greedy matching** by nucleus occupancy fraction; cell_id = nucleus_id |
| Nucleus sync | Not done | Unmatched nuclei zeroed in-place in `nuclei_labels` |
| Interface | Functions | `CellposeSegmenter` class inheriting `BaseSegmenter` |

**Key changes:**
- The custom distance-intensity algorithm (`DistanceIntensitySegmenter`) was removed. Benchmarking showed Cellpose v3 outperforms it in all cases.
- **Critical improvement:** `_match_cells_to_nuclei()` implements greedy matching that ensures cell ID = nucleus ID for the same physical object. This guarantees ID correspondence throughout the pipeline — essential for correct colocalization and relabeling.
- After matching, unmatched nuclei are zeroed from `nuclei_labels` in-place, preventing phantom objects in quantification.

### 4.4 Aggregate Segmentation

| Feature | Old (`processing/aggregates.py`) | New |
|---------|------|------|
| Classical method | Cap → BEQ → threshold (1.60) → median → connected components → size filter | Same algorithm in `aggregates/filter_based.py`, with LUT relabeling added |
| Neural network | None | `aggregates/neural_network.py` — UNet-based segmentation |
| Relabeling | Not done | LUT-based consecutive relabeling |
| Hole filling | `remove_small_holes` then `label` (wrong order) | **Fixed:** `remove_small_holes` before `label` per skimage docs |

**Key changes:**
- Bug fix: hole-filling order corrected (must happen before `label()`).
- Added neural network alternative path using trained UNet models.
- LUT-based relabeling for consistency with other segmenters.

### 4.5 Quantification / Colocalization

| Feature | Old (`processing/quantification.py`) | New (`colocalization.py`) |
|---------|------|------|
| Data structure | `Quantities` class with 7 named fields | Plain dict returned by `quantify_field()` |
| Overlap method | Per-aggregate loop counting cell overlaps | **Sparse cross-tabulation** (scipy CSR matrix) — ~6x speedup |
| Metrics | 7 QoI (% positive, cell count, area %, etc.) | 6 metrics (n_cells, n_nuclei, n_aggregates, n_positive, pct_positive, areas) |
| Ambiguous aggregates | Tracked (aggregates touching multiple cells) | Not tracked (simpler, more robust) |
| Output | Per-image `.txt` files | Per-field rows in a single `field_measurements.csv` |

**Key changes:**
- `build_overlap_table()` uses scipy sparse cross-tabulation instead of per-aggregate loops — major performance improvement.
- Output is a flat CSV with one row per field, combining focus metrics and quantification results.
- "Ambiguous aggregates" metric removed (rarely used, added complexity).

### 4.6 Statistics

| Feature | Old (`statistics/statistics.py`, `plate.py`, `diagnostics.py`) | New |
|---------|------|------|
| Well-level aggregation | Full implementation (per-well stats, SSMD, density maps, volcano plots) | **Not yet implemented** (Phase 2 milestone) |
| Plate model | `Plate` class (384-well, 16×24, 9 FOV) | `loaders/plate.py` — well ID utilities, supports 96 and 384 |
| Diagnostics | Montage generation (random sample, overlay, control columns) | Replaced by `visualization/` module |

**Key changes:**
- The old statistics module was fully functional for well-level aggregation and assay QC. This functionality has not been re-implemented yet in the new version (planned for Phase 2).
- Plate-level visualization is partially replaced by heatmaps and QC strip plots.

### 4.7 Visualization

| Feature | Old (`processing/montage.py`, `statistics/diagnostics.py`) | New (`visualization/`) |
|---------|------|------|
| Montages | 8×16 random sample grids, overlay views, control column grids | **Removed** (replaced by per-metric plate views) |
| Plate heatmaps | Density maps in statistics module | `heatmaps.py` — Plotly heatmaps for all metrics (cell count, aggregate %, areas) + focus metrics |
| QC plots | None | `qc_plots.py` — control well strip plots for assay validation |
| Interactive viewer | None | Streamlit-based interactive viewer in benchmarks (model comparison) |

---

## 5. Entirely New Components

### 5.1 Focus Quality Assessment (`focus.py`)

Not present in the old version. Computes image sharpness/blur metrics at two scales:

**Patch-based** (80×80 non-overlapping grid):
- VarianceLaplacian, LaplaceEnergy, Sobel, Brenner, FocusScore

**Global** (frequency domain, full image):
- Power Log-Log Slope (PLLS) — best single defocus metric per Bray et al. 2012
- Global variance of Laplacian
- High-frequency energy ratio

Focus metrics are computed per channel and saved alongside quantification results in the output CSV.

### 5.2 Deep Learning Module (`nn/`)

Entirely new. A full PyTorch-based deep learning infrastructure for training and deploying UNet models for aggregate segmentation. The module is designed for systematic ablation studies — start with baseline UNet and incrementally add modules to measure their individual effect on segmentation performance.

#### 5.2.1 UNet Architecture (`nn/architectures/unet.py`)

The core architecture is a modular UNet (Ronneberger et al., 2015) with pluggable components at every level. The constructor exposes 12 configuration parameters:

```python
UNet(
    in_channels=1,          # Grayscale microscopy
    out_channels=1,         # Binary segmentation
    features=[64, 128, 256, 512],  # Channel sizes per encoder level
    encoder_block="double_conv",   # or "residual", "convnext"
    decoder_block="double_conv",   # or "residual", "convnext"
    bridge_type="double_conv",     # or "residual", "aspp"
    use_attention_gates=False,     # Oktay et al. 2018
    use_se=False,                  # Hu et al. 2018
    use_cbam=False,                # Woo et al. 2018
    use_eca=False,                 # Wang et al. 2020
    use_deep_supervision=False,    # Multi-scale auxiliary losses
    upsample_mode="transpose",     # or "bilinear"
)
```

**Data flow:**
1. **Encoder path:** For each level, `EncoderBlock` applies a conv block (DoubleConv, ResidualBlock, or ConvNeXtBlock), optionally followed by channel attention (SE, CBAM, or ECA). Output is saved as a skip connection, then max-pooled 2×.
2. **Bridge/bottleneck:** At the lowest resolution, applies either a standard conv block, residual block, or ASPP (multi-scale dilated convolutions).
3. **Decoder path:** For each level, `DecoderBlock` upsamples (transposed conv or bilinear), optionally applies an attention gate to the skip connection, concatenates skip + upsampled, then applies a conv block with optional channel attention.
4. **Final:** 1×1 convolution to output channels.
5. **Deep supervision:** If enabled during training, intermediate decoder outputs are projected to output channels and returned alongside the main output.

**Weight initialization:** Kaiming normal for all Conv2d layers, constant (1, 0) for BatchNorm2d.

**Self-contained checkpoints:** `get_config()` returns a dict of all constructor kwargs, which is saved into the checkpoint by `Trainer.save_checkpoint()`. This means a checkpoint file contains everything needed to reconstruct the architecture + load weights — no external config file required.

#### 5.2.2 Building Blocks (`nn/architectures/blocks/`)

Eight pluggable blocks, each in its own module. The UNet selects blocks by string name at construction.

##### DoubleConv (`conv.py`) — Baseline block

The standard UNet block: two consecutive `SingleConv` (Conv2d → BatchNorm2d → ReLU) modules.

- `SingleConv`: 3×3 conv with `bias=False` (redundant before BatchNorm), `affine=True` BatchNorm, `inplace=True` ReLU
- `DoubleConv`: Two `SingleConv` in sequence, with optional `mid_channels`

**CV best practice:** Setting `bias=False` in Conv2d before BatchNorm follows the convention established in ResNet — BatchNorm's learnable bias (`affine=True`) subsumes the conv bias, so keeping both wastes parameters and can slow convergence.

**Reference:** Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

##### ResidualBlock (`residual.py`) — Skip addition for gradient flow

Implements `output = ReLU(input + F(input))` where F is a two-convolution path. If input/output channels differ, a 1×1 convolution matches dimensions for the skip connection.

Also provides `BottleneckResidualBlock` (1×1 → 3×3 → 1×1 pattern) for deeper networks with reduced computation.

**CV best practice:** Residual connections are essential for training segmentation networks beyond 4–5 encoder levels. They solve the degradation problem (deeper networks performing worse than shallow ones) by providing identity shortcuts for gradient flow. In medical/microscopy segmentation, ResUNet variants consistently outperform plain UNets.

**Reference:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). "Deep Residual Learning for Image Recognition." CVPR. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

##### AttentionGate (`attention.py`) — Learned skip connection filtering

The attention gate learns to suppress irrelevant regions in skip connections using the decoder features as a gating signal:

1. Project both gate (decoder) and skip (encoder) to intermediate space via 1×1 convs
2. Upsample gate to match skip spatial size
3. Sum, ReLU, 1×1 conv → sigmoid → attention coefficients
4. Multiply skip features by attention coefficients

Also provides `MultiHeadAttentionGate` (multiple parallel attention heads, analogous to multi-head attention in transformers).

**CV best practice:** Attention gates are particularly beneficial for microscopy segmentation where the target structures (aggregates) are small and sparse relative to the full field of view. Without attention, skip connections pass all encoder features equally, including background noise that the decoder must learn to ignore. Attention gates reduce false positives by suppressing irrelevant activations before they reach the decoder.

**Reference:** Oktay, O., et al. (2018). "Attention U-Net: Learning Where to Look for the Pancreas." MIDL. [arXiv:1804.03999](https://arxiv.org/abs/1804.03999)

##### SEBlock (`se.py`) — Channel recalibration

Squeeze-and-Excitation performs channel-wise recalibration in three steps:

1. **Squeeze:** Global average pooling → (B, C) channel statistics
2. **Excitation:** Two FC layers (`C → C/r → C`) with ReLU and sigmoid → channel weights
3. **Scale:** Multiply original features by learned weights

Also provides `SEConvBlock` (DoubleConv + SE) and `SEResidualBlock` (ResidualBlock + SE).

**CV best practice:** SE blocks add minimal overhead (~0.2% extra params for r=16) but provide measurable gains by letting the network adaptively emphasize informative channels. For microscopy where different channels capture different biological structures, channel recalibration helps the network focus on task-relevant features.

**Reference:** Hu, J., Shen, L., & Sun, G. (2018). "Squeeze-and-Excitation Networks." CVPR. [arXiv:1709.01507](https://arxiv.org/abs/1709.01507)

##### CBAM (`cbam.py`) — Channel + spatial attention

Sequential application of channel attention (shared MLP on avg-pooled + max-pooled features) followed by spatial attention (7×7 conv on channel-pooled statistics):

1. **Channel attention:** AvgPool + MaxPool → shared MLP → sigmoid → channel weights
2. **Spatial attention:** Channel-wise max + mean → 7×7 conv → sigmoid → spatial weights

Also provides `CBAMConvBlock` and `CBAMResidualBlock` integrated variants.

**CV best practice:** CBAM extends SE by adding spatial attention, which helps localize small structures. For aggregate segmentation, where objects can be as small as 9 pixels, spatial attention helps the network focus on specific image regions. The sequential (channel-first, spatial-second) order is important — applying spatial attention first degrades performance per the original paper.

**Reference:** Woo, S., Park, J., Lee, J.-Y., & Kweon, I. S. (2018). "CBAM: Convolutional Block Attention Module." ECCV. [arXiv:1807.06521](https://arxiv.org/abs/1807.06521)

##### ECABlock (`eca.py`) — Efficient channel attention

Replaces SE's FC bottleneck with a 1D convolution across channels:

- SE: `z → FC(C→C/r) → ReLU → FC(C/r→C) → sigmoid` (2C²/r parameters)
- ECA: `z → Conv1D(kernel=k) → sigmoid` (k parameters)

The kernel size k is adaptively determined: `k = |log₂(C) / γ + b / γ|_odd` (minimum 3), so each channel interacts with its k nearest neighbors.

**CV best practice:** ECA avoids the information loss from SE's dimensionality reduction (C → C/r → C) while using far fewer parameters. For smaller feature maps typical in segmentation decoders, ECA is more parameter-efficient than SE while maintaining equivalent or better performance.

**Reference:** Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., & Hu, Q. (2020). "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks." CVPR. [arXiv:1910.03151](https://arxiv.org/abs/1910.03151)

##### ConvNeXtBlock (`convnext.py`) — Modern encoder block

Modernizes standard convolution blocks by borrowing design choices from Vision Transformers:

```
output = x + γ · Linear_up(GELU(Linear_down(LayerNorm(DWConv7×7(x)))))
```

Key differences vs ResidualBlock:
- **7×7 depthwise conv:** Larger receptive field, fewer parameters (groups=channels)
- **LayerNorm** instead of BatchNorm: More stable for small batches
- **GELU** instead of ReLU: Smoother gradients
- **Inverted bottleneck:** Expand channels (C → 4C) then squeeze back
- **LayerScale:** Learnable per-channel scaling of the residual (initialized to 1e-6)

Includes a custom `LayerNorm2d` that permutes (B,C,H,W) → (B,H,W,C) for standard nn.LayerNorm.

**CV best practice:** ConvNeXt demonstrates that pure convolutional architectures can match Vision Transformers when modernized with their design choices. For microscopy segmentation, ConvNeXt's large 7×7 kernels provide broader context per layer, which is valuable for recognizing aggregate-sized structures without needing many pooling levels. LayerNorm is preferred over BatchNorm when batch sizes are small (common with large microscopy images).

**Reference:** Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022). "A ConvNet for the 2020s." CVPR. [arXiv:2201.03545](https://arxiv.org/abs/2201.03545)

##### ASPP / ASPPBridge (`aspp.py`) — Multi-scale context at bottleneck

Atrous Spatial Pyramid Pooling captures multi-scale context via parallel dilated convolutions:

1. **1×1 conv branch** (no dilation)
2. **3×3 conv branches** with dilation rates 6, 12, 18 (standard DeepLabV3 rates)
3. **Global average pooling branch** (image-level features, upsampled back)
4. **Concatenation + 1×1 projection** with 0.5 dropout

`ASPPBridge` wraps ASPP with a refinement conv, designed as a drop-in replacement for the standard DoubleConv bridge at the UNet bottleneck.

Also provides `LightASPP` with depthwise separable convolutions for memory-constrained settings.

**CV best practice:** ASPP is critical at the UNet bottleneck because it captures context at multiple scales without additional pooling. For aggregate segmentation, aggregates range from 9 to thousands of pixels — ASPP's multi-scale receptive fields (effective receptive fields of ~13, 25, 37 pixels from dilations 6, 12, 18) help distinguish true aggregates from noise at different scales. The global pooling branch provides plate-level context.

**Reference:** Chen, L.-C., Papandreou, G., Schroff, F., & Adam, H. (2017). "Rethinking Atrous Convolution for Semantic Image Segmentation." [arXiv:1706.05587](https://arxiv.org/abs/1706.05587)

#### 5.2.3 Model Registry (`nn/architectures/registry.py`)

Seven named presets for systematic ablation, organized in two groups:

**Incremental ablation** (each adds one module over the previous):

| # | Name | What's added | Purpose |
|---|------|-------------|---------|
| 1 | `baseline` | DoubleConv encoder/decoder | Reference point |
| 2 | `resunet` | + ResidualBlock | Test gradient flow benefit |
| 3 | `attention_resunet` | + Attention gates | Test skip filtering |
| 4 | `se_attention_resunet` | + SE channel attention | Test channel recalibration |
| 5 | `aspp_se_attention_resunet` | + ASPP bridge | Test multi-scale context |

**Structural alternatives:**

| # | Name | Architecture | Purpose |
|---|------|-------------|---------|
| 6 | `convnext_unet` | ConvNeXt encoder + attention gates | Modern vs classic encoder |
| 7 | `eca_attention_resunet` | ECA replaces SE | Efficient vs full channel attention |

`create_model("name")` instantiates by name; `list_models()` and `describe_models()` for discovery.

#### 5.2.4 Loss Functions (`nn/training/losses.py`)

Seven loss functions + a deep supervision wrapper, all accepting logits (pre-sigmoid):

| Loss | Formula | When to use |
|------|---------|-------------|
| `DiceLoss` | 1 − 2\|X∩Y\| / (\|X\|+\|Y\|) | Overlap-focused; handles class imbalance naturally |
| `DiceBCELoss` | α·Dice + β·BCE | Combined: BCE for pixel supervision, Dice for overlap |
| `FocalLoss` | −α(1−p)^γ log(p) | Severe class imbalance; down-weights easy examples |
| `TverskyLoss` | 1 − TP/(TP+αFN+βFP) | Controllable FP/FN trade-off (α=β=0.5 → Dice) |
| `FocalTverskyLoss` | (TverskyLoss)^γ | Focal + Tversky for hard example mining |
| `EdgeWeightedLoss` | base_loss + 0.5·weighted_BCE | Boundary emphasis via Laplacian edge detection |
| `DeepSupervisionLoss` | Σ wᵢ·loss(auxᵢ) | Wraps any base loss for multi-scale supervision |

`get_loss_function("name")` factory for config-driven loss selection.

**CV best practice:** For binary segmentation with class imbalance (aggregates are typically <5% of image area), `DiceBCELoss` is the standard starting point — Dice handles the imbalance, BCE provides stable gradients. `FocalLoss` is preferred when imbalance is extreme (>95:5). `EdgeWeightedLoss` helps when boundary precision matters for instance separation. Deep supervision accelerates convergence by providing gradient signal at multiple scales, preventing vanishing gradients in deep encoders.

**References:**
- Dice: Milletari, F., et al. (2016). "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation." [arXiv:1606.04797](https://arxiv.org/abs/1606.04797)
- Focal: Lin, T.-Y., et al. (2017). "Focal Loss for Dense Object Detection." ICCV. [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
- Tversky: Salehi, S. S. M., et al. (2017). "Tversky Loss Function for Image Segmentation Using 3D Fully Convolutional Deep Networks." MLMI.

#### 5.2.5 Training Pipeline (`nn/training/trainer.py`)

`Trainer` class with standard PyTorch training loop:

- **Inputs:** model, train_loader, val_loader, criterion, optimizer, optional scheduler
- **Per-epoch:** train → validate → step scheduler → log → checkpoint
- **Checkpointing:** Saves model state dict, optimizer state dict, scheduler state dict, model config (via `get_config()`), epoch number, and best val loss. Supports `save_best_only` mode.
- **Early stopping:** Monitors validation loss, stops after N epochs without improvement
- **`TrainingHistory` dataclass:** Tracks train/val loss, metrics, and learning rates per epoch. Serializable to JSON for post-training analysis.
- **ReduceLROnPlateau support:** Automatically detects scheduler type and passes val_loss

**CV best practice:** The trainer saves model architecture config alongside weights, making checkpoints fully self-contained. This prevents the common pitfall where a checkpoint file exists but the code that defines the architecture has changed. The `weights_only=True` flag in `torch.load` prevents arbitrary code execution from untrusted checkpoints.

#### 5.2.6 Data Pipeline (`nn/datatools/`)

**Patch extraction** (`dataset.py: extract_patches`):
- Grid-cuts full images + masks into non-overlapping patches
- Saves to `output_dir/images/` and `output_dir/masks/` with encoded filenames: `{stem}_y{row}_x{col}.tif`
- Incomplete edge patches (where grid doesn't fit) are skipped

**Dataset** (`dataset.py: PatchDataset`):
- Loads pre-extracted patch pairs (image + mask matched by filename)
- Percentile normalization (1st–99th percentile → [0, 1])
- Binary mask conversion (any nonzero → 1.0)
- Applies torchvision v2 transforms if provided

**DataLoader creation** (`dataset.py: create_dataloaders`):
- Discovers all patches, shuffles with fixed seed, splits by patch (not by image)
- This ensures patches from every source image appear in both train and val — no single patch appears in both
- Training loader: shuffled, `drop_last=True`, `pin_memory=True`
- Validation loader: unshuffled

**Augmentation** (`augmentation.py`):
Uses torchvision transforms v2 with `tv_tensors` for type-aware joint image/mask transforms:

| Category | Transforms | Applied to |
|----------|-----------|------------|
| Spatial | HorizontalFlip, VerticalFlip, RandomRotate90, RandomAffine | Image + Mask |
| Intensity | ColorJitter (brightness, contrast), RandomGamma | Image only |
| Noise | GaussianNoise, MultiplicativeNoise | Image only |
| Blur | GaussianBlur (kernel 3–5) | Image only |

Custom transforms: `RandomRotate90` (0/90/180/270°), `RandomGamma` (gamma correction), `MultiplicativeNoise` (per-pixel uniform multiplier).

**CV best practice:** The split is done at the patch level, not the image level. Since patches from the same image can end up in both train and val, this is appropriate when the goal is to evaluate pixel-level segmentation quality (not generalization to unseen images). For generalization testing, image-level splitting would be needed. Elastic deformation, shear, and random erasing are intentionally excluded because they distort small aggregate structures unnaturally — this follows domain-specific augmentation principles from nnU-Net (Isensee et al., 2021).

**Reference:** Isensee, F., et al. (2021). "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation." Nature Methods. [doi:10.1038/s41592-020-01008-z](https://doi.org/10.1038/s41592-020-01008-z)

#### 5.2.7 Inference (`nn/inference.py`)

Three inference modes:

1. **`predict_full()`:** Pad image to nearest multiple of 16 (reflection), single forward pass, sigmoid, crop back. Fast, requires full image in VRAM.

2. **`predict_tiled()`:** Split into overlapping tiles (default 256×256, stride 128 = 50% overlap), batch-process tiles, blend with 2D Gaussian weight map (center=1, edges→0), normalize by weight sum. Handles arbitrary image sizes.

3. **`predict()` (auto-detect):** Tries `predict_full()` first. On `torch.cuda.OutOfMemoryError`, clears cache and falls back to `predict_tiled()`.

**Post-processing** (`postprocess_predictions()`):
Probability map → binary (threshold 0.5) → fill small holes (< 6000 px) → connected components → remove small objects (< 9 px) → LUT relabel to consecutive uint32.

**CV best practice:** Gaussian blending for tiled inference is the standard approach for segmentation of large images (used by nnU-Net, StarDist, Cellpose). The key insight is that predictions are most reliable at tile centers and degrade at edges due to reduced context. The 50% overlap + Gaussian weighting ensures every pixel receives predictions from multiple tiles, weighted by confidence. Padding to multiples of 16 (not power-of-2) is more memory-efficient — UNet only needs divisibility by 2^n_pools.

#### 5.2.8 Evaluation Metrics (`nn/evaluation/metrics.py`)

**Single-pass confusion matrix:** All threshold-based metrics (Dice, IoU, precision, recall, specificity, accuracy) are derived from a single TP/TN/FP/FN computation.

**`SegmentationMetrics` class:** Computes multiple metrics in one call, avoiding redundant confusion matrix computation. Used by `Trainer` during validation.

**`soft_dice_score`:** Differentiable Dice using continuous probabilities (no thresholding). Useful for monitoring during training when hard metrics are noisy.

**`find_optimal_threshold`:** Grid search over thresholds (0.1–0.9 in 0.05 steps) to find the threshold maximizing a given metric. Used post-training to tune the binarization threshold.

**`evaluate_model`:** End-to-end model evaluation on a DataLoader — handles device placement, deep supervision output, and metric averaging.

#### 5.2.9 Pipeline Integration (`segmentation/aggregates/neural_network.py`)

`NeuralNetworkSegmenter` is a thin wrapper conforming to `BaseSegmenter`:
- **Lazy model loading:** Model is loaded from checkpoint on first `segment()` call, not at init
- **Architecture from checkpoint:** Reads `model_config` from the checkpoint dict and constructs `UNet(**config)` — no external config needed
- **Inference + post-processing:** Calls `predict()` (auto-detect mode) then `postprocess_predictions()`
- **Pipeline selection:** Controlled by YAML config `aggregate_method: "unet"` + `aggregate_model_path: path/to/checkpoint.pt`

### 5.3 GUI (`gui/`)

Not present in the old version. CustomTkinter-based desktop application:
- `app.py` — Main window with configuration widgets
- `pipeline_runner.py` — Background thread for pipeline execution
- `widgets/` — Reusable UI components

### 5.4 Benchmarking Suite (`benchmarks/`)

Not present in the old version. Three benchmark suites:

1. **Nuclei segmentation** — StarDist vs Cellpose comparison across preprocessing variants, with Streamlit interactive viewer for mask comparison
2. **Cell segmentation** — Cellpose configuration evaluation (8 model configs, FarRed channel), literature review, CellSAM comparison
3. **Mini Cellpose from scratch** — Minimal Cellpose implementation for educational purposes

### 5.5 Comprehensive Testing (`tests/`)

Old version had 1 test file (`test_nuclei.py`). New version has 131+ unit tests:

| Test module | Coverage |
|-------------|----------|
| `test_nn_blocks.py` | All 8 architectural blocks |
| `test_nn_unet.py` | UNet configs, registry, `get_config()` |
| `test_nn_inference.py` | Full-res, tiled, padding, Gaussian blending |
| `test_nn_losses.py` | All 4 loss functions |
| `test_nn_metrics.py` | Dice, Jaccard, IoU, soft metrics |
| `test_nuclei_segmentation.py` | StarDist end-to-end |
| `test_cell_segmentation.py` | Cellpose with nucleus matching |
| `test_aggregate_segmentation.py` | Filter-based segmentation |

---

## 6. Codebase Structure Improvements

This section documents the engineering improvements that transformed a collection of scripts into a maintainable software package.

### 6.1 From Scripts to Installable Package

**Old:** No packaging. Users had to clone the repo, manually install dependencies from `requirements.txt`, and run `main.py` directly. Imports worked only if `PYTHONPATH` was set or `sys.path` was hacked.

**New:** `pyproject.toml`-based package with:
- `pip install -e .` for development
- Optional dependency groups: `[nn]` (PyTorch + training deps), `[gui]` (CustomTkinter), `[dev]` (pytest, ruff, black)
- All imports are absolute: `from aggrequant.segmentation.stardist import StarDistSegmenter`
- No `sys.path` manipulation anywhere

### 6.2 Abstract Base Classes and Polymorphism

**Old:** Segmentation was implemented as bare functions (`_pre_process()`, `_segment_stardist()`, `_post_process_size_exclusion()` in `nuclei.py`). The pipeline called these functions directly, with ad-hoc branching for algorithm selection:

```python
# Old pipeline.py
if dataset.cell_segmentation_algorithm == "cellpose":
    cells.segment_cellpose(...)
elif dataset.cell_segmentation_algorithm == "distanceIntensity":
    cells.segment_distance_intensity(...)
```

**New:** All segmenters inherit from `BaseSegmenter` (ABC):

```python
class BaseSegmenter(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray: ...
```

The pipeline holds segmenter instances (`self._nuclei_segmenter`, `self._cell_segmenter`, `self._aggregate_segmenter`) and calls `.segment()` uniformly. Adding a new segmenter means implementing one class, not modifying the pipeline.

### 6.3 Configuration Validation

**Old:** YAML parsed by bare `yaml.safe_load()`. No validation — typos in field names or invalid values silently produced wrong behavior. Config keys accessed by string indexing: `dataset.setup_file['CELL_SEGMENTATION_ALGORITHM']`.

**New:** Pydantic-style dataclass hierarchy with `__post_init__` validation:

```
PipelineConfig
├── ChannelConfig (name, pattern, purpose — validated against allowed set)
├── SegmentationConfig (nuclei params, cell model, aggregate method)
├── QualityConfig (metric names validated against ALL_PATCH_METRICS / ALL_GLOBAL_METRICS)
├── OutputConfig (output_subdir, save_masks, overwrite_masks)
└── control_wells, n_workers, use_gpu, etc.
```

Invalid configs fail at construction with specific error messages. `from_yaml()` and `to_yaml()` handle serialization. `create_default_config()` provides sensible defaults.

### 6.4 Structured Logging

**Old:** `printer.py` with `msg()` and `err()` wrappers around `print()` and `sys.stderr.write()`. No log levels, no filtering, no file output.

**New:** Python `logging` module throughout:
- `get_logger(__name__)` returns a namespaced logger (`aggrequant.segmentation.stardist`)
- `setup_logging()` configures root logger with level, format, and optional file handler
- Log levels (DEBUG/INFO/WARNING/ERROR) used consistently
- Auto-configuration on first use if `setup_logging()` not called explicitly

### 6.5 Separation of Concerns

**Old:** `pipeline.py` was a monolithic orchestrator that mixed image loading, segmentation calls, quantification, montage generation, and statistics in one `process()` function (~200 lines of mixed logic).

**New:** Clear module boundaries:

| Concern | Module | Responsibility |
|---------|--------|---------------|
| Configuration | `loaders/config.py` | Parse and validate YAML |
| File discovery | `loaders/images.py` | Parse InCell filenames, build triplets |
| Plate layout | `loaders/plate.py` | Well ID utilities |
| Image I/O | `common/image_utils.py` | Load, normalize, find files |
| GPU config | `common/gpu_utils.py` | TF memory growth |
| Nuclei segmentation | `segmentation/stardist.py` | StarDist backend |
| Cell segmentation | `segmentation/cellpose.py` | Cellpose backend |
| Aggregate segmentation | `segmentation/aggregates/` | Filter or UNet backend |
| Post-processing | `segmentation/postprocessing.py` | Border removal, filtering, relabeling |
| Focus metrics | `focus.py` | Patch + global quality metrics |
| Quantification | `colocalization.py` | Cell-aggregate overlap |
| Visualization | `visualization/` | Heatmaps, QC plots |
| Orchestration | `pipeline.py` | `SegmentationPipeline` ties it all together |

### 6.6 Idempotent / Resumable Pipeline

**Old:** No resume capability. Restarting required reprocessing all images.

**New:**
- `_masks_exist()` checks if all 3 label TIFs exist for a field
- If `overwrite_masks=False` (default), processed fields are skipped
- Existing CSV results are loaded on resume, so skipped fields keep their measurements
- This makes large plate processing robust to interruptions

### 6.7 Post-Processing as Shared Module

**Old:** Post-processing was scattered across individual segmenters — each had its own size filtering, relabeling, and border handling code.

**New:** `segmentation/postprocessing.py` provides shared utilities:
- `remove_border_objects()` — zeros cells + matching nuclei at image edges
- `filter_aggregates_by_cells()` — zeros aggregates outside cells, relabels
- `relabel_consecutive()` — LUT-based relabeling preserving cell-nucleus ID correspondence
- `count_labels()` — fast object count + area from label map
- `remove_small_holes()` / `remove_small_objects()` — thin wrappers ensuring correct API usage

### 6.8 Image Loading Consolidation

**Old:** `image_functions.py` had multiple loading paths and ad-hoc normalization:
- `load_image()` with manual dtype handling
- Separate normalization in each segmenter
- No shared normalize function

**New:** `common/image_utils.py` provides a single canonical path:
- `load_image()` — tifffile (preferred for TIFF) with skimage fallback
- `normalize_image()` — three methods (minmax, percentile, zscore) with consistent float32 output
- `find_image_files()` — extension-aware file discovery
- `SUPPORTED_IMAGE_EXTENSIONS` — single source of truth

### 6.9 `__init__.py` Policy

**Old:** Some `__init__.py` files re-exported classes (`from .nuclei import *`), creating implicit imports and potential circular dependencies.

**New:** All `__init__.py` files are package markers only — docstring + `__version__`/`__author__` in the top-level one. Users import explicitly:

```python
# Always explicit
from aggrequant.segmentation.stardist import StarDistSegmenter
from aggrequant.nn.architectures.registry import create_model
```

This prevents circular imports, makes dependency chains visible, and improves IDE navigation.

### 6.10 Development Infrastructure

**Old:** `tests/` had 1 test file and a shell script runner. No linting, no formatting.

**New:**
- **131+ pytest tests** with fixtures, parametrization, and `conftest.py`
- **ruff** for linting
- **black** for formatting
- **pytest-cov** for coverage measurement
- All run via `conda run -n AggreQuant pytest tests/`
- Devlog with daily plans, progress tracking, and changelog

---

## 7. Bug Fixes from Old Version

1. **Hole-filling order** (`aggregates.py`): Old code called `label()` before `remove_small_holes()`. skimage requires binary input for hole filling — labeling first breaks this. Fixed in new version.

2. **Cell-nucleus ID mismatch**: Old code only removed cells without nuclei but did not ensure cell_id = nucleus_id. New greedy matching guarantees this correspondence, preventing silent errors in colocalization.

3. **Unmatched nuclei persisting**: Old code left orphan nuclei in the label map after cell segmentation. New code zeros them out in-place.

4. **Conv2d bias before BatchNorm**: Neural network blocks had redundant bias terms in Conv2d layers immediately followed by BatchNorm2d (which has its own bias). Fixed by setting `bias=False` in all such cases.

5. **Deprecated scikit-image API**: `binary_dilation` → `dilation`, `remove_small_objects` API updated to use `max_size` directly.

---

## 8. Removed / Deprecated Features

| Feature | Reason |
|---------|--------|
| Distance-intensity cell segmenter | Cellpose v3 outperforms it; unnecessary complexity |
| `main.py` / `main_multiplate.py` entry points | Replaced by `scripts/run_pipeline.py` |
| Operetta / ImageXpress parsers | Only InCell format is used in practice |
| Montage generation | Replaced by plate heatmaps and QC strip plots |
| Well-level statistics (SSMD, volcano plots) | Not yet re-implemented (Phase 2) |
| `Quantities` class | Replaced by plain dicts from `quantify_field()` |
| `FieldResult` dataclass | Removed; plain dicts used throughout |
| `printer.py` (print wrappers) | Replaced by Python `logging` module |
| `ImageJ` macros | Not carried forward (preprocessing done in Python) |
| 77 experimental YAML configs (`applications/Dalila/`) | Not migrated; new config schema differs |

---

## 9. Dependency Changes

| Old | New | Notes |
|-----|-----|-------|
| tensorflow 2.15.0 | tensorflow[and-cuda] | Still used for StarDist only |
| cellpose 2.2.3 | cellpose ≥3.0 | Major version upgrade (cyto2 → cyto3) |
| stardist 0.8.5 | stardist (latest) | Same model, updated API |
| — | torch, torchvision | New: UNet training and inference |
| — | pydantic | New: config validation |
| — | customtkinter | New: GUI |
| — | pytest, pytest-cov, ruff | New: testing and linting |
| — | tqdm, click | New: CLI utilities |
| plotly 5.18.0 | plotly | Still used for heatmaps |
| — | streamlit | New: interactive benchmark viewer |
| csbdeep 0.7.4 | csbdeep | StarDist dependency |

---

## 10. Performance Improvements

1. **Sparse cross-tabulation** for colocalization: ~6x speedup over per-aggregate loops.
2. **LUT-based relabeling**: O(N) lookup table instead of per-pixel relabeling with scikit-image.
3. **Skip processed fields**: Pipeline checks for existing masks on disk and skips already-processed fields.
4. **Lazy model loading**: StarDist and Cellpose models loaded only on first use, not at pipeline init.
5. **TF memory growth**: `configure_tensorflow_memory_growth()` prevents TensorFlow from allocating all GPU memory at startup.
6. **Tiled inference with OOM fallback**: Large images handled gracefully without manual intervention.

---

## 11. Summary of Status

**Fully functional and improved over old version:**
- Image loading and configuration
- Nuclei segmentation (StarDist)
- Cell segmentation (Cellpose v3 with proper nucleus matching)
- Aggregate segmentation (filter-based, bug-fixed)
- Per-field quantification and colocalization
- Focus quality assessment (entirely new)
- Plate visualization (heatmaps, QC plots)
- Unit testing

**New capabilities not in old version:**
- Deep learning aggregate segmentation (UNet)
- GUI application
- Benchmarking infrastructure
- pip-installable package

**Not yet re-implemented from old version:**
- Well-level statistical aggregation (SSMD, Z-factor)
- Volcano plots
- Batch multi-plate processing (`main_multiplate.py` equivalent)
- Montage generation (replaced by different visualization)

---

## 12. Improvements Relevant to Biology / Medical Collaborators

This section highlights changes that directly impact the scientific reliability and usability of the tool for collaborators who are not software engineers.

### 12.1 Correct Cell-Nucleus Correspondence (Critical Fix)

**Problem in old version:** The old pipeline removed cells without nuclei but did not ensure that cell ID #47 corresponds to nucleus ID #47. Cell and nucleus labels were independently assigned — so when a downstream analysis asked "which nucleus belongs to this cell?", the answer could be wrong.

**Fix:** The new `CellposeSegmenter._match_cells_to_nuclei()` implements greedy matching by nucleus occupancy fraction. After matching, cell_id = nucleus_id for the same physical object. Unmatched nuclei (those without a corresponding cell) are zeroed from the nuclei label map. This guarantees correctness for all downstream analyses (colocalization, per-cell aggregate counts, etc.).

**Impact:** Any analysis that correlates nuclear and cytoplasmic features (e.g., nuclear morphology vs. aggregate load) was potentially unreliable in the old version. The new version ensures this correspondence is correct by construction.

### 12.2 Focus Quality Metrics (New)

**Problem:** Out-of-focus fields produce unreliable segmentation. The old version had no way to detect or flag defocused images — bad fields silently contributed to results.

**Fix:** AggreQuant now computes focus quality metrics for every field:
- **Patch-level:** Variance of Laplacian (most widely used), Laplace energy, Sobel gradient, Brenner gradient, focus score — computed on an 80×80 grid to detect local defocus
- **Global:** Power log-log slope (PLLS, the best single defocus metric per Bray et al. 2012), global Laplacian variance, high-frequency ratio

These metrics are saved in `field_measurements.csv` alongside quantification results, enabling post-hoc quality filtering (e.g., exclude fields with PLLS below a threshold).

**Impact:** Collaborators can filter out unreliable measurements before statistical analysis. Focus metrics per well also reveal systematic plate effects (edge wells often have worse focus).

### 12.3 Plate Heatmaps for All Metrics (New)

**Problem:** The old version's plate-level visualization was limited to density maps generated by the statistics module.

**Fix:** AggreQuant automatically generates Plotly heatmaps for every metric:
- Cell count per well
- Aggregate count per well
- % aggregate-positive cells per well
- Total cell area, aggregate area
- Focus metrics per well

These are interactive HTML files — hover to see exact values, zoom into regions.

**Impact:** Collaborators can visually spot plate effects (edge effects, gradient patterns, contaminated wells) at a glance before diving into statistics.

### 12.4 QC Control Strip Plots (New)

**Problem:** No automated way to check assay quality using control wells.

**Fix:** `qc_plots.py` generates strip plots comparing positive and negative control wells for key metrics. These plots show the distribution of measurements across fields within control wells.

**Impact:** Quick visual check of assay window (separation between positive and negative controls) before full analysis.

### 12.5 Resumable Processing

**Problem:** The old version could not resume from an interruption. If processing failed at image 5000 of 10000, all 5000 completed images had to be reprocessed.

**Fix:** The pipeline checks if mask files exist on disk before processing each field. If `overwrite_masks=False` (default), already-processed fields are skipped. Existing CSV results are loaded on resume.

**Impact:** For large plates (384 wells × 9 fields = 3456 fields), processing can take hours. Resumability means interruptions don't waste completed work.

### 12.6 96-Well Plate Support (New)

**Problem:** The old version was hard-coded for 384-well plates only.

**Fix:** `plate_format` config field supports both "96" and "384". Well ID parsing and plate layout utilities handle both formats.

**Impact:** Collaborators working with different plate formats can use the same tool.

### 12.7 GUI for Non-Programmers (New)

**Problem:** Running the old version required editing Python scripts or YAML files and running from the command line.

**Fix:** CustomTkinter-based GUI (`scripts/run_gui.py`) with configuration widgets and background pipeline execution.

**Impact:** Collaborators who are not comfortable with the command line can use the tool directly.

### 12.8 Single CSV Output

**Problem:** The old version saved per-image text files for quantification, requiring manual aggregation.

**Fix:** All results (quantification + focus metrics) are saved in a single `field_measurements.csv` with columns: plate_name, well_id, field, n_cells, n_nuclei, n_aggregates, n_aggregate_positive_cells, pct_aggregate_positive_cells, total_cell_area_px, total_aggregate_area_px, plus focus metric columns.

**Impact:** Results can be directly loaded into R, Python, Excel, or Prism for downstream statistical analysis without any preprocessing.

### 12.9 Corrected Aggregate Segmentation Bug

**Problem:** The old version called `label()` before `remove_small_holes()` in aggregate segmentation. scikit-image's `remove_small_holes()` expects binary input — passing labeled input silently produces wrong results (holes within labeled objects are not correctly identified).

**Fix:** Correct order: `remove_small_holes()` on binary mask first, then `label()`.

**Impact:** The old version may have over-counted aggregates in some images (holes within aggregates that should have been filled were left unfilled, causing connected aggregates to be split).

---

## 13. Benchmarking Suite

Three systematic benchmarks were conducted to validate design decisions. All use 90–100 curated HCS images (2040×2040 px, 0.325 µm/pixel) sampled from 55 independent 384-well plates, organized into 9 difficulty categories (low/high confluency, clustered, mitotic, defocused, flat-field, low/high intensity, debris). No manual ground truth was available — evaluation is based on inter-model agreement, count concordance, and inference speed.

### 13.1 Nuclei Segmentation Benchmark

**Location:** `benchmarks/nuclei_segmentation/`
**Objective:** Compare 13 pretrained nuclei segmentation models to validate the choice of StarDist.

#### Models tested (13)

**Single-channel (7):**
- StarDist `2D_versatile_fluo`
- Cellpose `nuclei`, `cyto2`, `cyto3` (no nuclear hint)
- DeepCell `NuclearSegmentation`, `Mesmer` (nuclear mode)
- InstanSeg (fluorescence)

**Two-channel with DAPI as membrane hint (6):**
- Cellpose `cyto2`, `cyto3` (with DAPI dual-channel)
- Plus 4 additional two-channel configurations

#### Results

**Detection consistency:**
- StarDist consistently detects the highest or near-highest nuclei count across all 9 categories.
- InstanSeg detects the fewest (sparse detection pattern).
- DeepCell Nuclear produces broader boundaries, sometimes merging adjacent nuclei.

**Inter-model agreement (Coefficient of Variation):**
- Tight convergence on high-confluency (CV ~9%) and flat-field (CV ~15%) images.
- High disagreement on debris/artifacts (median CV ~42%, some images >100%).
- Elevated disagreement on mitotic and defocused fields.

**Inference speed (GPU):**

| Model | Time (s/image) |
|-------|----------------|
| InstanSeg | 0.20 |
| StarDist | 1.23 |
| DeepCell | 2.1–2.4 |
| Cellpose variants | 3.5–5.4 |

27-fold range from fastest to slowest.

#### Background normalization evaluation

The standard AggreQuant preprocessing (`denoised / Gaussian(σ=50)`) was evaluated separately:

- **For StarDist (pipeline model):** Essentially neutral — only ~2% grand median count change. StarDist's internal percentile normalization already handles intensity variation.
- **For Cellpose nuclei:** Beneficial on low-intensity images (+24% more detections), mostly stable otherwise.
- **Across all models:** Increases inter-model disagreement slightly but provides a safety net for spatial illumination gradients.

#### Conclusion

**StarDist `2D_versatile_fluo` is the clear winner** — consistently high detection, fast (1.23 s/image), robust to edge cases. Background normalization is justified: costs StarDist almost nothing and helps Cellpose on dim images.

---

### 13.2 Cell Segmentation Benchmark

**Location:** `benchmarks/cell_segmentation/`
**Objective:** Compare 8 pretrained whole-cell segmentation models, with emphasis on the effect of providing a nuclear channel.

#### Models tested (8)

**Single-channel (FarRed only, 4):**
- Cellpose `cyto2`, `cyto3`
- DeepCell `Mesmer` (zeroed nuclear input)
- InstanSeg fluorescence

**Two-channel (FarRed + DAPI nuclear hint, 4):**
- Cellpose `cyto2_with_nuc`, `cyto3_with_nuc`
- DeepCell `Mesmer_with_nuc`
- InstanSeg with DAPI

#### Results: Nuclear channel dependency

| Model | Single-channel | With DAPI | Assessment |
|-------|---------------|-----------|------------|
| Cellpose cyto2/cyto3 | Good | Modestly better | Designed for single-channel; DAPI is supplementary |
| DeepCell Mesmer | Very few cells | Dramatically better | Nuclear channel is architecturally required |
| InstanSeg | Sparse detections | Dramatically better | Trained on multi-channel tissue data; DAPI expected |
| CellSAM | ~50% undersegmentation | Substantially better | Benefits strongly from nuclear channel |

#### Literature review findings

- **Mesmer** was trained on TissueNet (tissue sections with paired DAPI/Histone H3). The nuclear channel is not optional — it is the primary seed signal.
- **InstanSeg** was trained on CPDMI (FFPE tissue, 8–32 channels, always including DAPI). Developers state: "If you only have one channel, you should probably only choose nuclei as the output."
- **Cellpose** training data explicitly includes fluorescence cell culture with single-channel cytoplasm — the only model validated for this use case.

#### Conclusion

**Only Cellpose (cyto2, cyto3) is designed and validated for single-channel cytoplasm fluorescence.** Mesmer, InstanSeg, and CellSAM fundamentally depend on a nuclear stain for reliable whole-cell segmentation. When given their intended two-channel inputs, they can perform competitively, but this requires DAPI — which is not always available as a separate raw channel in all pipeline configurations.

---

### 13.3 Preprocessing / Cellpose Segmentation Benchmark

**Location:** `benchmarks/preprocessing_cellpose_segmentation/`
**Objective:** Validate the AggreQuant pipeline design choice of feeding Cellpose a **binary StarDist nuclei mask** rather than raw DAPI or no nuclear input.

#### Three variants tested (all Cellpose cyto3)

| Variant | Nuclear Input | Description |
|---------|--------------|-------------|
| `cellpose_cell_only` | None | Single-channel FarRed only |
| `cellpose_raw_nuclei` | Raw DAPI image | Two-channel with unprocessed DAPI |
| `cellpose_nuclei_seeds` | StarDist binary mask | Two-channel with pre-segmented nuclei |

#### Results

**Cell counts (mean per category, selected categories):**

| Category | Cell-Only | Raw DAPI | Seeds (pipeline) |
|----------|-----------|----------|-------------------|
| Low confluency | 318 | 373 | **440** |
| High confluency | 0 | 0 | **1399** |
| Clustered | 159 | 0 | **1874** |
| Mitotic | 268 | 272 | **724** |
| Defocused | 447 | 871 | **943** |
| Flat-field | 772 | 1415 | **1327** |
| Low intensity | 563 | 780 | **783** |
| High intensity | 281 | 1619 | **1504** |
| Debris | 262 | 726 | **874** |

**Count difference summary:**
- Seeds − Cell only: mean +760 cells/image
- Seeds − Raw DAPI: mean +420 cells/image
- Seeds wins in every category

**Shape quality (solidity, 0–1):**
- Seeds: consistently ~0.92 across all categories
- Cell-only: variable (0.0–0.93); fails completely in high-confluency
- Raw DAPI: intermediate

**Inference time:**
- Cell only: 5.4 s, Raw DAPI: 5.2 s, Seeds: 7.0 s (includes StarDist preprocessing)

#### Conclusion

**Using StarDist binary masks as nuclear seeds dramatically outperforms both alternatives.** The seeds variant produces the highest cell counts (especially in dense/clustered fields where other methods fail completely), the highest and most consistent boundary quality, and is effective across all difficulty categories. The binary mask provides clean, denoised nuclear locations without intensity gradients, background noise, or out-of-focus blur that raw DAPI contains. This validates the AggreQuant pipeline's core architectural choice of pre-segmenting nuclei before cell segmentation.

---

### 13.4 Mini Cellpose from Scratch

**Location:** `benchmarks/mini_cellpose_from_scratch/`
**Objective:** Educational reimplementation of the Cellpose flow-based instance segmentation algorithm in PyTorch.

#### Implementation

- **Flow targets:** Heat diffusion from cell centers → spatial gradients → unit vector flow fields
- **Network:** Small UNet (4 encoder levels, ~2.5M params) with a style vector (global average pool of bottleneck, L2-normalized, projected and added to decoder)
- **Output:** 3 channels — flow_y, flow_x, cell_probability
- **Inference:** Euler-integrate predicted flow (200 steps via `grid_sample`) → pixels converge to cell centers → histogram peak detection → dilate for full masks
- **Training data:** 90 HCS FarRed images with Cellpose cyto3 pseudo ground truth (no manual annotations)

#### Purpose

This is not a production model — it is a minimal, self-contained reference implementation for understanding how flow-based instance segmentation works. Demonstrates that the core algorithm can be implemented in ~500 lines of PyTorch.

---

### 13.5 Overall Benchmark Conclusions

The three evaluation benchmarks collectively validate three key AggreQuant design decisions:

1. **StarDist for nuclei:** Best combination of detection accuracy, speed, and robustness among 13 models tested.
2. **Cellpose for cells:** The only model family designed for single-channel cytoplasm fluorescence in cell culture. All alternatives (Mesmer, InstanSeg, CellSAM) require a nuclear stain channel they were trained with.
3. **Binary mask preprocessing:** Feeding Cellpose pre-segmented StarDist masks instead of raw DAPI dramatically improves cell detection — especially in challenging categories (high confluency, clustered) where other input configurations fail completely.
