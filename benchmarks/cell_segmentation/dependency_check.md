# Dependency Check: CellSAM and Micro-SAM vs. cell-bench Environment

**Date:** 2026-02-26
**Context:** Assessing whether CellSAM and Micro-SAM can be installed into the existing
`cell-bench` conda environment alongside Cellpose, DeepCell (Mesmer), and InstanSeg.

---

## Current cell-bench Environment (Key Packages)

| Package | Version | Notes |
|---|---|---|
| Python | >=3.10 | |
| torch | 2.10.0 | CUDA 12.8 |
| tensorflow | 2.8.4 | Required by DeepCell |
| protobuf | 3.20.3 | Pinned for TF 2.8 compat |
| numpy | 1.26.4 | |
| scikit-image | 0.25.2 | |
| scikit-learn | 1.7.2 | |
| scipy | 1.15.3 | |
| cellpose | 3.1.1.3 | |
| DeepCell | 0.12.10 | |
| instanseg-torch | 0.1.1 | |

---

## CellSAM — Likely installable into cell-bench

**Source:** https://github.com/vanvalenlab/cellSAM (not on PyPI)
**Version:** 0.0.dev1 (pre-release)

### Dependency Compatibility

| Dependency | cell-bench status | Issue? |
|---|---|---|
| Python >=3.10 | Already satisfied | No |
| torch (unconstrained) | 2.10.0 installed | No |
| **torchvision** | **Not installed** | **Need to add** (must match torch 2.10.0) |
| numpy | 1.26.4 installed | No |
| scikit-image | 0.25.2 installed | No |
| scikit-learn | 1.7.2 installed | No |
| requests, tqdm, pyyaml | Already installed | No |
| **segment-anything** | **Not installed** | **GitHub-only install** |
| **kornia** | **Not installed** | New, pure PyTorch — low conflict risk |
| **dask[distributed] + dask-image** | **Not installed** | Heavyweight but no version pins |

### Verdict

**Yes, should work.** Three new packages needed (`torchvision`, `segment-anything`, `kornia`)
plus dask. No version pins means nothing should conflict with the existing TensorFlow 2.8 +
protobuf 3.20.3 constraint. The main risk is `torchvision` needing to match `torch==2.10.0`
exactly.

### Install Commands

```bash
conda activate cell-bench
pip install torchvision --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/vanvalenlab/cellSAM.git
```

---

## Micro-SAM — Requires a Separate Environment

**Source:** https://github.com/computational-cell-analytics/micro-sam (conda-forge only)
**Version:** 1.7.4

### Dependency Compatibility

| Dependency | cell-bench status | Issue? |
|---|---|---|
| Python >=3.10 | Already satisfied | No |
| torch >=2.4.0 | 2.10.0 installed | No |
| **napari >=0.5, <0.7** | **Not installed** | **Heavy GUI framework, huge dep tree** |
| **torch_em >=0.7.10** | **Not installed** | Niche conda package |
| **python-elf >=0.6.1** | **Not installed** | Niche conda package |
| **nifty >=1.2.3** | **Not installed** | C++ graph partitioning lib |
| **timm** | **Not installed** | PyTorch Image Models |
| segment-anything | Not installed | Same as CellSAM |
| kornia | Not installed | Same as CellSAM |

### Verdict

**Do not install into cell-bench.** Three problems:

1. **Conda-only distribution** — no PyPI package. `conda install -c conda-forge micro_sam`
   will re-solve the entire environment. The existing TensorFlow 2.8.4 + protobuf 3.20.3
   pin (required for DeepCell) will almost certainly break or force downgrades.

2. **napari is mandatory** — drags in Qt, pyqt, and ~30+ transitive GUI dependencies.
   High conflict surface with the existing environment.

3. **torch_em + python-elf + nifty** — niche conda-forge packages with C extension builds.
   These frequently cause solver conflicts in complex environments.

### Recommended Install (Separate Environment)

```bash
conda create -n micro-sam -c conda-forge python=3.11 micro_sam
```

---

## Summary

| Model | Install into cell-bench? | Method |
|---|---|---|
| CellSAM | Yes (low risk) | `pip install git+https://github.com/vanvalenlab/cellSAM.git` |
| Micro-SAM | No (high conflict risk) | Separate conda env via `conda install -c conda-forge micro_sam` |
