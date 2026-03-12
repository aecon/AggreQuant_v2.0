# Mini Cellpose From Scratch

Educational reimplementation of the [Cellpose](https://github.com/MouseLand/cellpose) flow-based cell instance segmentation algorithm in PyTorch.

## Algorithm

1. **Flow targets** (training): For each cell instance, run heat diffusion from its center, compute spatial gradients, normalize to unit vectors. Each pixel gets a 2D flow vector pointing toward its cell's center.
2. **Network**: Small UNet (4 encoder levels, ~2.5M params) with a **style vector** — global average pool of the bottleneck features, L2-normalized, linearly projected and added to each decoder level.
3. **Output**: 3 channels — flow_y, flow_x, cell_probability.
4. **Inference**: Euler-integrate the predicted flow field (200 steps via `grid_sample`). Pixels converge to cell centers. Build a histogram of final positions, find peaks, dilate to recover full masks.

## Setup

```bash
conda env create -f environment.yml
conda activate mini_cellpose_AE
```

### Data

Create symlinks (already done if you cloned the repo):
```bash
# Images: HCS FarRed TIFFs in category subdirectories
ln -s /path/to/NUCLEI-BENCHMARK_AE-CURATED-2026-02-19 data/images

# Masks: Cellpose cyto3 pseudo-GT label masks (flat directory)
ln -s /path/to/cell_segmentation/results/masks/cellpose_cyto3 data/masks
```

## Usage

### Train
```bash
conda run -n mini_cellpose_AE python train.py \
    --image-dir data/images \
    --mask-dir data/masks \
    --epochs 200 --batch-size 8 --lr 1e-3
```

Flow targets are computed once on first run and cached in `flow_cache/`.

### Predict
```bash
conda run -n mini_cellpose_AE python predict.py \
    --checkpoint checkpoints/best.pt \
    --image-dir data/images \
    --output-dir results/masks
```

## File Structure

| File | Description |
|------|-------------|
| `dynamics.py` | Heat diffusion, Euler integration, histogram-based mask recovery |
| `model.py` | UNet with style vector (ConvBlock, StyleBlock, MiniCellposeUNet) |
| `dataset.py` | PyTorch Dataset with flow caching, normalization, augmentation |
| `train.py` | Training loop (MSE on flows + BCE on cell prob) |
| `predict.py` | Inference pipeline: image -> flows -> masks |

## Notes

- Trained on 90 HCS images using Cellpose cyto3 masks as pseudo ground truth (no manual annotations)
- The model is intentionally small and simple — this is an educational benchmark, not a production model
- No diameter rescaling (all images are from the same microscope at the same resolution)
