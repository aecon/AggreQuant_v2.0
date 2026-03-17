# predict_and_plot.py

Visualize predictions of a trained model on an image.

---

## Output

A 2-panel figure:

| Panel | Content |
|---|---|
| 1 | Contrast-enhanced raw image (1st–99th percentile scaling) |
| 2 | Thresholded prediction vs ground truth overlay |

### Overlay colors (panel 2)

| Color | Meaning |
|---|---|
| Yellow | Overlap — both prediction and GT (TP) |
| Magenta | Prediction only (FP) |
| Cyan | GT only (FN) |

Without a ground truth mask, panel 2 shows the prediction in green on a black background.

### Interactive features

The figure always opens an interactive matplotlib window (after saving, if applicable):

- **Linked axes** — zoom and pan on one panel, the other follows
- **Crosshair cursor** — hover over either panel to see a white crosshair on both, highlighting the same location

---

## Usage

```bash
# Minimal — just the checkpoint (uses first image, auto-finds mask, saves + shows interactive):
conda run -n AggreQuant python scripts/predict_and_plot.py \
    training_output/loss_function/dice03_bce07_pw3/checkpoints/best.pt

# Pick image by index (0-based):
conda run -n AggreQuant python scripts/predict_and_plot.py \
    training_output/loss_function/dice03_bce07_pw3/checkpoints/best.pt --image 3

# Pick image by filename:
conda run -n AggreQuant python scripts/predict_and_plot.py \
    training_output/loss_function/dice03_bce07_pw3/checkpoints/best.pt --image image_0003.tif

# Custom threshold:
conda run -n AggreQuant python scripts/predict_and_plot.py \
    training_output/loss_function/dice03_bce07_pw3/checkpoints/best.pt --threshold 0.4

# Override all auto-resolved paths:
conda run -n AggreQuant python scripts/predict_and_plot.py \
    checkpoint.pt --image /path/to/image.tif --mask /path/to/mask.tif -o output.png

# Interactive only (skip saving):
conda run -n AggreQuant python scripts/predict_and_plot.py \
    training_output/loss_function/dice03_bce07_pw3/checkpoints/best.pt --no-save
```

---

## Options

| Argument | Required | Default | Description |
|---|---|---|---|
| `checkpoint` | Yes | — | Path to `.pt` checkpoint file |
| `--image` | No | First image in `symlinks/images/` | Image path, filename, or integer index |
| `--mask` | No | Auto-resolved from `symlinks/masks/` (same filename) | Path to ground truth mask |
| `--threshold` | No | `0.5` | Probability threshold for binarization |
| `-o` / `--output` | No | `training_output/<run>/prediction_<stem>.png` | Output path for saved figure |
| `--no-save` | No | `False` | Skip saving, only show interactive window |

---

## Path auto-resolution

The script resolves paths relative to `training_output/`:

- **Image**: `--image` accepts a full path, a filename (resolved from `training_output/symlinks/images/`), or an integer index into the sorted image list. Default: first image.
- **Mask**: Auto-discovered from `training_output/symlinks/masks/` using the same filename as the image. Skipped silently if not found.
- **Output**: Saved next to the checkpoint's parent directory as `prediction_<image_stem>.png`.

By default, the figure is both saved and shown interactively. Use `--no-save` to skip saving.
