# AggreQuant

Automated aggregate quantification for High Content Screening (HCS) image analysis.

## Overview

AggreQuant is a Python package for automated analysis of High Content Screens, specifically designed for quantifying α-synuclein protein aggregates in live-cell fluorescence microscopy data.

The input image data are assumed to be generated from HCS plates (96 or 384 wells), with multiple fields of view per well and 3 channels per field:
- **Nuclei** (Blue, 390nm)
- **Cells** (FarRed, 640nm)
- **Aggregates** (Green, 473nm)

For a 384-well plate with 9 fields per well, 10,368 images are processed to quantify aggregate-positive cells.

<IMG SRC="graphics/pipeline.jpg" style="float: left; margin-right: 10px;" />

## Features

- **Nuclei Segmentation**: StarDist pre-trained deep neural network
- **Cell Segmentation**: Cellpose or distance-intensity algorithm
- **Aggregate Segmentation**:
  - Filter-based method (calibrated image processing filters)
  - Neural network method (PyTorch UNet with modular architecture)
- **Focus Quality Assessment**: Blur detection to exclude out-of-focus regions
- **Colocalization Analysis**: Characterize aggregate inclusions in cells
- **Export**: Statistics in Parquet, CSV, or Excel format
- **GUI**: User-friendly interface for biologists to configure and run analyses

<IMG SRC="graphics/segmentation.jpg" style="float: left; margin-right: 10px;" />

## Installation

### Conda Environment

```bash
# Create and activate environment
conda create -n AggreQuant python=3.11
conda activate AggreQuant

# Install core dependencies
pip install numpy scikit-image tifffile pandas pyarrow pyyaml pydantic click tqdm matplotlib

# Install the package in development mode
pip install -e .
```

### Segmentation Dependencies (GPU)

StarDist and Cellpose require TensorFlow and PyTorch with GPU support for optimal performance.

**Prerequisites:**
- NVIDIA GPU with CUDA support
- NVIDIA Driver ≥ 525.60.13 (Linux)

**Step 1: Install TensorFlow with CUDA support**
```bash
pip install 'tensorflow[and-cuda]'

# Verify GPU detection
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

**Step 2: Install StarDist**
```bash
pip install stardist
```

**Step 3: Install Cellpose** (uses PyTorch, v3 API required)
```bash
pip install "cellpose>=3.0,<4.0"

# Verify Cellpose GPU detection
python -c "from cellpose import core; print('Cellpose GPU:', core.use_gpu())"
```

### Other Optional Dependencies

```bash
# For training neural networks
pip install torch torchvision albumentations segmentation-models-pytorch tensorboard

# For the GUI
pip install customtkinter
```

## Usage

### GUI (Recommended for Biologists)

Launch the graphical interface:

```bash
python scripts/run_gui.py
```

The GUI allows you to:
- Select input/output directories
- Choose plate format (96 or 384-well)
- Assign control wells (NT, negative) by clicking on the plate grid
- Configure analysis parameters (blur threshold, segmentation method)
- Monitor analysis progress in real-time
- Save/load configuration files for reproducibility

### Command Line

Run analysis from a YAML configuration file:

```bash
python scripts/run_pipeline.py config.yaml
```

Or specify options directly:

```bash
python scripts/run_pipeline.py --input /data/plate1 --output /results/plate1 \
    --plate-format 384 --method unet --blur-threshold 15.0
```

### As a Python Package

```python
from aggrequant import run_pipeline_from_config, run_pipeline_from_dict

# Run from config file
result = run_pipeline_from_config("config.yaml")

# Or run from dict
config = {
    "input_dir": "/data/plate1",
    "output_dir": "/results/plate1",
    "plate_format": "96",
    "aggregate_method": "unet",
    "blur_threshold": 15.0,
    "control_wells": {"A01": "negative", "A02": "NT"},
}

result = run_pipeline_from_dict(config)

print(f"Processed {result.total_n_wells_processed} wells")
print(f"Total cells: {result.total_n_cells}")
print(f"SSMD: {result.ssmd:.3f}")
```

### Configuration File Format

Create a YAML configuration file for reproducible analyses:

```yaml
# config.yaml
input_dir: /path/to/images
output_dir: /path/to/output
plate_format: "96"

# Segmentation settings
aggregate_method: unet  # or "filter"
model_path: /path/to/unet_weights.pt  # for unet method

# Quality control
blur_threshold: 15.0
blur_reject_pct: 50.0

# Control wells
control_wells:
  negative:
    - A01
    - A02
  NT:
    - A11
    - A12

# Output options
save_masks: true
save_overlays: true
export_format: parquet  # or "csv", "excel"
```

## Project Structure

```
AggreQuant/
├── aggrequant/              # Main package
│   ├── common/              # Shared utilities (image_utils, logging)
│   ├── loaders/             # Data loading (config, images, plate)
│   ├── quality/             # Image quality (focus/blur detection)
│   ├── segmentation/        # Segmentation backends
│   │   ├── nuclei/          # StarDist wrapper
│   │   ├── cells/           # Cellpose, distance-intensity
│   │   └── aggregates/      # Filter-based, neural network
│   ├── quantification/      # QoI calculations
│   ├── statistics/          # Well stats, export
│   ├── pipeline.py          # Main pipeline orchestrator
│   └── nn/                  # Neural network development
│       ├── architectures/   # Modular UNet with pluggable blocks
│       ├── data/            # Dataset, augmentation
│       ├── training/        # Losses, trainer
│       └── evaluation/      # Metrics, benchmarking
├── gui/                     # GUI application
│   ├── app.py               # Main application window
│   └── widgets/             # Custom UI components
├── scripts/                 # Entry point scripts
│   ├── run_pipeline.py      # CLI for running analysis
│   └── run_gui.py           # Launch GUI application
├── tests/                   # Unit and integration tests
├── PROJECT.md               # Detailed project documentation
└── pyproject.toml           # Package configuration
```

## Quantities of Interest (QoI)

AggreQuant computes various metrics including:
- Percentage of aggregate-positive cells
- Number of aggregates per cell
- Total aggregate area over cell area
- Focus quality metrics (variance of Laplacian, etc.)

<IMG SRC="graphics/raw_and_segmentation.jpg" style="float: left; margin-right: 10px;" />

## Development

See [PROJECT.md](PROJECT.md) for detailed development documentation including:
- Architecture design decisions
- Implementation phases and task lists
- Code style guidelines
- Literature review of UNet architectures

### Running Tests

```bash
conda activate AggreQuant
pytest tests/
```

## Author

**Athena Economides, PhD**  
Prof. Adriano Aguzzi Lab  
Institute of Neuropathology  
University of Zurich & University Hospital Zurich  
Schmelzbergstrasse 12  
CH-8091 Zurich  
Switzerland  

Contact: [athena.economides@uzh.ch](mailto:athena.economides@uzh.ch)

## License

This project is under development. Please contact the author for usage permissions.
