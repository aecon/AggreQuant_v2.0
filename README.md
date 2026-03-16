# AggreQuant v.2.0

A refactoring of the [AggreQuant](https://github.com/aecon/AggreQuant) codebase.
**WORK IN PROGRESS - non functioning code.**



## Installation

### Conda Environment

```bash
# Create and activate environment
conda create -n AggreQuant python=3.11
conda activate AggreQuant

# Install core dependencies
pip install numpy scikit-image opencv-python-headless tifffile pandas pyarrow pyyaml pydantic click tqdm matplotlib

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

### GUI

```bash
python scripts/run_gui.py
```

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


### Unit Tests

```bash
conda activate AggreQuant
pytest tests/
```

