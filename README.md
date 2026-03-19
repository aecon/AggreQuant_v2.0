# AggreQuant v.2.0

**!! WORK IN PROGRESS !!**


A refactoring of the [AggreQuant](https://github.com/aecon/AggreQuant) codebase.  



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

### Command Line

Run analysis from a YAML configuration file:

```bash
python scripts/run_pipeline.py configs/test_384well.yaml
```

Available options:

```bash
python scripts/run_pipeline.py configs/test_384well.yaml --verbose          # Enable detailed logging
python scripts/run_pipeline.py configs/test_384well.yaml --max-fields 5     # Process only 5 fields (quick test)
python scripts/run_pipeline.py configs/test_384well.yaml --segmentation-only  # Skip quantification and plots
```

See `configs/test_384well.yaml` for a documented example of all configuration options.

### Web GUI

```bash
python scripts/run_gui.py
```

Launches a Dash-based web interface for configuring and running the pipeline.

### As a Python Package

```python
from aggrequant.pipeline import SegmentationPipeline

pipeline = SegmentationPipeline(config_path="configs/test_384well.yaml", verbose=True)
pipeline.run()
```

### Unit Tests

```bash
conda activate AggreQuant
pytest tests/
```

