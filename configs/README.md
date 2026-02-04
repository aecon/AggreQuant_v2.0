# AggreQuant Configuration Files

This directory contains YAML configuration files for reproducible plate analysis.

## Quick Start

1. **Place your plate images** in the `data/plate1/` directory (or adjust `input_dir` in config)

2. **Edit the configuration** to match your data:
   - Update channel patterns to match your filenames
   - Adjust control well positions
   - Tune segmentation parameters if needed

3. **Run the analysis**:
   ```bash
   conda activate AggreQuant
   python scripts/run_plate_analysis.py configs/plate1_384well.yaml
   ```

## Configuration Files

| File | Description |
|------|-------------|
| `plate1_384well.yaml` | 384-well plate with StarDist/Cellpose/Filter segmentation |

## Image File Naming

The channel patterns in the config should match substrings in your filenames:

**Example filenames:**
```
Plate1_A01(fld 1 wv 390 - Blue).tif     # Matches pattern "390" (nuclei)
Plate1_A01(fld 1 wv 473 - Green).tif    # Matches pattern "473" (aggregates)
Plate1_A01(fld 1 wv 631 - FarRed).tif   # Matches pattern "631" (cells)
```

**Alternative naming (ImageXpress):**
```
Plate1_A01_s1_w1.tif    # Matches pattern "w1"
Plate1_A01_s1_w2.tif    # Matches pattern "w2"
Plate1_A01_s1_w3.tif    # Matches pattern "w3"
```

## Command-Line Options

```bash
# Basic usage
python scripts/run_plate_analysis.py configs/plate1_384well.yaml

# Verbose output
python scripts/run_plate_analysis.py configs/plate1_384well.yaml --verbose

# Process specific wells only
python scripts/run_plate_analysis.py configs/plate1_384well.yaml --wells A01 A02 B01 B02

# Dry run (validate config without processing)
python scripts/run_plate_analysis.py configs/plate1_384well.yaml --dry-run

# Disable GPU (run on CPU)
python scripts/run_plate_analysis.py configs/plate1_384well.yaml --no-gpu
```

## Output Files

Results are saved to the `output_dir` specified in the config:

```
output/plate1/
├── plate1_well_results.csv      # Well-level statistics
├── plate1_field_results.csv     # Field-level statistics
├── plate1_well_results.parquet  # Binary format (for large datasets)
├── plate1_summary.txt           # Plate summary
└── masks/                       # Segmentation masks (if enabled)
    ├── A01/
    │   ├── f1_nuclei.tif
    │   ├── f1_cells.tif
    │   └── f1_aggregates.tif
    └── ...
```

## Key Configuration Parameters

### Segmentation Thresholds

```yaml
segmentation:
  # Nuclei (StarDist)
  nuclei_prob_thresh: 0.5      # Detection probability threshold
  nuclei_nms_thresh: 0.4       # Non-max suppression overlap

  # Cells (Cellpose)
  cell_flow_threshold: 0.4     # Flow field threshold
  cell_cellprob_threshold: 0.0 # Cell probability threshold

  # Aggregates (Filter-based)
  aggregate_intensity_threshold: 1.6  # Lower = more sensitive
  aggregate_min_size: 9               # Minimum aggregate size (pixels)
```

### Quality Control

```yaml
quality:
  focus_blur_threshold: 15.0     # Variance of Laplacian threshold
  focus_reject_threshold: 50.0   # Reject if >50% patches are blurry
```

## Creating New Configurations

Copy an existing config and modify:

```bash
cp configs/plate1_384well.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your settings
python scripts/run_plate_analysis.py configs/my_experiment.yaml
```
