"""UI constants and defaults for the AggreQuant web GUI."""

# Application
APP_TITLE = "AggreQuant"
APP_HOST = "127.0.0.1"
APP_PORT = 8050

# Plate grid
WELL_SIZE_96 = 46      # px per well for 96-well plate
WELL_SIZE_384 = 26     # px per well for 384-well plate

# Control well colours
CONTROL_COLORS = {
    "NT": "#2ecc71",
    "rab13": "#3498db",
}
CUSTOM_CONTROL_COLOR = "#f39c12"
SELECTED_COLOR = "#f1c40f"
EMPTY_COLOR = "#ecf0f1"

# Default control well assignments for 384-well plates
# NT:    A-H col 5,  I-P col 13
# rab13: A-H col 13, I-P col 5
DEFAULT_CONTROL_ASSIGNMENTS = {}
for _r in range(8):       # rows A-H (0-7)
    from aggrequant.loaders.plate import indices_to_well_id as _iw
    DEFAULT_CONTROL_ASSIGNMENTS[_iw(_r, 4)] = "NT"       # col 5 (0-indexed: 4)
    DEFAULT_CONTROL_ASSIGNMENTS[_iw(_r, 12)] = "rab13"   # col 13 (0-indexed: 12)
for _r in range(8, 16):   # rows I-P (8-15)
    DEFAULT_CONTROL_ASSIGNMENTS[_iw(_r, 12)] = "NT"      # col 13
    DEFAULT_CONTROL_ASSIGNMENTS[_iw(_r, 4)] = "rab13"    # col 5
del _r, _iw

# Default channel configuration
DEFAULT_CHANNELS = [
    {"name": "DAPI", "pattern": "390", "purpose": "nuclei"},
    {"name": "GFP", "pattern": "473", "purpose": "aggregates"},
    {"name": "CellMask", "pattern": "631", "purpose": "cells"},
]

# Available focus metrics (from aggrequant.focus)
PATCH_METRIC_OPTIONS = [
    "VarianceLaplacian", "LaplaceEnergy", "Sobel", "Brenner", "FocusScore",
]
GLOBAL_METRIC_OPTIONS = [
    "power_log_log_slope", "global_variance_laplacian", "high_freq_ratio",
]

# Aggregate methods
AGGREGATE_METHODS = ["filter", "unet"]

# CSS colours
COLOR_PRIMARY = "#2c3e50"
COLOR_ACCENT = "#3498db"
COLOR_SUCCESS = "#27ae60"
COLOR_DANGER = "#e74c3c"
COLOR_WARNING = "#f39c12"
COLOR_BG = "#f8f9fa"
COLOR_CARD = "#ffffff"

# Progress polling interval (ms)
PROGRESS_INTERVAL_MS = 500
