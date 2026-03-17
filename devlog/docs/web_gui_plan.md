# AggreQuant Web GUI Plan

## Overview

Replace the current CustomTkinter desktop GUI with a Dash-based web interface served over HTTP. This brings a modern browser UI, better interactivity, and consistency with the PlateViewer project.

## Framework Choice: Dash

Dash (Plotly) over Streamlit because:

- **Interactive plate grid** needs JavaScript callbacks for click-to-select wells; Dash handles this natively, Streamlit would require a custom component
- **Consistency** with PlateViewer (same stack)
- **Layout control** — sidebar + main area + progress panel maps cleanly to Dash divs
- **Callbacks** — pipeline progress, live log updates, and conditional UI state are natural in Dash

## Current GUI Features (CustomTkinter)

What exists today in `gui/`:

| Feature | Status |
|---------|--------|
| Plate selector (96/384) with click/drag | Works |
| Control well assignment (negative, NT, custom) | Works |
| Input/output directory selection | Works |
| Plate format selector | Works |
| Aggregate method selector (UNet/filter) | Works |
| UNet model dropdown | Hardcoded list, may be outdated |
| Blur threshold / reject % sliders | Works |
| Output options (masks, overlays, stats) | Works |
| Config load/save (YAML) | Works |
| Pipeline execution with progress + log | Works, but imports `AggreQuantPipeline` (should be `SegmentationPipeline`) |
| Channel configuration | Missing — hardcoded DAPI/GFP/CellMask |
| Focus metric settings | Missing |
| Results visualization | Missing |

## New Web GUI Features

### 1. Configuration Tab

**Plate & Input/Output:**
- Input directory browser (or text input)
- Output directory browser (or text input)
- Plate format selector: 96-well / 384-well
- Plate name (auto-derived from directory, editable)

**Channel Configuration:**
- Dynamic channel list (add/remove rows)
- Each channel: name, filename pattern (wavelength), purpose (nuclei/cells/aggregates)
- No longer hardcoded — matches what the YAML config supports

**Segmentation Settings:**
- Aggregate method: filter-based / UNet
- UNet model: dropdown populated from actual available checkpoints in `training_output/symlinks/`
- Nuclei parameters: sigma denoise, sigma background, min/max area
- Cell model selector (Cellpose model name)
- Aggregate parameters: intensity threshold, min size

**Quality Control:**
- Focus metric toggle (enable/disable)
- Metric selection: VarianceLaplacian, power_log_log_slope, etc.
- Patch size configuration
- Compute on: nuclei, cells, or both

**Output Options:**
- Save segmentation masks (checkbox)
- Overwrite existing masks (checkbox)
- Output subdirectory name

**Config Persistence:**
- Load config from YAML
- Save current config to YAML

### 2. Plate Selector Tab

**Interactive Well Grid:**
- Visual plate grid (96 or 384 wells)
- Click to toggle individual wells
- Drag-select for rectangular regions
- Color-coded by control assignment

**Control Well Assignment:**
- Predefined types: negative, NT
- Custom control types (user-defined)
- Assign selected wells to a control type
- Clear selection / clear all assignments

**Text-Based Well Specification:**
- Flexible syntax input (e.g., `A-H:5, col:1-2, A01-A09`)
- Parsed and reflected on the grid
- Complements click-selection for power users

### 3. Batch Processing

**Multi-Plate Queue:**
- Add multiple input directories to a processing queue
- Each directory = one plate
- List view showing queued plates with status (pending / running / done / failed)
- Add/remove plates from the queue
- Option to apply the same config to all plates or load per-plate YAML configs
- Sequential execution: plates processed one after the other
- Global progress (plate N of M) + per-plate progress (field X of Y)
- Ability to cancel current plate or the entire queue

**Batch Input Methods:**
- Browse and add directories one by one
- Paste a list of directory paths (one per line)
- Browse a parent directory and auto-detect plate subdirectories

### 4. Progress & Log Tab

**Progress Tracking:**
- Overall batch progress bar (plate N of M)
- Current plate progress bar (field X of Y)
- Current status message
- Elapsed time display

**Live Log:**
- Scrollable log output with timestamps
- Log levels: INFO, WARN, ERROR
- Auto-scroll to latest
- Clear log button

**Run Controls:**
- Run / Cancel buttons
- Cancel current plate vs. cancel entire batch

### 5. Results Tab

**Post-Pipeline Visualization (after run completes):**

- **Plate Heatmaps**: interactive Plotly heatmaps for each quantified metric (n_cells, n_aggregates, pct_aggregate_positive_cells, etc.) — reuse `aggrequant.visualization.heatmaps`
- **QC Strip Plots**: control well comparisons (negative vs NT) — reuse `aggrequant.visualization.qc_plots`
- **Field Measurements Table**: searchable/sortable table of `field_measurements.csv`
- **Plate selector** to switch between plates when batch processing
- **Export**: save all plots as PNGs

### 6. Image Preview Tab (optional, lower priority)

- Thumbnail contact sheet of loaded plate (one thumbnail per well)
- Click a well to see all fields
- Useful for quick visual QC before running pipeline

## Architecture

```
aggrequant/
    gui_web/
        __init__.py
        app.py              # Dash app setup, layout, main()
        config.py            # UI constants (colors, sizes, defaults)
        callbacks/
            __init__.py
            configuration.py # Config tab callbacks
            plate_selector.py# Well grid callbacks
            batch.py         # Batch processing callbacks
            progress.py      # Progress + log callbacks
            results.py       # Results visualization callbacks
        components/
            __init__.py
            plate_grid.py    # Interactive well plate component
            settings_form.py # Configuration form components
            batch_queue.py   # Batch queue list component
            log_panel.py     # Log display component
        runner.py            # Pipeline execution (background thread/process)
```

**Key Design Decisions:**
- Callbacks separated by tab/feature for maintainability
- Components are reusable Dash layout builders
- Pipeline runs in a background thread; progress communicated via `dcc.Interval` polling or server-sent events
- Config dict ↔ YAML round-trip via existing `PipelineConfig` dataclasses
- Plate grid uses Plotly figure with click events (like PlateViewer contact sheet pattern)

## Entry Point

```
scripts/run_gui.py  →  updated to launch Dash app instead of CustomTkinter
```

Or a new script:
```
scripts/run_web_gui.py
```

CLI: `aggrequant-gui` console script entry point in `pyproject.toml`.

## Dependencies to Add

```
dash>=4.0
plotly>=5.0
```

Already in the environment from PlateViewer. Remove `customtkinter` dependency.

## Migration Notes

- The old `gui/` directory (CustomTkinter) will be replaced by `aggrequant/gui_web/`
- `gui/pipeline_runner.py` logic moves to `aggrequant/gui_web/runner.py`, fixing the `SegmentationPipeline` import
- Config load/save stays YAML-based, using the same schema as `configs/test_384well.yaml`

---

## Implementation Status (2026-03-17)

### Completed

All files have been created and the app imports and creates successfully.

**Files created:**
```
aggrequant/gui_web/
    __init__.py
    app.py                     # Dash app, layout, main(), entry point
    config.py                  # UI constants, colors, defaults
    runner.py                  # Background pipeline runner (threading + shared state)
    components/
        __init__.py
        plate_grid.py          # Interactive Plotly well plate with click-to-select
        settings_form.py       # Full config form (channels, segmentation, quality, output)
        batch_queue.py         # Multi-plate queue with 3 input methods
        log_panel.py           # Progress bars + live log panel
        results_panel.py       # Heatmaps, QC plots, measurements table
    callbacks/
        __init__.py
        configuration.py       # Load/save YAML, toggle UNet model row
        plate_selector.py      # Well click, text-based well spec, control assignment
        batch.py               # Add/paste/scan plates, clear queue
        progress.py            # Run/cancel, poll progress, collect GUI config
        results.py             # Heatmap rendering, QC plot, measurements table, export
```

**pyproject.toml updated:**
- `gui` optional dep: `dash>=4.0, plotly>=5.0` (was `customtkinter>=5.2`)
- `aggrequant-gui` entry point: `aggrequant.gui_web.app:main` (was `gui.app:main`)
- `setuptools.packages.find`: only `aggrequant*` (removed `gui*`)

**How to run:**
```bash
conda run -n AggreQuant python -m aggrequant.gui_web.app
# or after pip install -e .:
aggrequant-gui
# Options: --port 8050 --no-browser --debug
```

### Known TODOs / Remaining Work

1. **Dynamic channel rows**: The channel config form renders 3 default rows but the callbacks for add/remove channel rows (pattern-matching callbacks with `ALL`) are not yet wired. Currently falls back to `DEFAULT_CHANNELS` when saving config or running pipeline.

2. **Batch remove selected**: The "Remove Selected" button in the batch tab needs a callback using pattern-matching on `batch-select` checkboxes.

3. **Testing with real data**: The app needs end-to-end testing with actual plate data to verify:
   - Config load/save round-trip
   - Pipeline execution with progress tracking
   - Results visualization after pipeline completes

4. **Well drag-select**: Currently only click-to-toggle individual wells. Drag-select (box select) for rectangular regions could be added via Plotly's `selectedData` event.

5. **Auto-scroll log**: The log panel renders new lines but doesn't auto-scroll; would need a small clientside callback.

6. **Image Preview tab**: Listed as optional/low priority in the plan. Not implemented.

7. **Old `gui/` directory**: The old CustomTkinter GUI in `gui/` still exists. Can be removed once the web GUI is validated.

### How the Pipeline Integration Works

The web GUI is an alternative interface for `SegmentationPipeline` (same class used by `scripts/run_pipeline.py`). The flow:

1. User fills in config via the web form (or loads a YAML)
2. User adds plate(s) to the batch queue (or uses single-plate mode with the config tab's `input_dir`)
3. User clicks "Run Analysis" on the Progress tab
4. `callbacks/progress.py` collects all GUI state into a YAML-compatible dict
5. `runner.py` writes a temp YAML config per plate, instantiates `SegmentationPipeline(config_path=...)`, and calls `pipeline.run()`
6. Progress tracking: monkey-patches `pipeline._process_field` to increment a shared counter and check for cancellation
7. `dcc.Interval` polls `runner.state` every 500ms to update progress bars and log
8. After completion, the Results tab reads `field_measurements.csv` from the output directory and renders heatmaps/tables using existing `aggrequant.visualization` modules
