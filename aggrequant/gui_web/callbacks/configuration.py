"""Callbacks for the Configuration tab."""

import json
from pathlib import Path

import yaml
from dash import Input, Output, State, callback, ctx, ALL, MATCH, Patch
from dash.exceptions import PreventUpdate

from aggrequant.gui_web.components.settings_form import _channel_row


# ---- Load / Save config ----

@callback(
    Output("input-dir", "value"),
    Output("output-subdir", "value"),
    Output("plate-name", "value"),
    Output("plate-format", "value"),
    Output("nuclei-sigma-denoise", "value"),
    Output("nuclei-sigma-background", "value"),
    Output("nuclei-min-area", "value"),
    Output("nuclei-max-area", "value"),
    Output("cell-model", "value"),
    Output("aggregate-method", "value"),
    Output("aggregate-min-size", "value"),
    Output("aggregate-intensity-threshold", "value"),
    Output("aggregate-model-path", "value"),
    Output("focus-compute-on", "value"),
    Output("focus-patch-metrics", "value"),
    Output("focus-global-metrics", "value"),
    Output("focus-patch-h", "value"),
    Output("focus-patch-w", "value"),
    Output("output-options", "value"),
    Output("use-gpu", "value"),
    Output("verbose", "value"),
    Output("control-assignments", "data", allow_duplicate=True),
    Output("config-feedback", "children"),
    Input("btn-load-config", "n_clicks"),
    State("config-path", "value"),
    prevent_initial_call=True,
)
def load_config(n_clicks, config_path):
    """Load a YAML config file and populate all form fields."""
    if not config_path:
        raise PreventUpdate

    try:
        path = Path(config_path)
        if not path.exists():
            return [None] * 22 + [f"File not found: {config_path}"]

        with open(path) as f:
            data = yaml.safe_load(f)

        seg = data.get("segmentation", {})
        quality = data.get("quality", {})
        output = data.get("output", {})
        patch_size = quality.get("patch_size", [40, 40])

        output_opts = []
        if output.get("save_masks", True):
            output_opts.append("save_masks")
        if output.get("overwrite_masks", False):
            output_opts.append("overwrite_masks")

        # Convert control_wells {type: [wells]} -> {well: type}
        ctrl_assignments = {}
        for ctrl_type, wells in data.get("control_wells", {}).items():
            for w in wells:
                ctrl_assignments[w] = ctrl_type

        return (
            data.get("input_dir", ""),
            output.get("output_subdir", "aggrequant_output"),
            data.get("plate_name", ""),
            data.get("plate_format", "384"),
            seg.get("nuclei_sigma_denoise", 2.0),
            seg.get("nuclei_sigma_background", 50.0),
            seg.get("nuclei_min_area", 300),
            seg.get("nuclei_max_area", 15000),
            seg.get("cell_model", "cyto3"),
            seg.get("aggregate_method", "filter"),
            seg.get("aggregate_min_size", 9),
            seg.get("aggregate_intensity_threshold", 1.6),
            str(seg.get("aggregate_model_path", "")) if seg.get("aggregate_model_path") else "",
            quality.get("compute_on", ["nuclei"]),
            quality.get("patch_metrics", ["VarianceLaplacian"]),
            quality.get("global_metrics", ["power_log_log_slope"]),
            patch_size[0] if isinstance(patch_size, list) else patch_size,
            patch_size[1] if isinstance(patch_size, list) else patch_size,
            output_opts,
            ["yes"] if data.get("use_gpu", True) else [],
            ["yes"] if data.get("verbose", True) else [],
            ctrl_assignments,
            f"Loaded: {path.name}",
        )
    except Exception as e:
        return [None] * 22 + [f"Error: {e}"]


@callback(
    Output("config-feedback", "children", allow_duplicate=True),
    Input("btn-save-config", "n_clicks"),
    State("config-path", "value"),
    State("input-dir", "value"),
    State("output-subdir", "value"),
    State("plate-name", "value"),
    State("plate-format", "value"),
    State("nuclei-sigma-denoise", "value"),
    State("nuclei-sigma-background", "value"),
    State("nuclei-min-area", "value"),
    State("nuclei-max-area", "value"),
    State("cell-model", "value"),
    State("aggregate-method", "value"),
    State("aggregate-min-size", "value"),
    State("aggregate-intensity-threshold", "value"),
    State("aggregate-model-path", "value"),
    State("focus-compute-on", "value"),
    State("focus-patch-metrics", "value"),
    State("focus-global-metrics", "value"),
    State("focus-patch-h", "value"),
    State("focus-patch-w", "value"),
    State("output-options", "value"),
    State("use-gpu", "value"),
    State("verbose", "value"),
    State("control-assignments", "data"),
    prevent_initial_call=True,
)
def save_config(n_clicks, config_path,
                input_dir, output_subdir, plate_name, plate_format,
                nuc_sigma_d, nuc_sigma_b, nuc_min, nuc_max,
                cell_model, agg_method, agg_min, agg_thresh, agg_model,
                focus_on, focus_patch, focus_global, patch_h, patch_w,
                output_opts, use_gpu, verbose, ctrl_assignments):
    """Save the current form state to a YAML config file."""
    if not config_path:
        return "Enter a file path first"

    # Convert control assignments {well: type} -> {type: [wells]}
    control_wells = {}
    for well, ctrl_type in (ctrl_assignments or {}).items():
        control_wells.setdefault(ctrl_type, []).append(well)

    output_opts = output_opts or []

    data = {
        "input_dir": input_dir or "",
        "plate_format": plate_format or "384",
        "plate_name": plate_name or "",
        "channels": [],  # TODO: read from dynamic channel rows
        "segmentation": {
            "nuclei_sigma_denoise": float(nuc_sigma_d or 2.0),
            "nuclei_sigma_background": float(nuc_sigma_b or 50.0),
            "nuclei_min_area": int(nuc_min or 300),
            "nuclei_max_area": int(nuc_max or 15000),
            "cell_model": cell_model or "cyto3",
            "aggregate_method": agg_method or "filter",
            "aggregate_model_path": agg_model if agg_model else None,
            "aggregate_min_size": int(agg_min or 9),
            "aggregate_intensity_threshold": float(agg_thresh or 1.6),
        },
        "quality": {
            "compute_on": focus_on or [],
            "compute_patch_metrics": bool(focus_patch),
            "compute_global_metrics": bool(focus_global),
            "patch_metrics": focus_patch or [],
            "global_metrics": focus_global or [],
            "patch_size": [int(patch_h or 40), int(patch_w or 40)],
        },
        "output": {
            "output_subdir": output_subdir or "aggrequant_output",
            "save_masks": "save_masks" in output_opts,
            "overwrite_masks": "overwrite_masks" in output_opts,
        },
        "control_wells": control_wells,
        "use_gpu": bool(use_gpu),
        "verbose": bool(verbose),
    }

    try:
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        return f"Saved: {path.name}"
    except Exception as e:
        return f"Error: {e}"


# ---- Show/hide fields based on aggregate method ----

@callback(
    Output("unet-model-row", "style"),
    Output("filter-threshold-row", "style"),
    Input("aggregate-method", "value"),
)
def toggle_aggregate_fields(method):
    if method == "unet":
        return {"display": "block"}, {"display": "none"}
    return {"display": "none"}, {"display": "block"}


# ---- Browse for input directory ----

@callback(
    Output("input-dir", "value", allow_duplicate=True),
    Input("btn-browse-input", "n_clicks"),
    prevent_initial_call=True,
)
def browse_input_dir(n_clicks):
    """Open a native folder picker via tkinter subprocess."""
    import subprocess, sys
    code = (
        "import tkinter as tk; "
        "root = tk.Tk(); root.withdraw(); "
        "from tkinter import filedialog; "
        "p = filedialog.askdirectory(title='Select input directory'); "
        "print(p if p else '')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )
    path = proc.stdout.strip()
    if not path:
        from dash.exceptions import PreventUpdate
        raise PreventUpdate
    return path
