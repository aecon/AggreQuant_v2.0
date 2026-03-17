"""Callbacks for the Progress & Log tab and pipeline execution."""

from dash import html, Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from aggrequant.gui_web.runner import state as runner_state, start_batch, cancel_batch, PlateJob
from aggrequant.gui_web.config import DEFAULT_CHANNELS


def _collect_gui_config(
    input_dir, output_subdir, plate_name, plate_format,
    nuc_sigma_d, nuc_sigma_b, nuc_min, nuc_max,
    cell_model, agg_method, agg_min, agg_thresh, agg_model,
    focus_on, focus_patch, focus_global, patch_h, patch_w,
    output_opts, use_gpu, verbose, ctrl_assignments,
):
    """Build a YAML-compatible config dict from GUI form values."""
    output_opts = output_opts or []

    # Convert control assignments {well: type} -> {type: [wells]}
    control_wells = {}
    for well, ctrl_type in (ctrl_assignments or {}).items():
        control_wells.setdefault(ctrl_type, []).append(well)

    channels = DEFAULT_CHANNELS  # TODO: read from dynamic channel rows

    return {
        "input_dir": input_dir or "",
        "plate_format": plate_format or "384",
        "plate_name": plate_name or "",
        "channels": channels,
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


# ---- Run / Cancel ----

@callback(
    Output("progress-interval", "disabled", allow_duplicate=True),
    Output("btn-run", "disabled"),
    Output("btn-cancel", "disabled"),
    Output("run-status", "children", allow_duplicate=True),
    Input("btn-run", "n_clicks"),
    # All config states
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
    State("batch-queue-data", "data"),
    State("batch-per-plate-config", "value"),
    prevent_initial_call=True,
)
def run_analysis(
    n_clicks,
    input_dir, output_subdir, plate_name, plate_format,
    nuc_sigma_d, nuc_sigma_b, nuc_min, nuc_max,
    cell_model, agg_method, agg_min, agg_thresh, agg_model,
    focus_on, focus_patch, focus_global, patch_h, patch_w,
    output_opts, use_gpu, verbose, ctrl_assignments,
    batch_queue, per_plate_config,
):
    """Start pipeline execution."""
    if runner_state.is_running:
        raise PreventUpdate

    gui_config = _collect_gui_config(
        input_dir, output_subdir, plate_name, plate_format,
        nuc_sigma_d, nuc_sigma_b, nuc_min, nuc_max,
        cell_model, agg_method, agg_min, agg_thresh, agg_model,
        focus_on, focus_patch, focus_global, patch_h, patch_w,
        output_opts, use_gpu, verbose, ctrl_assignments,
    )

    # Build job list
    batch_queue = batch_queue or []
    use_per_plate = "yes" in (per_plate_config or [])

    if batch_queue:
        # Batch mode: use the queue
        jobs = []
        for item in batch_queue:
            job = PlateJob(
                input_dir=item["input_dir"],
                plate_name=item.get("plate_name", ""),
            )
            if use_per_plate:
                from pathlib import Path
                per_plate = Path(item["input_dir"]) / "aggrequant_config.yaml"
                if per_plate.exists():
                    job.config_path = str(per_plate)
            jobs.append(job)
    elif input_dir:
        # Single plate mode: use the config tab settings
        jobs = [PlateJob(input_dir=input_dir, plate_name=plate_name or "")]
    else:
        return True, False, True, "No input directory or batch queue specified"

    start_batch(gui_config, jobs)

    return False, True, False, "Running..."


@callback(
    Output("run-status", "children", allow_duplicate=True),
    Input("btn-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def cancel_analysis(n_clicks):
    cancel_batch()
    return "Cancelling..."


# ---- Progress polling ----

@callback(
    Output("batch-progress-bar", "style"),
    Output("batch-progress-text", "children"),
    Output("plate-progress-bar", "style"),
    Output("plate-progress-text", "children"),
    Output("log-output", "children"),
    Output("run-status", "children", allow_duplicate=True),
    Output("progress-interval", "disabled", allow_duplicate=True),
    Output("btn-run", "disabled", allow_duplicate=True),
    Output("btn-cancel", "disabled", allow_duplicate=True),
    Output("batch-queue-data", "data", allow_duplicate=True),
    Input("progress-interval", "n_intervals"),
    State("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def poll_progress(n_intervals, queue):
    s = runner_state

    # Batch progress
    if s.total_jobs > 0:
        batch_pct = (s.current_job_index + (1 if not s.is_running else 0)) / s.total_jobs * 100
        # Clamp between the completed jobs
        completed = sum(1 for j in s.jobs if j.status in ("done", "failed", "cancelled"))
        batch_pct = completed / s.total_jobs * 100
    else:
        batch_pct = 0
    batch_bar_style = {
        "width": f"{batch_pct:.0f}%", "height": "24px",
        "backgroundColor": "#3498db", "borderRadius": "4px",
        "transition": "width 0.3s",
    }
    batch_text = f"Plate {min(s.current_job_index + 1, s.total_jobs)} of {s.total_jobs}"

    # Plate progress
    if s.total_fields > 0:
        plate_pct = s.current_field / s.total_fields * 100
    else:
        plate_pct = 0
    plate_bar_style = {
        "width": f"{plate_pct:.0f}%", "height": "20px",
        "backgroundColor": "#27ae60", "borderRadius": "4px",
        "transition": "width 0.3s",
    }
    plate_text = f"Field {s.current_field} of {s.total_fields}"

    # Log
    log_lines = []
    level_colors = {"INFO": "#d4d4d4", "WARN": "#f39c12", "ERROR": "#e74c3c"}
    for ts, level, msg in s.log[-200:]:  # show last 200 lines
        color = level_colors.get(level, "#d4d4d4")
        log_lines.append(
            html.Span(f"[{ts}] [{level}] {msg}\n", style={"color": color})
        )

    # Status
    status_text = s.current_status.replace("_", " ").title()
    if s.current_status == "running":
        status_text = "Running..."
    elif s.current_status == "done":
        status_text = "Complete!"
    elif s.current_status == "cancelled":
        status_text = "Cancelled"

    # Update queue data with job statuses
    if s.jobs and queue:
        for i, job in enumerate(s.jobs):
            if i < len(queue):
                queue[i]["status"] = job.status

    # Stop polling when done
    is_done = not s.is_running and s.current_status != "idle"
    interval_disabled = is_done
    run_disabled = s.is_running
    cancel_disabled = not s.is_running

    return (
        batch_bar_style, batch_text,
        plate_bar_style, plate_text,
        log_lines,
        status_text,
        interval_disabled,
        run_disabled,
        cancel_disabled,
        queue or [],
    )


# ---- Clear log ----

@callback(
    Output("log-output", "children", allow_duplicate=True),
    Input("btn-clear-log", "n_clicks"),
    prevent_initial_call=True,
)
def clear_log(n_clicks):
    runner_state.log.clear()
    return []
