"""Callbacks for the Batch Processing tab."""

import os
from pathlib import Path

from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from aggrequant.gui_web.components.batch_queue import render_queue_table


@callback(
    Output("batch-queue-data", "data", allow_duplicate=True),
    Output("batch-queue-display", "children", allow_duplicate=True),
    Output("batch-dir-input", "value"),
    Input("btn-batch-add", "n_clicks"),
    State("batch-dir-input", "value"),
    State("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def add_single_plate(n_clicks, dir_path, queue):
    if not dir_path or not dir_path.strip():
        raise PreventUpdate

    dir_path = dir_path.strip()
    queue = list(queue or [])

    # Don't add duplicates
    existing_dirs = {j["input_dir"] for j in queue}
    if dir_path not in existing_dirs:
        queue.append({
            "input_dir": dir_path,
            "plate_name": Path(dir_path).name,
            "status": "pending",
        })

    return queue, render_queue_table(queue), ""


@callback(
    Output("batch-queue-data", "data", allow_duplicate=True),
    Output("batch-queue-display", "children", allow_duplicate=True),
    Output("batch-paste-input", "value"),
    Input("btn-batch-paste", "n_clicks"),
    State("batch-paste-input", "value"),
    State("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def add_pasted_plates(n_clicks, text, queue):
    if not text or not text.strip():
        raise PreventUpdate

    queue = list(queue or [])
    existing_dirs = {j["input_dir"] for j in queue}

    for line in text.strip().splitlines():
        d = line.strip()
        if d and d not in existing_dirs:
            queue.append({
                "input_dir": d,
                "plate_name": Path(d).name,
                "status": "pending",
            })
            existing_dirs.add(d)

    return queue, render_queue_table(queue), ""


@callback(
    Output("batch-queue-data", "data", allow_duplicate=True),
    Output("batch-queue-display", "children", allow_duplicate=True),
    Input("btn-batch-scan", "n_clicks"),
    State("batch-parent-input", "value"),
    State("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def scan_parent_directory(n_clicks, parent_dir, queue):
    if not parent_dir or not parent_dir.strip():
        raise PreventUpdate

    parent = Path(parent_dir.strip())
    if not parent.is_dir():
        raise PreventUpdate

    queue = list(queue or [])
    existing_dirs = {j["input_dir"] for j in queue}

    # Add all subdirectories that look like plate folders
    # (contain .tif files or have subdirectories with images)
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        dir_str = str(child)
        if dir_str not in existing_dirs:
            # Quick check: does it contain any .tif files?
            has_tifs = any(child.glob("*.tif")) or any(child.glob("**/*.tif"))
            if has_tifs:
                queue.append({
                    "input_dir": dir_str,
                    "plate_name": child.name,
                    "status": "pending",
                })
                existing_dirs.add(dir_str)

    return queue, render_queue_table(queue), ""


@callback(
    Output("batch-queue-data", "data", allow_duplicate=True),
    Output("batch-queue-display", "children", allow_duplicate=True),
    Input("btn-batch-clear", "n_clicks"),
    prevent_initial_call=True,
)
def clear_queue(n_clicks):
    return [], render_queue_table([])


@callback(
    Output("batch-queue-display", "children", allow_duplicate=True),
    Input("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def refresh_queue_display(queue):
    return render_queue_table(queue or [])
