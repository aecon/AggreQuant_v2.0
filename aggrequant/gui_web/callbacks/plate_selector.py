"""Callbacks for the Plate Selector tab."""

import re

from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from aggrequant.gui_web.components.plate_grid import make_plate_figure
from aggrequant.loaders.plate import PLATE_LAYOUTS, indices_to_well_id


def _parse_well_spec(spec, plate_format="384"):
    """Parse a flexible well specification string into a set of well IDs.

    Supports:
        - Single wells: A01, B12
        - Well ranges: A01-A12
        - Row-column ranges: A-H:5  (rows A through H, column 5)
        - Column ranges: col:1-2  (all rows, columns 1-2)
        - Comma-separated combinations

    Arguments:
        spec: Well specification string.
        plate_format: "96" or "384".

    Returns:
        Set of well ID strings.
    """
    layout = PLATE_LAYOUTS[plate_format]
    n_rows, n_cols = layout["rows"], layout["cols"]
    wells = set()

    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue

        # col:N or col:N-M
        m = re.match(r"col:(\d+)(?:-(\d+))?$", part, re.IGNORECASE)
        if m:
            c_start = int(m.group(1))
            c_end = int(m.group(2)) if m.group(2) else c_start
            for r in range(n_rows):
                for c in range(c_start, c_end + 1):
                    if 1 <= c <= n_cols:
                        wells.add(indices_to_well_id(r, c - 1))
            continue

        # Row range with column: A-H:5
        m = re.match(r"([A-Pa-p])-([A-Pa-p]):(\d+)$", part)
        if m:
            r_start = ord(m.group(1).upper()) - ord("A")
            r_end = ord(m.group(2).upper()) - ord("A")
            col = int(m.group(3))
            for r in range(r_start, r_end + 1):
                if r < n_rows and 1 <= col <= n_cols:
                    wells.add(indices_to_well_id(r, col - 1))
            continue

        # Well range: A01-A12
        m = re.match(r"([A-Pa-p])(\d+)-([A-Pa-p])(\d+)$", part)
        if m:
            r1 = ord(m.group(1).upper()) - ord("A")
            c1 = int(m.group(2)) - 1
            r2 = ord(m.group(3).upper()) - ord("A")
            c2 = int(m.group(4)) - 1
            for r in range(min(r1, r2), max(r1, r2) + 1):
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    if r < n_rows and c < n_cols:
                        wells.add(indices_to_well_id(r, c))
            continue

        # Single well: A01
        m = re.match(r"([A-Pa-p])(\d+)$", part)
        if m:
            r = ord(m.group(1).upper()) - ord("A")
            c = int(m.group(2)) - 1
            if r < n_rows and c < n_cols:
                wells.add(indices_to_well_id(r, c))
            continue

    return wells


# ---- Update plate grid on any state change ----

@callback(
    Output("plate-grid", "figure"),
    Input("plate-format", "value"),
    Input("selected-wells", "data"),
    Input("control-assignments", "data"),
)
def update_plate_grid(plate_format, selected_wells, control_assignments):
    return make_plate_figure(
        plate_format=plate_format or "384",
        selected_wells=set(selected_wells or []),
        control_assignments=control_assignments or {},
    )


# ---- Click on plate grid to toggle well selection ----

@callback(
    Output("selected-wells", "data", allow_duplicate=True),
    Input("plate-grid", "clickData"),
    State("selected-wells", "data"),
    prevent_initial_call=True,
)
def toggle_well_selection(click_data, selected):
    if not click_data or not click_data.get("points"):
        raise PreventUpdate

    point = click_data["points"][0]
    well_id = point.get("customdata")
    if not well_id:
        raise PreventUpdate

    selected = list(selected or [])
    if well_id in selected:
        selected.remove(well_id)
    else:
        selected.append(well_id)
    return selected


# ---- Text-based well spec ----

@callback(
    Output("selected-wells", "data", allow_duplicate=True),
    Output("well-spec-feedback", "children"),
    Input("btn-well-spec", "n_clicks"),
    State("well-spec-input", "value"),
    State("plate-format", "value"),
    State("selected-wells", "data"),
    prevent_initial_call=True,
)
def apply_well_spec(n_clicks, spec, plate_format, current_selected):
    if not spec:
        raise PreventUpdate
    try:
        new_wells = _parse_well_spec(spec, plate_format or "384")
        combined = list(set(current_selected or []) | new_wells)
        return combined, f"Added {len(new_wells)} wells"
    except Exception as e:
        return current_selected or [], f"Error: {e}"


# ---- Assign controls ----

@callback(
    Output("control-assignments", "data", allow_duplicate=True),
    Output("selected-wells", "data", allow_duplicate=True),
    Input("btn-assign-rab13", "n_clicks"),
    Input("btn-assign-nt", "n_clicks"),
    Input("btn-assign-custom", "n_clicks"),
    State("custom-control-name", "value"),
    State("selected-wells", "data"),
    State("control-assignments", "data"),
    prevent_initial_call=True,
)
def assign_controls(rab13_clicks, nt_clicks, custom_clicks,
                    custom_name, selected, assignments):
    triggered = ctx.triggered_id
    if not triggered or not selected:
        raise PreventUpdate

    assignments = dict(assignments or {})

    if triggered == "btn-assign-rab13":
        ctrl_type = "rab13"
    elif triggered == "btn-assign-nt":
        ctrl_type = "NT"
    elif triggered == "btn-assign-custom":
        ctrl_type = (custom_name or "").strip()
        if not ctrl_type:
            raise PreventUpdate
    else:
        raise PreventUpdate

    for well_id in selected:
        assignments[well_id] = ctrl_type

    return assignments, []  # clear selection


# ---- Clear buttons ----

@callback(
    Output("selected-wells", "data", allow_duplicate=True),
    Input("btn-clear-selection", "n_clicks"),
    prevent_initial_call=True,
)
def clear_selection(n_clicks):
    return []


@callback(
    Output("control-assignments", "data", allow_duplicate=True),
    Input("btn-clear-controls", "n_clicks"),
    prevent_initial_call=True,
)
def clear_controls(n_clicks):
    return {}
