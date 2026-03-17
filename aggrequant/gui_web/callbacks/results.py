"""Callbacks for the Results tab."""

from pathlib import Path

import pandas as pd
from dash import html, Input, Output, State, callback, dcc
from dash.exceptions import PreventUpdate

from aggrequant.gui_web.runner import state as runner_state


# ---- Toggle results sub-tab panels ----

@callback(
    Output("panel-heatmaps", "style"),
    Output("panel-qc", "style"),
    Output("panel-table", "style"),
    Input("results-tabs", "value"),
)
def toggle_results_panels(tab):
    hidden = {"display": "none"}
    visible = {"display": "block"}
    if tab == "tab-heatmaps":
        return visible, hidden, hidden
    elif tab == "tab-qc":
        return hidden, visible, hidden
    elif tab == "tab-table":
        return hidden, hidden, visible
    return visible, hidden, hidden


# ---- Populate plate selector after run ----

@callback(
    Output("results-plate-selector", "options"),
    Output("results-plate-selector", "value"),
    Input("progress-interval", "disabled"),
    State("batch-queue-data", "data"),
    prevent_initial_call=True,
)
def populate_plate_selector(interval_disabled, queue):
    """When progress polling stops, populate the plate selector."""
    if not interval_disabled:
        raise PreventUpdate

    jobs = runner_state.jobs
    if not jobs:
        raise PreventUpdate

    options = []
    first_done = None
    for j in jobs:
        if j.status == "done":
            label = j.plate_name or Path(j.input_dir).name
            options.append({"label": label, "value": j.input_dir})
            if first_done is None:
                first_done = j.input_dir

    return options, first_done


# ---- Load heatmap metrics for selected plate ----

@callback(
    Output("heatmap-metric-selector", "options"),
    Output("heatmap-metric-selector", "value"),
    Input("results-plate-selector", "value"),
    State("output-subdir", "value"),
    prevent_initial_call=True,
)
def load_heatmap_metrics(plate_dir, output_subdir):
    if not plate_dir:
        raise PreventUpdate

    output_subdir = output_subdir or "aggrequant_output"
    csv_path = Path(plate_dir) / output_subdir / "field_measurements.csv"
    if not csv_path.exists():
        return [], None

    df = pd.read_csv(csv_path)

    # Standard count metrics
    standard = ["n_nuclei", "n_cells", "n_aggregates",
                "n_aggregate_positive_cells", "pct_aggregate_positive_cells",
                "total_cell_area_px", "total_aggregate_area_px"]
    options = [{"label": m.replace("_", " ").title(), "value": m}
               for m in standard if m in df.columns]

    # Focus columns
    from aggrequant.visualization.heatmaps import detect_focus_columns
    for col in detect_focus_columns(df.columns):
        options.append({"label": col.replace("_", " ").title(), "value": col})

    first = options[0]["value"] if options else None
    return options, first


# ---- Render heatmap ----

@callback(
    Output("results-heatmap", "figure"),
    Input("heatmap-metric-selector", "value"),
    State("results-plate-selector", "value"),
    State("output-subdir", "value"),
    State("plate-format", "value"),
    prevent_initial_call=True,
)
def render_heatmap(metric, plate_dir, output_subdir, plate_format):
    if not metric or not plate_dir:
        raise PreventUpdate

    output_subdir = output_subdir or "aggrequant_output"
    csv_path = Path(plate_dir) / output_subdir / "field_measurements.csv"
    if not csv_path.exists():
        raise PreventUpdate

    from aggrequant.visualization.heatmaps import (
        load_field_measurements, aggregate_per_well, compute_ratio_per_well,
        well_values_to_plate_grid, make_plate_heatmap,
    )

    df = load_field_measurements(csv_path)
    plate_format = plate_format or "384"

    # Ratio metrics need special handling
    if metric == "pct_aggregate_positive_cells":
        vals = compute_ratio_per_well(df, "n_aggregate_positive_cells", "n_cells")
    elif metric in ("n_nuclei", "n_cells", "n_aggregates",
                     "n_aggregate_positive_cells",
                     "total_cell_area_px", "total_aggregate_area_px"):
        vals = aggregate_per_well(df, metric, "sum")
    else:
        # Focus metrics: use mean
        vals = aggregate_per_well(df, metric, "mean")

    grid = well_values_to_plate_grid(vals, plate_format)
    title = metric.replace("_", " ").title() + " per well"
    return make_plate_heatmap(grid, title=title, plate_format=plate_format)


# ---- QC plot ----

@callback(
    Output("qc-plot-container", "children"),
    Input("results-plate-selector", "value"),
    State("output-subdir", "value"),
    prevent_initial_call=True,
)
def render_qc_plot(plate_dir, output_subdir):
    if not plate_dir:
        raise PreventUpdate

    output_subdir = output_subdir or "aggrequant_output"
    qc_path = Path(plate_dir) / output_subdir / "plots" / "qc_control_strip.png"

    if qc_path.exists():
        import base64
        with open(qc_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return html.Img(
            src=f"data:image/png;base64,{encoded}",
            style={"maxWidth": "100%"},
        )

    return html.Div("No QC plot available. Run pipeline with control wells defined.",
                     style={"color": "#95a5a6", "fontStyle": "italic"})


# ---- Measurements table ----

@callback(
    Output("measurements-table-container", "children"),
    Input("results-plate-selector", "value"),
    State("output-subdir", "value"),
    prevent_initial_call=True,
)
def render_measurements_table(plate_dir, output_subdir):
    if not plate_dir:
        raise PreventUpdate

    output_subdir = output_subdir or "aggrequant_output"
    csv_path = Path(plate_dir) / output_subdir / "field_measurements.csv"
    if not csv_path.exists():
        return html.Div("No measurements available yet.",
                         style={"color": "#95a5a6", "fontStyle": "italic"})

    df = pd.read_csv(csv_path)

    # Build an HTML table (simple, no dash_table dependency)
    header = html.Tr([html.Th(col, style={"padding": "6px 10px",
                                           "borderBottom": "2px solid #ddd",
                                           "textAlign": "left",
                                           "whiteSpace": "nowrap"})
                       for col in df.columns])

    rows = []
    for _, row in df.head(500).iterrows():  # limit to 500 rows
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.3f}"
            cells.append(html.Td(str(val), style={"padding": "4px 10px",
                                                   "borderBottom": "1px solid #eee",
                                                   "whiteSpace": "nowrap"}))
        rows.append(html.Tr(cells))

    return html.Div([
        html.Div(f"Showing {min(500, len(df))} of {len(df)} rows",
                 style={"fontSize": "13px", "color": "#7f8c8d", "marginBottom": "6px"}),
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"borderCollapse": "collapse", "fontSize": "13px",
                   "fontFamily": "monospace"},
        ),
    ], style={"maxHeight": "500px", "overflowY": "auto"})


# ---- Export plots ----

@callback(
    Output("export-feedback", "children"),
    Input("btn-export-plots", "n_clicks"),
    State("results-plate-selector", "value"),
    State("output-subdir", "value"),
    State("plate-format", "value"),
    State("control-assignments", "data"),
    prevent_initial_call=True,
)
def export_plots(n_clicks, plate_dir, output_subdir, plate_format, ctrl_assignments):
    if not plate_dir:
        return "Select a plate first"

    output_subdir = output_subdir or "aggrequant_output"
    csv_path = Path(plate_dir) / output_subdir / "field_measurements.csv"
    if not csv_path.exists():
        return "No measurements CSV found"

    try:
        from aggrequant.visualization.heatmaps import generate_all_heatmaps

        # Convert control assignments for QC plot
        control_wells = {}
        for well, ctrl_type in (ctrl_assignments or {}).items():
            control_wells.setdefault(ctrl_type, []).append(well)

        plots_dir = generate_all_heatmaps(csv_path, plate_format=plate_format or "384")

        if control_wells:
            from aggrequant.visualization.qc_plots import plot_control_strip
            qc_path = Path(plate_dir) / output_subdir / "plots" / "qc_control_strip.png"
            plot_control_strip(csv_path, control_wells, output_path=qc_path)

        return f"Plots exported to {plots_dir}"
    except Exception as e:
        return f"Export failed: {e}"
