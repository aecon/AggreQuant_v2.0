"""Interactive well plate grid component using Plotly."""

import numpy as np
import plotly.graph_objects as go

from aggrequant.loaders.plate import PLATE_LAYOUTS, indices_to_well_id
from aggrequant.gui_web.config import DEFAULT_CONTROL_ASSIGNMENTS


def make_plate_figure(plate_format="96", selected_wells=None,
                      control_assignments=None):
    """Create an interactive plate grid figure.

    Arguments:
        plate_format: "96" or "384".
        selected_wells: Set of currently selected well IDs.
        control_assignments: Dict mapping well_id to control type name.

    Returns:
        plotly.graph_objects.Figure with clickable wells.
    """
    selected_wells = selected_wells or set()
    control_assignments = control_assignments or {}

    layout_info = PLATE_LAYOUTS[plate_format]
    n_rows, n_cols = layout_info["rows"], layout_info["cols"]

    row_labels = [chr(ord("A") + i) for i in range(n_rows)]
    col_labels = [str(i + 1) for i in range(n_cols)]

    # Build colour and hover arrays
    from aggrequant.gui_web.config import (
        CONTROL_COLORS, CUSTOM_CONTROL_COLOR, SELECTED_COLOR, EMPTY_COLOR,
    )

    colors = np.empty((n_rows, n_cols), dtype=object)
    hover = np.empty((n_rows, n_cols), dtype=object)
    # z values: 0=empty, 1=selected, 2+=control types
    z = np.zeros((n_rows, n_cols), dtype=float)

    control_types = sorted(set(control_assignments.values()))
    ctrl_index = {ct: i + 2 for i, ct in enumerate(control_types)}

    for r in range(n_rows):
        for c in range(n_cols):
            wid = indices_to_well_id(r, c)
            if wid in control_assignments:
                ct = control_assignments[wid]
                color = CONTROL_COLORS.get(ct, CUSTOM_CONTROL_COLOR)
                colors[r, c] = color
                hover[r, c] = f"{wid} [{ct}]"
                z[r, c] = ctrl_index.get(ct, 2)
            elif wid in selected_wells:
                colors[r, c] = SELECTED_COLOR
                hover[r, c] = f"{wid} (selected)"
                z[r, c] = 1
            else:
                colors[r, c] = EMPTY_COLOR
                hover[r, c] = wid
                z[r, c] = 0

    # Build a custom discrete colorscale from the z values
    # We use a heatmap with customdata for click identification
    fig = go.Figure()

    # One scatter marker per well for full control over colour
    for r in range(n_rows):
        for c in range(n_cols):
            wid = indices_to_well_id(r, c)
            fig.add_trace(go.Scatter(
                x=[c], y=[r],
                mode="markers",
                marker=dict(
                    size=18 if plate_format == "384" else 28,
                    color=colors[r, c],
                    line=dict(width=1, color="#bdc3c7"),
                    symbol="circle",
                ),
                customdata=[wid],
                hovertext=hover[r, c],
                hoverinfo="text",
                showlegend=False,
            ))

    well_size = 26 if plate_format == "384" else 46
    fig_width = n_cols * well_size + 80
    fig_height = n_rows * well_size + 60

    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=col_labels,
            side="top",
            range=[-0.5, n_cols - 0.5],
            constrain="domain",
            showgrid=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=row_labels,
            autorange="reversed",
            scaleanchor="x",
            constrain="domain",
            showgrid=False,
        ),
        width=fig_width,
        height=fig_height,
        margin=dict(l=30, r=10, t=30, b=10),
        plot_bgcolor="white",
        clickmode="event",
        dragmode=False,
    )

    return fig


def build_plate_layout():
    """Return the Dash layout for the plate selector tab."""
    from dash import html, dcc

    return html.Div([
        # Plate format selector
        html.Div([
            html.Label("Plate format:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="plate-format",
                options=[
                    {"label": "96-well", "value": "96"},
                    {"label": "384-well", "value": "384"},
                ],
                value="384",
                clearable=False,
                style={"width": "140px", "display": "inline-block"},
            ),
        ], style={"marginBottom": "10px", "display": "flex", "alignItems": "center"}),

        # Interactive plate grid
        dcc.Graph(
            id="plate-grid",
            config={"displayModeBar": False, "scrollZoom": False},
            style={"marginBottom": "10px"},
        ),

        # Control assignment
        html.Div([
            html.Div([
                html.Label("Assign selected wells as:", style={"fontWeight": "bold"}),
                html.Div([
                    html.Button("rab13", id="btn-assign-rab13",
                                style={"backgroundColor": "#3498db", "color": "white",
                                       "border": "none", "padding": "6px 16px",
                                       "borderRadius": "4px", "marginRight": "8px",
                                       "cursor": "pointer"}),
                    html.Button("NT", id="btn-assign-nt",
                                style={"backgroundColor": "#2ecc71", "color": "white",
                                       "border": "none", "padding": "6px 16px",
                                       "borderRadius": "4px", "marginRight": "8px",
                                       "cursor": "pointer"}),
                    dcc.Input(
                        id="custom-control-name",
                        placeholder="Custom type...",
                        type="text",
                        style={"width": "120px", "marginRight": "8px", "padding": "5px"},
                    ),
                    html.Button("Assign Custom", id="btn-assign-custom",
                                style={"backgroundColor": "#f39c12", "color": "white",
                                       "border": "none", "padding": "6px 16px",
                                       "borderRadius": "4px", "cursor": "pointer"}),
                ], style={"display": "flex", "alignItems": "center", "marginTop": "6px"}),
            ], style={"flex": "1"}),

            html.Div([
                html.Button("Clear Selection", id="btn-clear-selection",
                            style={"marginRight": "8px", "padding": "6px 16px",
                                   "borderRadius": "4px", "cursor": "pointer"}),
                html.Button("Clear All Controls", id="btn-clear-controls",
                            style={"padding": "6px 16px", "borderRadius": "4px",
                                   "cursor": "pointer"}),
            ]),
        ], style={"display": "flex", "justifyContent": "space-between",
                   "alignItems": "flex-start", "marginBottom": "10px"}),

        # Text-based well specification
        html.Div([
            html.Label("Or specify wells as text:", style={"fontWeight": "bold"}),
            html.Div([
                dcc.Input(
                    id="well-spec-input",
                    placeholder="e.g. A01-A12, B01, C-H:5, col:1-2",
                    type="text",
                    style={"flex": "1", "padding": "5px", "marginRight": "8px"},
                ),
                html.Button("Select", id="btn-well-spec",
                            style={"padding": "6px 16px", "borderRadius": "4px",
                                   "cursor": "pointer"}),
            ], style={"display": "flex", "marginTop": "4px"}),
            html.Div(id="well-spec-feedback",
                     style={"color": "#7f8c8d", "fontSize": "12px", "marginTop": "4px"}),
        ]),

        # Hidden stores
        dcc.Store(id="selected-wells", data=[]),
        dcc.Store(id="control-assignments",
                  data=DEFAULT_CONTROL_ASSIGNMENTS),
    ])
