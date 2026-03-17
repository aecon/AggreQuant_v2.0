"""Results visualization component."""

from dash import html, dcc


def build_results_layout():
    """Return the Dash layout for the results tab."""
    return html.Div([
        html.H4("Results", style={
            "borderBottom": "2px solid #3498db", "paddingBottom": "4px",
            "marginBottom": "12px",
        }),

        # Plate selector (for batch runs with multiple plates)
        html.Div([
            html.Label("Select plate:", style={"fontWeight": "bold", "marginRight": "10px"}),
            dcc.Dropdown(
                id="results-plate-selector",
                options=[],
                placeholder="Run the pipeline first...",
                style={"width": "400px"},
            ),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

        # Results sub-tabs
        dcc.Tabs(id="results-tabs", value="tab-heatmaps", children=[
            dcc.Tab(label="Heatmaps", value="tab-heatmaps"),
            dcc.Tab(label="QC Plots", value="tab-qc"),
            dcc.Tab(label="Measurements Table", value="tab-table"),
        ]),

        # Heatmaps panel
        html.Div(id="panel-heatmaps", children=[
            html.Div([
                html.Label("Metric:", style={"fontWeight": "bold", "marginRight": "10px"}),
                dcc.Dropdown(
                    id="heatmap-metric-selector",
                    options=[],
                    placeholder="Select metric...",
                    style={"width": "350px"},
                ),
            ], style={"display": "flex", "alignItems": "center",
                       "marginTop": "12px", "marginBottom": "12px"}),
            dcc.Graph(id="results-heatmap",
                      config={"displayModeBar": True},
                      style={"marginBottom": "12px"}),
        ]),

        # QC panel
        html.Div(id="panel-qc", children=[
            html.Div(id="qc-plot-container",
                     style={"marginTop": "12px"}),
        ], style={"display": "none"}),

        # Table panel
        html.Div(id="panel-table", children=[
            html.Div(id="measurements-table-container",
                     style={"marginTop": "12px", "overflowX": "auto"}),
        ], style={"display": "none"}),

        # Export
        html.Div([
            html.Button("Export All Plots as PNG", id="btn-export-plots",
                        style={"padding": "8px 20px", "borderRadius": "4px",
                               "cursor": "pointer", "marginTop": "16px"}),
            html.Div(id="export-feedback",
                     style={"color": "#7f8c8d", "marginTop": "4px"}),
        ]),
    ])
