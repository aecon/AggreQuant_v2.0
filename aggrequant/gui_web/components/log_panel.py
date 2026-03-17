"""Progress and log display component."""

from dash import html, dcc

from aggrequant.gui_web.config import (
    COLOR_SUCCESS, COLOR_DANGER, PROGRESS_INTERVAL_MS,
)


def build_progress_layout():
    """Return the Dash layout for the progress & log tab."""
    return html.Div([
        # Run / Cancel controls
        html.Div([
            html.Button(
                "Run Analysis", id="btn-run",
                style={
                    "backgroundColor": COLOR_SUCCESS, "color": "white",
                    "border": "none", "padding": "10px 28px", "fontSize": "16px",
                    "borderRadius": "6px", "cursor": "pointer", "marginRight": "12px",
                },
            ),
            html.Button(
                "Cancel", id="btn-cancel",
                disabled=True,
                style={
                    "backgroundColor": COLOR_DANGER, "color": "white",
                    "border": "none", "padding": "10px 28px", "fontSize": "16px",
                    "borderRadius": "6px", "cursor": "pointer",
                },
            ),
        ], style={"marginBottom": "16px"}),

        # Batch progress
        html.Div([
            html.Label("Batch progress:", style={"fontWeight": "bold"}),
            html.Div([
                html.Div(id="batch-progress-bar", style={
                    "width": "0%", "height": "24px",
                    "backgroundColor": "#3498db", "borderRadius": "4px",
                    "transition": "width 0.3s",
                }),
            ], style={
                "backgroundColor": "#ecf0f1", "borderRadius": "4px",
                "overflow": "hidden", "marginTop": "4px",
            }),
            html.Div(id="batch-progress-text",
                     style={"fontSize": "13px", "color": "#7f8c8d", "marginTop": "2px"}),
        ], style={"marginBottom": "12px"}),

        # Per-plate progress
        html.Div([
            html.Label("Current plate:", style={"fontWeight": "bold"}),
            html.Div([
                html.Div(id="plate-progress-bar", style={
                    "width": "0%", "height": "20px",
                    "backgroundColor": "#27ae60", "borderRadius": "4px",
                    "transition": "width 0.3s",
                }),
            ], style={
                "backgroundColor": "#ecf0f1", "borderRadius": "4px",
                "overflow": "hidden", "marginTop": "4px",
            }),
            html.Div(id="plate-progress-text",
                     style={"fontSize": "13px", "color": "#7f8c8d", "marginTop": "2px"}),
        ], style={"marginBottom": "16px"}),

        # Status
        html.Div(id="run-status", children="Ready",
                 style={"fontSize": "15px", "fontWeight": "bold", "marginBottom": "12px"}),

        # Log output
        html.Div([
            html.Div([
                html.Label("Log:", style={"fontWeight": "bold"}),
                html.Button("Clear", id="btn-clear-log",
                            style={"padding": "2px 10px", "borderRadius": "4px",
                                   "cursor": "pointer", "marginLeft": "12px",
                                   "fontSize": "12px"}),
            ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"}),
            html.Div(
                id="log-output",
                style={
                    "height": "300px", "overflowY": "auto",
                    "backgroundColor": "#1e1e1e", "color": "#d4d4d4",
                    "fontFamily": "monospace", "fontSize": "13px",
                    "padding": "10px", "borderRadius": "6px",
                    "whiteSpace": "pre-wrap",
                },
            ),
        ]),

        # Interval for polling progress
        dcc.Interval(id="progress-interval", interval=PROGRESS_INTERVAL_MS, disabled=True),
    ])
