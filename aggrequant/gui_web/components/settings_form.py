"""Configuration settings form component."""

from dash import html, dcc

from aggrequant.gui_web.config import (
    DEFAULT_CHANNELS,
    AGGREGATE_METHODS,
    PATCH_METRIC_OPTIONS,
    GLOBAL_METRIC_OPTIONS,
)


def _section(title, children):
    """Wrap children in a styled section with a header."""
    return html.Div([
        html.H4(title, style={
            "borderBottom": "2px solid #3498db", "paddingBottom": "4px",
            "marginBottom": "10px", "marginTop": "16px",
        }),
        *children,
    ])


def _row(*children, **style_kwargs):
    base = {"display": "flex", "gap": "12px", "marginBottom": "8px",
            "alignItems": "center"}
    base.update(style_kwargs)
    return html.Div(list(children), style=base)


def _label(text, width="160px"):
    return html.Label(text, style={"width": width, "fontWeight": "bold",
                                   "flexShrink": "0"})


def build_settings_layout():
    """Return the Dash layout for the configuration tab."""
    return html.Div([
        # Load / Save config
        html.Div([
            dcc.Input(id="config-path", placeholder="Path to YAML config...",
                      type="text", style={"flex": "1", "padding": "5px"}),
            html.Button("Load", id="btn-load-config",
                        style={"padding": "6px 16px", "borderRadius": "4px",
                               "cursor": "pointer", "marginLeft": "8px"}),
            html.Button("Save", id="btn-save-config",
                        style={"padding": "6px 16px", "borderRadius": "4px",
                               "cursor": "pointer", "marginLeft": "8px"}),
            html.Div(id="config-feedback", style={"marginLeft": "12px",
                                                   "color": "#7f8c8d"}),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px"}),

        # ================================================================
        # ESSENTIALS — what the user changes per experiment
        # ================================================================

        # Input / Output
        _section("Input / Output", [
            _row(
                _label("Input directory:"),
                dcc.Input(id="input-dir", type="text", placeholder="/path/to/images",
                          style={"flex": "1", "padding": "5px"}),
                html.Button("Browse...", id="btn-browse-input",
                            style={"padding": "5px 14px", "borderRadius": "4px",
                                   "cursor": "pointer", "marginLeft": "8px"}),
            ),
            _row(
                _label("Plate name:"),
                dcc.Input(id="plate-name", type="text", placeholder="(auto from folder name)",
                          style={"width": "250px", "padding": "5px"}),
            ),
            _row(
                _label("Output subdirectory:"),
                dcc.Input(id="output-subdir", type="text", value="aggrequant_output",
                          style={"width": "250px", "padding": "5px"}),
            ),
        ]),

        # Channel Configuration
        _section("Channels", [
            html.Div(id="channel-rows", children=_default_channel_rows()),
            html.Button("+ Add Channel", id="btn-add-channel",
                        style={"marginTop": "6px", "padding": "4px 12px",
                               "borderRadius": "4px", "cursor": "pointer"}),
        ]),

        # Output & Processing
        _section("Output & Processing", [
            _row(
                dcc.Checklist(
                    id="output-options",
                    options=[
                        {"label": " Save segmentation masks", "value": "save_masks"},
                        {"label": " Overwrite existing masks", "value": "overwrite_masks"},
                    ],
                    value=["save_masks"],
                    style={"display": "flex", "gap": "24px"},
                ),
            ),
            _row(
                _label("Use GPU:"),
                dcc.Checklist(
                    id="use-gpu",
                    options=[{"label": " Enabled", "value": "yes"}],
                    value=["yes"],
                ),
                _label("Verbose logging:", width="140px"),
                dcc.Checklist(
                    id="verbose",
                    options=[{"label": " Enabled", "value": "yes"}],
                    value=["yes"],
                ),
            ),
        ]),

        # ================================================================
        # ADVANCED SETTINGS — collapsible, rarely changed
        # ================================================================

        html.Details([
            html.Summary("Advanced Settings", style={
                "fontSize": "18px", "fontWeight": "bold", "cursor": "pointer",
                "color": "#2c3e50", "padding": "12px 0", "userSelect": "none",
            }),

            html.Div([
                # Nuclei segmentation
                _section("Nuclei Segmentation (StarDist)", [
                    _row(
                        _label("Denoise sigma:"),
                        dcc.Input(id="nuclei-sigma-denoise", type="number", value=2.0,
                                  step=0.1, style={"width": "80px", "padding": "5px"}),
                        _label("Background sigma:", width="140px"),
                        dcc.Input(id="nuclei-sigma-background", type="number", value=50.0,
                                  step=1, style={"width": "80px", "padding": "5px"}),
                    ),
                    _row(
                        _label("Min area (pixels):"),
                        dcc.Input(id="nuclei-min-area", type="number", value=300,
                                  step=10, style={"width": "80px", "padding": "5px"}),
                        _label("Max area (pixels):", width="160px"),
                        dcc.Input(id="nuclei-max-area", type="number", value=15000,
                                  step=100, style={"width": "80px", "padding": "5px"}),
                    ),
                ]),

                # Hidden — always cyto3
                dcc.Input(id="cell-model", type="hidden", value="cyto3"),

                # Aggregate segmentation
                _section("Aggregate Segmentation", [
                    _row(
                        _label("Method:"),
                        dcc.Dropdown(
                            id="aggregate-method",
                            options=[{"label": m, "value": m} for m in AGGREGATE_METHODS],
                            value="filter",
                            clearable=False,
                            style={"width": "140px"},
                        ),
                    ),
                    _row(
                        _label("Min area (pixels):"),
                        dcc.Input(id="aggregate-min-size", type="number", value=9,
                                  step=1, style={"width": "80px", "padding": "5px"}),
                    ),
                    html.Div(id="filter-threshold-row", children=[
                        _row(
                            _label("Intensity threshold:"),
                            dcc.Input(id="aggregate-intensity-threshold", type="number",
                                      value=1.6, step=0.1,
                                      style={"width": "80px", "padding": "5px"}),
                        ),
                    ]),
                    html.Div(id="unet-model-row", children=[
                        _row(
                            _label("UNet model path:"),
                            dcc.Input(id="aggregate-model-path", type="text",
                                      placeholder="/path/to/best.pt",
                                      style={"flex": "1", "padding": "5px"}),
                        ),
                    ]),
                ]),

                # Focus quality metrics
                _section("Focus Quality Metrics", [
                    _row(
                        _label("Compute on:"),
                        dcc.Checklist(
                            id="focus-compute-on",
                            options=[
                                {"label": " nuclei", "value": "nuclei"},
                                {"label": " cells", "value": "cells"},
                            ],
                            value=["nuclei"],
                            inline=True,
                            style={"display": "flex", "gap": "16px"},
                        ),
                    ),
                    _row(
                        _label("Patch metrics:"),
                        dcc.Dropdown(
                            id="focus-patch-metrics",
                            options=[{"label": m, "value": m} for m in PATCH_METRIC_OPTIONS],
                            value=["VarianceLaplacian"],
                            multi=True,
                            style={"flex": "1"},
                        ),
                    ),
                    _row(
                        _label("Global metrics:"),
                        dcc.Dropdown(
                            id="focus-global-metrics",
                            options=[{"label": m, "value": m} for m in GLOBAL_METRIC_OPTIONS],
                            value=["power_log_log_slope"],
                            multi=True,
                            style={"flex": "1"},
                        ),
                    ),
                    _row(
                        _label("Patch size:"),
                        dcc.Input(id="focus-patch-h", type="number", value=40,
                                  style={"width": "60px", "padding": "5px"}),
                        html.Span("x"),
                        dcc.Input(id="focus-patch-w", type="number", value=40,
                                  style={"width": "60px", "padding": "5px"}),
                    ),
                ]),

            ], style={"paddingLeft": "8px"}),

        ], style={"marginTop": "20px", "borderTop": "1px solid #ddd",
                   "paddingTop": "4px"}),

    ], style={"padding": "8px", "maxWidth": "900px"})


def _default_channel_rows():
    """Build the initial channel configuration rows."""
    rows = []
    for i, ch in enumerate(DEFAULT_CHANNELS):
        rows.append(_channel_row(i, ch["name"], ch["pattern"], ch["purpose"]))
    return rows


def _channel_row(index, name="", pattern="", purpose="nuclei"):
    """A single channel configuration row."""
    return html.Div([
        dcc.Input(value=name, type="text", placeholder="Name",
                  id={"type": "channel-name", "index": index},
                  style={"width": "100px", "padding": "4px", "marginRight": "8px"}),
        dcc.Input(value=pattern, type="text", placeholder="Pattern",
                  id={"type": "channel-pattern", "index": index},
                  style={"width": "80px", "padding": "4px", "marginRight": "8px"}),
        dcc.Dropdown(
            options=[
                {"label": "nuclei", "value": "nuclei"},
                {"label": "cells", "value": "cells"},
                {"label": "aggregates", "value": "aggregates"},
                {"label": "other", "value": "other"},
            ],
            value=purpose,
            clearable=False,
            id={"type": "channel-purpose", "index": index},
            style={"width": "130px", "marginRight": "8px"},
        ),
        html.Button("x", id={"type": "channel-remove", "index": index},
                    style={"padding": "2px 8px", "cursor": "pointer",
                           "borderRadius": "4px"}),
    ], style={"display": "flex", "alignItems": "center", "marginBottom": "4px"})
