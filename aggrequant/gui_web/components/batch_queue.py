"""Batch processing queue component."""

from dash import html, dcc


def build_batch_layout():
    """Return the Dash layout for the batch processing tab."""
    return html.Div([
        html.H4("Plate Queue", style={
            "borderBottom": "2px solid #3498db", "paddingBottom": "4px",
            "marginBottom": "12px",
        }),

        # Add plates
        html.Div([
            html.Div([
                html.Label("Add plate directories:", style={"fontWeight": "bold"}),
                html.Div([
                    dcc.Input(
                        id="batch-dir-input",
                        placeholder="/path/to/plate/directory",
                        type="text",
                        style={"flex": "1", "padding": "5px"},
                    ),
                    html.Button("Add", id="btn-batch-add",
                                style={"padding": "6px 16px", "borderRadius": "4px",
                                       "cursor": "pointer", "marginLeft": "8px"}),
                ], style={"display": "flex", "marginTop": "4px"}),
            ]),

            html.Div([
                html.Label("Or paste multiple paths (one per line):",
                           style={"fontWeight": "bold", "marginTop": "12px"}),
                dcc.Textarea(
                    id="batch-paste-input",
                    placeholder="/path/to/plate1\n/path/to/plate2\n/path/to/plate3",
                    style={"width": "100%", "height": "80px", "marginTop": "4px",
                           "padding": "5px", "fontFamily": "monospace"},
                ),
                html.Button("Add All", id="btn-batch-paste",
                            style={"padding": "6px 12px", "borderRadius": "4px",
                                   "cursor": "pointer", "marginTop": "4px"}),
            ]),

            html.Div([
                html.Label("Or scan parent directory for plate folders:",
                           style={"fontWeight": "bold", "marginTop": "12px"}),
                html.Div([
                    dcc.Input(
                        id="batch-parent-input",
                        placeholder="/path/to/parent/directory",
                        type="text",
                        style={"flex": "1", "padding": "5px"},
                    ),
                    html.Button("Scan", id="btn-batch-scan",
                                style={"padding": "6px 16px", "borderRadius": "4px",
                                       "cursor": "pointer", "marginLeft": "8px"}),
                ], style={"display": "flex", "marginTop": "4px"}),
            ]),
        ]),

        html.Hr(style={"margin": "16px 0"}),

        # Per-plate config option
        html.Div([
            dcc.Checklist(
                id="batch-per-plate-config",
                options=[{
                    "label": " Use per-plate YAML configs "
                             "(each directory must contain aggrequant_config.yaml)",
                    "value": "yes",
                }],
                value=[],
                style={"marginBottom": "12px"},
            ),
        ]),

        # Queue table
        html.Div(id="batch-queue-display", children=_empty_queue_message()),

        # Controls
        html.Div([
            html.Button("Remove Selected", id="btn-batch-remove",
                        style={"padding": "6px 16px", "borderRadius": "4px",
                               "cursor": "pointer", "marginRight": "8px"}),
            html.Button("Clear All", id="btn-batch-clear",
                        style={"padding": "6px 16px", "borderRadius": "4px",
                               "cursor": "pointer"}),
        ], style={"marginTop": "10px"}),

        # Hidden store for the queue data
        dcc.Store(id="batch-queue-data", data=[]),
    ])


def _empty_queue_message():
    return html.Div(
        "No plates in queue. Add directories above.",
        style={"color": "#95a5a6", "fontStyle": "italic", "padding": "20px",
               "textAlign": "center"},
    )


def render_queue_table(jobs):
    """Render the queue as an HTML table.

    Arguments:
        jobs: List of dicts with keys: input_dir, plate_name, status.

    Returns:
        Dash HTML component.
    """
    if not jobs:
        return _empty_queue_message()

    status_colors = {
        "pending": "#95a5a6",
        "running": "#3498db",
        "done": "#27ae60",
        "failed": "#e74c3c",
        "cancelled": "#f39c12",
    }

    header = html.Tr([
        html.Th("", style={"width": "30px"}),
        html.Th("#", style={"width": "30px"}),
        html.Th("Directory"),
        html.Th("Plate Name", style={"width": "150px"}),
        html.Th("Status", style={"width": "100px"}),
    ])

    rows = []
    for i, job in enumerate(jobs):
        status = job.get("status", "pending")
        color = status_colors.get(status, "#95a5a6")
        rows.append(html.Tr([
            html.Td(dcc.Checklist(
                options=[{"label": "", "value": str(i)}],
                value=[],
                id={"type": "batch-select", "index": i},
            )),
            html.Td(str(i + 1)),
            html.Td(job.get("input_dir", ""), style={
                "fontFamily": "monospace", "fontSize": "13px",
                "maxWidth": "400px", "overflow": "hidden",
                "textOverflow": "ellipsis", "whiteSpace": "nowrap",
            }),
            html.Td(job.get("plate_name", "")),
            html.Td(status.upper(), style={"color": color, "fontWeight": "bold"}),
        ]))

    table_style = {
        "width": "100%", "borderCollapse": "collapse",
        "fontSize": "14px",
    }
    th_style = {
        "textAlign": "left", "padding": "8px", "borderBottom": "2px solid #ddd",
    }
    td_style = {"padding": "6px 8px", "borderBottom": "1px solid #eee"}

    # Apply styles via className workaround: inline on each cell
    styled_header = html.Tr([
        html.Th(cell.children, style={**th_style, **cell.style})
        if hasattr(cell, "style") and cell.style
        else html.Th(cell.children, style=th_style)
        for cell in header.children
    ])

    return html.Table([
        html.Thead(styled_header),
        html.Tbody(rows),
    ], style=table_style)
