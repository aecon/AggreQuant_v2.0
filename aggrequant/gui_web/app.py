"""Dash web application for the AggreQuant pipeline.

Usage:
    python -m aggrequant.gui_web.app
    # or via entry point:
    aggrequant-gui
"""

import argparse
import webbrowser
from threading import Timer

from dash import Dash, html, dcc, Input, Output, callback

from aggrequant.gui_web.config import APP_TITLE, APP_HOST, APP_PORT, COLOR_PRIMARY


def create_app():
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        suppress_callback_exceptions=True,
        title=APP_TITLE,
    )

    # Import components (these define layout builders)
    from aggrequant.gui_web.components.settings_form import build_settings_layout
    from aggrequant.gui_web.components.plate_grid import build_plate_layout
    from aggrequant.gui_web.components.batch_queue import build_batch_layout
    from aggrequant.gui_web.components.log_panel import build_progress_layout
    from aggrequant.gui_web.components.results_panel import build_results_layout

    # Import callbacks (registration happens on import)
    import aggrequant.gui_web.callbacks.configuration  # noqa: F401
    import aggrequant.gui_web.callbacks.plate_selector  # noqa: F401
    import aggrequant.gui_web.callbacks.batch           # noqa: F401
    import aggrequant.gui_web.callbacks.progress        # noqa: F401
    import aggrequant.gui_web.callbacks.results         # noqa: F401

    app.layout = html.Div([
        # Header
        html.Div([
            html.H1(APP_TITLE, style={
                "margin": "0", "fontSize": "28px", "color": "white",
            }),
            html.Span("Web GUI", style={
                "color": "#bdc3c7", "fontSize": "14px", "marginLeft": "12px",
            }),
        ], style={
            "backgroundColor": COLOR_PRIMARY, "padding": "14px 24px",
            "display": "flex", "alignItems": "baseline",
        }),

        # Main tabs
        dcc.Tabs(id="main-tabs", value="tab-config", children=[
            dcc.Tab(label="Configuration", value="tab-config"),
            dcc.Tab(label="Plate Selector", value="tab-plate"),
            dcc.Tab(label="Batch Processing", value="tab-batch"),
            dcc.Tab(label="Progress & Log", value="tab-progress"),
            dcc.Tab(label="Results", value="tab-results"),
        ], style={"marginTop": "0"}),

        # Tab content panels
        html.Div(id="panel-config", children=build_settings_layout(),
                 style={"padding": "16px 24px"}),
        html.Div(id="panel-plate", children=build_plate_layout(),
                 style={"padding": "16px 24px", "display": "none"}),
        html.Div(id="panel-batch", children=build_batch_layout(),
                 style={"padding": "16px 24px", "display": "none"}),
        html.Div(id="panel-progress", children=build_progress_layout(),
                 style={"padding": "16px 24px", "display": "none"}),
        html.Div(id="panel-results", children=build_results_layout(),
                 style={"padding": "16px 24px", "display": "none"}),

    ], style={"fontFamily": "Segoe UI, Arial, sans-serif", "maxWidth": "1400px",
              "margin": "0 auto"})

    # Tab visibility callback
    @app.callback(
        [
            Output("panel-config", "style"),
            Output("panel-plate", "style"),
            Output("panel-batch", "style"),
            Output("panel-progress", "style"),
            Output("panel-results", "style"),
        ],
        Input("main-tabs", "value"),
    )
    def toggle_tabs(tab):
        hidden = {"padding": "16px 24px", "display": "none"}
        visible = {"padding": "16px 24px", "display": "block"}
        tab_map = {
            "tab-config": 0, "tab-plate": 1, "tab-batch": 2,
            "tab-progress": 3, "tab-results": 4,
        }
        styles = [hidden] * 5
        styles[tab_map.get(tab, 0)] = visible
        return styles

    return app


def main():
    """Entry point for the AggreQuant web GUI."""
    parser = argparse.ArgumentParser(description="AggreQuant Web GUI")
    parser.add_argument("--host", default=APP_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=APP_PORT, help="Port to serve on")
    parser.add_argument("--no-browser", action="store_true",
                        help="Don't auto-open browser")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    args = parser.parse_args()

    app = create_app()

    if not args.no_browser:
        Timer(1.5, lambda: webbrowser.open(f"http://{args.host}:{args.port}")).start()

    print(f"Starting AggreQuant GUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
