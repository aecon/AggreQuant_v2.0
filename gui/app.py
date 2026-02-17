"""
Main GUI application for AggreQuant.

Provides a user-friendly interface for biologists to:
- Select control wells on plate grids
- Configure analysis parameters
- Run the analysis pipeline
- Monitor progress
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional, Dict, Any
import threading
import yaml

from .widgets.plate_selector import PlateSelector
from .widgets.control_panel import ControlPanel
from .widgets.settings_panel import SettingsPanel
from .widgets.progress_panel import ProgressPanel


# Set appearance to light mode (white background)
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class AggreQuantApp(ctk.CTk):
    """
    Main application window for AggreQuant.

    Layout:
    +--------------------------------------------------+
    |  Menu Bar                                        |
    +--------------------------------------------------+
    |  Left Panel          |  Center Panel            |
    |  - Control Panel     |  - Plate Selector        |
    |  - Settings Panel    |                          |
    |                      +--------------------------+
    |                      |  Progress Panel          |
    +--------------------------------------------------+
    """

    def __init__(self):
        super().__init__()

        self.title("AggreQuant - Aggregate Quantification")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # State
        self._analysis_thread: Optional[threading.Thread] = None
        self._cancel_requested = False

        self._create_menu()
        self._create_layout()
        self._connect_callbacks()

    def _create_menu(self):
        """Create the menu bar."""
        self.menu_bar = ctk.CTkFrame(self, height=40)
        self.menu_bar.pack(fill="x", padx=5, pady=5)

        # File operations
        file_frame = ctk.CTkFrame(self.menu_bar, fg_color="transparent")
        file_frame.pack(side="left", padx=10)

        ctk.CTkButton(
            file_frame,
            text="📂 Load Config",
            width=100,
            corner_radius=0,
            command=self._load_config
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            file_frame,
            text="💾 Save Config",
            width=100,
            corner_radius=0,
            command=self._save_config
        ).pack(side="left", padx=2)

    def _create_layout(self):
        """Create the main layout."""
        # Main container
        main_frame = ctk.CTkFrame(self)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Configure grid
        main_frame.grid_columnconfigure(0, weight=0)  # Left panel (fixed)
        main_frame.grid_columnconfigure(1, weight=1)  # Center panel (expandable)
        main_frame.grid_rowconfigure(0, weight=1)

        # ========== Left Panel ==========
        left_panel = ctk.CTkFrame(main_frame, width=280)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        left_panel.grid_propagate(False)

        # Control Panel
        self.control_panel = ControlPanel(left_panel)
        self.control_panel.pack(fill="x", padx=5, pady=5)

        # Settings Panel
        self.settings_panel = SettingsPanel(left_panel, width=260)
        self.settings_panel.pack(fill="both", expand=True, padx=5, pady=5)

        # ========== Center Panel ==========
        center_panel = ctk.CTkFrame(main_frame)
        center_panel.grid(row=0, column=1, sticky="nsew")

        center_panel.grid_rowconfigure(0, weight=2)
        center_panel.grid_rowconfigure(1, weight=1)
        center_panel.grid_columnconfigure(0, weight=1)

        # Plate Selector (top)
        plate_frame = ctk.CTkFrame(center_panel)
        plate_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.plate_selector = PlateSelector(
            plate_frame,
            plate_format="96",
            on_selection_change=self._on_selection_change
        )
        self.plate_selector.pack(expand=True, fill="both", padx=10, pady=10)

        # Progress Panel (bottom)
        self.progress_panel = ProgressPanel(center_panel)
        self.progress_panel.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def _connect_callbacks(self):
        """Connect widget callbacks."""
        # Control panel -> Plate selector
        self.control_panel.set_on_assign(self._on_assign_control)
        self.control_panel.set_on_clear(self._on_clear_selection)

        # Settings panel -> Plate selector (plate format change)
        self.settings_panel.on_settings_change = self._on_settings_change

        # Progress panel -> Run/Cancel
        self.progress_panel.set_on_run(self._run_analysis)
        self.progress_panel.set_on_cancel(self._cancel_analysis)

    def _on_selection_change(self, selected_wells):
        """Handle well selection change."""
        count = len(selected_wells)
        if count > 0:
            self.progress_panel.log_info(f"Selected {count} well(s)")

    def _on_assign_control(self, control_type: str):
        """Handle control type assignment."""
        selected = self.plate_selector.get_selected_wells()
        if not selected:
            messagebox.showwarning(
                "No Selection",
                "Please select wells on the plate first."
            )
            return

        self.plate_selector.assign_control_type(control_type)
        self.progress_panel.log_info(
            f"Assigned {len(selected)} well(s) as '{control_type}'"
        )

    def _on_clear_selection(self):
        """Handle clear selection."""
        self.plate_selector.clear_selection()

    def _on_settings_change(self, settings: Dict[str, Any]):
        """Handle settings change."""
        # Update plate format if changed
        plate_format = settings.get("plate_format", "96")
        if plate_format != self.plate_selector.plate_format:
            self.plate_selector.set_plate_format(plate_format)
            self.progress_panel.log_info(f"Changed to {plate_format}-well plate")

    def _load_config(self):
        """Load configuration from YAML file."""
        filepath = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                config = yaml.safe_load(f)

            # Apply settings
            if "settings" in config:
                self.settings_panel.set_settings(config["settings"])

            # Apply control assignments
            if "control_wells" in config:
                # Convert from {type: [wells]} to {well: type}
                assignments = {}
                for control_type, wells in config["control_wells"].items():
                    for well in wells:
                        assignments[well] = control_type
                self.plate_selector.set_control_assignments(assignments)

            # Apply plate format
            if "plate_format" in config:
                self.plate_selector.set_plate_format(config["plate_format"])

            self.progress_panel.log_info(f"Loaded configuration from {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load configuration:\n{e}")
            self.progress_panel.log_error(f"Failed to load config: {e}")

    def _save_config(self):
        """Save configuration to YAML file."""
        filepath = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            # Get current settings
            settings = self.settings_panel.get_settings()

            # Get control assignments as {type: [wells]}
            assignments = self.plate_selector.get_control_assignments()
            control_wells: Dict[str, list] = {}
            for well, control_type in assignments.items():
                if control_type not in control_wells:
                    control_wells[control_type] = []
                control_wells[control_type].append(well)

            # Sort well lists
            for wells in control_wells.values():
                wells.sort()

            config = {
                "plate_format": self.plate_selector.plate_format,
                "settings": settings,
                "control_wells": control_wells,
            }

            with open(filepath, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            self.progress_panel.log_info(f"Saved configuration to {filepath}")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration:\n{e}")
            self.progress_panel.log_error(f"Failed to save config: {e}")

    def _run_analysis(self):
        """Start the analysis in a background thread."""
        settings = self.settings_panel.get_settings()

        # Validate inputs
        if not settings.get("input_dir"):
            messagebox.showwarning("Missing Input", "Please select an input directory.")
            return

        if not settings.get("output_dir"):
            messagebox.showwarning("Missing Output", "Please select an output directory.")
            return

        # Get control assignments
        control_assignments = self.plate_selector.get_control_assignments()
        if not control_assignments:
            result = messagebox.askyesno(
                "No Controls",
                "No control wells have been assigned. Continue anyway?"
            )
            if not result:
                return

        # Start analysis in background thread
        self._cancel_requested = False
        self.progress_panel.set_running(True)
        self.progress_panel.log_info("Starting analysis...")

        self._analysis_thread = threading.Thread(
            target=self._analysis_worker,
            args=(settings, control_assignments),
            daemon=True
        )
        self._analysis_thread.start()

    def _analysis_worker(self, settings: Dict[str, Any], controls: Dict[str, str]):
        """
        Background worker for running analysis.

        Runs the AggreQuant pipeline with progress reporting back to GUI.
        """
        try:
            from gui.pipeline_runner import run_pipeline_from_dict

            # Build config dict for pipeline
            config_dict = {
                "input_dir": settings.get("input_dir"),
                "output_dir": settings.get("output_dir"),
                "plate_format": self.plate_selector.plate_format,
                "aggregate_method": settings.get("aggregate_method", "unet"),
                "blur_threshold": settings.get("blur_threshold", 15.0),
                "blur_reject_pct": settings.get("blur_reject_pct", 50.0),
                "save_masks": settings.get("save_masks", True),
                "save_overlays": settings.get("save_overlays", True),
                "control_wells": controls,  # {well_id: control_type}
                "use_gpu": True,
            }

            # Progress callback that updates GUI
            def progress_callback(progress: float, message: str):
                if self._cancel_requested:
                    raise InterruptedError("Analysis cancelled by user")
                self.after(0, lambda p=progress, m=message: self._update_progress(p, m))

            # Run pipeline
            result = run_pipeline_from_dict(
                config_dict,
                progress_callback=progress_callback,
                verbose=True,
            )

            # Report completion
            self.after(0, lambda: self.progress_panel.log_info(
                f"Analysis complete: {result.total_n_wells_processed} wells, "
                f"{result.total_n_cells} cells detected"
            ))
            if result.ssmd is not None:
                self.after(0, lambda: self.progress_panel.log_info(
                    f"SSMD: {result.ssmd:.3f}"
                ))
            self.after(0, lambda: self.progress_panel.complete(True))

        except InterruptedError:
            self.after(0, lambda: self.progress_panel.log_warning("Analysis cancelled by user"))
            self.after(0, lambda: self.progress_panel.complete(False))

        except ImportError as e:
            # Pipeline dependencies not available - fall back to simulation
            self.after(0, lambda: self.progress_panel.log_warning(
                f"Pipeline dependencies not fully installed: {e}"
            ))
            self.after(0, lambda: self.progress_panel.log_info(
                "Running in simulation mode..."
            ))
            self._run_simulation()

        except Exception as e:
            self.after(0, lambda err=str(e): self.progress_panel.log_error(f"Analysis failed: {err}"))
            self.after(0, lambda: self.progress_panel.complete(False))

    def _run_simulation(self):
        """Run simulated analysis for testing when pipeline is not available."""
        import time

        steps = [
            ("Loading images...", 0.1),
            ("Segmenting nuclei...", 0.3),
            ("Segmenting cells...", 0.5),
            ("Segmenting aggregates...", 0.7),
            ("Computing statistics...", 0.9),
            ("Exporting results...", 1.0),
        ]

        for message, progress in steps:
            if self._cancel_requested:
                self.after(0, lambda: self.progress_panel.complete(False))
                return

            self.after(0, lambda m=message, p=progress: self._update_progress(p, m))
            time.sleep(0.5)

        self.after(0, lambda: self.progress_panel.log_info("Simulation complete"))
        self.after(0, lambda: self.progress_panel.complete(True))

    def _update_progress(self, progress: float, message: str):
        """Update progress from worker thread."""
        self.progress_panel.set_progress(progress, message)
        self.progress_panel.log_info(message)

    def _cancel_analysis(self):
        """Request analysis cancellation."""
        self._cancel_requested = True
        self.progress_panel.log_warning("Cancellation requested...")

    def get_config(self) -> Dict[str, Any]:
        """Get complete configuration as dict."""
        settings = self.settings_panel.get_settings()
        assignments = self.plate_selector.get_control_assignments()

        # Convert assignments to {type: [wells]} format
        control_wells: Dict[str, list] = {}
        for well, control_type in assignments.items():
            if control_type not in control_wells:
                control_wells[control_type] = []
            control_wells[control_type].append(well)

        return {
            "plate_format": self.plate_selector.plate_format,
            "settings": settings,
            "control_wells": control_wells,
        }


def main():
    """Launch the AggreQuant GUI application."""
    app = AggreQuantApp()
    app.mainloop()


if __name__ == "__main__":
    main()
