"""
Settings panel widget for configuring analysis parameters.

Provides controls for segmentation methods, quality thresholds,
and output options.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import customtkinter as ctk
from pathlib import Path
from tkinter import filedialog
from typing import Dict, Any, Optional, Callable


class SettingsPanel(ctk.CTkScrollableFrame):
    """
    Settings panel for configuring analysis parameters.

    Includes:
    - Input/output directory selection
    - Plate format selection
    - Channel configuration
    - Segmentation method selection
    - Quality threshold controls
    """

    def __init__(
        self,
        master,
        on_settings_change: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)

        self.on_settings_change = on_settings_change

        # Default settings
        self.settings: Dict[str, Any] = {
            "input_dir": "",
            "output_dir": "",
            "plate_format": "96",
            "aggregate_method": "unet",
            "blur_threshold": 15.0,
            "blur_reject_pct": 50.0,
            "save_masks": True,
            "save_overlays": True,
        }

        self._create_widgets()

    def _create_widgets(self):
        """Create the settings widgets."""
        # ========== Input/Output Section ==========
        self._create_section_header("Input / Output")

        # Input directory
        input_frame = ctk.CTkFrame(self, fg_color="transparent")
        input_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(input_frame, text="Input Directory:").pack(anchor="w")

        input_row = ctk.CTkFrame(input_frame, fg_color="transparent")
        input_row.pack(fill="x")

        self.input_entry = ctk.CTkEntry(input_row, width=200, corner_radius=0)
        self.input_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(
            input_row,
            text="...",
            width=40,
            corner_radius=0,
            command=self._browse_input
        ).pack(side="left")

        # Output directory
        output_frame = ctk.CTkFrame(self, fg_color="transparent")
        output_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(output_frame, text="Output Directory:").pack(anchor="w")

        output_row = ctk.CTkFrame(output_frame, fg_color="transparent")
        output_row.pack(fill="x")

        self.output_entry = ctk.CTkEntry(output_row, width=200, corner_radius=0)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))

        ctk.CTkButton(
            output_row,
            text="...",
            width=40,
            corner_radius=0,
            command=self._browse_output
        ).pack(side="left")

        # ========== Plate Section ==========
        self._create_section_header("Plate Configuration")

        plate_frame = ctk.CTkFrame(self, fg_color="transparent")
        plate_frame.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(plate_frame, text="Plate Format:").pack(anchor="w")

        self.plate_format_var = ctk.StringVar(value="96")
        self.plate_segmented = ctk.CTkSegmentedButton(
            plate_frame,
            values=["96-well", "384-well"],
            command=self._on_plate_format_segmented_change,
            corner_radius=0
        )
        self.plate_segmented.set("96-well")
        self.plate_segmented.pack(anchor="w", pady=(5, 0))

        # ========== Segmentation Section ==========
        self._create_section_header("Segmentation")

        seg_frame = ctk.CTkFrame(self, fg_color="transparent")
        seg_frame.pack(fill="x", padx=10, pady=5)

        # Aggregate method
        ctk.CTkLabel(seg_frame, text="Aggregate Segmentation:").pack(anchor="w")

        self.agg_method_var = ctk.StringVar(value="unet")
        self.agg_segmented = ctk.CTkSegmentedButton(
            seg_frame,
            values=["UNet", "Filter-based"],
            command=self._on_agg_method_segmented_change,
            corner_radius=0
        )
        self.agg_segmented.set("UNet")
        self.agg_segmented.pack(anchor="w", pady=(5, 10))

        # Model selection (for UNet)
        self.model_frame = ctk.CTkFrame(seg_frame, fg_color="transparent")
        self.model_frame.pack(fill="x", pady=5)

        ctk.CTkLabel(self.model_frame, text="UNet Model:").pack(anchor="w")

        self.model_var = ctk.StringVar(value="unet_baseline")
        self.model_dropdown = ctk.CTkOptionMenu(
            self.model_frame,
            values=[
                "unet_baseline",
                "unet_residual",
                "unet_attention",
                "unet_res_attention",
                "unet_full",
            ],
            variable=self.model_var,
            corner_radius=0,
            command=self._on_settings_change
        )
        self.model_dropdown.pack(anchor="w")

        # ========== Quality Section ==========
        self._create_section_header("Quality Control")

        quality_frame = ctk.CTkFrame(self, fg_color="transparent")
        quality_frame.pack(fill="x", padx=10, pady=5)

        # Blur threshold slider
        ctk.CTkLabel(quality_frame, text="Blur Threshold:").pack(anchor="w")

        blur_slider_frame = ctk.CTkFrame(quality_frame, fg_color="transparent")
        blur_slider_frame.pack(fill="x")

        self.blur_threshold_var = ctk.DoubleVar(value=15.0)
        self.blur_slider = ctk.CTkSlider(
            blur_slider_frame,
            from_=1,
            to=50,
            variable=self.blur_threshold_var,
            command=self._on_blur_threshold_change
        )
        self.blur_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.blur_value_label = ctk.CTkLabel(blur_slider_frame, text="15.0")
        self.blur_value_label.pack(side="left")

        # Blur reject percentage
        ctk.CTkLabel(
            quality_frame,
            text="Reject images with blur > (%):"
        ).pack(anchor="w", pady=(10, 0))

        reject_slider_frame = ctk.CTkFrame(quality_frame, fg_color="transparent")
        reject_slider_frame.pack(fill="x")

        self.reject_pct_var = ctk.DoubleVar(value=50.0)
        self.reject_slider = ctk.CTkSlider(
            reject_slider_frame,
            from_=10,
            to=90,
            variable=self.reject_pct_var,
            command=self._on_reject_pct_change
        )
        self.reject_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.reject_value_label = ctk.CTkLabel(reject_slider_frame, text="50%")
        self.reject_value_label.pack(side="left")

        # ========== Output Section ==========
        self._create_section_header("Output Options")

        output_options_frame = ctk.CTkFrame(self, fg_color="transparent")
        output_options_frame.pack(fill="x", padx=10, pady=5)

        self.save_masks_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            output_options_frame,
            text="Save segmentation masks",
            variable=self.save_masks_var,
            command=self._on_settings_change
        ).pack(anchor="w")

        self.save_overlays_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            output_options_frame,
            text="Save overlay images",
            variable=self.save_overlays_var,
            command=self._on_settings_change
        ).pack(anchor="w")

        self.save_stats_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            output_options_frame,
            text="Export statistics (Parquet)",
            variable=self.save_stats_var,
            command=self._on_settings_change
        ).pack(anchor="w")

    def _create_section_header(self, title: str):
        """Create a section header with separator."""
        frame = ctk.CTkFrame(self, fg_color="transparent")
        frame.pack(fill="x", padx=5, pady=(15, 5))

        ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(anchor="w")

        separator = ctk.CTkFrame(frame, height=1, fg_color="gray")
        separator.pack(fill="x", pady=(2, 0))

    def _browse_input(self):
        """Open directory browser for input."""
        directory = filedialog.askdirectory(title="Select Input Directory")
        if directory:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, directory)
            self.settings["input_dir"] = directory
            self._on_settings_change()

    def _browse_output(self):
        """Open directory browser for output."""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, directory)
            self.settings["output_dir"] = directory
            self._on_settings_change()

    def _on_plate_format_change(self):
        """Handle plate format change."""
        self.settings["plate_format"] = self.plate_format_var.get()
        self._on_settings_change()

    def _on_plate_format_segmented_change(self, value):
        """Handle plate format segmented button change."""
        plate_format = "96" if value == "96-well" else "384"
        self.plate_format_var.set(plate_format)
        self.settings["plate_format"] = plate_format
        self._on_settings_change()

    def _on_agg_method_segmented_change(self, value):
        """Handle aggregate method segmented button change."""
        method = "unet" if value == "UNet" else "filter"
        self.agg_method_var.set(method)
        self.settings["aggregate_method"] = method
        self._on_settings_change()

    def _on_blur_threshold_change(self, value):
        """Handle blur threshold slider change."""
        self.blur_value_label.configure(text=f"{value:.1f}")
        self.settings["blur_threshold"] = value
        self._on_settings_change()

    def _on_reject_pct_change(self, value):
        """Handle reject percentage slider change."""
        self.reject_value_label.configure(text=f"{value:.0f}%")
        self.settings["blur_reject_pct"] = value
        self._on_settings_change()

    def _on_settings_change(self, *args):
        """Notify callback of settings change."""
        self._update_settings()
        if self.on_settings_change:
            self.on_settings_change(self.get_settings())

    def _update_settings(self):
        """Update settings dict from widget values."""
        self.settings["input_dir"] = self.input_entry.get()
        self.settings["output_dir"] = self.output_entry.get()
        self.settings["plate_format"] = self.plate_format_var.get()
        self.settings["aggregate_method"] = self.agg_method_var.get()
        self.settings["blur_threshold"] = self.blur_threshold_var.get()
        self.settings["blur_reject_pct"] = self.reject_pct_var.get()
        self.settings["save_masks"] = self.save_masks_var.get()
        self.settings["save_overlays"] = self.save_overlays_var.get()

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings."""
        self._update_settings()
        return self.settings.copy()

    def set_settings(self, settings: Dict[str, Any]):
        """Set settings from dict."""
        if "input_dir" in settings:
            self.input_entry.delete(0, "end")
            self.input_entry.insert(0, settings["input_dir"])

        if "output_dir" in settings:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, settings["output_dir"])

        if "plate_format" in settings:
            self.plate_format_var.set(settings["plate_format"])
            plate_label = "96-well" if settings["plate_format"] == "96" else "384-well"
            self.plate_segmented.set(plate_label)

        if "aggregate_method" in settings:
            self.agg_method_var.set(settings["aggregate_method"])
            agg_label = "UNet" if settings["aggregate_method"] == "unet" else "Filter-based"
            self.agg_segmented.set(agg_label)

        if "blur_threshold" in settings:
            self.blur_threshold_var.set(settings["blur_threshold"])
            self.blur_value_label.configure(text=f"{settings['blur_threshold']:.1f}")

        if "blur_reject_pct" in settings:
            self.reject_pct_var.set(settings["blur_reject_pct"])
            self.reject_value_label.configure(text=f"{settings['blur_reject_pct']:.0f}%")

        if "save_masks" in settings:
            self.save_masks_var.set(settings["save_masks"])

        if "save_overlays" in settings:
            self.save_overlays_var.set(settings["save_overlays"])

        self.settings.update(settings)

    def get_plate_format(self) -> str:
        """Get current plate format."""
        return self.plate_format_var.get()
