"""
Unit tests for GUI components.

Tests widget logic, state management, and data flow without requiring
a display. Uses fixtures to manage tkinter root window lifecycle.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from typing import Set, Dict, Any


# Skip all tests if no display available or customtkinter not installed
try:
    import customtkinter as ctk
    # Try to create a root window to check if display is available
    _test_root = ctk.CTk()
    _test_root.withdraw()
    _test_root.destroy()
    DISPLAY_AVAILABLE = True
except Exception:
    DISPLAY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not DISPLAY_AVAILABLE,
    reason="No display available or customtkinter not installed"
)


@pytest.fixture(scope="module")
def tk_root():
    """Create a single root window for all tests in module."""
    root = ctk.CTk()
    root.withdraw()  # Hide the window
    yield root
    root.destroy()


@pytest.fixture
def plate_selector(tk_root):
    """Create a PlateSelector widget for testing."""
    from gui.widgets.plate_selector import PlateSelector

    selector = PlateSelector(tk_root, plate_format="96")
    yield selector
    selector.destroy()


@pytest.fixture
def plate_selector_384(tk_root):
    """Create a 384-well PlateSelector widget for testing."""
    from gui.widgets.plate_selector import PlateSelector

    selector = PlateSelector(tk_root, plate_format="384")
    yield selector
    selector.destroy()


@pytest.fixture
def control_panel(tk_root):
    """Create a ControlPanel widget for testing."""
    from gui.widgets.control_panel import ControlPanel

    panel = ControlPanel(tk_root)
    yield panel
    panel.destroy()


@pytest.fixture
def settings_panel(tk_root):
    """Create a SettingsPanel widget for testing."""
    from gui.widgets.settings_panel import SettingsPanel

    panel = SettingsPanel(tk_root)
    yield panel
    panel.destroy()


@pytest.fixture
def progress_panel(tk_root):
    """Create a ProgressPanel widget for testing."""
    from gui.widgets.progress_panel import ProgressPanel

    panel = ProgressPanel(tk_root)
    yield panel
    panel.destroy()


class TestPlateSelector:
    """Tests for PlateSelector widget."""

    def test_initialization_96_well(self, plate_selector):
        """Should initialize with correct dimensions for 96-well plate."""
        assert plate_selector.plate_format == "96"
        assert plate_selector.n_rows == 8
        assert plate_selector.n_cols == 12
        assert plate_selector.well_size == 40

    def test_initialization_384_well(self, plate_selector_384):
        """Should initialize with correct dimensions for 384-well plate."""
        assert plate_selector_384.plate_format == "384"
        assert plate_selector_384.n_rows == 16
        assert plate_selector_384.n_cols == 24
        assert plate_selector_384.well_size == 24

    def test_well_buttons_created(self, plate_selector):
        """Should create correct number of well buttons."""
        expected_wells = 8 * 12  # 96 wells
        assert len(plate_selector.well_buttons) == expected_wells

    def test_well_button_ids_format(self, plate_selector):
        """Well button IDs should have correct format."""
        # Check first well
        assert "A01" in plate_selector.well_buttons
        # Check last well
        assert "H12" in plate_selector.well_buttons
        # Check middle well
        assert "D06" in plate_selector.well_buttons

    def test_initial_selection_empty(self, plate_selector):
        """Initial selection should be empty."""
        assert len(plate_selector.selected_wells) == 0
        assert len(plate_selector.get_selected_wells()) == 0

    def test_initial_control_assignments_empty(self, plate_selector):
        """Initial control assignments should be empty."""
        assert len(plate_selector.control_assignments) == 0
        assert len(plate_selector.get_control_assignments()) == 0

    def test_set_selected_wells(self, plate_selector):
        """Should set selected wells correctly."""
        wells = {"A01", "A02", "B01"}
        plate_selector.set_selected_wells(wells)

        assert plate_selector.get_selected_wells() == wells

    def test_clear_selection(self, plate_selector):
        """Should clear all selections."""
        plate_selector.set_selected_wells({"A01", "A02"})
        plate_selector.clear_selection()

        assert len(plate_selector.get_selected_wells()) == 0

    def test_assign_control_type(self, plate_selector):
        """Should assign control type to selected wells."""
        # Select some wells
        plate_selector.set_selected_wells({"A01", "A02", "A03"})

        # Assign control type
        plate_selector.assign_control_type("negative")

        # Check assignments
        assignments = plate_selector.get_control_assignments()
        assert assignments["A01"] == "negative"
        assert assignments["A02"] == "negative"
        assert assignments["A03"] == "negative"

        # Selection should be cleared after assignment
        assert len(plate_selector.get_selected_wells()) == 0

    def test_assign_multiple_control_types(self, plate_selector):
        """Should handle multiple control types."""
        # Assign negative controls
        plate_selector.set_selected_wells({"A01", "A02"})
        plate_selector.assign_control_type("negative")

        # Assign NT controls
        plate_selector.set_selected_wells({"H11", "H12"})
        plate_selector.assign_control_type("NT")

        assignments = plate_selector.get_control_assignments()
        assert assignments["A01"] == "negative"
        assert assignments["H12"] == "NT"

    def test_remove_control_assignment(self, plate_selector):
        """Should remove single control assignment."""
        plate_selector.set_selected_wells({"A01", "A02"})
        plate_selector.assign_control_type("negative")

        plate_selector.remove_control_assignment("A01")

        assignments = plate_selector.get_control_assignments()
        assert "A01" not in assignments
        assert "A02" in assignments

    def test_clear_control_assignments(self, plate_selector):
        """Should clear all control assignments."""
        plate_selector.set_selected_wells({"A01", "A02"})
        plate_selector.assign_control_type("negative")

        plate_selector.clear_control_assignments()

        assert len(plate_selector.get_control_assignments()) == 0

    def test_get_wells_by_control_type(self, plate_selector):
        """Should return wells for specific control type."""
        plate_selector.set_selected_wells({"A01", "A02"})
        plate_selector.assign_control_type("negative")

        plate_selector.set_selected_wells({"H11", "H12"})
        plate_selector.assign_control_type("NT")

        negative_wells = plate_selector.get_wells_by_control_type("negative")
        assert set(negative_wells) == {"A01", "A02"}

        nt_wells = plate_selector.get_wells_by_control_type("NT")
        assert set(nt_wells) == {"H11", "H12"}

    def test_set_control_assignments(self, plate_selector):
        """Should set control assignments from dict."""
        assignments = {
            "A01": "negative",
            "A02": "negative",
            "H12": "NT"
        }
        plate_selector.set_control_assignments(assignments)

        assert plate_selector.get_control_assignments() == assignments

    def test_selection_callback(self, plate_selector):
        """Should call callback on selection change."""
        callback = Mock()
        plate_selector.on_selection_change = callback

        plate_selector.set_selected_wells({"A01"})

        callback.assert_called_once()
        # Callback receives a copy of selected wells
        call_arg = callback.call_args[0][0]
        assert "A01" in call_arg

    def test_set_plate_format_changes_grid(self, plate_selector):
        """Changing plate format should recreate the grid."""
        assert plate_selector.plate_format == "96"
        assert plate_selector.n_rows == 8

        plate_selector.set_plate_format("384")

        assert plate_selector.plate_format == "384"
        assert plate_selector.n_rows == 16
        assert plate_selector.n_cols == 24
        assert len(plate_selector.well_buttons) == 384

    def test_set_plate_format_clears_state(self, plate_selector):
        """Changing plate format should clear selections and assignments."""
        plate_selector.set_selected_wells({"A01"})
        plate_selector.set_control_assignments({"A02": "negative"})

        plate_selector.set_plate_format("384")

        assert len(plate_selector.get_selected_wells()) == 0
        assert len(plate_selector.get_control_assignments()) == 0


class TestControlPanel:
    """Tests for ControlPanel widget."""

    def test_initialization(self, control_panel):
        """Should initialize with default state."""
        assert control_panel.custom_types == []

    def test_on_assign_callback(self, control_panel):
        """Should call on_assign callback when assignment button clicked."""
        callback = Mock()
        control_panel.set_on_assign(callback)

        # Simulate clicking assignment
        control_panel._on_assign_click("negative")

        callback.assert_called_once_with("negative")

    def test_on_clear_callback(self, control_panel):
        """Should call on_clear callback."""
        callback = Mock()
        control_panel.set_on_clear(callback)

        control_panel._on_clear_selection()

        callback.assert_called_once()

    def test_add_custom_control_type(self, control_panel):
        """Should add custom control type."""
        control_panel.custom_entry.insert(0, "MyCustomControl")
        control_panel._on_add_custom()

        assert "MyCustomControl" in control_panel.custom_types

    def test_add_custom_clears_entry(self, control_panel):
        """Adding custom type should clear the entry field."""
        control_panel.custom_entry.insert(0, "TestControl")
        control_panel._on_add_custom()

        assert control_panel.custom_entry.get() == ""

    def test_duplicate_custom_not_added(self, control_panel):
        """Should not add duplicate custom types."""
        control_panel.custom_entry.insert(0, "Custom1")
        control_panel._on_add_custom()

        control_panel.custom_entry.insert(0, "Custom1")
        control_panel._on_add_custom()

        assert control_panel.custom_types.count("Custom1") == 1

    def test_empty_custom_not_added(self, control_panel):
        """Should not add empty custom type."""
        control_panel.custom_entry.insert(0, "   ")
        control_panel._on_add_custom()

        assert len(control_panel.custom_types) == 0

    def test_darken_color(self, control_panel):
        """Should darken hex colors correctly."""
        # White should become gray
        darkened = control_panel._darken_color("#ffffff")
        assert darkened == "#cccccc"

        # Black should stay black
        darkened = control_panel._darken_color("#000000")
        assert darkened == "#000000"


class TestSettingsPanel:
    """Tests for SettingsPanel widget."""

    def test_default_settings(self, settings_panel):
        """Should have correct default settings."""
        settings = settings_panel.get_settings()

        assert settings["input_dir"] == ""
        assert settings["output_dir"] == ""
        assert settings["plate_format"] == "96"
        assert settings["aggregate_method"] == "unet"
        assert settings["blur_threshold"] == 15.0
        assert settings["blur_reject_pct"] == 50.0
        assert settings["save_masks"] == True
        assert settings["save_overlays"] == True

    def test_set_settings(self, settings_panel):
        """Should set settings from dict."""
        new_settings = {
            "input_dir": "/path/to/input",
            "output_dir": "/path/to/output",
            "plate_format": "384",
            "aggregate_method": "filter",
            "blur_threshold": 20.0,
        }

        settings_panel.set_settings(new_settings)
        result = settings_panel.get_settings()

        assert result["input_dir"] == "/path/to/input"
        assert result["output_dir"] == "/path/to/output"
        assert result["plate_format"] == "384"
        assert result["aggregate_method"] == "filter"
        assert result["blur_threshold"] == 20.0

    def test_plate_format_segmented_change(self, settings_panel):
        """Should update plate format via segmented button."""
        settings_panel._on_plate_format_segmented_change("384-well")

        assert settings_panel.plate_format_var.get() == "384"
        assert settings_panel.get_settings()["plate_format"] == "384"

    def test_agg_method_segmented_change(self, settings_panel):
        """Should update aggregate method via segmented button."""
        settings_panel._on_agg_method_segmented_change("Filter-based")

        assert settings_panel.agg_method_var.get() == "filter"
        assert settings_panel.get_settings()["aggregate_method"] == "filter"

    def test_blur_threshold_change(self, settings_panel):
        """Should update blur threshold via slider."""
        # Set the variable first (simulating slider interaction)
        settings_panel.blur_threshold_var.set(25.0)
        settings_panel._on_blur_threshold_change(25.0)

        assert settings_panel.get_settings()["blur_threshold"] == 25.0

    def test_reject_pct_change(self, settings_panel):
        """Should update reject percentage via slider."""
        # Set the variable first (simulating slider interaction)
        settings_panel.reject_pct_var.set(70.0)
        settings_panel._on_reject_pct_change(70.0)

        assert settings_panel.get_settings()["blur_reject_pct"] == 70.0

    def test_settings_change_callback(self, settings_panel):
        """Should call callback on settings change."""
        callback = Mock()
        settings_panel.on_settings_change = callback

        settings_panel._on_settings_change()

        callback.assert_called_once()
        # Callback receives settings dict
        call_arg = callback.call_args[0][0]
        assert "plate_format" in call_arg

    def test_get_plate_format(self, settings_panel):
        """get_plate_format should return current plate format."""
        assert settings_panel.get_plate_format() == "96"

        settings_panel._on_plate_format_segmented_change("384-well")
        assert settings_panel.get_plate_format() == "384"


class TestProgressPanel:
    """Tests for ProgressPanel widget."""

    def test_initial_state(self, progress_panel):
        """Should start in non-running state."""
        assert progress_panel._is_running == False

    def test_set_running_true(self, progress_panel):
        """Setting running=True should update state and buttons."""
        progress_panel.set_running(True)

        assert progress_panel._is_running == True

    def test_set_running_false(self, progress_panel):
        """Setting running=False should update state and buttons."""
        progress_panel.set_running(True)
        progress_panel.set_running(False)

        assert progress_panel._is_running == False

    def test_set_progress(self, progress_panel):
        """Should set progress bar value."""
        progress_panel.set_progress(0.5, "Processing...")

        # Progress bar should be at 50%
        assert abs(progress_panel.progress_bar.get() - 0.5) < 0.01

    def test_set_status(self, progress_panel):
        """Should set status label text."""
        progress_panel.set_status("Custom status")

        # Can't easily check label text, but method shouldn't raise

    def test_log_methods(self, progress_panel):
        """Log methods should add text to log box."""
        progress_panel.log_info("Info message")
        progress_panel.log_warning("Warning message")
        progress_panel.log_error("Error message")

        # Get log content
        progress_panel.log_textbox.configure(state="normal")
        content = progress_panel.log_textbox.get("1.0", "end")
        progress_panel.log_textbox.configure(state="disabled")

        assert "Info message" in content
        assert "Warning message" in content
        assert "Error message" in content
        assert "[INFO]" in content
        assert "[WARN]" in content
        assert "[ERROR]" in content

    def test_clear_log(self, progress_panel):
        """Should clear log content."""
        progress_panel.log_info("Test message")
        progress_panel.clear_log()

        progress_panel.log_textbox.configure(state="normal")
        content = progress_panel.log_textbox.get("1.0", "end").strip()
        progress_panel.log_textbox.configure(state="disabled")

        assert content == ""

    def test_reset(self, progress_panel):
        """Reset should restore initial state."""
        progress_panel.set_running(True)
        progress_panel.set_progress(0.8, "Working")

        progress_panel.reset()

        assert progress_panel._is_running == False
        assert abs(progress_panel.progress_bar.get() - 0.0) < 0.01

    def test_complete_success(self, progress_panel):
        """complete(True) should set success state."""
        progress_panel.set_running(True)
        progress_panel.complete(True)

        assert progress_panel._is_running == False
        assert abs(progress_panel.progress_bar.get() - 1.0) < 0.01

    def test_complete_failure(self, progress_panel):
        """complete(False) should set failure state."""
        progress_panel.set_running(True)
        progress_panel.complete(False)

        assert progress_panel._is_running == False

    def test_on_run_callback(self, progress_panel):
        """Should call on_run callback when run button logic triggered."""
        callback = Mock()
        progress_panel.set_on_run(callback)

        progress_panel._on_run_click()

        callback.assert_called_once()

    def test_on_cancel_callback(self, progress_panel):
        """Should call on_cancel callback when cancel button logic triggered."""
        callback = Mock()
        progress_panel.set_on_cancel(callback)

        progress_panel._on_cancel_click()

        callback.assert_called_once()


class TestControlColors:
    """Tests for control color definitions."""

    def test_control_colors_defined(self):
        """Should have colors for negative and NT controls."""
        from gui.widgets.plate_selector import CONTROL_COLORS

        assert "negative" in CONTROL_COLORS
        assert "NT" in CONTROL_COLORS
        assert "custom" in CONTROL_COLORS

    def test_control_colors_are_hex(self):
        """Control colors should be valid hex color strings."""
        from gui.widgets.plate_selector import CONTROL_COLORS

        for name, color in CONTROL_COLORS.items():
            assert color.startswith("#")
            assert len(color) == 7
            # Should be valid hex
            int(color[1:], 16)


class TestWellConstants:
    """Tests for well color constants."""

    def test_well_colors_defined(self):
        """Should have well state colors defined."""
        from gui.widgets.plate_selector import WELL_EMPTY, WELL_HOVER, WELL_SELECTED

        assert WELL_EMPTY.startswith("#")
        assert WELL_HOVER.startswith("#")
        assert WELL_SELECTED.startswith("#")


class TestIntegration:
    """Integration tests for GUI component interactions."""

    def test_control_panel_plate_selector_flow(self, tk_root):
        """Test flow from ControlPanel to PlateSelector."""
        from gui.widgets.plate_selector import PlateSelector
        from gui.widgets.control_panel import ControlPanel

        plate_selector = PlateSelector(tk_root, plate_format="96")
        control_panel = ControlPanel(tk_root)

        # Connect control panel to plate selector
        control_panel.set_on_assign(plate_selector.assign_control_type)
        control_panel.set_on_clear(plate_selector.clear_selection)

        # Select wells on plate
        plate_selector.set_selected_wells({"A01", "A02"})

        # Simulate control type assignment via control panel
        control_panel._on_assign_click("negative")

        # Verify assignment on plate
        assignments = plate_selector.get_control_assignments()
        assert assignments["A01"] == "negative"
        assert assignments["A02"] == "negative"

        # Cleanup
        plate_selector.destroy()
        control_panel.destroy()

    def test_settings_panel_plate_selector_format_sync(self, tk_root):
        """Test plate format sync between SettingsPanel and PlateSelector."""
        from gui.widgets.plate_selector import PlateSelector
        from gui.widgets.settings_panel import SettingsPanel

        plate_selector = PlateSelector(tk_root, plate_format="96")
        settings_panel = SettingsPanel(tk_root)

        # Callback to update plate selector when settings change
        def on_settings_change(settings):
            plate_format = settings.get("plate_format", "96")
            if plate_format != plate_selector.plate_format:
                plate_selector.set_plate_format(plate_format)

        settings_panel.on_settings_change = on_settings_change

        # Change plate format in settings
        settings_panel._on_plate_format_segmented_change("384-well")

        # Verify plate selector updated
        assert plate_selector.plate_format == "384"
        assert plate_selector.n_rows == 16

        # Cleanup
        plate_selector.destroy()
        settings_panel.destroy()
