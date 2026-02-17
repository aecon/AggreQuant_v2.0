"""GUI widgets for AggreQuant."""

from .plate_selector import PlateSelector, CONTROL_COLORS
from .control_panel import ControlPanel
from .settings_panel import SettingsPanel
from .progress_panel import ProgressPanel

__all__ = [
    "PlateSelector",
    "ControlPanel",
    "SettingsPanel",
    "ProgressPanel",
    "CONTROL_COLORS",
]
