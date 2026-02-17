"""
Interactive plate grid widget for well selection.

Allows users to select wells and assign control types by clicking
or dragging on a 96/384-well plate visualization.
"""

import customtkinter as ctk
from typing import Dict, List, Optional, Callable, Set, Tuple


# Control type colors
CONTROL_COLORS = {
    "negative": "#3498db",  # Blue
    "NT": "#2ecc71",        # Green (Non-targeting)
    "custom": "#f39c12",    # Orange
}

# Default colors
WELL_EMPTY = "#ecf0f1"      # Light gray
WELL_HOVER = "#bdc3c7"      # Darker gray
WELL_SELECTED = "#f1c40f"   # Yellow


class PlateSelector(ctk.CTkFrame):
    """
    Interactive plate grid widget for selecting wells.

    Attributes:
        plate_format: "96" or "384"
        selected_wells: Set of currently selected well IDs
        control_assignments: Dict mapping well ID to control type
        on_selection_change: Callback when selection changes
    """

    def __init__(
        self,
        master,
        plate_format: str = "96",
        on_selection_change: Optional[Callable[[Set[str]], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)

        self.plate_format = plate_format
        self.on_selection_change = on_selection_change

        # State
        self.selected_wells: Set[str] = set()
        self.control_assignments: Dict[str, str] = {}
        self.well_buttons: Dict[str, ctk.CTkButton] = {}

        # Drag selection state
        self._drag_start: Optional[str] = None
        self._drag_selecting: bool = True  # True = selecting, False = deselecting

        # Grid dimensions
        if plate_format == "96":
            self.n_rows = 8
            self.n_cols = 12
            self.well_size = 40
        else:  # 384
            self.n_rows = 16
            self.n_cols = 24
            self.well_size = 24

        self._create_widgets()

    def _create_widgets(self):
        """Create the plate grid."""
        # Title
        title = ctk.CTkLabel(
            self,
            text=f"{self.plate_format}-Well Plate",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.grid(row=0, column=0, columnspan=self.n_cols + 1, pady=(0, 5))

        # Column headers (1-12 or 1-24)
        for col in range(self.n_cols):
            label = ctk.CTkLabel(
                self,
                text=str(col + 1),
                width=self.well_size,
                font=ctk.CTkFont(size=10)
            )
            label.grid(row=1, column=col + 1)

        # Row headers (A-H or A-P) and wells
        for row in range(self.n_rows):
            row_letter = chr(ord('A') + row)

            # Row label
            label = ctk.CTkLabel(
                self,
                text=row_letter,
                width=20,
                font=ctk.CTkFont(size=10)
            )
            label.grid(row=row + 2, column=0)

            # Well buttons
            for col in range(self.n_cols):
                well_id = f"{row_letter}{col + 1:02d}"

                btn = ctk.CTkButton(
                    self,
                    text="",
                    width=self.well_size,
                    height=self.well_size,
                    corner_radius=0,
                    fg_color=WELL_EMPTY,
                    hover_color=WELL_HOVER,
                    border_width=1,
                    border_color="#95a5a6",
                    command=lambda w=well_id: self._on_well_click(w)
                )
                btn.grid(row=row + 2, column=col + 1, padx=1, pady=1)

                # Bind mouse events for drag selection
                btn.bind("<Button-1>", lambda e, w=well_id: self._on_drag_start(w, e))
                btn.bind("<B1-Motion>", lambda e: self._on_drag_motion(e))
                btn.bind("<ButtonRelease-1>", lambda e: self._on_drag_end(e))
                btn.bind("<Enter>", lambda e, w=well_id: self._on_well_enter(w, e))

                self.well_buttons[well_id] = btn

        # Legend
        self._create_legend()

    def _create_legend(self):
        """Create color legend for control types."""
        legend_frame = ctk.CTkFrame(self)
        legend_frame.grid(row=self.n_rows + 3, column=0, columnspan=self.n_cols + 1, pady=10)

        ctk.CTkLabel(legend_frame, text="Legend:", font=ctk.CTkFont(weight="bold")).pack(side="left", padx=5)

        for control_type, color in CONTROL_COLORS.items():
            frame = ctk.CTkFrame(legend_frame, fg_color="transparent")
            frame.pack(side="left", padx=10)

            indicator = ctk.CTkButton(
                frame,
                text="",
                width=16,
                height=16,
                corner_radius=0,
                fg_color=color,
                hover=False,
                state="disabled"
            )
            indicator.pack(side="left", padx=2)

            ctk.CTkLabel(frame, text=control_type, font=ctk.CTkFont(size=11)).pack(side="left")

    def _on_well_click(self, well_id: str):
        """Handle single well click."""
        if well_id in self.selected_wells:
            self.selected_wells.remove(well_id)
        else:
            self.selected_wells.add(well_id)

        self._update_well_appearance(well_id)
        self._notify_selection_change()

    def _on_drag_start(self, well_id: str, event):
        """Start drag selection."""
        self._drag_start = well_id
        # If clicking on selected well, we're deselecting
        self._drag_selecting = well_id not in self.selected_wells

    def _on_drag_motion(self, event):
        """Continue drag selection."""
        pass  # Handled by _on_well_enter

    def _on_drag_end(self, event):
        """End drag selection."""
        self._drag_start = None

    def _on_well_enter(self, well_id: str, event):
        """Handle mouse entering a well during drag."""
        if self._drag_start is not None:
            if self._drag_selecting:
                self.selected_wells.add(well_id)
            else:
                self.selected_wells.discard(well_id)
            self._update_well_appearance(well_id)
            self._notify_selection_change()

    def _update_well_appearance(self, well_id: str):
        """Update the visual appearance of a well."""
        btn = self.well_buttons.get(well_id)
        if btn is None:
            return

        # Check if well has a control assignment
        if well_id in self.control_assignments:
            control_type = self.control_assignments[well_id]
            color = CONTROL_COLORS.get(control_type, CONTROL_COLORS["custom"])
            btn.configure(fg_color=color)
        elif well_id in self.selected_wells:
            btn.configure(fg_color=WELL_SELECTED)
        else:
            btn.configure(fg_color=WELL_EMPTY)

    def _update_all_wells(self):
        """Update appearance of all wells."""
        for well_id in self.well_buttons:
            self._update_well_appearance(well_id)

    def _notify_selection_change(self):
        """Notify callback of selection change."""
        if self.on_selection_change:
            self.on_selection_change(self.selected_wells.copy())

    # Public API

    def get_selected_wells(self) -> Set[str]:
        """Get currently selected wells."""
        return self.selected_wells.copy()

    def set_selected_wells(self, wells: Set[str]):
        """Set selected wells."""
        self.selected_wells = set(wells)
        self._update_all_wells()
        self._notify_selection_change()

    def clear_selection(self):
        """Clear all selections."""
        self.selected_wells.clear()
        self._update_all_wells()
        self._notify_selection_change()

    def assign_control_type(self, control_type: str):
        """Assign control type to currently selected wells."""
        for well_id in self.selected_wells:
            self.control_assignments[well_id] = control_type
        self.selected_wells.clear()
        self._update_all_wells()
        self._notify_selection_change()

    def remove_control_assignment(self, well_id: str):
        """Remove control assignment from a well."""
        if well_id in self.control_assignments:
            del self.control_assignments[well_id]
            self._update_well_appearance(well_id)

    def clear_control_assignments(self):
        """Clear all control assignments."""
        self.control_assignments.clear()
        self._update_all_wells()

    def get_control_assignments(self) -> Dict[str, str]:
        """Get all control assignments."""
        return self.control_assignments.copy()

    def set_control_assignments(self, assignments: Dict[str, str]):
        """Set control assignments."""
        self.control_assignments = dict(assignments)
        self._update_all_wells()

    def get_wells_by_control_type(self, control_type: str) -> List[str]:
        """Get all wells assigned to a specific control type."""
        return [w for w, t in self.control_assignments.items() if t == control_type]

    def set_plate_format(self, plate_format: str):
        """Change plate format (recreates the grid)."""
        if plate_format != self.plate_format:
            self.plate_format = plate_format
            self.selected_wells.clear()
            self.control_assignments.clear()
            self.well_buttons.clear()

            # Clear existing widgets
            for widget in self.winfo_children():
                widget.destroy()

            # Update dimensions
            if plate_format == "96":
                self.n_rows = 8
                self.n_cols = 12
                self.well_size = 40
            else:
                self.n_rows = 16
                self.n_cols = 24
                self.well_size = 24

            self._create_widgets()
