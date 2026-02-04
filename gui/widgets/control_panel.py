"""
Control panel widget for assigning control types to wells.

Provides buttons for common control types and allows custom types.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

import customtkinter as ctk
from typing import Optional, Callable, List

from .plate_selector import CONTROL_COLORS


class ControlPanel(ctk.CTkFrame):
    """
    Panel for assigning control types to selected wells.

    Provides predefined control type buttons (negative, positive, NT, RAB13)
    and allows adding custom control types.
    """

    def __init__(
        self,
        master,
        on_assign: Optional[Callable[[str], None]] = None,
        on_clear: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)

        self.on_assign = on_assign
        self.on_clear = on_clear
        self.custom_types: List[str] = []

        self._create_widgets()

    def _create_widgets(self):
        """Create the control panel widgets."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Assign Control Type",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=(10, 15))

        # Instructions
        instructions = ctk.CTkLabel(
            self,
            text="1. Select wells on the plate\n2. Click a control type below",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        instructions.pack(pady=(0, 10))

        # Predefined control type buttons
        predefined_frame = ctk.CTkFrame(self, fg_color="transparent")
        predefined_frame.pack(fill="x", padx=10)

        ctk.CTkLabel(
            predefined_frame,
            text="Predefined Controls:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))

        # Create buttons for predefined types
        for control_type in ["negative", "positive", "NT", "RAB13"]:
            color = CONTROL_COLORS.get(control_type, "#95a5a6")
            btn = ctk.CTkButton(
                predefined_frame,
                text=control_type.capitalize(),
                fg_color=color,
                hover_color=self._darken_color(color),
                width=120,
                command=lambda t=control_type: self._on_assign_click(t)
            )
            btn.pack(pady=3, fill="x")

        # Separator
        separator = ctk.CTkFrame(self, height=2, fg_color="gray")
        separator.pack(fill="x", padx=10, pady=15)

        # Custom control type
        custom_frame = ctk.CTkFrame(self, fg_color="transparent")
        custom_frame.pack(fill="x", padx=10)

        ctk.CTkLabel(
            custom_frame,
            text="Custom Control:",
            font=ctk.CTkFont(size=12)
        ).pack(anchor="w", pady=(0, 5))

        input_frame = ctk.CTkFrame(custom_frame, fg_color="transparent")
        input_frame.pack(fill="x")

        self.custom_entry = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter name...",
            width=100
        )
        self.custom_entry.pack(side="left", padx=(0, 5))

        add_btn = ctk.CTkButton(
            input_frame,
            text="Add",
            width=50,
            command=self._on_add_custom
        )
        add_btn.pack(side="left")

        # Frame for custom type buttons
        self.custom_buttons_frame = ctk.CTkFrame(custom_frame, fg_color="transparent")
        self.custom_buttons_frame.pack(fill="x", pady=(5, 0))

        # Separator
        separator2 = ctk.CTkFrame(self, height=2, fg_color="gray")
        separator2.pack(fill="x", padx=10, pady=15)

        # Clear buttons
        clear_frame = ctk.CTkFrame(self, fg_color="transparent")
        clear_frame.pack(fill="x", padx=10, pady=(0, 10))

        clear_selection_btn = ctk.CTkButton(
            clear_frame,
            text="Clear Selection",
            fg_color="#7f8c8d",
            hover_color="#5d6d7e",
            command=self._on_clear_selection
        )
        clear_selection_btn.pack(fill="x", pady=2)

        clear_all_btn = ctk.CTkButton(
            clear_frame,
            text="Clear All Assignments",
            fg_color="#c0392b",
            hover_color="#922b21",
            command=self._on_clear_all
        )
        clear_all_btn.pack(fill="x", pady=2)

    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color for hover effect."""
        # Remove # and convert to RGB
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

        # Darken by 20%
        factor = 0.8
        r = int(r * factor)
        g = int(g * factor)
        b = int(b * factor)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _on_assign_click(self, control_type: str):
        """Handle control type assignment button click."""
        if self.on_assign:
            self.on_assign(control_type)

    def _on_add_custom(self):
        """Add a custom control type."""
        custom_name = self.custom_entry.get().strip()
        if custom_name and custom_name not in self.custom_types:
            self.custom_types.append(custom_name)

            # Add button for custom type
            color = CONTROL_COLORS.get("custom", "#f39c12")
            btn = ctk.CTkButton(
                self.custom_buttons_frame,
                text=custom_name,
                fg_color=color,
                hover_color=self._darken_color(color),
                width=120,
                command=lambda t=custom_name: self._on_assign_click(t)
            )
            btn.pack(pady=2, fill="x")

            # Clear entry
            self.custom_entry.delete(0, "end")

    def _on_clear_selection(self):
        """Clear well selection."""
        if self.on_clear:
            self.on_clear()

    def _on_clear_all(self):
        """Clear all control assignments."""
        # This should be connected to plate selector's clear_control_assignments
        if self.on_clear:
            self.on_clear()

    def set_on_assign(self, callback: Callable[[str], None]):
        """Set the assignment callback."""
        self.on_assign = callback

    def set_on_clear(self, callback: Callable[[], None]):
        """Set the clear callback."""
        self.on_clear = callback
