"""
Progress panel widget for displaying analysis progress.

Shows progress bar, status messages, and log output.

Author: Athena Economides, 2026, UZH
"""

import customtkinter as ctk
from datetime import datetime
from typing import Optional, Callable


class ProgressPanel(ctk.CTkFrame):
    """
    Progress panel for displaying analysis progress.

    Shows:
    - Progress bar with percentage
    - Current status message
    - Scrollable log output
    - Run/Cancel buttons
    """

    def __init__(
        self,
        master,
        on_run: Optional[Callable[[], None]] = None,
        on_cancel: Optional[Callable[[], None]] = None,
        **kwargs
    ):
        super().__init__(master, **kwargs)

        self.on_run = on_run
        self.on_cancel = on_cancel
        self._is_running = False

        self._create_widgets()

    def _create_widgets(self):
        """Create the progress panel widgets."""
        # Title
        title = ctk.CTkLabel(
            self,
            text="Analysis Progress",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title.pack(pady=(10, 15))

        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready to run",
            font=ctk.CTkFont(size=12)
        )
        self.status_label.pack(pady=(0, 5))

        # Progress bar
        progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        progress_frame.pack(fill="x", padx=20, pady=5)

        self.progress_bar = ctk.CTkProgressBar(progress_frame, width=250)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.progress_bar.set(0)

        self.progress_label = ctk.CTkLabel(progress_frame, text="0%", width=40)
        self.progress_label.pack(side="left")

        # Current task
        self.task_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.task_label.pack(pady=5)

        # Run/Cancel buttons
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(pady=15)

        self.run_button = ctk.CTkButton(
            button_frame,
            text="▶ Run Analysis",
            fg_color="#27ae60",
            hover_color="#1e8449",
            width=120,
            corner_radius=0,
            command=self._on_run_click
        )
        self.run_button.pack(side="left", padx=5)

        self.cancel_button = ctk.CTkButton(
            button_frame,
            text="⏹ Cancel",
            fg_color="#e74c3c",
            hover_color="#c0392b",
            width=100,
            corner_radius=0,
            state="disabled",
            command=self._on_cancel_click
        )
        self.cancel_button.pack(side="left", padx=5)

        # Log output
        log_label = ctk.CTkLabel(
            self,
            text="Log Output",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        log_label.pack(anchor="w", padx=10, pady=(10, 5))

        self.log_textbox = ctk.CTkTextbox(
            self,
            height=200,
            font=ctk.CTkFont(family="Courier", size=11),
            state="disabled"
        )
        self.log_textbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        # Clear log button
        clear_btn = ctk.CTkButton(
            self,
            text="Clear Log",
            fg_color="#7f8c8d",
            hover_color="#5d6d7e",
            width=80,
            height=25,
            corner_radius=0,
            command=self.clear_log
        )
        clear_btn.pack(pady=(0, 10))

    def _on_run_click(self):
        """Handle run button click."""
        if self.on_run:
            self.on_run()

    def _on_cancel_click(self):
        """Handle cancel button click."""
        if self.on_cancel:
            self.on_cancel()

    # Public API

    def set_running(self, running: bool):
        """Set running state (enables/disables buttons)."""
        self._is_running = running
        if running:
            self.run_button.configure(state="disabled")
            self.cancel_button.configure(state="normal")
            self.status_label.configure(text="Running...")
        else:
            self.run_button.configure(state="normal")
            self.cancel_button.configure(state="disabled")

    def set_progress(self, value: float, message: str = ""):
        """
        Set progress bar value and optional message.

        Arguments:
            value: Progress value 0.0 to 1.0
            message: Optional status message
        """
        self.progress_bar.set(value)
        self.progress_label.configure(text=f"{int(value * 100)}%")

        if message:
            self.task_label.configure(text=message)

    def set_status(self, status: str):
        """Set the status message."""
        self.status_label.configure(text=status)

    def set_task(self, task: str):
        """Set the current task description."""
        self.task_label.configure(text=task)

    def log(self, message: str, level: str = "INFO"):
        """
        Add a message to the log.

        Arguments:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted = f"[{timestamp}] [{level}] {message}\n"

        self.log_textbox.configure(state="normal")
        self.log_textbox.insert("end", formatted)
        self.log_textbox.see("end")  # Scroll to end
        self.log_textbox.configure(state="disabled")

    def log_info(self, message: str):
        """Log an info message."""
        self.log(message, "INFO")

    def log_warning(self, message: str):
        """Log a warning message."""
        self.log(message, "WARN")

    def log_error(self, message: str):
        """Log an error message."""
        self.log(message, "ERROR")

    def clear_log(self):
        """Clear the log output."""
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.configure(state="disabled")

    def reset(self):
        """Reset progress panel to initial state."""
        self.set_progress(0)
        self.set_status("Ready to run")
        self.set_task("")
        self.set_running(False)

    def complete(self, success: bool = True):
        """
        Mark analysis as complete.

        Arguments:
            success: Whether analysis completed successfully
        """
        self.set_running(False)
        if success:
            self.set_progress(1.0)
            self.set_status("Analysis complete!")
            self.log_info("Analysis completed successfully.")
        else:
            self.set_status("Analysis failed or cancelled")
            self.log_error("Analysis did not complete.")

    def set_on_run(self, callback: Callable[[], None]):
        """Set the run callback."""
        self.on_run = callback

    def set_on_cancel(self, callback: Callable[[], None]):
        """Set the cancel callback."""
        self.on_cancel = callback
