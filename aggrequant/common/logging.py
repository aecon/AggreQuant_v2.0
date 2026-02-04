"""
Logging utilities for consistent output across the package.

Author: Athena Economides
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Configure logging for the aggrequant package.

    Arguments:
        level: Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        log_file: Optional path to log file
        format_string: Custom format string (default provides timestamp and level)

    Returns:
        Configured root logger
    """
    if format_string is None:
        format_string = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    # Get numeric level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Configure root logger
    logger = logging.getLogger("aggrequant")
    logger.setLevel(numeric_level)

    # Clear existing handlers
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Arguments:
        name: Module name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"aggrequant.{name}")


class SimpleLogger:
    """
    Simple logger that follows the existing codebase style.

    Uses print statements with function name prefix for compatibility
    with the existing verbose/debug pattern.
    """

    def __init__(self, name: str, verbose: bool = False, debug: bool = False):
        """
        Initialize simple logger.

        Arguments:
            name: Logger name (usually function or module name)
            verbose: Enable verbose output
            debug: Enable debug output
        """
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def msg(self, message: str):
        """Print message if verbose is enabled."""
        if self.verbose:
            print(f"({self.name}) {message}")

    def dbg(self, message: str):
        """Print message if debug is enabled."""
        if self.debug:
            print(f"({self.name}) [DEBUG] {message}")

    def err(self, message: str):
        """Print error message (always shown)."""
        print(f"({self.name}) [ERROR] {message}")

    def warn(self, message: str):
        """Print warning message (always shown)."""
        print(f"({self.name}) [WARNING] {message}")
