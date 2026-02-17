"""Logging utilities for consistent output across the package."""

import logging
import sys
from pathlib import Path
from typing import Optional


# Module-level flag to track if logging has been configured
_logging_configured = False


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
    global _logging_configured

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

    _logging_configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Arguments:
        name: Module name (typically __name__)

    Returns:
        Logger instance

    Note:
        If logging hasn't been configured via setup_logging(),
        a basic configuration will be applied automatically.
    """
    global _logging_configured

    logger = logging.getLogger(f"aggrequant.{name}")

    # Ensure at least basic logging is configured
    if not _logging_configured:
        root_logger = logging.getLogger("aggrequant")
        if not root_logger.handlers:
            # Set up minimal default logging
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
            root_logger.addHandler(handler)
            root_logger.setLevel(logging.INFO)
        _logging_configured = True

    return logger
