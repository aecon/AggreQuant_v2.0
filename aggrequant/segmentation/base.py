"""
Base classes and protocols for segmentation backends.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from abc import ABC, abstractmethod
import numpy as np

from aggrequant.common.logging import get_logger


class BaseSegmenter(ABC):
    """
    Abstract base class for segmenters.

    Provides common functionality and enforces the interface.
    """

    def __init__(self, verbose: bool = False, debug: bool = False):
        """
        Initialize segmenter.

        Arguments:
            verbose: Print progress messages
            debug: Print detailed debug information
        """
        self.verbose = verbose
        self.debug = debug

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        pass

    @abstractmethod
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment an image and return labeled mask.

        Arguments:
            image: Input image (2D grayscale)

        Returns:
            labels: Instance segmentation labels
        """
        pass

    def _log(self, message: str):
        """Log message if verbose is enabled."""
        if self.verbose:
            logger = get_logger(f"segmentation.{self.name}")
            logger.info(message)

    def _debug(self, message: str):
        """Log debug message if debug is enabled."""
        if self.debug:
            logger = get_logger(f"segmentation.{self.name}")
            logger.debug(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
