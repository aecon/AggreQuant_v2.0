"""Base classes and protocols for segmentation backends."""

from abc import ABC, abstractmethod
import numpy as np

from aggrequant.common.logging import get_logger


class BaseSegmenter(ABC):
    """
    Abstract base class for segmenters.

    Provides common functionality and enforces the interface.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize segmenter.

        Arguments:
            verbose: Print progress messages
        """
        self.verbose = verbose

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for logging."""
        pass

    @abstractmethod
    def segment(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Segment an image and return labeled mask.

        Arguments:
            image: Input image (2D grayscale)
            **kwargs: Additional arguments for subclasses

        Returns:
            labels: Instance segmentation labels
        """
        pass

    def _log(self, message: str):
        """Log message if verbose is enabled."""
        if self.verbose:
            logger = get_logger(f"segmentation.{self.name}")
            logger.info(message)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
