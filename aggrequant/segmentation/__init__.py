"""
Segmentation backends for AggreQuant.

Provides segmenters for:
- Nuclei (StarDist)
- Cells (Cellpose)
- Aggregates (Filter-based, Neural Network)

Author: Athena Economides, 2026, UZH
"""

from .base import BaseSegmenter
from .nuclei import StarDistSegmenter
from .cells import CellposeSegmenter
from .aggregates import FilterBasedSegmenter, NeuralNetworkSegmenter

__all__ = [
    # Base class
    "BaseSegmenter",
    # Nuclei
    "StarDistSegmenter",
    # Cells
    "CellposeSegmenter",
    # Aggregates
    "FilterBasedSegmenter",
    "NeuralNetworkSegmenter",
]
