"""
Segmentation backends for AggreQuant.

Provides segmenters for:
- Nuclei (StarDist)
- Cells (Cellpose)
- Aggregates (Filter-based, Neural Network)

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
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
