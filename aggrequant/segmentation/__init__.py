"""
Segmentation backends for AggreQuant.

Provides segmenters for:
- Nuclei (StarDist)
- Cells (Cellpose, Distance-Intensity watershed)
- Aggregates (Filter-based, Neural Network)

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .base import Segmenter, BaseSegmenter
from .nuclei import StarDistSegmenter
from .cells import CellposeSegmenter, DistanceIntensitySegmenter
from .aggregates import FilterBasedSegmenter, NeuralNetworkSegmenter

__all__ = [
    # Base classes
    "Segmenter",
    "BaseSegmenter",
    # Nuclei
    "StarDistSegmenter",
    # Cells
    "CellposeSegmenter",
    "DistanceIntensitySegmenter",
    # Aggregates
    "FilterBasedSegmenter",
    "NeuralNetworkSegmenter",
]
