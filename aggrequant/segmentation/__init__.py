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
from .postprocessing import (
    remove_border_objects,
    filter_aggregates_by_cells,
    relabel_consecutive,
)

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
    # Post-processing
    "remove_border_objects",
    "filter_aggregates_by_cells",
    "relabel_consecutive",
]
