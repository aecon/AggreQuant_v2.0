"""Segmentation backends for nuclei, cells, and aggregates."""

from aggrequant.segmentation.base import BaseSegmenter
from aggrequant.segmentation.nuclei import StarDistSegmenter
from aggrequant.segmentation.cells import CellposeSegmenter
from aggrequant.segmentation.aggregates import FilterBasedSegmenter, NeuralNetworkSegmenter
from aggrequant.segmentation.postprocessing import (
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
