"""
Aggregate segmentation backends.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .filter_based import FilterBasedSegmenter
from .neural_network import NeuralNetworkSegmenter

__all__ = ["FilterBasedSegmenter", "NeuralNetworkSegmenter"]
