"""
Aggregate segmentation backends.

Author: Athena Economides, 2026, UZH
"""

from .filter_based import FilterBasedSegmenter
from .neural_network import NeuralNetworkSegmenter

__all__ = ["FilterBasedSegmenter", "NeuralNetworkSegmenter"]
