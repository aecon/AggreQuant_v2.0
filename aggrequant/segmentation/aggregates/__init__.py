"""Aggregate segmentation backends."""

from aggrequant.segmentation.aggregates.filter_based import FilterBasedSegmenter
from aggrequant.segmentation.aggregates.neural_network import NeuralNetworkSegmenter

__all__ = ["FilterBasedSegmenter", "NeuralNetworkSegmenter"]
