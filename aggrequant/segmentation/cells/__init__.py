"""
Cell segmentation backends.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .cellpose import CellposeSegmenter

__all__ = ["CellposeSegmenter"]
