"""
Nuclei segmentation backends.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .stardist import StarDistSegmenter

__all__ = ["StarDistSegmenter"]
