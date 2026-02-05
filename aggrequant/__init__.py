"""
AggreQuant - Automated aggregate quantification for High Content Screening.

This package provides tools for:
1. Image analysis pipeline for CRISPR screen data
2. Neural network development for aggregate segmentation
"""

__version__ = "2.0.0"
__author__ = "Athena Economides"

from .pipeline import (
    AggreQuantPipeline,
    PipelineState,
    run_pipeline_from_config,
)

__all__ = [
    "__version__",
    "__author__",
    "AggreQuantPipeline",
    "PipelineState",
    "run_pipeline_from_config",
]
