"""
Quantification module for AggreQuant.

Computes quantities of interest (QoI) from segmentation results.

Author: Athena Economides, 2026, UZH
"""

from .results import FieldResult, WellResult, PlateResult
from .measurements import (
    compute_field_measurements,
    compute_masked_measurements,
    apply_focus_metrics_to_result,
    compute_aggregate_mask_inside_cells,
)

__all__ = [
    # Results containers
    "FieldResult",
    "WellResult",
    "PlateResult",
    # Measurement functions
    "compute_field_measurements",
    "compute_masked_measurements",
    "apply_focus_metrics_to_result",
    "compute_aggregate_mask_inside_cells",
]
