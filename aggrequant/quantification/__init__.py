"""
Quantification module for AggreQuant.

Computes quantities of interest (QoI) from segmentation results.

Author: Athena Economides, 2026, UZH
"""

from aggrequant.quantification.results import FieldResult, WellResult, PlateResult
from aggrequant.quantification.measurements import compute_field_measurements

__all__ = [
    # Results containers
    "FieldResult",
    "WellResult",
    "PlateResult",
    # Measurement functions
    "compute_field_measurements",
]
