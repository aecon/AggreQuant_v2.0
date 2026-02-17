"""Quantification of aggregate measurements from segmentation results."""

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
