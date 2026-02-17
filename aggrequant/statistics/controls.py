"""
Control well analysis and SSMD calculation.

Computes Strictly Standardized Mean Difference (SSMD) and other
quality metrics comparing control wells.

Author: Athena Economides, 2026, UZH
"""

import math
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from aggrequant.quantification.results import WellResult, PlateResult


@dataclass
class ControlComparison:
    """Results of comparing two control types."""

    control_type_1: str
    control_type_2: str

    # Basic statistics for control type 1
    n_wells_1: int
    mean_1: float
    std_1: float
    median_1: float

    # Basic statistics for control type 2
    n_wells_2: int
    mean_2: float
    std_2: float
    median_2: float

    # Comparison metrics
    ssmd: float
    z_factor: float
    fold_change: float

    # Raw values for detailed analysis
    values_1: List[float]
    values_2: List[float]


def compute_ssmd(
    values_1: np.ndarray,
    values_2: np.ndarray,
) -> float:
    """
    Compute Strictly Standardized Mean Difference (SSMD).

    SSMD = (μ1 - μ2) / sqrt(σ1² + σ2²)

    SSMD interpretation:
    - |SSMD| >= 3: Excellent
    - 2 <= |SSMD| < 3: Good
    - 1 <= |SSMD| < 2: Moderate
    - |SSMD| < 1: Poor

    Arguments:
        values_1: Values from first group (e.g., negative control)
        values_2: Values from second group (e.g., positive control)

    Returns:
        SSMD value
    """
    mean_1 = np.nanmean(values_1)
    mean_2 = np.nanmean(values_2)
    std_1 = np.nanstd(values_1)
    std_2 = np.nanstd(values_2)

    denominator = math.sqrt(std_1 ** 2 + std_2 ** 2)

    if denominator < 1e-10:
        return 0.0

    return (mean_1 - mean_2) / denominator


def compute_z_factor(
    values_1: np.ndarray,
    values_2: np.ndarray,
) -> float:
    """
    Compute Z-factor (Z').

    Z' = 1 - (3 * (σ1 + σ2)) / |μ1 - μ2|

    Z-factor interpretation:
    - Z' >= 0.5: Excellent assay
    - 0 < Z' < 0.5: Marginal assay
    - Z' <= 0: Poor assay

    Arguments:
        values_1: Values from first group
        values_2: Values from second group

    Returns:
        Z-factor value
    """
    mean_1 = np.nanmean(values_1)
    mean_2 = np.nanmean(values_2)
    std_1 = np.nanstd(values_1)
    std_2 = np.nanstd(values_2)

    denominator = abs(mean_1 - mean_2)

    if denominator < 1e-10:
        return -999.0  # Undefined

    return 1 - (3 * (std_1 + std_2)) / denominator


def compare_controls(
    plate_result: PlateResult,
    control_type_1: str,
    control_type_2: str,
    metric: str = "pct_aggregate_positive_cells",
) -> ControlComparison:
    """
    Compare two control types and compute quality metrics.

    Arguments:
        plate_result: PlateResult with well data
        control_type_1: First control type (e.g., "negative", "NT")
        control_type_2: Second control type (e.g., "positive", "RAB13")
        metric: Which metric to compare

    Returns:
        ControlComparison with statistics and quality metrics
    """
    # Get wells for each control type
    wells_1 = plate_result.get_control_wells(control_type_1)
    wells_2 = plate_result.get_control_wells(control_type_2)

    if not wells_1 or not wells_2:
        raise ValueError(
            f"No wells found for controls: {control_type_1} ({len(wells_1)}) "
            f"or {control_type_2} ({len(wells_2)})"
        )

    # Extract metric values
    values_1 = np.array([getattr(w, metric) for w in wells_1])
    values_2 = np.array([getattr(w, metric) for w in wells_2])

    # Remove NaN values
    values_1 = values_1[~np.isnan(values_1)]
    values_2 = values_2[~np.isnan(values_2)]

    # Compute statistics
    mean_1 = float(np.nanmean(values_1))
    mean_2 = float(np.nanmean(values_2))
    std_1 = float(np.nanstd(values_1))
    std_2 = float(np.nanstd(values_2))
    median_1 = float(np.nanmedian(values_1))
    median_2 = float(np.nanmedian(values_2))

    # Compute quality metrics
    ssmd = compute_ssmd(values_1, values_2)
    z_factor = compute_z_factor(values_1, values_2)
    fold_change = (mean_2 / mean_1) if mean_1 > 0 else 0.0

    return ControlComparison(
        control_type_1=control_type_1,
        control_type_2=control_type_2,
        n_wells_1=len(values_1),
        mean_1=mean_1,
        std_1=std_1,
        median_1=median_1,
        n_wells_2=len(values_2),
        mean_2=mean_2,
        std_2=std_2,
        median_2=median_2,
        ssmd=ssmd,
        z_factor=z_factor,
        fold_change=fold_change,
        values_1=values_1.tolist(),
        values_2=values_2.tolist(),
    )


def compute_plate_ssmd(
    plate_result: PlateResult,
    negative_control: str = "NT",
    positive_control: str = "RAB13",
) -> Optional[float]:
    """
    Compute SSMD for a plate using defined control types.

    Arguments:
        plate_result: PlateResult with well data
        negative_control: Name of negative control type
        positive_control: Name of positive control type

    Returns:
        SSMD value, or None if controls not found
    """
    try:
        comparison = compare_controls(
            plate_result,
            negative_control,
            positive_control,
            metric="pct_aggregate_positive_cells",
        )
        return comparison.ssmd
    except ValueError:
        return None


def get_control_statistics(
    well_results: List[WellResult],
    metric: str = "pct_aggregate_positive_cells",
) -> Dict[str, Dict[str, float]]:
    """
    Compute statistics for each control type.

    Arguments:
        well_results: List of WellResult objects
        metric: Metric to compute statistics for

    Returns:
        Dictionary mapping control_type to statistics dict
    """
    # Group by control type
    by_control: Dict[str, List[float]] = {}

    for well in well_results:
        if well.control_type:
            if well.control_type not in by_control:
                by_control[well.control_type] = []
            value = getattr(well, metric)
            if value is not None and not np.isnan(value):
                by_control[well.control_type].append(value)

    # Compute statistics
    result = {}
    for control_type, values in by_control.items():
        if values:
            arr = np.array(values)
            result[control_type] = {
                "n": len(values),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    return result


def format_ssmd_interpretation(ssmd: float) -> str:
    """
    Return interpretation of SSMD value.

    Arguments:
        ssmd: SSMD value

    Returns:
        Human-readable interpretation
    """
    abs_ssmd = abs(ssmd)

    if abs_ssmd >= 3:
        return f"Excellent (|SSMD| = {abs_ssmd:.2f} >= 3)"
    elif abs_ssmd >= 2:
        return f"Good (2 <= |SSMD| = {abs_ssmd:.2f} < 3)"
    elif abs_ssmd >= 1:
        return f"Moderate (1 <= |SSMD| = {abs_ssmd:.2f} < 2)"
    else:
        return f"Poor (|SSMD| = {abs_ssmd:.2f} < 1)"


def format_z_factor_interpretation(z_factor: float) -> str:
    """
    Return interpretation of Z-factor value.

    Arguments:
        z_factor: Z-factor value

    Returns:
        Human-readable interpretation
    """
    if z_factor >= 0.5:
        return f"Excellent assay (Z' = {z_factor:.2f} >= 0.5)"
    elif z_factor > 0:
        return f"Marginal assay (0 < Z' = {z_factor:.2f} < 0.5)"
    else:
        return f"Poor assay (Z' = {z_factor:.2f} <= 0)"
