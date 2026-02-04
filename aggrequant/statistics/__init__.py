"""
Statistics module for AggreQuant.

Aggregates field-level results to well-level statistics,
computes quality control metrics (SSMD), and exports results.

Original author: Athena Economides
Refactoring tool: Claude Opus 4.5
Date: 2026-02-04
"""

from .well_stats import (
    aggregate_field_to_well,
    compute_well_statistics_from_qoi_arrays,
    compute_area_percentage_from_arrays,
)
from .controls import (
    compute_ssmd,
    compute_z_factor,
    compare_controls,
    compute_plate_ssmd,
    get_control_statistics,
    ControlComparison,
    format_ssmd_interpretation,
    format_z_factor_interpretation,
)
from .export import (
    export_to_csv,
    export_to_parquet,
    export_to_excel,
    export_plate_summary,
    export_density_map_data,
    create_qoi_table,
)

__all__ = [
    # Well statistics
    "aggregate_field_to_well",
    "compute_well_statistics_from_qoi_arrays",
    "compute_area_percentage_from_arrays",
    # Control comparison
    "compute_ssmd",
    "compute_z_factor",
    "compare_controls",
    "compute_plate_ssmd",
    "get_control_statistics",
    "ControlComparison",
    "format_ssmd_interpretation",
    "format_z_factor_interpretation",
    # Export
    "export_to_csv",
    "export_to_parquet",
    "export_to_excel",
    "export_plate_summary",
    "export_density_map_data",
    "create_qoi_table",
]
