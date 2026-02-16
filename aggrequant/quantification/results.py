"""
Results container dataclasses for quantification outputs.

Provides structured containers for field-level and well-level results.

Author: Athena Economides, 2026, UZH
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime
import numpy as np


@dataclass
class FieldResult:
    """
    Quantification results for a single field of view.

    Contains all measurements from one image triplet (nuclei, cells, aggregates).
    """

    # Identifiers
    plate_name: str
    well_id: str  # e.g., "A01"
    row: str  # e.g., "A"
    column: int  # e.g., 1
    field: int  # e.g., 1

    # Control info
    control_type: Optional[str] = None  # "NT", "RAB13", etc.

    # Cell metrics
    n_cells: int = 0
    n_nuclei: int = 0
    total_nuclei_area_px: float = 0.0
    total_cell_area_px: float = 0.0

    # Aggregate metrics (total)
    n_aggregates: int = 0
    n_aggregate_positive_cells: int = 0
    pct_aggregate_positive_cells: float = 0.0
    total_aggregate_area_px: float = 0.0
    pct_aggregate_area_over_cell: float = 0.0
    avg_aggregates_per_positive_cell: float = 0.0
    pct_ambiguous_aggregates: float = 0.0

    # Focus quality metrics
    focus_variance_laplacian_mean: Optional[float] = None
    focus_variance_laplacian_min: Optional[float] = None
    focus_pct_patches_blurry: Optional[float] = None
    focus_pct_area_blurry: Optional[float] = None
    focus_is_likely_blurry: Optional[bool] = None

    # Blur-masked metrics (excluding blurry regions)
    n_cells_masked: Optional[int] = None
    n_aggregate_positive_cells_masked: Optional[int] = None
    pct_aggregate_positive_cells_masked: Optional[float] = None
    total_cell_area_masked_px: Optional[float] = None
    total_aggregate_area_masked_px: Optional[float] = None

    # Metadata
    segmentation_method: str = "unknown"
    model_weights: Optional[str] = None
    blur_threshold_used: Optional[float] = None
    timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        return asdict(self)

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class WellResult:
    """
    Aggregated quantification results for a single well.

    Combines measurements from all fields of view in a well.
    """

    # Identifiers
    plate_name: str
    well_id: str  # e.g., "A01"
    row: str  # e.g., "A"
    column: int  # e.g., 1
    n_fields: int = 0

    # Control info
    control_type: Optional[str] = None

    # Aggregated cell metrics
    total_n_cells: int = 0
    total_n_nuclei: int = 0
    total_cell_area_px: float = 0.0

    # Aggregated aggregate metrics
    total_n_aggregates: int = 0
    total_n_aggregate_positive_cells: int = 0
    pct_aggregate_positive_cells: float = 0.0
    total_aggregate_area_px: float = 0.0
    pct_aggregate_area_over_cell: float = 0.0
    avg_aggregates_per_positive_cell: float = 0.0

    # Focus quality (average across fields)
    avg_focus_variance_laplacian: Optional[float] = None
    n_blurry_fields: int = 0
    pct_blurry_fields: float = 0.0

    # Blur-masked metrics (aggregated)
    total_n_cells_masked: Optional[int] = None
    pct_aggregate_positive_cells_masked: Optional[float] = None

    # Per-field data (for detailed analysis)
    field_results: List[FieldResult] = field(default_factory=list)

    def to_dict(self, include_fields: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame creation."""
        result = {
            "plate_name": self.plate_name,
            "well_id": self.well_id,
            "row": self.row,
            "column": self.column,
            "n_fields": self.n_fields,
            "control_type": self.control_type,
            "total_n_cells": self.total_n_cells,
            "total_n_nuclei": self.total_n_nuclei,
            "total_cell_area_px": self.total_cell_area_px,
            "total_n_aggregates": self.total_n_aggregates,
            "total_n_aggregate_positive_cells": self.total_n_aggregate_positive_cells,
            "pct_aggregate_positive_cells": self.pct_aggregate_positive_cells,
            "total_aggregate_area_px": self.total_aggregate_area_px,
            "pct_aggregate_area_over_cell": self.pct_aggregate_area_over_cell,
            "avg_aggregates_per_positive_cell": self.avg_aggregates_per_positive_cell,
            "avg_focus_variance_laplacian": self.avg_focus_variance_laplacian,
            "n_blurry_fields": self.n_blurry_fields,
            "pct_blurry_fields": self.pct_blurry_fields,
            "total_n_cells_masked": self.total_n_cells_masked,
            "pct_aggregate_positive_cells_masked": self.pct_aggregate_positive_cells_masked,
        }
        if include_fields:
            result["field_results"] = [f.to_dict() for f in self.field_results]
        return result


@dataclass
class PlateResult:
    """
    Complete quantification results for a plate.

    Contains all well results and plate-level statistics.
    """

    # Identifiers
    plate_name: str
    plate_format: str = "96"  # "96" or "384"

    # Well results
    well_results: Dict[str, WellResult] = field(default_factory=dict)

    # Control information
    control_types: List[str] = field(default_factory=list)
    control_wells: Dict[str, List[str]] = field(default_factory=dict)

    # Plate-level statistics
    total_n_wells_processed: int = 0
    total_n_fields_processed: int = 0
    total_n_cells: int = 0
    avg_cells_per_well: float = 0.0

    # SSMD (if controls are defined)
    ssmd: Optional[float] = None
    ssmd_control_pair: Optional[tuple] = None

    # Metadata
    timestamp: Optional[datetime] = None
    processing_time_seconds: Optional[float] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def get_well(self, well_id: str) -> Optional[WellResult]:
        """Get well result by ID."""
        return self.well_results.get(well_id)

    def add_well(self, well_result: WellResult):
        """Add or update a well result."""
        self.well_results[well_result.well_id] = well_result
        self._update_plate_statistics()

    def _update_plate_statistics(self):
        """Update plate-level statistics from well results."""
        self.total_n_wells_processed = len(self.well_results)
        self.total_n_fields_processed = sum(
            w.n_fields for w in self.well_results.values()
        )
        self.total_n_cells = sum(w.total_n_cells for w in self.well_results.values())

        if self.total_n_wells_processed > 0:
            self.avg_cells_per_well = self.total_n_cells / self.total_n_wells_processed

    def get_control_wells(self, control_type: str) -> List[WellResult]:
        """Get all well results for a specific control type."""
        return [
            w for w in self.well_results.values()
            if w.control_type == control_type
        ]

    def to_dataframe(self, level: str = "well"):
        """
        Convert results to pandas DataFrame.

        Arguments:
            level: "well" for well-level data, "field" for field-level data

        Returns:
            pandas DataFrame with results
        """
        import pandas as pd

        if level == "well":
            data = [w.to_dict(include_fields=False) for w in self.well_results.values()]
        elif level == "field":
            data = []
            for well in self.well_results.values():
                for field_result in well.field_results:
                    data.append(field_result.to_dict())
        else:
            raise ValueError(f"Unknown level: {level}. Use 'well' or 'field'.")

        return pd.DataFrame(data)
