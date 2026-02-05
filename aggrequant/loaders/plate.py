"""
Plate and well data structures for HCS analysis.

Author: Athena Economides, 2026, UZH
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
import numpy as np


# Standard plate layouts
PLATE_LAYOUTS = {
    "96": {"rows": 8, "cols": 12},
    "384": {"rows": 16, "cols": 24},
}


def well_id_to_indices(well_id: str, plate_format: Optional[str] = None) -> Tuple[int, int]:
    """
    Convert well ID to row/column indices (0-based).

    Validates that the well ID is properly formatted. Optionally checks
    that indices are valid for the specified plate format.

    Arguments:
        well_id: Well identifier (e.g., "A01", "H12")
        plate_format: Optional plate format ("96" or "384") for bounds checking

    Returns:
        Tuple of (row_index, col_index)

    Raises:
        ValueError: If well ID is malformed or out of bounds for plate format
    """
    if len(well_id) < 2:
        raise ValueError(f"Invalid well ID: {well_id}")

    row_letter = well_id[0].upper()
    if not 'A' <= row_letter <= 'P':
        raise ValueError(f"Invalid row letter '{row_letter}' in well ID: {well_id}")

    try:
        col_num = int(well_id[1:])
    except ValueError:
        raise ValueError(f"Invalid column number in well ID: {well_id}")

    if col_num < 1:
        raise ValueError(f"Column number must be >= 1, got {col_num} in well ID: {well_id}")

    row_idx = ord(row_letter) - ord('A')
    col_idx = col_num - 1

    # Optional bounds checking for plate format
    if plate_format is not None:
        if plate_format not in PLATE_LAYOUTS:
            raise ValueError(f"Unknown plate format: {plate_format}")
        max_rows = PLATE_LAYOUTS[plate_format]["rows"]
        max_cols = PLATE_LAYOUTS[plate_format]["cols"]
        if row_idx >= max_rows or col_idx >= max_cols:
            raise ValueError(f"Well {well_id} invalid for {plate_format}-well plate")

    return row_idx, col_idx


def indices_to_well_id(row: int, col: int) -> str:
    """
    Convert row/column indices to well ID.

    Arguments:
        row: Row index (0-based)
        col: Column index (0-based)

    Returns:
        Well ID string (e.g., "A01")
    """
    row_letter = chr(ord('A') + row)
    return f"{row_letter}{col + 1:02d}"


@dataclass
class FieldOfView:
    """
    Represents a single field of view within a well.

    Attributes:
        field_id: Identifier for this field (e.g., "1", "01")
        images: Dictionary mapping channel name to image array
        masks: Dictionary mapping mask type to label array
        metrics: Dictionary of computed metrics
        file_paths: Dictionary mapping channel name to source file path
    """
    field_id: str
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    file_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def channels(self) -> List[str]:
        """List of available channels."""
        return list(self.images.keys())

    @property
    def has_nuclei_mask(self) -> bool:
        """Check if nuclei mask is available."""
        return "nuclei" in self.masks

    @property
    def has_cell_mask(self) -> bool:
        """Check if cell mask is available."""
        return "cells" in self.masks

    @property
    def has_aggregate_mask(self) -> bool:
        """Check if aggregate mask is available."""
        return "aggregates" in self.masks


@dataclass
class Well:
    """
    Represents a single well in a plate.

    Attributes:
        well_id: Well identifier (e.g., "A01")
        fields: Dictionary mapping field ID to FieldOfView
        treatment: Treatment/condition label
        control_type: Type of control if applicable ("negative", "positive", None)
        metadata: Additional well metadata
    """
    well_id: str
    fields: Dict[str, FieldOfView] = field(default_factory=dict)
    treatment: Optional[str] = None
    control_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def row(self) -> int:
        """Row index (0-based)."""
        return well_id_to_indices(self.well_id)[0]

    @property
    def col(self) -> int:
        """Column index (0-based)."""
        return well_id_to_indices(self.well_id)[1]

    @property
    def row_letter(self) -> str:
        """Row letter (A-P)."""
        return self.well_id[0].upper()

    @property
    def col_number(self) -> int:
        """Column number (1-based)."""
        return int(self.well_id[1:])

    @property
    def n_fields(self) -> int:
        """Number of fields of view."""
        return len(self.fields)

    @property
    def is_control(self) -> bool:
        """Check if this is a control well."""
        return self.control_type is not None

    def add_field(self, fov: FieldOfView) -> None:
        """Add a field of view to this well."""
        self.fields[fov.field_id] = fov

    def get_field(self, field_id: str) -> Optional[FieldOfView]:
        """Get a specific field of view."""
        return self.fields.get(field_id)

    def iter_fields(self) -> Iterator[FieldOfView]:
        """Iterate over all fields of view."""
        for fov in self.fields.values():
            yield fov


@dataclass
class Plate:
    """
    Represents a multi-well plate for HCS analysis.

    Attributes:
        plate_id: Plate identifier
        plate_format: Plate format ("96" or "384")
        wells: Dictionary mapping well ID to Well
        metadata: Plate-level metadata
    """
    plate_id: str
    plate_format: str = "96"
    wells: Dict[str, Well] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.plate_format not in PLATE_LAYOUTS:
            raise ValueError(f"Unknown plate format: {self.plate_format}")

    @property
    def n_rows(self) -> int:
        """Number of rows in plate."""
        return PLATE_LAYOUTS[self.plate_format]["rows"]

    @property
    def n_cols(self) -> int:
        """Number of columns in plate."""
        return PLATE_LAYOUTS[self.plate_format]["cols"]

    @property
    def n_wells(self) -> int:
        """Number of wells with data."""
        return len(self.wells)

    @property
    def max_wells(self) -> int:
        """Maximum number of wells for this plate format."""
        return self.n_rows * self.n_cols

    @property
    def well_ids(self) -> List[str]:
        """List of well IDs with data."""
        return sorted(self.wells.keys())

    def get_well(self, well_id: str) -> Optional[Well]:
        """Get a specific well."""
        return self.wells.get(well_id)

    def add_well(self, well: Well) -> None:
        """Add a well to the plate."""
        self.wells[well.well_id] = well

    def get_or_create_well(self, well_id: str) -> Well:
        """Get existing well or create a new one."""
        if well_id not in self.wells:
            self.wells[well_id] = Well(well_id=well_id)
        return self.wells[well_id]

    def iter_wells(self) -> Iterator[Well]:
        """Iterate over all wells in plate order."""
        for well_id in self.well_ids:
            yield self.wells[well_id]

    def get_control_wells(self, control_type: str) -> List[Well]:
        """Get all wells of a specific control type."""
        return [w for w in self.wells.values() if w.control_type == control_type]

    def get_treatment_wells(self, treatment: str) -> List[Well]:
        """Get all wells with a specific treatment."""
        return [w for w in self.wells.values() if w.treatment == treatment]

    def set_control_wells(
        self,
        well_ids: List[str],
        control_type: str
    ) -> None:
        """
        Mark wells as controls.

        Arguments:
            well_ids: List of well IDs to mark
            control_type: Type of control (e.g., "negative", "positive")
        """
        for well_id in well_ids:
            well = self.get_or_create_well(well_id)
            well.control_type = control_type

    def to_grid(self) -> np.ndarray:
        """
        Create a 2D boolean array showing which wells have data.

        Returns:
            2D boolean array (rows x cols)
        """
        grid = np.zeros((self.n_rows, self.n_cols), dtype=bool)
        for well in self.wells.values():
            grid[well.row, well.col] = True
        return grid

    def summary(self) -> str:
        """Return a summary string of the plate."""
        n_fields = sum(w.n_fields for w in self.wells.values())
        n_controls = sum(1 for w in self.wells.values() if w.is_control)

        return (
            f"Plate '{self.plate_id}' ({self.plate_format}-well)\n"
            f"  Wells with data: {self.n_wells}/{self.max_wells}\n"
            f"  Total fields: {n_fields}\n"
            f"  Control wells: {n_controls}"
        )


def create_plate_from_wells(
    plate_id: str,
    well_ids: List[str],
    plate_format: str = "96"
) -> Plate:
    """
    Create a plate with empty wells.

    Arguments:
        plate_id: Plate identifier
        well_ids: List of well IDs to create
        plate_format: Plate format ("96" or "384")

    Returns:
        Plate with empty Well objects
    """
    plate = Plate(plate_id=plate_id, plate_format=plate_format)

    for well_id in well_ids:
        plate.add_well(Well(well_id=well_id))

    return plate


def generate_all_well_ids(plate_format: str = "96") -> List[str]:
    """
    Generate all well IDs for a plate format.

    Arguments:
        plate_format: "96" or "384"

    Returns:
        List of well IDs in row-major order
    """
    layout = PLATE_LAYOUTS[plate_format]
    wells = []

    for row in range(layout["rows"]):
        for col in range(layout["cols"]):
            wells.append(indices_to_well_id(row, col))

    return wells
