"""Well-ID and plate-layout utilities for HCS analysis."""

from typing import Optional, Tuple


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
