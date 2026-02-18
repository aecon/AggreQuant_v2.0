"""Unit tests for aggrequant.loaders.plate module."""

import pytest
from aggrequant.loaders.plate import (
    PLATE_LAYOUTS,
    well_id_to_indices,
    indices_to_well_id,
)


class TestWellIdToIndices:
    """Tests for well_id_to_indices function."""

    def test_valid_well_ids(self):
        """Valid well IDs should convert correctly."""
        assert well_id_to_indices("A01") == (0, 0)
        assert well_id_to_indices("A12") == (0, 11)
        assert well_id_to_indices("H01") == (7, 0)
        assert well_id_to_indices("H12") == (7, 11)
        assert well_id_to_indices("P24") == (15, 23)

    def test_lowercase_well_id(self):
        """Lowercase row letters should be accepted."""
        assert well_id_to_indices("a01") == (0, 0)
        assert well_id_to_indices("h12") == (7, 11)

    def test_single_digit_column(self):
        """Single digit columns (without leading zero) should work."""
        assert well_id_to_indices("A1") == (0, 0)
        assert well_id_to_indices("B5") == (1, 4)

    def test_invalid_row_letter(self):
        """Invalid row letters should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid row letter"):
            well_id_to_indices("Z01")
        with pytest.raises(ValueError, match="Invalid row letter"):
            well_id_to_indices("Q01")
        with pytest.raises(ValueError, match="Invalid row letter"):
            well_id_to_indices("101")

    def test_invalid_column_number(self):
        """Invalid column numbers should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid column number"):
            well_id_to_indices("A0x")
        with pytest.raises(ValueError, match="Invalid column number"):
            well_id_to_indices("Axx")

    def test_zero_column(self):
        """Column 0 should raise ValueError."""
        with pytest.raises(ValueError, match="Column number must be >= 1"):
            well_id_to_indices("A00")
        with pytest.raises(ValueError, match="Column number must be >= 1"):
            well_id_to_indices("A0")

    def test_negative_column(self):
        """Negative columns should raise ValueError."""
        with pytest.raises(ValueError, match="Column number must be >= 1"):
            well_id_to_indices("A-1")

    def test_too_short_well_id(self):
        """Well IDs that are too short should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid well ID"):
            well_id_to_indices("A")
        with pytest.raises(ValueError, match="Invalid well ID"):
            well_id_to_indices("")

    def test_plate_format_validation_96(self):
        """Bounds should be checked for 96-well plate."""
        # Valid for 96-well
        assert well_id_to_indices("H12", plate_format="96") == (7, 11)

        # Invalid for 96-well (row out of bounds)
        with pytest.raises(ValueError, match="invalid for 96-well plate"):
            well_id_to_indices("I01", plate_format="96")

        # Invalid for 96-well (column out of bounds)
        with pytest.raises(ValueError, match="invalid for 96-well plate"):
            well_id_to_indices("A13", plate_format="96")

    def test_plate_format_validation_384(self):
        """Bounds should be checked for 384-well plate."""
        # Valid for 384-well
        assert well_id_to_indices("P24", plate_format="384") == (15, 23)

        # Invalid for 384-well (row out of bounds - no Q row)
        with pytest.raises(ValueError, match="Invalid row letter"):
            well_id_to_indices("Q01", plate_format="384")

        # Invalid for 384-well (column out of bounds)
        with pytest.raises(ValueError, match="invalid for 384-well plate"):
            well_id_to_indices("A25", plate_format="384")


class TestIndicesToWellId:
    """Tests for indices_to_well_id function."""

    def test_valid_indices(self):
        """Valid indices should convert correctly."""
        assert indices_to_well_id(0, 0) == "A01"
        assert indices_to_well_id(0, 11) == "A12"
        assert indices_to_well_id(7, 0) == "H01"
        assert indices_to_well_id(7, 11) == "H12"
        assert indices_to_well_id(15, 23) == "P24"

    def test_round_trip_conversion(self):
        """Converting to indices and back should give original well ID."""
        for well_id in ["A01", "B05", "H12", "P24"]:
            row, col = well_id_to_indices(well_id)
            assert indices_to_well_id(row, col) == well_id


