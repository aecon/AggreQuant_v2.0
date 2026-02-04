"""
Unit tests for aggrequant.loaders.plate module.

Author: Athena Economides
"""

import numpy as np
import pytest
from aggrequant.loaders.plate import (
    PLATE_LAYOUTS,
    well_id_to_indices,
    indices_to_well_id,
    FieldOfView,
    Well,
    Plate,
    create_plate_from_wells,
    generate_all_well_ids,
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


class TestFieldOfView:
    """Tests for FieldOfView dataclass."""

    def test_creation(self):
        """Should create FieldOfView with required fields."""
        fov = FieldOfView(field_id="1")
        assert fov.field_id == "1"
        assert fov.images == {}
        assert fov.masks == {}

    def test_channels_property(self):
        """channels property should return list of image keys."""
        fov = FieldOfView(
            field_id="1",
            images={"DAPI": np.zeros((10, 10)), "GFP": np.zeros((10, 10))}
        )
        assert set(fov.channels) == {"DAPI", "GFP"}

    def test_mask_properties(self):
        """Mask detection properties should work correctly."""
        fov = FieldOfView(field_id="1")
        assert not fov.has_nuclei_mask
        assert not fov.has_cell_mask
        assert not fov.has_aggregate_mask

        fov.masks["nuclei"] = np.zeros((10, 10))
        assert fov.has_nuclei_mask
        assert not fov.has_cell_mask

        fov.masks["cells"] = np.zeros((10, 10))
        fov.masks["aggregates"] = np.zeros((10, 10))
        assert fov.has_cell_mask
        assert fov.has_aggregate_mask


class TestWell:
    """Tests for Well dataclass."""

    def test_creation(self):
        """Should create Well with required fields."""
        well = Well(well_id="A01")
        assert well.well_id == "A01"
        assert well.fields == {}
        assert well.treatment is None
        assert well.control_type is None

    def test_row_col_properties(self):
        """Row and column properties should be correct."""
        well = Well(well_id="B05")
        assert well.row == 1
        assert well.col == 4
        assert well.row_letter == "B"
        assert well.col_number == 5

    def test_add_field(self):
        """add_field should add FieldOfView to well."""
        well = Well(well_id="A01")
        fov = FieldOfView(field_id="1")
        well.add_field(fov)
        assert well.n_fields == 1
        assert well.get_field("1") is fov

    def test_is_control(self):
        """is_control should reflect control_type."""
        well = Well(well_id="A01")
        assert not well.is_control

        well.control_type = "negative"
        assert well.is_control

    def test_iter_fields(self):
        """iter_fields should iterate over all fields."""
        well = Well(well_id="A01")
        well.add_field(FieldOfView(field_id="1"))
        well.add_field(FieldOfView(field_id="2"))

        fields = list(well.iter_fields())
        assert len(fields) == 2


class TestPlate:
    """Tests for Plate dataclass."""

    def test_creation_96_well(self):
        """Should create 96-well plate correctly."""
        plate = Plate(plate_id="test", plate_format="96")
        assert plate.n_rows == 8
        assert plate.n_cols == 12
        assert plate.max_wells == 96

    def test_creation_384_well(self):
        """Should create 384-well plate correctly."""
        plate = Plate(plate_id="test", plate_format="384")
        assert plate.n_rows == 16
        assert plate.n_cols == 24
        assert plate.max_wells == 384

    def test_invalid_plate_format(self):
        """Invalid plate format should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown plate format"):
            Plate(plate_id="test", plate_format="48")

    def test_add_and_get_well(self):
        """add_well and get_well should work correctly."""
        plate = Plate(plate_id="test")
        well = Well(well_id="A01")
        plate.add_well(well)

        assert plate.n_wells == 1
        assert plate.get_well("A01") is well
        assert plate.get_well("B01") is None

    def test_get_or_create_well(self):
        """get_or_create_well should create new wells if needed."""
        plate = Plate(plate_id="test")

        well1 = plate.get_or_create_well("A01")
        assert plate.n_wells == 1
        assert well1.well_id == "A01"

        # Should return same well
        well2 = plate.get_or_create_well("A01")
        assert well1 is well2
        assert plate.n_wells == 1

    def test_well_ids_sorted(self):
        """well_ids property should return sorted list."""
        plate = Plate(plate_id="test")
        plate.add_well(Well(well_id="B02"))
        plate.add_well(Well(well_id="A01"))
        plate.add_well(Well(well_id="A02"))

        assert plate.well_ids == ["A01", "A02", "B02"]

    def test_iter_wells(self):
        """iter_wells should iterate in sorted order."""
        plate = Plate(plate_id="test")
        plate.add_well(Well(well_id="B01"))
        plate.add_well(Well(well_id="A01"))

        wells = list(plate.iter_wells())
        assert wells[0].well_id == "A01"
        assert wells[1].well_id == "B01"

    def test_set_control_wells(self):
        """set_control_wells should mark wells as controls."""
        plate = Plate(plate_id="test")
        plate.set_control_wells(["A01", "A02"], "negative")

        assert plate.get_well("A01").control_type == "negative"
        assert plate.get_well("A02").control_type == "negative"

    def test_get_control_wells(self):
        """get_control_wells should return wells of specific control type."""
        plate = Plate(plate_id="test")
        plate.set_control_wells(["A01", "A02"], "negative")
        plate.set_control_wells(["H11", "H12"], "positive")
        plate.add_well(Well(well_id="B01"))  # Regular well

        neg_wells = plate.get_control_wells("negative")
        assert len(neg_wells) == 2
        assert all(w.control_type == "negative" for w in neg_wells)

        pos_wells = plate.get_control_wells("positive")
        assert len(pos_wells) == 2

    def test_to_grid(self):
        """to_grid should return correct boolean array."""
        plate = Plate(plate_id="test", plate_format="96")
        plate.add_well(Well(well_id="A01"))
        plate.add_well(Well(well_id="H12"))

        grid = plate.to_grid()
        assert grid.shape == (8, 12)
        assert grid[0, 0] == True  # A01
        assert grid[7, 11] == True  # H12
        assert grid[0, 1] == False  # A02 not added

    def test_summary(self):
        """summary should return informative string."""
        plate = Plate(plate_id="test_plate", plate_format="96")
        plate.add_well(Well(well_id="A01"))
        plate.set_control_wells(["A02"], "negative")

        summary = plate.summary()
        assert "test_plate" in summary
        assert "96-well" in summary
        assert "2/96" in summary  # 2 wells with data


class TestCreatePlateFromWells:
    """Tests for create_plate_from_wells function."""

    def test_creates_plate_with_wells(self):
        """Should create plate with specified wells."""
        plate = create_plate_from_wells(
            plate_id="test",
            well_ids=["A01", "A02", "B01"],
            plate_format="96"
        )

        assert plate.plate_id == "test"
        assert plate.n_wells == 3
        assert plate.get_well("A01") is not None


class TestGenerateAllWellIds:
    """Tests for generate_all_well_ids function."""

    def test_96_well_count(self):
        """Should generate 96 well IDs for 96-well plate."""
        wells = generate_all_well_ids("96")
        assert len(wells) == 96

    def test_384_well_count(self):
        """Should generate 384 well IDs for 384-well plate."""
        wells = generate_all_well_ids("384")
        assert len(wells) == 384

    def test_96_well_first_last(self):
        """First and last wells should be correct for 96-well."""
        wells = generate_all_well_ids("96")
        assert wells[0] == "A01"
        assert wells[-1] == "H12"

    def test_384_well_first_last(self):
        """First and last wells should be correct for 384-well."""
        wells = generate_all_well_ids("384")
        assert wells[0] == "A01"
        assert wells[-1] == "P24"

    def test_row_major_order(self):
        """Wells should be in row-major order."""
        wells = generate_all_well_ids("96")
        # First row
        assert wells[:12] == [f"A{i:02d}" for i in range(1, 13)]
        # Second row
        assert wells[12:24] == [f"B{i:02d}" for i in range(1, 13)]
