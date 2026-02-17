"""Unit tests for aggrequant.loaders.images module."""

import pytest
from aggrequant.loaders.images import parse_incell_filename


class TestParseIncellFilename:
    """Tests for InCell Analyzer filename parsing."""

    def test_standard_format(self):
        """Should parse standard InCell filename."""
        result = parse_incell_filename("A - 01(fld 1 wv 390 - Blue).tif")
        assert result == {
            "row": "A",
            "col": "01",
            "field": "1",
            "wavelength": "390",
        }

    def test_with_plate_prefix(self):
        """Should parse filename with plate prefix."""
        result = parse_incell_filename("Plate1_B - 01(fld 01 wv 390 - Blue).tif")
        assert result == {
            "row": "B",
            "col": "01",
            "field": "01",
            "wavelength": "390",
        }

    def test_with_multi_prefix(self):
        """Should parse filename with multiple prefix segments."""
        result = parse_incell_filename("Plate1_HA41_B - 01(fld 01 wv 390 - Blue).tif")
        assert result == {
            "row": "B",
            "col": "01",
            "field": "01",
            "wavelength": "390",
        }

    def test_different_wavelengths(self):
        """Should correctly extract different wavelength values."""
        for wv in ("390", "473", "631"):
            result = parse_incell_filename(f"A - 01(fld 1 wv {wv} - Blue).tif")
            assert result["wavelength"] == wv

    def test_multidigit_well_column(self):
        """Should handle multi-digit column numbers."""
        result = parse_incell_filename("C - 12(fld 3 wv 473 - Green).tif")
        assert result["row"] == "C"
        assert result["col"] == "12"

    def test_high_row_letter(self):
        """Should handle rows up to P (for 384-well plates)."""
        result = parse_incell_filename("P - 24(fld 1 wv 631 - Red).tif")
        assert result["row"] == "P"
        assert result["col"] == "24"

    def test_lowercase_row(self):
        """Should handle lowercase row letters (case-insensitive regex)."""
        result = parse_incell_filename("a - 01(fld 1 wv 390 - Blue).tif")
        assert result["row"] == "A"

    def test_no_spaces_around_dash(self):
        """Should handle missing spaces around the dash."""
        result = parse_incell_filename("A-01(fld 1 wv 390 - Blue).tif")
        assert result["row"] == "A"
        assert result["col"] == "01"

    def test_unrecognized_format_returns_empty(self):
        """Should return empty dict for unrecognized filenames."""
        assert parse_incell_filename("random_image.tif") == {}
        assert parse_incell_filename("") == {}

    def test_partial_match_missing_wavelength(self):
        """Should return empty dict if pattern is incomplete."""
        assert parse_incell_filename("A - 01(fld 1).tif") == {}
