"""Unit tests for aggrequant.loaders.images module."""

import pytest
from pathlib import Path
from aggrequant.loaders.images import parse_incell_filename, build_field_triplets


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


class TestBuildFieldTriplets:
    """Tests for build_field_triplets()."""

    PURPOSES = {"nuclei": "390", "cells": "548", "aggregates": "650"}

    def _make_files(self, tmp_path, filenames):
        """Create empty .tif files and return the directory."""
        for name in filenames:
            (tmp_path / name).write_bytes(b"")
        return tmp_path

    def test_complete_triplets(self, tmp_path):
        """Should discover complete triplets for all fields."""
        self._make_files(tmp_path, [
            "A - 01(fld 1 wv 390 - Blue).tif",
            "A - 01(fld 1 wv 548 - Green).tif",
            "A - 01(fld 1 wv 650 - Red).tif",
            "A - 01(fld 2 wv 390 - Blue).tif",
            "A - 01(fld 2 wv 548 - Green).tif",
            "A - 01(fld 2 wv 650 - Red).tif",
        ])
        triplets = build_field_triplets(tmp_path, self.PURPOSES)
        assert len(triplets) == 2
        assert triplets[0].well_id == "A01"
        assert triplets[0].field_id == "1"
        assert triplets[1].field_id == "2"
        assert set(triplets[0].paths.keys()) == {"nuclei", "cells", "aggregates"}

    def test_incomplete_triplet_skipped(self, tmp_path):
        """Should skip fields missing a channel."""
        self._make_files(tmp_path, [
            "A - 01(fld 1 wv 390 - Blue).tif",
            "A - 01(fld 1 wv 548 - Green).tif",
            # missing 650 for field 1
            "A - 01(fld 2 wv 390 - Blue).tif",
            "A - 01(fld 2 wv 548 - Green).tif",
            "A - 01(fld 2 wv 650 - Red).tif",
        ])
        triplets = build_field_triplets(tmp_path, self.PURPOSES)
        assert len(triplets) == 1
        assert triplets[0].field_id == "2"

    def test_sorted_by_well_then_field(self, tmp_path):
        """Should return triplets sorted by (well_id, field_id)."""
        self._make_files(tmp_path, [
            # Well B02 field 1
            "B - 02(fld 1 wv 390 - Blue).tif",
            "B - 02(fld 1 wv 548 - Green).tif",
            "B - 02(fld 1 wv 650 - Red).tif",
            # Well A01 field 2
            "A - 01(fld 2 wv 390 - Blue).tif",
            "A - 01(fld 2 wv 548 - Green).tif",
            "A - 01(fld 2 wv 650 - Red).tif",
            # Well A01 field 1
            "A - 01(fld 1 wv 390 - Blue).tif",
            "A - 01(fld 1 wv 548 - Green).tif",
            "A - 01(fld 1 wv 650 - Red).tif",
        ])
        triplets = build_field_triplets(tmp_path, self.PURPOSES)
        keys = [(t.well_id, t.field_id) for t in triplets]
        assert keys == [("A01", "1"), ("A01", "2"), ("B02", "1")]

    def test_empty_directory(self, tmp_path):
        """Should return empty list for a directory with no images."""
        triplets = build_field_triplets(tmp_path, self.PURPOSES)
        assert triplets == []

    def test_non_matching_files_ignored(self, tmp_path):
        """Should ignore files that don't match InCell format."""
        self._make_files(tmp_path, [
            "random_image.tif",
            "A - 01(fld 1 wv 390 - Blue).tif",
            "A - 01(fld 1 wv 548 - Green).tif",
            "A - 01(fld 1 wv 650 - Red).tif",
        ])
        triplets = build_field_triplets(tmp_path, self.PURPOSES)
        assert len(triplets) == 1
