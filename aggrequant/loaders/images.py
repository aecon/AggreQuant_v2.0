"""Image loading utilities for microscopy data."""

import re
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np

from aggrequant.common.logging import get_logger
from aggrequant.common.image_utils import load_image, load_image_stack, find_image_files

logger = get_logger(__name__)


# Sentinel value for unparseable well IDs
UNKNOWN_WELL_ID = "unknown"


def parse_incell_filename(filename: str) -> Dict[str, str]:
    """
    Parse GE InCell Analyzer microscope filename format.

    Expected formats (optional prefix before well ID):
        A - 01(fld 1 wv 390 - Blue).tif
        Plate1_B - 01(fld 01 wv 390 - Blue).tif
        Plate1_HA41_B - 01(fld 01 wv 390 - Blue).tif

    Arguments:
        filename: Filename to parse

    Returns:
        Dictionary with keys: row, col, field, wavelength
        Empty dict if the filename doesn't match.
    """
    pattern = r"([A-P])\s*-\s*(\d+)\(fld\s*(\d+)\s+wv\s+(\d+)"
    match = re.search(pattern, filename, re.IGNORECASE)

    if match:
        return {
            "row": match.group(1).upper(),
            "col": match.group(2),
            "field": match.group(3),
            "wavelength": match.group(4),
        }

    return {}


def group_files_by_well(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Group image files by well identifier.

    Arguments:
        files: List of file paths

    Returns:
        Dictionary mapping well ID (e.g. "A01") to list of files
    """
    wells: Dict[str, List[Path]] = {}

    for f in files:
        info = parse_incell_filename(f.name)

        if "row" in info and "col" in info:
            well_id = f"{info['row']}{int(info['col']):02d}"
        else:
            warnings.warn(
                f"Could not parse well ID from filename: {f.name}. "
                f"Grouping as '{UNKNOWN_WELL_ID}'."
            )
            well_id = UNKNOWN_WELL_ID

        if well_id not in wells:
            wells[well_id] = []
        wells[well_id].append(f)

    return wells


class FieldTriplet(NamedTuple):
    """One field of view with all its channel image paths."""
    well_id: str
    field_id: str
    paths: Dict[str, Path]  # purpose ("nuclei", "cells", ...) -> file path


def build_field_triplets(
    directory: Path,
    channel_purposes: Dict[str, str],
) -> List[FieldTriplet]:
    """
    Discover image files and group them into per-field triplets.

    Scans the directory once, parses every filename, matches channel patterns,
    and returns a sorted list of complete triplets (fields that have all channels).

    Arguments:
        directory: Root directory containing images
        channel_purposes: Mapping of purpose to filename pattern,
            e.g. {"nuclei": "390", "cells": "548", "aggregates": "650"}

    Returns:
        List of FieldTriplet sorted by (well_id, field_id), containing only
        fields where every expected channel was found.
    """
    all_files = find_image_files(directory, recursive=True)

    # (well_id, field_id) -> {purpose: path}
    grouped: Dict[Tuple[str, str], Dict[str, Path]] = defaultdict(dict)

    for f in all_files:
        info = parse_incell_filename(f.name)
        if not info:
            continue

        well_id = f"{info['row']}{int(info['col']):02d}"
        field_id = info["field"]

        # Match against channel patterns
        for purpose, pattern in channel_purposes.items():
            if pattern.lower() in f.name.lower():
                grouped[(well_id, field_id)][purpose] = f
                break  # each file matches at most one purpose

    # Keep only complete triplets (all purposes present)
    expected = set(channel_purposes.keys())
    triplets = []
    for (well_id, field_id), paths in sorted(grouped.items()):
        if paths.keys() >= expected:
            triplets.append(FieldTriplet(well_id, field_id, paths))
        else:
            missing = expected - paths.keys()
            logger.warning(
                f"Skipping {well_id}/f{field_id}: missing channel(s) {missing}"
            )

    return triplets


class ImageLoader:
    """
    High-level image loader for HCS data.

    Handles discovery and loading of multi-channel microscopy images
    organized by plate, well, and field of view.
    """

    def __init__(
        self,
        directory: Path,
        channel_patterns: Optional[Dict[str, str]] = None,
        verbose: bool = False
    ):
        """
        Initialize image loader.

        Arguments:
            directory: Root directory containing images
            channel_patterns: Dict mapping channel names to file patterns
                              e.g., {"DAPI": "C01", "GFP": "C02"}
            verbose: Print progress messages
        """
        self.directory = Path(directory)
        self.channel_patterns = channel_patterns or {}
        self.verbose = verbose

        # Discover files on init
        self._discover_files()

    def _discover_files(self):
        """Scan directory and organize files."""
        all_files = find_image_files(self.directory, recursive=True)
        if self.verbose:
            logger.info(f"Found {len(all_files)} image files")

        self.files_by_well = group_files_by_well(all_files)
        if self.verbose:
            logger.info(f"Organized into {len(self.files_by_well)} wells")

    @property
    def wells(self) -> List[str]:
        """List of discovered well IDs."""
        return sorted(self.files_by_well.keys())

    @property
    def n_wells(self) -> int:
        """Number of discovered wells."""
        return len(self.files_by_well)

    def get_well_files(self, well: str) -> List[Path]:
        """Get all files for a specific well."""
        return self.files_by_well.get(well, [])
