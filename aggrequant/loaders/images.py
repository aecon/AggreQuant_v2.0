"""Image loading utilities for microscopy data."""

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

from aggrequant.common.logging import get_logger
from aggrequant.common.image_utils import find_image_files

logger = get_logger(__name__)


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
