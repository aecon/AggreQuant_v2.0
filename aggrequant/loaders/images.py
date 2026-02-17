"""Image loading utilities for microscopy data."""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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


def find_channel_files(
    directory: Path,
    channel_pattern: str,
    well: Optional[str] = None
) -> List[Path]:
    """
    Find all files matching a channel pattern.

    Arguments:
        directory: Directory to search
        channel_pattern: Pattern to match (e.g., "C01", "ch1", "w1")
        well: Optional well filter (e.g., "A01")

    Returns:
        List of matching file paths
    """
    all_files = find_image_files(directory, recursive=True)

    matches = []
    for f in all_files:
        if channel_pattern.lower() in f.name.lower():
            if well is None or well in f.name:
                matches.append(f)

    return sorted(matches)


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


def group_files_by_field(files: List[Path]) -> Dict[str, List[Path]]:
    """
    Group files within a well by field of view.

    Arguments:
        files: List of file paths (typically from one well)

    Returns:
        Dictionary mapping field ID to list of files
    """
    fields: Dict[str, List[Path]] = {}

    for f in files:
        info = parse_incell_filename(f.name)
        field_id = info.get("field", "1")

        if field_id not in fields:
            fields[field_id] = []
        fields[field_id].append(f)

    return fields


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

    def load_well_channel(
        self,
        well: str,
        channel: str
    ) -> Tuple[List[np.ndarray], List[Path]]:
        """
        Load all images for a well and channel.

        Arguments:
            well: Well ID (e.g., "A01")
            channel: Channel name (must be in channel_patterns)

        Returns:
            Tuple of (list of images, list of file paths)
        """
        if channel not in self.channel_patterns:
            raise ValueError(f"Unknown channel '{channel}'. Known: {list(self.channel_patterns.keys())}")

        pattern = self.channel_patterns[channel]
        well_files = self.get_well_files(well)

        # Filter by channel pattern
        channel_files = [f for f in well_files if pattern.lower() in f.name.lower()]

        if self.verbose:
            logger.info(f"Loading {len(channel_files)} images for {well}/{channel}")

        images = [load_image(f) for f in channel_files]
        return images, channel_files

    def load_field(
        self,
        well: str,
        field: str
    ) -> Dict[str, np.ndarray]:
        """
        Load all channels for a specific field of view.

        Arguments:
            well: Well ID
            field: Field ID

        Returns:
            Dictionary mapping channel name to image
        """
        well_files = self.get_well_files(well)
        field_groups = group_files_by_field(well_files)

        if field not in field_groups:
            raise ValueError(f"Field '{field}' not found. Available: {list(field_groups.keys())}")

        field_files = field_groups[field]
        result = {}

        for channel_name, pattern in self.channel_patterns.items():
            matching = [f for f in field_files if pattern.lower() in f.name.lower()]
            if matching:
                result[channel_name] = load_image(matching[0])

        return result
