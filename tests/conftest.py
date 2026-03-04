"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path

from aggrequant.common.image_utils import load_image


DATA_DIR = Path(__file__).parent / "data"


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow running")


@pytest.fixture(scope="session")
def nuclei_image():
    path = next(DATA_DIR.glob("*390*Blue*"))
    return load_image(path)


@pytest.fixture(scope="session")
def cell_image():
    path = next(DATA_DIR.glob("*473*Green*"))
    return load_image(path)


@pytest.fixture(scope="session")
def aggregate_image():
    path = next(DATA_DIR.glob("*631*FarRed*"))
    return load_image(path)


@pytest.fixture(scope="session")
def nuclei_labels(nuclei_image):
    """Run StarDist on the nuclei image once per session (slow)."""
    from aggrequant.segmentation.stardist import StarDistSegmenter
    return StarDistSegmenter().segment(nuclei_image)


@pytest.fixture(scope="session")
def cell_labels(cell_image, nuclei_labels):
    """Run Cellpose on the cell image once per session (slow).

    Uses a copy of nuclei_labels so the session fixture is not mutated.
    """
    from aggrequant.segmentation.cellpose import CellposeSegmenter
    nuclei_copy = nuclei_labels.copy()
    return CellposeSegmenter().segment(cell_image, nuclei_copy)
