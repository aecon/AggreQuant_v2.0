"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "gui: mark test as requiring GUI/display"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root directory."""
    return project_root


@pytest.fixture
def sample_well_ids():
    """Return sample well IDs for testing."""
    return ["A01", "A02", "A03", "B01", "B02", "H12"]


@pytest.fixture
def sample_control_assignments():
    """Return sample control assignments for testing."""
    return {
        "A01": "negative",
        "A02": "negative",
        "H11": "NT",
        "H12": "NT",
    }
