"""Shared fixtures. Tests that need licensed or downloaded data skip cleanly when it is absent."""

import pytest

from mesqsim.data import amass, diodem


@pytest.fixture(scope="session")
def amass_files():
    files = amass.motion_files()
    if not files:
        pytest.skip(f"AMASS not downloaded (no motion files under {amass.amass_dir()}); see docs/DATA.md")
    return files


@pytest.fixture(scope="session")
def diodem_root():
    if not diodem.is_available():
        pytest.skip(f"DIODEM not downloaded ({diodem.diodem_dir()}); run scripts/download_diodem.py")
    return diodem.diodem_dir()
