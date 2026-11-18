"""Shared fixtures. Tests that need licensed or downloaded data skip cleanly when it is absent."""

import pytest

from mesqsim.data import amass, diodem
from mesqsim.real import recordings


@pytest.fixture(scope="session")
def real_results():
    """Φ1-Φ12 of all seven recordings (about 10 s); skips without the recordings."""
    if not recordings.available():
        pytest.skip(f"recordings not copied to {recordings.real_suit_dir()} (Step 3 brief §2.3)")
    from mesqsim.real.analysis import analyse

    return {r.id: analyse(r) for r in recordings.RECORDINGS}


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
