"""Rule 3 of the brief, enforced: no data in git and every committed file under 5 MB."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA_SUFFIXES = (".npz", ".npy", ".pkl", ".parquet", ".h5", ".hdf5", ".tar", ".bz2", ".xz", ".zip", ".pt", ".pth")


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [ROOT / p for p in out.decode().split("\0") if p]


def test_every_committed_file_is_under_5_mb():
    big = [(p.relative_to(ROOT).as_posix(), p.stat().st_size) for p in tracked_files()
           if p.is_file() and p.stat().st_size >= 5 * 1024 * 1024]
    assert not big, big


def test_no_data_files_committed():
    bad = [p.relative_to(ROOT).as_posix() for p in tracked_files()
           if p.name.endswith(DATA_SUFFIXES) and "tests/fixtures" not in p.as_posix()]
    assert not bad, bad


def test_package_data_folder_is_not_ignored():
    """A bare 'data/' rule in .gitignore once hid mesqsim/data/; the rules must be anchored to the root."""
    res = subprocess.run(["git", "check-ignore", "-q", "mesqsim/data/amass.py"], cwd=ROOT)
    assert res.returncode == 1  # 1 = not ignored
