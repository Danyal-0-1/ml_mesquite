"""Where data and outputs live.

``MESQ_DATA_DIR`` and ``MESQ_OUTPUT_DIR`` override the defaults ``<repo>/data`` and ``<repo>/outputs``
(both git-ignored). On the collaborator's machine they point outside OneDrive, to
``D:\\mesquite_work\\data`` and ``D:\\mesquite_work\\outputs``.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ROOT", "data_dir", "output_dir", "manifest_dir"]

REPO_ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Root folder of downloaded datasets and body models."""
    return Path(os.environ.get("MESQ_DATA_DIR", REPO_ROOT / "data"))


def output_dir() -> Path:
    """Root folder of simulator runs and results."""
    return Path(os.environ.get("MESQ_OUTPUT_DIR", REPO_ROOT / "outputs"))


def manifest_dir() -> Path:
    """Committed manifests (file name, size, SHA-256) of every data file. ``MESQ_MANIFEST_DIR`` overrides it
    (tests only, so they never write into the repository)."""
    return Path(os.environ.get("MESQ_MANIFEST_DIR", REPO_ROOT / "manifests"))
