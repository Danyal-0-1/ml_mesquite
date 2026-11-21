"""The twin's configuration, ``configs/twin.yaml``: Step 1's labelled-parameter format, a file of its own.

Every leaf carries ``value``, ``label`` (datasheet | firmware | assumption | measured | fitted) and ``source``;
:class:`mesqsim.config.Config` refuses anything else. Keys are dotted, e.g. ``twin.sensors.gyro_scale_max``.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Config, load_config
from ..paths import REPO_ROOT

__all__ = ["TWIN_CONFIG", "load_twin_config"]

TWIN_CONFIG = REPO_ROOT / "configs" / "twin.yaml"


def load_twin_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    """``configs/twin.yaml`` (or ``path``), with ``{dotted key: value}`` overrides applied."""
    cfg = load_config(TWIN_CONFIG if path is None else path)
    return cfg.with_overrides(overrides) if overrides else cfg
