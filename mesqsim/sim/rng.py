"""Random streams. Every draw of a run derives from its one seed (brief §6).

Each subsystem gets its own stream, seeded by ``(seed, name)``. Adding a new error term with its own stream
therefore leaves every other subsystem's draws unchanged, which keeps old runs comparable.
"""

from __future__ import annotations

import zlib

import numpy as np

__all__ = ["stream"]


def stream(seed: int, name: str) -> np.random.Generator:
    """The generator for subsystem ``name`` of the run with ``seed``."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])
