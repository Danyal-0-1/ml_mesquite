"""Read OptiTrack Motive CSV exports, format 1.25 (Step 3 brief §5.1).

Layout of a file:

* row 1: key, value pairs (``Format Version``, ``Take Name``, ``Capture Frame Rate``, ``Capture Start Time``
  written as ``2026-11-13 01.41.56.863 PM`` in the OptiTrack computer's local time, ``Length Units``, ...);
* row 2: empty;
* rows 3-7: per column, ``Type`` (``Marker``), ``Name`` (``Unlabeled 4603``: a fragment, one marker for as long as
  Motive kept it), ``ID``, ``Parent`` and the quantity (``Position``);
* row 8: ``Frame``, ``Time (Seconds)``, then ``X``, ``Y``, ``Z`` per marker;
* then one row per frame; an empty cell means that marker was not seen in that frame.

Coordinates stay in Motive's frame (+Z up, metres); :func:`mesqsim.twin.page_frame.motive_to_world_vec` converts them.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["MotiveTake", "read_motive_csv"]


@dataclass
class MotiveTake:
    path: str
    take_name: str
    capture_start_local: dt.datetime  # naive: the OptiTrack computer's clock, as written
    frame_rate_hz: float
    frame: np.ndarray  # (N,) frame numbers
    t_s: np.ndarray  # (N,) seconds from the first frame
    names: list[str]  # one per marker, e.g. "Unlabeled 4603"
    xyz: np.ndarray  # (N, M, 3) metres in Motive's frame; NaN where the marker was not seen
    meta: dict[str, str]

    @property
    def n_frames(self) -> int:
        return int(self.t_s.size)

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.frame_rate_hz


def _parse_start(text: str) -> dt.datetime:
    return dt.datetime.strptime(text.strip(), "%Y-%m-%d %I.%M.%S.%f %p")


def read_motive_csv(path) -> MotiveTake:
    """Parse one export. Raises ``ValueError`` for anything but marker positions in metres."""
    path = Path(path)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [r for _, r in zip(range(8), csv.reader(f), strict=False)]
    first = rows[0]
    meta = {first[k].strip(): first[k + 1].strip() for k in range(0, len(first) - 1, 2) if first[k].strip()}
    if meta.get("Length Units", "Meters") != "Meters":
        raise ValueError(f"{path.name}: length units {meta.get('Length Units')!r}, expected Meters")
    types, names, quantity, axes = rows[2], rows[3], rows[6], rows[7]
    if axes[:2] != ["Frame", "Time (Seconds)"]:
        raise ValueError(f"{path.name}: row 8 starts {axes[:2]}, expected Frame, Time (Seconds)")
    cols, marker_names = [], []
    for c in range(2, len(axes) - 2, 3):
        if types[c] == "Marker" and quantity[c] == "Position" and axes[c:c + 3] == ["X", "Y", "Z"]:
            cols.append(c)
            marker_names.append(names[c])
    data = pd.read_csv(path, skiprows=8, header=None, dtype=np.float64, encoding="utf-8-sig").to_numpy()
    xyz = np.stack([data[:, c:c + 3] for c in cols], axis=1) if cols else np.empty((data.shape[0], 0, 3))
    return MotiveTake(str(path), meta.get("Take Name", path.stem), _parse_start(meta["Capture Start Time"]),
                      float(meta.get("Export Frame Rate", meta.get("Capture Frame Rate", "120"))),
                      data[:, 0].astype(np.int64), data[:, 1], marker_names, xyz, meta)
