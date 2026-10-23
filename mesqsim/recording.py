"""What one uninterrupted recording is, and which missing frames count as *lost*. Shared by the simulator (to
write the labels) and Step 0 (to read real or simulated frames), so the two agree exactly.

* Recorded phases that touch are merged into one **recorded interval** (closed). The simulated idle interval is
  not recorded, so a session normally has two: box -> T-pose, and motion.
* A send tick that never reached the frames table counts as **lost** only when the host can know it existed:
  the pod's previous received frame and its next received frame lie in the same recorded interval, and the next
  one belongs to the same boot epoch as the tick. Ticks at the edges of a recording, or sent just before a
  reboot, are unknowable to any logger and are left out of both the labels and Step 0's series.
"""

from __future__ import annotations

import numpy as np

__all__ = ["merge_recorded", "interval_id", "knowable_ticks"]


def merge_recorded(phases) -> list[tuple[float, float]]:
    """``[(name, t0, t1, recorded), ...]`` -> closed intervals covering the recorded phases, touching ones merged."""
    out: list[list[float]] = []
    for _, t0, t1, rec in sorted(phases, key=lambda p: p[1]):
        if not rec or t1 < t0:
            continue
        if out and t0 <= out[-1][1]:
            out[-1][1] = max(out[-1][1], t1)
        else:
            out.append([t0, t1])
    return [(a, b) for a, b in out]


def interval_id(t_s, intervals) -> np.ndarray:
    """Index of the recorded interval containing each time; -1 outside every interval."""
    t = np.asarray(t_s, float)
    out = np.full(t.shape, -1, np.int64)
    for k, (a, b) in enumerate(intervals):
        out[(t >= a) & (t <= b)] = k
    return out


def knowable_ticks(delivered: np.ndarray, epoch: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """For one pod's ticks in send order: which ticks the host can know about (delivered, or lost between two
    delivered frames of the same recorded interval, with the later one in the tick's own epoch)."""
    n = delivered.size
    idx = np.arange(n)
    prev = np.maximum.accumulate(np.where(delivered, idx, -1))
    nxt = np.minimum.accumulate(np.where(delivered, idx, n)[::-1])[::-1]
    have = (prev >= 0) & (nxt < n)
    p, q = np.clip(prev, 0, n - 1), np.clip(nxt, 0, n - 1)
    bracketed = have & (interval[p] == interval[q]) & (interval[p] >= 0) & (epoch[q] == epoch)
    return delivered | bracketed
