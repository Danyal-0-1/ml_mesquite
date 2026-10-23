"""Step 0: deterministic repairs shared by simulated and real recordings (brief §4.6).

Input: a frames table (one row per received frame, the logger's schema) and, optionally, its ``session.json``.
It never reads labels. Output: a per-pod series with one row per send tick, received or lost.

1. **Time.** Per pod, in arrival order:
   - ``count`` is unwrapped into ``tick`` (sends since boot). The host-time gap between two frames says how many
     sends to expect; that resolves wraps across long gaps. A count that disagrees with the elapsed time by more
     than ``4 + 200 ppm`` of the expected sends means the pod rebooted, so a new ``epoch`` begins.
   - ``ms_lo`` is unwrapped into ``ms_unwrapped`` using the tick difference (31.25 ms per send).
   - A frame whose unwrapped ``ms_lo`` equals the previous received frame's is **held**, otherwise **fresh**.
   - Ticks missing between two received frames of the same recorded interval (``session.json``; touching
     recorded phases merge) are **lost**; after a reboot, so are the new epoch's ticks before its first received
     frame. Ticks the host cannot know about (edges of a recording, sends just before a reboot) are absent.
     :mod:`mesqsim.recording` defines this rule once, for Step 0 and the simulator's labels alike.
   - Host arrival time is kept, but never used as sample time.
2. **Integrity.** int16 / 32767; a frame with | |q| - 1 | > ``norm_tol`` is kept but marked ``valid = False``;
   valid quaternions are normalised and canonicalised to w >= 0.

3. **Box registration** (:func:`box_headings`). In the box every pod lies in an aligned slot, so the heading of
   each pod's box record (circular mean over its valid frames of the first boot epoch inside the ``box`` phase)
   estimates minus its private north, up to the slot error. Only differences between pods mean anything:
   delta_hat_pc = wrap(h_p - h_c).
4. **T-pose snapshot** (:func:`tpose_snapshot`, :func:`world_deltas`), as the suit does it: q_i(T) is the mean of
   pod i's frames in the ``tpose`` phase, and the world deltas are Delta_i(t) = q_i(t) · q_i(T)^-1.
5. Facing from an arm raise is Step 2 and is not here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .codec import Q_SCALE, WRAP
from .conventions import circ_mean_rad, heading_rad, quat_canonical, quat_conj, quat_mean, quat_mul
from .recording import interval_id, merge_recorded

__all__ = ["SEND_PERIOD_S", "NORM_TOL", "decode", "recorded_intervals", "phase_range", "box_headings",
           "tpose_snapshot", "world_deltas"]

SEND_PERIOD_S = 0.03125  # firmware: 31,250 us per send
NORM_TOL = 0.01  # quantisation alone moves | |q| - 1 | by < 1.3e-4; anything near 0.01 is corruption
REBOOT_TOL_SENDS = 4.0
REBOOT_TOL_PPM = 200.0  # pod and host clocks may disagree by this much in rate
NOMINAL_FIRST_SEND_MS = 1000.0  # firmware start-up; only places the first ms_lo in the right 65.5 s window


def recorded_intervals(session: dict | None) -> list[tuple[float, float]] | None:
    """Recorded host-time intervals from ``session.json`` (touching phases merged); ``None`` means "all"."""
    if not session:
        return None
    return merge_recorded([(p["name"], p["t_host_start_s"], p["t_host_end_s"], p.get("recorded", True))
                           for p in session["phases"]])  # fmt: skip


def _decode_pod(pod_id: int, df: pd.DataFrame, intervals, norm_tol: float) -> pd.DataFrame:
    h = df["t_host_s"].to_numpy(np.float64)
    c = df["count"].to_numpy(np.int64)
    m = df["ms_lo"].to_numpy(np.int64)
    n = h.size

    # --- count: unwrap, reboots, duplicates
    expect = np.diff(h) / SEND_PERIOD_S
    dc = np.mod(np.diff(c), WRAP)
    wraps = np.maximum(np.round((expect - dc) / WRAP), 0)
    dc_true = dc + WRAP * wraps
    reboot = np.abs(dc_true - expect) > REBOOT_TOL_SENDS + REBOOT_TOL_PPM * 1e-6 * expect
    epoch = np.r_[0, np.cumsum(reboot)].astype(np.int32)
    first_of_epoch = np.r_[True, reboot]
    start = np.maximum.accumulate(np.where(first_of_epoch, np.arange(n), 0))  # row where each row's epoch begins
    # tick = count at the epoch's first frame + the unwrapped count increments since then
    cum = np.cumsum(np.r_[0, dc_true]).astype(np.int64)
    tick = c[start] + cum - cum[start]
    dup = np.r_[False, (~reboot) & (dc_true == 0)]

    # --- ms_lo: unwrap within each epoch using the tick difference (31.25 ms per send)
    dm = np.mod(np.diff(m), WRAP)
    elapsed_ms = np.diff(tick) * SEND_PERIOD_S * 1000.0
    cum_ms = np.cumsum(np.r_[0, dm + WRAP * np.round((elapsed_ms - dm) / WRAP)]).astype(np.int64)
    guess = NOMINAL_FIRST_SEND_MS + tick * SEND_PERIOD_S * 1000.0  # places an epoch's first ms_lo
    base = m + WRAP * np.round((guess - m) / WRAP).astype(np.int64)
    ms = base[start] + cum_ms - cum_ms[start]
    new_sample = first_of_epoch | np.r_[False, ms[1:] != ms[:-1]]

    # --- integrity
    q = df[["qx", "qy", "qz", "qw"]].to_numpy(np.float64) / Q_SCALE
    norm = np.linalg.norm(q, axis=1)
    valid = np.abs(norm - 1.0) <= norm_tol
    q_unit = np.where(valid[:, None], quat_canonical(q / norm[:, None]), np.nan)

    keep = ~dup
    rec = pd.DataFrame({
        "pod_id": np.full(n, pod_id, np.uint8),
        "epoch": epoch,
        "tick": tick,
        "count": c.astype(np.uint16),
        "status": np.where(new_sample, "fresh", "held"),
        "received": np.ones(n, bool),
        "valid": valid,
        "t_host_s": h,
        "ms_lo": m.astype(np.float64),
        "ms_unwrapped": ms.astype(np.float64),
        "t_pod_s": ms / 1000.0,
        "batt": df["batt"].to_numpy(np.float64),
        "norm_err": norm - 1.0,
        "qx": q_unit[:, 0], "qy": q_unit[:, 1], "qz": q_unit[:, 2], "qw": q_unit[:, 3],
    })[keep]  # fmt: skip

    # --- missing ticks: only those bracketed by received frames of one recorded interval (mesqsim.recording)
    e = rec["epoch"].to_numpy()
    tk = rec["tick"].to_numpy()
    th = rec["t_host_s"].to_numpy()
    iid = np.zeros(tk.size, np.int64) if intervals is None else interval_id(th, intervals)
    # gap before row r (r >= 1): ticks between the previous received frame and this one, if both lie in one
    # recorded interval; after a reboot, the new epoch's ticks before its first received frame
    same_interval = (iid[1:] == iid[:-1]) & (iid[1:] >= 0)
    same_epoch = e[1:] == e[:-1]
    lo = np.where(same_epoch, tk[:-1] + 1, 0)
    length = np.where(same_interval, np.maximum(tk[1:] - lo, 0), 0)
    total = int(length.sum())
    if total:
        g = np.repeat(np.arange(length.size), length)  # which gap each lost tick belongs to
        ks = lo[g] + np.arange(total) - np.repeat(np.cumsum(length) - length, length)
        prev_t, prev_tick, next_t, next_tick = th[:-1][g], tk[:-1][g], th[1:][g], tk[1:][g]
        t_est = np.where(
            same_epoch[g],
            prev_t + (ks - prev_tick) * (next_t - prev_t) / np.maximum(next_tick - prev_tick, 1),  # interpolated
            next_t - (next_tick - ks) * SEND_PERIOD_S,
        )
        lost = pd.DataFrame({
            "pod_id": np.full(total, pod_id, np.uint8),
            "epoch": e[1:][g].astype(np.int32),
            "tick": ks.astype(np.int64),
            "count": (ks % WRAP).astype(np.uint16),
            "status": "lost",
            "received": False,
            "valid": False,
            "t_host_est_s": t_est,
        })  # fmt: skip
        rec = pd.concat([rec, lost], ignore_index=True)
    return rec


def decode(frames: pd.DataFrame, session: dict | None = None, norm_tol: float = NORM_TOL) -> pd.DataFrame:
    """The per-pod series: one row per (pod, epoch, tick), received (fresh or held) or lost."""
    intervals = recorded_intervals(session)
    parts = [
        _decode_pod(int(pod), df, intervals, norm_tol)
        for pod, df in frames.groupby("pod_id", sort=True)  # groupby keeps arrival order within a pod
    ]
    cols = ["pod_id", "epoch", "tick", "count", "status", "received", "valid", "t_host_s", "t_host_est_s",
            "ms_lo", "ms_unwrapped", "t_pod_s", "batt", "norm_err", "qx", "qy", "qz", "qw"]  # fmt: skip
    out = pd.concat(parts, ignore_index=True).reindex(columns=cols)
    return out.sort_values(["pod_id", "epoch", "tick"], kind="mergesort", ignore_index=True)


def phase_range(session: dict, name: str) -> tuple[float, float]:
    """Host-time range of one phase from ``session.json``."""
    for ph in session["phases"]:
        if ph["name"] == name:
            return ph["t_host_start_s"], ph["t_host_end_s"]
    raise KeyError(f"session.json has no phase {name!r}")


def _rows_in(series: pd.DataFrame, session: dict, name: str, epoch: int | None) -> pd.DataFrame:
    a, b = phase_range(session, name)
    sel = series["received"] & series["valid"] & (series["t_host_s"] >= a) & (series["t_host_s"] <= b)
    if epoch is not None:
        sel &= series["epoch"] == epoch
    return series[sel]


def box_headings(series: pd.DataFrame, session: dict) -> pd.DataFrame:
    """Per pod: circular-mean heading (radians) of its box record and the number of frames used.

    Only the first boot epoch counts: a pod that rebooted later keeps this (now stale) estimate.
    """
    rows = _rows_in(series, session, "box", epoch=0)
    out = []
    for pod, g in rows.groupby("pod_id", sort=True):
        h = heading_rad(g[["qx", "qy", "qz", "qw"]].to_numpy())
        out.append({"pod_id": int(pod), "heading_rad": float(circ_mean_rad(h)), "n_frames": len(g)})
    return pd.DataFrame(out, columns=["pod_id", "heading_rad", "n_frames"])


def tpose_snapshot(series: pd.DataFrame, session: dict) -> dict[int, np.ndarray]:
    """Per pod: q_i(T), the mean of its valid frames in the ``tpose`` phase."""
    rows = _rows_in(series, session, "tpose", epoch=None)
    return {int(pod): quat_mean(g[["qx", "qy", "qz", "qw"]].to_numpy()) for pod, g in rows.groupby("pod_id")}


def world_deltas(series: pd.DataFrame, snapshot: dict[int, np.ndarray]) -> np.ndarray:
    """Delta_i(t) = q_i(t) · q_i(T)^-1 for every row of ``series`` (NaN without a frame or a snapshot)."""
    q = series[["qx", "qy", "qz", "qw"]].to_numpy()
    ref = np.array([snapshot.get(int(pod), [np.nan] * 4) for pod in series["pod_id"]], dtype=float)
    return quat_mul(q, quat_conj(ref))
