"""Put a BVH recording and its OptiTrack take on one clock (Step 3 brief §5.1).

* **Coarse**, from the wall clocks: the BVH's name is its save time (the end of the take, laptop clock), so it starts
  ``n_frames x frame_time`` earlier; the Motive export states its capture start (OptiTrack computer's clock). The
  two clocks agreed within a few seconds on 2026-11-13.
* **Fine**: the lag, within ±4 s of the coarse one, that best correlates the four leg sites' (thighs, shins) height
  changes drawn by the page with those of their markers. A height change is the height with its centred 2 s moving
  mean removed (:func:`height_change`); drawn heights are taken relative to the drawn root, whose height follows
  the phone. The alignment uses the legs, so it fixes relative timing only; an absolute delay of the page cannot be
  measured from these files.
* **Frame time**: the page of 2026-11-13 wrote ``Frame Time: 0.0333333`` without measuring it. :func:`align`
  searches the frame interval too (:data:`FRAME_TIMES_S`), because a stretched clock spoils a single lag over a
  long take; the chosen interval is reported and used only for comparisons with OptiTrack.

``lag`` maps BVH time to Motive time: t_motive = k · frame_time + lag for BVH frame k (Motive time from its first
frame).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bvh import Bvh
from .motive import MotiveTake
from .recordings import bvh_end_local

__all__ = ["LEG_SITES", "SEARCH_S", "CHANGE_WINDOW_S", "FRAME_TIMES_S", "Alignment", "coarse_lag_s", "sample_at",
           "pearson", "height_change", "drawn_height_changes", "fine_lag_s", "align"]

LEG_SITES = ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")
SEARCH_S = 4.0
CHANGE_WINDOW_S = 2.0
FRAME_TIMES_S = tuple(np.round(np.arange(0.03280, 0.03361, 0.00001), 5))
"""Frame intervals searched by :func:`align`: 32.80 to 33.60 ms in 0.01 ms steps (1/30 s and the 33.0 ms measured on
2026-11-14 both inside)."""


@dataclass(frozen=True)
class Alignment:
    coarse_s: float
    lag_s: float  # t_motive = k · frame_time_s + lag_s
    score: float  # mean Pearson r of the four leg height changes at the best lag
    clock_difference_s: float  # lag_s - coarse_s: how far the two wall clocks disagreed
    frame_time_s: float = 0.0  # the BVH frame interval used (the file's own unless align() chose another)
    per_site: tuple[float, ...] = ()


def coarse_lag_s(bvh: Bvh, bvh_path: str, take: MotiveTake, frame_time_s: float | None = None) -> float:
    """BVH start minus Motive start, from the wall clocks, in seconds (BVH start = save time - its duration)."""
    dt = bvh.frame_time_s if frame_time_s is None else frame_time_s
    start = bvh_end_local(bvh_path).timestamp() - bvh.n_frames * dt
    return start - take.capture_start_local.timestamp()


def height_change(z: np.ndarray, dt_s: float, window_s: float = CHANGE_WINDOW_S) -> np.ndarray:
    """``z`` minus its centred moving mean over ``window_s`` (NaN-aware): the height's change, without drift."""
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z)
    v = np.where(ok, z, 0.0)
    c = np.concatenate([[0.0], np.cumsum(v)])
    k = np.concatenate([[0], np.cumsum(ok)])
    half = int(round(0.5 * window_s / dt_s))
    i = np.arange(len(z))
    lo, hi = np.clip(i - half, 0, len(z)), np.clip(i + half + 1, 0, len(z))
    with np.errstate(invalid="ignore", divide="ignore"):
        return z - (c[hi] - c[lo]) / (k[hi] - k[lo])


def drawn_height_changes(sites: dict[str, np.ndarray], root: np.ndarray, dt_s: float,
                         names=LEG_SITES) -> dict[str, np.ndarray]:
    """Height changes of drawn sites relative to the drawn root (whose height is the phone's, halved)."""
    return {s: height_change(sites[s][:, 2] - root[:, 2], dt_s) for s in names}


def sample_at(t_query_s: np.ndarray, t_s: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Nearest-sample lookup of ``values`` (N, ...) on the uniform grid ``t_s`` at ``t_query_s``; NaN outside."""
    dt = (t_s[-1] - t_s[0]) / max(len(t_s) - 1, 1)
    k = np.rint((np.asarray(t_query_s) - t_s[0]) / dt).astype(np.int64)
    ok = (k >= 0) & (k < len(t_s))
    out = np.full((len(k),) + values.shape[1:], np.nan)
    out[ok] = values[k[ok]]
    return out


def pearson(a: np.ndarray, b: np.ndarray, min_n: int = 30) -> float:
    """Pearson r over the samples where both are finite; NaN with fewer than ``min_n`` or a constant series."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n:
        return float("nan")
    x, y = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else float("nan")


def fine_lag_s(t_bvh_s: np.ndarray, drawn_change: dict[str, np.ndarray], t_motive_s: np.ndarray,
               truth: dict[str, np.ndarray], coarse_s: float, search_s: float = SEARCH_S,
               step_s: float | None = None) -> Alignment:
    """Search lags in ``coarse_s ± search_s`` (``step_s``, default one Motive frame) for the best mean correlation of
    the drawn leg height changes ``drawn_change`` (BVH times ``t_bvh_s``) with the markers' height changes."""
    dt_m = t_motive_s[1] - t_motive_s[0]
    step = step_s or dt_m
    lags = coarse_s + np.arange(-search_s, search_s + 0.5 * step, step)
    per = np.full((len(lags), len(LEG_SITES)), np.nan)
    for j, s in enumerate(LEG_SITES):
        tc = height_change(truth[s][:, 2], dt_m)
        d = np.asarray(drawn_change[s], dtype=np.float64)
        for lo in range(0, len(lags), 64):  # chunks of lags keep the (lags, frames) arrays small
            idx = np.rint((t_bvh_s[None, :] + lags[lo:lo + 64, None] - t_motive_s[0]) / dt_m).astype(np.int64)
            ok_i = (idx >= 0) & (idx < len(tc))
            y = np.where(ok_i, tc[np.clip(idx, 0, len(tc) - 1)], np.nan)
            ok = np.isfinite(y) & np.isfinite(d)[None, :]
            n = ok.sum(axis=1)
            x, y = np.where(ok, d[None, :], 0.0), np.where(ok, y, 0.0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mx, my = x.sum(1) / n, y.sum(1) / n
                cov = (x * y).sum(1) / n - mx * my
                r = cov / np.sqrt(((x * x).sum(1) / n - mx * mx) * ((y * y).sum(1) / n - my * my))
            per[lo:lo + 64, j] = np.where(n >= 30, r, np.nan)
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(per, axis=1) if np.isfinite(per).any() else np.full(len(lags), np.nan)
    if not np.isfinite(scores).any():
        raise ValueError("no overlap between the BVH and the take at any lag")
    i = int(np.nanargmax(scores))
    return Alignment(float(coarse_s), float(lags[i]), float(scores[i]), float(lags[i] - coarse_s),
                     float(t_bvh_s[1] - t_bvh_s[0]) if len(t_bvh_s) > 1 else 0.0, tuple(float(r) for r in per[i]))


def align(bvh: Bvh, bvh_path: str, drawn_sites: dict[str, np.ndarray], drawn_root: np.ndarray, take: MotiveTake,
          truth: dict[str, np.ndarray], frame_times_s=None) -> Alignment:
    """The best (frame interval, lag) pair: for each interval, the coarse lag from the wall clocks and the fine lag
    within ±4 s of it on a 1/60 s grid; then the lag refined to one Motive frame. By default the interval is searched
    over :data:`FRAME_TIMES_S` in two stages (0.05 ms steps, then 0.01 ms around the best)."""
    k = np.arange(bvh.n_frames)
    drawn = drawn_height_changes(drawn_sites, drawn_root, bvh.frame_time_s)

    def search(grid):
        best = None
        for dt in grid:
            a = fine_lag_s(k * dt, drawn, take.t_s, truth, coarse_lag_s(bvh, bvh_path, take, dt), step_s=1 / 60)
            if best is None or a.score > best.score:
                best = a
        return best

    if frame_times_s is None:
        best = search(FRAME_TIMES_S[::5])
        best = search([dt for dt in FRAME_TIMES_S if abs(dt - best.frame_time_s) <= 0.0000401])
    else:
        best = search(frame_times_s)
    fine = fine_lag_s(k * best.frame_time_s, drawn, take.t_s, truth, best.lag_s, search_s=1 / 60)
    return Alignment(best.coarse_s, fine.lag_s, fine.score, fine.lag_s - best.coarse_s, best.frame_time_s,
                     fine.per_site)
