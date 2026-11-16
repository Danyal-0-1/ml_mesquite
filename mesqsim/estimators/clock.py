"""One clock for parent and child: fresh samples, a causal pod-to-host time map, pairs and rotation rates
(Step 2 brief §5.3).

Every pod stamps its samples with its own clock (``t_pod_s``, Step 0's unwrapped ``ms_lo``: the decode time on the
pod clock) and the host stamps arrival (``t_host_s``). The geometry compares a parent and a child *at the same
instant*, so both are mapped onto one clock:

* **Pod time -> host time.** Per pod and boot epoch, ``t_host ≈ alpha + beta·t_pod``. Arrivals lie *above* the
  decode times by a positive latency (send wait + radio + hub), so the map follows the **lower envelope** of
  ``t_host - t_pod``: the minimum in each block of ``clock_block_s`` pod seconds, and a least-squares line through
  those minima (an offset only, beta = 1, when the data span less than two blocks). The fit uses only frames that
  arrived at or before the analysis time and within the last ``clock_window_s``: causal. A latency common to both
  pods of a joint cancels in the pair.
* **Fresh samples only** (``status == fresh`` and ``valid``): a held repeat carries no new information.
* **Pairs.** At each fresh parent sample, the child's orientation is slerped between its two bracketing fresh
  samples, if they are at most ``max_pair_gap_s`` apart; otherwise there is no pair.
* **Rotation rates** between consecutive fresh samples of one pod, in the pod (body) frame:
  ``omega = Log(q(t-)^-1 · q(t)) / (t - t-)``, assigned to the midpoint. A constant heading offset Y(-psi) on the
  left of every report cancels in ``q(t-)^-1·q(t)``, so rates do not see the pods' private norths (tested).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..body import N_PODS
from ..conventions import log_map, quat_conj, quat_mul, quat_slerp

__all__ = ["PodSamples", "pod_samples", "epoch_starts", "fit_host_map", "map_epoch", "Pairs", "pair_samples",
           "body_rates", "interp_rates"]  # fmt: skip


@dataclass
class PodSamples:
    """One pod's fresh, valid samples (all epochs), sorted by (epoch, tick)."""

    pod_id: int
    row: np.ndarray  # index into the series (for scorer-side diagnostics only)
    epoch: np.ndarray  # boot epoch as Step 0 numbers it
    t_host_s: np.ndarray  # arrival time on the host clock
    t_pod_s: np.ndarray  # decode time on the pod clock (Step 0's unwrapped ms_lo)
    q: np.ndarray  # (N, 4) decoded, normalised, canonical


def pod_samples(series: pd.DataFrame) -> list[PodSamples]:
    """Fresh, valid samples of every pod (index = pod id; empty arrays for a pod without frames)."""
    ok = (series["status"].to_numpy() == "fresh") & series["valid"].to_numpy()
    pod = series["pod_id"].to_numpy()
    q_all = series[["qx", "qy", "qz", "qw"]].to_numpy()
    out = []
    for i in range(N_PODS):
        idx = np.flatnonzero(ok & (pod == i))  # the series is sorted by (pod, epoch, tick)
        out.append(PodSamples(i, idx, series["epoch"].to_numpy()[idx].astype(np.int64),
                              series["t_host_s"].to_numpy()[idx], series["t_pod_s"].to_numpy()[idx],
                              q_all[idx]))  # fmt: skip
    return out


def epoch_starts(series: pd.DataFrame) -> list[list[tuple[int, float]]]:
    """Per pod: (epoch, host time of its first received frame) for every boot epoch, in order.

    A new epoch is when the host *learns* that a pod rebooted: from then on its north is new.
    """
    rec = series[series["received"].to_numpy()]
    out: list[list[tuple[int, float]]] = [[] for _ in range(N_PODS)]
    for (pod, ep), g in rec.groupby(["pod_id", "epoch"], sort=True):
        out[int(pod)].append((int(ep), float(g["t_host_s"].min())))
    return out


MIN_SLOPE_BLOCKS = 5
"""A slope is fitted only through at least this many block minima (10 s of data at 2 s blocks). Over shorter
spans two or three noisy minima give absurd rates (+1.5 % was seen 3 s after an idle gap), while assuming the
nominal rate costs at most 50 ppm x 10 s = 0.5 ms."""


def fit_host_map(t_pod_s: np.ndarray, t_host_s: np.ndarray, block_s: float) -> tuple[float, float]:
    """``(alpha, beta)`` of the lower envelope ``t_host ≈ alpha + beta·t_pod`` (see the module docstring).

    The slope ``beta - 1`` is fitted through the block minima when there are at least :data:`MIN_SLOPE_BLOCKS` of
    them, and clipped to the clock-rate tolerance Step 0 assumes (``REBOOT_TOL_PPM``); otherwise beta = 1 and
    alpha follows the lowest residual.
    """
    from ..step0 import REBOOT_TOL_PPM

    t_pod = np.asarray(t_pod_s, dtype=np.float64)
    r = np.asarray(t_host_s, dtype=np.float64) - t_pod
    if t_pod.size == 0:
        raise ValueError("no samples to fit a clock map")
    blk = np.floor((t_pod - t_pod.min()) / block_s).astype(np.int64)
    order = np.lexsort((r, blk))  # by block, then residual: the first row of each block is its minimum
    first = order[np.r_[True, blk[order][1:] != blk[order][:-1]]]
    tm, rm = t_pod[first], r[first]
    if first.size < MIN_SLOPE_BLOCKS:
        return float(rm.min()), 1.0
    t0 = tm.mean()
    slope = float(np.sum((tm - t0) * (rm - rm.mean())) / np.sum((tm - t0) ** 2))
    slope = float(np.clip(slope, -REBOOT_TOL_PPM * 1e-6, REBOOT_TOL_PPM * 1e-6))
    a0 = float(rm.mean())  # line through the block minima: r = a0 + slope·(t_pod - t0)
    return a0 - slope * t0, 1.0 + slope


def map_epoch(ps: PodSamples, epoch: int, t_now_host_s: float, clock_window_s: float, block_s: float,
              t_min_host_s: float = -np.inf) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Samples of one epoch that arrived by ``t_now`` (and at or after ``t_min``), with their mapped times.

    Returns ``(idx, tau)``: indices into ``ps`` and host-clock sample times ``alpha + beta·t_pod`` from the
    clock map fitted on that epoch's frames of the last ``clock_window_s`` before ``t_now`` (causal).
    """
    sel = (ps.epoch == epoch) & (ps.t_host_s <= t_now_host_s)
    idx = np.flatnonzero(sel)
    if idx.size == 0:
        return idx, np.zeros(0)
    fit = idx[ps.t_host_s[idx] >= t_now_host_s - clock_window_s]
    if fit.size == 0:
        fit = idx[-1:]
    alpha, beta = fit_host_map(ps.t_pod_s[fit], ps.t_host_s[fit], block_s)
    idx = idx[ps.t_host_s[idx] >= t_min_host_s]
    return idx, alpha + beta * ps.t_pod_s[idx]


@dataclass
class Pairs:
    """Parent and child orientations at the same instants: the parent's fresh sample times."""

    t_s: np.ndarray  # (N,) host-clock times of the pairs (the parent's mapped sample times)
    q_p: np.ndarray  # (N, 4) parent orientation
    q_c: np.ndarray  # (N, 4) child orientation slerped to t_s
    ip: np.ndarray  # (N,) index of the parent sample (into the parent's arrays)
    jc: np.ndarray  # (N,) index of the child's earlier bracketing sample
    u: np.ndarray  # (N,) slerp fraction between child samples jc and jc + 1

    def __len__(self) -> int:
        return int(self.t_s.size)


def pair_samples(tau_p: np.ndarray, q_p: np.ndarray, tau_c: np.ndarray, q_c: np.ndarray,
                 max_gap_s: float) -> Pairs:  # fmt: skip
    """Pair every parent sample with the child slerped between its two bracketing samples (gap <= max_gap_s)."""
    tau_p, tau_c = np.asarray(tau_p, float), np.asarray(tau_c, float)
    if tau_p.size == 0 or tau_c.size < 2:
        e = np.zeros(0)
        return Pairs(e, np.zeros((0, 4)), np.zeros((0, 4)), e.astype(int), e.astype(int), e)
    j = np.searchsorted(tau_c, tau_p, side="right") - 1
    ok = (j >= 0) & (j + 1 < tau_c.size)
    jj = np.clip(j, 0, tau_c.size - 2)
    gap = tau_c[jj + 1] - tau_c[jj]
    ok &= (gap > 0) & (gap <= max_gap_s)
    ip = np.flatnonzero(ok)
    jc = jj[ok]
    u = (tau_p[ok] - tau_c[jc]) / gap[ok]
    qc = quat_slerp(q_c[jc], q_c[jc + 1], u)
    return Pairs(tau_p[ok], q_p[ok], qc, ip, jc, u)


def body_rates(tau: np.ndarray, q: np.ndarray, max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame rotation rates between consecutive samples: ``(t_mid, omega (M, 3) rad/s)``.

    ``omega = Log(q[k-1]^-1 · q[k]) / (tau[k] - tau[k-1])`` at the midpoint, for gaps in (0, max_gap_s].
    """
    if tau.size < 2:
        return np.zeros(0), np.zeros((0, 3))
    dt = np.diff(tau)
    ok = (dt > 0) & (dt <= max_gap_s)
    rel = quat_mul(quat_conj(q[:-1][ok]), q[1:][ok])
    return 0.5 * (tau[:-1][ok] + tau[1:][ok]), log_map(rel) / dt[ok][:, None]


def interp_rates(t_src: np.ndarray, w_src: np.ndarray, t_query: np.ndarray,
                 max_gap_s: float) -> tuple[np.ndarray, np.ndarray]:  # fmt: skip
    """Linear interpolation of a rate series to ``t_query`` where two source points bracket it within the gap.

    Returns ``(ok mask over t_query, rates (n_ok, 3))``.
    """
    if t_src.size < 2 or t_query.size == 0:
        return np.zeros(t_query.size, bool), np.zeros((0, 3))
    j = np.clip(np.searchsorted(t_src, t_query, side="right") - 1, 0, t_src.size - 2)
    gap = t_src[j + 1] - t_src[j]
    ok = (t_query >= t_src[0]) & (t_query <= t_src[-1]) & (gap > 0) & (gap <= max_gap_s)
    u = ((t_query - t_src[j]) / np.where(gap > 0, gap, 1.0))[ok, None]
    return ok, (1.0 - u) * w_src[j[ok]] + u * w_src[j[ok] + 1]
