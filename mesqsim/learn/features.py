"""The model kit's inputs (Step 3 brief §7.1), from deployable data only: Step 0's series and ``phone.parquet``.

The training store and live inference call the same function, :func:`features`. It is causal: the value at step k
uses only what had arrived by its host time t_k, so computing it from the data cut at t_k (what a live stream holds
then) gives the stored value exactly (tested). Everything is unchanged by turning every pod and the phone by one
heading (check C5).

**One clock** (Step 2's clock module). Each pod sample's decode time on the pod clock is mapped onto the host clock
by :func:`mesqsim.estimators.clock.fit_host_map`, refitted once a second on that epoch's frames that have arrived
within the last ``geometry.clock_window_s`` (causal). The phone's messages carry no timestamp: their time is the
arrival time. Every quantity at step k is computed from what had arrived by its host time t_k, and describes the
body at t_k - ``LAG_S``, when each pod's samples usually bracket it (slerp between them; otherwise the latest held).

Per pod (15 × 8): tilt g = R̂ᵀ e_z (3); the rate between its last two fresh samples, Log(R̂(t₋)ᵀ R̂(t)) / (t - t₋),
rad/s in its own frame (3); the age of its last fresh sample, s (1); held (1: its last frame repeated a sample).
Per joint (14 × 6): the first two columns of R̂_pᵀ R̂_c (Zhou et al. 2019). The phone (8): the same eight, from its
messages (converted to +Z up as rotations within the reported frame), with "orientation only" for held. Per pod
relative to the phone (15 × 6): the first two columns of R̂_phᵀ R̂_i. States: per pod 0 lost, 1 held, 2 fresh (what
arrived since the previous step); the phone 0 no message, 1 orientation only, 2 full.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..body import JOINTS, N_JOINTS, N_PODS
from ..conventions import log_map, quat_conj, quat_mul, quat_normalize, quat_rotate, quat_slerp, quat_to_matrix

__all__ = ["FEATURE_VERSION", "LAG_S", "STEP_HZ", "Features", "pod_clock", "features", "six_d"]

FEATURE_VERSION = 1
LAG_S = 0.1
"""The features describe the body this long before the step: by then a pod at 32 Hz has usually sent a sample after
it, so its orientation is interpolated rather than held (`assumption`)."""
STEP_HZ = 32.0
CLOCK_REFIT_S = 1.0


@dataclass
class Features:
    t_host_s: np.ndarray  # (T,)
    x_pod: np.ndarray  # (T, 15, 8) float32
    x_joint: np.ndarray  # (T, 14, 6)
    x_phone: np.ndarray  # (T, 8)
    x_pod_phone: np.ndarray  # (T, 15, 6)
    state_pod: np.ndarray  # (T, 15) uint8
    state_phone: np.ndarray  # (T,) uint8
    q_pod: np.ndarray | None = None  # (T, 15, 4) float64: R̂_i, each pod in its own world (NaN before any sample)
    q_phone: np.ndarray | None = None  # (T, 4) float64: R̂_ph (+Z up); neither is stored: RIGID and the page test


def six_d(r_rel: np.ndarray) -> np.ndarray:
    """The first two columns of rotation matrices ``(..., 3, 3)``, as ``(..., 6)`` (column-major)."""
    return np.concatenate([r_rel[..., :, 0], r_rel[..., :, 1]], axis=-1)


def pod_clock(series: pd.DataFrame, clock_window_s: float, block_s: float) -> dict:
    """Per pod: its fresh valid samples with host-clock sample times, each mapped with the clock fitted on the frames
    of its epoch that had arrived by its own arrival (refitted once a second), and its held frames' arrivals."""
    from ..estimators.clock import fit_host_map, pod_samples

    out = {}
    pod = series["pod_id"].to_numpy()
    status = series["status"].to_numpy()
    received = series["received"].to_numpy()
    t_arr = series["t_host_s"].to_numpy()
    for ps in pod_samples(series):
        i = ps.pod_id
        tau = np.full(ps.t_host_s.size, np.nan)
        for e in np.unique(ps.epoch):
            idx = np.flatnonzero(ps.epoch == e)
            order = idx[np.argsort(ps.t_host_s[idx], kind="mergesort")]
            last_fit, alpha, beta = -np.inf, 0.0, 1.0
            for j in order:
                now = ps.t_host_s[j]
                if now - last_fit >= CLOCK_REFIT_S:
                    win = idx[(ps.t_host_s[idx] <= now) & (ps.t_host_s[idx] >= now - clock_window_s)]
                    alpha, beta = fit_host_map(ps.t_pod_s[win], ps.t_host_s[win], block_s)
                    last_fit = now
                tau[j] = alpha + beta * ps.t_pod_s[j]
        held = (pod == i) & received & (status == "held")
        out[i] = {"arrive": ps.t_host_s, "tau": tau, "q": ps.q, "epoch": ps.epoch, "held_arrive": np.sort(t_arr[held])}
    return out


def _slerp_rows(qa: np.ndarray, qb: np.ndarray, u: np.ndarray) -> np.ndarray:
    return quat_slerp(qa, qb, u) if len(u) else np.zeros((0, 4))


def _track_at(arr, tau, qs, ep, t_k, lag):
    """Vectorised: per step, the orientation at t_k - lag (slerp between the arrived samples that bracket it, in the
    current epoch; otherwise the latest), the rate between the last two samples, the age of the last one."""
    n = t_k.size
    q = np.full((n, 4), np.nan)
    rate = np.zeros((n, 3))
    age = np.zeros(n)
    n_arr = np.searchsorted(arr, t_k, side="right")
    has = n_arr > 0
    last = np.maximum(n_arr - 1, 0)
    tq = t_k - lag
    j = np.searchsorted(tau, tq, side="right") - 1  # tau is increasing: the host clock orders the samples
    interp = has & (j >= 0) & (j + 1 <= last) & (ep[np.clip(j, 0, None)] == ep[last]) \
        & (ep[np.clip(j + 1, 0, len(ep) - 1)] == ep[last])
    hold = has & ~interp
    q[hold] = qs[last[hold]]
    if interp.any():
        a, b = j[interp], j[interp] + 1
        u = np.clip((tq[interp] - tau[a]) / np.maximum(tau[b] - tau[a], 1e-9), 0.0, 1.0)
        q[interp] = _slerp_rows(qs[a], qs[b], u)
    two = has & (last >= 1)
    two &= (ep[np.maximum(last - 1, 0)] == ep[last]) & (tau[last] > tau[np.maximum(last - 1, 0)])
    if two.any():
        lb, la = last[two], last[two] - 1
        rate[two] = log_map(quat_mul(quat_conj(qs[la]), qs[lb])) / (tau[lb] - tau[la])[:, None]
    age[has] = t_k[has] - arr[last[has]]
    return q, rate, age, n_arr, last, has


def _pod_at(c: dict, t_k: np.ndarray, lag: float):
    """Per step: (R̂ at t_k - lag, rate, age, held flag, state) for one pod, causally."""
    n = t_k.size
    if c["arrive"].size == 0:
        return np.full((n, 4), np.nan), np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n, np.uint8)
    order = np.argsort(c["tau"], kind="mergesort")
    arr, tau, qs, ep = c["arrive"][order], c["tau"][order], c["q"][order], c["epoch"][order]
    if np.any(np.diff(arr) < 0):  # arrival order and clock order agree for one pod's FIFO link; keep it honest
        arr = np.maximum.accumulate(arr)
    q, rate, age, n_arr, last, has = _track_at(arr, tau, qs, ep, t_k, lag)
    held_a = c["held_arrive"]
    held_n = np.searchsorted(held_a, t_k, side="right")
    prev_n, prev_h = np.r_[0, n_arr[:-1]], np.r_[0, held_n[:-1]]
    state = np.zeros(n, np.uint8)
    state[held_n > prev_h] = 1
    state[n_arr > prev_n] = 2
    held = np.zeros(n)
    okh = has & (held_n > 0)
    held[okh] = (held_a[np.maximum(held_n[okh] - 1, 0)] > arr[last[okh]]).astype(float)
    return q, rate, age, held, state


def _phone_at(phone: pd.DataFrame, t_k: np.ndarray, lag: float):
    from ..twin.page_frame import arkit_to_world_turn

    n = t_k.size
    if phone is None or not len(phone):
        return np.full((n, 4), np.nan), np.zeros((n, 3)), np.zeros(n), np.zeros(n), np.zeros(n, np.uint8)
    arr = phone["t_host_s"].to_numpy(dtype=float)
    qm = quat_normalize(arkit_to_world_turn(phone[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)))
    has_p = np.isfinite(phone["px"].to_numpy(dtype=float))
    q, rate, age, n_arr, last, has = _track_at(arr, arr, qm, np.zeros(arr.size, np.int64), t_k, lag)
    only = np.zeros(n)
    only[has] = (~has_p[last[has]]).astype(float)
    prev = np.r_[0, n_arr[:-1]]
    state = np.zeros(n, np.uint8)
    new = n_arr > prev
    state[new] = np.where(has_p[last[new]], 2, 1)
    return q, rate, age, only, state


def features(series: pd.DataFrame, phone: pd.DataFrame | None, t_host_s, clock_window_s: float, block_s: float,
             lag_s: float = LAG_S) -> Features:  # fmt: skip
    """The §7.1 inputs at host times ``t_host_s`` (the 32 Hz steps), from Step 0's series and ``phone.parquet``."""
    t = np.asarray(t_host_s, dtype=np.float64)
    n = t.size
    clocks = pod_clock(series, clock_window_s, block_s)
    q = np.full((n, N_PODS, 4), np.nan)
    x_pod = np.zeros((n, N_PODS, 8))
    state_pod = np.zeros((n, N_PODS), np.uint8)
    up = np.array([0.0, 0.0, 1.0])
    for i in range(N_PODS):
        if i not in clocks:
            continue
        qi, rate, age, held, st = _pod_at(clocks[i], t, lag_s)
        ok = np.isfinite(qi[:, 3])
        q[:, i] = qi
        x_pod[ok, i, 0:3] = quat_rotate(quat_conj(qi[ok]), up)
        x_pod[ok, i, 3:6] = rate[ok]
        x_pod[ok, i, 6] = age[ok]
        x_pod[ok, i, 7] = held[ok]
        state_pod[:, i] = st
    x_joint = np.zeros((n, N_JOINTS, 6))
    for k, jt in enumerate(JOINTS):
        qp, qc = q[:, jt.parent], q[:, jt.child]
        ok = np.isfinite(qp[:, 3]) & np.isfinite(qc[:, 3])
        if ok.any():
            x_joint[ok, k] = six_d(quat_to_matrix(quat_mul(quat_conj(qp[ok]), qc[ok])))
    qph, rph, aph, only, st_ph = _phone_at(phone, t, lag_s)
    x_phone = np.zeros((n, 8))
    okp = np.isfinite(qph[:, 3])
    x_phone[okp, 0:3] = quat_rotate(quat_conj(qph[okp]), up)
    x_phone[okp, 3:6] = rph[okp]
    x_phone[okp, 6] = aph[okp]
    x_phone[okp, 7] = only[okp]
    x_pp = np.zeros((n, N_PODS, 6))
    for i in range(N_PODS):
        ok = okp & np.isfinite(q[:, i, 3])
        if ok.any():
            x_pp[ok, i] = six_d(quat_to_matrix(quat_mul(quat_conj(qph[ok]), q[ok, i])))
    f32 = np.float32
    return Features(t, x_pod.astype(f32), x_joint.astype(f32), x_phone.astype(f32), x_pp.astype(f32), state_pod,
                    st_ph, q, qph)  # fmt: skip
