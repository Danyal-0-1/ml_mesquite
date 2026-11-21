"""RIGID and the chain along the tree (Step 3 brief §7.3): geometry, not learning.

**RIGID** estimates ε_h, the heading of HipsAlt relative to the phone: two sensors on one rigid body (the pelvis) see
one physical rotation, so their **world-frame** rates differ only by the two private norths,

    ω̂_ph = Y(ψ_h - ψ_ph) · ω̂_h = Y(ε_h) · ω̂_h,      ω̂ = Log(q(t) · q(t₋)⁻¹) / (t - t₋).

With a_t and b_t the horizontal parts of ω̂_ph and ω̂_h at paired instants, Step 2's closed form
(:func:`mesqsim.estimators.hinge.hinge_closed_form`) gives ε̂_h = atan2(Σ b × a, Σ a·b) and κ = I / σ̂². The vertical
parts must agree (a check, :attr:`RigidFit.vertical_rms`). Mountings do not enter: the rates are in world frames.
A rotation about the vertical alone has no horizontal part: κ = 0.

**The chain.** ε_c = ε_p + δ_pc, since ψ_c - ψ_ph = (ψ_c - ψ_p) + (ψ_p - ψ_ph): from ε̂_HipsAlt outward along the tree
(:data:`mesqsim.body.JOINTS` lists parents before children). For large κ the variance adds: 1/κ_c ≈ 1/κ_p + 1/κ_pc.
Without HipsAlt the phone is the pelvis sensor: start from ε_root = 0 and give the pelvis joints' δ as measured
against the phone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..body import JOINTS, N_PODS, ROOT_POD
from ..conventions import KAPPA_MAX, log_map, quat_conj, quat_mul, wrap_rad
from ..estimators.hinge import hinge_closed_form

__all__ = ["RigidFit", "world_rates", "rigid", "rigid_track", "chain", "chain_kappa", "TRACK_WINDOW_S", "KAPPA_HOLD"]

TRACK_WINDOW_S = 30.0
"""RIGID's causal track: the trailing window of rate pairs at each step (`assumption`)."""
KAPPA_HOLD = 100.0
"""Below this κ (a 1σ of about 6°) the track holds its last confident estimate (`assumption`)."""


@dataclass(frozen=True)
class RigidFit:
    eps_rad: float
    kappa: float
    info: float
    n: int
    vertical_rms: float  # RMS of the two vertical rates' difference, rad/s: near zero on one rigid body


def world_rates(t_s, q) -> tuple[np.ndarray, np.ndarray]:
    """World-frame rates between consecutive orientations: (midpoint times (N-1,), ω (N-1, 3) rad/s)."""
    t = np.asarray(t_s, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    dt = np.diff(t)
    w = log_map(quat_mul(q[1:], quat_conj(q[:-1]))) / dt[:, None]
    return 0.5 * (t[1:] + t[:-1]), w


def rigid(t_ph, q_ph, t_h, q_h, sigma_floor_rad: float = np.deg2rad(0.5)) -> RigidFit:
    """ε̂_h from the phone's and HipsAlt's orientation streams (their own private worlds, +Z up): HipsAlt's rates
    paired with the phone's interpolated at the same instants."""
    tp, wp = world_rates(t_ph, q_ph)
    th, wh = world_rates(t_h, q_h)
    inside = (th >= tp[0]) & (th <= tp[-1])
    if inside.sum() < 1:
        return RigidFit(0.0, 0.0, 0.0, 0, float("nan"))
    a = np.stack([np.interp(th[inside], tp, wp[:, k]) for k in range(3)], axis=1)
    b = wh[inside]
    fit = hinge_closed_form(a[:, :2], b[:, :2], sigma_floor_rad)
    vert = float(np.sqrt(np.mean((a[:, 2] - b[:, 2]) ** 2)))
    return RigidFit(fit.delta_rad, fit.kappa, fit.info, fit.n, vert)


def rigid_track(q_ph, q_h, step_hz: float = 32.0, window_s: float = TRACK_WINDOW_S,
                sigma_floor_rad: float = np.deg2rad(0.5), kappa_hold: float = KAPPA_HOLD):  # fmt: skip
    """Causal RIGID on the step grid, from the phone's and HipsAlt's orientation tracks (``(T, 4)``, each in its own
    world, NaN where nothing has arrived; :class:`mesqsim.learn.features.Features` ``q_phone`` and ``q_pod[:, 2]``).
    At step k, the closed form over the world-rate pairs between consecutive steps within the trailing ``window_s``;
    where that window gives κ below ``kappa_hold`` (still, or turning about the vertical only), the last confident
    estimate is held, with its κ. Returns ``(eps (T,), kappa (T,))``; 0 and 0 before the first confident window."""
    q_ph, q_h = np.asarray(q_ph, dtype=np.float64), np.asarray(q_h, dtype=np.float64)
    n = q_ph.shape[0]
    eps, kap = np.zeros(n), np.zeros(n)
    if n < 2:
        return eps, kap
    ok = np.isfinite(q_ph[:, 3]) & np.isfinite(q_h[:, 3])
    pair = ok[1:] & ok[:-1]
    a, b = np.zeros((n - 1, 2)), np.zeros((n - 1, 2))
    if pair.any():
        a[pair] = (log_map(quat_mul(q_ph[1:][pair], quat_conj(q_ph[:-1][pair]))) * step_hz)[:, :2]
        b[pair] = (log_map(quat_mul(q_h[1:][pair], quat_conj(q_h[:-1][pair]))) * step_hz)[:, :2]
    per = np.stack([np.sum(a * b, axis=1), b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0],
                    np.sum(a * a, axis=1) + np.sum(b * b, axis=1), pair.astype(float)], axis=1)  # fmt: skip
    cum = np.vstack([np.zeros((1, 4)), np.cumsum(per, axis=0)])  # cum[j] = sum of pairs 0 .. j-1
    w = max(1, int(round(window_s * step_hz)))
    k = np.arange(1, n)  # step k closes pair k - 1
    win = cum[k] - cum[np.maximum(k - w, 0)]
    c, s, e, m = win[:, 0], win[:, 1], win[:, 2], win[:, 3]
    info = np.hypot(c, s)
    rss = np.maximum(e - 2.0 * info, 0.0)
    sigma2 = np.maximum(rss / np.maximum(2.0 * m - 1.0, 1.0), sigma_floor_rad**2)
    kk = np.where(info > 1e-12, np.minimum(info / sigma2, KAPPA_MAX), 0.0)
    est = np.arctan2(s, c)
    last_e, last_k = 0.0, 0.0
    for j in range(n - 1):  # hold the last confident window
        if kk[j] >= kappa_hold:
            last_e, last_k = float(est[j]), float(kk[j])
        eps[j + 1], kap[j + 1] = last_e, last_k
    return eps, kap


def chain(eps_root, delta) -> np.ndarray:
    """ε (..., 15) from ε of the root pod (..., ) and δ per joint (..., 14): ε_c = ε_p + δ_pc outward."""
    delta = np.asarray(delta, dtype=np.float64)
    eps = np.full(delta.shape[:-1] + (N_PODS,), np.nan)
    eps[..., ROOT_POD] = np.asarray(eps_root, dtype=np.float64)
    for k, jt in enumerate(JOINTS):
        eps[..., jt.child] = wrap_rad(eps[..., jt.parent] + delta[..., k])
    return eps


def chain_kappa(kappa_root, kappa_joints) -> np.ndarray:
    """κ (..., 15) along the chain: 1/κ_c = 1/κ_p + 1/κ_pc (κ = 0 is no information, and stays so outward)."""
    kj = np.asarray(kappa_joints, dtype=np.float64)
    var = np.full(kj.shape[:-1] + (N_PODS,), np.inf)
    with np.errstate(divide="ignore"):
        var[..., ROOT_POD] = 1.0 / np.asarray(kappa_root, dtype=np.float64)
        for k, jt in enumerate(JOINTS):
            var[..., jt.child] = var[..., jt.parent] + 1.0 / kj[..., k]
    with np.errstate(divide="ignore"):
        return np.minimum(1.0 / var, KAPPA_MAX)
