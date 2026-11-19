"""What each pod's gyroscope and accelerometer feel (Step 3 brief §4.3).

    ω_i(t) = (R_iᵀ Ṙ_i)^∨            angular rate in the pod's frame, rad/s
    f_i(t) = R_iᵀ (p̈_i(t) - g)       specific force in the pod's frame, m/s²,  g = (0, 0, -9.81)

AMASS gives poses at 60-120 Hz. Rotations are interpolated with ``scipy.spatial.transform.RotationSpline``, which is
C² and whose first derivative is the angular rate *in the body frame* (scipy 1.17.1; ``tests/test_twin_body.py``
pins it), i.e. ω directly. Positions are differentiated twice, which amplifies the jitter of motion capture by
(2π f)², so they go through a cubic smoothing spline (``scipy.interpolate.make_smoothing_spline``, which minimises
Σ (y - s)² + λ ∫ s''²). The sum counts samples, so for samples h apart it acts like the filter 1 / (1 + λ h ω⁴), and
λ = 1 / (h (2π f_c)⁴) halves the amplitude at f_c (``twin.physics.position_cutoff_hz``); the test measures the
response, which follows 1 / (1 + (f / f_c)⁴) to three decimals. ``cutoff_hz=None`` uses an interpolating cubic
spline instead, for the exact checks. Both are differentiated analytically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.spatial.transform import Rotation, RotationSpline

__all__ = ["G_M_S2", "ImuTruth", "smoothing_lambda", "position_spline", "imu_truth"]

G_M_S2 = np.array([0.0, 0.0, -9.81])
"""Gravity in the world, +Z up (Step 1 brief §4.2)."""


@dataclass
class ImuTruth:
    t_s: np.ndarray  # (T,)
    q: np.ndarray  # (T, 4) pod -> world
    omega_rad_s: np.ndarray  # (T, 3) in the pod's frame
    f_m_s2: np.ndarray  # (T, 3) specific force in the pod's frame
    p_m: np.ndarray  # (T, 3) world


def smoothing_lambda(dt_s: float, cutoff_hz: float) -> float:
    """λ of a cubic smoothing spline whose response is 1/2 at ``cutoff_hz`` for samples ``dt_s`` apart."""
    return 1.0 / (dt_s * (2 * np.pi * cutoff_hz) ** 4)


def position_spline(t_s, p, cutoff_hz: float | None):
    """A callable ``(t, nu)`` -> positions or their ``nu``-th derivative ``(T, 3)``."""
    t_s = np.asarray(t_s, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    if cutoff_hz is None:
        cs = CubicSpline(t_s, p, axis=0)
        return lambda t, nu=0: cs(t, nu)
    lam = smoothing_lambda(float(np.median(np.diff(t_s))), cutoff_hz)
    parts = [make_smoothing_spline(t_s, p[:, k], lam=lam) for k in range(p.shape[1])]
    return lambda t, nu=0: np.stack([s(t, nu) for s in parts], axis=-1)


def imu_truth(t_src, q_src, p_src, rate_hz: float, cutoff_hz: float | None = None, t_eval=None) -> ImuTruth:
    """The true rotation, angular rate and specific force of one pod at ``rate_hz`` (or at ``t_eval``).

    ``q_src`` (N, 4) pod -> world and ``p_src`` (N, 3) world metres, at ``t_src`` (N,) seconds.
    """
    t_src = np.asarray(t_src, dtype=np.float64)
    rs = RotationSpline(t_src, Rotation.from_quat(np.asarray(q_src)))
    t = np.arange(t_src[0], t_src[-1], 1.0 / rate_hz) if t_eval is None else np.asarray(t_eval, dtype=np.float64)
    rot = rs(t)
    omega = np.asarray(rs(t, 1))
    ps = position_spline(t_src, p_src, cutoff_hz)
    acc = ps(t, 2)
    f = rot.inv().apply(acc - G_M_S2)
    return ImuTruth(t, rot.as_quat(), omega, f, ps(t, 0))
