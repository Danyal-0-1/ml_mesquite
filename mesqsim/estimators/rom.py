"""Range of motion (Lehmann et al. 2020): the only clue at ball joints (Step 2 brief §5.6).

    J(delta, t) = (q_p·m_p^-1)^-1 · Y(delta) · (q_c·m_c^-1)       equal to s_p^-1·s_c when delta = delta_pc
    cost(delta) = sum_t dist(angles(J(delta, t)), Omega)²         dist = distance outside the box, 0 inside

``angles`` is the joint's documented Cardan sequence (:data:`mesqsim.estimators.anatomy.CARDAN_SEQ`) and Omega one
interval per angle, fitted on the training subjects' true joint angles (:func:`fit_limits_deg`).

**Fast form.** With rotation matrices A_t = R(q_p·m_p^-1), B_t = R(q_c·m_c^-1) and
Rz(delta) = P + cos(delta)·(I - P) + sin(delta)·K (P = e_z·e_z^T, K = [e_z]_x),

    J(delta, t) = M0_t + cos(delta)·M1_t + sin(delta)·M2_t,   M0 = A^T·P·B,  M1 = A^T·(I - P)·B,  M2 = A^T·K·B

so a whole grid of delta costs a few multiply-adds per sample.

**Confidence.** The cost is exactly flat (zero) for every delta that keeps the motion inside the box, so the
curvature at a minimum (the Laplace approximation used for the hinge and 2-DoF joints) is zero there and says
nothing about how wide the flat region is. Instead the window's von Mises is fitted to the weights
w(delta) ∝ exp(-(cost(delta) - min cost)/(2·sigma_rom²)) on the grid by matching their circular mean:
mu = its direction, kappa = A^-1(its length). If the motion never reaches a limit, w is uniform and kappa = 0 - the
expected result at ball joints in everyday motion, not a bug.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..body import JOINTS
from ..conventions import (
    matrix_to_cardan_rad,
    quat_conj,
    quat_mul,
    quat_to_matrix,
    vm_kappa_from_resultant,
    wrap_rad,
)
from .anatomy import CARDAN_SEQ

__all__ = ["GRID_RAD", "RomFit", "rom_terms", "rom_angles", "rom_cost", "rom_solve", "joint_angles_rad",
           "fit_limits_deg", "UNBOUNDED_DEG"]  # fmt: skip

GRID_RAD = np.linspace(-np.pi, np.pi, 361)[1:]  # 1° steps over (-180°, 180°]
UNBOUNDED_DEG = [[-180.0, 180.0], [-90.0, 90.0], [-180.0, 180.0]]
"""A box that every Cardan angle satisfies: no constraint (the value before Omega is fitted)."""

_AX = {"X": 0, "Y": 1, "Z": 2}
_P = np.diag([0.0, 0.0, 1.0])
_K = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


@dataclass(frozen=True)
class RomFit:
    delta_rad: float  # circular mean of the weights
    kappa: float  # A^-1(mean resultant length of the weights); 0 = the motion never reached a limit
    cost_min: float
    frac_violating: float  # share of the delta grid with cost above the minimum (0 = flat everywhere)
    n: int


def rom_terms(a_q: np.ndarray, b_q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(M0, M1, M2), each (N, 3, 3), from A = q_p·m_p^-1 and B = q_c·m_c^-1 given as quaternions (N, 4)."""
    a = quat_to_matrix(a_q)
    b = quat_to_matrix(b_q)
    at = np.swapaxes(a, -1, -2)
    return at @ _P @ b, at @ (np.eye(3) - _P) @ b, at @ _K @ b


def rom_angles(m0: np.ndarray, m1: np.ndarray, m2: np.ndarray, seq: str, grid: np.ndarray = GRID_RAD) -> np.ndarray:
    """Cardan angles (G, N, 3) of J(delta, t) for every delta of the grid (only the five entries the sequence needs)."""
    i, j, k = (_AX[c] for c in seq)
    c, s = np.cos(grid)[:, None], np.sin(grid)[:, None]

    def entry(r, col):
        return m0[None, :, r, col] + c * m1[None, :, r, col] + s * m2[None, :, r, col]

    e = 1.0 if (j - i) % 3 == 1 else -1.0
    b = np.arcsin(np.clip(e * entry(i, k), -1.0, 1.0))
    a = np.arctan2(-e * entry(j, k), entry(k, k))
    g = np.arctan2(-e * entry(i, j), entry(i, i))
    return np.stack([a, b, g], axis=-1)


def rom_cost(angles: np.ndarray, lo_rad: np.ndarray, hi_rad: np.ndarray) -> np.ndarray:
    """Sum over samples of the squared distance outside the box, for every delta: (G,)."""
    d = np.maximum(0.0, np.maximum(lo_rad - angles, angles - hi_rad))
    return np.sum(d * d, axis=(1, 2))


def rom_solve(a_q: np.ndarray, b_q: np.ndarray, seq: str, lo_rad: np.ndarray, hi_rad: np.ndarray,
              sigma_rad: float, grid: np.ndarray = GRID_RAD) -> RomFit:  # fmt: skip
    """The range-of-motion window estimate (moment-matched von Mises of exp(-cost/(2 sigma²)))."""
    n = int(a_q.shape[0])
    if n == 0:
        return RomFit(0.0, 0.0, 0.0, 0.0, 0)
    cost = rom_cost(rom_angles(*rom_terms(a_q, b_q), seq, grid), np.asarray(lo_rad), np.asarray(hi_rad))
    cmin = float(cost.min())
    excess = cost - cmin
    frac = float(np.mean(excess > 1e-12))
    if frac == 0.0:  # flat: every delta of the grid fits the box equally well
        return RomFit(0.0, 0.0, cmin, 0.0, n)
    w = np.exp(-excess / (2.0 * sigma_rad**2))
    z = np.sum(w * np.exp(1j * grid)) / np.sum(w)
    return RomFit(float(wrap_rad(np.angle(z))), float(vm_kappa_from_resultant(min(abs(z), 1.0))), cmin, frac, n)


def joint_angles_rad(seg_q: np.ndarray, joint: int) -> np.ndarray:
    """True Cardan angles (N, 3) of joint ``joint`` from segment orientations (N, 15, 4): angles(s_p^-1·s_c)."""
    jt = JOINTS[joint]
    rel = quat_mul(quat_conj(seg_q[:, jt.parent]), seg_q[:, jt.child])
    return matrix_to_cardan_rad(quat_to_matrix(rel), CARDAN_SEQ[jt.name])


def fit_limits_deg(angles_rad: np.ndarray, percentiles=(0.5, 99.5)) -> list[list[float]]:
    """Omega: per Cardan angle, the [lo, hi] percentiles (degrees) of the training subjects' true angles (N, 3)."""
    a = np.rad2deg(np.asarray(angles_rad, dtype=np.float64))
    lo, hi = np.percentile(a, percentiles[0], axis=0), np.percentile(a, percentiles[1], axis=0)
    return [[round(float(x), 3), round(float(y), 3)] for x, y in zip(lo, hi, strict=True)]
