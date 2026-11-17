"""2-DoF joints (elbows, wrists, ankles): the constraint of Laidig et al. 2019 (Step 2 brief §5.5).

The first axis j1 is fixed in the parent, the second j2 in the child, at a constant angle gamma:

    (q_p·j1) · (Y(delta)·q_c·j2) = cos(gamma)

With a = q_p·j1 and b = q_c·j2 (each pod's private-north world frame), horizontal parts a_h, b_h, vertical parts
a_z, b_z, and Y(delta) acting on the horizontal plane:

    P_t·cos(delta) + Q_t·sin(delta) + R_t = 0,   P = a_h·b_h,  Q = b_h × a_h,  R = a_z·b_z - cos(gamma)

one equation per instant. The cost ``sum_t (P_t cos d + Q_t sin d + R_t)²`` is a trigonometric polynomial of degree
two in d, fully described by five sums:

    cost(d) = A0 + A1 cos d + B1 sin d + A2 cos 2d + B2 sin 2d
    A0 = sum (P² + Q²)/2 + R²,  A1 = 2 sum P·R,  B1 = 2 sum Q·R,  A2 = sum (P² - Q²)/2,  B2 = sum P·Q

so it is evaluated on a fine grid over (-180°, 180°] and each local minimum refined by Newton steps.

* **Confidence** (Laplace approximation): kappa = cost''(delta_hat) / (2·sigma_hat²), sigma_hat² = cost/(N - 1)
  (floored by ``geometry.twodof_sigma_floor``).
* **Two minima.** A degree-two polynomial can have two local minima. When the vertical terms vanish (R_t = 0, e.g.
  gamma = 90° with one axis horizontal) the cost is symmetric under d -> d + 180° and both are equally good. The
  rule: if the second minimum's cost exceeds the best by less than ``2·sigma_hat²·llr_min`` (a likelihood ratio
  below e^llr_min), the window is **ambiguous** and kappa = 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conventions import KAPPA_MAX, wrap_rad

__all__ = ["TwoDofFit", "twodof_terms", "twodof_coefficients", "twodof_cost", "twodof_solve"]

_GRID = np.linspace(-np.pi, np.pi, 721)[1:]  # 0.5° steps over (-180°, 180°]


@dataclass(frozen=True)
class TwoDofFit:
    delta_rad: float
    kappa: float  # Laplace concentration (0 when ambiguous)
    cost: float  # cost at delta_hat
    second_cost: float  # cost at the second-best local minimum (inf if none)
    ambiguous: bool
    n: int


def twodof_terms(a3: np.ndarray, b3: np.ndarray, cos_gamma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(P, Q, R) per instant from a = q_p·j1 and b = q_c·j2, (N, 3) each."""
    p = a3[:, 0] * b3[:, 0] + a3[:, 1] * b3[:, 1]
    q = b3[:, 0] * a3[:, 1] - b3[:, 1] * a3[:, 0]
    r = a3[:, 2] * b3[:, 2] - cos_gamma
    return p, q, r


def twodof_coefficients(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """(A0, A1, B1, A2, B2) of the cost's trigonometric polynomial."""
    return np.array([np.sum((p * p + q * q) / 2 + r * r), 2 * np.sum(p * r), 2 * np.sum(q * r),
                     np.sum((p * p - q * q) / 2), np.sum(p * q)])  # fmt: skip


def twodof_cost(coef: np.ndarray, d) -> np.ndarray:
    a0, a1, b1, a2, b2 = coef
    d = np.asarray(d, dtype=np.float64)
    return a0 + a1 * np.cos(d) + b1 * np.sin(d) + a2 * np.cos(2 * d) + b2 * np.sin(2 * d)


def _derivatives(coef: np.ndarray, d: float) -> tuple[float, float]:
    _, a1, b1, a2, b2 = coef
    g = -a1 * np.sin(d) + b1 * np.cos(d) - 2 * a2 * np.sin(2 * d) + 2 * b2 * np.cos(2 * d)
    h = -a1 * np.cos(d) - b1 * np.sin(d) - 4 * a2 * np.cos(2 * d) - 4 * b2 * np.sin(2 * d)
    return float(g), float(h)


def _refine(coef: np.ndarray, d: float) -> float:
    for _ in range(20):
        g, h = _derivatives(coef, d)
        if h <= 0:
            break
        step = g / h
        d -= step
        if abs(step) < 1e-13:
            break
    return float(wrap_rad(d))


def twodof_solve(a3: np.ndarray, b3: np.ndarray, cos_gamma: float, sigma_floor: float, llr_min: float) -> TwoDofFit:
    """Minimise the 2-DoF constraint cost over delta; Laplace kappa; the two-minima ambiguity rule."""
    n = int(a3.shape[0])
    if n < 2:
        return TwoDofFit(0.0, 0.0, 0.0, np.inf, False, n)
    coef = twodof_coefficients(*twodof_terms(a3, b3, cos_gamma))
    c = twodof_cost(coef, _GRID)
    is_min = (c <= np.roll(c, 1)) & (c <= np.roll(c, -1))
    minima = sorted({round(_refine(coef, float(d)), 12) for d in _GRID[is_min]})
    if not minima:  # a constant cost (no information at all)
        return TwoDofFit(0.0, 0.0, float(c[0]), np.inf, False, n)
    costs = np.array([float(twodof_cost(coef, d)) for d in minima])
    order = np.argsort(costs)
    best = minima[order[0]]
    # distinct second minimum: more than 1° away from the best
    others = [k for k in order[1:] if abs(float(wrap_rad(minima[k] - best))) > np.deg2rad(1.0)]
    second = float(costs[others[0]]) if others else np.inf
    cost = float(costs[order[0]])
    sigma2 = max(cost / (n - 1), sigma_floor**2)
    curv = _derivatives(coef, best)[1]
    kappa = float(np.clip(curv / (2 * sigma2), 0.0, KAPPA_MAX))
    ambiguous = np.isfinite(second) and (second - cost) < 2 * sigma2 * llr_min
    return TwoDofFit(best, 0.0 if ambiguous else kappa, cost, second, bool(ambiguous), n)
