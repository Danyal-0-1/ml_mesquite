"""Hinge joints (the knees): closed form, likelihood, axis identification and the sign rule (Step 2 brief §5.4).

**The closed form.** The hinge axis is one physical line h. In the parent pod's frame it is u_p, in the child's u_c,
and the reports carry each pod's private north on the left, so

    q_p·u_p = Y(-psi_p)·h,   q_c·u_c = Y(-psi_c)·h   =>   q_p·u_p = Y(delta)·q_c·u_c,   delta = psi_c - psi_p.

With a_t, b_t the horizontal (x, y) parts of q_p·u_p and q_c·u_c over a window:

    C = sum a_t·b_t,   S = sum (b_t × a_t) = sum (b_x a_y - b_y a_x),   delta_hat = atan2(S, C),   I = sqrt(C² + S²)
    sum |a_t - Y(delta)·b_t|² = sum (|a_t|² + |b_t|²) - 2·I·cos(delta - delta_hat)

With independent Gaussian noise of variance sigma² per component, the log-likelihood of delta is
const + (I/sigma²)·cos(delta - delta_hat): exactly von Mises, mean delta_hat, **kappa = I/sigma_hat²** with
sigma_hat² = RSS/(2N - 1) (floored by the estimator's belief ``geometry.hinge_sigma_floor_deg``). A vertical axis
has no horizontal part: I = 0, kappa = 0 (the blind spot). One instant with a non-vertical axis already fixes delta.

**Axis identification** (GEO-ID; Seel, Raisch & Schauer 2014). For a hinge the rate components perpendicular to the
axis are one physical vector, so ``|omega_p × u_p| = |omega_c × u_c|`` at every instant. u_p and u_c are fitted in
spherical coordinates by least squares from several starts. Rates do not see the private norths (clock.py).

**The sign.** The condition cannot tell u from -u; a wrong *relative* sign turns b into -b, so delta_hat moves by
180°. Two rules vote on it (:func:`vote_vertical`, :func:`vote_tpose`); without a vote the window is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conventions import KAPPA_MAX, geodesic_angle_rad, quat_conj, quat_mul, quat_rotate, wrap_rad, yaw_quat

__all__ = ["HingeFit", "AxisFit", "axis_in_world", "hinge_closed_form", "identify_hinge_axes", "align_sign",
           "vote_vertical", "vote_tpose", "unit_from_angles", "angles_from_unit"]  # fmt: skip


@dataclass(frozen=True)
class HingeFit:
    delta_rad: float  # delta_hat = atan2(S, C)
    kappa: float  # I / sigma_hat² (before calibration); 0 = no information
    info: float  # I = sqrt(C² + S²), the hinge information of the window
    rss: float  # residual sum of squares at delta_hat
    n: int  # pairs used


def axis_in_world(q: np.ndarray, u: np.ndarray) -> np.ndarray:
    """q·u for every orientation q (N, 4): the axis in the pod's (private-north) world frame, (N, 3)."""
    return quat_rotate(q, np.broadcast_to(np.asarray(u, dtype=np.float64), (q.shape[0], 3)))


def hinge_closed_form(a_xy: np.ndarray, b_xy: np.ndarray, sigma_floor_rad: float) -> HingeFit:
    """Least-squares delta with a_t = Y(delta)·b_t on horizontal parts (N, 2), and its von Mises concentration."""
    a = np.asarray(a_xy, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(b_xy, dtype=np.float64).reshape(-1, 2)
    n = a.shape[0]
    if n == 0:
        return HingeFit(0.0, 0.0, 0.0, 0.0, 0)
    c = float(np.sum(a * b))
    s = float(np.sum(b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]))
    info = float(np.hypot(c, s))
    rss = max(float(np.sum(a * a) + np.sum(b * b)) - 2.0 * info, 0.0)
    if info < 1e-12:
        return HingeFit(0.0, 0.0, info, rss, n)
    sigma2 = max(rss / (2 * n - 1), sigma_floor_rad**2)
    return HingeFit(float(np.arctan2(s, c)), float(min(info / sigma2, KAPPA_MAX)), info, rss, n)


# ------------------------------------------------------------------------------------------ axis identification

def unit_from_angles(phi: float, theta: float) -> np.ndarray:
    """Unit vector at elevation phi and azimuth theta (radians)."""
    return np.array([np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)])


def angles_from_unit(u: np.ndarray) -> tuple[float, float]:
    u = np.asarray(u, dtype=np.float64) / np.linalg.norm(u)
    return float(np.arcsin(np.clip(u[2], -1, 1))), float(np.arctan2(u[1], u[0]))


@dataclass(frozen=True)
class AxisFit:
    u_p: np.ndarray  # hinge axis in the parent pod frame (unit; sign arbitrary)
    u_c: np.ndarray  # hinge axis in the child pod frame
    rms_rad_s: float  # RMS of |omega_p × u_p| - |omega_c × u_c|
    rel_residual: float  # rms / RMS of the rate magnitudes: 0 for a perfect hinge without noise
    n: int  # rate pairs used


_STARTS = np.random.default_rng(20260508).normal(size=(16, 2, 3))  # fixed: identification is deterministic


def identify_hinge_axes(w_p: np.ndarray, w_c: np.ndarray, start: tuple[np.ndarray, np.ndarray] | None = None,
                        n_starts: int = 8) -> AxisFit:  # fmt: skip
    """Fit u_p, u_c with |w_p × u_p| = |w_c × u_c| at every instant (rates (N, 3) in each pod's frame, rad/s).

    Least squares in spherical coordinates from ``start`` (a previous fit) and ``n_starts`` fixed starts; the best
    solution wins. Signs are normalised by :func:`align_sign` (largest component positive) when there is no start.
    """
    from scipy.optimize import least_squares

    w_p = np.asarray(w_p, dtype=np.float64)
    w_c = np.asarray(w_c, dtype=np.float64)

    def resid(x):
        up, uc = unit_from_angles(x[0], x[1]), unit_from_angles(x[2], x[3])
        return np.linalg.norm(np.cross(w_p, up), axis=1) - np.linalg.norm(np.cross(w_c, uc), axis=1)

    x0s = []
    if start is not None:
        x0s.append([*angles_from_unit(start[0]), *angles_from_unit(start[1])])
    for k in range(n_starts):
        x0s.append([*angles_from_unit(_STARTS[k % 16, 0]), *angles_from_unit(_STARTS[k % 16, 1])])
    best = None
    for x0 in x0s:
        r = least_squares(resid, np.asarray(x0, dtype=np.float64), method="trf", x_scale=1.0, max_nfev=200)
        if best is None or r.cost < best.cost - 1e-15:
            best = r
    x = best.x
    up, uc = unit_from_angles(x[0], x[1]), unit_from_angles(x[2], x[3])
    if start is not None:
        up, uc = align_sign(up, start[0]), align_sign(uc, start[1])
    else:
        up, uc = align_sign(up), align_sign(uc)
    rms = float(np.sqrt(np.mean(resid(x) ** 2)))
    scale = float(np.sqrt(np.mean(0.5 * (np.sum(w_p**2, axis=1) + np.sum(w_c**2, axis=1)))))
    return AxisFit(up, uc, rms, rms / scale if scale > 0 else np.inf, int(w_p.shape[0]))


def align_sign(u: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """u or -u: the one pointing like ``ref``, or (no ref) the one whose largest component is positive."""
    u = np.asarray(u, dtype=np.float64)
    if ref is None:
        return u if u[np.argmax(np.abs(u))] >= 0 else -u
    return u if float(np.dot(u, ref)) >= 0 else -u


# ------------------------------------------------------------------------------------------ the relative sign

def vote_vertical(a3: np.ndarray, b3: np.ndarray, min_mean_product: float) -> int:
    """Vertical-component rule: for the true axis (q_p·u_p)_z = (q_c·u_c)_z, so sum a_z·b_z > 0.

    +1 if the mean product is at least ``min_mean_product`` (sin² of the minimum RMS elevation), -1 if it is at most
    minus that, 0 (no vote) when the axis stays too close to horizontal to tell.
    """
    if a3.shape[0] == 0:
        return 0
    v = float(np.mean(a3[:, 2] * b3[:, 2]))
    return 1 if v >= min_mean_product else (-1 if v <= -min_mean_product else 0)


def vote_tpose(q_p: np.ndarray, q_c: np.ndarray, delta_rad: float, j_tpose: np.ndarray, margin_rad: float) -> int:
    """T-pose rule: the knee cannot be turned 180° about the vertical relative to its T-pose configuration.

    ``j_tpose`` = q_p(T)^-1·Y(delta_T)·q_c(T), the pod-to-pod rotation in the T-pose with box registration's
    delta_T. For each pair, K = q_p^-1·Y(d)·q_c·j_tpose^-1 is the knee's rotation since the T-pose under the
    hypothesis d: about |flexion| for the right d, exactly 180° for d + 180° when the axis is horizontal (a half turn
    about the vertical composed with a turn about a horizontal axis). +1 if delta_rad is the better hypothesis by at
    least ``margin_rad`` in median angle, -1 if delta_rad + 180° is, 0 otherwise.
    """
    if q_p.shape[0] == 0:
        return 0
    ang = []
    for d in (delta_rad, delta_rad + np.pi):
        j = quat_mul(quat_mul(quat_conj(q_p), yaw_quat(np.full(q_p.shape[0], wrap_rad(d)))), q_c)
        ang.append(float(np.median(geodesic_angle_rad(quat_mul(j, quat_conj(j_tpose))))))
    diff = ang[1] - ang[0]
    return 1 if diff >= margin_rad else (-1 if diff <= -margin_rad else 0)
