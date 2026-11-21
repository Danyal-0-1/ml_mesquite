"""The twin's heading labels: one twist rule for every pod and for the phone (Step 3 brief §4.5, §4.7, §4.11).

    E(t)    = q(t) · R(t)⁻¹                                  the error rotation, world frame, canonical (E_w >= 0)
    ψ(t)    = -wrap(2 atan2(E_z, E_w))                       minus E's twist about +Z: the report's north
    tilt(t) = 2 atan2(sqrt(E_x² + E_y²), sqrt(E_z² + E_w²))  the swing that is left once the twist is removed
    δ_pc    = wrap(ψ_c - ψ_p)                                a joint's relative heading
    ε_i     = wrap(ψ_i - ψ_ph)                               pod i's heading relative to the phone

``q`` is what a pod reports (pod -> its private world) and ``R`` its true orientation (pod -> world).

**Hand check** (the brief's): E = Y(-12°) has E_z = -sin 6° and E_w = cos 6°, so ψ = 12°.

**Why the twist.** A report q = Y(-ψ) · R · Exp(η) (Step 1's model) gives E = Y(-ψ) · (R Exp(η) R⁻¹), a turn about
the vertical times a small tilt; with η = 0, E = Y(-ψ) and the rule returns ψ exactly. In general split E into a
twist T about +Z and a swing S about a horizontal axis (S_z = 0): for E = S · T, E_w = S_w T_w and E_z = S_w T_z,
and the same holds for E = T · S, so (E_z, E_w) is proportional to (T_z, T_w) in either order; the swing's angle
is 2 acos(S_w), with S_w = sqrt(E_z² + E_w²). A filter whose tilt correction is a rotation about a horizontal axis
therefore leaves ψ alone to first order in its tilt error.

``ψ`` is returned in (-π, π], so the brief's -wrap(·) is used except at exactly ±π, where it reads +π.
"""

from __future__ import annotations

import numpy as np

from ..conventions import as_quat, quat_canonical, quat_conj, quat_mul, wrap_rad

__all__ = ["error_rotation", "heading_error_rad", "tilt_error_rad", "report_labels", "delta_rad", "epsilon_rad"]


def error_rotation(q_report, r_true) -> np.ndarray:
    """E = q · R⁻¹, canonical (w >= 0), for reports ``q`` and true orientations ``R`` (both ``(..., 4)``)."""
    q, r = np.broadcast_arrays(as_quat(q_report), as_quat(r_true))
    return quat_canonical(quat_mul(q, quat_conj(r)))


def heading_error_rad(e) -> np.ndarray:
    """ψ = -wrap(2 atan2(E_z, E_w)) of error rotations ``e`` ``(..., 4)``, in (-π, π]."""
    e = as_quat(e)
    return wrap_rad(-2.0 * np.arctan2(e[..., 2], e[..., 3]))


def tilt_error_rad(e) -> np.ndarray:
    """The angle in [0, π] of E's swing about a horizontal axis, once its twist about +Z is removed."""
    e = as_quat(e)
    return 2.0 * np.arctan2(np.hypot(e[..., 0], e[..., 1]), np.hypot(e[..., 2], e[..., 3]))


def report_labels(q_report, r_true) -> tuple[np.ndarray, np.ndarray]:
    """(ψ, tilt) in radians of reports against the truth."""
    e = error_rotation(q_report, r_true)
    return heading_error_rad(e), tilt_error_rad(e)


def delta_rad(psi_child, psi_parent) -> np.ndarray:
    """δ_pc = wrap(ψ_c - ψ_p)."""
    return wrap_rad(np.asarray(psi_child, float) - np.asarray(psi_parent, float))


def epsilon_rad(psi_pod, psi_phone) -> np.ndarray:
    """ε_i = wrap(ψ_i - ψ_ph): pod i's heading relative to the phone's, the target of TCN-P."""
    return wrap_rad(np.asarray(psi_pod, float) - np.asarray(psi_phone, float))
