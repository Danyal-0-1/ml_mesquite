"""Relative-heading targets and the outward heading correction (brief §4.4).

Target for joint (parent p, child c)::

    delta_pc(t) = wrap(psi_c(t) - psi_p(t))

Correction, outward from the pelvis::

    c_root = 0,   c_c = c_p + delta_hat_pc,   q~_c = Y(c_c) · q_c

With exact estimates every corrected pod becomes ``Y(-psi_root) · s_i · m_i``: all pods share the
pelvis pod's (unknown) heading. Arrays put pods on the last axis (length 15, by pod id) and
joints on the last axis (length 14, in :data:`mesqsim.body.JOINTS` order).
"""

from __future__ import annotations

import numpy as np

from .body import JOINTS, N_JOINTS, N_PODS, ROOT_POD
from .conventions import quat_mul, wrap_rad, yaw_quat

__all__ = ["relative_heading_rad", "chain_corrections_rad", "apply_corrections"]


def relative_heading_rad(psi_rad) -> np.ndarray:
    """Per-joint targets ``delta_pc = wrap(psi_c - psi_p)`` from per-pod headings ``psi_rad`` (..., 15)."""
    psi = np.asarray(psi_rad, dtype=np.float64)
    if psi.shape[-1] != N_PODS:
        raise ValueError(f"last axis must hold {N_PODS} pods, got {psi.shape}")
    parent = np.array([j.parent for j in JOINTS])
    child = np.array([j.child for j in JOINTS])
    return wrap_rad(psi[..., child] - psi[..., parent])


def chain_corrections_rad(delta_hat_rad) -> np.ndarray:
    """Per-pod corrections ``c`` (..., 15) from per-joint estimates ``delta_hat_rad`` (..., 14).

    ``c_root = 0`` and ``c_child = c_parent + delta_hat``, accumulated outward from the pelvis.
    """
    d = np.asarray(delta_hat_rad, dtype=np.float64)
    if d.shape[-1] != N_JOINTS:
        raise ValueError(f"last axis must hold {N_JOINTS} joints, got {d.shape}")
    c = np.full((*d.shape[:-1], N_PODS), np.nan)
    c[..., ROOT_POD] = 0.0
    for k, j in enumerate(JOINTS):  # parents come first in JOINTS
        c[..., j.child] = c[..., j.parent] + d[..., k]
    return wrap_rad(c)


def apply_corrections(q_pods, c_rad) -> np.ndarray:
    """Corrected reports ``q~_i = Y(c_i) · q_i`` for quaternions (..., 15, 4) and corrections (..., 15)."""
    return quat_mul(yaw_quat(c_rad), q_pods)
