"""The heading of each pod's private frame, psi_i(t) (brief §4.2).

    psi_i(t) = a_i + b_i·t + k_i·Theta_i(t) + w_i(t)    (+ a jump at each reboot)

M3 implements the offset a_i only; the drift, scale-factor, random-walk and reboot terms are added in M4.
"""

from __future__ import annotations

import numpy as np

from .session import SessionPlan

__all__ = ["psi_rad"]


def psi_rad(plan: SessionPlan, pod: int, t_s) -> np.ndarray:
    """psi_i at true times ``t_s`` (radians, not wrapped)."""
    t = np.asarray(t_s, dtype=np.float64)
    return np.full(t.shape, plan.pods[pod].offset_rad)
