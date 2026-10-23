"""True pod orientations P_i(t) = s_i(t) · m_i (pod -> world) at arbitrary true times, phase by phase."""

from __future__ import annotations

import numpy as np

from ..conventions import quat_mul, yaw_quat
from .motion import MotionSource, strap_path
from .session import SessionPlan, tpose_segment_q

__all__ = ["box_pod_q", "worn_tpose_pod_q", "pod_orientation"]


def box_pod_q(plan: SessionPlan, pod: int) -> np.ndarray:
    """Pod lying flat in its box slot: Y(box heading + slot error)."""
    return yaw_quat(plan.box_heading_rad + plan.pods[pod].box_slot_rad)


def worn_tpose_pod_q(plan: SessionPlan, pod: int) -> np.ndarray:
    """Pod strapped on, wearer in the T-pose: s_i(T) · m_i."""
    return quat_mul(tpose_segment_q(plan, pod), np.asarray(plan.pods[pod].mount_q))


def pod_orientation(plan: SessionPlan, motion: MotionSource, pod: int, t_s) -> np.ndarray:
    """P_i(t) for true times ``t_s`` (N,), anywhere in the session.

    The idle interval is never recorded, but a frame sent just after it can carry a sample taken just before
    the motion starts. So the wearer holds the T-pose for the first half of the idle interval and the motion's
    first pose for the second half; the jump between them is never observed.
    """
    t = np.asarray(t_s, dtype=np.float64)
    out = np.full((*t.shape, 4), np.nan)
    p = plan.pods[pod]
    box_end = plan.phases["box"][1]
    tpose_end = plan.phases["tpose"][1]
    m0, m1 = plan.phases["motion"]
    mid_idle = 0.5 * (tpose_end + m0)
    q_box = box_pod_q(plan, pod)
    q_worn = worn_tpose_pod_q(plan, pod)

    pre = (t >= 0) & (t < mid_idle)  # box, strapping, T-pose: the box pose before w0, worn after w1
    out[pre] = strap_path(t[pre], *p.strap_window_s, q_box, q_worn, p.strap_spins)
    out[(t >= 0) & (t < box_end)] = q_box
    mot = (t >= mid_idle) & (t <= m1) & ~pre  # times before m0 clamp to the motion's first frame
    if np.any(mot):
        seg = quat_mul(yaw_quat(plan.facing_motion_rad), motion.at(t[mot] - m0, pod))
        out[mot] = quat_mul(seg, np.asarray(p.mount_q))
    if np.any(np.isnan(out[..., 3])):
        raise ValueError("pod orientation requested inside the idle interval or outside the session")
    return out
