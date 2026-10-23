"""Where the pods truly are: the motion source and each pod's true orientation P_i(t) (pod -> world).

* Box: P_i = Y(eps_i) · B, pods lying flat with aligned axes (B = Y(box heading)), eps_i the slot error.
* Strapping: each pod moves from its box pose to its worn T-pose pose during its own window, with whole
  turns about the vertical: P = Y(2 pi n g) · slerp(P_box, P_T, g), g a smoothstep of time.
* T-pose: s_i = F_T · Exp(e_i) (SMPL rest pose rotated to the T-pose facing, plus posture error); P = s_i · m_i.
* Motion: s_i(t) = Y(phi_m) · S_i(t - t_motion), the AMASS (or synthetic) segment orientations S_i turned to the
  motion facing; P = s_i · m_i.

Segment frames are SMPL's canonical body frames (x: subject's left, y: up, z: forward); m_i maps pod axes to
segment axes. The idle interval has no orientation: nothing is recorded there.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..body import N_PODS
from ..conventions import (
    exp_map,
    heading_rad,
    interp_quat,
    quat_conj,
    quat_from_matrix,
    quat_mul,
    quat_normalize,
    quat_slerp,
    yaw_quat,
)
from ..data import amass

__all__ = ["UPRIGHT", "NOMINAL_MOUNT", "MotionSource", "facing_heading_rad", "strap_path"]

UPRIGHT = exp_map([np.pi / 2, 0.0, 0.0])
"""Rx(90°): maps the SMPL body up axis (+y) to world +Z. An upright body facing heading phi is Y(phi)·UPRIGHT."""


def _frame(x_axis, z_axis) -> np.ndarray:
    """Quaternion of the rotation whose columns are (x, z × x, z): pod axes written in segment coordinates."""
    x, z = np.asarray(x_axis, float), np.asarray(z_axis, float)
    return quat_from_matrix(np.column_stack([x, np.cross(z, x), z]))


_L, _UP, _FWD = np.eye(3)  # segment frame: x = subject's left, y = up, z = forward
NOMINAL_MOUNT = np.stack(
    [
        _frame(_UP, _FWD),  # 0 Head: on the forehead, face forward, pod x up
        _frame(_UP, _FWD),  # 1 Spine: front of the chest
        _frame(_UP, -_FWD),  # 2 HipsAlt: back of the pelvis, face backward
        _frame(_L, _UP),  # 3 LeftArm: outer side (up in T-pose), pod x toward the hand
        _frame(_L, _UP),  # 4 LeftForeArm
        _frame(_L, _UP),  # 5 LeftHand: back of the hand
        _frame(-_L, _UP),  # 6 RightArm
        _frame(-_L, _UP),  # 7 RightForeArm
        _frame(-_L, _UP),  # 8 RightHand
        _frame(-_UP, _FWD),  # 9 LeftUpLeg: front of the thigh, pod x down the bone
        _frame(-_UP, _FWD),  # 10 LeftLeg: front of the shin
        _frame(_FWD, _UP),  # 11 LeftFoot: top of the foot, pod x toward the toes
        _frame(-_UP, _FWD),  # 12 RightUpLeg
        _frame(-_UP, _FWD),  # 13 RightLeg
        _frame(_FWD, _UP),  # 14 RightFoot
    ]
)
"""Nominal placement of each pod on its segment (pod -> segment), by pod id. Label: assumption."""
assert NOMINAL_MOUNT.shape == (N_PODS, 4)


def facing_heading_rad(root_q) -> np.ndarray:
    """Facing of a body from its pelvis orientation: the heading of R·UPRIGHT⁻¹ (body left axis, horizontal)."""
    return heading_rad(quat_mul(root_q, quat_conj(UPRIGHT)))


@dataclass
class MotionSource:
    """Segment orientations S_i (N, 15, 4) in a +Z-up world at ``fps``, from AMASS or the synthetic generator."""

    seg_q: np.ndarray
    fps: float
    name: str

    @property
    def duration_s(self) -> float:
        return (self.seg_q.shape[0] - 1) / self.fps

    @property
    def t_s(self) -> np.ndarray:
        return np.arange(self.seg_q.shape[0]) / self.fps

    @classmethod
    def from_sequence(cls, seq: amass.AmassSequence, spine_joint: int = amass.SPINE3) -> MotionSource:
        return cls(amass.segment_orientations(seq.poses, spine_joint), seq.fps, seq.source)

    @classmethod
    def from_spec(cls, spec: str, spine_joint: int = amass.SPINE3) -> MotionSource:
        """``synthetic:<kind>[:<seconds>[:<seed>]]`` or a path to an AMASS ``.npz`` file."""
        if spec.startswith("synthetic:"):
            from ..synth import make_sequence

            parts = spec.split(":")
            duration = float(parts[2]) if len(parts) > 2 else 20.0
            seed = int(parts[3]) if len(parts) > 3 else 0
            return cls.from_sequence(make_sequence(parts[1], duration, 60.0, seed=seed), spine_joint)
        return cls.from_sequence(amass.load_sequence(spec), spine_joint)

    def window(self, start_s: float, length_s: float) -> MotionSource:
        """The clip from ``start_s`` lasting ``length_s`` (whole source frames; both ends included)."""
        k0 = int(round(start_s * self.fps))
        k1 = k0 + int(round(length_s * self.fps))
        if k1 >= self.seg_q.shape[0]:
            raise ValueError(f"window [{start_s}, {start_s + length_s}] s exceeds the {self.duration_s:.2f} s clip")
        return MotionSource(self.seg_q[k0 : k1 + 1].copy(), self.fps, f"{self.name}@{start_s:g}s")

    def n_windows(self, length_s: float) -> int:
        """Number of whole non-overlapping windows of ``length_s`` in the clip."""
        frames = int(round(length_s * self.fps))
        return max(0, (self.seg_q.shape[0] - 1) // frames)

    def at(self, tau_s: np.ndarray, pod: int) -> np.ndarray:
        """Slerp of segment ``pod`` to times ``tau_s`` (seconds from the first frame), clamped to the clip."""
        return interp_quat(self.t_s, self.seg_q[:, pod], tau_s)


def strap_path(t_s, w0_s: float, w1_s: float, q_box, q_worn, spins: int) -> np.ndarray:
    """Pod orientation while carried from the box to its strap: box before w0, worn after w1."""
    u = np.clip((np.asarray(t_s, float) - w0_s) / max(w1_s - w0_s, 1e-9), 0.0, 1.0)
    g = u * u * (3.0 - 2.0 * u)  # smoothstep: zero velocity at both ends
    base = quat_slerp(np.broadcast_to(q_box, (*g.shape, 4)), np.broadcast_to(q_worn, (*g.shape, 4)), g)
    return quat_normalize(quat_mul(yaw_quat(2.0 * np.pi * spins * g), base))
