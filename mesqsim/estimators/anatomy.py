"""Anatomical joint models, written in SMPL segment frames (Step 2 brief §5.4-§5.6).

Every Mesquite segment frame is SMPL's canonical body frame of that bone (``mesqsim.data.amass``): at the rest pose
(the T-pose, arms horizontal, palms down) every segment frame equals the root's, with **x = the subject's left,
y = up, z = forward**. A joint rotation is ``s_p^-1 · s_c`` (child segment frame expressed in the parent's).

Documented choices (the brief leaves them to us):

* **Knee hinge:** the flexion axis is the thigh's left-right axis at the rest pose, x, in both segments (SMPL flexes
  the knee by a positive rotation about +x). One physical line, so ``s_thigh·e_x = s_shin·e_x`` for a perfect hinge.
* **2-DoF joints:** a universal joint ``s_p^-1·s_c = R(j1, a)·R(j2, b)`` with j1 fixed in the parent and j2 fixed in
  the child, at a constant angle gamma = 90° (Laidig et al. 2019):

  | joint | j1 (parent) | j2 (child) | first, second rotation |
  |---|---|---|---|
  | elbow | y: flexion axis, vertical at the rest pose | x: the forearm's long axis | flexion, pronation/supination |
  | wrist | z: the forearm's forward axis | y: the hand's up axis | flexion/extension, radial/ulnar deviation |
  | ankle | x: the shin's left-right axis | z: the foot's long axis | dorsi/plantarflexion, inversion/eversion |

* **Cardan sequences** (intrinsic, SMPL axes; :func:`mesqsim.conventions.matrix_to_cardan_rad`), one per joint:
  - lower back, neck, hips: ``XZY``, the ISB order Z-X-Y of Wu et al. (2002) - flexion/extension about the lateral
    axis, then lateral bending/abduction about the floating forward axis, then axial rotation about the long axis -
    rewritten in SMPL axes (ISB Z = right = -SMPL x, X = anterior = SMPL z, Y = superior = SMPL y);
  - shoulders: ``ZYX``: elevation in the frontal plane (z), horizontal flexion (y), then axial rotation about the
    humerus (x, its long axis at the rest pose). ISB recommends Y-X-Y for the humerus with the arm hanging down;
    SMPL's rest pose has it horizontal, so a Cardan sequence that is regular at the rest pose is used instead
    (gimbal lock only with the arm pointing straight forward or backward);
  - 2-DoF joints: the sequence of their universal-joint model (elbow ``YXZ``, wrist ``ZYX``, ankle ``XZY``), so the
    third angle is the rotation the 2-DoF model ignores;
  - knees: ``XZY`` (flexion, ab/adduction, axial rotation), used to report how hinge-like the knee is.
"""

from __future__ import annotations

import numpy as np

from ..body import JOINT_BY_NAME, JOINTS

__all__ = ["E_X", "E_Y", "E_Z", "HINGE_AXES", "TWODOF_AXES", "CARDAN_SEQ", "joint_index"]

E_X, E_Y, E_Z = np.eye(3)

HINGE_AXES: dict[str, tuple[np.ndarray, np.ndarray]] = {
    "l_knee": (E_X, E_X),
    "r_knee": (E_X, E_X),
}
"""Hinge axis (in the parent segment frame, in the child segment frame) of every hinge joint."""

TWODOF_AXES: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
    "l_elbow": (E_Y, E_X, np.pi / 2),
    "r_elbow": (E_Y, E_X, np.pi / 2),
    "l_wrist": (E_Z, E_Y, np.pi / 2),
    "r_wrist": (E_Z, E_Y, np.pi / 2),
    "l_ankle": (E_X, E_Z, np.pi / 2),
    "r_ankle": (E_X, E_Z, np.pi / 2),
}
"""(j1 in the parent segment frame, j2 in the child segment frame, gamma = angle between them) per 2-DoF joint."""

CARDAN_SEQ: dict[str, str] = {
    "lower_back": "XZY", "neck": "XZY", "l_hip": "XZY", "r_hip": "XZY",
    "l_shoulder": "ZYX", "r_shoulder": "ZYX",
    "l_elbow": "YXZ", "r_elbow": "YXZ", "l_wrist": "ZYX", "r_wrist": "ZYX", "l_ankle": "XZY", "r_ankle": "XZY",
    "l_knee": "XZY", "r_knee": "XZY",
}  # fmt: skip
"""One documented intrinsic Cardan sequence per joint, in SMPL segment axes."""


def joint_index(name: str) -> int:
    """Column of joint ``name`` in every (..., 14) array."""
    return [j.name for j in JOINTS].index(JOINT_BY_NAME[name].name)


def _check() -> None:
    names = {j.name for j in JOINTS}
    assert set(CARDAN_SEQ) == names, "every joint needs a Cardan sequence"
    assert set(HINGE_AXES) == {j.name for j in JOINTS if j.jtype == "hinge"}
    assert set(TWODOF_AXES) == {j.name for j in JOINTS if j.jtype == "2dof"}


_check()
