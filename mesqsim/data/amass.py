"""AMASS motion → orientations of the 15 Mesquite segments (brief §5.1).

AMASS stores, per sequence, ``poses`` (N, 156) for SMPL+H (or (N, 165) for SMPL-X): axis-angle rotations,
three numbers per joint. The first 66 numbers are the root (pelvis) and the 21 body joints of the SMPL tree,
identical in SMPL+H and SMPL-X. Global rotations compose down the tree::

    R_0 = Exp(root),   R_j = R_parent(j) · Exp(local_j)

R_j maps the canonical SMPL body frame of the bone that starts at joint j (x: subject's left, y: up,
z: forward, at the rest pose) to the AMASS world frame. The Mesquite segment orientation s_i is the global
rotation of one SMPL joint (:data:`POD_TO_SMPL`). At the rest pose (all local rotations zero, a T-pose with
arms horizontal) every segment equals R_0.

AMASS worlds are +Z up (:data:`AMASS_WORLD_UP`); :func:`estimate_world_up` checks it from the data and
``tests/test_amass.py`` asserts it on the downloaded files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..body import N_PODS, PODS
from ..conventions import exp_map, quat_mul, quat_normalize, quat_rotate

__all__ = [
    "SMPL_JOINT_NAMES",
    "SMPL_PARENTS",
    "POD_TO_SMPL",
    "AMASS_WORLD_UP",
    "AmassSequence",
    "load_sequence",
    "global_rotations",
    "segment_orientations",
    "estimate_world_up",
    "knee_flexion_rad",
    "hip_flexion_rad",
]

SMPL_JOINT_NAMES = (
    "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
    "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)  # fmt: skip
SMPL_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
N_BODY_JOINTS = 22

SPINE3, SPINE2 = 9, 6
_POD_TO_SMPL_BY_NAME = {
    "Head": 15,
    "Spine": SPINE3,  # chest: spine3 (brief §5.1); spine2 is the documented alternative
    "HipsAlt": 0,
    "LeftArm": 16,
    "LeftForeArm": 18,
    "LeftHand": 20,
    "RightArm": 17,
    "RightForeArm": 19,
    "RightHand": 21,
    "LeftUpLeg": 1,
    "LeftLeg": 4,
    "LeftFoot": 7,
    "RightUpLeg": 2,
    "RightLeg": 5,
    "RightFoot": 8,
}
POD_TO_SMPL = tuple(_POD_TO_SMPL_BY_NAME[p.name] for p in PODS)
"""SMPL joint whose global rotation is pod i's segment orientation, indexed by pod id."""

AMASS_WORLD_UP = np.array([0.0, 0.0, 1.0])
"""AMASS world up axis, verified from the data by :func:`estimate_world_up`. Equal to our +Z, so no conversion."""

SMPL_BODY_UP = np.array([0.0, 1.0, 0.0])  # canonical SMPL body frame: y up at the rest pose
SMPL_BODY_LEFT = np.array([1.0, 0.0, 0.0])  # x points to the subject's left


@dataclass
class AmassSequence:
    poses: np.ndarray  # (N, 66) axis-angle, radians: root then 21 body joints
    trans: np.ndarray  # (N, 3) root translation, metres, AMASS world
    betas: np.ndarray  # shape coefficients
    gender: str
    fps: float  # frames per second
    source: str  # file path or description

    @property
    def n_frames(self) -> int:
        return self.poses.shape[0]

    @property
    def t_s(self) -> np.ndarray:
        return np.arange(self.n_frames) / self.fps


def load_sequence(path: str | Path) -> AmassSequence:
    """Load one AMASS ``.npz`` (SMPL+H ``*_poses.npz`` or SMPL-X ``*_stageii.npz``).

    Handles both frame-rate keys (``mocap_framerate`` in SMPL+H releases, ``mocap_frame_rate`` in SMPL-X).
    Raises ``ValueError`` for files without motion (AMASS ``shape.npz`` files hold only betas and gender).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        keys = set(z.files)
        if "poses" not in keys:
            raise ValueError(f"{path.name}: no 'poses' key (keys: {sorted(keys)}); not a motion file")
        poses = np.asarray(z["poses"], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] < 3 * N_BODY_JOINTS:
            raise ValueError(f"{path.name}: poses shape {poses.shape}, need (N, >= 66)")
        for key in ("mocap_framerate", "mocap_frame_rate"):
            if key in keys:
                fps = float(z[key])
                break
        else:
            raise ValueError(f"{path.name}: no frame-rate key")
        trans = np.asarray(z["trans"], dtype=np.float64) if "trans" in keys else np.zeros((poses.shape[0], 3))
        betas = np.asarray(z["betas"], dtype=np.float64) if "betas" in keys else np.zeros(16)
        gender = "neutral"
        if "gender" in keys:  # stored as a 0-d unicode or bytes array depending on the release
            g = z["gender"].item() if z["gender"].ndim == 0 else z["gender"][0]
            gender = g.decode() if isinstance(g, bytes) else str(g)
    return AmassSequence(poses[:, : 3 * N_BODY_JOINTS], trans, betas, gender, fps, str(path))


def global_rotations(poses_66) -> np.ndarray:
    """Global rotations (N, 22, 4) of the 22 SMPL body joints from axis-angle poses (N, 66)."""
    local = exp_map(np.asarray(poses_66, dtype=np.float64).reshape(-1, N_BODY_JOINTS, 3))
    glob = np.empty_like(local)
    for j, parent in enumerate(SMPL_PARENTS):
        glob[:, j] = local[:, j] if parent < 0 else quat_mul(glob[:, parent], local[:, j])
    return quat_normalize(glob)


def segment_orientations(poses_66, spine_joint: int = SPINE3) -> np.ndarray:
    """Segment orientations s_i (N, 15, 4), indexed by pod id, in the AMASS (= our +Z-up) world."""
    if spine_joint not in (SPINE3, SPINE2):
        raise ValueError("spine_joint must be 9 (spine3) or 6 (spine2)")
    idx = list(POD_TO_SMPL)
    idx[1] = spine_joint
    glob = global_rotations(poses_66)
    out = glob[:, idx]
    assert out.shape[1] == N_PODS
    return out


def estimate_world_up(root_quats) -> tuple[np.ndarray, float]:
    """World axis that the pelvis's body up axis (SMPL +y) points along, on average.

    Returns ``(axis, mean_cosine)``: ``axis`` is the signed world basis vector closest to the mean of
    R_0·e_y over all frames, and ``mean_cosine`` how well it aligns (1 = always exactly upright).
    """
    up = quat_rotate(root_quats, np.broadcast_to(SMPL_BODY_UP, (*np.shape(root_quats)[:-1], 3)))
    mean = up.reshape(-1, 3).mean(axis=0)
    k = int(np.argmax(np.abs(mean)))
    axis = np.zeros(3)
    axis[k] = np.sign(mean[k])
    return axis, float(np.mean(up.reshape(-1, 3) @ axis))


def _relative(parent_q, child_q) -> np.ndarray:
    from ..conventions import quat_conj

    return quat_mul(quat_conj(parent_q), child_q)


def knee_flexion_rad(seg_q, side: str) -> np.ndarray:
    """Knee flexion (radians, positive = bent) from segment orientations (N, 15, 4).

    The knee rotation thigh⁻¹·shin is decomposed as a rotation about the SMPL lateral axis (x); flexion moves
    the shin's down axis backward, i.e. a positive rotation about +x: ``atan2(R[2,1], R[1,1])``.
    """
    from ..conventions import quat_to_matrix

    thigh, shin = (9, 10) if side == "left" else (12, 13)
    r = quat_to_matrix(_relative(seg_q[:, thigh], seg_q[:, shin]))
    return np.arctan2(r[:, 2, 1], r[:, 1, 1])


def amass_dir() -> Path:
    """``<data dir>/amass``: archives in ``archives/``, extracted subsets as ``<Subset>/<subject>/<seq>.npz``."""
    from ..paths import data_dir

    return data_dir() / "amass"


def body_model_dir() -> Path:
    """``<data dir>/body_models/smplh``: the SMPL+H ``.npz`` model files (male, female, neutral)."""
    from ..paths import data_dir

    return data_dir() / "body_models" / "smplh"


def motion_files(root: Path | None = None) -> list[Path]:
    """Every AMASS motion file under ``root`` (``*_poses.npz`` or ``*_stageii.npz``; ``shape.npz`` excluded)."""
    root = amass_dir() if root is None else root
    if not root.is_dir():
        return []
    files = [p for p in root.rglob("*.npz") if p.name.endswith(("_poses.npz", "_stageii.npz"))]
    return sorted(p for p in files if "archives" not in p.relative_to(root).parts)


def subset_subject(path: Path, root: Path | None = None) -> tuple[str, str]:
    """``(subset, subject)`` of a motion file laid out as ``<root>/<Subset>/<subject>/.../<seq>.npz``."""
    rel = path.relative_to(amass_dir() if root is None else root).parts
    if len(rel) < 3:
        raise ValueError(f"{path}: expected <Subset>/<subject>/<file>")
    return rel[0], rel[1]


def find_walking_sequence(root: Path | None = None) -> Path | None:
    """First motion file whose name mentions walking, or ``None`` when AMASS is absent."""
    for p in motion_files(root):
        if "walk" in p.name.lower() and "totalcapture" not in str(p).lower():
            return p
    return None


def hip_flexion_rad(seg_q, side: str) -> np.ndarray:
    """Hip flexion (radians, positive = thigh forward) relative to the pelvis, about the SMPL lateral axis."""
    from ..conventions import quat_to_matrix

    thigh = 9 if side == "left" else 12
    r = quat_to_matrix(_relative(seg_q[:, 2], seg_q[:, thigh]))
    return -np.arctan2(r[:, 2, 1], r[:, 1, 1])
