"""SMPL+H in numpy, for a handful of vertices: each sequence's own shape and gender (Step 3 brief §4.2).

The model files are the licensed SMPL+H release (``$MESQ_DATA_DIR/body_models/smplh/<gender>/model.npz``;
``manifests/smplh.csv``): 6,890 vertices, 52 joints (22 body, 15 per hand), 16 shape and 459 pose-corrective blend
shapes. The steps are those of ``smplx.lbs.lbs`` (Loper et al. 2015; Romero et al. 2017), in the same order:

1. shape: v_shaped = v_template + shapedirs · β; rest joints J = J_regressor · v_shaped;
2. pose correctives: v_posed = v_shaped + posedirs · vec(R_j - I, j = 1..51);
3. forward kinematics: G_j = G_parent · [R_j | J_j - J_parent], posed joints = G_j's translation;
4. skinning: v = Σ_j w_vj (R_j^G v_posed + t_j^G - R_j^G J_j) + trans.

Only the vertices asked for are skinned, with each vertex's (at most four, in this model) non-zero weights, so a
ten-minute session of a dozen pod sites costs seconds. Frames: the AMASS world, +Z up, metres.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..paths import data_dir

__all__ = ["N_JOINTS", "SmplhModel", "SmplhSequence", "load_model", "load_smplh_sequence", "rest_pose",
           "forward", "skin", "vertex_normals"]

N_JOINTS = 52


@dataclass
class SmplhModel:
    gender: str
    v_template: np.ndarray  # (V, 3)
    shapedirs: np.ndarray  # (V, 3, 16)
    posedirs: np.ndarray  # (V, 3, 459)
    j_regressor: np.ndarray  # (52, V)
    weights: np.ndarray  # (V, 52)
    parents: np.ndarray  # (52,), -1 for the root
    faces: np.ndarray  # (F, 3)


@dataclass
class SmplhSequence:
    poses: np.ndarray  # (N, 156) axis-angle, radians: root, 21 body joints, 30 hand joints (zeros if absent)
    trans: np.ndarray  # (N, 3) metres, AMASS world
    betas: np.ndarray  # (16,)
    gender: str
    fps: float
    source: str

    @property
    def t_s(self) -> np.ndarray:
        return np.arange(self.poses.shape[0]) / self.fps


def load_model(gender: str, root: Path | None = None) -> SmplhModel:
    root = root or data_dir() / "body_models" / "smplh"
    g = gender if gender in ("male", "female") else "neutral"
    with np.load(Path(root) / g / "model.npz", allow_pickle=False) as z:
        parents = z["kintree_table"][0].astype(np.int64)
        parents[0] = -1
        return SmplhModel(g, z["v_template"].astype(np.float64), z["shapedirs"].astype(np.float64),
                          z["posedirs"].astype(np.float64), z["J_regressor"].astype(np.float64),
                          z["weights"].astype(np.float64), parents, z["f"].astype(np.int64))


def load_smplh_sequence(path) -> SmplhSequence:
    """An AMASS SMPL+H ``*_poses.npz`` with its hands (Step 1's loader keeps the 22 body joints only)."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        poses = np.asarray(z["poses"], dtype=np.float64)
        full = np.zeros((poses.shape[0], 3 * N_JOINTS))
        full[:, : min(poses.shape[1], 3 * N_JOINTS)] = poses[:, : 3 * N_JOINTS]
        fps = float(z["mocap_framerate"] if "mocap_framerate" in z.files else z["mocap_frame_rate"])
        g = z["gender"].item() if "gender" in z.files and z["gender"].ndim == 0 else "neutral"
        g = g.decode() if isinstance(g, bytes) else str(g)
        betas = np.zeros(16)
        b = np.asarray(z["betas"], dtype=np.float64).ravel() if "betas" in z.files else np.zeros(0)
        betas[: min(16, b.size)] = b[:16]
        trans = np.asarray(z["trans"], dtype=np.float64) if "trans" in z.files else np.zeros((poses.shape[0], 3))
    return SmplhSequence(full, trans, betas, g, fps, str(path))


def rest_pose(model: SmplhModel, betas) -> tuple[np.ndarray, np.ndarray]:
    """Shaped template (V, 3) and rest joints (52, 3) for shape ``betas``."""
    b = np.zeros(model.shapedirs.shape[-1])
    b[: min(b.size, np.size(betas))] = np.ravel(betas)[: b.size]
    v = model.v_template + model.shapedirs @ b
    return v, model.j_regressor @ v


def forward(model: SmplhModel, poses, joints_rest) -> tuple[np.ndarray, np.ndarray]:
    """Global joint rotations (N, 52, 3, 3) and posed joint positions (N, 52, 3) (without ``trans``)."""
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, N_JOINTS, 3)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(poses.shape[0], N_JOINTS, 3, 3)
    rot = np.empty_like(local)
    pos = np.empty((poses.shape[0], N_JOINTS, 3))
    for j, p in enumerate(model.parents):
        if p < 0:
            rot[:, j], pos[:, j] = local[:, j], joints_rest[j]
        else:
            rot[:, j] = rot[:, p] @ local[:, j]
            pos[:, j] = pos[:, p] + np.einsum("nab,b->na", rot[:, p], joints_rest[j] - joints_rest[p])
    return rot, pos


def skin(model: SmplhModel, poses, v_shaped, joints_rest, idx, trans=None, fk=None) -> np.ndarray:
    """Posed positions (N, len(idx), 3) of vertices ``idx``, pose correctives included."""
    idx = np.asarray(idx, dtype=np.int64)
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, N_JOINTS, 3)
    rot, pos = fk if fk is not None else forward(model, poses, joints_rest)
    local = Rotation.from_rotvec(poses[:, 1:].reshape(-1, 3)).as_matrix().reshape(poses.shape[0], -1, 9)
    feature = (local - np.eye(3).ravel()).reshape(poses.shape[0], -1)  # (N, 459), smplx's pose_feature
    offsets = feature @ model.posedirs[idx].reshape(-1, model.posedirs.shape[-1]).T  # one BLAS product
    v_posed = v_shaped[idx][None] + offsets.reshape(poses.shape[0], idx.size, 3)
    out = np.zeros((poses.shape[0], idx.size, 3))
    w = model.weights[idx]
    for k in range(idx.size):
        for j in np.flatnonzero(w[k] > 0):
            r, t = rot[:, j], pos[:, j] - np.einsum("nab,b->na", rot[:, j], joints_rest[j])
            out[:, k] += w[k, j] * (np.einsum("nab,nb->na", r, v_posed[:, k]) + t)
    if trans is not None:
        out += np.asarray(trans)[:, None, :]
    return out


def vertex_normals(model: SmplhModel, verts: np.ndarray, idx, ring: dict[int, np.ndarray]) -> np.ndarray:
    """Outward unit normals (N, len(idx), 3) of vertices ``idx``: the area-weighted sum of their faces' normals.

    ``verts`` maps every vertex the faces touch (``ring[v]`` lists the faces around v) to (N, 3) positions:
    a dict ``{vertex: (N, 3)}``.
    """
    out = np.zeros((next(iter(verts.values())).shape[0], len(idx), 3))
    for k, v in enumerate(idx):
        for f in ring[int(v)]:
            a, b, c = (verts[int(i)] for i in model.faces[f])
            out[:, k] += np.cross(b - a, c - a)
    return out / np.linalg.norm(out, axis=-1, keepdims=True)
