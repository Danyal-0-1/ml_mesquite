"""The pods' and the phone's true poses on a moving SMPL+H body (Step 3 brief §4.2).

For one sequence: the site vertices and their neighbours are skinned (:func:`mesqsim.twin.smplh.skin`), each site's
outward normal and bone give its surface frame F (:func:`mesqsim.twin.sites.surface_frames`), and the pod's (or the
phone's) orientation without strap error is F · M. Positions are the site vertices. The segment orientation of
Step 1 (the global rotation of the site's proximal joint) comes along for the ``pod_frame: segment`` ablation and for
comparing the two.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .sites import MOUNT_CHIP, MOUNT_PHONE, SITE_VERTEX, SITES, face_rings, surface_frames
from .smplh import SmplhModel, SmplhSequence, forward, load_model, rest_pose, skin, vertex_normals

__all__ = ["BodyTracks", "body_tracks", "skin_against_bone_deg"]


@dataclass
class BodyTracks:
    t_s: np.ndarray  # (N,) the sequence's frame times
    names: tuple[str, ...]  # site names, SITES order (15 pods, then "Phone")
    q: np.ndarray  # (N, S, 4) pod (or phone device) -> world: F · M, no strap error
    p: np.ndarray  # (N, S, 3) site positions, metres, AMASS world (+Z up)
    q_frame: np.ndarray  # (N, S, 4) the surface frame F itself
    q_segment: np.ndarray  # (N, S, 4) global rotation of each site's proximal joint (Step 1's segment frame)


def body_tracks(seq: SmplhSequence, model: SmplhModel | None = None) -> BodyTracks:
    model = model or load_model(seq.gender)
    v_shaped, j_rest = rest_pose(model, seq.betas)
    fk = forward(model, seq.poses, j_rest)
    rot, pos = fk
    site_idx = [SITE_VERTEX[s.name] for s in SITES]
    rings = face_rings(model, site_idx)
    needed = sorted({int(v) for f in np.concatenate(list(rings.values())) for v in model.faces[f]} | set(site_idx))
    posed = skin(model, seq.poses, v_shaped, j_rest, needed, trans=seq.trans, fk=fk)
    by_vertex = {v: posed[:, k] for k, v in enumerate(needed)}
    normals = vertex_normals(model, by_vertex, site_idx, rings)  # (N, S, 3)
    n, s_count = seq.poses.shape[0], len(SITES)
    q = np.empty((n, s_count, 4))
    q_frame = np.empty((n, s_count, 4))
    q_seg = np.empty((n, s_count, 4))
    p = np.empty((n, s_count, 3))
    for k, site in enumerate(SITES):
        a, b = site.bone
        axis = rot[:, b, :, 1] if site.axis_up else pos[:, b] - pos[:, a]  # the head: its joint's own +y
        frame = Rotation.from_matrix(surface_frames(normals[:, k], axis))
        mount = MOUNT_PHONE if site.name == "Phone" else MOUNT_CHIP[site.name]
        q_frame[:, k] = frame.as_quat()
        q[:, k] = (frame * Rotation.from_quat(mount)).as_quat()
        q_seg[:, k] = Rotation.from_matrix(rot[:, b if site.axis_up else a]).as_quat()
        p[:, k] = by_vertex[SITE_VERTEX[site.name]]
    return BodyTracks(seq.t_s, tuple(s.name for s in SITES), q, p, q_frame, q_seg)


def skin_against_bone_deg(tracks: BodyTracks, k: int) -> np.ndarray:
    """How far site ``k``'s surface frame turns against its bone: the angle (N,) of E(t) = s(t)⁻¹ · F(t) away from
    its mean over the sequence (constant, so 0, if the skin moved rigidly with the bone)."""
    e = Rotation.from_quat(tracks.q_segment[:, k]).inv() * Rotation.from_quat(tracks.q_frame[:, k])
    return np.degrees((e * e.mean().inv()).magnitude())
