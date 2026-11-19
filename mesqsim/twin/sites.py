"""Where the pods and the phone sit on the body's skin, their surface frames and mountings (Step 3 brief §4.2).

**Placement** comes from Danyal's answer in S3-M0: the velvet wear guide of mesquite-mocap ``87d7c3b``
(``trees/velvet/wearguide/threejsscene.js:334-404``), the HipsAlt pod on the lower back, the phone on the front of
the belt. Each site is a ray in the SMPL+H rest pose (a T-pose; canonical frame x = the subject's left, y = up,
z = forward): it starts on the skeleton, a fraction of the way along the segment's bone (the wear guide's), and
points front, back or up; the site is the vertex skinned mostly to the segment that lies nearest that ray, at its
far end. The vertex index is shared by every shape of one gender.

**The surface frame** F = [t̂, b̂, n̂]: n̂ the outward vertex normal, t̂ the segment's bone (proximal to distal joint)
projected into the tangent plane, b̂ = n̂ x t̂.

**The pod** R_i(t) = F_i(t) · M_i · E_i. M_i says how the chip's axes lie in the surface frame (Danyal's answer,
`assumption` until the bench test): the chip's Z points out of the case face, so out of the skin; its X runs along
the case, which the wear guide lays along the limb pointing away from the trunk on the arms, up the thigh and shin,
up the head and chest, forward on the foot. So M is the identity except on the thighs and shins, where the case
points up and the bone down: a half turn about n̂. E_i is the strap error, uniform within ±max per axis
(``twin.body.strap_error_max_deg``). **The phone**: screen against the body, rear camera forward, upright: its
device axes are (b̂, t̂, -n̂) (x to the right, y up, z out of the screen, into the body).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .smplh import SmplhModel, rest_pose

__all__ = ["SITES", "SITE_VERTEX", "MOUNT_CHIP", "MOUNT_PHONE", "Site", "site_vertex", "face_rings",
           "surface_frames", "strap_errors"]

# SMPL+H joints: 0 pelvis, 1/2 hips, 3 spine1, 4/5 knees, 6 spine2, 7/8 ankles, 9 spine3, 10/11 feet, 12 neck,
# 13/14 collars, 15 head, 16/17 shoulders, 18/19 elbows, 20/21 wrists; 25 and 40 the middle fingers' first joints
_L, _UP, _FWD = np.eye(3)


@dataclass(frozen=True)
class Site:
    name: str
    joints: tuple[int, ...]  # the segment's joints: the vertex must be skinned mostly to one of them
    bone: tuple[int, int]  # proximal, distal joint: the segment's axis
    fraction: float  # where the ray starts, along the bone (0: proximal joint)
    lift_m: tuple[float, float, float]  # then this far (canonical frame), e.g. up the head
    direction: tuple[float, float, float]  # the ray, canonical frame
    source: str
    axis_up: bool = False  # the segment's axis is its joint's own up (canonical +y) rather than the bone


SITES = (
    Site("Head", (15,), (12, 15), 1.0, (0.0, 0.07, 0.0), tuple(_FWD), "forehead: wear guide (-2, 168, 8), front",
         axis_up=True),  # the neck -> head bone turns with the neck, not the head
    Site("Spine", (6, 9, 12), (9, 12), 0.3, (0.0, 0.0, 0.0), tuple(_FWD), "sternum: wear guide (-2, 137, 14), front"),
    Site("HipsAlt", (0, 3), (0, 3), 0.5, (0.0, 0.0, 0.0), tuple(-_FWD), "lower back (S3-M0 answer; marker 1.04 m)"),
    Site("LeftArm", (16,), (16, 18), 0.41, (0.0, 0.0, 0.0), tuple(_UP), "top of the upper arm, 0.41 along it"),
    Site("LeftForeArm", (18,), (18, 20), 0.39, (0.0, 0.0, 0.0), tuple(_UP), "top of the forearm, 0.39 along it"),
    Site("LeftHand", (20,), (20, 25), 0.35, (0.0, 0.0, 0.0), tuple(_UP), "back of the hand"),
    Site("RightArm", (17,), (17, 19), 0.41, (0.0, 0.0, 0.0), tuple(_UP), "top of the upper arm, 0.41 along it"),
    Site("RightForeArm", (19,), (19, 21), 0.39, (0.0, 0.0, 0.0), tuple(_UP), "top of the forearm, 0.39 along it"),
    Site("RightHand", (21,), (21, 40), 0.35, (0.0, 0.0, 0.0), tuple(_UP), "back of the hand"),
    Site("LeftUpLeg", (1,), (1, 4), 0.69, (0.0, 0.0, 0.0), tuple(_FWD), "front of the thigh, 0.69 down it"),
    Site("LeftLeg", (4,), (4, 7), 0.53, (0.0, 0.0, 0.0), tuple(_FWD), "front of the shin, 0.53 down it"),
    Site("LeftFoot", (7, 10), (7, 10), 0.65, (0.0, 0.0, 0.0), tuple(_UP), "top of the foot"),
    Site("RightUpLeg", (2,), (2, 5), 0.69, (0.0, 0.0, 0.0), tuple(_FWD), "front of the thigh, 0.69 down it"),
    Site("RightLeg", (5,), (5, 8), 0.53, (0.0, 0.0, 0.0), tuple(_FWD), "front of the shin, 0.53 down it"),
    Site("RightFoot", (8, 11), (8, 11), 0.65, (0.0, 0.0, 0.0), tuple(_UP), "top of the foot"),
    Site("Phone", (0, 3), (0, 3), 0.5, (0.0, 0.0, 0.0), tuple(_FWD), "front of the belt (wear guide (0, 100, 12))"),
)
"""The 15 pod sites (named by pod) and the phone's belt site. The fractions are the wear guide's positions measured
along the avatar's bones (upper arm 0.41, forearm 0.39, thigh 0.69, shin 0.53); the head's lift (7 cm) and the
feet's fraction (0.65) were chosen so the site's normal tilts as the wear guide's pod face does (head 11-13° up,
guide 20°; feet 59-61° up, guide 70°); the rest are this module's choices (`assumption`)."""

SITE_VERTEX = {"Head": 331, "Spine": 3075, "HipsAlt": 1783, "LeftArm": 1505, "LeftForeArm": 1570, "LeftHand": 2184,
               "RightArm": 4978, "RightForeArm": 5039, "RightHand": 5662, "LeftUpLeg": 1187, "LeftLeg": 1075,
               "LeftFoot": 3400, "RightUpLeg": 4671, "RightLeg": 4563, "RightFoot": 6800, "Phone": 1769}
""":func:`site_vertex` of each site on the neutral SMPL+H template, mean shape. SMPL+H numbers its vertices the same
in every gender and shape, so one table serves every subject (``tests/test_twin_body.py`` recomputes it)."""

_HALF_TURN_Z = Rotation.from_rotvec([0.0, 0.0, np.pi]).as_quat()
MOUNT_CHIP = {s.name: (_HALF_TURN_Z if s.name in ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")
                       else np.array([0.0, 0.0, 0.0, 1.0])) for s in SITES if s.name != "Phone"}
"""M_i: chip axes in the surface frame (x, y, z) = (t̂, b̂, n̂), as quaternions (x, y, z, w). Label: assumption."""
MOUNT_PHONE = Rotation.from_matrix(np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])).as_quat()
"""The phone's device axes (x right, y up, z out of the screen) in its belt site's surface frame: (b̂, t̂, -n̂)."""


def site_vertex(model: SmplhModel, site: Site, betas=None) -> int:
    """The vertex of ``site`` on the rest pose of ``model`` (mean shape unless ``betas``)."""
    v, j = rest_pose(model, np.zeros(16) if betas is None else betas)
    a, b = j[site.bone[0]], j[site.bone[1]]
    start = a + site.fraction * (b - a) + np.asarray(site.lift_m)
    d = np.asarray(site.direction, dtype=float)
    own = np.isin(np.argmax(model.weights, axis=1), site.joints)
    rel = v - start
    along = rel @ d
    off = np.linalg.norm(rel - along[:, None] * d, axis=1)
    ok = own & (along > 0)
    cand = np.flatnonzero(ok & (off < np.min(off[ok]) + 0.01))  # within 1 cm of the ray ...
    return int(cand[np.argmax(along[cand])])  # ... the outermost: on the skin


def face_rings(model: SmplhModel, idx) -> dict[int, np.ndarray]:
    """The faces around each vertex of ``idx``."""
    return {int(v): np.flatnonzero(np.any(model.faces == v, axis=1)) for v in idx}


def surface_frames(normals: np.ndarray, bone: np.ndarray) -> np.ndarray:
    """F = [t̂, b̂, n̂] as rotation matrices (N, 3, 3) from outward normals and bone vectors (N, 3)."""
    n = normals / np.linalg.norm(normals, axis=-1, keepdims=True)
    t = bone - np.sum(bone * n, axis=-1, keepdims=True) * n
    t = t / np.linalg.norm(t, axis=-1, keepdims=True)
    return np.stack([t, np.cross(n, t), n], axis=-1)


def strap_errors(rng: np.random.Generator, n: int, max_deg: float) -> np.ndarray:
    """E_i: rotations (n, 4) with each rotation-vector component uniform in ±max (as Step 1's mounting error)."""
    return Rotation.from_rotvec(rng.uniform(-1.0, 1.0, size=(n, 3)) * np.deg2rad(max_deg)).as_quat()
