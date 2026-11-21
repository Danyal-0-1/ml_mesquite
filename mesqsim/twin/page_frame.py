"""The one module that converts between the page's frames, Motive's and the repository's (Step 3 brief rule 10).

Frames
------
* **three.js world**: the page, its avatar and its BVH files. +Y up; the avatar faces +Z with its left at +X
  (Step 3 brief §5.1). The page and its BVH files measure lengths in centimetres.
* **Motive world**: the OptiTrack exports of 2026-11-13. +Z up, metres; Danyal faced -Y, so left was +X.
* **The repository's world** (:mod:`mesqsim.conventions`): +Z up, right-handed.
* **ARKit's world** and the phone app's **reported frame**: +Y up, metres, the axes of three.js (below).

Every conversion here is a proper rotation, so handedness and the sense of turning about the vertical survive it:

* three.js -> world: (x, y, z) -> (x, -z, y), the rotation by +90° about +X. The avatar's forward (+Z) becomes world
  -Y and its left stays +X, so the avatar in the world faces the way Danyal faced in Motive's world.
* Motive -> three.js: (x, y, z) -> (x, z, -y), the rotation by -90° about +X (Step 3 brief §5.1).
* Motive -> world: the composition of the two, which is the identity. Both worlds are +Z up and right-handed; a
  recording and its truth differ only by a turn about the vertical and a shift, which no statistic assumes known.

BVH rotations
-------------
A joint's CHANNELS list its rotations in order, and its local rotation is their product in that order,
R = R_c1(a1) · R_c2(a2) · R_c3(a3): three.js r157's ``BVHLoader`` right-multiplies one axis rotation per channel
(``build-static/BVHLoader.js:177-187`` at mesquite-mocap ``87d7c3b``), and the page writes ``Euler(..., "XYZ")`` with
channels ``Xrotation Yrotation Zrotation`` (``js/bvh_converter.js:7-13``), i.e. R = Rx · Ry · Rz.
"""

from __future__ import annotations

import numpy as np

from ..conventions import as_quat, quat_mul

__all__ = [
    "R_THREE_TO_WORLD",
    "Q_THREE_TO_WORLD",
    "R_MOTIVE_TO_THREE",
    "three_to_world_vec",
    "world_to_three_vec",
    "three_to_world_rot",
    "motive_to_three_vec",
    "motive_to_world_vec",
    "arkit_to_world_vec",
    "world_to_arkit_vec",
    "arkit_to_world_rot",
    "world_to_arkit_rot",
    "arkit_to_world_turn",
    "axis_quat",
    "bvh_euler_to_quat",
]

R_THREE_TO_WORLD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
"""Rotation matrix of three.js -> world: (x, y, z) -> (x, -z, y), +90° about +X."""

Q_THREE_TO_WORLD = np.array([np.sin(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)])
"""The same rotation as a quaternion (x, y, z, w)."""

R_MOTIVE_TO_THREE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
"""Rotation matrix of Motive -> three.js: (x, y, z) -> (x, z, -y), -90° about +X."""

_AXES = {"X": 0, "Y": 1, "Z": 2}


def three_to_world_vec(v) -> np.ndarray:
    """Vectors or points ``(..., 3)`` in three.js coordinates -> world coordinates (units unchanged)."""
    return np.asarray(v, dtype=np.float64) @ R_THREE_TO_WORLD.T


def world_to_three_vec(v) -> np.ndarray:
    """World coordinates ``(..., 3)`` -> three.js coordinates (the inverse of :func:`three_to_world_vec`)."""
    return np.asarray(v, dtype=np.float64) @ R_THREE_TO_WORLD


def three_to_world_rot(q) -> np.ndarray:
    """An orientation body -> three.js world as body -> world: ``Q_THREE_TO_WORLD · q``.

    The body's own axes are kept (for a bone, the rig's axes), so ``three_to_world_rot(q)`` applied to a body vector
    equals :func:`three_to_world_vec` of ``q`` applied to it.
    """
    q = as_quat(q)
    return quat_mul(np.broadcast_to(Q_THREE_TO_WORLD, q.shape), q)


def motive_to_three_vec(v) -> np.ndarray:
    """Motive coordinates ``(..., 3)`` -> three.js coordinates: (x, y, z) -> (x, z, -y)."""
    return np.asarray(v, dtype=np.float64) @ R_MOTIVE_TO_THREE.T


def motive_to_world_vec(v) -> np.ndarray:
    """Motive coordinates -> world coordinates, through three.js (the composition is the identity)."""
    return three_to_world_vec(motive_to_three_vec(v))


# ARKit's world with ``.gravity`` alignment, and the phone app's reported frame R, are +Y up and right-handed, the
# same axes as three.js (Apple, ARConfiguration.WorldAlignment.gravity; the app's MATH.md §2), so one matrix serves.
# Two kinds of quaternion cross this boundary: an *orientation* (a body's axes -> the frame, e.g. ARKit's camera
# pose) keeps its body axes and is multiplied on the left; a *rotation within the frame* (the app's reported pose,
# which is a turn of R relative to the Zero, or a Y(θ) about +Y) is conjugated, so its axis is converted too.

def arkit_to_world_vec(v) -> np.ndarray:
    """ARKit (or the app's reported frame) coordinates ``(..., 3)`` -> world coordinates: (x, y, z) -> (x, -z, y)."""
    return three_to_world_vec(v)


def world_to_arkit_vec(v) -> np.ndarray:
    """World coordinates ``(..., 3)`` -> ARKit coordinates, the inverse of :func:`arkit_to_world_vec`."""
    return world_to_three_vec(v)


def arkit_to_world_rot(q) -> np.ndarray:
    """An orientation body -> ARKit world as body -> world (body axes kept): ``Q · q``."""
    return three_to_world_rot(q)


def world_to_arkit_rot(q) -> np.ndarray:
    """An orientation body -> world as body -> ARKit world (body axes kept): ``Q⁻¹ · q``."""
    q = as_quat(q)
    qi = np.array([-Q_THREE_TO_WORLD[0], 0.0, 0.0, Q_THREE_TO_WORLD[3]])
    return quat_mul(np.broadcast_to(qi, q.shape), q)


def arkit_to_world_turn(q) -> np.ndarray:
    """A rotation within ARKit's (or the reported) frame -> the same rotation within the world: ``Q · q · Q⁻¹``.

    A turn by θ about ARKit's +Y becomes a turn by θ about the world's +Z (a proper rotation keeps the sense).
    """
    q = as_quat(q)
    qi = np.array([-Q_THREE_TO_WORLD[0], 0.0, 0.0, Q_THREE_TO_WORLD[3]])
    return quat_mul(quat_mul(np.broadcast_to(Q_THREE_TO_WORLD, q.shape), q), np.broadcast_to(qi, q.shape))


def axis_quat(axis: str, angle_rad) -> np.ndarray:
    """Quaternions ``(..., 4)`` of rotations by ``angle_rad`` about the body axis ``"X"``, ``"Y"`` or ``"Z"``."""
    half = 0.5 * np.asarray(angle_rad, dtype=np.float64)
    q = np.zeros(half.shape + (4,))
    q[..., _AXES[axis.upper()]] = np.sin(half)
    q[..., 3] = np.cos(half)
    return q


def bvh_euler_to_quat(angles_deg, order: str) -> np.ndarray:
    """BVH rotation channels ``(..., 3)`` in degrees, listed in ``order`` (e.g. ``"XYZ"``), -> quaternions.

    The product is taken in channel order, R = R_order[0] · R_order[1] · R_order[2], as three.js's BVHLoader does.
    """
    a = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
    order = order.upper()
    if len(order) != 3 or set(order) != set("XYZ"):
        raise ValueError(f"order must be a permutation of XYZ, got {order!r}")
    q = axis_quat(order[0], a[..., 0])
    for k in (1, 2):
        q = quat_mul(q, axis_quat(order[k], a[..., k]))
    return q
