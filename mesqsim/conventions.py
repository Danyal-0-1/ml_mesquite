"""The one place that defines rotation conventions for the whole project (brief §4.1).

Every other module imports these functions instead of re-implementing them; the tests in
``tests/test_conventions.py`` pin each definition.

Conventions
-----------
* A quaternion is a float64 array whose last axis is ``(x, y, z, w)``: scalar **last**, as in
  ``scipy.spatial.transform.Rotation``. ``q`` and ``-q`` are the same rotation.
* Products are Hamilton products. ``quat_mul(a, b)`` is the rotation "first ``b``, then ``a``"
  (composition reads right to left, like matrices).
* An orientation maps body coordinates to world coordinates: ``v_world = R @ v_body``.
* The world frame is gravity-aligned with **+Z up**; its heading is arbitrary but fixed.
* ``Y(alpha)`` (:func:`yaw_quat`) rotates by ``alpha`` about world +Z, right-handed: ``Y(90°)``
  maps +x to +y.
* The heading of an orientation is the first angle of the intrinsic Z-Y-X (yaw-pitch-roll)
  decomposition, so ``heading(Y(a) R) = wrap(heading(R) + a)``.
* :func:`wrap_rad` / :func:`wrap_deg` return angles in the half-open interval (-180°, 180°].
* Radians inside the code, degrees in reports. Function names carry the unit.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "QUAT_ORDER",
    "as_quat",
    "quat_identity",
    "quat_mul",
    "quat_conj",
    "quat_inv",
    "quat_normalize",
    "quat_canonical",
    "quat_rotate",
    "quat_to_matrix",
    "quat_from_matrix",
    "yaw_quat",
    "heading_rad",
    "wrap_rad",
    "wrap_deg",
    "exp_map",
    "log_map",
    "geodesic_angle_rad",
    "quat_slerp",
    "interp_quat",
    "gravity_in_body",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
    "circ_mean_rad",
    "circ_std_rad",
    "quat_mean",
]

QUAT_ORDER = ("x", "y", "z", "w")
"""Component order of every quaternion array in this project (scalar last)."""


def as_quat(q) -> np.ndarray:
    """Return ``q`` as a float64 array whose last axis has length 4 ``(x, y, z, w)``."""
    q = np.asarray(q, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion last axis must have length 4 (x, y, z, w), got shape {q.shape}")
    return q


def quat_identity(shape: tuple[int, ...] = ()) -> np.ndarray:
    """Identity quaternions ``(0, 0, 0, 1)`` with the given leading shape."""
    q = np.zeros((*shape, 4))
    q[..., 3] = 1.0
    return q


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product ``a ⊗ b``: the rotation that applies ``b`` first, then ``a``.

    Broadcasts over leading axes.
    """
    a = as_quat(a)
    b = as_quat(b)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_conj(q) -> np.ndarray:
    """Conjugate ``(-x, -y, -z, w)``; the inverse rotation for a unit quaternion."""
    q = as_quat(q)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_inv(q) -> np.ndarray:
    """Inverse quaternion (conjugate divided by the squared norm)."""
    q = as_quat(q)
    return quat_conj(q) / np.sum(q * q, axis=-1, keepdims=True)


def quat_normalize(q) -> np.ndarray:
    """Scale each quaternion to unit norm."""
    q = as_quat(q)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_canonical(q) -> np.ndarray:
    """Choose the sign with ``w >= 0`` (the wire convention; brief §3.2).

    Quaternions with ``w == 0`` exactly are returned unchanged: both signs are valid on the wire.
    """
    q = as_quat(q)
    return np.where(q[..., 3:4] < 0.0, -q, q)


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vectors ``v`` (last axis 3) by unit quaternions ``q``: ``v' = q v q*``."""
    q = as_quat(q)
    v = np.asarray(v, dtype=np.float64)
    qv = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * np.cross(qv, v)
    return v + w * t + np.cross(qv, t)


def quat_to_matrix(q) -> np.ndarray:
    """3x3 rotation matrices ``R`` with ``v_world = R @ v_body`` for unit quaternions ``q``."""
    q = quat_normalize(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    r = np.empty((*q.shape[:-1], 3, 3))
    r[..., 0, 0] = 1 - 2 * (y * y + z * z)
    r[..., 0, 1] = 2 * (x * y - w * z)
    r[..., 0, 2] = 2 * (x * z + w * y)
    r[..., 1, 0] = 2 * (x * y + w * z)
    r[..., 1, 1] = 1 - 2 * (x * x + z * z)
    r[..., 1, 2] = 2 * (y * z - w * x)
    r[..., 2, 0] = 2 * (x * z - w * y)
    r[..., 2, 1] = 2 * (y * z + w * x)
    r[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return r


def quat_from_matrix(r) -> np.ndarray:
    """Unit quaternions (``w >= 0``) from rotation matrices; inverse of :func:`quat_to_matrix`."""
    from scipy.spatial.transform import Rotation

    r = np.asarray(r, dtype=np.float64)
    q = Rotation.from_matrix(r.reshape(-1, 3, 3)).as_quat()  # scipy: (x, y, z, w)
    return quat_canonical(q.reshape(*r.shape[:-2], 4))


def yaw_quat(alpha_rad) -> np.ndarray:
    """``Y(alpha)``: right-handed rotation by ``alpha_rad`` about world +Z (``Y(pi/2)`` maps +x to +y)."""
    half = 0.5 * np.asarray(alpha_rad, dtype=np.float64)
    zero = np.zeros_like(half)
    return np.stack([zero, zero, np.sin(half), np.cos(half)], axis=-1)


def heading_rad(q) -> np.ndarray:
    """Heading: first angle of the intrinsic Z-Y-X decomposition ``R = Rz(heading) Ry(pitch) Rx(roll)``.

    Equals ``atan2(R[1, 0], R[0, 0])``: the direction of the body x axis projected on the horizontal
    plane. It is ill-conditioned when the body x axis is vertical (pitch = ±90°). Works for
    non-unit quaternions (the scale cancels in ``atan2``).
    """
    q = as_quat(q)
    x, y, z, w = np.moveaxis(q, -1, 0)
    r10 = 2.0 * (x * y + w * z)
    r00 = w * w + x * x - y * y - z * z
    return np.arctan2(r10, r00)


def _wrap(x, half_period: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    period = 2.0 * half_period
    y = np.mod(x + half_period, period) - half_period  # in [-half, half)
    return np.where(y <= -half_period, y + period, y)  # move -half to +half


def wrap_rad(x) -> np.ndarray:
    """Wrap angles in radians into (-pi, pi]."""
    return _wrap(x, np.pi)


def wrap_deg(x) -> np.ndarray:
    """Wrap angles in degrees into (-180, 180]."""
    return _wrap(x, 180.0)


def exp_map(rotvec_rad) -> np.ndarray:
    """Quaternion of the rotation vector ``rotvec_rad`` (axis times angle in radians), ``Exp`` in §4.2."""
    v = np.asarray(rotvec_rad, dtype=np.float64)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    # sin(angle/2)/angle written with np.sinc so that it is exact at angle = 0.
    xyz = v * 0.5 * np.sinc(angle / (2.0 * np.pi))
    return np.concatenate([xyz, np.cos(0.5 * angle)], axis=-1)


def log_map(q) -> np.ndarray:
    """Rotation vector (radians) of a unit quaternion; inverse of :func:`exp_map`, angle in [0, pi]."""
    q = quat_canonical(as_quat(q))
    v = q[..., :3]
    s = np.linalg.norm(v, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(s, q[..., 3:4])
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(s > 1e-12, angle / np.where(s > 1e-12, s, 1.0), 2.0 / q[..., 3:4])
    return v * scale


def geodesic_angle_rad(q) -> np.ndarray:
    """Rotation angle in [0, pi] of each quaternion: ``2 atan2(|v|, |w|)`` (brief §4.7)."""
    q = as_quat(q)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3]))


def quat_slerp(q0, q1, u) -> np.ndarray:
    """Spherical linear interpolation along the shorter arc; ``u = 0`` gives ``q0``, ``u = 1`` gives ``q1``.

    Implemented as ``q0 ⊗ Exp(u · Log(q0⁻¹ ⊗ q1))``.
    """
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    u = np.asarray(u, dtype=np.float64)[..., None]
    rel = quat_mul(quat_conj(q0), q1)
    return quat_mul(q0, exp_map(u * log_map(rel)))


def interp_quat(t_src_s, q_src, t_query_s) -> np.ndarray:
    """Slerp a quaternion time series ``q_src`` (N, 4) sampled at increasing ``t_src_s`` to ``t_query_s``.

    Queries outside the source range are clamped to the first or last sample.
    """
    t_src = np.asarray(t_src_s, dtype=np.float64)
    q_src = as_quat(q_src)
    t_q = np.asarray(t_query_s, dtype=np.float64)
    if t_src.ndim != 1 or q_src.shape[0] != t_src.shape[0]:
        raise ValueError("t_src_s must be 1-D and match q_src's first axis")
    if t_src.size == 1:
        return np.broadcast_to(q_src[0], (*t_q.shape, 4)).copy()
    idx = np.clip(np.searchsorted(t_src, t_q, side="right") - 1, 0, t_src.size - 2)
    t0 = t_src[idx]
    t1 = t_src[idx + 1]
    u = np.clip((t_q - t0) / (t1 - t0), 0.0, 1.0)
    return quat_slerp(q_src[idx], q_src[idx + 1], u)


def gravity_in_body(q) -> np.ndarray:
    """World up (+Z) expressed in body coordinates, ``R^T e_z``. It describes tilt and ignores heading."""
    q = as_quat(q)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (*q.shape[:-1], 3))
    return quat_rotate(quat_conj(quat_normalize(q)), up)


def wxyz_to_xyzw(q_wxyz) -> np.ndarray:
    """Reorder scalar-first quaternions (as VQF and some datasets store them) to this project's order."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    return np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)


def xyzw_to_wxyz(q_xyzw) -> np.ndarray:
    """Reorder this project's quaternions to scalar-first ``(w, x, y, z)``."""
    q = as_quat(q_xyzw)
    return np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)


def circ_mean_rad(angles_rad, axis=None) -> np.ndarray:
    """Circular mean (radians, in (-pi, pi]) of angles: ``atan2(mean sin, mean cos)``."""
    a = np.asarray(angles_rad, dtype=np.float64)
    return wrap_rad(np.arctan2(np.mean(np.sin(a), axis=axis), np.mean(np.cos(a), axis=axis)))


def circ_std_rad(angles_rad, axis=None) -> np.ndarray:
    """Circular standard deviation ``sqrt(-2 ln R)`` (radians), R the mean resultant length."""
    a = np.asarray(angles_rad, dtype=np.float64)
    r = np.hypot(np.mean(np.sin(a), axis=axis), np.mean(np.cos(a), axis=axis))
    return np.sqrt(-2.0 * np.log(np.clip(r, 1e-300, 1.0)))


def quat_mean(q, axis: int = 0) -> np.ndarray:
    """Average of nearby unit quaternions along ``axis``: signs aligned to the first, mean, renormalised.

    Accurate for tightly clustered samples (a box record, a T-pose window); not a general Karcher mean.
    """
    q = np.moveaxis(as_quat(q), axis, 0)
    ref = q[0]
    sign = np.where(np.sum(q * ref, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return quat_canonical(quat_normalize(np.mean(q * sign, axis=0)))
