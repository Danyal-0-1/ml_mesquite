"""The arithmetic of three.js r157 that the page uses, operation for operation, for the page twin.

The page twin must reproduce the page's numbers, including the ones that depend on floating point: a bone stops
moving when ``Quaternion.slerp`` finds its target equal to it (``cosHalfTheta >= 1``), and the exact repeats of
Step 3 brief §5.2 (Φ8) come from that. So these functions follow ``build-static/three.module.js`` at mesquite-mocap
``87d7c3b`` term by term, on plain tuples ``(x, y, z, w)``. Everything else in the repository uses
:mod:`mesqsim.conventions`; the conventions are the same (Hamilton product, scalar last), only the order of the
operations differs.

:func:`js_number` writes a number as JavaScript's ``Number.prototype.toString`` does, which is how the page's BVH
writer prints every value.
"""

from __future__ import annotations

import math
from decimal import Decimal

__all__ = ["IDENTITY", "q_mul", "q_conj", "q_length", "q_normalize", "q_slerp", "q_from_axis_angle", "euler_xyz",
           "v_lerp", "js_number"]

IDENTITY = (0.0, 0.0, 0.0, 1.0)
_EPSILON = 2.0 ** -52  # Number.EPSILON


def q_mul(a, b) -> tuple[float, float, float, float]:
    """``Quaternion.multiplyQuaternions(a, b)``: a · b."""
    qax, qay, qaz, qaw = a
    qbx, qby, qbz, qbw = b
    return (qax * qbw + qaw * qbx + qay * qbz - qaz * qby,
            qay * qbw + qaw * qby + qaz * qbx - qax * qbz,
            qaz * qbw + qaw * qbz + qax * qby - qay * qbx,
            qaw * qbw - qax * qbx - qay * qby - qaz * qbz)


def q_conj(q) -> tuple[float, float, float, float]:
    """``Quaternion.invert()``, which is the conjugate (three.js assumes a unit quaternion)."""
    return (q[0] * -1, q[1] * -1, q[2] * -1, q[3])


def q_length(q) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def q_normalize(q) -> tuple[float, float, float, float]:
    """``Quaternion.normalize()``; a zero quaternion becomes the identity."""
    n = q_length(q)
    if n == 0:
        return IDENTITY
    n = 1 / n
    return (q[0] * n, q[1] * n, q[2] * n, q[3] * n)


def q_slerp(a, b, t: float) -> tuple[float, float, float, float]:
    """``a.slerp(b, t)``: the new value of ``a``, with every branch of three.js r157."""
    if t == 0:
        return tuple(a)
    if t == 1:
        return tuple(b)
    x, y, z, w = a
    cos_half = w * b[3] + x * b[0] + y * b[1] + z * b[2]
    if cos_half < 0:
        bw, bx, by, bz = -b[3], -b[0], -b[1], -b[2]
        cos_half = -cos_half
    else:
        bx, by, bz, bw = b
    if cos_half >= 1.0:
        return (x, y, z, w)
    sqr_sin_half = 1.0 - cos_half * cos_half
    if sqr_sin_half <= _EPSILON:
        s = 1 - t
        return q_normalize((s * x + t * bx, s * y + t * by, s * z + t * bz, s * w + t * bw))
    sin_half = math.sqrt(sqr_sin_half)
    half = math.atan2(sin_half, cos_half)
    ratio_a = math.sin((1 - t) * half) / sin_half
    ratio_b = math.sin(t * half) / sin_half
    return (x * ratio_a + bx * ratio_b, y * ratio_a + by * ratio_b, z * ratio_a + bz * ratio_b,
            w * ratio_a + bw * ratio_b)


def q_from_axis_angle(axis, angle: float) -> tuple[float, float, float, float]:
    """``Quaternion.setFromAxisAngle(axis, angle)`` (the axis assumed unit)."""
    half = angle / 2
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def euler_xyz(q) -> tuple[float, float, float]:
    """``new Euler().setFromQuaternion(q, "XYZ")`` in radians: ``Matrix4.compose`` then ``setFromRotationMatrix``."""
    x, y, z, w = q
    x2, y2, z2 = x + x, y + y, z + z
    xx, xy, xz = x * x2, x * y2, x * z2
    yy, yz, zz = y * y2, y * z2, z * z2
    wx, wy, wz = w * x2, w * y2, w * z2
    m11 = (1 - (yy + zz)) * 1.0
    m12 = (xy - wz) * 1.0
    m13 = (xz + wy) * 1.0
    m22 = (1 - (xx + zz)) * 1.0
    m23 = (yz - wx) * 1.0
    m32 = (yz + wx) * 1.0
    m33 = (1 - (xx + yy)) * 1.0
    ey = math.asin(max(-1.0, min(1.0, m13)))
    if abs(m13) < 0.9999999:
        return (math.atan2(-m23, m33), ey, math.atan2(-m12, m11))
    return (math.atan2(m32, m22), ey, 0.0)


def v_lerp(v, target, alpha: float) -> tuple[float, float, float]:
    """``Vector3.lerp(target, alpha)``."""
    return (v[0] + (target[0] - v[0]) * alpha, v[1] + (target[1] - v[1]) * alpha, v[2] + (target[2] - v[2]) * alpha)


def js_number(x: float) -> str:
    """``String(x)`` in JavaScript (ECMAScript Number::toString): shortest round-trip digits, plain notation for
    decimal exponents in (-7, 21], exponential beyond, ``0`` for both zeros."""
    x = float(x)
    if x == 0:
        return "0"
    if math.isnan(x):
        return "NaN"
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"
    if x < 0:
        return "-" + js_number(-x)
    sign, digits, exp = Decimal(repr(x)).as_tuple()
    digits = list(digits)
    while len(digits) > 1 and digits[-1] == 0:
        digits.pop()
        exp += 1
    s = "".join(str(d) for d in digits)
    k = len(s)
    n = exp + k  # x = 0.s x 10^n
    if k <= n <= 21:
        return s + "0" * (n - k)
    if 0 < n <= 21:
        return s[:n] + "." + s[n:]
    if -6 < n <= 0:
        return "0." + "0" * (-n) + s
    e = n - 1
    return (s if k == 1 else s[0] + "." + s[1:]) + "e" + ("+" if e >= 0 else "-") + str(abs(e))
