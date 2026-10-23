"""The numeric checks of brief §9, written so that they can both be asserted by tests and reported.

Each check is registered with what it tests, why, the expected value and how to reproduce it.
``scripts/validate.py`` runs them all and writes ``docs/VALIDATION.md``; the tests call the same
functions (often with ``quick=True``, which uses fewer Monte Carlo samples) and assert ``passed``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .codec import decode_frames, encode_frames, quantize_component
from .conventions import (
    geodesic_angle_rad,
    gravity_in_body,
    heading_rad,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_to_matrix,
    wrap_deg,
    wrap_rad,
    yaw_quat,
)

__all__ = ["Check", "CheckResult", "REGISTRY", "register", "run_checks"]


@dataclass(frozen=True)
class CheckResult:
    result: str
    passed: bool | None  # None: skipped (data absent)

    def __post_init__(self):
        if self.passed is not None:  # numpy booleans would break `is True` tests downstream
            object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    what: str
    why: str
    expected: str
    command: str
    fn: Callable[[bool], CheckResult]


REGISTRY: list[Check] = []


def register(id: str, title: str, what: str, why: str, expected: str, command: str):
    def deco(fn: Callable[[bool], CheckResult]):
        REGISTRY.append(Check(id, title, what, why, expected, command, fn))
        return fn

    return deco


def run_checks(quick: bool = False, prefix: str | None = None) -> list[tuple[Check, CheckResult]]:
    out = []
    for chk in sorted(REGISTRY, key=lambda c: [int(p) for p in c.id.split(".")]):
        if prefix is None or chk.id == prefix or chk.id.startswith(prefix + "."):
            out.append((chk, chk.fn(quick)))
    return out


def _deg(x) -> float:
    return float(np.rad2deg(x))


# ---------------------------------------------------------------- 1. codec

@register(
    "1.1",
    "Golden vector, both ways",
    "Encode id 2, batt 87, (x, y, z, w) = (0.5, -0.5, 0.25, 0.75), count 0x1234, ms_lo 0xABCD; decode it back.",
    "The codec must match the firmware frame byte for byte, or nothing decoded later can be trusted.",
    "aa550257ff3f01c0ff1fff5f3412cdab, and decoding returns the same fields",
    "pytest tests/test_codec.py -k golden",
)
def _golden(quick: bool) -> CheckResult:
    buf = encode_frames(2, 87, (0.5, -0.5, 0.25, 0.75), 0x1234, 0xABCD)
    f = decode_frames(bytes.fromhex("aa550257ff3f01c0ff1fff5f3412cdab"))[0]
    back = (int(f["pod_id"]), int(f["batt"]), int(f["count"]), int(f["ms_lo"]))
    ok = buf.hex() == "aa550257ff3f01c0ff1fff5f3412cdab" and back == (2, 87, 0x1234, 0xABCD)
    return CheckResult(f"encoded {buf.hex()}; decoded id/batt/count/ms_lo {back}", ok)


@register(
    "1.2",
    "Saturation at ±32767",
    "Quantise ±1.0, ±1.5 and ±1e9.",
    "int16 can hold -32768, but the firmware saturates symmetrically; an off-by-one here flips signs.",
    "+32767 and -32767, never -32768",
    "pytest tests/test_codec.py -k saturation",
)
def _saturation(quick: bool) -> CheckResult:
    v = quantize_component([1.0, -1.0, 1.5, -1.5, 1e9, -1e9])
    return CheckResult(f"{v.tolist()}", v.max() == 32767 and v.min() == -32767)


@register(
    "1.3",
    "Battery clamped at 100",
    "Encode battery values 150, 100, -5.",
    "The frame field is 0-100 percent.",
    "100, 100, 0",
    "pytest tests/test_codec.py -k battery",
)
def _battery(quick: bool) -> CheckResult:
    got = [int(decode_frames(encode_frames(0, b, (0, 0, 0, 1), 0, 0))[0]["batt"]) for b in (150, 100, -5)]
    return CheckResult(f"{got}", got == [100, 100, 0])


@register(
    "1.4",
    "count and ms_lo wrap at 65,536",
    "Encode count = ms_lo = 65535, 65536, 65537 and 3·65536 + 7.",
    "Both fields are uint16 on the wire; Step 0 must unwrap them.",
    "65535, 0, 1, 7",
    "pytest tests/test_codec.py -k wrap",
)
def _wrap16(quick: bool) -> CheckResult:
    vals = [65535, 65536, 65537, 3 * 65536 + 7]
    f = decode_frames(encode_frames(1, 50, [(0, 0, 0, 1)] * 4, vals, vals))
    got = (f["count"].tolist(), f["ms_lo"].tolist())
    return CheckResult(f"count {got[0]}, ms_lo {got[1]}", got[0] == got[1] == [65535, 0, 1, 7])


# ---------------------------------------------------------------- 2. conventions

@register(
    "2.1",
    "Y(90°) maps +x to +y",
    "Rotate the unit x vector by Y(90°).",
    "Pins the sign of Y(α): right-handed about world +Z.",
    "(0, 1, 0)",
    "pytest tests/test_conventions.py -k y90",
)
def _y90(quick: bool) -> CheckResult:
    v = quat_rotate(yaw_quat(np.pi / 2), [1.0, 0.0, 0.0])
    return CheckResult(f"({v[0]:.3g}, {v[1]:.3g}, {v[2]:.3g})", bool(np.allclose(v, [0, 1, 0], atol=1e-15)))


@register(
    "2.2",
    "heading(Y(α)·R) = wrap(heading(R) + α)",
    "Random orientations with |pitch| < 80° and random α; also heading equals scipy's first intrinsic "
    "Z-Y-X angle.",
    "Every heading quantity in the project (targets, box registration) relies on this identity.",
    "max error < 1e-9 rad",
    "pytest tests/test_conventions.py -k heading",
)
def _heading(quick: bool) -> CheckResult:
    from scipy.spatial.transform import Rotation

    n = 2000 if quick else 100000
    rng = np.random.default_rng(21)
    yaw = rng.uniform(-np.pi, np.pi, n)
    pitch = rng.uniform(-1.39, 1.39, n)
    roll = rng.uniform(-np.pi, np.pi, n)
    alpha = rng.uniform(-np.pi, np.pi, n)
    q = Rotation.from_euler("ZYX", np.c_[yaw, pitch, roll]).as_quat()
    e1 = np.max(np.abs(wrap_rad(heading_rad(q) - yaw)))
    e2 = np.max(np.abs(wrap_rad(heading_rad(quat_mul(yaw_quat(alpha), q)) - (heading_rad(q) + alpha))))
    err = max(e1, e2)
    return CheckResult(f"max error {err:.2e} rad over {n} orientations", bool(err < 1e-9))


@register(
    "2.3",
    "wrap into (-180°, 180°]",
    "wrap(190°), wrap(-180°), wrap(180°).",
    "The interval is half-open; -180° and 180° must not both appear.",
    "-170, 180, 180",
    "pytest tests/test_conventions.py -k wrap",
)
def _wrapdeg(quick: bool) -> CheckResult:
    got = [float(wrap_deg(x)) for x in (190.0, -180.0, 180.0)]
    return CheckResult(f"{got}", got == [-170.0, 180.0, 180.0])


@register(
    "2.4",
    "q and -q are the same rotation",
    "Compare the rotation matrices and headings of random q and -q.",
    "The wire canonicalises to w ≥ 0; nothing downstream may depend on the sign.",
    "max difference < 1e-12",
    "pytest tests/test_conventions.py -k minus_q",
)
def _qminusq(quick: bool) -> CheckResult:
    q = quat_normalize(np.random.default_rng(22).normal(size=(1000, 4)))
    d = max(
        np.max(np.abs(quat_to_matrix(q) - quat_to_matrix(-q))),
        np.max(np.abs(wrap_rad(heading_rad(q) - heading_rad(-q)))),
    )
    return CheckResult(f"max difference {d:.1e}", bool(d < 1e-12))


@register(
    "2.5",
    "Composition order pinned",
    "quat_mul(Rz90, Rx90) applied to +y; and quat_mul(a, b)·v = a·(b·v).",
    "Composition reads right to left; swapping the order is the most common silent bug.",
    "+y -> +z (and -x for the swapped order)",
    "pytest tests/test_conventions.py -k composition",
)
def _composition(quick: bool) -> CheckResult:
    from .conventions import exp_map

    rz, rx = yaw_quat(np.pi / 2), exp_map([np.pi / 2, 0, 0])
    a = quat_rotate(quat_mul(rz, rx), [0, 1.0, 0])
    b = quat_rotate(quat_mul(rx, rz), [0, 1.0, 0])
    ok = np.allclose(a, [0, 0, 1], atol=1e-15) and np.allclose(b, [-1, 0, 0], atol=1e-15)
    return CheckResult(f"Rz90·Rx90·y = {np.round(a, 12).tolist()}, Rx90·Rz90·y = {np.round(b, 12).tolist()}", ok)


# ---------------------------------------------------------------- 3. the §4.4 example

@register(
    "3.1",
    "The §4.4 worked example",
    "Parent at the root with ψ_p = 12°, child ψ_c = 5°. δ = wrap(ψ_c − ψ_p); correct with Y(+δ) and with the "
    "wrong sign Y(−δ); measure the residual rotation against Y(−12°)·s_c·m_c.",
    "A sign error in the correction is invisible in most plots but leaves twice the error.",
    "δ = -7°, residual 0° with the right sign, 14° with the wrong sign",
    "pytest tests/test_correction.py -k worked_example",
)
def _worked_example(quick: bool) -> CheckResult:
    psi_p, psi_c = np.deg2rad(12.0), np.deg2rad(5.0)
    sm = quat_normalize(np.random.default_rng(23).normal(size=4))
    q_c = quat_mul(yaw_quat(-psi_c), sm)
    delta = wrap_rad(psi_c - psi_p)
    shared = quat_mul(yaw_quat(-psi_p), sm)
    good = _deg(geodesic_angle_rad(quat_mul(quat_mul(yaw_quat(delta), q_c), quat_conj(shared))))
    bad = _deg(geodesic_angle_rad(quat_mul(quat_mul(yaw_quat(-delta), q_c), quat_conj(shared))))
    ok = abs(_deg(delta) + 7) < 1e-12 and good < 1e-6 and abs(bad - 14) < 1e-9
    return CheckResult(f"δ = {_deg(delta):.6f}°, right sign {good:.2e}°, wrong sign {bad:.6f}°", ok)


# ---------------------------------------------------------------- 4. invariances

@register(
    "4.1",
    "A correction never changes tilt",
    "Apply random corrections Y(c_i) to random orientations; compare world-up expressed in body axes.",
    "A heading correction acts about the vertical only; changing tilt would corrupt what gravity measured.",
    "max change < 1e-12",
    "pytest tests/test_correction.py -k tilt",
)
def _tilt(quick: bool) -> CheckResult:
    rng = np.random.default_rng(24)
    q = quat_normalize(rng.normal(size=(5000, 4)))
    c = rng.uniform(-np.pi, np.pi, 5000)
    d = np.max(np.abs(gravity_in_body(quat_mul(yaw_quat(c), q)) - gravity_in_body(q)))
    return CheckResult(f"max change {d:.1e}", bool(d < 1e-12))
