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


# ---------------------------------------------------------------- 10. AMASS mapping

_NO_AMASS = CheckResult("AMASS not downloaded (docs/DATA.md)", None)


def _amass_files():
    from .data import amass

    return amass.motion_files()


@register(
    "10.1",
    "Rest pose: every segment equals the facing",
    "Zero every local rotation, give the root an upright pose with a random facing, run the forward kinematics "
    "and the mapping.",
    "At the SMPL rest pose (a T-pose) every global rotation equals R_0; this is the simulated T-pose.",
    "max angle between any segment and R_0 < 1e-9°",
    "pytest tests/test_amass.py -k rest_pose",
)
def _rest_pose(quick: bool) -> CheckResult:
    from .conventions import exp_map, log_map
    from .data import amass

    rng = np.random.default_rng(25)
    worst = 0.0
    for _ in range(20):
        root = quat_mul(yaw_quat(rng.uniform(-np.pi, np.pi)), exp_map([np.pi / 2, 0.0, 0.0]))
        poses = np.zeros((1, 66))
        poses[0, :3] = log_map(root)
        seg = amass.segment_orientations(poses)[0]
        worst = max(worst, _deg(np.max(geodesic_angle_rad(quat_mul(quat_conj(seg), root)))))
    return CheckResult(f"max angle {worst:.1e}° over 20 facings", worst < 1e-9)


@register(
    "10.2",
    "One AMASS sequence gives 15 unit orientations",
    "Load the first AMASS motion file and map it.",
    "The loader must handle the release's keys and frame rate; every segment must come out finite and unit.",
    "shape (N, 15, 4), all finite, |q| = 1",
    "pytest tests/test_amass.py -m amass -k 15_orientations",
)
def _amass_one(quick: bool) -> CheckResult:
    from .data import amass

    files = _amass_files()
    if not files:
        return _NO_AMASS
    seq = amass.load_sequence(files[0])
    seg = amass.segment_orientations(seq.poses)
    ok = seg.shape == (seq.n_frames, 15, 4) and np.all(np.isfinite(seg))
    ok = ok and np.max(np.abs(np.linalg.norm(seg, axis=-1) - 1)) < 1e-9
    return CheckResult(f"{files[0].name}: {seq.n_frames} frames at {seq.fps:g} Hz, shape {seg.shape}", ok)


@register(
    "10.3",
    "AMASS world is +Z up",
    "Over up to 20 sequences, average the pelvis's body up axis (SMPL +y) in world coordinates.",
    "Every dataset is converted to the +Z-up world at load time; for AMASS that conversion is the identity, "
    "which this verifies from the data rather than assuming it.",
    "closest world axis +Z, mean cosine > 0.8",
    "pytest tests/test_amass.py -m amass -k z_up",
)
def _amass_up(quick: bool) -> CheckResult:
    from .data import amass

    files = _amass_files()
    if not files:
        return _NO_AMASS
    roots = np.concatenate([amass.global_rotations(amass.load_sequence(f).poses)[:, 0] for f in files[:20]])
    axis, cos = amass.estimate_world_up(roots)
    return CheckResult(f"axis {axis.tolist()}, mean cosine {cos:.3f}", bool(np.all(axis == [0, 0, 1]) and cos > 0.8))


@register(
    "10.4",
    "Walking: left and right thighs alternate",
    "Hip flexion (thigh relative to pelvis about the lateral axis) on the first AMASS file named 'walk'.",
    "Left and right must be the subject's own sides; swapped or mirrored legs would still look like walking "
    "but would correlate positively.",
    "correlation(left, right) < -0.3",
    "pytest tests/test_amass.py -m amass -k thighs_alternate",
)
def _amass_walk(quick: bool) -> CheckResult:
    from .data import amass

    path = amass.find_walking_sequence()
    if path is None:
        return _NO_AMASS
    seg = amass.segment_orientations(amass.load_sequence(path).poses)
    r = float(np.corrcoef(amass.hip_flexion_rad(seg, "left"), amass.hip_flexion_rad(seg, "right"))[0, 1])
    return CheckResult(f"{path.name}: correlation {r:+.2f}", r < -0.3)


@register(
    "10.5",
    "Knee flexion in an anatomical range",
    "Knee flexion (shin relative to thigh about the lateral axis) over up to 50 sequences, both sides.",
    "A wrong joint index or axis would produce impossible knees.",
    "all frames within (-20°, 170°)",
    "pytest tests/test_amass.py -m amass -k knee",
)
def _amass_knee(quick: bool) -> CheckResult:
    from .data import amass

    files = _amass_files()
    if not files:
        return _NO_AMASS
    lo, hi = np.inf, -np.inf
    for f in files[: 10 if quick else 50]:
        seg = amass.segment_orientations(amass.load_sequence(f).poses)
        for side in ("left", "right"):
            k = np.rad2deg(amass.knee_flexion_rad(seg, side))
            lo, hi = min(lo, k.min()), max(hi, k.max())
    return CheckResult(f"range [{lo:.1f}°, {hi:.1f}°]", lo > -20.0 and hi < 170.0)


# ---------------------------------------------------------------- 11. DIODEM

_NO_DIODEM = CheckResult("DIODEM not downloaded (scripts/download_diodem.py)", None)


@register(
    "11.1",
    "DIODEM files match the manifest",
    "Verify size and SHA-256 of every file in manifests/diodem.csv (downloads were checked against the "
    "Dataverse MD5 when fetched).",
    "The sanity check is only meaningful on the published data.",
    "267 of 267 files ok",
    "python scripts/check_env.py --require diodem",
)
def _diodem_manifest(quick: bool) -> CheckResult:
    from .data import diodem
    from .manifest import verify_manifest
    from .paths import data_dir, manifest_dir

    if not diodem.is_available():
        return _NO_DIODEM
    res = verify_manifest(manifest_dir() / diodem.MANIFEST, data_dir())
    n_ok = sum(s == "ok" for _, s, _ in res)
    return CheckResult(f"{n_ok} of {len(res)} files ok", n_ok == len(res) == 267)


@register(
    "11.2",
    "DIODEM rate 40 Hz, quaternion norms 1",
    "Read the header of every IMU and optical CSV; compute every optical quaternion's norm.",
    "Confirms the format the loader assumes before any analysis.",
    "IMU 40 Hz everywhere; max | |q| - 1 | < 1e-6",
    "pytest tests/test_diodem.py -m diodem -k rates",
)
def _diodem_rates(quick: bool) -> CheckResult:
    from .data import diodem

    if not diodem.is_available():
        return _NO_DIODEM
    s = diodem.scan_rates_and_norms()
    ok = s["imu_rates_hz"] == [40.0] and s["max_norm_dev"] < 1e-6
    return CheckResult(
        f"IMU rates {s['imu_rates_hz']} Hz; optical rates {s['omc_rates_hz']} Hz; max | |q| - 1 | "
        f"{s['max_norm_dev']:.1e} over {s['n_quaternions']} quaternions", ok
    )


@register(
    "11.3",
    "Six-axis relative heading drifts, optical stays bounded",
    "Arm trial exp01 (all 13 motions, 6.1 min), hinge seg2 -> seg3, rigid IMUs: six-axis VQF (magnetometer "
    "dropped) against optical truth. Figure: docs/figures/diodem_relative_heading.png.",
    "Reproduces, on real sensors, the phenomenon the project corrects. No claims are drawn from it in Step 1.",
    "|VQF − optical| at the end > 2 × its P95 over the first 5 s; optical relative heading spans < 180°",
    "python scripts/make_figures.py",
)
def _diodem_drift(quick: bool) -> CheckResult:
    from .data import diodem

    if not diodem.is_available():
        return _NO_DIODEM
    res = diodem.six_axis_relative_heading(diodem.load_trial(1), "seg2", "seg3")
    first = np.percentile(np.abs(res["delta_deg"][res["t_s"] < 5]), 95)
    end, span = res["delta_deg"][-1], np.ptp(res["rel_opt_deg"])
    return CheckResult(
        f"VQF − optical: end {end:+.1f}°, max |.| {np.max(np.abs(res['delta_deg'])):.1f}°, P95 first 5 s "
        f"{first:.2f}°; optical span {span:.1f}°", abs(end) > 2 * first and span < 180.0
    )
