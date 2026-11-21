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


# ---------------------------------------------------------------- 5, 7, 8: the simulator and Step 0

def _simulate(preset: str | None, seed: int = 1, motion: str = "synthetic:walk:20", plan_hook=None, **overrides):
    """Simulate one session in memory, decode it with Step 0, return (SessionData, data dict, series)."""
    from . import step0
    from .config import load_config
    from .paths import REPO_ROOT
    from .sim.motion import MotionSource
    from .sim.simulate import simulate_session

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    if preset:
        cfg = cfg.with_preset(preset)
    if overrides:
        cfg = cfg.with_overrides(overrides)
    d = simulate_session(cfg, seed, MotionSource.from_spec(motion), plan_hook=plan_hook)
    data = {"frames": d.frames, "session": d.session, "labels_ticks": d.labels_ticks,
            "labels_joints": d.labels_joints, "meta": d.meta}  # fmt: skip
    return d, data, step0.decode(d.frames, d.session)


@register(
    "5.1",
    "Zero corruption: B0 = 0 up to quantisation, O2 = 0",
    "Preset 'zero' (no offsets, exact timing): simulate a 20 s synthetic walk, decode, score B0 and O2.",
    "With nothing to correct, any error left is a pipeline bug; only int16 quantisation may remain.",
    "heading error 0 for both; B0 orientation error ≤ 0.01° (P99) and < 0.02° (max)",
    "pytest tests/test_sim_slice.py -k zero_corruption",
)
def _zero(quick: bool) -> CheckResult:
    from . import evaluate as ev

    _, data, series = _simulate("zero")
    e = ev.evaluate(data, series)
    b0, o2 = e[e["baseline"] == "B0"], e[e["baseline"] == "O2"]
    hmax = max(b0["err_heading_deg"].max(), o2["err_heading_deg"].max())
    p99, mx = b0["err_orient_deg"].quantile(0.99), b0["err_orient_deg"].max()
    return CheckResult(f"heading max {hmax:.1e}°; B0 orientation P99 {p99:.4f}°, max {mx:.4f}°",
                       hmax == 0.0 and p99 <= 0.01 and mx < 0.02)  # fmt: skip


@register(
    "5.2",
    "Offsets only: B0 equals the offset differences, O2 = 0",
    "Preset 'offsets_only' (arbitrary offsets a_i, nothing else): B0's orientation error through the decoded "
    "frames against |wrap(a_c − a_p)|; O2's heading error.",
    "The M3 end-to-end slice: frames, Step 0 and the metric agree with the analytic answer.",
    "|B0 − |a_c − a_p|| < 0.02° at every step; O2 heading error exactly 0",
    "pytest tests/test_sim_slice.py -k 'offset_differences or o2_error'",
)
def _offsets(quick: bool) -> CheckResult:
    from . import evaluate as ev
    from .body import JOINTS

    d, data, series = _simulate("offsets_only")
    e = ev.evaluate(data, series)
    offs = np.array([p["offset_rad"] for p in d.meta["plan"]["pods"]])
    exp = np.rad2deg(np.abs(wrap_rad([offs[j.child] - offs[j.parent] for j in JOINTS])))
    b0 = e[e["baseline"] == "B0"]
    dev = float(np.max(np.abs(b0["err_orient_deg"].to_numpy() - exp[b0["joint"].to_numpy()])))
    o2 = float(e.loc[e["baseline"] == "O2", "err_heading_deg"].max())
    return CheckResult(f"max |B0 − offset difference| {dev:.4f}° (offset differences up to {exp.max():.1f}°); "
                       f"O2 heading max {o2:.1e}°", dev < 0.02 and o2 == 0.0)  # fmt: skip


@register(
    "7.3",
    "Step 0 unwraps ms_lo across ≥ 3 wraps",
    "Simulate 200 s of standing (about 238 s of recording, over 3 wraps of 65.536 s); compare Step 0's unwrapped "
    "ms_lo with the pod clock at each true decode time.",
    "ms_lo wraps every 65.536 s; a wrong unwrap shifts every later sample time by a minute.",
    "exact equality for every received frame",
    "pytest tests/test_sim_slice.py -k three_wraps",
)
def _unwrap(quick: bool) -> CheckResult:
    from .sim.timing import pod_ms

    d, _, series = _simulate("offsets_only", motion="synthetic:stand:200")
    m = series[series["received"]].merge(d.labels_ticks, on=["pod_id", "epoch", "tick"])
    truth = np.empty(len(m))
    for p in d.plan.pods:
        sel = (m["pod_id"] == p.pod_id).to_numpy()
        truth[sel] = np.floor(pod_ms(p, m.loc[sel, "t_decode_s"].to_numpy()))
    wraps = m["ms_unwrapped"].max() / 65536
    bad = int(np.sum(m["ms_unwrapped"].to_numpy() != truth))
    return CheckResult(f"{len(m)} frames, up to {wraps:.2f} wraps, {bad} mismatches", bad == 0 and wraps > 3)


@register(
    "7.4",
    "Step 0's fresh, held and lost flags equal the truth",
    "Join Step 0's series with the simulator's per-tick labels on (pod, epoch, tick) and compare every flag, under "
    "the default config and under the stress preset (5 % loss in bursts of 10, 100 ms stalls, reboots; seed 5 has "
    "three reboots), both with a 10-minute idle gap.",
    "Step 0 is the shared decoder of real recordings; its bookkeeping must be exact where the truth is known.",
    "same set of (pod, epoch, tick); 0 flag mismatches; reboot epochs found",
    "pytest tests/test_validation.py -k check_7_4",
)
def _flags(quick: bool) -> CheckResult:
    texts, ok = [], True
    for preset, seed in ((None, 1), ("stress", 5)):
        d, _, series = _simulate(preset, seed=seed, motion="synthetic:walk:30", **{"session.idle_s": 600.0})
        m = series.merge(d.labels_ticks, on=["pod_id", "epoch", "tick"], how="outer", indicator=True,
                         suffixes=("", "_true"))  # fmt: skip
        unmatched = int((m["_merge"] != "both").sum())
        both = m[m["_merge"] == "both"]
        bad = int(np.sum((both["status"] == "fresh").to_numpy() != both["new_to_host"].to_numpy()))
        bad += int(np.sum(both["received"].to_numpy() != both["received_true"].to_numpy()))
        c = series["status"].value_counts()
        reboots = int(series.groupby("pod_id")["epoch"].max().sum())
        texts.append(f"{preset or 'default'}: {len(both)} ticks (fresh {c.get('fresh', 0)}, held {c.get('held', 0)}, "
                     f"lost {c.get('lost', 0)}), {reboots} reboots, {unmatched} unmatched, {bad} mismatches")
        ok &= unmatched == 0 and bad == 0
    return CheckResult("; ".join(texts), ok)


@register(
    "8.1",
    "Same config and seed: identical bytes; another seed differs",
    "Simulate a 5 s synthetic walk three times (seeds 7, 7, 8), write each session, hash every file.",
    "A run must be reproducible from (commit, config hash, seed) alone.",
    "seed 7 twice: identical SHA-256 for every file; seed 8: different frames",
    "pytest tests/test_sim_slice.py -k same_seed",
)
def _repro(quick: bool) -> CheckResult:
    import hashlib
    import tempfile
    from pathlib import Path

    from . import io

    hashes = []
    with tempfile.TemporaryDirectory() as tmp:
        for k, seed in enumerate((7, 7, 8)):
            d, _, _ = _simulate(None, seed=seed, motion="synthetic:walk:5")
            folder = io.write_session(Path(tmp) / f"r{k}", d, {"seed": seed, **d.meta})
            hashes.append({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())})
    same = hashes[0] == hashes[1]
    differ = hashes[0]["frames.parquet"] != hashes[2]["frames.parquet"]
    return CheckResult(f"{len(hashes[0])} files; seed 7 twice identical: {same}; seed 8 differs: {differ}",
                       same and differ)  # fmt: skip


# ---------------------------------------------------------------- 4. invariances at the simulator level

def _shift_offsets(pods_deltas: dict):
    def hook(plan):
        for pod, d in pods_deltas.items():
            plan.pods[pod].offset_rad += d

    return hook


@register(
    "4.2",
    "Turning every pod's north by one Y(β) changes nothing observable",
    "Simulate the same session twice (default config, seed 2), the second time with β = 73° added to every pod's "
    "north; compare the labels and B1's estimates (Step 0 box registration on the re-simulated frames).",
    "The room's heading is unknowable (brief §2, §4.5): labels must not move, and estimators may only move by "
    "quantisation.",
    "labels: max change < 1e-9°; B1: max change ≤ 0.5°",
    "pytest tests/test_validation.py -k check_4_2",
)
def _common_yaw(quick: bool) -> CheckResult:
    from . import evaluate as ev

    beta = np.deg2rad(73.0)
    _, d0, s0 = _simulate(None, seed=2)
    _, d1, s1 = _simulate(None, seed=2, plan_hook=_shift_offsets({i: beta for i in range(15)}))
    lab0 = d0["labels_joints"]["delta_rad"].to_numpy()
    lab1 = d1["labels_joints"]["delta_rad"].to_numpy()
    dl = np.max(np.abs(wrap_rad(lab1 - lab0)))
    g0, g1 = ev.Grid(d0["labels_joints"]), ev.Grid(d1["labels_joints"])
    b0, b1 = ev.estimate("B1", g0, s0, d0["session"]), ev.estimate("B1", g1, s1, d1["session"])
    db = np.max(np.abs(wrap_rad(b1 - b0)))
    return CheckResult(f"labels max change {_deg(dl):.1e}°; B1 max change {_deg(db):.4f}°",
                       _deg(dl) < 1e-9 and _deg(db) <= 0.5)  # fmt: skip


@register(
    "4.3",
    "Turning one pod's north by Y(γ) moves only its joints, by exactly ±γ",
    "Re-simulate with γ = 23° added to the LeftUpLeg pod's north (pod 9, child of l_hip, parent of l_knee).",
    "Each pod's north enters only the joints it touches: as child +γ, as parent −γ (brief §4.5).",
    "l_hip +23°, l_knee −23°, every other joint 0 (to 1e-9°); other pods' frames identical",
    "pytest tests/test_validation.py -k check_4_3",
)
def _one_pod(quick: bool) -> CheckResult:
    from .body import JOINTS

    gamma = np.deg2rad(23.0)
    a, d0, _ = _simulate(None, seed=2)
    b, d1, _ = _simulate(None, seed=2, plan_hook=_shift_offsets({9: gamma}))
    lj0, lj1 = d0["labels_joints"], d1["labels_joints"]
    diff = wrap_rad(lj1["delta_rad"].to_numpy() - lj0["delta_rad"].to_numpy())
    ok = True
    parts = []
    for k, j in enumerate(JOINTS):
        dk = diff[lj0["joint"].to_numpy() == k]
        want = gamma if j.child == 9 else (-gamma if j.parent == 9 else 0.0)
        ok &= _deg(float(np.max(np.abs(dk - want)))) < 1e-9
        if want != 0.0:
            parts.append(f"{j.name} {_deg(np.mean(dk)):+.6f}°")
    others = a.frames[a.frames["pod_id"] != 9].reset_index(drop=True)
    others_b = b.frames[b.frames["pod_id"] != 9].reset_index(drop=True)
    same = others.equals(others_b)
    return CheckResult(f"{', '.join(parts)}, others 0; other pods' frames identical: {same}", ok and same)


# ---------------------------------------------------------------- 5. offsets only: B1 = O1 = 0

@register(
    "5.3",
    "Offsets only (arbitrary, zero slot error): B1 = O1 = 0, B0 is not",
    "Preset 'offsets_only': score B0, B1 and O1 over the motion phase.",
    "Box registration recovers arbitrary offsets exactly when the box slots are perfect; nothing else may leak in.",
    "B1 and O1 ≤ 0.01° (quantisation); B0 mean > 10°",
    "pytest tests/test_validation.py -k check_5_3",
)
def _offsets_b1(quick: bool) -> CheckResult:
    from . import evaluate as ev

    _, data, series = _simulate("offsets_only")
    e = ev.evaluate(data, series, baselines=("B0", "B1", "O1"))
    e = e[e["phase"] == "motion"]
    mx = e.groupby("baseline")["err_heading_deg"].max()
    mean_b0 = e.loc[e["baseline"] == "B0", "err_heading_deg"].mean()
    return CheckResult(f"max heading error B1 {mx['B1']:.4f}°, O1 {mx['O1']:.1e}°; B0 mean {mean_b0:.1f}°",
                       mx["B1"] <= 0.01 and mx["O1"] <= 0.01 and mean_b0 > 10.0)  # fmt: skip


# ---------------------------------------------------------------- 6. analytic growth

def _heading_pods(n_sessions: int, cfg_overrides: dict, motion: str, hook=None):
    """Plans and PodHeading objects for many pods, without frames (fast Monte Carlo)."""
    from .config import load_config
    from .paths import REPO_ROOT
    from .sim.heading import build_pod_heading
    from .sim.motion import MotionSource
    from .sim.rng import stream
    from .sim.session import plan_session
    from .sim.simulate import pod_epochs

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_preset("zero").with_overrides(cfg_overrides)
    mot = MotionSource.from_spec(motion)
    arw = np.deg2rad(cfg["error_model.arw_deg_per_sqrt_s"])
    out = []
    for seed in range(n_sessions):
        plan = plan_session(cfg, seed, mot)
        if hook:
            hook(plan)
        r_ep, r_walk = stream(seed, "epochs"), stream(seed, "walk")
        for p in plan.pods:
            eps = pod_epochs(cfg, plan, mot, p, r_ep)
            hd = build_pod_heading(plan, mot, p.pod_id, eps, p.drift_rad_per_s, p.scale_factor, arw, r_walk)
            out.append((plan, p, hd))
    return out


@register(
    "6.1",
    "Random walk only: σ of ψ at 600 s is 0.367°, of a pair 0.519°",
    "Preset 'zero' plus the datasheet angle random walk (0.015 °/s/√Hz); idle chosen so that 600 s after each pod's "
    "boot falls in the motion phase; ψ over 4,500 pods (600 in quick mode) and over the 14 parent-child pairs.",
    "Pins the random walk's scaling σ = N_g √t (datasheet: 0.367° after 10 min, √2 × for a pair).",
    "within 5 % of 0.367° and 0.519° (15 % in quick mode)",
    "pytest tests/test_validation.py -k check_6_1",
)
def _arw(quick: bool) -> CheckResult:
    from .body import JOINTS

    n = 40 if quick else 300
    pods = _heading_pods(n, {"error_model.arw_deg_per_sqrt_s": 0.015, "session.idle_s": 560.0}, "synthetic:stand:10")
    psi = np.array([float(hd.psi(p.boot_s + 600.0)) for _, p, hd in pods])
    by_session = psi.reshape(n, 15)
    pair = np.concatenate([by_session[:, j.child] - by_session[:, j.parent] for j in JOINTS])
    s1, s2 = _deg(np.std(psi)), _deg(np.std(pair))
    e1, e2 = 0.015 * np.sqrt(600.0), 0.015 * np.sqrt(1200.0)
    tol = 0.15 if quick else 0.05
    return CheckResult(f"σ ψ {s1:.4f}° (expected {e1:.4f}°, {len(psi)} pods); σ pair {s2:.4f}° (expected {e2:.4f}°, "
                       f"{len(pair)} pairs)", abs(s1 / e1 - 1) < tol and abs(s2 / e2 - 1) < tol)  # fmt: skip


@register(
    "6.2",
    "Drift only: δ(t) = (b_c − b_p)·t exactly",
    "Preset 'zero' plus drift (σ_b = 1 °/min), all pods booted at t = 0 (boot spread 0); compare every label δ(t_k) "
    "with (b_c − b_p)·t_k from the drawn b_i.",
    "Pins the drift term's sign, units (deg/min) and time origin.",
    "max |δ − (b_c − b_p)·t| < 1e-9° over every step and joint",
    "pytest tests/test_validation.py -k check_6_2",
)
def _drift(quick: bool) -> CheckResult:
    from .body import JOINTS

    d, data, _ = _simulate("zero", **{"error_model.drift_sigma_deg_per_min": 1.0, "session.boot_spread_s": 0.0})
    lj = data["labels_joints"]
    b = np.array([p.drift_rad_per_s for p in d.plan.pods])
    jt = lj["joint"].to_numpy()
    expected = np.array([b[JOINTS[k].child] - b[JOINTS[k].parent] for k in jt]) * lj["t_s"].to_numpy()
    err = _deg(np.max(np.abs(wrap_rad(lj["delta_rad"].to_numpy() - expected))))
    largest = _deg(np.max(np.abs(expected)))
    return CheckResult(f"max deviation {err:.1e}° over {len(lj)} labels; largest |δ| {largest:.2f}°", err < 1e-9)


@register(
    "6.3",
    "One full turn with k = 1.5 % gives 5.4°",
    "Synthetic walk in a circle at 30 °/s for exactly 360°; set the pelvis pod's scale factor to 1.5 %; ψ change of "
    "that pod over the motion, through the simulator's integration of the vertical angular velocity.",
    "Pins Θ (signed rotation about the vertical) and the scale-factor term: 0.015 × 360° = 5.4° (brief §4.3).",
    "5.4° within 5 %",
    "pytest tests/test_validation.py -k check_6_3",
)
def _turn(quick: bool) -> CheckResult:
    def hook(plan):
        plan.pods[2].scale_factor = 0.015

    pods = _heading_pods(1, {}, "synthetic:circle:12.0166667", hook)
    plan, _, hd = pods[2]
    m0, m1 = plan.phases["motion"]
    dpsi = _deg(float(hd.psi(m1) - hd.psi(m0)))
    dtheta = _deg(float(hd.theta_at(m1) - hd.theta_at(m0)))
    return CheckResult(f"Θ over the motion {dtheta:.3f}°; ψ change {dpsi:.4f}° (expected 5.4°)",
                       abs(dpsi / 5.4 - 1) < 0.05)  # fmt: skip


# ---------------------------------------------------------------- 7. timing statistics

def _one_pod_timing(rate_hz: float, seconds: float, seed: int, **overrides):
    from .config import load_config
    from .paths import REPO_ROOT
    from .sim.session import PodParams
    from .sim.timing import simulate_timing

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides({"timing.pod_clock_ppm": 0.0, **overrides})
    p = PodParams(pod_id=0, boot_s=0.0, clock_ppm=0.0, chip_rate_hz=rate_hz, chip_start_s=0.3, grid_first_ms=1000.0,
                  mount_q=[0, 0, 0, 1], box_slot_rad=0.0, offset_rad=0.0, strap_window_s=(0, 0), strap_spins=0,
                  posture_q=[0, 0, 0, 1], battery_start_pct=100.0)  # fmt: skip
    return cfg, p, simulate_timing(cfg, p, seconds, np.random.default_rng(seed))


@register(
    "7.1",
    "Discarded fractions 0.36, 0.42, 0.47 at 50.1, 55, 60 Hz",
    "One pod, 600 s, decode delay 0-11 ms, no stalls: share of chip samples that are never sent.",
    "The pod sends the newest sample at each 32 Hz send; the rest are discarded (brief §3.3): 1 − 32/f.",
    "within 5 % of 1 − 32/f (0.361, 0.418, 0.467)",
    "pytest tests/test_validation.py -k check_7_1",
)
def _discard(quick: bool) -> CheckResult:
    texts, ok = [], True
    for f in (50.1, 55.0, 60.0):
        _, _, tm = _one_pod_timing(f, 600.0, 1)
        used = tm.t_decode_s <= tm.t_send_s[-1]
        frac = 1.0 - np.unique(tm.sample_idx).size / used.sum()
        exp = 1.0 - 32.0 / f
        ok &= abs(frac / exp - 1) < 0.05
        texts.append(f"{f:g} Hz: {frac:.4f} (expected {exp:.4f})")
    return CheckResult("; ".join(texts), ok)


@register(
    "7.2",
    "Held frames: none without stalls; the expected rate with stalls",
    "Stalls off: 15 pods at rates spread over 50.1-60 Hz, 600 s each. Stalls on (S = 100 ms every 3 s ± 0.5 s): "
    "15 pods for 3,600 s (1,200 s in quick mode); expected held per stall = (S + E[u])/T − 1 with T = 31.25 ms and "
    "E[u] = 1/(2f) + f·D²/12 the mean time since the last decode (D = 11 ms decode spread).",
    "Held frames come only from stalls (brief §3.3); their rate must follow from the stall model, not from a bug.",
    "0 held without stalls; held rate within 5 % of the expectation with stalls (10 % in quick mode)",
    "pytest tests/test_validation.py -k check_7_2",
)
def _held(quick: bool) -> CheckResult:
    rates = np.linspace(50.1, 60.0, 15)
    held0 = sum(int(np.sum(~_one_pod_timing(f, 600.0, k)[2].fresh[1:])) for k, f in enumerate(rates))
    secs = 1200.0 if quick else 3600.0
    s_ms, d_s, t_ms = 100.0, 0.011, 31.25
    held, sends, expected = 0, 0, 0.0
    for k, f in enumerate(rates):
        _, _, tm = _one_pod_timing(f, secs, 100 + k, **{"timing.stall_ms": s_ms})
        held += int(np.sum(~tm.fresh[1:]))
        sends += tm.fresh.size
        eu_ms = 1000.0 * (1.0 / (2 * f) + f * d_s**2 / 12)
        expected += tm.n_stalls * ((s_ms + eu_ms) / t_ms - 1.0)
    rate, exp_rate = held / sends, expected / sends
    tol = 0.10 if quick else 0.05
    return CheckResult(f"stalls off: {held0} held; stalls on: rate {rate:.5f} vs expected {exp_rate:.5f} "
                       f"({held} held in {sends} sends)", held0 == 0 and abs(rate / exp_rate - 1) < tol)  # fmt: skip


@register(
    "7.5",
    "Loss rate and burst length match the loss model",
    "The Gilbert-Elliott chain alone, 2,000,000 sends, at the brief's sweep corners (0.5 %, 1), (2 %, 3), (5 %, 10).",
    "The loss model's two parameters must mean what their names say.",
    "loss fraction and mean burst within 5 % of the parameters",
    "pytest tests/test_validation.py -k check_7_5",
)
def _loss(quick: bool) -> CheckResult:
    from .sim.radio import gilbert_elliott_lost

    texts, ok = [], True
    for k, (loss, burst) in enumerate(((0.005, 1.0), (0.02, 3.0), (0.05, 10.0))):
        lost = gilbert_elliott_lost(2_000_000, loss, burst, np.random.default_rng(k))
        starts = np.flatnonzero(np.diff(np.r_[0, lost.astype(np.int8)]) == 1)
        rate, mean_burst = lost.mean(), lost.sum() / max(starts.size, 1)
        ok &= abs(rate / loss - 1) < 0.05 and abs(mean_burst / burst - 1) < 0.05
        texts.append(f"({loss:g}, {burst:g}): rate {rate:.5f}, burst {mean_burst:.3f}")
    return CheckResult("; ".join(texts), ok)


# ---------------------------------------------------------------- 12. baseline order

@register(
    "12.1",
    "Baseline order: O2 = 0 ≤ O1 ≤ B1, and B0 ≫ B1 in the arbitrary mode",
    "Default config (arbitrary offsets) with a 10-minute idle; mean heading error over the motion phase, averaged "
    "over 10 sessions of a 20 s synthetic walk (3 in quick mode).",
    "Oracles bound the honest baseline from below; without box registration arbitrary offsets dominate.",
    "O2 = 0 ≤ O1 ≤ B1; B0 > 5 × B1",
    "pytest tests/test_validation.py -k check_12_1",
)
def _order(quick: bool) -> CheckResult:
    from . import evaluate as ev

    means = []
    for seed in range(3 if quick else 10):
        _, data, series = _simulate(None, seed=seed, **{"session.idle_s": 600.0})
        e = ev.evaluate(data, series)
        means.append(e[e["phase"] == "motion"].groupby("baseline")["err_heading_deg"].mean())
    m = sum(means) / len(means)
    ok = m["O2"] == 0.0 and m["O1"] <= m["B1"] and m["B0"] > 5 * m["B1"]
    return CheckResult(", ".join(f"{k} {m[k]:.3f}°" for k in ("O2", "O1", "B1", "B0")), ok)


# ---------------------------------------------------------------- 9. splits

@register(
    "9.1",
    "Splits by subject; TotalCapture held out; deterministic",
    "Split a synthetic AMASS-like file list (CMU 20 subjects, KIT 3, SFU 1, TotalCapture 5; 2 sequences each), "
    "shuffle the rows and split again.",
    "Leakage between splits would make every later comparison optimistic (brief §5.1).",
    "no subject in two splits; CMU 14/3/3; every TotalCapture sequence in 'holdout'; identical output after shuffling",
    "pytest tests/test_splits.py",
)
def _splits_logic(quick: bool) -> CheckResult:
    import pandas as pd

    from .splits import assign_splits, check_manifest

    rows = [{"path": f"{sub}/S{s:02d}/q{q}_poses.npz", "subset": sub, "subject": f"S{s:02d}"}
            for sub, n in (("CMU", 20), ("KIT", 3), ("SFU", 1), ("TotalCapture", 5))
            for s in range(n) for q in range(2)]  # fmt: skip
    df = pd.DataFrame(rows)
    a, b = assign_splits(df), assign_splits(df.sample(frac=1.0, random_state=7))
    per = a[a["subset"] == "CMU"].groupby("split")["subject"].nunique().to_dict()
    tc = a.loc[a["subset"] == "TotalCapture", "split"].unique().tolist()
    problems = check_manifest(a)
    ok = not problems and per == {"train": 14, "val": 3, "test": 3} and tc == ["holdout"] and a.equals(b)
    return CheckResult(f"CMU {per}; TotalCapture {tc}; problems {problems or 'none'}; shuffle-invariant {a.equals(b)}",
                       ok)  # fmt: skip


@register(
    "9.2",
    "The AMASS split manifest: no subject in two splits, hash unchanged",
    "Read manifests/split_manifest.csv; check it; compare its SHA-256 with the frozen manifests/split_manifest.sha256.",
    "After Danyal freezes the split, any change to it must be deliberate and visible.",
    "no problems; SHA-256 equal to the frozen value",
    "python -m mesqsim split",
)
def _splits_manifest(quick: bool) -> CheckResult:
    import pandas as pd

    from .paths import manifest_dir
    from .splits import check_manifest, manifest_sha256

    path, frozen = manifest_dir() / "split_manifest.csv", manifest_dir() / "split_manifest.sha256"
    if not path.is_file():
        return CheckResult("no split manifest yet (needs AMASS: python -m mesqsim split --write)", None)
    df = pd.read_csv(path, dtype={"subject": str})  # CMU's "01" must not become 1
    problems = check_manifest(df)
    digest = manifest_sha256(path)
    want = frozen.read_text().split()[0] if frozen.is_file() else None
    # A subject is (subset, subject id): CMU and KIT reuse ids 10, 11, 12 and 63 for different people.
    counts = df.drop_duplicates(["subset", "subject"]).groupby("split").size().to_dict()
    return CheckResult(f"{len(df)} sequences, subjects per split {counts}; problems {problems or 'none'}; "
                       f"SHA-256 {digest[:12]}… frozen {want[:12] + '…' if want else 'not yet'}",
                       not problems and (want is None or want == digest))  # fmt: skip


@register(
    "9.3",
    "Every window and variant of a sequence stays in its split",
    "Generate a small dataset (synthetic sources, 2 windows, 2 variants per window) and read its dataset.csv.",
    "Windows of one sequence are near-duplicates; if they straddled splits, test data would leak into training.",
    "each source sequence has exactly one split; every session seed distinct",
    "pytest tests/test_splits.py -k variant",
)
def _splits_dataset(quick: bool) -> CheckResult:
    import tempfile
    from pathlib import Path

    import pandas as pd

    from .config import load_config
    from .dataset import generate, synthetic_sources
    from .paths import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides({
        "dataset.window_s": 4.0, "dataset.windows_per_sequence": 2, "dataset.motion_start_min": [1],
        "dataset.variants": 2,
    })  # fmt: skip
    with tempfile.TemporaryDirectory() as tmp:
        folder = generate(cfg, 3, synthetic_sources(2, 10.0), out_dir=Path(tmp) / "run", log=lambda s: None)
        ds = pd.read_csv(folder / "dataset.csv")
    per_source = ds.groupby("source")["split"].nunique()
    ok = bool((per_source == 1).all()) and ds["seed"].nunique() == len(ds)
    return CheckResult(f"{len(ds)} sessions from {ds['source'].nunique()} sequences; splits per sequence "
                       f"{sorted(per_source.unique().tolist())}; distinct seeds {ds['seed'].nunique()}", ok)


# ================================================================ Step 2 (Step 2 brief §8): item n is check 20 + n

def _geo_session(preset: str | None = None, seed: int = 11, motion: str = "synthetic:walk:20", **overrides):
    """One simulated session with Step 0's series and the scorer's query times (host times of the 32 Hz grid)."""
    from . import evaluate as ev
    from .config import load_config
    from .paths import REPO_ROOT

    d, data, series = _simulate(preset, seed=seed, motion=motion, **overrides)
    host = ev.Grid(d.labels_joints).t_s + d.meta["plan"]["host_epoch_s"]
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    return d, data, series, host, cfg


@register(
    "21.1",
    "Contract: causal (deleting later frames changes nothing before)",
    "Default config, 20 s synthetic walk with a 10-minute idle; for GEO-ID and GEO-K, cut the frames table at 5 host "
    "times (box, T-pose, three in the motion), decode the cut recording with Step 0, re-run the estimator and compare "
    "every output (delta, kappa, uncertain) at the query times up to the cut.",
    "An estimate at time t may use only frames that arrived at or before t (brief rule 9); a real-time display "
    "cannot see the future.",
    "identical bytes at every query time at or before each cut, for both estimators",
    "pytest tests/test_geo_contract.py -k causal",
)
def _geo_causal(quick: bool) -> CheckResult:
    from . import step0
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator
    from .geo_score import truncate_frames

    d, _, series, host, cfg = _geo_session(**{"session.idle_s": 600.0})
    ph = {p["name"]: (p["t_host_start_s"], p["t_host_end_s"]) for p in d.session["phases"]}
    m0 = ph["motion"][0]
    cuts = [ph["box"][1] - 1.0, ph["tpose"][0] + 1.5, m0 + 2.2, m0 + 9.0, m0 + 15.5]
    worst, compared = 0, 0
    for truth in (None, GeoTruth.from_meta(d.meta)):
        full = GeoEstimator(cfg, truth).estimate(series, d.session, host)
        for t_cut in cuts[: 3 if quick else 5]:
            s_cut = step0.decode(truncate_frames(d.frames, t_cut), d.session)
            cut = GeoEstimator(cfg, truth).estimate(s_cut, d.session, host)
            before = host <= t_cut
            compared += int(before.sum())
            worst += sum(int(not np.array_equal(a[before], b[before])) for a, b in zip(full, cut, strict=True))
    return CheckResult(f"{compared} query steps compared over {len(cuts[: 3 if quick else 5])} cuts × 2 estimators; "
                       f"{worst} outputs differ", worst == 0)  # fmt: skip


@register(
    "21.2",
    "Contract: deterministic (same inputs, identical bytes)",
    "Run GEO-ID and GEO-K twice on the same series, session and query times; compare the bytes of every output.",
    "Results must be reproducible from (commit, config, seed); a random start or set ordering would break it.",
    "identical bytes",
    "pytest tests/test_geo_contract.py -k deterministic",
)
def _geo_deterministic(quick: bool) -> CheckResult:
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator

    d, _, series, host, cfg = _geo_session()
    same = True
    for truth in (None, GeoTruth.from_meta(d.meta)):
        a = GeoEstimator(cfg, truth).estimate(series, d.session, host)
        b = GeoEstimator(cfg, truth).estimate(series.copy(), dict(d.session), host.copy())
        same &= all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))
    return CheckResult(f"GEO-ID and GEO-K, two runs each: identical bytes {same}", same)


@register(
    "21.3",
    "Contract: label-blind (GEO-ID identical with the labels deleted)",
    "Write a simulated session folder, run GEO-ID; delete labels_ticks.parquet, labels_joints.parquet and meta.json; "
    "read the folder as a real recording (frames and session.json only), run GEO-ID again.",
    "A deployable estimator must run on the real suit, which has no labels (brief rule 7).",
    "identical bytes",
    "pytest tests/test_geo_contract.py -k label_blind",
)
def _geo_label_blind(quick: bool) -> CheckResult:
    import tempfile
    from pathlib import Path

    from . import io, step0
    from .estimators.geo import estimate_geo_id

    d, _, _, host, cfg = _geo_session()
    with tempfile.TemporaryDirectory() as tmp:
        folder = io.write_session(Path(tmp) / "s", d, d.meta)
        rec = io.read_session(folder)
        a = estimate_geo_id(step0.decode(rec["frames"], rec["session"]), rec["session"], host, cfg)
        for name in ("labels_ticks.parquet", "labels_joints.parquet", "meta.json"):
            (folder / name).unlink()
        blind = io.read_session(folder, labels=False)
        b = estimate_geo_id(step0.decode(blind["frames"], blind["session"]), blind["session"], host, cfg)
    same = all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))
    return CheckResult(f"files left: frames.parquet, session.json; identical bytes {same}", same)


@register(
    "31.1",
    "Timing: differential timing error of the parent/child pairs",
    "Default config, 30 s synthetic walk with a 10-minute idle (3 seeds; 1 in quick mode): every pair GEO-ID formed, "
    "joined to the labels on (pod, epoch, tick): the child's true time at the slerp point minus the parent's true "
    "sample time. Also reported, not tested: the stress preset (100 ms sample-task stalls).",
    "The geometry compares two pods at one instant; the pods' clocks differ, and ms_lo is the decode time, 0-11 ms "
    "after the sample (Step 1 brief §3.3). Bound tested: 15 ms = the 11 ms decode-delay spread + 4 ms for the clock "
    "fit. After a stall a sample is decoded up to one chip period late, so stress can exceed it.",
    "max |error| ≤ 15 ms under the default config; P50 and P95 reported",
    "pytest tests/test_geo_contract.py -k pair_timing",
)
def _geo_timing(quick: bool) -> CheckResult:
    from .estimators.geo import GeoEstimator
    from .geo_score import TIMING_BOUND_MS, pair_timing_errors_ms

    def errors(preset, seed):
        d, _, series, host, cfg = _geo_session(preset, seed=seed, motion="synthetic:walk:30",
                                               **{"session.idle_s": 600.0})  # fmt: skip
        _, diag = GeoEstimator(cfg).run(series, d.session, host, diagnostics=True)
        return np.abs(pair_timing_errors_ms(diag, series, d.labels_ticks))

    e = np.concatenate([errors(None, s) for s in ((1,) if quick else (1, 2, 3))])
    st = errors("stress", 5)
    ok = e.max() <= TIMING_BOUND_MS
    p50, p95 = np.percentile(e, [50, 95])
    return CheckResult(f"default: {e.size} pairs, P50 {p50:.2f} ms, P95 {p95:.2f} ms, max {e.max():.2f} ms; "
                       f"stress (stalls): P95 {np.percentile(st, 95):.2f} ms, max {st.max():.2f} ms", ok)  # fmt: skip


@register(
    "22.1",
    "Hinge, exact: closed form, the 12°/5° example, the blind spot, and GEO-K through the pipeline",
    "(a) 500 random hinge windows (random delta, axis, mountings, poses, 1-40 pairs), no noise; (b) h = x horizontal "
    "with psi_p = 12°, psi_c = 5°; (c) a vertical axis; (d) GEO-K on a simulated 20 s walk with every error off except "
    "arbitrary offsets and drift (preset zero, offsets arbitrary, drift 0.2 °/min): every knee window's estimate "
    "against the true delta at the window end (the window solver through frames, Step 0, clock and pairs).",
    "Exact data must give the exact answer; (d) catches m against m^-1 (pod -> segment) and frame-order bugs that "
    "would make GEO-K fail on clean data (brief §5.10).",
    "(a) |delta_hat - delta| < 1e-9 rad; (b) -7.000°; (c) I < 1e-12 and kappa = 0; (d) every knee window < 0.1°",
    "pytest tests/test_hinge.py",
)
def _hinge_exact(quick: bool) -> CheckResult:
    from .body import JOINT_BY_NAME, JOINTS
    from .conventions import exp_map
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator
    from .estimators.hinge import axis_in_world, hinge_closed_form

    rng = np.random.default_rng(2201)
    worst = 0.0
    for _ in range(100 if quick else 500):
        n = int(rng.integers(1, 41))
        delta, psi_p = rng.uniform(-np.pi, np.pi, 2)
        u_seg = quat_normalize(np.r_[rng.normal(size=3), 0.0])[:3]
        m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
        s_p = quat_normalize(rng.normal(size=(n, 4)))
        s_c = quat_mul(s_p, exp_map(rng.uniform(0, 2, n)[:, None] * u_seg))
        q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p)), s_p), m_p)
        q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p - delta)), s_c), m_c)
        a = axis_in_world(q_p, quat_rotate(quat_conj(m_p), u_seg))[:, :2]
        b = axis_in_world(q_c, quat_rotate(quat_conj(m_c), u_seg))[:, :2]
        worst = max(worst, abs(float(wrap_rad(hinge_closed_form(a, b, 1e-3).delta_rad - delta))))
    a = quat_rotate(yaw_quat(-np.deg2rad(12.0)), [1.0, 0.0, 0.0])[None, :2]
    b = quat_rotate(yaw_quat(-np.deg2rad(5.0)), [1.0, 0.0, 0.0])[None, :2]
    ex = _deg(hinge_closed_form(a, b, 1e-3).delta_rad)
    vert = hinge_closed_form(quat_rotate(yaw_quat(np.linspace(0, 3, 20)), [0.0, 0.0, 1.0])[:, :2],
                             quat_rotate(yaw_quat(np.linspace(1, -2, 20)), [0.0, 0.0, 1.0])[:, :2], 1e-3)  # fmt: skip
    d, _, series, host, cfg = _geo_session("zero", **{"error_model.offset_mode": "arbitrary",
                                                      "error_model.drift_sigma_deg_per_min": 0.2})  # fmt: skip
    _, diag = GeoEstimator(cfg, GeoTruth.from_meta(d.meta)).run(series, d.session, host, diagnostics=True)
    from . import evaluate as ev

    g = ev.Grid(d.labels_joints)
    knees = {k for k, j in enumerate(JOINTS) if j.jtype == "hinge"}
    win = [r for r in diag.windows if r["joint"] in knees and r["status"] == "ok"]
    k_end = np.clip(np.searchsorted(host, [r["t_s"] for r in win], side="right") - 1, 0, host.size - 1)
    err = _deg(np.max(np.abs(wrap_rad(np.array([r["delta_rad"] for r in win])
                                      - g.delta_rad[k_end, [r["joint"] for r in win]]))))  # fmt: skip
    assert JOINT_BY_NAME["l_knee"].jtype == "hinge" and len(win) > 20
    ok = worst < 1e-9 and abs(ex + 7.0) < 5e-4 and vert.info < 1e-12 and vert.kappa == 0.0 and err < 0.1
    return CheckResult(f"(a) max error {worst:.1e} rad; (b) {ex:.4f}°; (c) I {vert.info:.1e}, kappa {vert.kappa:g}; "
                       f"(d) GEO-K knee windows: {len(win)}, max error {err:.4f}°", ok)  # fmt: skip


@register(
    "23.1",
    "Hinge likelihood: the spread of delta_hat matches 1/sqrt(kappa)",
    "One window's hinge geometry (40 pairs) with independent Gaussian noise (sigma 0.05 per component), 2,000 Monte "
    "Carlo draws: the circular RMS spread of delta_hat against the RMS of the predicted 1/sqrt(kappa).",
    "Verifies the formula kappa = I/sigma_hat^2 (the exact von Mises likelihood), not the real noise: correlated "
    "residuals are handled by the fitted kappa scale.",
    "ratio within 10 % of 1",
    "pytest tests/test_hinge.py -k likelihood",
)
def _hinge_likelihood(quick: bool) -> CheckResult:
    from .conventions import circ_mean_rad, exp_map
    from .estimators.hinge import axis_in_world, hinge_closed_form

    rng = np.random.default_rng(2301)
    n = 40
    u_seg = np.array([0.6, 0.0, 0.8])
    s_p = quat_normalize(rng.normal(size=(n, 4)))
    s_c = quat_mul(s_p, exp_map(rng.uniform(0, 2, n)[:, None] * u_seg))
    a0 = axis_in_world(s_p, u_seg)[:, :2]
    b0 = axis_in_world(quat_mul(yaw_quat(np.full(n, -0.3)), s_c), u_seg)[:, :2]
    est, kap = [], []
    for _ in range(2000):
        f = hinge_closed_form(a0 + rng.normal(0, 0.05, a0.shape), b0 + rng.normal(0, 0.05, b0.shape), 1e-6)
        est.append(f.delta_rad)
        kap.append(f.kappa)
    spread = float(np.sqrt(np.mean(wrap_rad(np.array(est) - circ_mean_rad(est)) ** 2)))
    predicted = float(np.sqrt(np.mean(1.0 / np.array(kap))))
    return CheckResult(f"spread {_deg(spread):.3f}°, predicted {_deg(predicted):.3f}°, ratio {spread / predicted:.3f}",
                       abs(spread / predicted - 1) < 0.10)  # fmt: skip


@register(
    "24.1",
    "Axis identification: within 1° on noise-free hinge motion; a flipped axis is corrected or flagged",
    "Five noise-free synthetic hinge motions (parent tumbling, child flexing; random mountings and private norths): "
    "fit u_p, u_c from body rates (Seel et al. 2014). Then deliberately flip u_c: the vertical rule must vote to "
    "flip back when the axis leaves the horizontal, and abstain when it stays horizontal; the T-pose rule must vote to "
    "flip back for a horizontal knee axis. Also GEO-ID through the pipeline: its knee axes on a simulated walk.",
    "A wrong axis or a wrong relative sign (180° error) would make GEO-ID's hinge worse than nothing (brief §5.10).",
    "both axes within 1° in every trial; votes +1 / -1 / 0 as stated; GEO-ID's knee axes within 2° on the walk",
    "pytest tests/test_hinge.py -k 'identified or flipped'",
)
def _axis_identification(quick: bool) -> CheckResult:
    from .conventions import exp_map
    from .estimators.clock import body_rates
    from .estimators.geo import GeoEstimator
    from .estimators.hinge import axis_in_world, identify_hinge_axes, vote_tpose, vote_vertical

    rng = np.random.default_rng(2401)
    worst = 0.0
    votes_ok = True
    for trial in range(5):
        t = np.arange(0.0, 8.0, 0.02)
        u_seg = np.array([1.0, 0.0, 0.0])
        s_p = exp_map(np.c_[0.6 * np.sin(1.3 * t), 0.4 * np.sin(0.7 * t + 1.0), 0.5 * np.sin(0.9 * t + 2.0)])
        s_c = quat_mul(s_p, exp_map((0.8 + 0.7 * np.sin(2.1 * t))[:, None] * u_seg))
        m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
        q_p, q_c = quat_mul(s_p, m_p), quat_mul(s_c, m_c)
        u_p, u_c = quat_rotate(quat_conj(m_p), u_seg), quat_rotate(quat_conj(m_c), u_seg)
        psi = rng.uniform(-np.pi, np.pi, 2)
        _, w_p = body_rates(t, quat_mul(yaw_quat(np.full(t.size, -psi[0])), q_p), 0.1)
        _, w_c = body_rates(t, quat_mul(yaw_quat(np.full(t.size, -psi[1])), q_c), 0.1)
        fit = identify_hinge_axes(w_p, w_c, n_starts=8)
        for got, true in ((fit.u_p, u_p), (fit.u_c, u_c)):
            worst = max(worst, _deg(np.arccos(np.clip(abs(np.dot(got, true)), 0, 1))))
        a3, b3 = axis_in_world(q_p, u_p), axis_in_world(q_c, u_c)
        mp = np.sin(np.deg2rad(5.0)) ** 2
        flat = a3.copy()
        flat[:, 2] = 0.0
        votes_ok &= vote_vertical(a3, b3, mp) == 1 and vote_vertical(a3, -b3, mp) == -1
        votes_ok &= vote_vertical(flat, flat, mp) == 0
        if trial == 0:  # the T-pose rule on a horizontal knee axis
            up = quat_mul(yaw_quat(0.7), exp_map([np.pi / 2, 0, 0]))
            flex = np.deg2rad(rng.uniform(5, 70, 64))
            qp = quat_mul(quat_mul(yaw_quat(np.full(64, -0.4)), np.broadcast_to(up, (64, 4))), m_p)
            qc = quat_mul(quat_mul(yaw_quat(np.full(64, 1.1)), quat_mul(np.broadcast_to(up, (64, 4)),
                                                                        exp_map(flex[:, None] * u_seg))), m_c)
            j_t = quat_mul(quat_mul(quat_conj(quat_mul(quat_mul(yaw_quat(-0.4), up), m_p)), yaw_quat(-1.5)),
                           quat_mul(quat_mul(yaw_quat(1.1), up), m_c))  # fmt: skip
            votes_ok &= vote_tpose(qp, qc, -1.5, j_t, np.deg2rad(60)) == 1
            votes_ok &= vote_tpose(qp, qc, -1.5 + np.pi, j_t, np.deg2rad(60)) == -1
    d, _, series, host, cfg = _geo_session(None, seed=1, **{"session.idle_s": 600.0})
    _, diag = GeoEstimator(cfg).run(series, d.session, host, diagnostics=True)
    mq = np.array([p["mount_q"] for p in d.meta["plan"]["pods"]])
    pipe = []
    for j, (p, c) in ((10, (9, 10)), (11, (12, 13))):
        f = diag.state["axes"].get(j)
        if f is None:
            pipe.append(np.inf)
            continue
        for got, pod in ((f.u_p, p), (f.u_c, c)):
            true = quat_rotate(quat_conj(mq[pod]), [1.0, 0.0, 0.0])
            pipe.append(_deg(np.arccos(np.clip(abs(np.dot(got, true)), 0, 1))))
    ok = worst < 1.0 and votes_ok and max(pipe) < 2.0
    return CheckResult(f"noise-free: worst axis error {worst:.3f}° over 5 motions; sign votes as expected: "
                       f"{votes_ok}; GEO-ID on a simulated walk (fusion noise on): knee axes within {max(pipe):.2f}°",
                       ok)  # fmt: skip


@register(
    "29.1",
    "Ordering with drift on, fusion noise off, known axes: O2 = 0 ≤ GEO-K ≤ B1 at every hinge",
    "Default config with drift 1 °/min, fusion tilt error 0 and a 10-minute idle; mean primary error over the motion "
    "at each knee, averaged over 4 sessions of a 20 s synthetic walk (2 in quick mode).",
    "With exact tilt and known axes the hinge geometry sees the drift that box registration cannot (brief §8 item 9).",
    "O2 = 0 ≤ GEO-K ≤ B1 at both knees",
    "pytest tests/test_validation.py -k check_29_1",
)
def _ordering_hinge(quick: bool) -> CheckResult:
    from . import evaluate as ev
    from .body import JOINTS

    rows = []
    for seed in range(2 if quick else 4):
        _, data, series, _, cfg = _geo_session(None, seed=seed, **{"session.idle_s": 600.0,
                                                                   "error_model.drift_sigma_deg_per_min": 1.0,
                                                                   "error_model.tilt_sigma_deg": 0.0})  # fmt: skip
        e = ev.evaluate(data, series, ("O2", "GEO-K", "B1"), cfg)
        rows.append(e[e["phase"] == "motion"].groupby(["baseline", "joint"])["err_heading_deg"].mean())
    m = sum(rows) / len(rows)
    ok, parts = True, []
    for k, j in enumerate(JOINTS):
        if j.jtype != "hinge":
            continue
        o2, gk, b1 = m[("O2", k)], m[("GEO-K", k)], m[("B1", k)]
        ok &= o2 == 0.0 and gk <= b1
        parts.append(f"{j.name}: O2 {o2:.3f}°, GEO-K {gk:.3f}°, B1 {b1:.3f}°")
    return CheckResult("; ".join(parts), ok)


@register(
    "25.1",
    "2-DoF, exact: the constraint recovers delta; two equal minima are flagged ambiguous",
    "120 random universal-joint windows (elbow, wrist and ankle axes; random delta, mountings, poses, private norths), "
    "no noise, known axes: delta_hat against delta. Then a built case with j1 horizontal and gamma = 90°, where the "
    "vertical terms vanish and delta, delta + 180° fit equally well.",
    "Exact data must give the exact answer; the 180° twin must never be reported with confidence (brief §5.5).",
    "|delta_hat - delta| < 1e-6 rad; the built case ambiguous with kappa = 0",
    "pytest tests/test_twodof_rom.py -k 'twodof or ambiguous'",
)
def _twodof_exact(quick: bool) -> CheckResult:
    from .conventions import exp_map
    from .estimators.anatomy import TWODOF_AXES
    from .estimators.hinge import axis_in_world
    from .estimators.twodof import twodof_solve

    rng = np.random.default_rng(2501)
    worst, n_amb = 0.0, 0
    for joint in ("l_elbow", "l_wrist", "l_ankle"):
        j1, j2, gamma = TWODOF_AXES[joint]
        for _ in range(20 if quick else 40):
            n, delta, psi = 60, rng.uniform(-np.pi, np.pi), rng.uniform(-np.pi, np.pi)
            m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
            s_p = exp_map(rng.normal(0, 1.0, (n, 3)))
            s_c = quat_mul(quat_mul(s_p, exp_map(rng.uniform(-1.2, 1.2, n)[:, None] * j1)),
                           exp_map(rng.uniform(-1.2, 1.2, n)[:, None] * j2))  # fmt: skip
            q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi)), s_p), m_p)
            q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi - delta)), s_c), m_c)
            fit = twodof_solve(axis_in_world(q_p, quat_rotate(quat_conj(m_p), j1)),
                               axis_in_world(q_c, quat_rotate(quat_conj(m_c), j2)), np.cos(gamma), 1e-6, 3.0)
            n_amb += int(fit.ambiguous)
            worst = max(worst, abs(float(wrap_rad(fit.delta_rad - delta))))
    a3 = np.c_[rng.normal(size=(50, 2)), np.zeros(50)]
    rot = np.array([[np.cos(-0.7), -np.sin(-0.7)], [np.sin(-0.7), np.cos(-0.7)]])
    b3 = np.c_[(np.c_[-a3[:, 1], a3[:, 0]] * rng.uniform(0.5, 1.5, (50, 1))) @ rot.T, rng.normal(size=50)]
    twin = twodof_solve(a3, b3, 0.0, 1e-3, 3.0)
    ok = worst < 1e-6 and n_amb == 0 and twin.ambiguous and twin.kappa == 0.0
    return CheckResult(f"max error {worst:.1e} rad over {3 * (20 if quick else 40)} windows ({n_amb} ambiguous); "
                       f"built twin: ambiguous {twin.ambiguous}, kappa {twin.kappa:g}, cost gap "
                       f"{abs(twin.second_cost - twin.cost):.1e}", ok)  # fmt: skip


@register(
    "26.1",
    "Range of motion: within 1° when the motion reaches the limits; almost no information inside them",
    "Noise-free hip windows (Cardan XZY; 120 samples; random delta, mountings, pelvis facing): (a) axial rotation "
    "sweeping its whole allowed range ±40° against the box [-10°, 60°] x [-10°, 10°] x [-40°, 40°]; (b) a ±10° sweep "
    "against a wide box [-90°, 120°] x [-60°, 60°] x [-90°, 90°] that it never reaches; (c) no box at all.",
    "At ball joints the only clue is the joint limit; where the motion stays inside, kappa must stay near 0 instead "
    "of inventing confidence (brief §5.6).",
    "(a) |delta_hat - delta| < 1°; (b) kappa < 3 (95 % interval wider than ±60°); (c) kappa = 0",
    "pytest tests/test_twodof_rom.py -k rom",
)
def _rom_check(quick: bool) -> CheckResult:
    from scipy.spatial.transform import Rotation

    from .conventions import exp_map, vm_halfwidth_rad
    from .estimators.anatomy import CARDAN_SEQ
    from .estimators.rom import UNBOUNDED_DEG, rom_solve

    rng = np.random.default_rng(2601)
    seq = CARDAN_SEQ["l_hip"]

    def window(delta, axial_deg):
        n, psi = 120, rng.uniform(-np.pi, np.pi)
        m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
        s_p = quat_mul(yaw_quat(rng.uniform(-np.pi, np.pi)), exp_map([np.pi / 2, 0.0, 0.0]))
        s_p = np.broadcast_to(s_p, (n, 4))
        rel = Rotation.from_euler(seq, np.c_[np.deg2rad(rng.uniform(-10, 60, n)), np.deg2rad(rng.uniform(-10, 10, n)),
                                             np.deg2rad(np.linspace(-axial_deg, axial_deg, n))]).as_quat()  # fmt: skip
        q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi)), s_p), m_p)
        q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi - delta)), quat_mul(s_p, rel)), m_c)
        return quat_mul(q_p, quat_conj(m_p)), quat_mul(q_c, quat_conj(m_c))

    tight = np.deg2rad([[-10, 60], [-10, 10], [-40, 40]])
    wide = np.deg2rad([[-90, 120], [-60, 60], [-90, 90]])
    free = np.deg2rad(np.array(UNBOUNDED_DEG))
    errs = []
    for delta in rng.uniform(-np.pi, np.pi, 3 if quick else 8):
        a_q, b_q = window(delta, 40.0)
        errs.append(abs(_deg(wrap_rad(rom_solve(a_q, b_q, seq, tight[:, 0], tight[:, 1],
                                                np.deg2rad(2.0)).delta_rad - delta))))  # fmt: skip
    a_q, b_q = window(0.8, 10.0)
    k_wide = rom_solve(a_q, b_q, seq, wide[:, 0], wide[:, 1], np.deg2rad(5.0)).kappa
    k_free = rom_solve(a_q, b_q, seq, free[:, 0], free[:, 1], np.deg2rad(5.0)).kappa
    hw = _deg(vm_halfwidth_rad(k_wide, 0.95))
    ok = max(errs) < 1.0 and k_wide < 3.0 and hw > 60.0 and k_free == 0.0
    return CheckResult(f"(a) max error {max(errs):.3f}° over {len(errs)} windows; (b) kappa {k_wide:.2f} "
                       f"(95 % half-width {hw:.0f}°); (c) kappa {k_free:g}", ok)  # fmt: skip


@register(
    "27.1",
    "Filter: exact prediction, large-kappa limit, product update, reboot reset; GEO-ID = B1 without information",
    "(a) Beliefs left alone for 1 s to 600 s: the mean stays and A(kappa) = A(kappa0)·exp(-q·dt/2); (b) kappa0 = 5e4: "
    "kappa against (1/kappa0 + q·dt)^-1; (c) one update against the closed-form product; (d) the stress preset, "
    "seed 5 (reboots), 10-minute idle: at the first query after the host learns of each reboot, before the next "
    "window, kappa of every joint of that pod; (e) noise-free standing (preset zero, arbitrary offsets): GEO-ID's "
    "error against B1's at every motion step.",
    "The filter is the estimator's memory; a reboot gives the pod a new north, so its joints must forget (brief "
    "§5.7). With nothing to learn, GEO-ID must not do worse than box registration (S2-M4).",
    "(a) relative error < 1e-9 and the mean unchanged; (b) within 1 %; (c) identical; (d) kappa = 0; (e) never worse",
    "pytest tests/test_filter.py",
)
def _filter_check(quick: bool) -> CheckResult:
    from . import evaluate as ev
    from .conventions import vm_mean_resultant, vm_product
    from .estimators.geo import GeoEstimator
    from .estimators.vmfilter import JointFilter

    f = JointFilter(3, 4e-6)
    k0 = np.array([5.0, 80.0, 2000.0])
    f.init(10.0, np.array([0.3, -2.0, 3.1]), k0)
    t = np.array([11.0, 70.0, 610.0])
    mu, kap, _ = f.query(t)
    rel = float(np.max(np.abs(vm_mean_resultant(kap) / (vm_mean_resultant(k0) * np.exp(-4e-6 * (t - 10) / 2)[:, None])
                              - 1)))  # fmt: skip
    mean_ok = float(np.max(np.abs(mu - [0.3, -2.0, 3.1]))) < 1e-15
    g = JointFilter(1, 1e-5)
    g.init(0.0, np.zeros(1), np.array([5e4]))
    lim = max(abs(float(g.query(np.array([dt]))[1][0, 0]) * (1 / 5e4 + 1e-5 * dt) - 1) for dt in (1.0, 60.0, 600.0))
    h = JointFilter(2, 0.0)
    h.init(0.0, np.array([0.1, 3.0]), np.array([10.0, 0.0]))
    h.update(1.0, np.array([0.5, -2.0]), np.array([30.0, 7.0]))
    pm, pk = vm_product(np.array([0.1, 3.0]), np.array([10.0, 0.0]), np.array([0.5, -2.0]), np.array([30.0, 7.0]))
    prod_ok = np.array_equal(h.mu, pm) and np.array_equal(h.kappa, pk)
    d, _, series, host, cfg = _geo_session("stress", seed=5, motion="synthetic:walk:30", **{"session.idle_s": 600.0})
    est, diag = GeoEstimator(cfg).run(series, d.session, host, diagnostics=True)
    resets = [(tt, jj) for tt, kind, jj in diag.events.events if kind == 1]
    wins = [tt for tt, kind, _ in diag.events.events if kind == 2]
    checked, reset_ok = 0, True
    for tt, jj in resets:
        k = int(np.searchsorted(host, tt, side="left"))
        nxt = [w for w in wins if w >= tt]
        if k < host.size and (not nxt or host[k] < nxt[0]):
            checked += 1
            reset_ok &= bool(np.all(est.kappa[k, jj] == 0.0))
    d2, _, s2, h2, _ = _geo_session("zero", seed=4, motion="synthetic:stand:20",
                                    **{"error_model.offset_mode": "arbitrary"})  # fmt: skip
    g2 = ev.Grid(d2.labels_joints)
    geo = GeoEstimator(cfg).estimate(s2, d2.session, h2)
    b1 = ev.estimate("B1", g2, s2, d2.session)
    mot = g2.phase == "motion"
    worse = float(np.max(np.abs(wrap_rad(geo.delta_rad[mot] - g2.delta_rad[mot]))
                         - np.abs(wrap_rad(b1[mot] - g2.delta_rad[mot]))))  # fmt: skip
    ok = rel < 1e-9 and mean_ok and lim < 0.01 and prod_ok and reset_ok and checked > 0 and worse <= 1e-12
    return CheckResult(f"(a) max relative error {rel:.1e}, mean unchanged {mean_ok}; (b) max {100 * lim:.3f} % off; "
                       f"(c) identical {prod_ok}; (d) {len(resets)} reboots, {checked} checked before the next window, "
                       f"kappa 0: {reset_ok}; (e) GEO-ID minus B1 error at most {_deg(worse):+.1e}°", ok)  # fmt: skip


@register(
    "28.1",
    "Invariances of GEO: one common yaw, one pod's yaw, tilt (brief §5.9)",
    "Default config, 20 s synthetic walk, 10-minute idle, seed 2. (a) Re-simulate with β = 73° added to every pod's "
    "north: GEO-ID and GEO-K over the motion; (b) with γ = 23° added to pod 9 (LeftUpLeg): every joint's change "
    "against +γ for l_hip, −γ for l_knee and 0 elsewhere; (c) GEO-ID's chain corrections applied to every pod's "
    "reports: the world-up direction in each pod's frame before and after.",
    "North and the room's heading are unknowable; an estimator that depends on them, or lets one pod's north leak "
    "into other joints, is broken (Step 1 brief §4.5).",
    "(a) every output within 0.5°; (b) each joint's change within 0.5° of its expected value; (c) < 1e-12",
    "pytest tests/test_validation.py -k check_28_1",
)
def _geo_invariance(quick: bool) -> CheckResult:
    from .body import JOINTS
    from .conventions import gravity_in_body
    from .correction import apply_corrections, chain_corrections_rad
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator

    def run(hook):
        d, _, series, host, cfg = _geo_session(None, seed=2, plan_hook=hook, **{"session.idle_s": 600.0})
        from . import evaluate as ev

        mot = ev.Grid(d.labels_joints).phase == "motion"
        out = {}
        for name, truth in (("GEO-ID", None), ("GEO-K", GeoTruth.from_meta(d.meta))):
            out[name] = GeoEstimator(cfg, truth).estimate(series, d.session, host).delta_rad[mot]
        return out, series

    base, series = run(None)
    common, _ = run(_shift_offsets({i: np.deg2rad(73.0) for i in range(15)}))
    one, _ = run(_shift_offsets({9: np.deg2rad(23.0)}))
    d_common = max(_deg(np.max(np.abs(wrap_rad(common[k] - base[k])))) for k in base)
    expect = np.array([np.deg2rad(23.0) if j.child == 9 else (-np.deg2rad(23.0) if j.parent == 9 else 0.0)
                       for j in JOINTS])  # fmt: skip
    d_one = max(_deg(np.max(np.abs(wrap_rad(one[k] - base[k] - expect)))) for k in base)
    rows = series[series["received"] & series["valid"]].head(600)
    q = rows[["qx", "qy", "qz", "qw"]].to_numpy()
    c = chain_corrections_rad(base["GEO-ID"][-1])[rows["pod_id"].to_numpy()]
    tilt = float(np.max(np.abs(gravity_in_body(apply_corrections(q, c)) - gravity_in_body(q))))
    ok = d_common < 0.5 and d_one < 0.5 and tilt < 1e-12
    return CheckResult(f"(a) common 73°: max change {d_common:.4f}° (GEO-ID and GEO-K); (b) pod 9 by 23°: max "
                       f"deviation from ±γ / 0 {d_one:.4f}°; (c) tilt change {tilt:.1e}", ok)  # fmt: skip


@register(
    "30.1",
    "Coverage on validation after recalibration: 68 % and 95 % per joint type",
    "GEO-ID with the configured (fitted) calibration on the validation sessions of the synthetic stand-in dataset "
    "(seed 1; 2 sequences and 12 sessions; quick: 1 sequence): the share of motion-phase steps whose error lies "
    "inside the central 68 % and 95 % von Mises intervals of the reported kappa, per joint type. The AMASS validation "
    "numbers come from `python -m mesqsim fit-geometry calibration --source amass` and docs/RESULTS_STEP2.md.",
    "Confidence is only useful if it is calibrated (H5); the brief asks that the numbers exist here and treats the "
    "±5-point target as a result, not a check.",
    "every coverage a number in [0, 1]",
    "python -m mesqsim fit-geometry calibration --source synthetic",
)
def _coverage_check(quick: bool) -> CheckResult:
    from .body import JOINTS
    from .config import load_config
    from .conventions import vm_halfwidth_rad
    from .dataset import synthetic_sources
    from .estimators.geo import GeoParams, replay
    from .geo_fit import calibration_worker
    from .geo_sessions import dataset_jobs
    from .paths import REPO_ROOT

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    jobs = dataset_jobs(cfg, synthetic_sources(), 1, splits=("val",), max_sequences=1 if quick else None)
    gp = GeoParams(cfg)
    jt = np.array([j.jtype for j in JOINTS])
    cov = {t: {0.68: [], 0.95: []} for t in ("hinge", "2dof", "3dof")}
    for job in jobs:
        r = calibration_worker(job, cfg)
        mu, kap, _ = replay(r["events"], r["t_query"], float(cfg["geometry.drift_q_rad2_per_s"]), gp.scales(),
                              gp.model_sigmas_rad(), gp.weight(True))  # fmt: skip
        err = np.abs(wrap_rad(mu - r["truth"]))
        for t in cov:
            for lev in (0.68, 0.95):
                cov[t][lev].append(err[:, jt == t] <= vm_halfwidth_rad(kap[:, jt == t], lev))
    res = {t: {lev: float(np.mean(np.concatenate([x.ravel() for x in v]))) for lev, v in cov[t].items()} for t in cov}
    ok = all(0.0 <= x <= 1.0 for t in res.values() for x in t.values())
    txt = "; ".join(f"{t} {res[t][0.68]:.3f} / {res[t][0.95]:.3f}" for t in res)
    return CheckResult(f"{len(jobs)} sessions, 68 % / 95 % coverage: {txt}", ok)


@register(
    "32.1",
    "DIODEM: files, the truth's sign, and GEO-ID against the optical truth on one hinge",
    "(a) Every DIODEM file against manifests/diodem.csv; (b) the truth's sign on a synthetic rotation: q_vqf = "
    "Y(-psi)·q_opt with psi = 30° must give psi = -heading(q_vqf·q_opt^-1) = 30°, and psi_p = 12°, psi_c = 5° must "
    "give delta = -7°; (c) GEO-ID, unchanged, on arm exp01, hinge seg2 -> seg3 (6.1 min, rigid IMUs, six-axis VQF at "
    "40 Hz, no box, no T-pose), against the optical relative heading; regenerates docs/figures/diodem_geo_hinge.png.",
    "A reality check on real sensors whose north drifts (−80° in 6 min here): the geometry must track what a perfect "
    "registration at the start cannot. Out of distribution (mechanical chains): no claim about people.",
    "(a) 267 of 267 files ok; (b) 30.000°, -7.000°; (c) GEO-ID's median error below the start registration's",
    "python scripts/make_figures.py",
)
def _diodem_geo(quick: bool) -> CheckResult:
    from .config import load_config
    from .data import diodem
    from .diodem_geo import psi_from_vqf_and_optical, run_hinge
    from .manifest import verify_manifest
    from .paths import REPO_ROOT, data_dir, manifest_dir

    q_opt = quat_normalize(np.random.default_rng(3201).normal(size=(50, 4)))
    psi = float(np.rad2deg(np.median(psi_from_vqf_and_optical(quat_mul(yaw_quat(np.full(50, -np.deg2rad(30.0))),
                                                                       q_opt), q_opt))))  # fmt: skip
    p_p = psi_from_vqf_and_optical(quat_mul(yaw_quat(-np.deg2rad(12.0)), q_opt[:1]), q_opt[:1])
    p_c = psi_from_vqf_and_optical(quat_mul(yaw_quat(-np.deg2rad(5.0)), q_opt[:1]), q_opt[:1])
    delta = _deg(float(wrap_rad(p_c - p_p)[0]))
    sign_ok = abs(psi - 30.0) < 1e-9 and abs(delta + 7.0) < 1e-9
    if not diodem.is_available():
        return CheckResult(f"(b) psi {psi:.3f}°, delta {delta:.3f}°; DIODEM not downloaded", None if sign_ok else False)
    res = verify_manifest(manifest_dir() / diodem.MANIFEST, data_dir())
    n_ok = sum(s == "ok" for _, s, _ in res)
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    r = run_hinge(diodem.load_trial(1), "seg2", "seg3", cfg)
    err = np.rad2deg(np.abs(wrap_rad(r["delta_rad"] - r["truth_rad"])))
    ref = np.rad2deg(np.abs(wrap_rad(r["start_oracle_rad"] - r["truth_rad"])))
    if not quick:
        from .figures import diodem_geo_figure

        diodem_geo_figure(r, "arm exp01, hinge seg2 → seg3", REPO_ROOT / "docs" / "figures" / "diodem_geo_hinge.png")
    ok = sign_ok and n_ok == len(res) == 267 and np.nanmedian(err) < np.nanmedian(ref)
    return CheckResult(f"(a) {n_ok} of {len(res)} files ok; (b) psi {psi:.3f}°, delta {delta:.3f}°; (c) GEO-ID median "
                       f"{np.nanmedian(err):.2f}° (P95 {np.nanpercentile(err, 95):.2f}°, first information after "
                       f"{r['t_first_info_s']:.1f} s) against {np.nanmedian(ref):.2f}° for a perfect registration at "
                       f"the start (end {ref[-1]:.1f}°)", ok)  # fmt: skip


@register(
    "33.1",
    "Reproducibility: the same config and seed give an identical RESULTS_STEP2.md apart from its Run line",
    "Run `scripts/make_results_step2.py --source synthetic --max-sequences 1 --no-sweeps --no-diodem` twice into "
    "temporary files (seed 1; 6 test sessions) and compare them line by line, except the line starting '**Run:**' "
    "(date, CPU, wall time and compute time live there and nowhere else).",
    "Every table must be traceable to (commit, config, seed); worker scheduling, dictionary order or a random "
    "start must not change a number (brief §8 item 13).",
    "identical apart from the Run line",
    "python scripts/make_results_step2.py --source synthetic --max-sequences 1 --no-sweeps --no-diodem",
)
def _results_repro(quick: bool) -> CheckResult:
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    from .paths import REPO_ROOT

    texts = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(2):
            out = Path(tmp) / f"r{k}.md"
            subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_results_step2.py"), "--source",
                            "synthetic", "--max-sequences", "1", "--no-sweeps", "--no-diodem", "--n-boot", "2000",
                            "--out", str(out), "--figdir", str(Path(tmp) / f"fig{k}")], check=True,
                           capture_output=True)  # fmt: skip
            texts.append([ln for ln in out.read_text(encoding="utf-8").split("\n") if not ln.startswith("**Run:**")])
    diff = [i for i, (a, b) in enumerate(zip(*texts, strict=False)) if a != b]
    same = not diff and len(texts[0]) == len(texts[1])
    return CheckResult(f"{len(texts[0])} lines compared; identical apart from the Run line: {same}"
                       + (f" (first difference at line {diff[0] + 1})" if diff else ""), same)


# ---------------------------------------------------------------- Step 3 (check 39 + n is Step 3 brief §10, item n)

@register(
    "40.1",
    "The page twin draws what the page draws (Step 3 brief §4.10)",
    "Replay the eight golden scenarios (a T-pose then a leg swing; a turn while the phone moves; HipsAlt silent for "
    "2 s; the phone's positions holding exact zeros; each in the velvet and smooth trees) through "
    "mesqsim/twin/page.py and compare every drawn bone quaternion "
    "and hips position after each message, and every number of the recorded BVH, with what js/custom_icm.js and "
    "js/bvh_converter.js wrote when run unmodified in Node (tools/page_golden/run.mjs, three.js r157, the avatar "
    "glbs/scene2.glb, mesquite-mocap 87d7c3b).",
    "The twin's BVH files are judged against Danyal's, so the twin must draw with the page's own rules, defects "
    "included; a port that agrees only on simple inputs would hide exactly the effects Step 3 studies.",
    "quaternions within 1e-5, BVH numbers within 1e-4, the BVH header identical",
    "pytest tests/test_page_twin.py; python tools/page_golden/make_fixtures.py (needs Node)",
)
def _page_twin(quick: bool) -> CheckResult:
    from .twin import golden

    names = golden.cases()
    if not names:
        return CheckResult("no golden fixtures under tests/fixtures/page_golden", None)
    rig_ = golden.rig()
    rs = [golden.compare(golden.load_case(n), rig_) for n in names]
    q = max(max(r["quat"], r["hips"]) for r in rs)
    b = max(r["bvh_numbers"] for r in rs)
    ok = q <= golden.QUAT_TOL and b <= golden.BVH_TOL and all(r["bvh_header_equal"] and r["bvh_count_equal"]
                                                              for r in rs)
    return CheckResult(f"{len(rs)} scenarios, {sum(r['messages'] for r in rs)} messages: largest quaternion or hips "
                       f"difference {q:.1e}, largest BVH number difference {b:.1e}, headers identical "
                       f"{all(r['bvh_header_equal'] for r in rs)}", ok)


@register(
    "41.1",
    "What a pod's sensors feel: the four exact cases of Step 3 brief §4.3",
    "mesqsim.twin.physics.imu_truth on synthetic motion sampled at 1 kHz with the interpolating spline: (a) a still "
    "pod; (b) a constant spin about a body axis; (c) free fall; (d) a site on a circle at constant speed; and "
    "(e) the smoothing spline's response at the cutoff (8 Hz, samples at 120 Hz).",
    "ω and f are the twin's sensor inputs: a wrong frame, sign or derivative here reaches every pod's fusion, and "
    "no later statistic would reveal which.",
    "(a) ω = 0, f = Rᵀ(0, 0, 9.81) within 1e-9; (b) ω equals the spin within 1e-6 rad/s; (c) f = 0 within 1e-6; "
    "(d) the centripetal force within 1e-4 relative; (e) amplitude 0.5 ± 0.05 at the cutoff",
    "pytest tests/test_twin_body.py",
)
def _twin_physics(quick: bool) -> CheckResult:
    from scipy.spatial.transform import Rotation

    from .twin.physics import G_M_S2, imu_truth, position_spline

    t = np.linspace(0.0, 4.0, 4001)
    q0 = Rotation.from_euler("xyz", [20, -35, 110], degrees=True)
    still = imu_truth(t, np.tile(q0.as_quat(), (t.size, 1)), np.tile([0.3, -1.2, 0.9], (t.size, 1)), 225.0)
    e_a = max(np.abs(still.omega_rad_s).max(), np.abs(still.f_m_s2 - q0.inv().apply(-G_M_S2)).max())
    w = np.array([0.4, -1.1, 2.3])
    spin = imu_truth(t, (q0 * Rotation.from_rotvec(np.outer(t, w))).as_quat(), np.zeros((t.size, 3)), 225.0)
    e_b = np.abs(spin.omega_rad_s - w).max()
    p = np.array([0.1, 0.2, 30.0]) + np.outer(t, [1.0, -0.5, 2.0]) + 0.5 * np.outer(t ** 2, G_M_S2)
    e_c = np.abs(imu_truth(t, np.tile(q0.as_quat(), (t.size, 1)), p, 225.0).f_m_s2).max()
    r, om = 0.8, 2 * np.pi * 0.7
    pc = np.stack([r * np.cos(om * t), r * np.sin(om * t), np.ones(t.size)], axis=1)
    circ = imu_truth(t, np.tile([0.0, 0.0, 0.0, 1.0], (t.size, 1)), pc, 225.0, t_eval=np.linspace(0.5, 3.5, 301))
    te = circ.t_s
    want = np.stack([-r * om ** 2 * np.cos(om * te), -r * om ** 2 * np.sin(om * te), np.full(te.size, 9.81)], axis=1)
    e_d = np.abs(circ.f_m_s2 - want).max() / (r * om ** 2)
    ts = np.arange(0.0, 20.0, 1 / 120)
    x = np.sin(2 * np.pi * 8.0 * ts)
    y = position_spline(ts, np.stack([x, x, x], axis=1), 8.0)(ts)[240:-240, 0]
    amp = float(np.sqrt(2 * np.mean(y ** 2)))
    ok = e_a < 1e-9 and e_b < 1e-6 and e_c < 1e-6 and e_d < 1e-4 and abs(amp - 0.5) < 0.05
    return CheckResult(f"(a) {e_a:.1e}; (b) {e_b:.1e} rad/s; (c) {e_c:.1e} m/s²; (d) {e_d:.1e} relative; "
                       f"(e) amplitude {amp:.3f} at the cutoff", ok)


@register(
    "42.1",
    "The chip's fusion in the loop meets its analytic expectations (Step 3 brief §4.5)",
    "VQF six-axis and Mahony at 55 Hz, their own bias estimation off, fed ideal samples (mesqsim.twin.sensors: the "
    "mean rate over each interval, gravity rotated in): (a) a pod turning about every axis, no sensor error; (b) a "
    "pod lying flat turning three times, a bias of 0.5 °/s on its vertical axis, 60 s; (c) the same turns with a 1 % "
    "scale error on that axis; (d) Step 1's simulator (its default config, η = 0) on synthetic:walk:10, its reports "
    "caught before the int16 quantisation and put through the twin's twist rule.",
    "The twin replaces Step 1's analytic heading model by real filters; their heading must still follow the rules "
    "the analysis rests on, with the signs q = Y(-ψ) R fixes: a bias b gives ψ = -b t, a scale error k over a turn Θ "
    "gives ψ = -k Θ (Step 1's b_1 = -b, k_1 = -k), and no error gives a constant ψ.",
    "(a) ψ constant within 1e-6°; (b) ψ(t) = -b t within 1 %; (c) ψ = -k · 3 · 360° within 1 %; (d) Step 1's ψ "
    "within 1e-12 rad",
    "pytest tests/test_twin_fusion.py",
)
def _twin_fusion(quick: bool) -> CheckResult:
    from .twin.checks import analytic_equals_step1, fusion_numbers

    f = fusion_numbers()
    n_pods, worst = analytic_equals_step1()
    ok = n_pods == 15 and worst < 1e-12
    parts = []
    for name, r in f.items():
        ok &= r["psi_range_deg"] < 1e-6 and r["bias_rel_err"] < 0.01 and r["scale_rel_err"] < 0.01
        parts.append(f"{name}: (a) range {r['psi_range_deg']:.1e}°, tilt {r['tilt_max_deg']:.1e}°; (b) "
                     f"{r['bias_psi_end_deg']:.3f}° against {r['bias_expected_deg']:.3f}° ({r['bias_rel_err']:.1e}); "
                     f"(c) {r['scale_psi_deg']:.3f}° against {r['scale_expected_deg']:.1f}° "
                     f"({r['scale_rel_err']:.1e})")
    return CheckResult("; ".join(parts) + f"; (d) {n_pods} pods, worst {worst:.1e} rad", ok)


@register(
    "43.1",
    "The heading labels: one twist rule (Step 3 brief §4.5, §4.11)",
    "mesqsim.twin.labels: ψ = -wrap(2 atan2(E_z, E_w)) of E = q R⁻¹ on the brief's hand check E = Y(-12°); on "
    "q = Y(-ψ) R for 1,000 random R and ψ; and on E = S T and E = T S for 1,000 random swings S about horizontal "
    "axes and twists T. The δ and ε tables are recomputed from ψ once the twin writes sessions (S3-M5).",
    "Every pod's and the phone's heading label is this rule; a sign or order slip would move every δ and ε.",
    "ψ = 12° exactly; ψ recovered within 1e-12 rad and no tilt; the split order-free within 1e-12 rad",
    "pytest tests/test_twin_fusion.py -k 'twist or swing'",
)
def _twin_labels(quick: bool) -> CheckResult:
    from .twin.checks import label_numbers

    r = label_numbers()
    ok = (abs(r["hand_deg"] - 12.0) < 1e-12 and r["recover_max_rad"] < 1e-12 and r["recover_tilt_max_rad"] < 1e-7
          and r["order_max_rad"] < 1e-12 and r["swing_angle_max_err_rad"] < 1e-7)  # fmt: skip
    return CheckResult(f"hand check {r['hand_deg']:.12f}°; ψ recovered within {r['recover_max_rad']:.1e} rad (tilt "
                       f"{r['recover_tilt_max_rad']:.1e}); both orders within {r['order_max_rad']:.1e} rad, the "
                       f"swing's angle within {r['swing_angle_max_err_rad']:.1e} rad", ok)


@register(
    "44.1",
    "The wire, both suits (Step 3 brief §4.6)",
    "Step 1's golden frame (id 2, batt 87, q (0.5, -0.5, 0.25, 0.75), count 0x1234, ms_lo 0xABCD) through "
    "mesqsim.twin.wire, and the message the page's parser makes of it; the old dongle's JSON line at 2-6 decimals "
    "for 500 random reports each, parsed back. Nov 4's exact-repeat share with the fitted json_decimals is added "
    "when S3-M6 fits them.",
    "The page draws what arrives: the twin's frames must be the pod's bytes and the old suit's lines the dongle's "
    "text, or every statistic of the page twin inherits the difference.",
    "the golden bytes and message exactly; every JSON number within half a step of the float32 report",
    "pytest tests/test_twin_wire.py tests/test_codec.py -k golden",
)
def _twin_wire(quick: bool) -> CheckResult:
    from .twin.checks import wire_numbers

    r = wire_numbers()
    worst = r["json_worst_over_half_step"]
    ok = r["golden_ok"] and r["message_ok"] and max(worst.values()) <= 1.0 + 1e-9
    return CheckResult(f"golden frame {r['golden_hex']} ({'equal' if r['golden_ok'] else 'DIFFERENT'}), parsed "
                       f"message {'equal' if r['message_ok'] else 'DIFFERENT'}; JSON worst error / half step: "
                       + ", ".join(f"{d} decimals {v:.3f}" for d, v in worst.items()), ok)


@register(
    "45.1",
    "The phone app as its Swift code, 95 bytes, and the page's zero guard (Step 3 brief §4.7)",
    "mesqsim.twin.phone_app against tests/fixtures/app_golden (the app's unmodified Core/*.swift at mesquite-mocap "
    "87d7c3b, compiled by tools/app_golden): 1,260 pipeline frames, 2,110 encoded poses, 410 matrix conversions, "
    "4 lever-arm fits. Then 10,000 uniformly random poses (boxes of ±5, ±50, ±500 m, and orientation only) through "
    "the port's encoder; the nudged zero; and the zero-guard scenarios of the page golden test replayed through "
    "the page twin.",
    "The phone drives the drawn root; the twin's messages must be the app's to the byte, never exceed the hub's "
    "95-byte record, and meet the page's habit of dropping a position that holds an exact 0.",
    "no difference from Swift (0 ulps); no message over 95 bytes; 0.0004, -0.0004, 0, -0 sent as 0.001, -0.001, "
    "0.001, 0.001; the hips never move on a position with a 0, as on the page (1e-5)",
    "pytest tests/test_twin_phone.py tests/test_page_twin.py -k zero",
)
def _twin_phone(quick: bool) -> CheckResult:
    from .twin.checks import phone_numbers

    r = phone_numbers()
    p = r["port"]
    port_ok = p["max_ulps"] == 0 and not (p["text_diffs"] or p["encode_diffs"] or p["column_diffs"]
                                          or p["lever_diffs"])  # fmt: skip
    guard_ok = all(g["held"] and g["diff"] <= 1e-5 for g in r["guard"])
    ok = port_ok and r["longest_bytes"] <= 95 and r["nudged"] == ["0.001", "-0.001", "0.001", "0.001"] and guard_ok
    g = "; ".join(f"{x['tree']}: {x['zero_positions']} of {x['messages']} positions with a 0, hips held "
                  f"{x['held']}, largest difference from the page {x['diff']:.1e}" for x in r["guard"])
    return CheckResult(f"port vs Swift: {p['rows']} frames, {p['encode']} encodes, {p['columns']} conversions, "
                       f"{p['lever_cases']} lever fits, {p['text_diffs'] + p['encode_diffs'] + p['lever_diffs']} "
                       f"differences, {p['max_ulps']} ulps; {r['random_poses']} random poses, longest "
                       f"{r['longest_bytes']} bytes; nudged {r['nudged']}; {g}", ok)


@register(
    "49.1",
    "The analysis of 2026-11-14 is reproduced (Step 3 brief §5.3)",
    "Φ1-Φ12 of Danyal's seven recordings (mesqsim.real: the BVH and Motive readers, marker naming and following, "
    "the clock alignment, the statistics) against the 62 reference values of §5.3, each within its tolerance.",
    "The twin will be judged by these statistics, so the code that measures them must first reproduce the "
    "analysis it inherits; every value it does not reproduce needs a written reason (Step 3 brief §5.3).",
    "every value reproduced or listed with its reason in mesqsim.real.recordings.EXPLAINED_DIFFERENCES",
    "python -m mesqsim real-stats; pytest -m real tests/test_real_recordings.py",
)
def _real_reproduced(quick: bool) -> CheckResult:
    from .real import recordings
    from .real.analysis import analyse, compare

    if not recordings.available():
        return CheckResult(f"recordings not under {recordings.real_suit_dir()}", None)
    rows = compare({r.id: analyse(r) for r in recordings.RECORDINGS})
    failed = {(x["statistic"], x["recording"]) for x in rows if not x["ok"]}
    new = sorted(failed - set(recordings.EXPLAINED_DIFFERENCES))
    return CheckResult(f"{len(rows) - len(failed)} of {len(rows)} reference values within tolerance; "
                       f"{len(failed)} explained differences" + (f"; UNEXPLAINED: {new}" if new else ""), not new)
