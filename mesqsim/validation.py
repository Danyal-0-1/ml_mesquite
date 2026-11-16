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
    turns = cfg["error_model.idle_turns_sigma_per_sqrt_min"]
    out = []
    for seed in range(n_sessions):
        plan = plan_session(cfg, seed, mot)
        if hook:
            hook(plan)
        r_ep, r_walk = stream(seed, "epochs"), stream(seed, "walk")
        for p in plan.pods:
            eps = pod_epochs(cfg, plan, mot, p, r_ep)
            hd = build_pod_heading(plan, mot, p.pod_id, eps, p.drift_rad_per_s, p.scale_factor, arw, turns, r_walk)
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
