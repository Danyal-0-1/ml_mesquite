"""The numbers behind validation checks 42.1-45.1 (Step 3 brief §10, items 3-6): fusion, labels, the wire, the phone.

Each function returns plain numbers, so ``mesqsim/validation.py`` states them and the tests can assert on them.
"""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from ..conventions import exp_map, quat_conj, quat_mul, quat_rotate, wrap_rad, yaw_quat
from .fusion import mahony, vqf6d
from .labels import heading_error_rad, report_labels, tilt_error_rad
from .sensors import ideal_gyro_rad_s

__all__ = ["RATE_HZ", "FILTERS", "fusion_numbers", "analytic_equals_step1", "label_numbers", "wire_numbers",
           "phone_numbers"]

RATE_HZ = 55.0
DT = 1.0 / RATE_HZ
G_UP = np.array([0.0, 0.0, 9.81])
FILTERS = {"vqf6d": lambda g, a: vqf6d(g, a, DT, 3.0, False, False),
           "mahony": lambda g, a: mahony(g, a, DT, 1 / 3, 0.0)}
"""Both filters with their own bias estimation switched off (VQF's rest and motion estimation, Mahony's integral)."""


def _flat_turns(seconds=60.0, turns=3, still_s=5.0):
    t = np.arange(0.0, seconds, DT)
    th = np.clip((t - still_s) / (seconds - 2 * still_s), 0.0, 1.0) * turns * 2 * np.pi
    r = yaw_quat(th)
    return t, r, ideal_gyro_rad_s(np.vstack([r[:1], r]), DT), quat_rotate(quat_conj(r), G_UP)


def _tumbling(seconds=40.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, DT)
    w = np.stack([a * np.sin(2 * np.pi * f * t + p) for a, f, p in rng.uniform([0.5, 0.1, 0], [2.0, 0.6, 6], (3, 3))])
    r = Rotation.from_euler("zyx", [30, 10, -20], degrees=True)
    q = [r.as_quat()]
    for k in range(1, t.size):
        r = r * Rotation.from_rotvec(w[:, k] * DT)
        q.append(r.as_quat())
    q = np.array(q)
    return t, q, ideal_gyro_rad_s(np.vstack([q[:1], q]), DT), quat_rotate(quat_conj(q), G_UP)


def fusion_numbers() -> dict:
    """Per filter: (a) the range of ψ with perfect samples while the pod turns about every axis; (b) a bias of
    0.5 °/s on the pod axis that stays vertical, ψ(60 s) against -b t; (c) a scale error of 1 % on it over three
    turns, the change of ψ against -k · 3 · 360°."""
    out = {}
    b, k = np.deg2rad(0.5), 0.01
    _, q, gyr, acc = _tumbling()
    t, r, g2, a2 = _flat_turns()
    for name, f in FILTERS.items():
        psi, tilt = report_labels(f(gyr, acc).q, q)
        pb, _ = report_labels(f(g2 + [0.0, 0.0, b], a2).q, r)
        db = np.unwrap(pb) - pb[0]
        want_b = -b * (t - t[0])
        pk, _ = report_labels(f(g2 * [1.0, 1.0, 1.0 + k], a2).q, r)
        dk = np.unwrap(pk)[-1] - pk[0]
        out[name] = {"psi_range_deg": float(np.rad2deg(np.ptp(np.unwrap(psi)))),
                     "tilt_max_deg": float(np.rad2deg(tilt.max())),
                     "bias_psi_end_deg": float(np.rad2deg(db[-1])), "bias_expected_deg": float(np.rad2deg(want_b[-1])),
                     "bias_rel_err": float(np.abs(db - want_b).max() / abs(want_b[-1])),
                     "scale_psi_deg": float(np.rad2deg(dk)), "scale_expected_deg": -k * 3 * 360.0,
                     "scale_rel_err": float(abs(dk / (-k * 3 * 2 * np.pi) - 1))}  # fmt: skip
    return out


def analytic_equals_step1(seed: int = 4) -> tuple[int, float]:
    """Step 1's simulator (its default config, η = 0) on ``synthetic:walk:10``: its reports, caught before the int16
    quantisation, through the twin's twist rule, against its own ψ labels. Returns (pods, worst |Δψ| rad)."""
    from ..config import load_config
    from ..paths import REPO_ROOT
    from ..sim import simulate as sim
    from ..sim.motion import MotionSource

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides({"error_model.tilt_sigma_deg": 0.0})
    seen = []
    real = sim.quantize_component
    with mock.patch.object(sim, "quantize_component", lambda q: (seen.append(np.array(q)), real(q))[1]):
        d = sim.simulate_session(cfg, seed, MotionSource.from_spec("synthetic:walk:10"))
    worst = 0.0
    for i, q in enumerate(seen):
        lab = d.labels_ticks[d.labels_ticks["pod_id"] == i]
        psi, _ = report_labels(q, lab[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]].to_numpy())
        worst = max(worst, float(np.abs(wrap_rad(psi - lab["psi_rad"].to_numpy())).max()))
    return len(seen), worst


def label_numbers() -> dict:
    """The hand check; ψ recovered from Y(-ψ) · R over random R; the swing-twist split in both orders."""
    rng = np.random.default_rng(1)
    hand = float(np.rad2deg(heading_error_rad(yaw_quat(np.deg2rad(-12.0)))))
    r = Rotation.random(1000, random_state=rng).as_quat()
    psi = rng.uniform(-np.pi, np.pi, 1000)
    got, tilt = report_labels(quat_mul(yaw_quat(-psi), r), r)
    swing = exp_map(np.c_[rng.normal(0, 0.3, (1000, 2)), np.zeros(1000)])
    twist = rng.uniform(-3, 3, 1000)
    order = max(float(np.abs(wrap_rad(heading_error_rad(e) + twist)).max())
                for e in (quat_mul(swing, yaw_quat(twist)), quat_mul(yaw_quat(twist), swing)))  # fmt: skip
    tilt_err = float(np.abs(tilt_error_rad(quat_mul(swing, yaw_quat(twist))) - 2 * np.arccos(swing[:, 3])).max())
    return {"hand_deg": hand, "recover_max_rad": float(np.abs(wrap_rad(got - psi)).max()),
            "recover_tilt_max_rad": float(tilt.max()), "order_max_rad": order, "swing_angle_max_err_rad": tilt_err}


def wire_numbers() -> dict:
    """Step 1's golden frame through the twin's wire, and the old JSON line's rounding over random reports."""
    from .wire import binary16_bytes, binary16_messages, json_line

    buf = binary16_bytes(2, 87, (0.5, -0.5, 0.25, 0.75), 0x1234, 0xABCD)
    (m,) = binary16_messages(buf)
    rng = np.random.default_rng(0)
    worst = {}
    for d in (2, 3, 4, 5, 6):
        q = rng.normal(size=(500, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        err = 0.0
        for qi in q:
            o = json.loads(json_line(3, qi, 0, 0, 50, d))
            err = max(err, float(np.abs(np.array([o[c] for c in "xyzw"]) - qi.astype(np.float32)).max()))
        worst[d] = err / (0.5 * 10.0**-d)
    return {"golden_hex": buf.hex(), "golden_ok": buf.hex() == "aa550257ff3f01c0ff1fff5f3412cdab",
            "message_ok": m["bone"] == "HipsAlt" and m["x"] == 16383 / 32767 and m["millis"] == 0xABCD,
            "json_worst_over_half_step": worst}


def phone_numbers(n_random: int = 10_000) -> dict:
    """The port against the Swift core's fixture; the 95-byte budget over random poses; the nudged zero; and the
    page's zero guard as the page twin replays it on the zero-guard fixtures."""
    from . import golden, phone_app
    from .app_golden import compare, load_fixture

    port = compare(load_fixture())
    rng = np.random.default_rng(7)
    longest, n = 0, 0
    for box in (5.0, 50.0, 500.0, None):
        q = Rotation.random(n_random // 4, random_state=rng).as_quat()
        p = rng.uniform(-(box or 1.0), box or 1.0, (n_random // 4, 3))
        for qi, pi in zip(q, p, strict=True):
            msg = phone_app.encode(tuple(qi), None if box is None else tuple(pi))
            longest, n = max(longest, msg.bytes), n + 1
    nudged = [phone_app.fixed(v, 3, avoid_zero=True) for v in (0.0004, -0.0004, 0.0, -0.0)]
    guard = []
    rig_ = golden.rig()
    for tree in ("velvet", "smooth"):
        case = golden.load_case(f"phone_zero_guard_{tree}")
        page, per, _ = golden.replay(case, rig_)
        hips = [(e["msg"], r) for e, r in zip(case["input"]["events"], per, strict=True)
                if e["type"] == "msg" and e["msg"]["bone"] == "Hips"]
        ref = [r for e, r in zip(case["input"]["events"], case["golden"]["per_event"], strict=True)
               if e["type"] == "msg" and e["msg"]["bone"] == "Hips"]
        zero = [not (m["px"] and m["py"] and m["pz"]) for m, _ in hips]
        held = all(hips[k][1][4:7] == hips[k - 1][1][4:7] for k in range(1, len(hips)) if zero[k])
        diff = max(max(abs(a - b) for a, b in zip(x[1], y, strict=True)) for x, y in zip(hips, ref, strict=True))
        guard.append({"tree": tree, "messages": len(hips), "zero_positions": sum(zero), "held": held, "diff": diff})
    return {"port": port, "random_poses": n, "longest_bytes": longest, "nudged": nudged, "guard": guard}
