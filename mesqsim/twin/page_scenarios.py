"""Scripted message sequences for the page golden test (Step 3 brief §4.10), deterministic for a seed.

Each scenario is a list of events in arrival order, as the page would see them: pod messages ``{bone, x, y, z, w,
batt, count, millis}``, phone messages ``{bone: "Hips", x, y, z, w, px, py, pz}``, one ``calibrate`` (the page's
Set T-pose) and a recording (``record_start``, a ``record_tick`` every 33 ms, ``record_stop``).

A pod reports q_i(t) = Y(-ψ_i) · S_i(t) · M_i in its own private world (+Z up): a random north ψ_i, a random
mounting M_i and its segment's orientation S_i(t). The phone reports in ARKit's +Y-up Zero frame, metres. The motions
are simple on purpose; the test is that the page twin draws what the page draws, whatever it is given.

* ``tpose_leg_swing``: a T-pose, Set T-pose at 1.0 s, then the left thigh swings ±30° and the left knee bends.
* ``turn_phone_moving``: a T-pose, then the whole body turns 90° about the vertical while the phone walks 1 m.
* ``hipsalt_silent``: the pelvis sways; HipsAlt sends nothing from 2.0 s to 4.0 s, so the phone drives the hips.
* ``phone_zero_guard``: the phone sways about its Zero point with positions rounded plainly to 1 mm, so many contain
  an exact 0 (or -0), and py is 0 for 0.2 s: the page's zero guard (Step 3 brief §10, item 6).
"""

from __future__ import annotations

import numpy as np

from ..conventions import exp_map, quat_canonical, quat_mul, yaw_quat

__all__ = ["PODS", "SCENARIOS", "make_scenario"]

PODS = ("HipsAlt", "Spine", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot")
POD_HZ = 16.0
PHONE_HZ = 30.0
TICK_MS = 33.0
SCENARIOS = {
    # name: (duration s, calibrate at s, record from s, record to s)
    "tpose_leg_swing": (3.5, 1.0, 1.1, 3.4),
    "turn_phone_moving": (3.5, 0.8, 0.9, 3.4),
    "hipsalt_silent": (4.6, 0.8, 0.9, 4.5),
    "phone_zero_guard": (3.0, 0.8, 0.9, 2.9),
}


def _axis_angle(axis, angle_rad):
    axis = np.asarray(axis, dtype=float)
    return exp_map(axis / np.linalg.norm(axis) * angle_rad)


def _segment(name: str, scenario: str, t: float) -> np.ndarray:
    """S_i(t): the segment's orientation in the world (+Z up; identity in the T-pose)."""
    q = np.array([0.0, 0.0, 0.0, 1.0])
    if scenario == "tpose_leg_swing" and t >= 1.5:
        swing = np.deg2rad(30.0) * np.sin(2 * np.pi * 0.8 * (t - 1.5))
        if name in ("LeftUpLeg", "LeftLeg", "LeftFoot"):
            q = _axis_angle([1, 0, 0], swing)
        if name in ("LeftLeg", "LeftFoot"):
            knee = -np.deg2rad(40.0) * max(0.0, np.sin(2 * np.pi * 0.8 * (t - 1.5)))
            q = quat_mul(q, _axis_angle([1, 0, 0], knee))
    elif scenario == "turn_phone_moving" and t >= 1.0:
        q = yaw_quat(np.deg2rad(90.0) * min(1.0, (t - 1.0) / 2.5))
        if name in ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm"):
            q = quat_mul(q, _axis_angle([0, 1, 0], np.deg2rad(15.0) * np.sin(2 * np.pi * (t - 1.0))))
    elif scenario == "hipsalt_silent" and t >= 1.0:
        q = yaw_quat(np.deg2rad(20.0) * np.sin(2 * np.pi * 0.4 * (t - 1.0)))
    return q


def _phone(scenario: str, t: float) -> tuple[np.ndarray, np.ndarray]:
    """The phone's orientation (x, y, z, w) and position (m) in ARKit's Zero frame (+Y up, forward -Z)."""
    yaw, pos = 0.0, np.array([0.05, 1.0, 0.02])
    if scenario == "turn_phone_moving" and t >= 1.0:
        u = min(1.0, (t - 1.0) / 2.5)
        yaw = np.deg2rad(90.0) * u
        pos = pos + np.array([0.0, 0.01 * np.sin(4 * np.pi * t), -1.0 * u])
    elif scenario == "hipsalt_silent" and t >= 1.0:
        yaw = np.deg2rad(20.0) * np.sin(2 * np.pi * 0.4 * (t - 1.0))
        pos = pos + np.array([0.02 * np.sin(t), 0.0, -0.1 * (t - 1.0)])
    elif scenario == "tpose_leg_swing":
        pos = pos + np.array([0.0, 0.005 * np.sin(3 * t), 0.0])
    elif scenario == "phone_zero_guard":
        # positions as plain rounding to 1 mm would send them around the Zero point: many are exactly 0 or -0, and
        # py is 0 for 0.2 s; the page skips such a position (js/custom_icm.js:931), the app nudges it (MATH.md §5.3)
        yaw = np.deg2rad(10.0) * np.sin(2 * np.pi * 0.5 * t)
        raw = np.array([0.003 * np.sin(2 * np.pi * 0.7 * t), 0.0 if 1.5 <= t < 1.7 else 1.0,
                        -0.002 * np.cos(2 * np.pi * 0.4 * t)])
        pos = np.round(raw, 3)
    q = np.array([0.0, np.sin(yaw / 2), 0.0, np.cos(yaw / 2)])  # about ARKit's +Y
    return q, pos


def make_scenario(name: str, tree: str, seed: int = 0) -> dict:
    """The events of scenario ``name`` for tree ``tree`` (the tree only labels the scenario; the page applies it)."""
    duration, t_cal, t_rec0, t_rec1 = SCENARIOS[name]
    rng = np.random.default_rng(seed)
    north = {p: rng.uniform(-np.pi, np.pi) for p in PODS}
    mount = {p: exp_map(rng.normal(size=3) * 0.6) for p in PODS}
    jitter = np.random.default_rng(seed + 1)
    events = []  # (t_ms, tie-break, event)
    for k, pod in enumerate(PODS):
        n_msg = int(duration * POD_HZ)
        for n in range(n_msg):
            t = n / POD_HZ + k * 0.0025
            if name == "hipsalt_silent" and pod == "HipsAlt" and 2.0 <= t < 4.0:
                continue
            s = quat_mul(_segment(pod, name, t), exp_map(jitter.normal(size=3) * 1e-3))
            q = quat_canonical(quat_mul(quat_mul(yaw_quat(-north[pod]), s), mount[pod]))
            events.append((t * 1000.0, 1, {"type": "msg", "msg": {
                "bone": pod, "x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3]),
                "batt": 0.9, "count": n + 1, "millis": 0}}))
    for n in range(int(duration * PHONE_HZ)):
        t = n / PHONE_HZ + 0.001
        q, p = _phone(name, t)
        events.append((t * 1000.0, 0, {"type": "msg", "msg": {
            "bone": "Hips", "x": float(q[0]), "y": float(q[1]), "z": float(q[2]), "w": float(q[3]),
            "px": float(p[0]), "py": float(p[1]), "pz": float(p[2])}}))
    events.append((t_cal * 1000.0, 2, {"type": "calibrate"}))
    events.append((t_rec0 * 1000.0, 3, {"type": "record_start"}))
    t = t_rec0 * 1000.0 + TICK_MS
    while t < t_rec1 * 1000.0:
        events.append((t, 4, {"type": "record_tick"}))
        t += TICK_MS
    events.append((t_rec1 * 1000.0, 5, {"type": "record_stop"}))
    events.sort(key=lambda e: (e[0], e[1]))
    return {"scenario": name, "tree": tree, "seed": seed,
            "events": [dict(e[2], t_ms=float(e[0])) for e in events]}
