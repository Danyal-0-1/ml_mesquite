"""The phone app's golden test (Step 3 brief §4.7): scripted inputs, the real Swift core's outputs, and the port's.

``tools/app_golden/make_fixtures.py`` compiles ``tools/app_golden/main.swift`` with the app's own ``Core/*.swift``
from mesquite-mocap ``87d7c3b`` (unmodified), runs it on :func:`golden_inputs`, and writes the inputs and the real
outputs to ``tests/fixtures/app_golden/app_golden.json``. :func:`compare` replays the same inputs through
:mod:`mesqsim.twin.phone_app` and counts every difference, so the test runs where Swift is missing.

Inputs, all in ARKit's frames (+Y up, metres), every double as its IEEE bit pattern in hexadecimal:

* ``cases``: three :class:`PosePipeline` runs at about 60 Hz with timestamp jitter. A phone on the front of the
  belt (screen against the body, rear camera forward, portrait), its camera matrix rounded to float32 as ARKit's
  ``simd_float4x4`` is, turned into a quaternion by the port's Shepperd method (the ``columns`` group checks that
  step against Swift). ``walk``: initializing, the automatic Zero, a requested Zero in the T-pose, a walk with a
  left turn, lean and sway (velvet family, forward, 30 Hz). ``episodes``: excessive motion, relocalizing, not
  available, insufficient features, a requested Zero while a sideways lean tips the camera toward the floor (it
  fails), one requested during limited tracking (it waits), the link refusing some sends (thorny, the camera facing
  left, a measured lever, 60 Hz). ``sway``: a 2 cm sway about the Zero point, so positions round to zero,
  and the camera quaternion's sign flipping every frame (matched, backward, 15 Hz, hip height 0.95 m).
* ``encode``: 2,100 uniformly random poses in boxes of ±5, ±50 and ±500 m, and edge cases (exact zeros, ties,
  -0, a non-unit or non-finite quaternion, no position).
* ``columns``: 400 random rotation matrices and the branch edges of Shepperd's method.
* ``lever``: the app's own lever-arm cases (``MATH.md`` §4.5): a full turn, a half turn, a 36° twist, too few samples.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..paths import REPO_ROOT
from . import phone_app as app

__all__ = ["FIXTURE", "hx", "unhx", "golden_inputs", "load_fixture", "replay", "compare"]

FIXTURE = REPO_ROOT / "tests" / "fixtures" / "app_golden" / "app_golden.json"
# camera axes in the body frame B (+X right, +Y up, -Z forward): X down, Y right, Z into the body (sites.py)
_M_CAM = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def hx(v: float) -> str:
    return format(struct.unpack("<Q", struct.pack("<d", float(v)))[0], "x")


def unhx(s: str) -> float:
    return struct.unpack("<d", struct.pack("<Q", int(s, 16)))[0]


def _f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32).astype(np.float64)


def _camera(heading_rad: float, lean_rad: float, roll_rad: float) -> np.ndarray:
    """Camera -> ARKit rotation matrix for a body turned by ``heading`` about +Y, leaning forward and sideways."""
    body = Rotation.from_rotvec([0.0, heading_rad, 0.0]) * Rotation.from_rotvec([-lean_rad, 0.0, 0.0]) \
        * Rotation.from_rotvec([0.0, 0.0, roll_rad])
    return body.as_matrix() @ _M_CAM


def _frame_row(t, pos, mat, tracking="normal", zero=0, accept=1, flip=False) -> list:
    m = _f32(mat)
    q = app.q_from_columns(tuple(m[:, 0]), tuple(m[:, 1]), tuple(m[:, 2]))
    if flip:
        q = app.q_neg(q)
    p = _f32(pos)
    return [hx(t), hx(p[0]), hx(p[1]), hx(p[2]), *(hx(c) for c in q), tracking, int(zero), int(accept)]


def _settings(lever, hip_height, facing, tree, rate_hz) -> dict:
    return {"lever": [hx(c) for c in lever], "hipHeight": hx(hip_height), "facing": facing, "tree": tree,
            "rateHz": hx(rate_hz)}


def _case_walk(rng) -> dict:
    rows, n = [], 480
    t = np.arange(n) / 60.0 + rng.uniform(-1.5e-3, 1.5e-3, n)
    for k in range(n):
        tk = t[k]
        walk = max(0.0, tk - 3.0)
        heading = np.deg2rad(90.0) * min(1.0, walk / 4.0)
        lean = np.deg2rad(4.0) * math.sin(2 * math.pi * 1.8 * walk) if walk > 0 else 0.0
        mat = _camera(heading, lean, np.deg2rad(2.0) * math.sin(2 * math.pi * 0.9 * walk))
        fwd = np.array([-math.sin(heading), 0.0, -math.cos(heading)])
        bob = 0.02 * math.sin(2 * math.pi * 1.8 * walk)
        hip = np.array([0.0, 1.0, 0.0]) + 1.0 * walk * fwd + np.array([0.0, bob, 0.0])
        pos = hip + mat @ np.array([0.0, 0.0, -0.1])  # the camera, 10 cm in front of the hip centre
        tracking = "limited:initializing" if tk < 0.5 else "normal"
        rows.append(_frame_row(tk, pos, mat, tracking, zero=int(k == 120)))
    return {"name": "walk", "settings": _settings((0.0, 0.0, 0.1), 1.0, "forward", "velvetFamily", 30.0),
            "frames": rows}


# the camera turned about the body's vertical, which is the camera's -X: a quarter turn looks left, a half backward
_LOOK_LEFT = Rotation.from_rotvec([-np.pi / 2, 0.0, 0.0]).as_matrix()
_LOOK_BACK = Rotation.from_rotvec([np.pi, 0.0, 0.0]).as_matrix()


def _case_episodes(rng) -> dict:
    rows, n = [], 420
    t = 100.0 + np.arange(n) / 60.0 + rng.uniform(-1.5e-3, 1.5e-3, n)
    for k in range(n):
        tk = t[k] - 100.0
        heading = np.deg2rad(30.0) + np.deg2rad(60.0) * math.sin(2 * math.pi * 0.2 * tk)
        lean = np.deg2rad(10.0) * math.sin(2 * math.pi * 0.5 * tk)
        roll = np.deg2rad(80.0) if 2.0 <= tk < 2.4 else 0.0  # a sideways lean tips the left-looking camera down
        mat = _camera(heading, lean, roll) @ _LOOK_LEFT
        pos = np.array([0.3 * math.sin(tk), 1.1, -0.5 * tk]) + rng.normal(0.0, 0.002, 3)
        if 1.0 <= tk < 1.5:
            tracking = "limited:excessiveMotion"
        elif 3.0 <= tk < 3.4:
            tracking = "limited:relocalizing"
        elif 4.0 <= tk < 4.2:
            tracking = "not_available"
        elif 5.0 <= tk < 5.3:
            tracking = "limited:insufficientFeatures"
        else:
            tracking = "normal"
        zero = int(k in (130, 300))  # 2.17 s: the camera looks at the floor, so the Zero fails; 5.0 s: limited
        rows.append(_frame_row(t[k], pos, mat, tracking, zero=zero, accept=int(k % 7 != 3)))
    return {"name": "episodes", "settings": _settings((0.02, -0.05, 0.12), 1.0, "left", "thorny", 60.0),
            "frames": rows}


def _case_sway(rng) -> dict:
    rows, n = [], 360
    t = 7.0 + np.arange(n) / 60.0 + rng.uniform(-1.5e-3, 1.5e-3, n)
    for k in range(n):
        tk = t[k] - 7.0
        heading = np.pi + np.deg2rad(3.0) * math.sin(2 * math.pi * 0.3 * tk)
        mat = _camera(heading, 0.0, 0.0) @ _LOOK_BACK
        pos = np.array([0.02 * math.sin(2 * math.pi * 0.5 * tk), 0.9 + 0.0004 * math.sin(3 * tk),
                        0.02 * math.cos(2 * math.pi * 0.5 * tk) - 0.02])
        rows.append(_frame_row(t[k], pos, mat, "normal", zero=0, accept=1, flip=bool(k % 2)))
    return {"name": "sway", "settings": _settings((0.0, 0.0, 0.08), 0.95, "backward", "matched", 15.0),
            "frames": rows}


def _encode_inputs(rng) -> list:
    out = []
    for box in (5.0, 50.0, 500.0):
        q = Rotation.random(700, random_state=rng).as_quat()
        q[rng.random(700) < 0.5] *= -1.0
        p = rng.uniform(-box, box, (700, 3))
        for qi, pi in zip(q, p, strict=True):
            out.append([*(hx(c) for c in qi), *(hx(c) for c in pi), 1])
    edge = [((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0), 1), ((0.0, 0.0, 0.0, 1.0), (-0.0, 0.0004, -0.0005), 1),
            ((0.00005, -0.00005, 0.00015, 1.0), (0.0005, -0.0005, 0.0015), 1), ((1e-5, 2e-5, -3e-5, 1.0), None, 0),
            ((0.5, 0.5, 0.5, 0.5), (9.9995, -9.9995, 99.995), 1), ((2.0, 0.0, 0.0, 0.0), (1e6, 1e6, 1e6), 1),
            ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 1), ((0.3, -0.4, 0.5, 0.7), (math.inf, 0.0, 1.0), 1),
            ((math.nan, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0), 1), ((-0.70710678, 0.0, 0.70710678, 0.0), (1e9, -1e9, 0.5), 1)]
    for q, p, has in edge:
        p = (0.0, 0.0, 0.0) if p is None else p
        out.append([*(hx(c) for c in q), *(hx(c) for c in p), has])
    return out


def _columns_inputs(rng) -> list:
    mats = list(Rotation.random(400, random_state=rng).as_matrix())
    for axis in np.eye(3):  # half turns (trace -1) and near-half turns: every branch of Shepperd's method
        for ang in (np.pi, np.pi - 1e-7, np.pi / 2):
            mats.append(Rotation.from_rotvec(axis * ang).as_matrix())
    mats.append(np.eye(3))
    return [[hx(c) for c in _f32(m).T.ravel()] for m in mats]


def _lever_inputs(rng) -> list:
    """The app's lever-arm check (MATH.md §4.5): a phone 12 cm in front of the hip centre and 4 cm above, tilted
    10°, turned at 60 Hz with 2 mm position noise."""
    lever_body = np.array([0.0, -0.04, 0.12])  # camera -> hip, in B (the hip is behind and below the camera)
    tilt = Rotation.from_rotvec([np.deg2rad(10.0), 0.0, 0.0])
    cases = []
    for arc_deg, n in ((370.0, 600), (190.0, 300), (36.0, 300), (360.0, 40)):
        samples = []
        hip = np.array([0.4, 0.95, -1.2])
        for k in range(n):
            th = np.deg2rad(arc_deg) * k / (n - 1)
            body = Rotation.from_rotvec([0.0, th, 0.0]) * tilt
            r_cam = body.as_matrix() @ _M_CAM
            p = hip - r_cam @ (_M_CAM.T @ lever_body) + rng.normal(0.0, 0.002, 3)
            q = app.q_from_columns(tuple(r_cam[:, 0]), tuple(r_cam[:, 1]), tuple(r_cam[:, 2]))
            samples.append([*(hx(c) for c in p), *(hx(c) for c in q)])
        cases.append({"prior": [hx(0.0), hx(0.0), hx(0.1)], "samples": samples})
    return cases


def golden_inputs(seed: int = 20261119) -> dict:
    rng = np.random.default_rng(seed)
    return {"cases": [_case_walk(rng), _case_episodes(rng), _case_sway(rng)], "encode": _encode_inputs(rng),
            "columns": _columns_inputs(rng), "lever": _lever_inputs(rng)}


def load_fixture(path: Path = FIXTURE) -> dict:
    return json.loads(Path(path).read_text())


def replay(inputs: dict) -> dict:
    """The port's outputs for ``inputs``, in the harness's format."""
    out = {"cases": [], "encode": [], "columns": [], "lever": []}
    for c in inputs["cases"]:
        s = c["settings"]
        frame = app.HipsFrame(lever=tuple(unhx(v) for v in s["lever"]), hip_height=unhx(s["hipHeight"]),
                              facing=s["facing"])  # fmt: skip
        pipe = app.PosePipeline(frame=frame, tree=s["tree"], rate_hz=unhx(s["rateHz"]))
        rows = []
        for fr in c["frames"]:
            f = app.FrameInput(unhx(fr[0]), tuple(unhx(v) for v in fr[1:4]), tuple(unhx(v) for v in fr[4:8]),
                               app.Tracking(fr[8]))  # fmt: skip
            if fr[9] == 1:
                pipe.request_zero()
            o = pipe.process(f)
            row = [None if o.reported is None else [hx(v) for v in (*o.reported[0], *o.reported[1])],
                   None if o.message is None else o.message.text, None if o.message is None else o.message.level,
                   None if o.zero is None else [o.zero.epoch, hx(o.zero.heading), int(o.zero_was_requested)],
                   o.zero_error]  # fmt: skip
            if o.message is not None and fr[10] == 1:
                pipe.mark_sent(f.time)
            rows.append(row)
        out["cases"].append({"name": c["name"], "out": rows})
    for e in inputs["encode"]:
        q = tuple(unhx(v) for v in e[0:4])
        p = tuple(unhx(v) for v in e[4:7]) if e[7] == 1 else None
        m = app.encode(q, p)
        out["encode"].append(None if m is None else [m.text, m.level])
    for m in inputs["columns"]:
        v = [unhx(x) for x in m]
        out["columns"].append([hx(c) for c in app.q_from_columns(tuple(v[0:3]), tuple(v[3:6]), tuple(v[6:9]))])
    for lv in inputs["lever"]:
        fit = app.LeverArmFit(prior=tuple(unhx(v) for v in lv["prior"]))
        for s in lv["samples"]:
            fit.add(tuple(unhx(v) for v in s[0:3]), tuple(unhx(v) for v in s[3:7]))
        try:
            r = fit.solve()
            out["lever"].append({"ok": 1, "lever": [hx(v) for v in r.lever], "pivot": [hx(v) for v in r.pivot],
                                 "rms": hx(r.rms), "headingRange": hx(r.heading_range), "samples": r.samples})
        except app.LeverFitFailure as e:
            kind, value = e.args
            out["lever"].append({"ok": 0, "failure": kind, "value": hx(float(value))})
    return out


def _ulps(a: str, b: str) -> int:
    """Units in the last place between two doubles given as bit patterns (+0 and -0 are 0 apart)."""

    def line(i: int) -> int:  # sign-magnitude -> a monotonic integer line
        return i if i < (1 << 63) else (1 << 63) - i

    return abs(line(int(a, 16)) - line(int(b, 16)))


def _lever_values(r: dict) -> list[str]:
    if r["ok"] == 1:
        return [*r["lever"], *r["pivot"], r["rms"], r["headingRange"], hx(r["samples"])]
    return [r["value"]]


def compare(fixture: dict) -> dict:
    """Replay the fixture's inputs and count differences from the Swift outputs: texts, levels, Zero epochs and
    errors must be equal; ``number_diffs`` counts outputs whose numbers differ at all, ``max_ulps`` the largest
    difference in units in the last place."""
    mine = replay(fixture["inputs"])
    ref = fixture["outputs"]
    n_text = n_num = n_rows = 0
    ulps = [0]
    for cm, cr in zip(mine["cases"], ref["cases"], strict=True):
        for rm, rr in zip(cm["out"], cr["out"], strict=True):
            n_rows += 1
            n_text += int(rm[1] != rr[1] or rm[2] != rr[2] or rm[4] != rr[4] or (rm[0] is None) != (rr[0] is None)
                          or (rm[3] is None) != (rr[3] is None))  # fmt: skip
            nums = []
            if rm[0] is not None and rr[0] is not None:
                nums += list(zip(rm[0], rr[0], strict=True))
            if rm[3] is not None and rr[3] is not None:
                n_text += int(rm[3][0] != rr[3][0] or rm[3][2] != rr[3][2])
                nums.append((rm[3][1], rr[3][1]))
            u = max((_ulps(a, b) for a, b in nums), default=0)
            ulps.append(u)
            n_num += int(u > 0)
    n_enc = sum(int(a != b) for a, b in zip(mine["encode"], ref["encode"], strict=True))
    col = [max(_ulps(a, b) for a, b in zip(qm, qr, strict=True))
           for qm, qr in zip(mine["columns"], ref["columns"], strict=True)]  # fmt: skip
    n_lever = 0
    for lm, lr in zip(mine["lever"], ref["lever"], strict=True):
        if lm["ok"] != lr["ok"] or lm.get("failure") != lr.get("failure"):
            n_lever += 1
            continue
        u = max(_ulps(a, b) for a, b in zip(_lever_values(lm), _lever_values(lr), strict=True))
        ulps.append(u)
        n_lever += int(u > 0)
    return {"rows": n_rows, "text_diffs": n_text, "number_diffs": n_num, "encode": len(ref["encode"]),
            "encode_diffs": n_enc, "columns": len(col), "column_diffs": sum(int(u > 0) for u in col),
            "lever_cases": len(ref["lever"]), "lever_diffs": n_lever, "max_ulps": int(max(ulps + col))}
