"""The phone: the ARKit boundary, the app's port against its Swift core, the 95-byte message, ARKit-like reports and
the phone's labels (Step 3 brief §4.7; check 45.1)."""

import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mesqsim.conventions import quat_mul, quat_rotate, wrap_rad, yaw_quat
from mesqsim.paths import data_dir
from mesqsim.twin import phone_app as app
from mesqsim.twin.app_golden import FIXTURE, compare, load_fixture
from mesqsim.twin.config import load_twin_config
from mesqsim.twin.page_frame import (
    arkit_to_world_rot,
    arkit_to_world_turn,
    arkit_to_world_vec,
    world_to_arkit_rot,
    world_to_arkit_vec,
)
from mesqsim.twin.phone import arkit_track, frame_times, phone_labels, phone_tables, run_app
from mesqsim.twin.sites import MOUNT_PHONE

QUIET = {"twin.phone.heading_walk_deg_sqrt_min": 0.0, "twin.phone.tilt_sigma_deg": 0.0,
         "twin.phone.limited_per_min": 0.0, "twin.phone.reloc_per_min": 0.0, "twin.phone.scale_error_max": 0.0,
         "twin.phone.position_walk_cm_sqrt_min": 0.0}


# ------------------------------------------------------------------------------------------- the ARKit boundary
def test_arkit_frame_conversions():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3))
    assert np.allclose(arkit_to_world_vec([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])  # ARKit's up is the world's up
    assert np.allclose(arkit_to_world_vec([0.0, 0.0, -1.0]), [0.0, 1.0, 0.0])
    assert np.allclose(world_to_arkit_vec(arkit_to_world_vec(v)), v)
    q = Rotation.random(50, random_state=rng).as_quat()
    assert np.allclose(world_to_arkit_rot(arkit_to_world_rot(q)), q)
    # an orientation keeps its body axes: its image of a body vector is that image converted
    assert np.allclose(quat_rotate(arkit_to_world_rot(q), v), arkit_to_world_vec(quat_rotate(q, v)))
    # a turn about ARKit's +Y is the same turn about the world's +Z
    th = rng.uniform(-3, 3, 50)
    y_arkit = np.stack([np.zeros(50), np.sin(th / 2), np.zeros(50), np.cos(th / 2)], axis=1)
    assert np.allclose(arkit_to_world_turn(y_arkit), yaw_quat(th))


# ------------------------------------------------------------------------------------------- the port vs Swift
def test_port_matches_the_swift_core_bit_for_bit():
    r = compare(load_fixture())
    assert r["rows"] == 1260 and r["encode"] == 2110 and r["columns"] == 410 and r["lever_cases"] == 4
    assert r["text_diffs"] == r["number_diffs"] == r["encode_diffs"] == r["column_diffs"] == r["lever_diffs"] == 0, r
    assert r["max_ulps"] == 0, r


@pytest.mark.swift
def test_fixture_is_what_the_swift_core_says_today(tmp_path):
    mocap = data_dir() / "mesquite-mocap"
    if shutil.which("swiftc") is None or not (mocap / "new_device_code" / "phone_app").is_dir():
        pytest.skip("needs swiftc and the mesquite-mocap clone ($MESQ_DATA_DIR/mesquite-mocap)")
    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "app_golden.json"
    subprocess.run([sys.executable, str(root / "tools" / "app_golden" / "make_fixtures.py"), "--mocap", str(mocap),
                    "--out", str(out)], check=True, capture_output=True)  # fmt: skip
    new, old = json.loads(out.read_text()), json.loads(FIXTURE.read_text())
    new["provenance"].pop("swiftc")
    old["provenance"].pop("swiftc")
    assert new == old


# ------------------------------------------------------------------------------------------- the message
def test_no_message_exceeds_95_bytes_over_10000_random_poses():
    rng = np.random.default_rng(1)
    keys = {"bone", "x", "y", "z", "w", "px", "py", "pz"}
    longest, levels = 0, [0, 0, 0, 0]
    for box in (5.0, 50.0, 500.0, None):
        q = Rotation.random(2500, random_state=rng).as_quat() * rng.choice([-1.0, 1.0], (2500, 1))
        p = rng.uniform(-(box or 1), box or 1, (2500, 3))
        for qi, pi in zip(q, p, strict=True):
            m = app.encode(tuple(qi), None if box is None else tuple(pi))
            assert m is not None and m.bytes <= 95
            longest = max(longest, m.bytes)
            levels[m.level] += 1
            obj = json.loads(m.text)
            assert set(obj) == (keys if m.has_position else keys - {"px", "py", "pz"})
            assert re.search(r"\d[eE]", m.text) is None  # no exponent
            q_dec, p_dec = app.LEVELS[m.level]
            u = qi / np.linalg.norm(qi)
            assert max(abs(obj[k] - u[i]) for i, k in enumerate("xyzw")) <= 0.5 * 10.0**-q_dec + 1e-12
            if p_dec is not None:
                assert max(abs(obj[k] - pi[i]) for i, k in enumerate(("px", "py", "pz"))) <= 10.0**-p_dec
    assert longest == 95 and sum(levels) == 10_000 and all(levels)


def test_a_zero_is_nudged_and_never_minus_zero():
    assert app.fixed(0.0004, 3, avoid_zero=True) == "0.001" and app.fixed(-0.0004, 3, avoid_zero=True) == "-0.001"
    assert app.fixed(0.0, 3, avoid_zero=True) == "0.001" and app.fixed(-0.00004, 4) == "0"
    assert app.fixed(2.5, 0) == "3" and app.fixed(-2.5, 0) == "-3"  # Swift's rounded(): ties away from zero
    m = json.loads(app.encode((0.0, 0.0, 0.0, 1.0), (0.0, 1.0, -0.0)).text)
    assert m["px"] == 0.001 and m["pz"] == 0.001 and all(m[k] != 0 for k in ("px", "py", "pz"))


# ------------------------------------------------------------------------------------------- the twin's phone
def belt(t, turn_deg=90.0):
    """A wearer standing for 8 s at heading 40°, then walking at 1 m/s while turning left; the phone 12 cm in front
    of the pelvis, worn as sites.py says. Returns the camera's true orientations and positions (world, +Z up)."""
    heading = np.deg2rad(40.0) + np.deg2rad(turn_deg) * np.clip((t - 8.0) / 6.0, 0.0, 1.0)
    fwd = np.stack([-np.sin(heading), np.cos(heading), 0 * t], 1)
    right = np.stack([np.cos(heading), np.sin(heading), 0 * t], 1)
    frame = Rotation.from_matrix(np.stack([np.tile([0, 0, 1.0], (t.size, 1)), right, fwd], axis=2))
    r_cam = (frame * Rotation.from_quat(MOUNT_PHONE)).as_quat()
    speed = np.where(t < 8.0, 0.0, 1.0)
    p = np.cumsum(fwd * speed[:, None] * np.r_[0.0, np.diff(t)][:, None], axis=0) + [0.0, 0.0, 1.0] + 0.12 * fwd
    return r_cam, p, heading


def test_quiet_arkit_gives_a_constant_phone_heading_set_by_the_zero():
    cfg = load_twin_config(overrides=QUIET)
    rng = np.random.default_rng(2)
    t = frame_times(0.0, 20.0, cfg, rng)
    r_cam, p, heading = belt(t)
    track = arkit_track(cfg, t, r_cam, p, rng)
    run = run_app(cfg, track, rng, zero_requests_s=[4.0])
    assert [z["epoch"] for z in run.zeros] == [1, 2] and run.zeros[1]["requested"] and not run.zeros[0]["requested"]
    lab = phone_labels(run, track, r_cam, cfg["twin.app.tree"])
    e2 = lab[lab.epoch == 2]
    # ψ_ph is the wearer's heading at the Zero (forward = the reported frame's -Z, the world's +Y at ψ = 0)
    assert np.abs(wrap_rad(e2.psi_ph_rad.to_numpy() - heading[run.zeros[1]["frame"]])).max() < 1e-6
    assert e2.tilt_ph_rad.max() < 1e-6
    # after the 90° left turn the message reads Y(+90°) about ARKit's +Y
    last = json.loads(run.messages.line.iloc[-1])
    half = math.pi / 4
    assert np.allclose([last["x"], last["y"], last["z"], last["w"]], [0, math.sin(half), 0, math.cos(half)], atol=1e-4)


def test_phone_heading_is_zero_heading_plus_arkit_drift_and_ignores_arkits_start():
    cfg = load_twin_config(overrides=QUIET | {"twin.phone.heading_walk_deg_sqrt_min": 2.0})
    t = frame_times(0.0, 20.0, cfg, np.random.default_rng(3))
    r_cam, p, _ = belt(t)
    psis = []
    for start_turn in (0.0, 1.3):  # ARKit started with the camera looking elsewhere: its own north moves
        r0 = r_cam.copy()
        r0[: int(0.5 * 60)] = quat_mul(yaw_quat(np.full(int(0.5 * 60), start_turn)), r_cam[: int(0.5 * 60)])
        rng = np.random.default_rng(4)
        track = arkit_track(cfg, t, r0, p, rng)
        run = run_app(cfg, track, rng, zero_requests_s=[4.0])
        lab = phone_labels(run, track, r0, cfg["twin.app.tree"])
        e2 = lab[lab.epoch == 2]
        z = run.zeros[1]
        pred = wrap_rad(z["heading_rad"] + track.psi_a_rad[run.messages.frame[run.messages.epoch == 2]])
        assert np.abs(wrap_rad(e2.psi_ph_rad.to_numpy() - pred)).max() < 1e-6
        psis.append(e2.psi_ph_rad.to_numpy())
    assert np.abs(wrap_rad(psis[0] - psis[1])).max() < 1e-6  # the app's Zero cancels ARKit's arbitrary north


@pytest.mark.parametrize("tree, facing", [("thorny", "forward"), ("matched", "forward"), ("velvetFamily", "left")])
def test_labels_hold_for_every_tree_setting(tree, facing):
    cfg = load_twin_config(overrides=QUIET)
    rng = np.random.default_rng(5)
    t = frame_times(0.0, 16.0, cfg, rng)
    r_cam, p, heading = belt(t)
    if facing == "left":  # the phone on the left hip, camera looking left: a quarter turn about the vertical
        r_cam = quat_mul(yaw_quat(np.full(t.size, np.pi / 2)), r_cam)
    track = arkit_track(cfg, t, r_cam, p, rng)
    run = run_app(cfg, track, rng, zero_requests_s=[4.0], settings={"tree": tree, "facing": facing})
    lab = phone_labels(run, track, r_cam, tree)
    e2 = lab[lab.epoch == 2]
    shift = -np.pi if tree == "thorny" else 0.0
    want = wrap_rad(heading[run.zeros[1]["frame"]] + shift)
    assert np.abs(wrap_rad(e2.psi_ph_rad.to_numpy() - want)).max() < 1e-6 and e2.tilt_ph_rad.max() < 1e-6


def test_episodes_link_and_tables():
    cfg = load_twin_config(overrides={"twin.phone.limited_per_min": 6.0, "twin.phone.reloc_per_min": 3.0})
    rng = np.random.default_rng(6)
    t = frame_times(0.0, 60.0, cfg, rng)
    r_cam, p, _ = belt(t)
    track = arkit_track(cfg, t, r_cam, p, rng)
    kinds = {e[0] for e in track.episodes}
    assert kinds == {"limited:excessiveMotion", "limited:relocalizing"}
    run = run_app(cfg, track, rng, zero_requests_s=[4.0])
    phone, lab = phone_tables(run, phone_labels(run, track, r_cam, "velvetFamily"), 1.79e9)
    assert list(phone.columns) == ["t_host_s", "line", "qx", "qy", "qz", "qw", "px", "py", "pz", "len_bytes"]
    assert len(phone) == len(lab) and phone.len_bytes.max() <= 95 and np.all(np.diff(phone.t_host_s) >= 0)
    only_q = phone.px.isna()
    assert only_q.any() and (~only_q).any() and phone.loc[only_q, ["py", "pz"]].isna().all().all()
    assert set(run.messages.tracking[only_q.to_numpy()]) == {"limited:excessiveMotion"}
    assert not (run.messages.tracking == "limited:relocalizing").any()
    sent = run.messages.t_send_s.to_numpy()
    assert np.median(np.diff(sent)) == pytest.approx(1 / 30, abs=0.004)  # 30 Hz, every second ARKit frame
    for line, row in zip(phone.line, phone.itertuples(), strict=True):
        assert re.fullmatch(r'\{"bone":"Hips","x":[-0-9.]+,"y":[-0-9.]+,"z":[-0-9.]+,"w":[-0-9.]+'
                            r'(,"px":[-0-9.]+,"py":[-0-9.]+,"pz":[-0-9.]+)?\}', line)
        assert json.loads(line)["x"] == row.qx
