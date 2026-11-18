"""The statistics Φ and the clock alignment of mesqsim.real, on synthetic motion with known answers."""

import datetime as dt

import numpy as np
import pytest

from mesqsim.real import stats as st
from mesqsim.real.align import LEG_SITES, align, fine_lag_s, height_change
from mesqsim.real.bvh import Bvh, read_bvh
from mesqsim.real.motive import MotiveTake

DT = 1 / 30


def _walker(seconds=20.0, axis_deg=0.0, swing_m=0.2, facing_deg=-90.0, walk_until=None, travel_deg=0.0,
            speed=0.4):
    """A series whose shins swing along the body axis atan2(right, forward) = axis_deg while the root travels at
    travel_deg anticlockwise from the facing."""
    t = np.arange(int(seconds / DT)) * DT
    fwd = np.array([np.cos(np.deg2rad(facing_deg)), np.sin(np.deg2rad(facing_deg))])
    right = np.array([fwd[1], -fwd[0]])
    a = np.deg2rad(axis_deg)
    axis = np.cos(a) * fwd + np.sin(a) * right
    move = np.ones_like(t) if walk_until is None else (t < walk_until).astype(float)
    tr = np.deg2rad(facing_deg + travel_deg)
    root = np.stack([speed * t * np.cos(tr), speed * t * np.sin(tr), np.full_like(t, 0.8)], axis=1)
    sites = {"Hips": root.copy()}
    for s, sign in (("LeftLeg", 1.0), ("RightLeg", -1.0)):
        d = sign * swing_m * np.sin(2 * np.pi * t) * move
        sites[s] = root + np.concatenate([d[:, None] * axis[None, :], np.full((len(t), 1), -0.45)], axis=1)
    facing = np.broadcast_to(fwd, (len(t), 2)).copy()
    return st.Series(t, DT, sites, facing, root, "bvh")


@pytest.mark.parametrize("axis_deg, expected", [(0.0, 0.0), (30.0, 30.0), (-60.0, -60.0), (120.0, -60.0)])
def test_swing_axis_is_an_axis(axis_deg, expected):
    s = _walker(axis_deg=axis_deg)
    assert np.isclose(st.swing_axis_deg(s, "LeftLeg", st.walking_mask(s)), expected, atol=0.01)
    assert st.swing_roundness(s, "LeftLeg", st.walking_mask(s)) < 1e-6  # a line


def test_walking_needs_both_shins_swinging():
    s = _walker(walk_until=10.0)
    w = st.walking_mask(s)
    assert 9.0 < w.sum() * DT < 10.6 and not w[s.t_s > 11.0].any()


def test_umeyama_recovers_a_similarity():
    rng = np.random.default_rng(4)
    src = rng.normal(size=(200, 2))
    th = np.deg2rad(-20.0)
    r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    dst = 0.45 * src @ r.T + [1.5, -0.3]
    s, turn, rms, t = st.umeyama_2d(src, dst)
    assert np.isclose(s, 0.45) and np.isclose(turn, -20.0) and rms < 1e-12 and np.allclose(t, [1.5, -0.3])


@pytest.mark.parametrize("travel", [30.0, -74.0])
def test_travel_minus_facing_is_anticlockwise_positive(travel):
    s = _walker(travel_deg=travel)
    assert np.isclose(st.travel_minus_facing(s, st.walking_mask(s)), travel, atol=1e-6)
    speed, slow = st.root_speed(s, st.walking_mask(s))
    assert np.isclose(speed, 0.4) and slow == 0.0


def _bvh(tmp_path, columns: dict[str, np.ndarray], frame_time=DT) -> Bvh:
    """A BVH file with a root Hips (6 channels) and children, one rotation-channel triple (X, Y, Z) each."""
    names = list(columns)
    lines = ["HIERARCHY", f"ROOT {names[0]}", "{", "  OFFSET 0 100 0",
             "  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"]
    for nm in names[1:]:
        lines += [f"  JOINT {nm}", "  {", "    OFFSET 0 10 0", "    CHANNELS 3 Xrotation Yrotation Zrotation",
                  "  }"]
    lines += ["}", "MOTION", f"Frames: {len(columns[names[0]])}", f"Frame Time: {frame_time}"]
    n = len(columns[names[0]])
    data = np.concatenate([np.tile([0.0, 100.0, 0.0], (n, 1))] + [columns[nm] for nm in names], axis=1)
    lines += [" ".join(f"{v:.9f}" for v in row) for row in data]
    p = tmp_path / "s.bvh"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return read_bvh(p)


def test_exact_repeats_count_identical_still_frames(tmp_path):
    x = np.zeros((100, 3))
    x[50:, 0] = 0.001 * (1 + np.arange(50))  # from frame 50 it moves 0.001° per frame: still, but not exact
    b = _bvh(tmp_path, {"Hips": x})
    assert np.isclose(st.exact_repeats(b, ["Hips"])["Hips"], 49 / 99)


def test_jumps_and_snapping_back(tmp_path):
    x = np.zeros((60, 3))
    x[10:13, 0] = 30.0  # a glitch of three frames
    x[30:, 0] = 30.0  # a turn that stays
    j = st.jumps(_bvh(tmp_path, {"Hips": x}), ["Hips"])["Hips"]
    assert [(round(t / DT), round(a), back) for t, a, back in j] == [(10, 30, True), (13, 30, False), (30, 30, False)]


def test_static_hold_window_drift_and_restart(tmp_path):
    n = 600  # ten minutes at 1 frame per second
    t_min = np.arange(n) / 60.0
    hips = np.zeros((n, 3))
    hips[:, 1] = 0.05 * t_min  # Y rotation in three.js = heading: 0.05° per minute
    head = np.zeros((n, 3))
    head[:, 1] = 0.95 * t_min  # local, so its world heading drifts 1° per minute
    hand = np.zeros((n, 3))
    hand[480:, 1] = 40.0  # a restart at 8 min
    h = st.static_hold(_bvh(tmp_path, {"Hips": hips, "Head": head, "LeftHand": hand}, frame_time=1.0),
                       ["Hips", "Head", "LeftHand"])
    assert np.isclose(h["window_min"], 8.0)
    assert np.isclose(h["pods"]["Hips"]["drift_deg_per_min"], 0.05, atol=1e-6)
    assert np.isclose(h["pods"]["Head"]["drift_deg_per_min"], 1.0, atol=1e-6)
    assert h["steady"] == ["Hips", "Head"] and h["holding"] == ["Hips"] and h["restarted"] == ["LeftHand"]


def test_height_change_removes_drift():
    t = np.arange(3000) / 120.0
    z = 0.01 * t + 0.05 * np.sin(2 * np.pi * t)
    hc = height_change(z, 1 / 120.0)
    core = slice(240, -240)
    assert abs(np.mean(hc[core])) < 1e-3 and np.corrcoef(hc[core], np.sin(2 * np.pi * t[core]))[0, 1] > 0.999


def _leg_signal(t, k):
    rng = np.random.default_rng(10 + k)
    f = rng.uniform(0.3, 2.0, size=6)
    ph = rng.uniform(0, 2 * np.pi, size=6)
    return 0.03 * np.sin(2 * np.pi * f[None, :] * t[:, None] + ph[None, :]).sum(axis=1)


def test_fine_lag_recovers_a_known_lag():
    t_b = np.arange(900) * DT
    t_m = np.arange(int(40 * 120)) / 120.0
    lag = 1.25
    drawn = {s: height_change(_leg_signal(t_b, k), DT) for k, s in enumerate(LEG_SITES)}
    truth = {s: np.stack([np.zeros_like(t_m), np.zeros_like(t_m), 0.5 + _leg_signal(t_m - lag, k)], axis=1)
             for k, s in enumerate(LEG_SITES)}
    a = fine_lag_s(t_b, drawn, t_m, truth, coarse_s=0.0)
    assert abs(a.lag_s - lag) <= 1 / 120 + 1e-9 and a.score > 0.99


def test_align_finds_the_frame_interval_of_an_asserted_clock():
    """A page that writes 1/30 s but records every 33.0 ms, against a take whose clock is 0.6 s off."""
    n, true_dt, lag0, clock_err = 3000, 0.0330, 2.0, 0.6
    end = dt.datetime(2026, 11, 13, 13, 41, 17)
    t_bvh_true = np.arange(n) * true_dt
    t_m = np.arange(int((n * true_dt + 10) * 120)) / 120.0
    take_start = end - dt.timedelta(seconds=n * true_dt + lag0) + dt.timedelta(seconds=clock_err)
    drawn_sites = {s: np.stack([np.zeros(n), np.zeros(n), 0.8 + _leg_signal(t_bvh_true, k)], axis=1)
                   for k, s in enumerate(LEG_SITES)}
    root = np.tile([0.0, 0.0, 0.8], (n, 1))
    truth = {s: np.stack([np.zeros_like(t_m), np.zeros_like(t_m), 0.5 + _leg_signal(t_m - lag0, k)], axis=1)
             for k, s in enumerate(LEG_SITES)}
    bvh = Bvh("MMcap_bvh_2026-11-13-13-41-17.bvh", [], 1 / 30, "asserted", None, None, {}, np.zeros((n, 0)))
    take = MotiveTake("t", "t", take_start, 120.0, np.arange(len(t_m)), t_m, [], np.zeros((len(t_m), 0, 3)), {})
    a = align(bvh, bvh.path, drawn_sites, root, take, truth)
    assert abs(a.frame_time_s - true_dt) <= 0.0000101 and abs(a.lag_s - lag0) <= 0.02 and a.score > 0.99
    assert abs(a.clock_difference_s - clock_err) <= 0.02
