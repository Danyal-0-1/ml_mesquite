"""Unit tests of the M4 error model and Step 0 additions (the §9 Monte Carlo checks live in test_validation)."""

import numpy as np
import pytest

from mesqsim import step0
from mesqsim.config import load_config
from mesqsim.conventions import (
    exp_map,
    geodesic_angle_rad,
    heading_rad,
    quat_conj,
    quat_mul,
    wrap_rad,
    yaw_quat,
)
from mesqsim.paths import REPO_ROOT
from mesqsim.sim.heading import Epoch, PodHeading, theta_increments_rad, twist_z_rad
from mesqsim.sim.motion import MotionSource
from mesqsim.sim.radio import gilbert_elliott_lost
from mesqsim.sim.simulate import fusion_error, simulate_session

CFG = load_config(REPO_ROOT / "configs" / "default.yaml")


def test_twist_about_vertical():
    for alpha in np.deg2rad([-170.0, -30.0, 0.0, 45.0, 179.0]):
        q = quat_mul(yaw_quat(alpha), exp_map([0.4, 0.0, 0.0]))
        assert twist_z_rad(q) == pytest.approx(alpha, abs=1e-12)


def test_theta_counts_one_full_turn_despite_tilt_wobble():
    """Tilting about a horizontal axis adds no vertical rotation: one turn is exactly 2 pi."""
    u = np.linspace(0.0, 1.0, 2401)
    p = quat_mul(yaw_quat(2 * np.pi * u), exp_map(np.c_[np.deg2rad(20.0) * np.sin(4 * np.pi * u), 0 * u, 0 * u]))
    assert np.sum(theta_increments_rad(p)) == pytest.approx(2 * np.pi, abs=1e-4)
    back_and_forth = yaw_quat(np.pi * np.sin(2 * np.pi * u))  # signed: turning back cancels
    assert np.sum(theta_increments_rad(back_and_forth)) == pytest.approx(0.0, abs=1e-9)


def test_gilbert_elliott_edge_cases():
    rng = np.random.default_rng(0)
    assert not gilbert_elliott_lost(10_000, 0.0, 3.0, rng).any()
    lost = gilbert_elliott_lost(200_000, 0.05, 1.0, rng)
    assert not np.any(lost[1:] & lost[:-1])  # burst length 1: never two in a row
    with pytest.raises(ValueError):
        gilbert_elliott_lost(10, 1.5, 3.0, rng)
    with pytest.raises(ValueError):
        gilbert_elliott_lost(10, 0.1, 0.5, rng)


def test_fusion_error_is_stationary_with_the_right_correlation():
    t = np.arange(0, 2000.0, 1 / 55.0)
    eta = fusion_error(t, np.deg2rad(0.5), 1.0, np.random.default_rng(1))
    assert np.rad2deg(eta.std(axis=0)) == pytest.approx([0.5, 0.5, 0.5], rel=0.05)
    lag = 55  # one second = one correlation time
    r = np.mean([np.corrcoef(eta[:-lag, k], eta[lag:, k])[0, 1] for k in range(3)])
    assert r == pytest.approx(np.exp(-1.0), abs=0.05)


def test_psi_is_nan_while_a_pod_is_off_and_restarts_after_reboot():
    t = np.linspace(0, 100, 1001)
    hd = PodHeading(t, np.zeros_like(t), np.zeros_like(t), 0.01, 0.0,
                    [Epoch(0.0, 40.0, 0.3, 1.0), Epoch(42.0, 100.0, 42.3, -2.0)])  # fmt: skip
    psi = hd.psi([10.0, 41.0, 50.0])
    assert psi[0] == pytest.approx(1.0 + 0.01 * 10.0)
    assert np.isnan(psi[1])
    assert psi[2] == pytest.approx(-2.0 + 0.01 * 8.0)  # drift counts from the new boot


def _session(preset, seed=1, **overrides):
    cfg = CFG.with_preset(preset) if preset else CFG
    if overrides:
        cfg = cfg.with_overrides(overrides)
    return simulate_session(cfg, seed, MotionSource.from_spec("synthetic:walk:10"))


def test_tpose_snapshot_gives_world_deltas_of_the_segments():
    """With every error off, Delta_i(t) = q_i(t) q_i(T)^-1 is the true world rotation of the pod since the T-pose."""
    d = _session("zero")
    series = step0.decode(d.frames, d.session)
    snap = step0.tpose_snapshot(series, d.session)
    assert len(snap) == 15
    rec = series[series["received"]].merge(d.labels_ticks, on=["pod_id", "epoch", "tick"])
    rec = rec[rec["phase"] == "motion"]
    delta = step0.world_deltas(rec, snap)
    p_now = rec[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]].to_numpy()
    worn = {i: d.labels_ticks.query("pod_id == @i and phase == 'tpose'")[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]]
            .to_numpy()[0] for i in range(15)}  # fmt: skip
    p_t = np.array([worn[i] for i in rec["pod_id"]])
    err = np.rad2deg(geodesic_angle_rad(quat_mul(quat_conj(quat_mul(p_now, quat_conj(p_t))), delta)))
    assert err.max() < 0.02


def test_box_headings_are_zero_in_aligned_mode_and_minus_offsets_in_arbitrary_mode():
    d = _session("offsets_only", **{"error_model.offset_mode": "aligned"})
    bh = step0.box_headings(step0.decode(d.frames, d.session), d.session)
    assert len(bh) == 15 and (bh["n_frames"] > 50).all()
    assert np.rad2deg(np.max(np.abs(bh["heading_rad"]))) < 0.01  # the chip's north is its own start-up heading
    d = _session("offsets_only")
    bh = step0.box_headings(step0.decode(d.frames, d.session), d.session)
    a = np.array([p.offset_rad for p in d.plan.pods])
    expected = wrap_rad(d.plan.box_heading_rad - a)  # heading(Y(-a)·Y(box)) with zero slot error
    assert np.rad2deg(np.max(np.abs(wrap_rad(bh["heading_rad"].to_numpy() - expected)))) < 0.01


def test_reboot_gives_a_new_north_seen_by_o1_and_step0():
    cfg = CFG.with_preset("stress").with_overrides({"session.idle_s": 600.0})
    d = simulate_session(cfg, 5, MotionSource.from_spec("synthetic:walk:30"))
    rebooted = [p.pod_id for p in d.plan.pods if p.reboots_s]
    assert rebooted, "seed 5 is chosen because it reboots three pods"
    series = step0.decode(d.frames, d.session)
    last_epoch = series.groupby("pod_id")["epoch"].max().to_dict()
    assert last_epoch == d.labels_ticks.groupby("pod_id")["epoch"].max().to_dict()
    epochs = d.meta["epochs"][rebooted[0]]
    assert len(epochs) == 2 and epochs[1]["offset_rad"] != epochs[0]["offset_rad"]


def test_aligned_report_heading_is_zero_at_start_up():
    d = _session("offsets_only", **{"error_model.offset_mode": "aligned"})
    first = d.labels_ticks.groupby("pod_id").head(1)
    q = quat_mul(yaw_quat(-first["psi_rad"].to_numpy()), first[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]].to_numpy())
    assert np.rad2deg(np.max(np.abs(heading_rad(q)))) < 1e-9
