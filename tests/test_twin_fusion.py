"""The twin's labels, sensor errors and fusion in the loop (Step 3 brief §4.4-4.5; checks 42.1 and 43.1)."""

from unittest import mock

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mesqsim.conventions import exp_map, quat_conj, quat_mul, quat_rotate, wrap_rad, yaw_quat
from mesqsim.twin.config import load_twin_config
from mesqsim.twin.fusion import align_to_gravity, analytic_heading, analytic_reports, fuse, fusion_times, mahony, vqf6d
from mesqsim.twin.labels import delta_rad, epsilon_rad, error_rotation, heading_error_rad, report_labels, tilt_error_rad
from mesqsim.twin.sensors import G0_M_S2, draw_sensor_errors, ideal_gyro_rad_s, measure, no_sensor_errors

RATE = 55.0
DT = 1.0 / RATE
G_UP = np.array([0.0, 0.0, 9.81])


def still_acc(r):
    return quat_rotate(quat_conj(r), G_UP)


def flat_turns(seconds=60.0, turns=3, still_s=5.0):
    """A pod lying face up (its z vertical) that turns ``turns`` times about the vertical, then stops."""
    t = np.arange(0.0, seconds, DT)
    span = seconds - 2 * still_s
    th = np.clip((t - still_s) / span, 0.0, 1.0) * turns * 2 * np.pi
    r = yaw_quat(th)
    return t, r, ideal_gyro_rad_s(np.vstack([r[:1], r]), DT), still_acc(r)


def tumbling(seconds=40.0, seed=0):
    """A pod turning about every axis, with no linear acceleration (it rotates about its own centre)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, seconds, DT)
    w = np.stack([a * np.sin(2 * np.pi * f * t + p) for a, f, p in rng.uniform([0.5, 0.1, 0], [2.0, 0.6, 6], (3, 3))])
    r = Rotation.from_euler("zyx", [30, 10, -20], degrees=True)
    q = [r.as_quat()]
    for k in range(1, t.size):
        r = r * Rotation.from_rotvec(w[:, k] * DT)
        q.append(r.as_quat())
    q = np.array(q)
    return t, q, ideal_gyro_rad_s(np.vstack([q[:1], q]), DT), still_acc(q)


# ----------------------------------------------------------------------------------------------- labels (43.1)
def test_twist_rule_hand_check():
    e = yaw_quat(np.deg2rad(-12.0))
    assert np.isclose(e[2], -np.sin(np.deg2rad(6.0))) and np.isclose(e[3], np.cos(np.deg2rad(6.0)))
    assert np.isclose(np.rad2deg(heading_error_rad(e)), 12.0, atol=1e-12)
    assert tilt_error_rad(e) == 0.0


def test_twist_rule_recovers_psi_of_a_pure_heading_error():
    rng = np.random.default_rng(1)
    r = Rotation.random(500, random_state=rng).as_quat()
    psi = rng.uniform(-np.pi, np.pi, 500)
    got, tilt = report_labels(quat_mul(yaw_quat(-psi), r), r)
    assert np.abs(wrap_rad(got - psi)).max() < 1e-12 and tilt.max() < 1e-7


def test_swing_twist_split_is_order_free():
    rng = np.random.default_rng(2)
    swing = exp_map(np.c_[rng.normal(0, 0.3, (200, 2)), np.zeros(200)])  # about horizontal axes
    twist = yaw_quat(rng.uniform(-3, 3, 200))
    for e in (quat_mul(swing, twist), quat_mul(twist, swing)):
        assert np.allclose(wrap_rad(heading_error_rad(e) + 2 * np.arctan2(twist[:, 2], twist[:, 3])), 0, atol=1e-12)
        assert np.allclose(tilt_error_rad(e), 2 * np.arccos(np.clip(swing[:, 3], -1, 1)), atol=1e-7)


def test_error_rotation_is_canonical_and_delta_epsilon_wrap():
    e = error_rotation(np.array([0.0, 0.0, 0.0, -1.0]), np.array([0.0, 0.0, 0.0, 1.0]))
    assert e[3] == 1.0
    assert np.isclose(delta_rad(np.deg2rad(170.0), np.deg2rad(-170.0)), np.deg2rad(-20.0))
    assert np.isclose(epsilon_rad(np.deg2rad(-175.0), np.deg2rad(10.0)), np.deg2rad(175.0))


# ----------------------------------------------------------------------------------------------- sensors
def test_no_errors_is_the_truth_and_noise_has_its_density():
    cfg = load_twin_config()
    rng = np.random.default_rng(3)
    w, f = rng.normal(0, 1, (1000, 3)), rng.normal(0, 5, (1000, 3))
    gyr, acc, _ = measure(no_sensor_errors(), w, f, DT, rng)
    assert np.array_equal(gyr, w) and np.array_equal(acc, f)
    err = draw_sensor_errors(cfg, rng)
    err.gyro_scale[:], err.accel_scale[:] = 0.0, 0.0
    err.gyro_misalign[:], err.accel_misalign[:] = [0, 0, 0, 1], [0, 0, 0, 1]
    err.gyro_bias0_rad_s[:], err.accel_bias_m_s2[:] = 0.0, 0.0
    gyr, _, _ = measure(err, np.zeros((200_000, 3)), np.tile(G_UP, (200_000, 1)), DT, rng)
    sigma = np.deg2rad(cfg["twin.sensors.gyro_noise_deg_s_sqrt_hz"]) * np.sqrt(RATE)  # N_g / sqrt(dt)
    assert abs(gyr.std() / sigma - 1) < 0.02  # 16-bit rounding (0.061 °/s) is coarse against 0.11 °/s of noise


def test_scale_misalignment_bias_and_clipping():
    cfg = load_twin_config()
    err = no_sensor_errors(ranges=True, cfg=cfg)
    err.gyro_scale[:] = [0.01, -0.02, 0.0]
    err.gyro_bias0_rad_s[:] = [0.0, 0.0, 0.001]
    gyr, acc, _ = measure(err, np.array([[1.0, 1.0, 1.0], [100.0, 0.0, 0.0]]), np.array([[0, 0, 9.81], [0, 0, 80.0]]),
                          DT)  # fmt: skip
    step = np.deg2rad(2000.0) / 2**15
    assert np.allclose(gyr[0], [1.01, 0.98, 1.001], atol=step / 2 + 1e-12)
    assert gyr[1, 0] == np.round(np.deg2rad(2000.0) / step - 1) * step or np.isclose(gyr[1, 0], 32767 * step)
    assert np.isclose(acc[1, 2], 32767 * 4 * G0_M_S2 / 2**15)  # 80 m/s² clips at +4 g


def test_ideal_gyro_integrates_back_exactly():
    _, q, gyr, _ = tumbling(10.0)
    r = q[0]
    for k in range(1, q.shape[0]):
        r = quat_mul(r, exp_map(gyr[k] * DT))
    assert np.abs(np.abs(np.sum(r * q[-1])) - 1) < 1e-12


# ----------------------------------------------------------------------------------------------- fusion (42.1)
@pytest.fixture(params=["vqf6d", "mahony"])
def filt(request):
    if request.param == "vqf6d":
        return lambda g, a, est=False: vqf6d(g, a, DT, 3.0, est, est)
    return lambda g, a, est=False: mahony(g, a, DT, 1 / 3, 1 / 30 if est else 0.0)


def test_start_is_aligned_to_gravity_with_no_twist(filt):
    r = Rotation.from_euler("xyz", [20, -10, 70], degrees=True).as_quat()
    out = filt(np.zeros((50, 3)), np.tile(still_acc(r), (50, 1)))
    assert np.allclose(quat_rotate(quat_conj(out.q[0]), [0, 0, 1]), quat_rotate(quat_conj(r), [0, 0, 1]), atol=1e-9)
    assert abs(out.q[0][2]) < 1e-12  # the shortest arc from gravity: no twist about +Z
    assert np.allclose(quat_rotate(align_to_gravity([0, 0, -9.81]), [0, 0, -1]), [0, 0, 1])


def test_no_error_keeps_psi_constant(filt):
    _, q, gyr, acc = tumbling()
    psi, tilt = report_labels(filt(gyr, acc).q, q)
    assert np.ptp(np.unwrap(psi)) < 1e-9 and tilt.max() < 1e-9


def test_vertical_bias_turns_the_report(filt):
    t, r, gyr, acc = flat_turns()
    b = np.deg2rad(0.5)
    psi, _ = report_labels(filt(gyr + [0.0, 0.0, b], acc).q, r)
    d = np.unwrap(psi) - psi[0]
    expected = -b * (t - t[0])
    assert abs(d[-1] / expected[-1] - 1) < 0.01 and np.abs(d - expected).max() < 0.01 * abs(expected[-1])


def test_vertical_scale_error_adds_k_turns(filt):
    _, r, gyr, acc = flat_turns(turns=3)
    k = 0.01
    psi, _ = report_labels(filt(gyr * [1.0, 1.0, 1.0 + k], acc).q, r)
    assert abs((np.unwrap(psi)[-1] - psi[0]) / (-k * 3 * 2 * np.pi) - 1) < 0.01


def test_vqf_rest_bias_estimation_removes_a_still_pods_vertical_bias():
    t = np.arange(0.0, 60.0, DT)
    r = np.tile([0.0, 0.0, 0.0, 1.0], (t.size, 1))
    b = np.deg2rad(np.array([0.2, -0.3, 0.5]))
    out = vqf6d(np.zeros((t.size, 3)) + b, still_acc(r), DT, 3.0, True, True)
    assert np.allclose(out.bias_rad_s[int(10 * RATE)], b, rtol=0.01)  # 99 % within 10 s of rest
    psi, _ = report_labels(out.q, r)
    assert abs(np.rad2deg(psi[-1] - psi[0])) < 1.0  # instead of -30° without the estimate


def test_fuse_reads_the_config_and_switches_estimation_off():
    cfg = load_twin_config()
    t, r, gyr, acc = flat_turns(seconds=30.0, turns=1)
    a = fuse(cfg, gyr + [0, 0, 0.01], acc, DT, bias_estimation=False)
    b = vqf6d(gyr + [0, 0, 0.01], acc, DT, cfg["twin.fusion.vqf_tau_acc_s"], False, False)
    assert np.array_equal(a.q, b.q)
    with pytest.raises(ValueError):
        fuse(cfg, gyr, acc, DT, mode="analytic")
    assert np.allclose(np.diff(fusion_times(1.0, 5, 55.0, 4)), 1 / 220) and fusion_times(1.0, 5, 55.0, 4).size == 17


def test_analytic_mode_with_eta_zero_equals_step1_psi():
    """The twist rule applied to Step 1's own reports (its simulator, unquantised, η = 0) returns Step 1's ψ."""
    from mesqsim.config import load_config
    from mesqsim.paths import REPO_ROOT
    from mesqsim.sim import simulate as sim
    from mesqsim.sim.motion import MotionSource

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides({"error_model.tilt_sigma_deg": 0.0})
    seen = []
    real = sim.quantize_component
    with mock.patch.object(sim, "quantize_component", lambda q: (seen.append(np.array(q)), real(q))[1]):
        d = sim.simulate_session(cfg, 4, MotionSource.from_spec("synthetic:walk:10"))
    worst = 0.0
    for i, q in enumerate(seen):
        lab = d.labels_ticks[d.labels_ticks["pod_id"] == i]
        r = lab[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]].to_numpy()
        psi, _ = report_labels(q, r)
        worst = max(worst, np.abs(wrap_rad(psi - lab["psi_rad"].to_numpy())).max())
    assert len(seen) == 15 and worst < 1e-12


def test_twin_analytic_mode_is_step1s_formula():
    rng = np.random.default_rng(5)
    t = np.arange(0.0, 20.0, 1 / 240)
    r = yaw_quat(0.4 * np.sin(t))
    hd = analytic_heading(t, r, 0.0, 20.0, 0.3, 0.7, np.deg2rad(0.2) / 60, 0.01, np.deg2rad(0.015), rng)
    ts = np.arange(0.3, 19.0, 1 / 55)
    rs = yaw_quat(0.4 * np.sin(ts))
    psi, _ = report_labels(analytic_reports(rs, hd.psi(ts)), rs)
    assert np.abs(wrap_rad(psi - hd.psi(ts))).max() < 1e-12
