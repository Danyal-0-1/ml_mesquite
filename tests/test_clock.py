"""The common clock (Step 2 brief §5.3): lower-envelope map, pairs and rotation rates."""

import numpy as np
import pytest

from mesqsim.conventions import exp_map, geodesic_angle_rad, quat_conj, quat_mul, yaw_quat
from mesqsim.estimators.clock import body_rates, fit_host_map, interp_rates, pair_samples


def _pod_clock(rng, n, ppm, epoch_s, send_wait_ms=18.0):
    """Decode times on a pod clock and their host arrival times: send wait + 2 ms + Exp(3 ms) above decode."""
    t_true = np.sort(rng.uniform(0.0, 60.0, n))
    t_pod = t_true * (1.0 + ppm * 1e-6)
    t_host = epoch_s + t_true + rng.uniform(0, send_wait_ms, n) / 1e3 + 2e-3 + rng.exponential(3e-3, n)
    return t_true, t_pod, t_host


def test_lower_envelope_maps_decode_time_to_host_time_within_two_ms():
    rng = np.random.default_rng(0)
    t_true, t_pod, t_host = _pod_clock(rng, 1900, 45.0, 400.0)
    alpha, beta = fit_host_map(t_pod, t_host, block_s=2.0)
    mapped = alpha + beta * t_pod
    # the map follows the lower envelope: decode time + the minimum latency (2 ms) + a small positive bias, nearly
    # constant over the minute (a common offset cancels in a pair; its drift is what matters)
    above = mapped - (400.0 + t_true)
    assert 2e-3 <= above.min() and above.max() < 5e-3
    assert np.ptp(above) < 1.5e-3
    assert beta == pytest.approx(1.0 / (1.0 + 45e-6), abs=30e-6)  # the slope is estimated, not assumed 1


def test_a_latency_common_to_two_pods_cancels_in_the_pair():
    rng = np.random.default_rng(1)
    tt1, tp1, th1 = _pod_clock(rng, 1900, -40.0, 400.0)
    tt2, tp2, th2 = _pod_clock(rng, 1900, 35.0, 400.0)
    a1, b1 = fit_host_map(tp1, th1, 2.0)
    a2, b2 = fit_host_map(tp2, th2, 2.0)
    m1, m2 = a1 + b1 * tp1, a2 + b2 * tp2
    # two samples at the same true time map to the same host time, to well under the 0-11 ms decode spread
    probe = np.linspace(5, 55, 50)
    d = np.interp(probe, tt1, m1) - np.interp(probe, tt2, m2)
    assert np.max(np.abs(d)) < 1.5e-3


def test_short_data_gives_an_offset_only_map():
    alpha, beta = fit_host_map(np.array([1.0, 1.5]), np.array([101.004, 101.506]), block_s=2.0)
    assert beta == 1.0 and alpha == pytest.approx(100.004)


def test_pairs_slerp_the_child_to_the_parent_instants():
    rate = np.array([0.3, -1.2, 0.8])  # rad/s, constant
    t_c = np.arange(0.0, 2.0, 1 / 55.0)
    q_c = exp_map(t_c[:, None] * rate)
    t_p = np.arange(0.013, 1.9, 1 / 50.0)
    q_p = yaw_quat(0.4 * t_p)
    pairs = pair_samples(t_p, q_p, t_c, q_c, max_gap_s=0.1)
    assert len(pairs) == t_p.size
    truth = exp_map(pairs.t_s[:, None] * rate)
    assert np.max(geodesic_angle_rad(quat_mul(quat_conj(truth), pairs.q_c))) < 1e-12


def test_pairs_need_two_bracketing_samples_within_the_gap():
    t_c = np.array([0.0, 0.05, 0.3, 0.35])
    q_c = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    t_p = np.array([-0.01, 0.02, 0.2, 0.34, 0.4])
    pairs = pair_samples(t_p, np.tile([0.0, 0.0, 0.0, 1.0], (5, 1)), t_c, q_c, max_gap_s=0.1)
    assert pairs.ip.tolist() == [1, 3]  # before the first, inside the 0.25 s gap, and after the last: no pair


def test_rates_ignore_a_constant_heading_offset():
    """A constant psi on the left of every report cancels in q(t-)^-1·q(t) (brief §5.3)."""
    rng = np.random.default_rng(2)
    t = np.cumsum(rng.uniform(0.015, 0.035, 300))
    q = exp_map(np.c_[np.sin(t), 0.5 * np.cos(2 * t), 0.2 * t])
    for psi in (0.0, 1.0, -2.7):
        q_off = quat_mul(yaw_quat(-psi), q)
        tm, w = body_rates(t, q_off, max_gap_s=0.1)
        tm0, w0 = body_rates(t, q, max_gap_s=0.1)
        np.testing.assert_allclose(w, w0, atol=1e-12)
        np.testing.assert_allclose(tm, tm0)


def test_rates_of_a_constant_rotation_are_exact():
    rate = np.array([0.5, -0.2, 1.1])
    t = np.arange(0.0, 1.0, 0.031)
    _, w = body_rates(t, exp_map(t[:, None] * rate), max_gap_s=0.1)
    np.testing.assert_allclose(w, np.broadcast_to(rate, w.shape), atol=1e-12)


def test_interp_rates_only_inside_short_gaps():
    t = np.array([0.0, 0.03, 0.06, 0.5])
    w = np.c_[t, 2 * t, 3 * t]
    ok, out = interp_rates(t, w, np.array([0.015, 0.05, 0.3, 0.6]), max_gap_s=0.1)
    assert ok.tolist() == [True, True, False, False]
    np.testing.assert_allclose(out[:, 0], [0.015, 0.05])
