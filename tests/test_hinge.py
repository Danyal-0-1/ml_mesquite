"""Hinge joints (Step 2 brief §5.4, §8 items 2-4): closed form, likelihood, blind spot, axis identification, sign."""

import numpy as np
import pytest

from mesqsim.conventions import (
    circ_mean_rad,
    exp_map,
    quat_conj,
    quat_mul,
    quat_normalize,
    quat_rotate,
    wrap_rad,
    yaw_quat,
)
from mesqsim.estimators.clock import body_rates
from mesqsim.estimators.hinge import (
    axis_in_world,
    hinge_closed_form,
    identify_hinge_axes,
    vote_tpose,
    vote_vertical,
)


def _axis_angle_deg(u, v):
    return float(np.rad2deg(np.arccos(np.clip(abs(np.dot(u, v)) / np.linalg.norm(u) / np.linalg.norm(v), 0, 1))))


def test_worked_example_12_and_5_degrees_gives_minus_7():
    """h = x horizontal, psi_p = 12°, psi_c = 5°: a = (cos 12°, -sin 12°), b = (cos 5°, -sin 5°)."""
    psi_p, psi_c = np.deg2rad(12.0), np.deg2rad(5.0)
    a = quat_rotate(yaw_quat(-psi_p), [1.0, 0.0, 0.0])[None, :2]
    b = quat_rotate(yaw_quat(-psi_c), [1.0, 0.0, 0.0])[None, :2]
    np.testing.assert_allclose(a[0], [np.cos(psi_p), -np.sin(psi_p)], atol=1e-15)
    c = float(np.sum(a * b))
    s = float(b[0, 0] * a[0, 1] - b[0, 1] * a[0, 0])
    assert c == pytest.approx(0.992546, abs=5e-7) and s == pytest.approx(-0.121869, abs=5e-7)
    fit = hinge_closed_form(a, b, sigma_floor_rad=np.deg2rad(0.5))
    assert np.rad2deg(fit.delta_rad) == pytest.approx(-7.000, abs=5e-4)


def test_vertical_axis_is_the_blind_spot():
    """An axis along the vertical has no shadow on the floor: I = 0, kappa = 0, whatever the private norths."""
    n = 50
    a = axis_in_world(yaw_quat(np.linspace(0.0, 3.0, n)), [0.0, 0.0, 1.0])[:, :2]
    b = axis_in_world(yaw_quat(np.linspace(1.0, -2.0, n)), [0.0, 0.0, 1.0])[:, :2]
    fit = hinge_closed_form(a, b, np.deg2rad(0.5))
    assert fit.info < 1e-12 and fit.kappa == 0.0
    assert hinge_closed_form(np.zeros((0, 2)), np.zeros((0, 2)), 0.01).kappa == 0.0


def _hinge_pairs(rng, n, delta, sigma=0.0):
    """Exact hinge geometry: random segment poses, axis h; parent and child reports with private norths."""
    psi_p = rng.uniform(-np.pi, np.pi)
    psi_c = psi_p + delta
    u_seg = quat_normalize(rng.normal(size=3).tolist() + [0.0])[:3]
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    s_p = quat_normalize(rng.normal(size=(n, 4)))
    theta = rng.uniform(0, 2.0, n)
    s_c = quat_mul(s_p, exp_map(theta[:, None] * u_seg))  # rotation about the hinge axis
    q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p)), s_p), m_p)
    q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_c)), s_c), m_c)
    u_p, u_c = quat_rotate(quat_conj(m_p), u_seg), quat_rotate(quat_conj(m_c), u_seg)
    a = axis_in_world(q_p, u_p)[:, :2] + rng.normal(0, sigma, (n, 2))
    b = axis_in_world(q_c, u_c)[:, :2] + rng.normal(0, sigma, (n, 2))
    return a, b


def test_exact_recovery_over_random_deltas_axes_and_poses():
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        delta = rng.uniform(-np.pi, np.pi)
        a, b = _hinge_pairs(rng, int(rng.integers(1, 40)), delta)
        worst = max(worst, abs(float(wrap_rad(hinge_closed_form(a, b, 1e-3).delta_rad - delta))))
    assert worst < 1e-9


def test_likelihood_spread_matches_one_over_sqrt_kappa():
    """2,000 noisy windows: the circular spread of delta_hat equals 1/sqrt(kappa) within 10 % (brief §8 item 3)."""
    rng = np.random.default_rng(2)
    a0, b0 = _hinge_pairs(rng, 40, 0.0)  # one window's geometry, fixed
    est, kap = [], []
    for _ in range(2000):
        fit = hinge_closed_form(a0 + rng.normal(0, 0.05, a0.shape), b0 + rng.normal(0, 0.05, b0.shape), 1e-6)
        est.append(fit.delta_rad)
        kap.append(fit.kappa)
    spread = float(np.sqrt(np.mean(wrap_rad(np.array(est) - circ_mean_rad(est)) ** 2)))
    predicted = float(np.sqrt(np.mean(1.0 / np.array(kap))))
    assert spread == pytest.approx(predicted, rel=0.10)


def _hinge_motion(rng, seconds=8.0, rate_hz=50.0):
    """Noise-free hinge motion: a thigh-like parent tumbling slowly, the child flexing about the hinge."""
    t = np.arange(0.0, seconds, 1.0 / rate_hz)
    u_seg = np.array([1.0, 0.0, 0.0])
    s_p = exp_map(np.c_[0.6 * np.sin(1.3 * t), 0.4 * np.sin(0.7 * t + 1.0), 0.5 * np.sin(0.9 * t + 2.0)])
    s_c = quat_mul(s_p, exp_map((0.8 + 0.7 * np.sin(2.1 * t))[:, None] * u_seg))
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    u_p, u_c = quat_rotate(quat_conj(m_p), u_seg), quat_rotate(quat_conj(m_c), u_seg)
    return t, quat_mul(s_p, m_p), quat_mul(s_c, m_c), u_p, u_c


def test_axes_identified_within_one_degree_on_noise_free_hinge_motion():
    rng = np.random.default_rng(3)
    for _ in range(5):
        t, q_p, q_c, u_p, u_c = _hinge_motion(rng)
        psi = rng.uniform(-np.pi, np.pi, 2)  # private norths do not matter to rates
        _, w_p = body_rates(t, quat_mul(yaw_quat(np.full(t.size, -psi[0])), q_p), 0.1)
        _, w_c = body_rates(t, quat_mul(yaw_quat(np.full(t.size, -psi[1])), q_c), 0.1)
        fit = identify_hinge_axes(w_p, w_c, n_starts=8)
        assert _axis_angle_deg(fit.u_p, u_p) < 1.0 and _axis_angle_deg(fit.u_c, u_c) < 1.0
        assert fit.rel_residual < 0.02


def test_flipped_axis_is_corrected_by_the_vertical_rule():
    rng = np.random.default_rng(4)
    t, q_p, q_c, u_p, u_c = _hinge_motion(rng)  # the parent tumbles, so the axis leaves the horizontal
    a3, b3 = axis_in_world(q_p, u_p), axis_in_world(q_c, u_c)
    min_prod = np.sin(np.deg2rad(5.0)) ** 2
    assert vote_vertical(a3, b3, min_prod) == 1
    assert vote_vertical(a3, axis_in_world(q_c, -u_c), min_prod) == -1  # deliberately flipped: vote to flip back
    flat = a3.copy()
    flat[:, 2] = 0.0
    assert vote_vertical(flat, flat, min_prod) == 0  # an axis that stays horizontal cannot vote


def test_flipped_axis_is_corrected_by_the_tpose_rule():
    """A horizontal knee axis: the vertical rule cannot vote, the T-pose rule can (brief §5.4 Should)."""
    rng = np.random.default_rng(5)
    n = 64
    psi_p, psi_c = 0.4, -1.1
    delta = psi_c - psi_p
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    facing = yaw_quat(0.7)
    s_t = quat_mul(facing, exp_map([np.pi / 2, 0, 0]))  # upright T-pose segment frame: knee axis x horizontal
    q_pt = quat_mul(quat_mul(yaw_quat(-psi_p), s_t), m_p)
    q_ct = quat_mul(quat_mul(yaw_quat(-psi_c), s_t), m_c)
    j_t = quat_mul(quat_mul(quat_conj(q_pt), yaw_quat(delta)), q_ct)
    flex = np.deg2rad(rng.uniform(5, 70, n))
    s_p = np.broadcast_to(s_t, (n, 4))
    s_c = quat_mul(s_p, exp_map(flex[:, None] * np.array([1.0, 0.0, 0.0])))
    q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p)), s_p), m_p)
    q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_c)), s_c), m_c)
    margin = np.deg2rad(60.0)
    assert vote_tpose(q_p, q_c, delta, j_t, margin) == 1
    assert vote_tpose(q_p, q_c, delta + np.pi, j_t, margin) == -1  # the flipped-axis answer is 180° off


def test_tpose_tied_identification_recovers_the_axis_and_refuses_rigid_motion():
    """In the T-pose the knee axis is horizontal and one line in both segments: one unknown (brief §5.4, GEO-ID)."""
    from mesqsim.conventions import quat_to_matrix
    from mesqsim.estimators.hinge import identify_hinge_axes_tpose

    rng = np.random.default_rng(12)
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    u_seg = np.array([1.0, 0.0, 0.0])
    s_t = quat_mul(yaw_quat(0.4), exp_map([np.pi / 2, 0.0, 0.0]))  # upright T-pose: segment x is horizontal
    psi_p, psi_c = 0.3, -1.2
    q_pt = quat_mul(quat_mul(yaw_quat(-psi_p), s_t), m_p)
    q_ct = quat_mul(quat_mul(yaw_quat(-psi_c), s_t), m_c)
    g_p = quat_rotate(quat_conj(q_pt), [0.0, 0.0, 1.0])
    m_rel = quat_to_matrix(quat_mul(quat_mul(quat_conj(q_ct), yaw_quat(-(psi_c - psi_p))), q_pt))
    t = np.arange(0.0, 8.0, 0.02)
    s_p = exp_map(np.c_[0.6 * np.sin(1.3 * t), 0.4 * np.sin(0.7 * t + 1.0), 0.5 * np.sin(0.9 * t + 2.0)])
    s_c = quat_mul(s_p, exp_map((0.8 + 0.7 * np.sin(2.1 * t))[:, None] * u_seg))
    _, w_p = body_rates(t, quat_mul(s_p, m_p), 0.1)
    _, w_c = body_rates(t, quat_mul(s_c, m_c), 0.1)
    fit = identify_hinge_axes_tpose(w_p, w_c, g_p, m_rel)
    assert _axis_angle_deg(fit.u_p, quat_rotate(quat_conj(m_p), u_seg)) < 0.5
    assert _axis_angle_deg(fit.u_c, quat_rotate(quat_conj(m_c), u_seg)) < 0.5
    assert fit.contrast < 0.01
    # thigh and shin turning together (straight knee): every axis fits, nothing is identified
    s_c_rigid = s_p
    _, w_c_rigid = body_rates(t, quat_mul(s_c_rigid, m_c), 0.1)
    m_rel_rigid = quat_to_matrix(quat_mul(quat_conj(m_c), m_p))
    rigid = identify_hinge_axes_tpose(w_p, w_c_rigid, g_p, m_rel_rigid)
    assert rigid.contrast > 0.25
