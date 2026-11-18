"""Every worked example of docs/explain/MATH_EXPLAINED.md, recomputed from the code (Step 2 brief §10).

Each test is named after the example's label (E1, E2, ...). If a number in the document and the code disagree, this
file fails: fix whichever is wrong and record it in WORKLOG.md.
"""

import numpy as np
import pandas as pd
import pytest

from mesqsim.codec import quantize_component
from mesqsim.conventions import (
    exp_map,
    geodesic_angle_rad,
    heading_rad,
    quat_conj,
    quat_mul,
    quat_rotate,
    quat_to_matrix,
    vm_halfwidth_rad,
    vm_kappa_from_resultant,
    vm_mean_resultant,
    vm_predict_kappa,
    vm_product,
    wrap_deg,
    wrap_rad,
    yaw_quat,
)
from mesqsim.estimators.clock import body_rates, fit_host_map
from mesqsim.estimators.hinge import axis_in_world, hinge_closed_form
from mesqsim.estimators.twodof import twodof_solve
from mesqsim.results_step2 import bootstrap_contrast, holm
from mesqsim.step0 import SEND_PERIOD_S, decode

D = np.deg2rad


def test_E1_yaw_90_turns_x_to_y():
    q = yaw_quat(D(90.0))
    np.testing.assert_allclose(q, [0, 0, 0.707107, 0.707107], atol=5e-7)
    np.testing.assert_allclose(quat_rotate(q, [1.0, 0.0, 0.0]), [0, 1, 0], atol=1e-15)


def test_E2_composition_order():
    rz, rx = yaw_quat(D(90.0)), exp_map([D(90.0), 0.0, 0.0])
    np.testing.assert_allclose(quat_rotate(quat_mul(rz, rx), [0.0, 1.0, 0.0]), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(quat_rotate(quat_mul(rx, rz), [0.0, 1.0, 0.0]), [-1, 0, 0], atol=1e-12)


def test_E3_q_and_minus_q():
    q = np.array([0.1, -0.5, 0.3, 0.8]) / np.linalg.norm([0.1, -0.5, 0.3, 0.8])
    np.testing.assert_array_equal(quat_to_matrix(q), quat_to_matrix(-q))


def test_E4_wrap():
    assert [float(wrap_deg(x)) for x in (190.0, -180.0, 540.0)] == [-170.0, 180.0, 180.0]


def test_E5_heading_shifts_by_alpha():
    r = quat_mul(yaw_quat(D(30.0)), exp_map([0.2, -0.3, 0.0]))
    assert np.rad2deg(heading_rad(r)) == pytest.approx(28.221, abs=5e-4)
    assert np.rad2deg(heading_rad(quat_mul(yaw_quat(D(50.0)), r))) == pytest.approx(78.221, abs=5e-4)


def test_E6_geodesic_angle():
    assert geodesic_angle_rad(exp_map([0.0, 0.0, 0.3])) == pytest.approx(0.3, abs=1e-15)


def test_E7_turning_the_room_changes_no_relative_heading():
    assert float(wrap_deg(5.0 - 12.0)) == -7.0 == float(wrap_deg((5.0 + 73.0) - (12.0 + 73.0)))


def test_E8_the_briefs_example_and_the_14_degree_sign_error():
    s_m = np.array([0.2, -0.4, 0.1, 0.88]) / np.linalg.norm([0.2, -0.4, 0.1, 0.88])
    q_c = quat_mul(yaw_quat(-D(5.0)), s_m)
    shared = quat_mul(yaw_quat(-D(12.0)), s_m)
    delta = wrap_rad(D(5.0) - D(12.0))
    good = geodesic_angle_rad(quat_mul(quat_mul(yaw_quat(delta), q_c), quat_conj(shared)))
    bad = geodesic_angle_rad(quat_mul(quat_mul(yaw_quat(-delta), q_c), quat_conj(shared)))
    assert np.rad2deg(delta) == pytest.approx(-7.0) and np.rad2deg(good) < 1e-6
    assert np.rad2deg(bad) == pytest.approx(14.0, abs=1e-9)


def _frames(t, count, ms):
    n = len(t)
    qi = quantize_component(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)))
    return pd.DataFrame({"t_host_s": np.asarray(t, float), "pod_id": np.full(n, 3, np.uint8),
                         "batt": np.full(n, 80, np.uint8), "qx": qi[:, 0], "qy": qi[:, 1], "qz": qi[:, 2],
                         "qw": qi[:, 3], "count": np.asarray(count, np.int64).astype(np.uint16),
                         "ms_lo": np.asarray(ms, np.int64).astype(np.uint16)})  # fmt: skip


def test_E9_count_wrap():
    s = decode(_frames(100 + np.arange(5) * SEND_PERIOD_S, [65533, 65534, 65535, 0, 1],
                       [(1000 + k * 31) % 65536 for k in range(5)]))  # fmt: skip
    assert s["tick"].tolist() == [65533, 65534, 65535, 65536, 65537]


def test_E10_lost_and_held():
    s = decode(_frames(100 + np.array([0, 1, 4, 5]) * SEND_PERIOD_S, [10, 11, 14, 15], [1300, 1331, 1425, 1425]))
    assert s["status"].tolist() == ["fresh", "fresh", "lost", "lost", "fresh", "held"]


def test_E12_lower_envelope_on_four_samples():
    tp = np.array([10.0, 10.5, 11.0, 11.5])
    alpha, beta = fit_host_map(tp, 500 + tp + np.array([0.004, 0.012, 0.003, 0.020]), 2.0)
    assert beta == 1.0 and alpha == pytest.approx(500.003, abs=1e-9)


def test_E13_rates_ignore_a_constant_north():
    t = np.arange(0.0, 1.0, 0.031)
    q = exp_map(t[:, None] * np.array([0.5, -0.2, 1.1]))
    for psi in (0.0, 1.3):
        _, w = body_rates(t, quat_mul(yaw_quat(np.full(t.size, -psi)), q), 0.1)
        np.testing.assert_allclose(w, np.broadcast_to([0.5, -0.2, 1.1], w.shape), atol=1e-12)


def test_E14_hinge_worked_example():
    a = quat_rotate(yaw_quat(-D(12.0)), [1.0, 0.0, 0.0])[None, :2]
    b = quat_rotate(yaw_quat(-D(5.0)), [1.0, 0.0, 0.0])[None, :2]
    c, s = float(np.sum(a * b)), float(b[0, 0] * a[0, 1] - b[0, 1] * a[0, 0])
    assert round(c, 6) == 0.992546 and round(s, 6) == -0.121869
    assert np.rad2deg(hinge_closed_form(a, b, 1e-3).delta_rad) == pytest.approx(-7.000, abs=5e-4)


def test_E15_hinge_kappa():
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    b = np.array([[np.cos(D(-10)), np.sin(D(-10))], [np.cos(D(80)), np.sin(D(80))]])
    exact = hinge_closed_form(a, b, 1e-6)
    assert np.rad2deg(exact.delta_rad) == pytest.approx(10.0) and exact.info == pytest.approx(2.0)
    noisy = hinge_closed_form(np.array([[1.0, 0.02], [-0.01, 1.0]]), b, 1e-6)
    assert np.rad2deg(noisy.delta_rad) == pytest.approx(10.86, abs=0.01)
    assert noisy.info == pytest.approx(2.0002, abs=1e-4) and noisy.rss == pytest.approx(5.0e-5, rel=0.01)
    assert noisy.kappa == pytest.approx(1.2e5, rel=0.01)
    assert np.rad2deg(1 / np.sqrt(noisy.kappa)) == pytest.approx(0.17, abs=0.005)


def test_E16_blind_spot():
    a = axis_in_world(yaw_quat(np.linspace(0.0, 3.0, 20)), [0.0, 0.0, 1.0])[:, :2]
    fit = hinge_closed_form(a, a, D(0.5))
    assert fit.info < 1e-12 and fit.kappa == 0.0


@pytest.mark.parametrize("theta_deg", [0.0, 30.0, 90.0])
def test_E17_half_turn_about_the_vertical_composed_with_flexion_is_180_degrees(theta_deg):
    k = quat_mul(yaw_quat(np.pi), exp_map([D(theta_deg), 0.0, 0.0]))
    assert abs(k[3]) < 1e-15 and np.rad2deg(geodesic_angle_rad(k)) == pytest.approx(180.0)


def test_E19_built_twin_is_ambiguous():
    rng = np.random.default_rng(2501)
    a3 = np.c_[rng.normal(size=(50, 2)), np.zeros(50)]
    rot = np.array([[np.cos(-0.7), -np.sin(-0.7)], [np.sin(-0.7), np.cos(-0.7)]])
    b3 = np.c_[(np.c_[-a3[:, 1], a3[:, 0]] * rng.uniform(0.5, 1.5, (50, 1))) @ rot.T, rng.normal(size=50)]
    fit = twodof_solve(a3, b3, 0.0, 1e-3, 3.0)
    assert fit.ambiguous and fit.kappa == 0.0 and abs(fit.second_cost - fit.cost) < 1e-9


def test_E21_uniform_over_plus_minus_80_degrees():
    r = np.sin(D(80.0)) / D(80.0)
    assert r == pytest.approx(0.7053, abs=1e-4)
    assert vm_kappa_from_resultant(r) == pytest.approx(2.05, abs=0.01)


def test_E22_mean_resultant_length():
    assert vm_mean_resultant(1.0) == pytest.approx(0.4464, abs=1e-4)
    assert vm_mean_resultant(1000.0) == pytest.approx(1 - 1 / 2000, abs=1e-6)


def test_E23_interval_half_widths():
    assert np.rad2deg(vm_halfwidth_rad(30.0, 0.95)) == pytest.approx(20.70, abs=0.01)
    assert np.rad2deg(vm_halfwidth_rad(30.0, 0.68)) == pytest.approx(10.46, abs=0.01)


def test_E24_product_of_two_beliefs():
    mu, k = vm_product(D(10.0), 4.0, D(40.0), 2.0)
    assert np.rad2deg(mu) == pytest.approx(19.90, abs=0.01) and k == pytest.approx(5.819, abs=1e-3)


def test_E25_prediction():
    assert vm_predict_kappa(1000.0, 1e-5, 60.0) == pytest.approx(625.3, abs=0.05)
    assert 1 / (1 / 1000 + 1e-5 * 60) == pytest.approx(625.0)


def test_E26_calibration_cap():
    from mesqsim.estimators.geo import calibrated_kappa

    s, sm = 0.011776, D(0.9385)
    assert s / sm**2 == pytest.approx(43.89, abs=0.01)
    assert np.rad2deg(1 / np.sqrt(s / sm**2)) == pytest.approx(8.65, abs=0.005)
    for k_stat, k_w in ((1e4, 31.97), (1e5, 42.31)):
        k = np.zeros(14)
        k[10] = k_stat  # joint 10, the left knee
        assert calibrated_kappa(k, {"hinge": s, "2dof": 1.0, "3dof": 1.0},
                                {"hinge": sm, "2dof": 0.0, "3dof": 0.0})[10] == pytest.approx(k_w, abs=0.01)


def test_E28_paired_bootstrap():
    bs = bootstrap_contrast(np.array([-3.0, -1.0, -2.0, 0.5, -1.5]), n_boot=10_000)
    assert bs["mean"] == pytest.approx(-1.4) and bs["lo"] == pytest.approx(-2.4) and bs["hi"] == pytest.approx(-0.4)
    assert bs["p"] == pytest.approx(0.0092)


def test_E29_holm():
    assert holm({"C1": 0.01, "C2": 0.04, "C3": 0.03}) == pytest.approx({"C1": 0.03, "C2": 0.06, "C3": 0.06})
