"""2-DoF constraint and range of motion (Step 2 brief §5.5, §5.6, §8 items 5-6)."""

import numpy as np
import pytest

from mesqsim.conventions import exp_map, quat_conj, quat_mul, quat_normalize, quat_rotate, wrap_rad, yaw_quat
from mesqsim.estimators.anatomy import CARDAN_SEQ, TWODOF_AXES
from mesqsim.estimators.hinge import axis_in_world
from mesqsim.estimators.rom import fit_limits_deg, joint_angles_rad, rom_angles, rom_solve, rom_terms
from mesqsim.estimators.twodof import twodof_coefficients, twodof_cost, twodof_solve, twodof_terms


def _universal_joint(rng, n, j1, j2, delta, tumble=1.0):
    """Exact 2-DoF motion: s_c = s_p·R(j1, a)·R(j2, b); reports with private norths; axes in pod frames."""
    psi_p = rng.uniform(-np.pi, np.pi)
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    s_p = exp_map(rng.normal(0, tumble, (n, 3)))
    a, b = rng.uniform(-1.2, 1.2, n), rng.uniform(-1.2, 1.2, n)
    s_c = quat_mul(quat_mul(s_p, exp_map(a[:, None] * j1)), exp_map(b[:, None] * j2))
    q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p)), s_p), m_p)
    q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p - delta)), s_c), m_c)
    return q_p, q_c, quat_rotate(quat_conj(m_p), j1), quat_rotate(quat_conj(m_c), j2), m_p, m_c


@pytest.mark.parametrize("joint", ["l_elbow", "l_wrist", "r_ankle"])
def test_twodof_exact_recovery(joint):
    j1, j2, gamma = TWODOF_AXES[joint]
    rng = np.random.default_rng(5)
    worst = 0.0
    for _ in range(40):
        delta = rng.uniform(-np.pi, np.pi)
        q_p, q_c, u1, u2, _, _ = _universal_joint(rng, 60, j1, j2, delta)
        fit = twodof_solve(axis_in_world(q_p, u1), axis_in_world(q_c, u2), np.cos(gamma), 1e-6, 3.0)
        assert not fit.ambiguous and fit.kappa > 0
        worst = max(worst, abs(float(wrap_rad(fit.delta_rad - delta))))
    assert worst < 1e-6


def test_every_equation_holds_at_the_true_delta():
    j1, j2, gamma = TWODOF_AXES["l_elbow"]
    q_p, q_c, u1, u2, _, _ = _universal_joint(np.random.default_rng(6), 30, j1, j2, 0.9)
    p, q, r = twodof_terms(axis_in_world(q_p, u1), axis_in_world(q_c, u2), np.cos(gamma))
    np.testing.assert_allclose(p * np.cos(0.9) + q * np.sin(0.9) + r, 0.0, atol=1e-12)


def test_trig_polynomial_equals_the_sum_of_squares():
    rng = np.random.default_rng(7)
    a3, b3 = rng.normal(size=(25, 3)), rng.normal(size=(25, 3))
    p, q, r = twodof_terms(a3, b3, 0.3)
    d = np.linspace(-3, 3, 13)
    direct = [np.sum((p * np.cos(x) + q * np.sin(x) + r) ** 2) for x in d]
    np.testing.assert_allclose(twodof_cost(twodof_coefficients(p, q, r), d), direct, rtol=1e-12)


def test_two_equal_minima_are_flagged_ambiguous():
    """j1 horizontal and gamma = 90°: the vertical terms vanish, so delta and delta + 180° fit equally well."""
    rng = np.random.default_rng(8)
    n = 50
    a3 = np.c_[rng.normal(size=(n, 2)), np.zeros(n)]  # horizontal first axis
    delta = 0.7
    # b chosen so that a·(Y(delta) b) = 0 with b's own vertical part anything
    b_h = np.c_[-a3[:, 1], a3[:, 0]] * rng.uniform(0.5, 1.5, (n, 1))
    b_h = np.einsum("ij,nj->ni", np.array([[np.cos(-delta), -np.sin(-delta)], [np.sin(-delta), np.cos(-delta)]]),
                    b_h)  # fmt: skip
    b3 = np.c_[b_h, rng.normal(size=n)]
    fit = twodof_solve(a3, b3, 0.0, 1e-3, 3.0)
    assert fit.ambiguous and fit.kappa == 0.0
    assert abs(fit.second_cost - fit.cost) < 1e-9
    assert min(abs(float(wrap_rad(fit.delta_rad - delta))), abs(float(wrap_rad(fit.delta_rad - delta - np.pi)))) < 1e-6


def test_rom_fast_form_equals_direct_joint_rotation():
    rng = np.random.default_rng(9)
    a_q, b_q = quat_normalize(rng.normal(size=(7, 4))), quat_normalize(rng.normal(size=(7, 4)))
    grid = np.array([-2.0, 0.0, 0.4, 3.0])
    ang = rom_angles(*rom_terms(a_q, b_q), "XZY", grid)
    from mesqsim.conventions import quat_to_cardan_rad

    for g, d in enumerate(grid):
        direct = quat_mul(quat_mul(quat_conj(a_q), yaw_quat(np.full(7, d))), b_q)
        np.testing.assert_allclose(ang[g], quat_to_cardan_rad(direct, "XZY"), atol=1e-9)


def _hip_session(rng, n, delta, axial_range_deg):
    """A hip whose axial rotation sweeps its whole allowed range (touching both limits), flexion varying."""
    seq = CARDAN_SEQ["l_hip"]
    psi_p = rng.uniform(-np.pi, np.pi)
    m_p, m_c = quat_normalize(rng.normal(size=(2, 4)))
    s_p = quat_mul(yaw_quat(rng.uniform(-np.pi, np.pi)), exp_map(np.c_[np.pi / 2 + rng.normal(0, 0.05, n),
                                                                      np.zeros(n), np.zeros(n)]))  # fmt: skip
    from scipy.spatial.transform import Rotation

    flex = np.deg2rad(rng.uniform(-10, 60, n))
    abd = np.deg2rad(rng.uniform(-10, 10, n))
    axial = np.deg2rad(np.linspace(-axial_range_deg, axial_range_deg, n))
    rel = Rotation.from_euler(seq, np.c_[flex, abd, axial]).as_quat()
    s_c = quat_mul(s_p, rel)
    q_p = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p)), s_p), m_p)
    q_c = quat_mul(quat_mul(yaw_quat(np.full(n, -psi_p - delta)), s_c), m_c)
    return q_p, q_c, m_p, m_c, seq


def test_rom_recovers_delta_when_the_motion_reaches_the_limits():
    rng = np.random.default_rng(10)
    box = np.deg2rad([[-10, 60], [-10, 10], [-40, 40]])
    for delta in (0.3, -2.5, 1.7):
        q_p, q_c, m_p, m_c, seq = _hip_session(rng, 120, delta, 40.0)
        fit = rom_solve(quat_mul(q_p, quat_conj(m_p)), quat_mul(q_c, quat_conj(m_c)), seq, box[:, 0], box[:, 1],
                        np.deg2rad(2.0))  # fmt: skip
        assert abs(np.rad2deg(float(wrap_rad(fit.delta_rad - delta)))) < 1.0
        assert fit.kappa > 100


def test_rom_has_almost_no_information_inside_the_limits():
    """Motion that never reaches a limit only rules out delta errors so large that the joint would leave its range
    (here about ±80°): kappa < 3, a 95 % interval wider than ±60°. Without any limit, kappa is exactly 0."""
    from mesqsim.conventions import vm_halfwidth_rad
    from mesqsim.estimators.rom import UNBOUNDED_DEG

    rng = np.random.default_rng(11)
    box = np.deg2rad([[-90, 120], [-60, 60], [-90, 90]])  # wide box: a small motion never reaches it
    q_p, q_c, m_p, m_c, seq = _hip_session(rng, 120, 0.8, 10.0)
    a_q, b_q = quat_mul(q_p, quat_conj(m_p)), quat_mul(q_c, quat_conj(m_c))
    fit = rom_solve(a_q, b_q, seq, box[:, 0], box[:, 1], np.deg2rad(5.0))
    assert fit.kappa < 3.0 and np.rad2deg(vm_halfwidth_rad(fit.kappa, 0.95)) > 60.0
    free = np.deg2rad(np.array(UNBOUNDED_DEG))
    assert rom_solve(a_q, b_q, seq, free[:, 0], free[:, 1], np.deg2rad(5.0)).kappa == 0.0


def test_fit_limits_are_percentiles_of_the_true_angles():
    ang = np.deg2rad(np.c_[np.linspace(-10, 30, 1001), np.linspace(-5, 5, 1001), np.zeros(1001)])
    box = fit_limits_deg(ang, (0.5, 99.5))
    assert box[0] == pytest.approx([-9.8, 29.8], abs=1e-6) and box[2] == pytest.approx([0.0, 0.0])


def test_true_joint_angles_follow_the_cardan_sequence():
    from scipy.spatial.transform import Rotation

    seg = np.tile([0.0, 0.0, 0.0, 1.0], (3, 15, 1))
    rel = Rotation.from_euler("XZY", [[0.3, -0.1, 0.2]] * 3).as_quat()
    seg[:, 9] = quat_mul(seg[:, 2], rel)  # left hip: pelvis (2) -> left thigh (9)
    np.testing.assert_allclose(joint_angles_rad(seg, 8), [[0.3, -0.1, 0.2]] * 3, atol=1e-12)
