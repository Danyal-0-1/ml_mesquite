"""Checks 3 and 4 (brief §9): the §4.4 worked example and the correction-level invariances of §4.5."""

import numpy as np
import pytest

from mesqsim.body import JOINTS, N_JOINTS, N_PODS, ROOT_POD
from mesqsim.conventions import (
    geodesic_angle_rad,
    gravity_in_body,
    heading_rad,
    quat_conj,
    quat_mul,
    quat_normalize,
    wrap_rad,
    yaw_quat,
)
from mesqsim.correction import apply_corrections, chain_corrections_rad, relative_heading_rad


def random_rotations(shape, seed):
    return quat_normalize(np.random.default_rng(seed).normal(size=(*shape, 4)))


def reports(psi_rad, s, m):
    """q_i = Y(-psi_i) · s_i · m_i (no fusion error)."""
    return quat_mul(yaw_quat(-psi_rad), quat_mul(s, m))


def test_worked_example_section_4_4():
    """Parent at the root with psi_p = 12°, child psi_c = 5°: delta = -7°, residual 0°; wrong sign 14°."""
    psi_p, psi_c = np.deg2rad(12.0), np.deg2rad(5.0)
    s_c, m_c = random_rotations((2,), 0)
    q_c = reports(psi_c, s_c, m_c)
    delta = wrap_rad(psi_c - psi_p)
    assert np.rad2deg(delta) == pytest.approx(-7.0, abs=1e-12)

    c_c = 0.0 + delta  # c_root = 0, c_c = c_p + delta_pc
    shared = reports(psi_p, s_c, m_c)  # Y(-12°)·s_c·m_c: the child shares the parent's north
    good = quat_mul(yaw_quat(c_c), q_c)
    bad = quat_mul(yaw_quat(-c_c), q_c)
    assert np.rad2deg(geodesic_angle_rad(quat_mul(good, quat_conj(shared)))) == pytest.approx(0.0, abs=1e-6)
    assert np.rad2deg(geodesic_angle_rad(quat_mul(bad, quat_conj(shared)))) == pytest.approx(14.0, abs=1e-9)
    # The residual is a pure twist about the vertical, so the heading shows the same 14°.
    assert np.rad2deg(wrap_rad(heading_rad(bad) - heading_rad(shared))) == pytest.approx(14.0, abs=1e-9)


def test_same_example_through_the_tree_functions():
    psi = np.zeros(N_PODS)
    psi[ROOT_POD] = np.deg2rad(12.0)
    lhip = next(k for k, j in enumerate(JOINTS) if j.name == "l_hip")
    child = JOINTS[lhip].child
    psi[child] = np.deg2rad(5.0)
    delta = relative_heading_rad(psi)
    assert np.rad2deg(delta[lhip]) == pytest.approx(-7.0)
    c = chain_corrections_rad(delta)
    assert c[ROOT_POD] == 0.0
    assert np.rad2deg(c[child]) == pytest.approx(-7.0)


def test_exact_estimates_give_one_shared_heading():
    """With delta_hat = delta, every corrected pod equals Y(-psi_root)·s_i·m_i."""
    rng = np.random.default_rng(1)
    psi = rng.uniform(-np.pi, np.pi, (8, N_PODS))
    s = random_rotations((8, N_PODS), 2)
    m = random_rotations((8, N_PODS), 3)
    q = reports(psi, s, m)
    q_tilde = apply_corrections(q, chain_corrections_rad(relative_heading_rad(psi)))
    target = reports(np.repeat(psi[:, ROOT_POD : ROOT_POD + 1], N_PODS, axis=1), s, m)
    assert np.max(geodesic_angle_rad(quat_mul(q_tilde, quat_conj(target)))) < 1e-9


def test_correction_never_changes_tilt():
    rng = np.random.default_rng(4)
    q = random_rotations((50, N_PODS), 5)
    c = chain_corrections_rad(rng.uniform(-np.pi, np.pi, (50, N_JOINTS)))
    np.testing.assert_allclose(gravity_in_body(apply_corrections(q, c)), gravity_in_body(q), atol=1e-12)


def test_common_yaw_leaves_targets_unchanged_exactly():
    """Rotating every report by Y(beta) is psi_i -> psi_i - beta for all i; delta must not move."""
    rng = np.random.default_rng(6)
    psi = rng.uniform(-np.pi, np.pi, (100, N_PODS))
    beta = rng.uniform(-np.pi, np.pi, (100, 1))
    d0 = relative_heading_rad(psi)
    d1 = relative_heading_rad(psi - beta)
    assert np.max(np.abs(wrap_rad(d1 - d0))) < 1e-12


def test_single_pod_offset_moves_only_its_joints_by_gamma():
    rng = np.random.default_rng(7)
    psi = rng.uniform(-np.pi, np.pi, N_PODS)
    gamma = np.deg2rad(23.0)
    for pod in range(N_PODS):
        moved = psi.copy()
        moved[pod] += gamma
        diff = wrap_rad(relative_heading_rad(moved) - relative_heading_rad(psi))
        for k, j in enumerate(JOINTS):
            if j.child == pod:
                assert diff[k] == pytest.approx(gamma, abs=1e-12)
            elif j.parent == pod:
                assert diff[k] == pytest.approx(-gamma, abs=1e-12)
            else:
                assert diff[k] == 0.0
