"""Check 2 (brief §9): every rotation convention of §4.1 is pinned here."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.spatial.transform import Rotation

from mesqsim.conventions import (
    circ_mean_rad,
    circ_std_rad,
    exp_map,
    geodesic_angle_rad,
    gravity_in_body,
    heading_rad,
    interp_quat,
    log_map,
    quat_canonical,
    quat_conj,
    quat_from_matrix,
    quat_mean,
    quat_mul,
    quat_normalize,
    quat_rotate,
    quat_slerp,
    quat_to_matrix,
    wrap_deg,
    wrap_rad,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
    yaw_quat,
)

EX, EY, EZ = np.eye(3)
angles = st.floats(-np.pi, np.pi, allow_nan=False)
unit_quats = st.lists(st.floats(-1, 1), min_size=4, max_size=4).filter(lambda v: np.linalg.norm(v) > 0.2)


def rand_quats(n, seed=0):
    return quat_normalize(np.random.default_rng(seed).normal(size=(n, 4)))


def test_y90_maps_x_to_y():
    np.testing.assert_allclose(quat_rotate(yaw_quat(np.pi / 2), EX), EY, atol=1e-15)
    np.testing.assert_allclose(quat_rotate(yaw_quat(np.pi / 2), EY), -EX, atol=1e-15)
    np.testing.assert_allclose(quat_rotate(yaw_quat(np.pi / 2), EZ), EZ, atol=1e-15)


def test_quaternion_order_is_xyzw_like_scipy():
    q = rand_quats(50)
    np.testing.assert_allclose(quat_to_matrix(q), Rotation.from_quat(q).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(quat_canonical(quat_from_matrix(quat_to_matrix(q))), quat_canonical(q), atol=1e-12)
    np.testing.assert_array_equal(wxyz_to_xyzw([1.0, 2.0, 3.0, 4.0]), [2.0, 3.0, 4.0, 1.0])
    np.testing.assert_array_equal(xyzw_to_wxyz([2.0, 3.0, 4.0, 1.0]), [1.0, 2.0, 3.0, 4.0])


def test_hamilton_product():
    i, j, k = np.array([1.0, 0, 0, 0]), np.array([0, 1.0, 0, 0]), np.array([0, 0, 1.0, 0])
    np.testing.assert_array_equal(quat_mul(i, j), k)  # ij = k (Hamilton, not JPL)
    np.testing.assert_array_equal(quat_mul(j, i), -k)
    a, b = rand_quats(30, 1), rand_quats(30, 2)
    expected = (Rotation.from_quat(a) * Rotation.from_quat(b)).as_quat()
    np.testing.assert_allclose(quat_canonical(quat_mul(a, b)), quat_canonical(expected), atol=1e-12)


def test_composition_order_is_right_to_left():
    """quat_mul(a, b) applies b first: Rz(90)·Rx(90) sends +y to +z, Rx(90)·Rz(90) sends +y to -x."""
    rz, rx = yaw_quat(np.pi / 2), exp_map([np.pi / 2, 0, 0])
    np.testing.assert_allclose(quat_rotate(quat_mul(rz, rx), EY), EZ, atol=1e-15)
    np.testing.assert_allclose(quat_rotate(quat_mul(rx, rz), EY), -EX, atol=1e-15)
    a, b = rand_quats(20, 3), rand_quats(20, 4)
    v = np.random.default_rng(5).normal(size=(20, 3))
    np.testing.assert_allclose(quat_rotate(quat_mul(a, b), v), quat_rotate(a, quat_rotate(b, v)), atol=1e-12)


def test_orientation_maps_body_to_world():
    q = rand_quats(10, 6)
    v_body = np.random.default_rng(7).normal(size=(10, 3))
    v_world = np.einsum("nij,nj->ni", quat_to_matrix(q), v_body)
    np.testing.assert_allclose(quat_rotate(q, v_body), v_world, atol=1e-12)


@settings(max_examples=300, deadline=None)
@given(unit_quats)
def test_q_and_minus_q_are_the_same_rotation(v):
    q = quat_normalize(np.array(v))
    np.testing.assert_allclose(quat_to_matrix(q), quat_to_matrix(-q), atol=1e-12)
    np.testing.assert_allclose(heading_rad(q), heading_rad(-q), atol=1e-12)
    assert quat_canonical(q)[3] >= 0 and quat_canonical(-q)[3] >= 0


@settings(max_examples=500, deadline=None)
@given(angles, st.floats(-1.4, 1.4), angles, angles)
def test_heading_shifts_by_alpha(yaw, pitch, roll, alpha):
    """heading(Y(a)·R) = wrap(heading(R) + a), away from gimbal lock (|pitch| < 80°)."""
    q = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_quat()
    h = heading_rad(q)
    assert abs(wrap_rad(h - yaw)) < 1e-9  # heading is the first intrinsic Z-Y-X angle
    shifted = heading_rad(quat_mul(yaw_quat(alpha), q))
    assert abs(wrap_rad(shifted - wrap_rad(h + alpha))) < 1e-9


def test_wrap_values():
    assert wrap_deg(190.0) == -170.0
    assert wrap_deg(-180.0) == 180.0
    assert wrap_deg(180.0) == 180.0
    assert wrap_deg(540.0) == 180.0
    assert wrap_deg(-190.0) == 170.0
    assert wrap_deg(0.0) == 0.0
    assert wrap_rad(-np.pi) == np.pi
    np.testing.assert_allclose(wrap_rad(3 * np.pi / 2), -np.pi / 2)


@settings(max_examples=500, deadline=None)
@given(st.floats(-1e4, 1e4))
def test_wrap_range_and_equivalence(x):
    w = float(wrap_deg(x))
    assert -180.0 < w <= 180.0
    k = (x - w) / 360.0
    assert abs(k - round(k)) < 1e-9


def test_exp_log_roundtrip_and_geodesic_angle():
    rv = np.random.default_rng(8).normal(size=(100, 3))
    rv *= (np.random.default_rng(9).uniform(0, np.pi - 1e-6, 100) / np.linalg.norm(rv, axis=1))[:, None]
    np.testing.assert_allclose(log_map(exp_map(rv)), rv, atol=1e-9)
    np.testing.assert_allclose(geodesic_angle_rad(exp_map(rv)), np.linalg.norm(rv, axis=1), atol=1e-12)
    np.testing.assert_allclose(exp_map(np.zeros(3)), [0, 0, 0, 1])
    np.testing.assert_allclose(log_map([0, 0, 0, 1.0]), np.zeros(3))
    np.testing.assert_allclose(geodesic_angle_rad(-exp_map([0.3, 0, 0])), 0.3, atol=1e-12)
    np.testing.assert_allclose(exp_map(rv), quat_canonical(Rotation.from_rotvec(rv).as_quat()), atol=1e-12)


def test_slerp_and_interpolation():
    q0, q1 = rand_quats(1, 10)[0], rand_quats(1, 11)[0]
    np.testing.assert_allclose(quat_canonical(quat_slerp(q0, q1, 0.0)), quat_canonical(q0), atol=1e-12)
    np.testing.assert_allclose(quat_canonical(quat_slerp(q0, q1, 1.0)), quat_canonical(q1), atol=1e-12)
    half = quat_slerp(q0, q1, 0.5)
    rel = quat_mul(quat_conj(q0), q1)
    assert abs(geodesic_angle_rad(quat_mul(quat_conj(q0), half)) - 0.5 * geodesic_angle_rad(rel)) < 1e-12
    # a constant-rate rotation about z is interpolated exactly
    t = np.arange(5.0)
    q = yaw_quat(0.3 * t)
    np.testing.assert_allclose(heading_rad(interp_quat(t, q, [0.5, 2.25, 9.0])), [0.15, 0.675, 1.2], atol=1e-12)


def test_gravity_in_body_ignores_heading():
    q = rand_quats(20, 12)
    g = gravity_in_body(q)
    np.testing.assert_allclose(gravity_in_body(quat_mul(yaw_quat(1.234), q)), g, atol=1e-12)
    np.testing.assert_allclose(gravity_in_body([0, 0, 0, 1.0]), EZ)


def test_circular_statistics():
    a = np.deg2rad([179.0, -179.0, 178.0, -178.0])
    assert abs(wrap_deg(np.rad2deg(circ_mean_rad(a)) - 180.0)) < 1e-9
    assert np.rad2deg(circ_std_rad(a)) < 2.0
    x = np.random.default_rng(13).normal(0, np.deg2rad(1.0), 100000)
    assert abs(np.rad2deg(circ_std_rad(x)) - 1.0) < 0.01


def test_quat_mean_handles_sign_flips():
    base = rand_quats(1, 14)[0]
    noise = exp_map(np.random.default_rng(15).normal(0, 1e-3, (50, 3)))
    q = quat_mul(base, noise)
    q[::2] *= -1
    assert geodesic_angle_rad(quat_mul(quat_conj(quat_mean(q)), base)) < 1e-3
