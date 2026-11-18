"""mesqsim.twin.page_frame: the one place that converts the page's, Motive's and the repository's frames."""

import numpy as np
from scipy.spatial.transform import Rotation

from mesqsim.conventions import heading_rad, quat_rotate, quat_to_matrix
from mesqsim.twin import page_frame as pf


def test_three_to_world_is_a_proper_rotation_taking_up_to_up():
    r = pf.R_THREE_TO_WORLD
    assert np.allclose(r @ r.T, np.eye(3)) and np.isclose(np.linalg.det(r), 1.0)
    assert np.allclose(pf.three_to_world_vec([0, 1, 0]), [0, 0, 1])  # +Y up -> +Z up
    assert np.allclose(pf.three_to_world_vec([0, 0, 1]), [0, -1, 0])  # the avatar's forward -> world -Y
    assert np.allclose(pf.three_to_world_vec([1, 0, 0]), [1, 0, 0])  # its left stays +X
    v = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(pf.world_to_three_vec(pf.three_to_world_vec(v)), v)
    assert np.allclose(quat_to_matrix(pf.Q_THREE_TO_WORLD), r)


def test_motive_conversions():
    assert np.isclose(np.linalg.det(pf.R_MOTIVE_TO_THREE), 1.0)
    assert np.allclose(pf.motive_to_three_vec([1, 2, 3]), [1, 3, -2])  # (x, y, z) -> (x, z, -y), brief §5.1
    v = np.random.default_rng(1).normal(size=(10, 3))
    assert np.allclose(pf.motive_to_world_vec(v), v)  # both worlds +Z up and right-handed: the identity
    # Danyal facing -Y (left +X) in Motive faces the avatar's forward (+Z, left +X) in three.js
    assert np.allclose(pf.motive_to_three_vec([0, -1, 0]), [0, 0, 1])


def test_three_to_world_rot_agrees_with_vectors_and_turns_about_up():
    q = Rotation.random(20, random_state=2).as_quat()
    v = np.array([0.3, -0.2, 0.9])
    assert np.allclose(quat_rotate(pf.three_to_world_rot(q), v), pf.three_to_world_vec(quat_rotate(q, v)))
    # a turn by a about three.js +Y is a heading change of a in the world
    for a in (-120.0, 15.0, 170.0):
        h = heading_rad(pf.three_to_world_rot(pf.axis_quat("Y", np.deg2rad(a))))
        assert np.isclose(np.rad2deg(h), a)


def test_bvh_euler_is_the_product_in_channel_order():
    a = np.random.default_rng(3).uniform(-180, 180, size=(50, 3))
    for order in ("XYZ", "ZXY", "YZX"):
        q = pf.bvh_euler_to_quat(a, order)
        ref = Rotation.from_euler(order, a, degrees=True).as_quat()  # upper case: intrinsic, R1 · R2 · R3
        assert np.allclose(np.abs(np.sum(q * ref, axis=1)), 1.0)
    assert np.allclose(quat_rotate(pf.bvh_euler_to_quat([90.0, 0.0, 0.0], "XYZ"), [0, 1, 0]), [0, 0, 1])
