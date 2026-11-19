"""The twin's body and physics (Step 3 brief §4.2-4.3; check 41.1): exact checks on synthetic motion, then SMPL+H on
AMASS (marked amass, skipped without the licensed model files)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation, RotationSpline

from mesqsim.twin.physics import G_M_S2, imu_truth, position_spline
from mesqsim.twin.sites import MOUNT_CHIP, MOUNT_PHONE, SITE_VERTEX, SITES, surface_frames

T = np.linspace(0.0, 4.0, 4001)  # 1 kHz samples, so the interpolating spline is exact enough to compare at 1e-6


def test_still_pod_feels_only_gravity():
    q = np.tile(Rotation.from_euler("xyz", [20, -35, 110], degrees=True).as_quat(), (T.size, 1))
    imu = imu_truth(T, q, np.tile([0.3, -1.2, 0.9], (T.size, 1)), 225.0)
    assert np.abs(imu.omega_rad_s).max() < 1e-12
    expected = Rotation.from_quat(q[0]).inv().apply(-G_M_S2)  # +1 g up, in the pod's frame
    assert np.abs(imu.f_m_s2 - expected).max() < 1e-9


def test_constant_spin_about_a_body_axis_is_a_constant_rate():
    w = np.array([0.4, -1.1, 2.3])  # rad/s, in the pod's frame
    r0 = Rotation.from_euler("zyx", [30, 10, -60], degrees=True)
    q = (r0 * Rotation.from_rotvec(np.outer(T, w))).as_quat()
    imu = imu_truth(T, q, np.zeros((T.size, 3)), 225.0)
    assert np.abs(imu.omega_rad_s - w).max() < 1e-6


def test_rotation_spline_rate_is_in_the_body_frame():
    """scipy's RotationSpline(t, 1) is the body-frame rate: a spin about *world* z from a tilted start reads
    R0⁻¹ (0, 0, w), not (0, 0, w) (the brief asks to test which frame it is)."""
    w, r0 = 1.3, Rotation.from_euler("x", 50, degrees=True)
    rs = RotationSpline(T, Rotation.from_rotvec(np.outer(w * T, [0, 0, 1])) * r0)
    assert np.allclose(rs(np.array([1.7]), 1)[0], r0.inv().apply([0, 0, w]), atol=1e-9)


def test_free_fall_feels_nothing():
    q = np.tile(Rotation.from_euler("y", 25, degrees=True).as_quat(), (T.size, 1))
    p = np.array([0.1, 0.2, 30.0]) + np.outer(T, [1.0, -0.5, 2.0]) + 0.5 * np.outer(T ** 2, G_M_S2)
    # a cubic spline holds a parabola exactly; what is left is rounding: tens of metres twice differentiated 1 ms apart
    assert np.abs(imu_truth(T, q, p, 225.0).f_m_s2).max() < 1e-6


def test_circle_at_constant_speed_gives_the_centripetal_force():
    r, w = 0.8, 2 * np.pi * 0.7
    p = np.stack([r * np.cos(w * T), r * np.sin(w * T), np.full(T.size, 1.0)], axis=1)
    q = np.tile([0.0, 0.0, 0.0, 1.0], (T.size, 1))
    imu = imu_truth(T, q, p, 225.0, t_eval=np.linspace(0.5, 3.5, 301))
    t = imu.t_s
    expected = np.stack([-r * w ** 2 * np.cos(w * t), -r * w ** 2 * np.sin(w * t), np.full(t.size, 9.81)], axis=1)
    # exact up to the interpolating spline's own error, O(h² ω⁴ r) = 1e-6 x 19.3² x 0.8 / 12 at 1 kHz
    assert np.abs(imu.f_m_s2 - expected).max() < 1e-4 * r * w ** 2


@pytest.mark.parametrize("ratio, lo, hi", [(0.25, 0.98, 1.0), (1.0, 0.4, 0.6), (4.0, 0.0, 0.02)])
def test_smoothing_halves_the_amplitude_at_the_cutoff(ratio, lo, hi):
    """λ = h / (2π f_c)⁴ makes the smoothing spline a filter 1 / (1 + (f / f_c)⁴): a half at f_c."""
    t = np.arange(0.0, 20.0, 1 / 120)
    fc = 8.0
    x = np.sin(2 * np.pi * ratio * fc * t)
    y = position_spline(t, np.stack([x, x, x], axis=1), fc)(t)[:, 0]
    core = slice(240, -240)
    amp = np.sqrt(2 * np.mean(y[core] ** 2))
    assert lo <= amp <= hi, amp


def test_surface_frames_are_right_handed_with_the_normal_last():
    rng = np.random.default_rng(3)
    n, bone = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    f = surface_frames(n, bone)
    assert np.allclose(np.einsum("nij,nik->njk", f, f), np.eye(3), atol=1e-12)
    assert np.allclose(np.linalg.det(f), 1.0)
    assert np.allclose(f[:, :, 2], n / np.linalg.norm(n, axis=1, keepdims=True))
    assert np.allclose(np.sum(f[:, :, 0] * bone, axis=1) > 0, True)  # t̂ keeps the bone's sense


def test_mountings():
    """The chip's z is the face (out of the skin) on every pod; the case points up the thigh and shin, a half turn
    about the normal from the bone's downward axis; the phone's device axes are (b̂, t̂, -n̂)."""
    for name, q in MOUNT_CHIP.items():
        m = Rotation.from_quat(q).as_matrix()
        assert np.allclose(m[:, 2], [0, 0, 1]), name
        legs = name in ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")
        assert np.allclose(m[:, 0], [-1, 0, 0] if legs else [1, 0, 0], atol=1e-12), name
    m = Rotation.from_quat(MOUNT_PHONE).as_matrix()
    assert np.allclose(m, [[0, 1, 0], [1, 0, 0], [0, 0, -1]], atol=1e-12) and np.isclose(np.linalg.det(m), 1.0)


# ------------------------------------------------------------------------------------------- with SMPL+H

@pytest.fixture(scope="module")
def walk(amass_files):
    from mesqsim.data.amass import find_walking_sequence
    from mesqsim.twin import smplh as sm

    path = find_walking_sequence()
    if path is None:
        pytest.skip("no AMASS walking sequence")
    seq = sm.load_smplh_sequence(path)
    return seq, sm.load_model(seq.gender)


@pytest.mark.amass
def test_numpy_smplh_matches_smplx(walk):
    torch = pytest.importorskip("torch")
    from smplx.lbs import lbs

    from mesqsim.twin import smplh as sm

    seq, model = walk
    frames = slice(0, 30)
    v_shaped, j_rest = sm.rest_pose(model, seq.betas)
    idx = np.array(sorted(SITE_VERTEX.values()))
    mine = sm.skin(model, seq.poses[frames], v_shaped, j_rest, idx, trans=seq.trans[frames])
    with torch.no_grad():
        verts, joints = lbs(torch.tensor(np.tile(seq.betas[:16], (30, 1))), torch.tensor(seq.poses[frames]),
                            torch.tensor(model.v_template), torch.tensor(model.shapedirs),
                            torch.tensor(model.posedirs.reshape(-1, model.posedirs.shape[-1]).T),
                            torch.tensor(model.j_regressor), torch.tensor(model.parents),
                            torch.tensor(model.weights))
    assert np.abs(mine - (verts.numpy()[:, idx] + seq.trans[frames][:, None])).max() < 1e-7
    assert np.abs(sm.forward(model, seq.poses[frames], j_rest)[1] - joints.numpy()).max() < 1e-7


@pytest.mark.amass
def test_site_table_and_normals(walk):
    from mesqsim.twin import smplh as sm
    from mesqsim.twin.sites import face_rings, site_vertex

    neutral = sm.load_model("neutral")
    for s in SITES:
        assert site_vertex(neutral, s) == SITE_VERTEX[s.name], s.name
        for gender in ("female", "male"):
            m = sm.load_model(gender)
            v = SITE_VERTEX[s.name]
            ring = face_rings(m, [v])
            pts = {int(i): m.v_template[int(i)][None] for f in ring[v] for i in m.faces[f]}
            n = sm.vertex_normals(m, pts, [v], ring)[0, 0]
            assert n @ np.asarray(s.direction) > 0.4, (s.name, gender, n)  # outward, the way the pod faces


@pytest.mark.amass
def test_skin_moves_with_its_bone_on_a_walk(walk):
    from mesqsim.twin.bodymotion import body_tracks, skin_against_bone_deg

    seq, model = walk
    tr = body_tracks(seq, model)
    assert tr.q.shape == (seq.poses.shape[0], 16, 4) and np.allclose(np.linalg.norm(tr.q, axis=-1), 1.0)
    for k, name in enumerate(tr.names):
        assert np.median(skin_against_bone_deg(tr, k)) < 5.0, name
