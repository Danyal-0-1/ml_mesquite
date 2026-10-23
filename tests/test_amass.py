"""Check 10 (brief §9): the AMASS loader and the AMASS → Mesquite mapping.

Synthetic tests always run. Tests marked ``amass`` use the downloaded data and skip without it.
"""

import numpy as np
import pytest

from mesqsim.body import POD_BY_NAME
from mesqsim.conventions import (
    exp_map,
    geodesic_angle_rad,
    log_map,
    quat_conj,
    quat_mul,
    quat_rotate,
    yaw_quat,
)
from mesqsim.data import amass
from mesqsim.synth import make_sequence, write_amass_npz


def max_angle_deg(a, b):
    return float(np.rad2deg(np.max(geodesic_angle_rad(quat_mul(quat_conj(a), b)))))


def test_mapping_table_matches_the_brief():
    expected = {
        "HipsAlt": 0, "Spine": 9, "Head": 15, "LeftUpLeg": 1, "RightUpLeg": 2, "LeftLeg": 4, "RightLeg": 5,
        "LeftFoot": 7, "RightFoot": 8, "LeftArm": 16, "RightArm": 17, "LeftForeArm": 18, "RightForeArm": 19,
        "LeftHand": 20, "RightHand": 21,
    }  # fmt: skip
    for name, joint in expected.items():
        assert amass.POD_TO_SMPL[POD_BY_NAME[name].id] == joint, name
    assert amass.SMPL_JOINT_NAMES[9] == "spine3" and amass.SMPL_JOINT_NAMES[6] == "spine2"


def test_forward_kinematics_composes_down_the_tree():
    rng = np.random.default_rng(0)
    poses = rng.normal(0, 0.4, (5, 66))
    glob = amass.global_rotations(poses)
    local = exp_map(poses.reshape(5, 22, 3))
    # the left wrist's chain: pelvis 0 -> spine1 3 -> spine2 6 -> spine3 9 -> collar 13 -> shoulder 16 -> elbow 18 -> 20
    chain = [0, 3, 6, 9, 13, 16, 18, 20]
    manual = local[:, chain[0]]
    for j in chain[1:]:
        manual = quat_mul(manual, local[:, j])
    assert max_angle_deg(glob[:, 20], manual) < 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_rest_pose_every_segment_equals_the_facing(seed):
    """Check 10.1: with every local rotation zero, all 15 segments equal R_0 (the facing rotation)."""
    rng = np.random.default_rng(seed)
    root = quat_mul(yaw_quat(rng.uniform(-np.pi, np.pi)), exp_map([np.pi / 2, 0, 0]))
    poses = np.zeros((1, 66))
    poses[0, :3] = log_map(root)
    seg = amass.segment_orientations(poses)
    assert seg.shape == (1, 15, 4)
    assert max_angle_deg(seg[0], np.broadcast_to(root, (15, 4))) < 1e-9


def test_spine2_alternative_and_validation():
    seq = make_sequence("walk", 2.0, 60.0, seed=1)
    s3 = amass.segment_orientations(seq.poses)
    s2 = amass.segment_orientations(seq.poses, spine_joint=amass.SPINE2)
    np.testing.assert_allclose(np.delete(s3, 1, axis=1), np.delete(s2, 1, axis=1))
    assert max_angle_deg(s3[:, 1], s2[:, 1]) > 1.0  # spine3 differs from spine2 in this motion
    with pytest.raises(ValueError):
        amass.segment_orientations(seq.poses, spine_joint=3)


@pytest.mark.parametrize("smplx_style", [False, True])
def test_loader_reads_both_amass_layouts(tmp_path, smplx_style):
    seq = make_sequence("walk", 1.0, 120.0, seed=2)
    path = write_amass_npz(seq, tmp_path / "x_poses.npz", smplx_style=smplx_style)
    back = amass.load_sequence(path)
    assert back.fps == 120.0 and back.n_frames == seq.n_frames and back.gender == "neutral"
    np.testing.assert_allclose(back.poses, seq.poses)
    assert amass.segment_orientations(back.poses).shape == (seq.n_frames, 15, 4)


def test_loader_rejects_shape_files(tmp_path):
    np.savez(tmp_path / "shape.npz", betas=np.zeros(16), gender=np.array("female"))
    with pytest.raises(ValueError, match="not a motion file"):
        amass.load_sequence(tmp_path / "shape.npz")


def test_gender_stored_as_bytes(tmp_path):
    np.savez(tmp_path / "a_poses.npz", poses=np.zeros((5, 156)), trans=np.zeros((5, 3)), betas=np.zeros(16),
             gender=np.array(b"female"), mocap_framerate=30.0)
    assert amass.load_sequence(tmp_path / "a_poses.npz").gender == "female"


def test_motion_file_discovery_and_subjects(tmp_path):
    seq = make_sequence("walk", 0.5, 30.0)
    for rel in ("CMU/01/01_01_poses.npz", "CMU/02/02_05_poses.npz", "KIT/3/walking_run01_poses.npz"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        write_amass_npz(seq, tmp_path / rel)
    np.savez(tmp_path / "CMU" / "01" / "shape.npz", betas=np.zeros(16))
    files = amass.motion_files(tmp_path)
    assert [f.relative_to(tmp_path).as_posix() for f in files] == [
        "CMU/01/01_01_poses.npz", "CMU/02/02_05_poses.npz", "KIT/3/walking_run01_poses.npz"]  # fmt: skip
    assert amass.subset_subject(files[2], tmp_path) == ("KIT", "3")
    assert amass.find_walking_sequence(tmp_path) == files[2]


def test_synthetic_up_axis_left_right_and_knees():
    """The checks of 10.2-10.4 on synthetic motion, where the answers are known by construction."""
    seq = make_sequence("walk", 6.0, 60.0, seed=4)
    seg = amass.segment_orientations(seq.poses)
    axis, cos = amass.estimate_world_up(seg[:, POD_BY_NAME["HipsAlt"].id])
    np.testing.assert_array_equal(axis, [0, 0, 1])
    assert cos > 0.99
    left, right = amass.hip_flexion_rad(seg, "left"), amass.hip_flexion_rad(seg, "right")
    assert np.corrcoef(left, right)[0, 1] < -0.9
    assert np.rad2deg(left.max()) == pytest.approx(25.0, abs=0.5)  # forward swing is positive
    knee = np.rad2deg(amass.knee_flexion_rad(seg, "left"))
    assert knee.min() == pytest.approx(5.0, abs=0.5) and knee.max() == pytest.approx(60.0, abs=0.5)


def test_subject_left_is_body_plus_x():
    """At the rest pose the subject's left (SMPL +x) is up × forward in the world."""
    seq = make_sequence("tpose", 0.1, 30.0, seed=5)
    r0 = amass.global_rotations(seq.poses)[0, 0]
    up = quat_rotate(r0, amass.SMPL_BODY_UP)
    forward = quat_rotate(r0, [0.0, 0.0, 1.0])
    left = quat_rotate(r0, amass.SMPL_BODY_LEFT)
    np.testing.assert_allclose(up, [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(np.cross(up, forward), left, atol=1e-12)


# ------------------------------------------------------------------ real AMASS (skips without data)


@pytest.mark.amass
def test_one_amass_sequence_gives_15_orientations(amass_files):
    seq = amass.load_sequence(amass_files[0])
    seg = amass.segment_orientations(seq.poses)
    assert seg.shape == (seq.n_frames, 15, 4)
    assert np.all(np.isfinite(seg))
    np.testing.assert_allclose(np.linalg.norm(seg, axis=-1), 1.0, atol=1e-9)


@pytest.mark.amass
def test_amass_world_is_z_up(amass_files):
    roots = np.concatenate([amass.global_rotations(amass.load_sequence(f).poses)[:, 0] for f in amass_files[:20]])
    axis, cos = amass.estimate_world_up(roots)
    np.testing.assert_array_equal(axis, amass.AMASS_WORLD_UP)
    assert cos > 0.8


@pytest.mark.amass
def test_amass_walking_thighs_alternate(amass_files):
    path = amass.find_walking_sequence()
    if path is None:
        pytest.skip("no AMASS file name mentions walking")
    seg = amass.segment_orientations(amass.load_sequence(path).poses)
    assert np.corrcoef(amass.hip_flexion_rad(seg, "left"), amass.hip_flexion_rad(seg, "right"))[0, 1] < -0.3


@pytest.mark.amass
def test_amass_knee_flexion_in_anatomical_range(amass_files):
    for f in amass_files[:50]:
        seg = amass.segment_orientations(amass.load_sequence(f).poses)
        for side in ("left", "right"):
            k = np.rad2deg(amass.knee_flexion_rad(seg, side))
            assert k.min() > -20.0 and k.max() < 170.0, (f, side, k.min(), k.max())
