"""Long sessions, the procedure and one twin session end to end (Step 3 brief §4.8-4.12; checks 40.2, 43.2,
46.1-48.1). Marked amass: they need the licensed AMASS files and SMPL+H model, and skip cleanly without them."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mesqsim.body import JOINTS, N_PODS
from mesqsim.conventions import wrap_rad
from mesqsim.twin import session_checks as sc

pytestmark = pytest.mark.amass


@pytest.fixture(scope="module", autouse=True)
def _data():
    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")


def test_protocol_program_is_the_briefs_script():
    from mesqsim.twin.program import join_check

    joined = sc._program("protocol", sc.SEED, None)[0]
    kinds = [p["kind"] for p in joined.pieces]
    assert kinds[0] == "tpose" and kinds[-1] == "tpose" and kinds[-2] == "still" and kinds[1] == "twist"
    assert kinds.count("walk") == 6 and kinds.count("squat") == 3 and kinds.count("turn_spot") == 2
    assert kinds.count("leg_swing") == 2 and kinds.count("leg_side") == 2 and kinds.count("arm_forward") == 2
    assert join_check(joined)["ok"]
    from mesqsim.twin.smplh import rest_pose

    model = sc._program("protocol", sc.SEED, None)[1]
    pel = joined.seq.trans + rest_pose(model, joined.seq.betas)[1][0]
    walks = [p for p in joined.pieces if p["kind"] == "walk"]
    for p in walks:  # every leg on the same line, there and back
        d = pel[p["last"], :2] - pel[p["first"], :2]
        assert np.linalg.norm(d) > 3.9 and abs(pel[p["first"], 0]) < 0.2 and abs(pel[p["last"], 0]) < 0.2
    fwd = Rotation.from_rotvec(joined.seq.poses[:, :3]).apply([0, 0, 1.0])
    yaw = np.unwrap(np.arctan2(fwd[:, 1], fwd[:, 0]))
    spots = [p for p in joined.pieces if p["kind"] == "turn_spot"]
    assert [round(np.rad2deg(yaw[p["last"]] - yaw[p["first"]])) for p in spots] == [360, -360]


def test_random_program_stays_with_one_subject_and_joins_smoothly():
    from mesqsim.twin.program import join_check

    joined = sc._program("random", sc.SEED, sc.MINUTES)[0]
    clips = [p["source"] for p in joined.pieces if p["source"].startswith("clip:")]
    assert clips and all(c.startswith("clip:KIT/3/") for c in clips)
    assert joined.seq.poses.shape[0] / joined.seq.fps >= 60.0 * sc.MINUTES - 5.0
    r = join_check(joined)
    assert r["ok"] and r["transition_max_rad_s"] <= r["p99_rad_s"]
    assert joined.transition.sum() == 30 * r["joins"]  # 0.5 s at 60 Hz per join


def test_session_tables_and_labels():
    s = sc.short_session()
    assert set(s.frames.columns) == {"t_host_s", "pod_id", "batt", "qx", "qy", "qz", "qw", "count", "ms_lo"}
    assert s.frames["pod_id"].nunique() == N_PODS and np.all(np.diff(s.frames["t_host_s"]) >= 0)
    assert [p["name"] for p in s.session["phases"]] == ["box", "strapping", "tpose", "motion"]
    assert len(s.labels_joints) == len(s.labels_steps) * len(JOINTS)
    assert list(s.phone.columns) == ["t_host_s", "line", "qx", "qy", "qz", "qw", "px", "py", "pz", "len_bytes"]
    assert s.phone["len_bytes"].max() <= 95 and len(s.labels_phone) == len(s.phone)
    slipped = s.plan.slips["zero_at_start"]  # seed 3 draws this slip: only the automatic Zero at Start
    assert s.meta["counts"]["zeros"] == (1 if slipped else 2) and (s.plan.phone["zero_request_s"] is None) == slipped
    r = sc.table_numbers()
    assert r["eps_max_rad"] < 1e-12 and r["delta_max_rad"] < 1e-12  # recomputed from the stored ψ
    t = s.labels_ticks
    assert np.nanmax(np.abs(t["tilt_rad"])) < np.deg2rad(30.0)  # tilt from strapping spins and walking only
    box = t[t["phase"] == "box"]
    # still in the box, the fused tilt is the accelerometer's: misalignment within ±1° per axis (√3° together) plus a
    # bias within 0.05 m/s² (0.3°) and scale errors of 1 %
    assert np.nanmax(np.rad2deg(box["tilt_rad"])) < np.sqrt(3.0) + 0.3 + 0.6


def test_slips_restart_a_limb_and_mistake_the_tree():
    """Seed 6 draws a USB restart of the right arm's two upper pods and the wrong tree setting in the app."""
    from mesqsim.conventions import quat_canonical
    from mesqsim.sim.heading import twist_z_rad

    s = sc.short_session(seed=6)
    lr = s.plan.slips["limb_restart"]
    assert lr["what"] == "right_arm" and lr["pods"] == [6, 7] and s.plan.slips["tree_mismatch"]
    assert s.plan.phone["settings"]["tree"] == "matched"
    t = s.labels_ticks
    for pod in lr["pods"]:
        g = t[t["pod_id"] == pod]
        assert sorted(g["epoch"].unique()) == [0, 1]
        first = g[g["epoch"] == 1].iloc[0]
        r = quat_canonical(first[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]].to_numpy(dtype=float))
        # the restarted filter starts with no twist: its new north is the pod's own heading where it faces
        assert abs(wrap_rad(first["psi_rad"] - twist_z_rad(r))) < np.deg2rad(0.5)
        assert first["t_sample_s"] > lr["t_s"]
    others = t[~t["pod_id"].isin(lr["pods"])]
    assert set(others["epoch"].unique()) == {0}
    steps = s.labels_steps
    assert steps["slip_restarted"].any() and steps["slip_tree_mismatch"].all()


def test_invariances_of_section_4_12():
    r = sc.invariance_numbers()
    c, o, m = r["common"], r["one"], r["room"]
    assert c["psi_shift_err"] < 1e-9 and c["psi_ph_shift_err"] < 1e-9 and c["eps"] < 1e-9 and c["delta"] < 1e-9
    assert o["psi_pod_err"] < 1e-9 and o["psi_others"] == 0.0 and o["eps_pod_err"] < 1e-9 and o["eps_others"] == 0.0
    assert o["delta_err"] < 1e-9 and o["joints_moved"] == ["l_ankle", "l_knee"]  # pod 10, the left shin
    assert m["psi"] < 1e-9 and m["psi_ph"] < 1e-9 and m["delta"] < 1e-9 and m["phone_lines_equal"]


@pytest.mark.slow
def test_estimators_do_not_see_a_common_turn():
    r = sc.estimator_invariance()
    assert r["B1"] < 0.5 and r["GEO-ID"] < 0.5, r


@pytest.mark.parametrize("over", [{"twin.fusion.mode": "mahony"}, {"twin.fusion.mode": "analytic"},
                                  {"twin.wire.mode": "json", "twin.wire.json_decimals": 2},
                                  {"twin.imu.oversample": 4}])
def test_session_runs_in_every_mode(over):
    from mesqsim.config import load_config
    from mesqsim.paths import REPO_ROOT
    from mesqsim.twin.config import load_twin_config
    from mesqsim.twin.session import simulate_twin_session

    joined, model, label = sc._program("random", sc.SEED, sc.MINUTES)
    s = simulate_twin_session(load_config(REPO_ROOT / "configs" / "default.yaml"), load_twin_config(overrides=over),
                              11, joined, model, label)  # fmt: skip
    t = s.labels_ticks
    assert s.frames["pod_id"].nunique() == N_PODS and len(s.bvh) > 0
    assert np.nanmedian(np.rad2deg(t["tilt_rad"])) < 3.0
    if over.get("twin.wire.mode") == "json":
        # the old wire: 2 decimals, so the frames hold the dongle's rounded numbers, and the send grid is 32 ms
        q = s.frames[["qx", "qy", "qz", "qw"]].to_numpy() / 32767.0
        assert np.all(np.abs(q * 100 - np.round(q * 100)) < 0.01)
        sent = t[t["pod_id"] == 0]["t_send_s"].to_numpy()
        assert abs(np.median(np.diff(sent)) - 0.032) < 2e-4


def test_same_seed_same_bytes():
    r = sc.determinism_numbers()
    assert r["identical"] and r["files"] == 9, r["differ"]


@pytest.mark.node
def test_twin_bvh_in_the_pages_own_loader():
    r = sc.bvh_numbers()
    if r is None:
        pytest.skip("needs Node and the mesquite-mocap clone")
    tw = r["twin"]
    assert tw["bones"] == 27 and tw["keyframes"] == r["frames"] and tw["finite"] and tw["comment_lines"] == 3
    if r["real"] is not None:  # the page's own file behaves the same way
        for k in ("constructed", "construct_error", "parsed_as_is", "parse_error", "bones", "names", "comment_lines"):
            assert tw[k] == r["real"][k], k


def test_wrap_of_delta_signs_documented():
    """The signs §4.12 asks to write out: ψ_i -> ψ_i - γ moves δ_pc = ψ_c - ψ_p by -γ when i is the child."""
    psi_p, psi_c, g = 0.3, -1.2, 0.5
    assert np.isclose(wrap_rad((psi_c - g) - psi_p), wrap_rad(psi_c - psi_p) - g)
    assert np.isclose(wrap_rad(psi_c - (psi_p - g)), wrap_rad(psi_c - psi_p) + g)


@pytest.mark.amass
def test_the_segment_ablation_puts_every_pod_rigid_on_its_segment():
    """twin.body.pod_frame: segment (§8.1's ablation): the pods' true orientation against their segment is constant
    (it moves with the skin in the twin), and the session's reports differ from the surface twin's."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    from mesqsim.twin import session_checks as sc
    from mesqsim.twin.bodymotion import body_tracks, on_segments

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    joined, model, _ = sc._program("random", sc.SEED, sc.MINUTES)
    tr = body_tracks(joined.seq, model)
    seg = on_segments(tr)
    k = tr.names.index("LeftLeg")
    rel_seg = Rotation.from_quat(seg.q_segment[:, k]).inv() * Rotation.from_quat(seg.q[:, k])
    rel_skin = Rotation.from_quat(tr.q_segment[:, k]).inv() * Rotation.from_quat(tr.q[:, k])
    assert np.degrees((rel_seg * rel_seg[0].inv()).magnitude()).max() < 1e-6
    assert np.degrees((rel_skin * rel_skin[0].inv()).magnitude()).max() > 1.0  # the skin does move against the bone
