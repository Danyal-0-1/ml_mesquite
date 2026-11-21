"""The calibration's machinery (Step 3 brief §5.4-5.7): the fit on a toy problem with a known answer, the predictive
coverage, the classifier two-sample test and gate G-T; then, with AMASS and SMPL+H, Φ on twin sessions, the old suit
and the phone zeroed in the hand."""

import numpy as np
import pandas as pd
import pytest

from mesqsim.twin import calibration as cal


def test_rejection_abc_recovers_a_known_parameter():
    """One statistic = 1000 k + noise, the real value 15: the closest 10 % of 400 draws centre on k = 0.015."""
    rng = np.random.default_rng(0)
    key = "twin.sensors.gyro_scale_max"
    k = rng.uniform(0.0, 0.03, 400)
    thetas = pd.DataFrame({"draw": np.arange(400), key: k})
    sims = pd.DataFrame({"draw": np.arange(400), "rec": "nov14", "stat": "phi6",
                         "value": 1000.0 * k + rng.normal(0.0, 0.5, 400)})  # fmt: skip
    dist = cal.distances(sims, {"nov14": {"phi6": 15.0}})
    post, rep = cal.posterior(thetas, dist, keep=0.1)
    assert len(post) == 40
    r = rep[key]
    assert abs(r["median"] - 0.015) < 0.001 and r["p5"] < 0.015 < r["p95"] and r["width_share"] < 0.2


def test_axial_statistics_wrap_in_the_distance_and_the_coverage():
    sims = pd.DataFrame({"draw": np.arange(50), "rec": "nov13_1341", "stat": "phi3_LeftLeg",
                         "value": np.r_[np.full(25, 89.0), np.full(25, -89.0)] + np.linspace(-0.5, 0.5, 50)})
    d = cal.distances(sims, {"nov13_1341": {"phi3_LeftLeg": -90.0}})
    assert d["distance"].max() < 5.0  # 89° and -89° are 1° from -90° on an axis
    cov = cal.coverage(sims, {"nov13_1341": {"phi3_LeftLeg": 90.0}})
    assert bool(cov["inside"].iloc[0])


def test_classifier_two_sample_test_known_answers():
    rng = np.random.default_rng(1)
    real = {r: rng.normal(size=(60, 6)) for r in ("a", "b", "c", "d")}
    same = cal.classifier_test(real, {r: rng.normal(size=(60, 6)) for r in real}, boot=300)
    far = cal.classifier_test(real, {r: rng.normal(size=(60, 6)) + 1.5 for r in real}, boot=300)
    assert same["p5"] < 0.5 < same["p95"] + 0.05 and abs(same["auc"] - 0.5) < 0.1
    assert far["auc"] > 0.95 and far["p5"] > 0.9
    assert cal.auc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([0, 0, 1, 1])) == 1.0


def test_gate_gt():
    cov = pd.DataFrame({"inside": [True, True, False]})
    assert cal.gate_gt(cov, pd.DataFrame({"inside": [True, False, False]}), {"auc": 0.7}, {"auc": 0.8})["passes"]
    assert not cal.gate_gt(cov, cov, {"auc": 0.81}, {"auc": 0.8})["passes"]


def test_priors_and_their_split_by_suit():
    rng = np.random.default_rng(2)
    th = cal.draw_theta(rng)
    tw13, s1 = cal.split_theta(th, "suit_2026_11_13")
    tw14, _ = cal.split_theta(th, "suit_2026_11_14")
    tw04, _ = cal.split_theta(th, "suit_2026_11_04")
    assert "twin.wire.json_decimals" in tw13 and "twin.wire.json_decimals" not in tw14
    assert tw04["twin.wire.json_decimals"] == th["twin.wire.json_decimals@suit_2026_11_04"]  # Nov 4's own precision
    assert tw13["twin.wire.json_decimals"] == th["twin.wire.json_decimals"] and not any("@" in k for k in tw04)
    assert "twin.procedure.p_camera_mismatch" in tw14 and "twin.procedure.p_camera_mismatch" not in tw13
    assert set(s1) == {"error_model.box_slot_deg"} and 0.5 <= s1["error_model.box_slot_deg"] <= 30.0
    assert cal.net_turn_rate_deg_s(np.c_[np.arange(100.0), np.zeros(100)], 0.1, np.ones(100, bool)) == 0.0


# ------------------------------------------------------------------------------------------- with SMPL+H
@pytest.fixture(scope="module")
def program():
    from mesqsim.twin import session_checks as sc

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    return sc._program("random", sc.SEED, sc.MINUTES)


def _session(program, seed, over):
    from mesqsim.config import load_config
    from mesqsim.paths import REPO_ROOT
    from mesqsim.twin.config import load_twin_config
    from mesqsim.twin.session import simulate_twin_session

    joined, model, label = program
    return simulate_twin_session(load_config(REPO_ROOT / "configs" / "default.yaml"),
                                 load_twin_config(overrides=over), seed, joined, model, label)  # fmt: skip


@pytest.mark.amass
def test_twin_statistics_by_the_recordings_code_path(program):
    from mesqsim.twin.twin_stats import analyse_twin

    s = _session(program, 1, {"twin.procedure.p_camera_mismatch": 0.0, "twin.procedure.p_zero_at_start": 0.0})
    r = analyse_twin(s, "optitrack")
    assert 0.40 < r["phi5"]["scale"] < 0.55  # the page walks half the distance (x* = (100 p + c) / 2)
    assert 90.0 < r["phi5"]["pelvis_height_cm"] < 110.0 and abs(r["phi7_deg"]) < 15.0
    assert r["phi10"]["source"] == "measured" and r["phi9"]["leg_jumps"] == 0
    assert set(r["phi3_truth_deg"]) == {"LeftLeg", "RightLeg", "LeftUpLeg", "RightUpLeg"}


@pytest.mark.amass
def test_old_suit_wears_eleven_pods_and_the_old_recorder(program):
    from mesqsim.twin.suits import OLD_SUIT_PODS, suit_overrides
    from mesqsim.twin.twin_stats import analyse_twin

    s = _session(program, 2, suit_overrides("suit_2026_11_13"))
    assert sorted(s.frames["pod_id"].unique().tolist()) == OLD_SUIT_PODS
    assert "Frame Time: 0.03333333333333333" in s.bvh and ";" not in s.bvh
    r = analyse_twin(s, "free")
    assert r["phi10"]["source"] == "asserted" and len(r["pod_bones"]) == 11


@pytest.mark.amass
def test_a_zero_in_the_hand_lifts_the_root_and_turns_the_travel(program):
    """§2.4 item 3: zeroed in the hand at Start, h below the belt and turned by a, the phone then reports from the
    belt y = H + h, so the root stands at (100 (1 + h) + 100) / 2 cm, and the travel runs about a off the facing."""
    from mesqsim.twin.twin_stats import analyse_twin

    quiet = {"twin.procedure.p_camera_mismatch": 0.0, "twin.procedure.p_tree_mismatch": 0.0,
             "twin.procedure.p_limb_restart": 0.0}  # fmt: skip
    ok = analyse_twin(_session(program, 1, quiet | {"twin.procedure.p_zero_at_start": 0.0}), "free")
    s = _session(program, 1, quiet | {"twin.procedure.p_zero_at_start": 1.0})
    slip = analyse_twin(s, "free")
    h = s.plan.phone["hand"]
    assert len(s.meta["zeros"]) == 1 and not s.meta["zeros"][0]["requested"]
    want = (100.0 * (1.0 - h["height_m"]) + 100.0) / 2.0
    assert abs(slip["root_height_cm"] - want) < 5.0 and abs(ok["root_height_cm"] - 100.0) < 5.0
    assert abs(((slip["phi7_deg"] - ok["phi7_deg"]) + np.rad2deg(h["yaw_rad"]) + 180) % 360 - 180) < 15.0


def test_checkpointed_runs_resume_after_the_last_batch(tmp_path, monkeypatch):
    import pickle

    calls = []

    def fake(part, programs, workers):
        calls.append(len(part))
        return [{"rec": t[0], "seed": t[2]} for t in part]

    monkeypatch.setattr(cal, "run_many", fake)
    tasks = [("nov04", {}, s, {}, False, False, s) for s in range(7)]
    path = tmp_path / "fit_partial.pkl"
    out = cal.run_checkpointed(tasks, {}, 1, path, log=lambda *_: None, batch=3)
    assert [r["seed"] for r in out] == list(range(7)) and calls == [3, 3, 1]
    saved = pickle.loads(path.read_bytes())  # as if stopped after the first batch
    saved["results"] = saved["results"][:3]
    path.write_bytes(pickle.dumps(saved))
    calls.clear()
    assert cal.run_checkpointed(tasks, {}, 1, path, log=lambda *_: None, batch=3) == out and calls == [3, 1]
    calls.clear()  # other tasks: the checkpoint is not theirs
    cal.run_checkpointed(tasks[:5], {}, 1, path, log=lambda *_: None, batch=3)
    assert calls == [3, 2]
