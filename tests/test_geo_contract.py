"""The estimator contract (Step 2 brief §5.2, §8 item 1): causal, deterministic, label-blind, own beliefs only."""

import shutil

import numpy as np
import pytest

from mesqsim import evaluate as ev
from mesqsim import io, step0
from mesqsim.config import load_config
from mesqsim.estimators.contract import GeoTruth
from mesqsim.estimators.geo import GeoEstimator, GeoParams, estimate_geo_id, estimate_geo_k
from mesqsim.geo_score import pair_timing_errors_ms, truncate_frames
from mesqsim.paths import REPO_ROOT
from mesqsim.sim.motion import MotionSource
from mesqsim.sim.simulate import simulate_session

CFG = load_config(REPO_ROOT / "configs" / "default.yaml")


@pytest.fixture(scope="module")
def session_walk():
    d = simulate_session(CFG, 11, MotionSource.from_spec("synthetic:walk:12"))
    series = step0.decode(d.frames, d.session)
    host = ev.Grid(d.labels_joints).t_s + d.meta["plan"]["host_epoch_s"]
    return d, series, host


def _run(cfg, frames, session, host, truth=None):
    series = step0.decode(frames, session)
    return GeoEstimator(cfg, truth).estimate(series, session, host)


@pytest.mark.parametrize("variant", ["ID", "K"])
def test_causal_deleting_later_frames_changes_nothing_before(session_walk, variant):
    d, _, host = session_walk
    truth = GeoTruth.from_meta(d.meta) if variant == "K" else None
    full = _run(CFG, d.frames, d.session, host, truth)
    ph = {p["name"]: (p["t_host_start_s"], p["t_host_end_s"]) for p in d.session["phases"]}
    cuts = [ph["box"][1] - 1.0, ph["tpose"][0] + 1.5, ph["motion"][0] + 3.3, ph["motion"][0] + 7.0]
    for t_cut in cuts:
        cut = _run(CFG, truncate_frames(d.frames, t_cut), d.session, host, truth)
        before = host <= t_cut
        assert before.any()
        for a, b in zip(full, cut, strict=True):
            assert np.array_equal(a[before], b[before]), f"output before {t_cut} changed after deleting later frames"


def test_deterministic_identical_bytes(session_walk):
    d, series, host = session_walk
    a = GeoEstimator(CFG).estimate(series, d.session, host)
    b = GeoEstimator(CFG).estimate(series.copy(), dict(d.session), host.copy())
    assert all(x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def test_label_blind_identical_without_labels(session_walk, tmp_path):
    d, series, host = session_walk
    folder = io.write_session(tmp_path / "s", d, d.meta)
    with_labels = estimate_geo_id(step0.decode(io.read_session(folder)["frames"], d.session), d.session, host, CFG)
    for name in ("labels_ticks.parquet", "labels_joints.parquet", "meta.json"):
        (folder / name).unlink()
    blind = io.read_session(folder, labels=False)
    assert set(blind) == {"frames", "session"}
    without = estimate_geo_id(step0.decode(blind["frames"], blind["session"]), blind["session"], host, CFG)
    assert all(x.tobytes() == y.tobytes() for x, y in zip(with_labels, without, strict=True))
    shutil.rmtree(folder)


def test_the_estimator_sees_only_its_own_beliefs():
    p = GeoParams(CFG)
    assert p["window_s"] == CFG["geometry.window_s"]
    with pytest.raises(KeyError):
        p.cfg["error_model.drift_sigma_deg_per_min"]  # the simulator's truth is not reachable
    section = CFG.section("geometry")
    assert set(section.tree) == {"geometry"}
    GeoParams(section)  # the section alone is enough


def test_geo_k_needs_truth_and_geo_id_refuses_it(session_walk):
    d, series, host = session_walk
    with pytest.raises(ValueError):
        estimate_geo_k(series, d.session, host, CFG, truth=None)
    with pytest.raises(ValueError):
        estimate_geo_id(series, d.session, host, CFG, truth=GeoTruth.from_meta(d.meta))


def test_output_shapes_ranges_and_hold_flag(session_walk):
    d, series, host = session_walk
    est = GeoEstimator(CFG).estimate(series, d.session, host)
    assert est.delta_rad.shape == est.kappa.shape == est.uncertain.shape == (host.size, 14)
    assert np.all((est.delta_rad > -np.pi) & (est.delta_rad <= np.pi))
    assert np.all(est.kappa >= 0) and est.uncertain.dtype == bool
    assert np.array_equal(est.uncertain, est.kappa < CFG["geometry.kappa_hold"])


def test_scorer_columns_kappa_and_uncertain(session_walk):
    d, series, _ = session_walk
    data = {"frames": d.frames, "session": d.session, "labels_ticks": d.labels_ticks,
            "labels_joints": d.labels_joints, "meta": d.meta}  # fmt: skip
    e = ev.evaluate(data, series, ("B1", "GEO-ID"), CFG)
    b1, geo = e[e["baseline"] == "B1"], e[e["baseline"] == "GEO-ID"]
    assert b1["kappa"].isna().all() and b1["uncertain"].isna().all()
    assert geo["kappa"].notna().all() and geo["uncertain"].notna().all()


def test_pair_timing_within_the_decode_delay_bound(session_walk):
    d, series, host = session_walk
    _, diag = GeoEstimator(CFG).run(series, d.session, host, diagnostics=True)
    err = pair_timing_errors_ms(diag, series, d.labels_ticks)
    assert err.size > 1000
    assert np.max(np.abs(err)) < 15.0 and np.percentile(np.abs(err), 50) < 6.0
