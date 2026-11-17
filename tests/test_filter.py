"""The von Mises filter, replay, reboots and the hold rule (Step 2 brief §5.7, §8 item 7)."""

import numpy as np
import pytest

from mesqsim import evaluate as ev
from mesqsim import step0
from mesqsim.body import JOINTS
from mesqsim.config import load_config
from mesqsim.conventions import vm_mean_resultant, vm_product, wrap_rad
from mesqsim.estimators.contract import GeoTruth
from mesqsim.estimators.geo import GeoEstimator, GeoParams, replay
from mesqsim.estimators.vmfilter import JointFilter
from mesqsim.paths import REPO_ROOT
from mesqsim.sim.motion import MotionSource
from mesqsim.sim.simulate import simulate_session

CFG = load_config(REPO_ROOT / "configs" / "default.yaml")


def test_without_updates_the_mean_stays_and_kappa_follows_the_exact_rule():
    f = JointFilter(3, q_rad2_per_s=4e-6)
    f.init(10.0, np.array([0.3, -2.0, 3.1]), np.array([5.0, 80.0, 2000.0]))
    t = np.array([10.0, 11.0, 70.0, 610.0])
    mu, kap, defined = f.query(t)
    assert defined.all()
    np.testing.assert_allclose(mu, np.broadcast_to([0.3, -2.0, 3.1], mu.shape), atol=1e-15)  # wrap: 4e-16
    expect = vm_mean_resultant(np.array([5.0, 80.0, 2000.0]))[None, :] * np.exp(-4e-6 * (t - 10.0) / 2)[:, None]
    np.testing.assert_allclose(vm_mean_resultant(kap), expect, rtol=1e-10)


def test_large_kappa_prediction_matches_the_variance_sum():
    f = JointFilter(1, q_rad2_per_s=1e-5)
    f.init(0.0, np.zeros(1), np.array([5e4]))
    for dt in (1.0, 60.0, 600.0):
        _, kap, _ = f.query(np.array([dt]))
        assert kap[0, 0] == pytest.approx(1.0 / (1.0 / 5e4 + 1e-5 * dt), rel=0.01)


def test_update_is_the_product_of_the_two_densities():
    f = JointFilter(2, q_rad2_per_s=0.0)
    f.init(0.0, np.array([0.1, 3.0]), np.array([10.0, 0.0]))
    f.update(1.0, np.array([0.5, -2.0]), np.array([30.0, 7.0]))
    mu, kap = vm_product(np.array([0.1, 3.0]), np.array([10.0, 0.0]), np.array([0.5, -2.0]), np.array([30.0, 7.0]))
    np.testing.assert_allclose(f.mu, mu)
    np.testing.assert_allclose(f.kappa, kap)
    assert f.mu[1] == pytest.approx(-2.0) and f.kappa[1] == pytest.approx(7.0)  # a kappa-0 prior adds nothing


def test_reset_drops_kappa_to_zero_and_events_must_be_ordered():
    f = JointFilter(3, q_rad2_per_s=1e-6)
    f.init(0.0, np.zeros(3), np.full(3, 100.0))
    f.reset(5.0, [1])
    _, kap, _ = f.query(np.array([5.0, 6.0]))
    assert kap[0, 1] == 0.0 and kap[1, 1] == 0.0 and kap[0, 0] > 99.0
    with pytest.raises(ValueError):
        f.update(4.0, np.zeros(3), np.ones(3))


def test_query_before_the_first_event_is_undefined():
    f = JointFilter(2, q_rad2_per_s=1e-6)
    f.init(3.0, np.ones(2), np.ones(2))
    _, kap, defined = f.query(np.array([1.0, 3.0]))
    assert defined.tolist() == [False, True] and kap[0].tolist() == [0.0, 0.0]


@pytest.fixture(scope="module")
def stress_session():
    """The stress preset with a 10-minute idle: seed 5 has three reboots in check 7.4."""
    cfg = CFG.with_preset("stress").with_overrides({"session.idle_s": 600.0})
    d = simulate_session(cfg, 5, MotionSource.from_spec("synthetic:walk:30"))
    series = step0.decode(d.frames, d.session)
    host = ev.Grid(d.labels_joints).t_s + d.meta["plan"]["host_epoch_s"]
    return d, series, host


@pytest.mark.parametrize("variant", ["ID", "K", "nofilter"])
def test_replay_reproduces_the_estimator_exactly(stress_session, variant):
    d, series, host = stress_session
    truth = GeoTruth.from_meta(d.meta) if variant == "K" else None
    geo = GeoEstimator(CFG, truth, use_filter=variant != "nofilter")
    est, diag = geo.run(series, d.session, host, diagnostics=True)
    p = GeoParams(CFG)
    mode = "K" if variant == "K" else "ID"
    mu, kap, defined = replay(diag.events, host, float(CFG["geometry.drift_q_rad2_per_s"]), p.scales(mode),
                              p.model_sigmas_rad(mode), p.weight(variant != "nofilter"), variant != "nofilter")
    after = defined & (host >= diag.events.t_init)
    assert np.array_equal(wrap_rad(mu[after]), est.delta_rad[after]) and np.array_equal(kap[after], est.kappa[after])


def test_a_reboot_resets_the_joints_of_that_pod(stress_session):
    d, series, host = stress_session
    est, diag = GeoEstimator(CFG).run(series, d.session, host, diagnostics=True)
    resets = [(t, joints) for t, kind, joints in diag.events.events if kind == 1]
    assert resets, "seed 5 under stress must reboot at least one pod"
    for t, joints in resets:
        k = int(np.searchsorted(host, t, side="left"))  # the first query at or after the host learns of the reboot
        nxt = [tw for tw, kind, _ in diag.events.events if kind == 2 and tw >= t]
        if k < host.size and (not nxt or host[k] < nxt[0]):
            assert np.all(est.kappa[k, joints] == 0.0)
            assert np.all(est.uncertain[k, joints])


def test_geo_id_equals_b1_on_noise_free_motion_without_information():
    """Standing still, every error off except offsets: nothing to identify, no facing; GEO-ID holds B1 exactly."""
    cfg = CFG.with_preset("zero").with_overrides({"error_model.offset_mode": "arbitrary"})
    d = simulate_session(cfg, 4, MotionSource.from_spec("synthetic:stand:20"))
    series = step0.decode(d.frames, d.session)
    grid = ev.Grid(d.labels_joints)
    host = grid.t_s + d.meta["plan"]["host_epoch_s"]
    est = GeoEstimator(CFG).estimate(series, d.session, host)
    b1 = ev.estimate("B1", grid, series, d.session)
    mot = grid.phase == "motion"
    err_geo = np.abs(wrap_rad(est.delta_rad[mot] - grid.delta_rad[mot]))
    err_b1 = np.abs(wrap_rad(b1[mot] - grid.delta_rad[mot]))
    assert np.all(err_geo <= err_b1 + 1e-12)
    assert len(JOINTS) == err_geo.shape[1]
