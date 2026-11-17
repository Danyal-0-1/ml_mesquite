"""The Step 2 results machinery: replays equal the registered ablations; statistics (Step 2 brief §6)."""

import numpy as np
import pytest

from mesqsim import evaluate as ev
from mesqsim.config import load_config
from mesqsim.dataset import synthetic_sources
from mesqsim.geo_sessions import dataset_jobs, simulate_job
from mesqsim.paths import REPO_ROOT
from mesqsim.results_step2 import bootstrap_contrast, coverage, holm, selective_risk, session_worker

CFG = load_config(REPO_ROOT / "configs" / "default.yaml")


def test_replayed_ablations_equal_the_registered_estimators():
    job = dataset_jobs(CFG, synthetic_sources(), 1, splits=("test",), motion_start_min=[10], max_sequences=1)[0]
    res = session_worker(job, CFG, methods=("GEO-ID", "GEO-ID-nofilter", "GEO-ID-hinge"))
    d, series, host, grid = simulate_job(job, CFG)
    mot = grid.phase == "motion"
    for name in ("GEO-ID-nofilter", "GEO-ID-hinge"):
        dlt, kap, _ = ev.method_estimate(name, grid, series, d.session, d.meta, CFG)
        err = np.rad2deg(np.abs(np.angle(np.exp(1j * (dlt - grid.delta_rad)))))[mot].astype(np.float32)
        np.testing.assert_allclose(res[name]["err"], err, atol=1e-4)
        np.testing.assert_allclose(res[name]["kappa"], kap[mot].astype(np.float32), rtol=1e-6, atol=1e-6)


def test_bootstrap_and_holm():
    rng = np.random.default_rng(0)
    clear = bootstrap_contrast(rng.normal(-3.0, 1.0, 15))
    assert clear["hi"] < 0 and clear["p"] < 0.01
    null = bootstrap_contrast(rng.normal(0.0, 1.0, 15))
    assert null["lo"] < 0 < null["hi"]
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})


def test_coverage_and_selective_risk():
    rng = np.random.default_rng(1)
    kappa = np.full(20000, 50.0)
    err = np.rad2deg(np.abs(rng.vonmises(0.0, 50.0, 20000)))
    cov = coverage(err, kappa)
    assert cov[0.68] == pytest.approx(0.68, abs=0.015) and cov[0.95] == pytest.approx(0.95, abs=0.01)
    k2 = np.r_[np.full(1000, 1000.0), np.full(1000, 1.0)]
    e2 = np.r_[np.full(1000, 1.0), np.full(1000, 50.0)]
    rows = selective_risk(e2, k2, fractions=(1.0, 0.5))
    assert rows[1]["median"] == 1.0 and rows[0]["p95"] == 50.0
