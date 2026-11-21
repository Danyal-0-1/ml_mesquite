"""The numbers to beat (Step 3 brief §8; check 53.1): the mergeable error summaries, one session's methods, and the
results document reproduced by the same command."""

import numpy as np
import pytest

from mesqsim import results_step3 as r3


def test_agg_quantiles_are_exact_to_the_bin_and_merge():
    rng = np.random.default_rng(0)
    e = np.abs(rng.normal(0.0, 8.0, 100_000))
    a, b = r3.Agg(), r3.Agg()
    a.add(e[:50_000])
    b.add(e[50_000:])
    a.merge(b)
    s = a.stats()
    assert s["n"] == e.size and abs(s["mean_deg"] - e.mean()) < 1e-9
    assert abs(s["median_deg"] - np.median(e)) <= 0.02 and abs(s["p95_deg"] - np.percentile(e, 95)) <= 0.02
    big = r3.Agg()
    big.add([179.9, 180.0, 45.2])
    assert big.quantile(1.0) == 180.0 and big.quantile(0.5) == 179.75 and big.quantile(0.2) == 45.25  # bin middles


@pytest.fixture(scope="module")
def tiny():
    from mesqsim.learn import checks

    if not checks.available():
        pytest.skip("needs AMASS and the SMPL+H model")
    return checks.tiny_dataset("a")


@pytest.mark.amass
def test_one_session_every_method(tiny):
    rows = r3.test_rows(tiny, "test")
    res = r3.session_result((rows[0][0], rows[0][1], None))
    pooled = r3.merge([res])[None]
    tb = r3.tables(pooled)
    allrows = tb[(tb["stratum"] == "all") & (tb["unit"] == "all")].set_index(["target", "method"])
    assert set(allrows.index) == {("delta", m) for m in r3.DELTA_METHODS} | {("eps", m) for m in r3.EPS_METHODS}
    assert allrows.loc[("delta", "O2"), "median_deg"] <= 0.02 and allrows.loc[("eps", "O2"), "median_deg"] <= 0.02
    n68, n95, n = pooled["coverage"][("delta", "GEO-ID")]
    assert 0 <= n68 <= n95 <= n == allrows.loc[("delta", "GEO-ID"), "n"]
    types = tb[(tb["target"] == "delta") & tb["unit"].str.startswith("type:") & (tb["method"] == "B1")]
    assert sorted(types["unit"]) == ["type:2dof", "type:3dof", "type:hinge"]


@pytest.mark.amass
def test_the_results_document_is_reproduced(tiny):
    from mesqsim.learn.checks import results_repro_numbers

    r = results_repro_numbers()
    assert r["identical"] and r["lines"] > 20, r
