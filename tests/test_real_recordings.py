"""Danyal's recordings: the analysis of 2026-11-14 reproduced (Step 3 brief §5.3, check 49.1). Skips without them."""

import numpy as np
import pytest

from mesqsim.real.analysis import compare
from mesqsim.real.recordings import EXPLAINED_DIFFERENCES, NOV13

pytestmark = pytest.mark.real


def test_every_reference_value_is_reproduced_or_explained(real_results):
    rows = compare(real_results)
    assert len(rows) == 62
    failed = {(r["statistic"], r["recording"]) for r in rows if not r["ok"]}
    assert failed <= set(EXPLAINED_DIFFERENCES), sorted(failed - set(EXPLAINED_DIFFERENCES))


def test_leg_heights_identify_the_markers(real_results):
    """Step 3 brief §5.1: in the well-tracked takes the legs' height changes correlate with the drawing at 0.64 to
    0.95, and the correlation turns negative when left and right are swapped."""
    for rid in ("nov13_1341", "nov13_1342", "nov13_1344", "nov13_1347"):
        r = real_results[rid]
        legs = [r["phi11_change_r"][s] for s in ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")]
        assert np.mean(legs) >= 0.64, (rid, legs)
        assert np.mean(list(r["phi11_swapped_change_r"].values())) < 0.0, rid


def test_the_old_page_recorded_every_33_ms(real_results):
    """The page of 2026-11-13 wrote 1/30 s; the long takes align with OptiTrack at 33.0 ms (as measured on Nov 14)."""
    for rid in ("nov13_1341", "nov13_1359"):
        assert abs(real_results[rid]["alignment"]["frame_time_s"] - 0.0330) <= 0.00005, rid
    for rid in NOV13:
        assert real_results[rid]["phi10"]["source"] == "asserted"
        assert abs(real_results[rid]["alignment"]["clock_difference_s"]) < 1.5, rid


def test_static_hold(real_results):
    h = real_results["nov04"]["phi12"]
    assert (len(h["holding"]), len(h["steady"])) == (12, 13) and h["restarted"] == ["LeftHand"]
    assert set(h["steady"]) - set(h["holding"]) == {"Head"}
    assert 17.0 < h["window_min"] < 17.6
