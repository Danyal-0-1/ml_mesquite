"""The page-level test (Step 3 brief §8.3): the page drawn again with pod readings turned by an estimate, Nov 14's
slips forced."""

import numpy as np
import pytest


@pytest.fixture(scope="module")
def short():
    from mesqsim.twin import session_checks as sc

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    return sc.short_session()


@pytest.mark.amass
def test_drive_page_redraws_the_session_exactly_and_a_correction_moves_it(short):
    from mesqsim.learn.through_page import corrected_bvh
    from mesqsim.twin.config import load_twin_config

    cfgt = load_twin_config()
    t = short.labels_steps["t_s"].to_numpy()
    assert corrected_bvh(short, cfgt, "as_is", None, t) == short.bvh
    assert corrected_bvh(short, cfgt, "as_is", np.zeros((t.size, 15)), t) == short.bvh  # a zero turn is no turn
    assert corrected_bvh(short, cfgt, "as_is", np.full((t.size, 15), 0.3), t) != short.bvh


@pytest.mark.amass
def test_the_forced_slips_are_nov14s():
    from mesqsim.learn.checks import tiny_dataset
    from mesqsim.learn.through_page import NOV14, page_rows

    rows = page_rows(tiny_dataset("a"), "test", 1)
    assert len(rows) == 1
    row, cond = rows[0]
    assert cond["suit"] == "suit_2026_11_14" and cond["hipsalt"] and not cond["protocol"]
    assert all(cond["twin"][k] == v for k, v in NOV14.items()) and cond["twin"]["twin.suit.pods"] == list(range(15))
    from mesqsim.config import load_config
    from mesqsim.learn.store import build_session
    from mesqsim.paths import REPO_ROOT
    from mesqsim.twin.config import load_twin_config

    built = build_session(row, cond, load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml"),
                          row["length_min"], keep_session=True, page=True)  # fmt: skip
    slips = built["session"].plan.slips
    assert slips["limb_restart"]["what"] == "left_leg" and slips["limb_restart"]["pods"] == [9, 10]
    assert slips["zero_at_start"] and built["session"].bvh
