"""The page twin (mesqsim/twin/page.py) against the real page code (Step 3 brief §4.10, check 40.1).

The golden outputs in tests/fixtures/page_golden/ come from tools/page_golden/run.mjs, which runs the page's
js/custom_icm.js and js/bvh_converter.js unmodified in Node; the Python test needs neither Node nor the clone.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mesqsim.paths import data_dir
from mesqsim.twin.golden import BVH_TOL, FIXTURES, QUAT_TOL, cases, compare, load_case, rig
from mesqsim.twin.page import PageTwin
from mesqsim.twin.three_math import js_number, q_normalize, q_slerp

CASES = cases()


@pytest.mark.parametrize("name", CASES)
def test_page_twin_matches_the_page(name):
    r = compare(load_case(name), rig())
    assert r["quat"] <= QUAT_TOL and r["hips"] <= QUAT_TOL, r  # every bone quaternion (brief: 1e-5)
    assert r["bvh_count_equal"] and r["bvh_numbers"] <= BVH_TOL, r  # every BVH number (brief: 1e-4)
    assert r["bvh_header_equal"], r  # hierarchy, offsets, frame count and time, provenance lines: identical text


def test_all_four_scenarios_in_both_trees():
    assert {c.rsplit("_", 1)[1] for c in CASES} == {"velvet", "smooth"} and len(CASES) == 8
    assert {c.rsplit("_", 1)[0] for c in CASES} == {"tpose_leg_swing", "turn_phone_moving", "hipsalt_silent",
                                                     "phone_zero_guard"}


def test_zero_guard_fixture_shows_the_page_skipping_zero_positions():
    """In the zero-guard scenario the real page left the hips where they were for every phone message whose
    position held an exact 0 (custom_icm.js:931), and moved them otherwise: the fixture's own record shows it."""
    for tree in ("velvet", "smooth"):
        case = load_case(f"phone_zero_guard_{tree}")
        pairs = [(e["msg"], r) for e, r in zip(case["input"]["events"], case["golden"]["per_event"], strict=True)
                 if e["type"] == "msg" and e["msg"]["bone"] == "Hips"]
        zero = [not (m["px"] and m["py"] and m["pz"]) for m, _ in pairs]
        assert 20 < sum(zero) < len(pairs) - 20, sum(zero)
        moved = [pairs[k][1][4:7] != pairs[k - 1][1][4:7] for k in range(1, len(pairs))]
        assert not any(m for m, z in zip(moved, zero[1:], strict=True) if z), tree  # a 0 never moves the hips
        assert sum(m for m, z in zip(moved, zero[1:], strict=True) if not z) > 20, tree


P_TPOSE = np.array([0.31, 1.05, -0.42])
C_CM = np.array([0.0, 100.0, 0.0])


def _static_phone(mode: str, n: int = 400):
    """The phone still at P_TPOSE for n messages, Set T-pose, then still 0.5 m further along x for n messages."""
    case = load_case("tpose_leg_swing_velvet")
    page = PageTwin(rig(), case["provenance"]["tree"], case["provenance"]["slerp"], mode=mode)
    p = P_TPOSE
    msg = {"bone": "Hips", "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0, "px": p[0], "py": p[1], "pz": p[2]}
    for k in range(n):
        page.handle(dict(msg), 33.3 * k)
    before = np.array(page.hips_position())
    page.calibrate()
    for k in range(n, 2 * n):
        page.handle(dict(msg, px=p[0] + 0.5), 33.3 * k)
    return before, np.array(page.hips_position())


def test_root_fixed_point_is_half_way():
    """as_is: x <- 0.8 x + 0.1 (100 p + c), so x* = (100 p + c) / 2 with c = (0, 100, 0) cm (brief §2.4, item 2)."""
    before, after = _static_phone("as_is")
    assert np.allclose(before, 100 * P_TPOSE + C_CM, atol=1e-9)  # before Set T-pose initialPosition is {0, 0, 0}
    assert np.allclose(after, (100 * (P_TPOSE + [0.5, 0, 0]) + C_CM) / 2, atol=1e-9)


def test_ideal_root_box_draws_one_metre_as_one_metre():
    """The counterfactual copies initialPosition (the hips at the T-pose): x* = 100 (p - p_T), so 0.5 m of phone
    travel is 50 cm of drawn travel; heights are then relative to the T-pose's."""
    _, after = _static_phone("ideal_root_box")
    assert np.allclose(after, [50.0, 0.0, 0.0], atol=1e-9)


def test_a_zero_coordinate_drops_the_phone_message():
    case = load_case("tpose_leg_swing_velvet")
    page = PageTwin(rig(), case["provenance"]["tree"], case["provenance"]["slerp"])
    start = page.hips_position()
    page.handle({"bone": "Hips", "x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0, "px": 0.0, "py": 1.0, "pz": -0.3}, 0.0)
    assert page.hips_position() == start  # obj.px && obj.py && obj.pz (custom_icm.js:931)


@pytest.mark.parametrize("x, s", [(5.648319542253949e-7, "5.648319542253949e-7"), (0.000025202082524802354,
                                  "0.000025202082524802354"), (100.0, "100"), (0.0, "0"), (-0.0, "0"),
                                  (-3.520553700092965, "-3.520553700092965"), (1e21, "1e+21"),
                                  (1.2345678901234568e20, "123456789012345680000"), (0.1, "0.1"),
                                  (1.5e-7, "1.5e-7"), (0.0000015, "0.0000015")])
def test_js_number(x, s):
    assert js_number(x) == s


def test_slerp_branches():
    a = q_normalize((0.1, -0.2, 0.3, 0.9))
    assert q_slerp(a, (0.5, 0.5, 0.5, 0.5), 0) == a
    assert q_slerp(a, (0.5, 0.5, 0.5, 0.5), 1) == (0.5, 0.5, 0.5, 0.5)
    e = (0.0, 1.0, 0.0, 0.0)  # a dot product of exactly 1: three.js keeps the old value (cosHalfTheta >= 1)
    assert q_slerp(e, e, 0.3) == e and q_slerp(e, (0.0, -1.0, 0.0, -0.0), 0.3) == e  # also from the other side
    # a normalised quaternion's dot with itself may be 0.9999999999999999: then it moves by an ulp, not stays
    assert max(abs(u - v) for u, v in zip(q_slerp(a, a, 0.3), a, strict=True)) < 1e-15


@pytest.mark.node
def test_fixtures_are_what_the_page_writes_today(tmp_path):
    """Rerun the real page code in Node (needs Node and the clone at 87d7c3b) and compare with the fixtures."""
    mocap = data_dir() / "mesquite-mocap"
    if shutil.which("node") is None or not (mocap / "js" / "custom_icm.js").is_file():
        pytest.skip("needs Node and the mesquite-mocap clone ($MESQ_DATA_DIR/mesquite-mocap)")
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools" / "page_golden" / "make_fixtures.py"), "--mocap", str(mocap),
                    "--out", str(tmp_path)], check=True, capture_output=True)
    for name in CASES + ["rig_scene2"]:
        new = json.loads((tmp_path / f"{name}.json").read_text(encoding="utf-8"))
        old = json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))
        if name != "rig_scene2":
            new["provenance"].pop("node", None)
            old["provenance"].pop("node", None)
        assert new == old, name
