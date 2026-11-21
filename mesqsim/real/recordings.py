"""The seven recordings of Danyal's suit (Step 3 brief §2.3), their pairing with OptiTrack, and the analysis of
2026-11-14 that S3-M1 must reproduce (Step 3 brief §5.3).

Paths are relative to ``$MESQ_DATA_DIR/real_suit/`` (``manifests/real_suit.csv``). A BVH file is named when it is
saved, so its name is the end of the take, in the laptop's local time. ``role`` is the split *proposed* in Step 3
brief §5.4 (calibrate on the long takes and Nov 14, check on the three best-tracked takes); Danyal freezes it in
``docs/analysis_plan_DRAFT.md``.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from ..paths import data_dir

__all__ = ["Recording", "RECORDINGS", "NOV13", "REFERENCE", "EXPLAINED_DIFFERENCES", "real_suit_dir", "available",
           "bvh_end_local"]

_N13 = "Test_Nove13_Optitrack_vs_M"
_TAKE = f"{_N13}/Optitrack/Take_{{}}_MESQUITE_benckmark_2026-11-13 01.23.50 PM{{}}.csv"


@dataclass(frozen=True)
class Recording:
    id: str
    bvh: str  # relative to real_suit/
    motive: str | None  # the OptiTrack take that overlaps it, if any
    kind: str  # "static" (Nov 4), "optitrack" (Nov 13) or "free" (Nov 14)
    suit_version: str  # the twin's suit version (Step 3 brief §5.4)
    role: str  # proposed split: "calibrate" or "check"


RECORDINGS = (
    Recording("nov04", "Test_Nov4_calibration_tests/MMcap_bvh_2026-11-4-15-29-14.bvh", None, "static",
              "suit_2026_11_04", "calibrate"),
    Recording("nov13_1341", f"{_N13}/Mesquite/MMcap_bvh_2026-11-13-13-41-17.bvh", _TAKE.format(0, ""), "optitrack",
              "suit_2026_11_13", "calibrate"),
    Recording("nov13_1342", f"{_N13}/Mesquite/MMcap_bvh_2026-11-13-13-42-39.bvh", _TAKE.format(1, "_2026"),
              "optitrack", "suit_2026_11_13", "check"),
    Recording("nov13_1344", f"{_N13}/Mesquite/MMcap_bvh_2026-11-13-13-44-44.bvh", _TAKE.format(1, "_2027"),
              "optitrack", "suit_2026_11_13", "check"),
    Recording("nov13_1347", f"{_N13}/Mesquite/MMcap_bvh_2026-11-13-13-47-28.bvh", _TAKE.format(2, "_2028"),
              "optitrack", "suit_2026_11_13", "check"),
    Recording("nov13_1359", f"{_N13}/Mesquite/MMcap_bvh_2026-11-13-13-59-24.bvh", _TAKE.format(2, "_2029"),
              "optitrack", "suit_2026_11_13", "calibrate"),
    Recording("nov14", "Test_Nov14_new_code/MMcap_bvh_2026-11-14-17-52-25.bvh", None, "free", "suit_2026_11_14",
              "calibrate"),
)
NOV13 = tuple(r.id for r in RECORDINGS if r.kind == "optitrack")


def real_suit_dir() -> Path:
    return data_dir() / "real_suit"


def available() -> bool:
    """True when every recording's files are present (tests marked ``real`` skip otherwise)."""
    root = real_suit_dir()
    return all((root / r.bvh).is_file() and (r.motive is None or (root / r.motive).is_file()) for r in RECORDINGS)


def bvh_end_local(bvh_path: str) -> dt.datetime:
    """The save time in a BVH file's name, ``MMcap_bvh_Y-M-D-h-m-s``: the end of the take, laptop local time."""
    m = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})-(\d{1,2})", Path(bvh_path).name)
    if not m:
        raise ValueError(f"no save time in {bvh_path!r}")
    return dt.datetime(*(int(g) for g in m.groups()))


# The analysis of 2026-11-14 and 15 (Step 3 brief §5.3): value per recording, and the tolerance. Nov 13 takes in file
# order. Left/right pairs are (left, right). Angles in degrees; "axial" values compare modulo 180°.
REFERENCE = {
    "phi1_walking_s": ({"nov13_1341": 172, "nov13_1342": 10, "nov13_1344": 17, "nov13_1347": 19,
                        "nov13_1359": 141, "nov14": 96}, ("rel", 0.15)),
    "phi3_shin_drawn_deg": ({"nov13_1341": (-30, 19), "nov13_1342": (-63, 43), "nov13_1344": (-58, 50),
                             "nov13_1347": (-40, 64), "nov13_1359": (-31, 17), "nov14": (73, 9)}, ("axial", 5.0)),
    "phi3_shin_truth_deg": ({"nov13_1341": (11, -1), "nov13_1342": (12, 14), "nov13_1344": (5, 3),
                             "nov13_1347": (1, -10), "nov13_1359": (2, -2)}, ("axial", 5.0)),
    "phi3_thigh_drawn_deg": ({"nov14": (78, 15)}, ("axial", 5.0)),
    "phi4_shin_first_last_deg": ({"nov13_1341": ((-31, -73), (3, 38))}, ("abs", 7.0)),  # (L at 0 s, at 240 s), R
    "phi5_scale": ({"nov13_1341": 0.43, "nov13_1342": 0.45, "nov13_1344": 0.50, "nov13_1347": 0.48,
                    "nov13_1359": 0.47}, ("abs", 0.03)),
    "phi5_heights_cm": ({"nov13_1341": (103, 66), "nov13_1342": (102, 77), "nov13_1344": (104, 82),
                         "nov13_1347": (103, 82), "nov13_1359": (105, 83)}, ("abs", 3.0)),  # (pelvis marker, root)
    "root_height_cm": ({"nov14": 116.5}, ("abs", 3.0)),  # 116-117 cm, one 15 s sample at 92 cm
    "phi6_speed_cm_s": ({"nov13_1341": 43.3, "nov13_1342": 38.8, "nov13_1344": 41.5, "nov13_1347": 31.8,
                         "nov13_1359": 41.7, "nov14": 18.1}, ("rel", 0.10)),
    "phi7_travel_minus_facing_deg": ({"nov13_1341": -24.9, "nov13_1342": -15.0, "nov13_1344": -10.6,
                                      "nov13_1347": -7.6, "nov13_1359": -6.6, "nov14": 73.7}, ("abs", 5.0)),
    "phi8_exact_repeats_pct": ({"nov13_1341": 11.8, "nov13_1342": 11.4, "nov13_1344": 2.6, "nov13_1347": 5.1,
                                "nov13_1359": 40.8, "nov14": 6.7, "nov04": 79.0}, ("abs", 3.0)),
    "phi9_leg_jumps": ({"nov14": 0}, ("exact", 0)),
    "phi10_frame_time": ({"nov13_1341": (1 / 30, "asserted"), "nov13_1342": (1 / 30, "asserted"),
                          "nov13_1344": (1 / 30, "asserted"), "nov13_1347": (1 / 30, "asserted"),
                          "nov13_1359": (1 / 30, "asserted"), "nov14": (0.033000, "measured")}, ("exact", 0)),
    "phi11_upper_arm_r": ({"nov13_1341": (-0.42, 0.64), "nov13_1342": (-0.49, 0.97), "nov13_1344": (-0.92, 0.99),
                           "nov13_1347": (-0.17, 0.49), "nov13_1359": (0.57, 0.35)}, ("abs", 0.10)),
    "phi12_pods_within_1p8": ({"nov04": (12, 13)}, ("exact", 0)),  # 12 of 13 pods within 1.8° for 17.3 min
}
"""Step 3 brief §5.3, transcribed. Tolerance kinds: ``rel`` (fraction), ``abs``, ``axial`` (degrees, modulo 180°),
``exact``."""

_ROUND = ("the left shin's swing is nearly round ({:.2f}, minor over major spread; the right shin's 0.38-0.66), so "
          "its major axis is ill-determined in these short takes (10-19 s of walking) and moves by more than 5° "
          "with the frames counted as walking; the left shin of the same two takes differs in the drawing and in "
          "the truth alike").format
_ARMS = ("arm markers are the hardest to follow (occlusion, the arms cross the torso); r over a take depends on "
         "which frames each analysis names an arm; definitions tried: heights, heights relative to the root, "
         "height changes")
EXPLAINED_DIFFERENCES = {
    ("phi3_shin_drawn_deg", "nov13_1342"): _ROUND(0.85),
    ("phi3_shin_drawn_deg", "nov13_1347"): _ROUND(0.84),
    ("phi3_shin_truth_deg", "nov13_1342"): _ROUND(0.70),
    ("phi3_shin_truth_deg", "nov13_1344"): "the left value is 5.3° from the reference (tolerance 5°); round swing "
                                           "(0.71)",
    ("phi3_shin_truth_deg", "nov13_1347"): _ROUND(0.87),
    ("phi7_travel_minus_facing_deg", "nov14"): (
        "the root crawled (18 cm/s, 8 % of walking frames below 5 cm/s), so travel minus facing spreads over the "
        "whole circle (median +64°, circular mean +83°) and its median moves with the frames counted as walking "
        "(99 s here, 96 s in the reference)"),
    ("phi8_exact_repeats_pct", "nov13_1341"): (
        "definition: the brief's median over bones gives 15.5 %; the share pooled over bones without Hips gives "
        "10.5 % (reference 11.8 %)"),
    ("phi8_exact_repeats_pct", "nov13_1359"): "definition: median over bones 47.8 %, pooled 53-55 % (reference 40.8 %)",
    ("phi8_exact_repeats_pct", "nov04"): (
        "definition: median over bones 85.6 %; pooled over bones 79.5 % and the mean over bones 79.3 % match the "
        "reference 79 %"),
    **{("phi11_upper_arm_r", rid): _ARMS for rid in NOV13},
}
"""The reference values S3-M1 does not reproduce within tolerance, each with its reason (WORKLOG.md, S3-M1). The
test ``tests/test_real_recordings.py`` fails if any other value stops reproducing."""
