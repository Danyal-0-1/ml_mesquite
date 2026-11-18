"""The page golden comparison (Step 3 brief §4.10): replay a fixture's events through the page twin and measure how
far it lands from what the real page code wrote (``tools/page_golden/run.mjs``). Used by
``tests/test_page_twin.py`` and validation check 40.1."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..paths import REPO_ROOT
from .page import PageTwin, Rig

__all__ = ["FIXTURES", "QUAT_TOL", "BVH_TOL", "cases", "load_case", "rig", "replay", "compare"]

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "page_golden"
QUAT_TOL = 1e-5  # every bone quaternion (brief §4.10)
BVH_TOL = 1e-4  # every BVH number
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def cases(root: Path = FIXTURES) -> list[str]:
    return sorted(p.stem for p in root.glob("*_*.json") if p.stem != "rig_scene2")


def load_case(name: str, root: Path = FIXTURES) -> dict:
    return json.loads((root / f"{name}.json").read_text(encoding="utf-8"))


def rig(root: Path = FIXTURES) -> Rig:
    return Rig.from_json(json.loads((root / "rig_scene2.json").read_text(encoding="utf-8")))


def replay(case: dict, rig_: Rig, mode: str = "as_is"):
    """Feed a fixture's events to the page twin; return (page, per-event records shaped like run.mjs's, BVH)."""
    prov = case["provenance"]
    page = PageTwin(rig_, prov["tree"], prov["slerp"], mode=mode)
    per, bvh = [], None
    for ev in case["input"]["events"]:
        t = ev["t_ms"]
        if ev["type"] == "msg":
            page.handle(dict(ev["msg"]), t)
            rec = list(page.bone_quaternion(ev["msg"]["bone"].replace("Alt", "", 1)))
            if ev["msg"]["bone"] == "Hips":
                rec += list(page.hips_position())
            per.append(rec)
        else:
            if ev["type"] == "calibrate":
                page.calibrate()
            elif ev["type"] == "record_start":
                page.record_start(t)
            elif ev["type"] == "record_tick":
                page.record_tick(t)
            elif ev["type"] == "record_stop":
                bvh = page.record_stop(t)
            per.append(None)
    return page, per, bvh


def compare(case: dict, rig_: Rig) -> dict:
    """The largest differences between the page twin and the page on one fixture."""
    page, per, bvh = replay(case, rig_)
    g = case["golden"]
    d_msg = max((max(abs(a - b) for a, b in zip(x, y, strict=True))
                 for x, y in zip(per, g["per_event"], strict=True) if x is not None), default=0.0)
    d_final = max(max(abs(a - b) for a, b in zip(page.bone_quaternion(n), q, strict=True))
                  for n, q in g["final_state"].items())
    d_hips = max(abs(a - b) for a, b in zip(page.hips_position(), g["final_hips_position"], strict=True))
    head_py = bvh.split("\n; MESQ_FRAME_TIME_SOURCE")[0]
    head_js = g["bvh"].split("\n; MESQ_FRAME_TIME_SOURCE")[0]
    na = np.array([float(v) for v in _NUMBER.findall(bvh)])
    nb = np.array([float(v) for v in _NUMBER.findall(g["bvh"])])
    return {"messages": sum(x is not None for x in per), "quat": max(d_msg, d_final), "hips": d_hips,
            "bvh_numbers": float(np.max(np.abs(na - nb))) if na.size == nb.size else float("inf"),
            "bvh_count_equal": na.size == nb.size, "bvh_header_equal": head_py == head_js,
            "throws_node": len(g["threw"])}
