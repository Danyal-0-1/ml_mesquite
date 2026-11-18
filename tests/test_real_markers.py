"""Marker naming and following (mesqsim.real.markers), on a synthetic take: 11 markers on a walker."""

import datetime as dt

import numpy as np
import pytest

from mesqsim.real.markers import MARKER_SITES, name_tpose, track_sites
from mesqsim.real.motive import MotiveTake

# body coordinates (forward, left, height) in metres of each marker; the T-pose, then walking with the arms down
TPOSE = {"Head": (0.08, 0.0, 1.72), "LeftArm": (0.0, 0.35, 1.50), "RightArm": (0.0, -0.35, 1.50),
         "LeftForeArm": (0.0, 0.58, 1.47), "RightForeArm": (0.0, -0.58, 1.47), "Spine": (0.12, 0.0, 1.285),
         "Hips": (-0.15, 0.0, 1.045), "LeftUpLeg": (0.07, 0.10, 0.63), "RightUpLeg": (0.07, -0.10, 0.63),
         "LeftLeg": (0.05, 0.10, 0.36), "RightLeg": (0.05, -0.10, 0.36)}
WALK = dict(TPOSE, LeftArm=(0.0, 0.22, 1.30), RightArm=(0.0, -0.22, 1.30), LeftForeArm=(0.02, 0.25, 1.05),
            RightForeArm=(0.02, -0.25, 1.05))
SWING = {"LeftUpLeg": 0.10, "RightUpLeg": -0.10, "LeftLeg": 0.15, "RightLeg": -0.15, "LeftArm": -0.05,
         "RightArm": 0.05, "LeftForeArm": -0.08, "RightForeArm": 0.08}


def _world(body, heading_rad, origin):
    fwd, left, z = body
    c, s = np.cos(heading_rad), np.sin(heading_rad)
    return np.array([origin[0] + fwd * c - left * s, origin[1] + fwd * s + left * c, z])


def test_tpose_naming_is_independent_of_the_heading():
    for heading in (-np.pi / 2, 0.4, 2.9):
        pts = np.stack([_world(TPOSE[s], heading, (1.0, -2.0)) for s in MARKER_SITES])
        order = np.random.default_rng(0).permutation(11)
        named = name_tpose(pts[order])
        assert {s: MARKER_SITES[order[k]] for s, k in named.items()} == {s: s for s in MARKER_SITES}


def _take():
    """A T-pose (1 s), then a walk along -Y at 1 m/s with legs and arms swinging at 1 Hz (6 s), at 120 Hz.

    The left shin's marker is lost for 0.1 s at 3 s and comes back as a new fragment; the right upper arm's is lost
    for 1.5 s at 4 s and comes back as a new fragment; a stray marker lies on the floor from 2.5 s to 3.5 s.
    """
    hz, n = 120.0, 840
    t = np.arange(n) / hz
    heading = -np.pi / 2  # facing -Y, so left is +X, as Danyal faced on 2026-11-13
    walking = t >= 1.0
    origin = np.stack([np.zeros(n), -np.where(walking, t - 1.0, 0.0)], axis=1)
    cols = {}
    for s in MARKER_SITES:
        xyz = np.empty((n, 3))
        for k in range(n):
            fwd, left, z = (WALK if walking[k] else TPOSE)[s]
            fwd += SWING.get(s, 0.0) * np.sin(2 * np.pi * (t[k] - 1.0)) * walking[k]
            xyz[k] = _world((fwd, left, z), heading, origin[k])
        cols[s] = xyz
    out = []
    for s in MARKER_SITES:
        if s == "LeftLeg":
            a, b = cols[s].copy(), cols[s].copy()
            a[t >= 3.0] = np.nan
            b[t < 3.1] = np.nan
            out += [a, b]
        elif s == "RightArm":
            a, b = cols[s].copy(), cols[s].copy()
            a[t >= 4.0] = np.nan
            b[t < 5.5] = np.nan
            out += [a, b]
        else:
            out.append(cols[s])
    stray = np.full((n, 3), np.nan)
    stray[(t >= 2.5) & (t < 3.5)] = [2.0, -2.0, 0.02]
    out.append(stray)
    names = []
    for s in MARKER_SITES:
        names += [f"{s} a", f"{s} b"] if s in ("LeftLeg", "RightArm") else [s]
    names.append("stray")
    return MotiveTake("synthetic", "synthetic", dt.datetime(2026, 11, 13, 13, 0, 0), hz, np.arange(n), t, names,
                      np.stack(out, axis=1), {})


@pytest.fixture(scope="module")
def tracked():
    take = _take()
    return take, track_sites(take)


def test_fragments_inherit_their_site(tracked):
    take, tr = tracked
    named = {s: [take.names[c] for c in cs] for s, cs in tr.fragments.items()}
    assert named["LeftLeg"] == ["LeftLeg a", "LeftLeg b"]  # by continuity, across 0.1 s
    assert named["RightArm"] == ["RightArm a", "RightArm b"]  # by template, after 1.5 s
    for s in MARKER_SITES:
        if s not in ("LeftLeg", "RightArm"):
            assert named[s] == [s]
    assert [take.names[c] for c in tr.unnamed] == ["stray"]


def test_sites_and_facing(tracked):
    take, tr = tracked
    assert tr.tpose_frame == 0 and np.isclose(tr.tpose_heights_m["Head"], 1.72)
    cov = tr.coverage()
    assert np.isclose(cov["LeftLeg"], 1 - 0.1 / 7.0, atol=0.01) and np.isclose(cov["Head"], 1.0)
    # facing: pelvis (back) -> head (front), averaged: -Y throughout
    ok = np.all(np.isfinite(tr.facing), axis=1)
    assert ok.all() and np.allclose(tr.facing, [0.0, -1.0], atol=1e-9)
