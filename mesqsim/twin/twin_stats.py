"""Φ1-Φ12 of a twin session, by the code path the recordings use (Step 3 brief §5.2: "one code path for real files and
twin files").

The drawn side is the twin BVH read by :func:`mesqsim.real.bvh.read_bvh` and :func:`mesqsim.real.stats.series_from_bvh`,
exactly as a recording's. The truth side is the twin's own markers (the pods' sites, :data:`.session.MARKER_SITES`)
on the session grid, with the facing from the pelvis marker to the head marker as for OptiTrack
(:func:`mesqsim.real.markers.facing_from_sites`). The two clocks are aligned as the recordings' are: the recorder's
tick times are known (frame k at record-on + k × 33 ms), and the remaining lag of the drawing behind the body is
fitted by the same correlation of the four leg sites' height changes (:func:`mesqsim.real.align.fine_lag_s`),
searched over ±1 s.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from ..real import stats as st
from ..real.align import drawn_height_changes, fine_lag_s, height_change, pearson
from ..real.analysis import CORR_SITES, LEG_BONES, SWING_SITES
from ..real.bvh import forward_kinematics, pod_bones, read_bvh
from ..real.markers import facing_from_sites, smooth_facing

__all__ = ["truth_series", "analyse_twin"]

LAG_SEARCH_S = 1.0


def _f(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def truth_series(truth: dict) -> st.Series:
    """The twin's markers as an OptiTrack-like series on the session grid."""
    t = np.asarray(truth["t_s"], dtype=np.float64)
    dt = float(t[1] - t[0])
    sites = {k: np.asarray(v, dtype=np.float64) for k, v in truth["sites"].items()}
    return st.Series(t, dt, sites, smooth_facing(facing_from_sites(sites), dt), None, "optitrack")


def analyse_twin(session, kind: str = "free") -> dict:
    """The statistics of :func:`mesqsim.real.analysis.analyse` for a twin session. ``kind`` as a recording's:
    ``static`` adds Φ12; ``optitrack`` adds the comparisons with the truth (Φ3 truth, Φ5, Φ11)."""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "twin.bvh"
        path.write_text(session.bvh, encoding="utf-8", newline="\n")
        bvh = read_bvh(path)
    fk = forward_kinematics(bvh)
    d = st.series_from_bvh(bvh, fk)
    bones = pod_bones(bvh)
    walk = st.walking_mask(d)
    out: dict = {"kind": kind, "n_frames": bvh.n_frames, "duration_s": bvh.duration_s,
                 "phi10": {"frame_time_s": bvh.frame_time_s, "source": bvh.frame_time_source}, "pod_bones": bones,
                 "phi1_walking_s": float(walk.sum() * d.dt_s)}  # fmt: skip
    if walk.any():
        out["phi3_drawn_deg"] = {s: _f(st.swing_axis_deg(d, s, walk)) for s in SWING_SITES}
        out["phi4"] = {}
        for s in ("LeftLeg", "RightLeg"):
            starts, ang, slope = st.swing_windows(d, s, walk)
            out["phi4"][s] = {"starts_s": starts.tolist(), "angles_deg": [_f(a) for a in ang],
                              "slope_deg_per_min": _f(slope)}
        speed, slow = st.root_speed(d, walk)
        out["phi6_speed_cm_s"], out["phi6_slow_share"] = _f(100.0 * speed), _f(slow)
        out["phi7_deg"] = _f(st.travel_minus_facing(d, walk))
    out["root_height_cm"] = float(100.0 * np.median(d.root[:, 2]))
    rep = st.exact_repeats(bvh, bones)
    out["phi8_pct"] = _f(100.0 * np.nanmedian(list(rep.values())))
    jumps = st.jumps(bvh, bones)
    out["phi9"] = {"leg_jumps": int(sum(len(jumps[b]) for b in bones if b in LEG_BONES))}
    if kind == "static":
        out["phi12"] = st.static_hold(bvh, bones, fk)
    if kind != "optitrack" or not session.truth:
        return out

    tr = truth_series(session.truth)
    t_true = session.truth["record_on_s"] + np.arange(bvh.n_frames) * session.truth["frame_interval_s"]
    change = drawn_height_changes(d.sites, d.root, d.dt_s, SWING_SITES)
    al = fine_lag_s(t_true, change, tr.t_s, tr.sites, 0.0, search_s=LAG_SEARCH_S)
    truth = st.resample(tr, t_true + al.lag_s, d.dt_s)
    out["alignment"] = {"lag_s": al.lag_s, "score": al.score}
    walk_t = st.walking_mask(truth)
    out["phi1_truth_walking_s"] = float(walk_t.sum() * d.dt_s)
    out["phi3_truth_deg"] = {s: _f(st.swing_axis_deg(truth, s, walk_t)) for s in SWING_SITES}
    tr_root = st.Series(truth.t_s, truth.dt_s, truth.sites, truth.facing, truth.sites["Hips"], "optitrack")
    out["phi7_truth_deg"] = _f(st.travel_minus_facing(tr_root, walk_t))  # the pelvis marker's travel
    pel = truth.sites["Hips"]
    scale, turn, rms, _ = st.umeyama_2d(pel[:, :2], d.root[:, :2])
    out["phi5"] = {"scale": scale, "turn_deg": turn, "rms_cm": 100.0 * rms,
                   "pelvis_height_cm": float(100.0 * np.nanmedian(pel[:, 2])),
                   "root_height_cm": float(100.0 * np.median(d.root[:, 2]))}  # fmt: skip
    change_c = drawn_height_changes(d.sites, d.root, d.dt_s, CORR_SITES)
    out["phi11_change_r"] = {s: _f(pearson(change_c[s], height_change(truth.sites[s][:, 2], d.dt_s)))
                             for s in CORR_SITES}  # fmt: skip
    return out
