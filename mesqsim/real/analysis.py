"""Φ1-Φ12 for one recording, and the comparison with the analysis of 2026-11-14 (Step 3 brief §5.3).

:func:`analyse` returns plain numbers (JSON-ready). Statistics of the BVH alone use the file's own frame time, as
the reference analysis did; comparisons with OptiTrack use the clock of :func:`mesqsim.real.align.align`, which also
fits the frame interval. :func:`compare` lines the results up against :data:`mesqsim.real.recordings.REFERENCE`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..twin.page_frame import motive_to_world_vec
from . import stats as st
from .align import align, drawn_height_changes, height_change, pearson, sample_at
from .bvh import forward_kinematics, pod_bones, read_bvh
from .markers import track_sites
from .motive import read_motive_csv
from .recordings import REFERENCE, Recording, real_suit_dir

__all__ = ["LEG_BONES", "SWING_SITES", "analyse", "compare"]

SWING_SITES = ("LeftLeg", "RightLeg", "LeftUpLeg", "RightUpLeg")
LEG_BONES = ("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot")
CORR_SITES = ("LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")


def _f(x) -> float | None:
    x = float(x)
    return None if not np.isfinite(x) else x


def analyse(rec: Recording, root: Path | None = None) -> dict:
    """Every statistic of Step 3 brief §5.2 that applies to ``rec``."""
    root = root or real_suit_dir()
    bvh = read_bvh(root / rec.bvh)
    fk = forward_kinematics(bvh)
    d = st.series_from_bvh(bvh, fk)
    bones = pod_bones(bvh)
    walk = st.walking_mask(d)
    out: dict = {"id": rec.id, "kind": rec.kind, "n_frames": bvh.n_frames, "duration_s": bvh.duration_s,
                 "phi10": {"frame_time_s": bvh.frame_time_s, "source": bvh.frame_time_source}, "pod_bones": bones,
                 "phi1_walking_s": float(walk.sum() * d.dt_s)}
    if walk.any():
        out["phi3_drawn_deg"] = {s: _f(st.swing_axis_deg(d, s, walk)) for s in SWING_SITES}
        out["phi3_drawn_roundness"] = {s: _f(st.swing_roundness(d, s, walk)) for s in SWING_SITES}
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
    out["phi8_per_bone_pct"] = {b: _f(100.0 * v) for b, v in rep.items()}
    jumps = st.jumps(bvh, bones)
    out["phi9"] = {"jumps": {b: v for b, v in jumps.items() if v},
                   "leg_jumps": int(sum(len(jumps[b]) for b in bones if b in LEG_BONES))}
    if rec.kind == "static":
        out["phi12"] = st.static_hold(bvh, bones, fk)
    if rec.motive is None:
        return out

    take = read_motive_csv(root / rec.motive)
    trk = track_sites(take)
    al = align(bvh, rec.bvh, d.sites, d.root, take, trk.sites)
    tq = np.arange(bvh.n_frames) * al.frame_time_s + al.lag_s  # Motive time of each BVH frame
    truth = st.resample(st.series_from_track(trk, take.t_s, 1.0 / take.frame_rate_hz), tq, d.dt_s)
    vel = np.gradient(trk.sites["Hips"][:, :2], 1.0 / take.frame_rate_hz, axis=0)
    moving = np.linalg.norm(vel, axis=1) > 0.3
    ok_f = moving & np.all(np.isfinite(trk.facing), axis=1)
    out["alignment"] = {"coarse_s": al.coarse_s, "lag_s": al.lag_s, "frame_time_s": al.frame_time_s,
                        "clock_difference_s": al.clock_difference_s, "score": al.score,
                        "per_leg_site": dict(zip(("LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg"), al.per_site,
                                                 strict=True))}
    out["tracking"] = {"coverage": trk.coverage(), "unnamed": len(trk.unnamed), "fragments": len(take.names),
                       "tpose_heights_m": trk.tpose_heights_m,
                       "walking_toward_chest": _f(np.mean(np.sum(vel[ok_f] * trk.facing[ok_f], axis=1) > 0))}
    walk_t = st.walking_mask(truth)
    out["phi1_truth_walking_s"] = float(walk_t.sum() * d.dt_s)
    out["phi3_truth_deg"] = {s: _f(st.swing_axis_deg(truth, s, walk_t)) for s in SWING_SITES}
    out["phi3_truth_roundness"] = {s: _f(st.swing_roundness(truth, s, walk_t)) for s in SWING_SITES}
    # Φ5: the pelvis marker's path, or the median of the visible markers where it is hidden
    med = np.nanmedian(motive_to_world_vec(take.xyz), axis=1)
    pelvis = np.where(np.isfinite(trk.sites["Hips"][:, :1]), trk.sites["Hips"], med)
    pel = sample_at(tq, take.t_s, pelvis)
    scale, turn, rms, _ = st.umeyama_2d(pel[:, :2], d.root[:, :2])
    both = np.isfinite(pel[:, 2])
    out["phi5"] = {"scale": scale, "turn_deg": turn, "rms_cm": 100.0 * rms,
                   "pelvis_height_cm": float(100.0 * np.median(pel[both, 2])),
                   "root_height_cm": float(100.0 * np.median(d.root[both, 2]))}
    change = drawn_height_changes(d.sites, d.root, d.dt_s, CORR_SITES)
    out["phi11_r"] = {s: _f(pearson(d.sites[s][:, 2], truth.sites[s][:, 2])) for s in CORR_SITES}
    out["phi11_change_r"] = {s: _f(pearson(change[s], height_change(truth.sites[s][:, 2], d.dt_s)))
                             for s in CORR_SITES}
    twin = {"LeftUpLeg": "RightUpLeg", "RightUpLeg": "LeftUpLeg", "LeftLeg": "RightLeg", "RightLeg": "LeftLeg"}
    out["phi11_swapped_change_r"] = {s: _f(pearson(change[s], height_change(truth.sites[t][:, 2], d.dt_s)))
                                     for s, t in twin.items()}  # the left/right check of Step 3 brief §5.1
    return out


def _axial_diff(a: float, b: float) -> float:
    return abs(float(st.axial_deg(a - b)))


def compare(results: dict[str, dict]) -> list[dict]:
    """One row per reference value of §5.3: statistic, recording, value, reference, tolerance, ``ok``."""
    rows = []

    def row(stat, rid, value, ref, ok, tol):
        rows.append({"statistic": stat, "recording": rid, "value": value, "reference": ref, "tolerance": tol,
                     "ok": bool(ok) if ok is not None else None})

    for key, (refs, (_kind, tol)) in REFERENCE.items():
        for rid, ref in refs.items():
            r = results.get(rid)
            if r is None:
                continue
            if key == "phi1_walking_s":
                v = r["phi1_walking_s"]
                row(key, rid, v, ref, abs(v - ref) <= tol * ref, f"±{100 * tol:.0f} %")
            elif key in ("phi3_shin_drawn_deg", "phi3_shin_truth_deg", "phi3_thigh_drawn_deg"):
                src = r["phi3_truth_deg" if "truth" in key else "phi3_drawn_deg"]
                sites = ("LeftUpLeg", "RightUpLeg") if "thigh" in key else ("LeftLeg", "RightLeg")
                v = tuple(src[s] for s in sites)
                row(key, rid, v, ref, all(_axial_diff(a, b) <= tol for a, b in zip(v, ref, strict=True)),
                    f"±{tol:.0f}°, axial")
            elif key == "phi4_shin_first_last_deg":
                p = r["phi4"]
                v = tuple((p[s]["angles_deg"][0], p[s]["angles_deg"][8]) for s in ("LeftLeg", "RightLeg"))
                row(key, rid, v, ref, all(abs(a - b) <= tol for va, vb in zip(v, ref, strict=True)
                                          for a, b in zip(va, vb, strict=True)), f"±{tol:.0f}°")
            elif key == "phi5_scale":
                v = r["phi5"]["scale"]
                row(key, rid, v, ref, abs(v - ref) <= tol, f"±{tol}")
            elif key == "phi5_heights_cm":
                v = (r["phi5"]["pelvis_height_cm"], r["phi5"]["root_height_cm"])
                row(key, rid, v, ref, all(abs(a - b) <= tol for a, b in zip(v, ref, strict=True)), f"±{tol:.0f} cm")
            elif key == "root_height_cm":
                v = r["root_height_cm"]
                row(key, rid, v, ref, abs(v - ref) <= tol, f"±{tol:.0f} cm")
            elif key == "phi6_speed_cm_s":
                v = r["phi6_speed_cm_s"]
                row(key, rid, v, ref, abs(v - ref) <= tol * ref, f"±{100 * tol:.0f} %")
            elif key == "phi7_travel_minus_facing_deg":
                v = r["phi7_deg"]
                row(key, rid, v, ref, abs(v - ref) <= tol, f"±{tol:.0f}°")
            elif key == "phi8_exact_repeats_pct":
                v = r["phi8_pct"]
                row(key, rid, v, ref, abs(v - ref) <= tol, f"±{tol:.0f} points")
            elif key == "phi9_leg_jumps":
                v = r["phi9"]["leg_jumps"]
                row(key, rid, v, ref, v == ref, "exact")
            elif key == "phi10_frame_time":
                v = (r["phi10"]["frame_time_s"], r["phi10"]["source"])
                row(key, rid, v, ref, abs(v[0] - ref[0]) < 1e-6 and v[1] == ref[1], "exact")
            elif key == "phi11_upper_arm_r":
                v = (r["phi11_r"]["LeftArm"], r["phi11_r"]["RightArm"])
                row(key, rid, v, ref, all(abs(a - b) <= tol for a, b in zip(v, ref, strict=True)), f"±{tol:.2f}")
            elif key == "phi12_pods_within_1p8":
                h = r["phi12"]
                v = (len(h["holding"]), len(h["steady"]))
                row(key, rid, v, ref, v == tuple(ref), "exact count")
    return rows
