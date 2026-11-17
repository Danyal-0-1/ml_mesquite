"""The Step 2 results: every method on the test sessions, scored in memory (Step 2 brief §6, S2-M6). Scorer side.

:func:`session_worker` simulates one dataset session (``mesqsim.geo_sessions``), runs every method and keeps the
motion-phase errors as compact arrays. GEO-ID-nofilter and GEO-ID-hinge are exact replays of GEO-ID's recorded filter
events (no filter, or the non-knee windows' kappa set to 0), which equal the registered estimators
(``tests/test_results_step2.py``). The tables, statistics and figures of docs/RESULTS_STEP2.md are built from these
arrays by ``scripts/make_results_step2.py``.
"""

from __future__ import annotations

import numpy as np

from .body import JOINTS, N_JOINTS
from .conventions import vm_halfwidth_rad, wrap_rad

__all__ = ["METHODS_ALL", "METHODS_SWEEP", "ERR_ORIENT", "session_worker", "excitation", "bootstrap_contrast",
           "holm", "coverage", "selective_risk"]  # fmt: skip

METHODS_ALL = ("B0", "B1", "O1", "O2", "GEO-K", "GEO-ID", "GEO-ID-nofilter", "GEO-ID-hinge", "GEO-ID-w4",
               "GEO-ID-w16")  # fmt: skip
METHODS_SWEEP = ("B1", "O1", "O2", "GEO-K", "GEO-ID")
ERR_ORIENT = ("B1", "O1", "O2", "GEO-K", "GEO-ID")  # secondary metric kept for these
_JT = np.array([j.jtype for j in JOINTS])
_KNEES = tuple(k for k, j in enumerate(JOINTS) if j.jtype == "hinge")
NEAR_LIMIT_DEG = 5.0  # a 3-DoF joint is "near its limit" within this of an edge of Omega (or outside it)


def excitation(clip_seg_q: np.ndarray, cfg) -> np.ndarray:
    """Per joint, from the truth: hinge and 2-DoF joints the spread (std, degrees) of their main Cardan angle over the
    motion (how much they moved); 3-DoF joints the share of frames within 5° of a range-of-motion limit or outside."""
    from .estimators.rom import joint_angles_rad

    out = np.zeros(N_JOINTS)
    for k, j in enumerate(JOINTS):
        ang = np.rad2deg(joint_angles_rad(clip_seg_q, k))
        if j.jtype in ("hinge", "2dof"):
            out[k] = float(np.std(ang[:, 0]))
        else:
            box = np.asarray(cfg[f"geometry.rom_limits_deg.{j.name}"], dtype=np.float64)
            near = np.any((ang < box[:, 0] + NEAR_LIMIT_DEG) | (ang > box[:, 1] - NEAR_LIMIT_DEG), axis=1)
            out[k] = float(np.mean(near))
    return out


def session_worker(job: dict, cfg, methods=METHODS_ALL) -> dict:
    """Every method on one session: motion-phase errors (degrees, float32), kappa, uncertain, and context."""
    from . import evaluate as ev
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator, GeoParams, replay
    from .geo_sessions import simulate_job

    d, series, host, grid, clip = simulate_job(job, cfg, with_clip=True)
    data = {"labels_joints": d.labels_joints, "labels_ticks": d.labels_ticks, "meta": d.meta, "session": d.session}
    sc = ev.Scorer(data, series)
    mot = grid.phase == "motion"
    est: dict = {}
    for name in ("B0", "B1", "O1", "O2"):
        if name in methods:
            est[name] = (ev.estimate(name, grid, series, d.session), None, None)
    seconds = np.nan
    if any(m.startswith("GEO-ID") for m in methods):
        e_id, diag = GeoEstimator(cfg).run(series, d.session, host, diagnostics=True)
        seconds = diag.seconds
        est["GEO-ID"] = (e_id.delta_rad, e_id.kappa, e_id.uncertain)
        p = GeoParams(cfg)
        q, sc_, sg = float(cfg["geometry.drift_q_rad2_per_s"]), p.scales(), p.model_sigmas_rad()
        after = host >= diag.events.t_init
        hold = float(cfg["geometry.kappa_hold"])
        if "GEO-ID-nofilter" in methods:
            mu, kap, _ = replay(diag.events, host, q, sc_, sg, p.weight(False), use_filter=False)
            dlt = np.where(after[:, None], wrap_rad(mu), e_id.delta_rad)
            kap = np.where(after[:, None], kap, 0.0)
            est["GEO-ID-nofilter"] = (dlt, kap, kap < hold)
        if "GEO-ID-hinge" in methods:
            hinge = _JT == "hinge"
            ev_h = type(diag.events)(diag.events.t_init, diag.events.box_delta_rad,
                                     [(t, k, (pl[0], np.where(hinge, pl[1], 0.0)) if k == 2 else pl)
                                      for t, k, pl in diag.events.events])  # fmt: skip
            mu, kap, _ = replay(ev_h, host, q, sc_, sg, p.weight(True))
            dlt = np.where(after[:, None], wrap_rad(mu), e_id.delta_rad)
            kap = np.where(after[:, None], kap, 0.0)
            est["GEO-ID-hinge"] = (dlt, kap, kap < hold)
        for w in (4, 16):
            if f"GEO-ID-w{w}" in methods:
                geo_w = GeoEstimator(cfg.with_overrides({"geometry.window_s": float(w)}))
                e_w = geo_w.estimate(series, d.session, host)
                est[f"GEO-ID-w{w}"] = (e_w.delta_rad, e_w.kappa, e_w.uncertain)
        resets = [(t, joints) for t, kind, joints in diag.events.events if kind == 1]
        # what GEO-ID could identify: without accepted knee axes it has no facing, hence no T-pose frames and no
        # range of motion either, and every joint keeps its box estimate
        identified = {"knee_axes": np.array([k in diag.state["axes"] for k in _KNEES]),
                      "facing": diag.state["facing_rad"] is not None}  # fmt: skip
    else:
        resets, identified = [], None
    if "GEO-K" in methods:
        e_k = GeoEstimator(cfg, GeoTruth.from_meta(d.meta)).estimate(series, d.session, host)
        est["GEO-K"] = (e_k.delta_rad, e_k.kappa, e_k.uncertain)
    out = {"job": job, "n_query": int(host.size), "geo_seconds": seconds, "excitation": excitation(clip.seg_q, cfg),
           "geo_id": identified}  # fmt: skip
    since = np.full((int(mot.sum()), N_JOINTS), np.nan)
    for t, joints in resets:
        dt = host[mot] - t
        for j in joints:
            since[:, j] = np.where(dt >= 0, dt, since[:, j])
    out["since_reset_s"] = since.astype(np.float32)
    for name, (dlt, kap, unc) in est.items():
        err_h, err_o = sc.errors(dlt)
        out[name] = {"err": np.rad2deg(err_h[mot]).astype(np.float32)}
        if name in ERR_ORIENT:
            out[name]["err_orient"] = np.rad2deg(err_o[mot]).astype(np.float32)
        if kap is not None:
            out[name]["kappa"] = np.asarray(kap)[mot].astype(np.float32)
            out[name]["uncertain"] = np.asarray(unc)[mot].astype(bool)
    return out


# ------------------------------------------------------------------------------------------ statistics

def bootstrap_contrast(diff_by_unit: np.ndarray, n_boot: int = 10_000, seed: int = 20261117) -> dict:
    """Paired contrast over units (subjects): mean of the per-unit differences, its 95 % percentile bootstrap
    interval (units resampled with replacement) and a two-sided bootstrap p-value."""
    d = np.asarray(diff_by_unit, dtype=np.float64)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"n": 0, "mean": np.nan, "lo": np.nan, "hi": np.nan, "p": np.nan}
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, (n_boot, d.size))].mean(axis=1)
    p = float(min(1.0, 2.0 * min(np.mean(means <= 0.0), np.mean(means >= 0.0))))
    return {"n": int(d.size), "mean": float(d.mean()), "lo": float(np.percentile(means, 2.5)),
            "hi": float(np.percentile(means, 97.5)), "p": max(p, 1.0 / n_boot)}  # fmt: skip


def holm(pvalues: dict) -> dict:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, running, out = len(keys), 0.0, {}
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def coverage(err_deg: np.ndarray, kappa: np.ndarray, levels=(0.68, 0.95)) -> dict:
    """Share of steps whose error lies inside the central von Mises interval of the reported kappa."""
    e = np.deg2rad(np.asarray(err_deg, dtype=np.float64))
    k = np.asarray(kappa, dtype=np.float64)
    ok = np.isfinite(e) & np.isfinite(k)
    return {lev: float(np.mean(e[ok] <= vm_halfwidth_rad(k[ok], lev))) for lev in levels}


def selective_risk(err_deg: np.ndarray, kappa: np.ndarray, fractions=(1.0, 0.8, 0.6, 0.4, 0.2, 0.1)) -> list[dict]:
    """Error of the steps whose kappa is in the top ``f`` share, for each f (the most confident first)."""
    e = np.asarray(err_deg, dtype=np.float64)
    k = np.asarray(kappa, dtype=np.float64)
    ok = np.isfinite(e) & np.isfinite(k)
    e, k = e[ok], k[ok]
    order = np.argsort(-k, kind="mergesort")
    rows = []
    for f in fractions:
        n = max(1, int(round(f * e.size)))
        sel = e[order[:n]]
        rows.append({"fraction": f, "kappa_min": float(k[order[n - 1]]), "median": float(np.median(sel)),
                     "p95": float(np.percentile(sel, 95)), "mean": float(np.mean(sel))})  # fmt: skip
    return rows
