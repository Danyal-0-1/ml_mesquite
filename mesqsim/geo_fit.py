"""Fitting the estimator's data-dependent beliefs on the training and validation splits (Step 2 brief rule 6).

This module reads truth (source motion, simulated labels), so it lives outside :mod:`mesqsim.estimators`; the
estimator only ever sees the fitted *values*, through the ``geometry:`` section of the config, labelled ``fitted``
with a source note naming the command, the data and the split. Nothing here ever touches the test split.

* :func:`fit_rom` (train): the range-of-motion box Omega per joint - the 0.5th and 99.5th percentiles of the
  training subjects' true Cardan angles (Step 2 brief §5.6) - and two descriptive numbers the brief asks for: how
  hinge-like the knee is (RMS of its rotation about its two other axes) and how much each 2-DoF joint turns about
  its third axis.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .body import JOINTS
from .estimators.anatomy import CARDAN_SEQ
from .estimators.rom import joint_angles_rad

__all__ = ["ROM_JOINTS", "RomFitResult", "fit_rom", "rom_source_note"]

ROM_JOINTS = tuple(k for k, j in enumerate(JOINTS) if j.jtype != "hinge")
"""Joints with a range-of-motion box: GEO-K uses it at 3-DoF joints, GEO-ID at 2-DoF and 3-DoF joints."""
_BINS = np.linspace(-180.0, 180.0, 36001)  # 0.01° histogram bins: percentiles exact to 0.01°
_TARGET_HZ = 30.0  # frames are subsampled to about this rate (neighbouring frames carry no new posture)


@dataclass
class RomFitResult:
    limits_deg: dict  # joint name -> [[lo, hi] x 3]
    n_frames: int
    n_sequences: int
    knee_off_axis_rms_deg: dict  # knee name -> (RMS of Cardan angle 2, RMS of angle 3)
    third_axis_rms_deg: dict  # 2-DoF joint name -> RMS of its third Cardan angle


def _sequence_stats(args) -> tuple[np.ndarray, np.ndarray, int]:
    """Histogram counts (14, 3, bins) and sums of squares (14, 3) of one sequence's true joint angles."""
    path, spine_joint = args
    from .sim.motion import MotionSource

    m = MotionSource.from_spec(path, spine_joint=spine_joint)
    step = max(1, int(round(m.fps / _TARGET_HZ)))
    seg = m.seg_q[::step]
    counts = np.zeros((len(JOINTS), 3, _BINS.size - 1), np.int64)
    sumsq = np.zeros((len(JOINTS), 3))
    for k in range(len(JOINTS)):
        ang = np.rad2deg(joint_angles_rad(seg, k))
        for a in range(3):
            counts[k, a] += np.histogram(ang[:, a], bins=_BINS)[0]
        sumsq[k] += np.sum(ang * ang, axis=0)
    return counts, sumsq, seg.shape[0]


def _percentile_from_hist(counts: np.ndarray, q: float) -> float:
    c = np.cumsum(counts)
    target = q / 100.0 * c[-1]
    i = int(np.searchsorted(c, target))
    prev = c[i - 1] if i > 0 else 0
    frac = (target - prev) / max(counts[i], 1)
    return float(_BINS[i] + frac * (_BINS[i + 1] - _BINS[i]))


def fit_rom(sources: pd.DataFrame, spine_joint: int, percentiles=(0.5, 99.5), workers: int | None = None,
            log=print) -> RomFitResult:  # fmt: skip
    """Omega from the **training** rows of ``sources`` (columns path, split); raises if any other split is given."""
    train = sources[sources["split"] == "train"]
    if train.empty:
        raise ValueError("no training sequences")
    args = [(p, spine_joint) for p in sorted(train["path"])]
    counts = np.zeros((len(JOINTS), 3, _BINS.size - 1), np.int64)
    sumsq = np.zeros((len(JOINTS), 3))
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (c, s, k) in enumerate(ex.map(_sequence_stats, args, chunksize=8)):
            counts += c
            sumsq += s
            n += k
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(args)} sequences")
    limits = {}
    for k in ROM_JOINTS:
        limits[JOINTS[k].name] = [[round(_percentile_from_hist(counts[k, a], percentiles[0]), 2),
                                   round(_percentile_from_hist(counts[k, a], percentiles[1]), 2)] for a in range(3)]
    rms = np.sqrt(sumsq / max(n, 1))
    knees = {JOINTS[k].name: (round(float(rms[k, 1]), 3), round(float(rms[k, 2]), 3))
             for k, j in enumerate(JOINTS) if j.jtype == "hinge"}  # fmt: skip
    third = {JOINTS[k].name: round(float(rms[k, 2]), 3) for k, j in enumerate(JOINTS) if j.jtype == "2dof"}
    return RomFitResult(limits, n, len(args), knees, third)


def rom_source_note(command: str, data: str, n_seq: int, n_frames: int, percentiles, commit: str, date: str) -> str:
    return (f"fitted by `{command}` on {data}, training split only: the {percentiles[0]:g}th and {percentiles[1]:g}th "
            f"percentiles of the true Cardan angles ({{seq}}) over {n_frames:,} frames of {n_seq:,} sequences "
            f"(subsampled to about {_TARGET_HZ:g} Hz); commit {commit[:12]}, {date}")


def cardan_seq_note(joint: str) -> str:
    return CARDAN_SEQ[joint]


# ---------------------------------------------------------------------------------------------- calibration (val)

CALIBRATED = ("kappa_scale", "model_sigma_deg")


def calibration_worker(job: dict, cfg) -> dict:
    """One validation session: GEO-ID's filter events, its window estimates with the truth at each window end, and
    the motion-phase query times with their truth. Scorer side (reads the labels)."""
    from .estimators.geo import GeoEstimator
    from .geo_sessions import simulate_job

    d, series, host, grid = simulate_job(job, cfg)
    _, diag = GeoEstimator(cfg).run(series, d.session, host, diagnostics=True)
    mot = grid.phase == "motion"
    wins = diag.windows
    t_w = np.array([r["t_s"] for r in wins])
    j_w = np.array([r["joint"] for r in wins], dtype=np.int64)
    k_end = np.clip(np.searchsorted(host, t_w, side="right") - 1, 0, host.size - 1)
    return {
        "job": job,
        "events": diag.events,
        "t_query": host[mot],
        "truth": grid.delta_rad[mot],
        "win_joint": j_w,
        "win_delta": np.array([r["delta_rad"] for r in wins]),
        "win_kappa_stat": np.array([r["kappa_stat"] for r in wins]),
        "win_truth": grid.delta_rad[k_end, j_w] if len(wins) else np.zeros(0),
        "seconds": diag.seconds,
    }


_BOUNDS = [(-8.0, 2.0), (0.0, 120.0)]  # log10(scale), model sigma in degrees


def _fit_window_type(err: np.ndarray, k_stat: np.ndarray, sigma_fixed_deg: float) -> tuple[float, float, float, str]:
    """Maximum-likelihood calibration of kappa_w = s·k/(1 + k·sigma²) for one joint type.

    Both s and sigma when the data identify them; when the optimum sits on a bound (the windows' kappa_stat hardly
    varies, so only s·k/(1 + k·sigma²) at that k is determined) the brief's single scale factor is fitted with sigma
    held at its stated assumption. Returns (scale, sigma_deg, mean log-likelihood, "scale+sigma" or "scale").
    """
    from scipy.optimize import minimize, minimize_scalar

    from .conventions import vm_logpdf

    def nll(x):
        s, sig = 10.0 ** x[0], np.deg2rad(x[1])
        return -float(np.sum(vm_logpdf(err, s * k_stat / (1.0 + k_stat * sig * sig))))

    best = None
    for x0 in ([0.0, 2.0], [-2.0, 5.0], [-1.0, 20.0], [-3.0, 1.0]):
        r = minimize(nll, x0, method="L-BFGS-B", bounds=_BOUNDS)
        if best is None or r.fun < best.fun:
            best = r
    on_bound = any(abs(v - b) < 1e-3 for v, (lo, hi) in zip(best.x, _BOUNDS, strict=True) for b in (lo, hi))
    if not on_bound or best.x[1] < 1e-3:  # sigma = 0 is a legitimate answer (an exact model), not a failure
        return float(10.0 ** best.x[0]), float(best.x[1]), float(-best.fun / max(err.size, 1)), "scale+sigma"
    r = minimize_scalar(lambda x: nll([x, sigma_fixed_deg]), bounds=_BOUNDS[0], method="bounded")
    return float(10.0 ** r.x), float(sigma_fixed_deg), float(-r.fun / max(err.size, 1)), "scale"


def _replay_all(results: list, q: float, scales: dict, sigmas_rad: dict, weight: float):
    from .estimators.geo import replay

    errs, kaps, joints = [], [], []
    for r in results:
        mu, kap, _ = replay(r["events"], r["t_query"], q, scales, sigmas_rad, weight)
        errs.append(np.abs(np.angle(np.exp(1j * (mu - r["truth"])))))
        kaps.append(kap)
        joints.append(np.broadcast_to(np.arange(mu.shape[1]), mu.shape))
    return np.concatenate([e.ravel() for e in errs]), np.concatenate([k.ravel() for k in kaps]), \
        np.concatenate([j.ravel() for j in joints])  # fmt: skip


def fit_calibration(results: list, weight: float, sigma_assumed_deg: dict, q_grid=None,
                    hold_target_p95_deg: float = 10.0, log=print) -> dict:  # fmt: skip
    """Fit the window calibration per joint type, then q, then kappa_hold, on validation sessions (GEO-ID).

    1. (kappa_scale, model_sigma_deg) per joint type: maximum likelihood of the window errors under
       VM(0, s·k/(1 + k·sigma²)) (brief §5.4 recalibration, with the model-error term).
    2. drift_q_rad2_per_s: the value on a log grid that maximises the von Mises log-likelihood of the filter's
       outputs at every motion-phase 32 Hz step (replays of the recorded events).
    3. kappa_hold: the smallest threshold whose retained steps (kappa >= threshold) have a P95 error at most
       ``hold_target_p95_deg``: "uncertain" then means "not shown, on validation, to be within that bound".
    """
    from .body import JOINTS
    from .conventions import vm_halfwidth_rad, vm_logpdf, wrap_rad

    jt = np.array([j.jtype for j in JOINTS])
    jw = np.concatenate([r["win_joint"] for r in results])
    dw = np.concatenate([r["win_delta"] for r in results])
    kw = np.concatenate([r["win_kappa_stat"] for r in results])
    tw = np.concatenate([r["win_truth"] for r in results])
    ew = np.abs(wrap_rad(dw - tw))
    scales, sigmas_deg, win_ll, n_win, how = {}, {}, {}, {}, {}
    for t in ("hinge", "2dof", "3dof"):
        sel = (jt[jw] == t) & (kw > 0) & np.isfinite(tw)
        n_win[t] = int(sel.sum())
        if n_win[t] < 20:
            scales[t], sigmas_deg[t], win_ll[t], how[t] = 1.0, float(sigma_assumed_deg[t]), float("nan"), "none"
            log(f"  {t}: only {n_win[t]} informative windows; scale 1 and the assumed model sigma kept")
            continue
        scales[t], sigmas_deg[t], win_ll[t], how[t] = _fit_window_type(ew[sel], kw[sel], float(sigma_assumed_deg[t]))
        log(f"  {t}: {n_win[t]} windows, fitted {how[t]}: scale {scales[t]:.4g}, model sigma {sigmas_deg[t]:.3f}°, "
            f"mean log-lik {win_ll[t]:.3f}")
    sig_rad = {t: float(np.deg2rad(v)) for t, v in sigmas_deg.items()}
    q_grid = np.asarray(q_grid if q_grid is not None else 10.0 ** np.arange(-9.0, -2.74, 0.25))
    ll = []
    for q in q_grid:
        e, k, _ = _replay_all(results, float(q), scales, sig_rad, weight)
        ll.append(float(np.mean(vm_logpdf(e, k))))
    q_best = float(q_grid[int(np.argmax(ll))])
    log(f"  q: {q_best:.3g} rad²/s (mean log-lik {max(ll):.4f}; grid {q_grid[0]:.0e} to {q_grid[-1]:.0e})")
    e, k, j = _replay_all(results, q_best, scales, sig_rad, weight)
    thr_grid = np.r_[0.0, 10.0 ** np.arange(-1.0, 6.01, 0.05)]
    target = np.deg2rad(hold_target_p95_deg)
    kappa_hold, kept = float(thr_grid[-1]), 0.0
    for thr in thr_grid:
        sel = k >= thr
        if sel.mean() < 0.01:
            break
        if np.percentile(e[sel], 95) <= target:
            kappa_hold, kept = float(thr), float(sel.mean())
            break
    log(f"  kappa_hold {kappa_hold:.4g}: {100 * kept:.1f} % of validation steps certain")
    coverage = {}
    for t in ("hinge", "2dof", "3dof", "all"):
        sel = np.ones(e.size, bool) if t == "all" else (jt[j] == t)
        coverage[t] = {lev: float(np.mean(e[sel] <= vm_halfwidth_rad(k[sel], lev))) for lev in (0.68, 0.95)}
    return {"kappa_scale": scales, "model_sigma_deg": sigmas_deg, "fitted_how": how, "window_mean_loglik": win_ll,
            "n_windows": n_win,
            "q_grid": q_grid.tolist(), "q_loglik": ll, "drift_q_rad2_per_s": q_best, "kappa_hold": kappa_hold,
            "certain_fraction": kept, "coverage": coverage, "n_sessions": len(results),
            "n_steps": int(e.size)}  # fmt: skip
