"""GEO-ID on DIODEM: real IMUs on mechanical chains against optical truth (Step 2 brief §6.4). Scorer side.

* **Data.** The rigid IMU of each segment, six-axis VQF (magnetometer dropped), 40 Hz, as in Step 1's sanity
  analysis (:mod:`mesqsim.data.diodem`). The brief's *Must*: the estimator runs on the 40 Hz series directly.
* **Truth.** The rigid IMUs' axes are the optical segment axes (verified in Step 1), so q_vqf = Y(-psi)·q_opt up to
  fusion error and psi = -heading(q_vqf·q_opt^-1) (:func:`psi_from_vqf_and_optical`; sign tested on a synthetic
  rotation); then delta = wrap(psi_c - psi_p), as in the simulator.
* **The estimator** is GEO-ID itself, unchanged: the two IMUs become pods 9 and 10 (the tree's left knee, a hinge)
  of a Step 0-shaped series with one boot epoch, every sample fresh and valid, pod time = host time = sample time,
  and a ``session.json`` with a single ``motion`` phase. With no box and no T-pose it starts with kappa = 0 and must
  find delta from the geometry alone (free axis fit, vertical sign rule).
* **Reference.** "Start oracle": delta held at its true value at the first sample (a perfect box registration at
  the start); its error is the six-axis drift that geometry has to track.
* **Joints covered.** Hinges only. The saddle joints need 2-DoF axes identified from motion (a *Should*, not built)
  and the spherical joint a range of motion fitted on these chains (none exists): reported as not covered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conventions import heading_rad, quat_conj, quat_mul, wrap_rad, wxyz_to_xyzw
from .data.diodem import ARM_EXPS, CHAINS, GAIT_EXPS

__all__ = ["HZ", "psi_from_vqf_and_optical", "vqf_six_axis", "diodem_series", "run_hinge", "hinge_joints"]

HZ = 40.0
PARENT_POD, CHILD_POD, JOINT = 9, 10, 10  # the tree's left knee: a hinge joint between pods 9 and 10


def psi_from_vqf_and_optical(q_vqf: np.ndarray, q_opt: np.ndarray) -> np.ndarray:
    """psi with q_vqf = Y(-psi)·q_opt: psi = -heading(q_vqf·q_opt^-1). Both (x, y, z, w), segment -> +Z-up world."""
    return -heading_rad(quat_mul(q_vqf, quat_conj(q_opt)))


def vqf_six_axis(imu: dict, hz: float = HZ) -> np.ndarray:
    """Six-axis VQF orientation (x, y, z, w) of one IMU (gyr rad/s, acc m/s²); the magnetometer is not used."""
    from vqf import VQF

    out = VQF(1.0 / hz).updateBatch(np.ascontiguousarray(imu["gyr"]), np.ascontiguousarray(imu["acc"]))
    return wxyz_to_xyzw(out["quat6D"])


def diodem_series(q_parent: np.ndarray, q_child: np.ndarray, hz: float = HZ, t0_s: float = 100.0):
    """A Step 0-shaped series and session.json for two IMUs as pods 9 (parent) and 10 (child)."""
    n = q_parent.shape[0]
    t = t0_s + np.arange(n) / hz
    parts = []
    for pod, q in ((PARENT_POD, q_parent), (CHILD_POD, q_child)):
        ok = np.all(np.isfinite(q), axis=1)
        parts.append(pd.DataFrame({
            "pod_id": np.full(n, pod, np.uint8), "epoch": np.zeros(n, np.int32), "tick": np.arange(n, dtype=np.int64),
            "count": (np.arange(n) % 65536).astype(np.uint16), "status": "fresh", "received": True, "valid": ok,
            "t_host_s": t, "t_host_est_s": np.nan, "ms_lo": np.nan, "ms_unwrapped": (t - t0_s) * 1000.0,
            "t_pod_s": t - t0_s, "batt": 100.0, "norm_err": 0.0,
            "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
        }))  # fmt: skip
    series = pd.concat(parts, ignore_index=True)
    session = {"schema_version": 1, "pods": [PARENT_POD, CHILD_POD],
               "phases": [{"name": "motion", "t_host_start_s": float(t[0]), "t_host_end_s": float(t[-1]),
                           "recorded": True}]}  # fmt: skip
    return series, session, t


def hinge_joints(exp_id: int) -> list[tuple[str, str]]:
    chain = "arm" if exp_id in ARM_EXPS else "gait"
    return [(p, c) for p, c, kind in CHAINS[chain] if kind == "hinge"]


def run_hinge(data: dict, parent: str, child: str, cfg, hz: float = HZ) -> dict:
    """GEO-ID on one DIODEM hinge: the time series of truth, estimate, kappa and the start oracle (radians)."""
    from .estimators.geo import GeoEstimator

    q = {s: vqf_six_axis(data[s]["imu_rigid"], hz) for s in (parent, child)}
    opt = {s: wxyz_to_xyzw(np.asarray(data[s]["quat"], dtype=np.float64)) for s in (parent, child)}
    psi = {s: psi_from_vqf_and_optical(q[s], opt[s]) for s in (parent, child)}
    truth = wrap_rad(psi[child] - psi[parent])
    series, session, t = diodem_series(q[parent], q[child], hz)
    est, diag = GeoEstimator(cfg).run(series, session, t, diagnostics=True)
    first = np.flatnonzero(np.isfinite(truth))
    start = truth[first[0]] if first.size else np.nan
    informative = [r["t_s"] for r in diag.windows if r["joint"] == JOINT and r["kappa_stat"] > 0]
    axes = diag.state["axes"].get(JOINT)
    return {"t_s": t - t[0], "truth_rad": truth, "delta_rad": est.delta_rad[:, JOINT], "kappa": est.kappa[:, JOINT],
            "uncertain": est.uncertain[:, JOINT], "start_oracle_rad": np.full(t.size, start),
            "t_first_info_s": (informative[0] - t[0]) if informative else np.nan,
            "axes": None if axes is None else (axes.u_p.tolist(), axes.u_c.tolist()),
            "seconds": diag.seconds}  # fmt: skip


def all_hinges(cfg, exps=ARM_EXPS + GAIT_EXPS, log=print) -> list[dict]:
    """Every hinge of every experiment: per-run error summaries (degrees). The experiment is the unit."""
    from .conventions import vm_halfwidth_rad
    from .data.diodem import load_trial

    rows = []
    for e in exps:
        data = load_trial(e)
        for parent, child in hinge_joints(e):
            r = run_hinge(data, parent, child, cfg)
            err = np.rad2deg(np.abs(wrap_rad(r["delta_rad"] - r["truth_rad"])))
            ref = np.rad2deg(np.abs(wrap_rad(r["start_oracle_rad"] - r["truth_rad"])))
            ok = np.isfinite(err)
            after = ok & (r["t_s"] >= (r["t_first_info_s"] if np.isfinite(r["t_first_info_s"]) else np.inf))
            cov95 = np.mean(err[ok] <= np.rad2deg(vm_halfwidth_rad(r["kappa"][ok], 0.95)))
            rows.append({"experiment": f"exp{e:02d}", "chain": "arm" if e in ARM_EXPS else "gait",
                         "joint": f"{parent}-{child}", "minutes": float(r["t_s"][-1] / 60.0),
                         "geo_mean_deg": float(np.mean(err[ok])), "geo_median_deg": float(np.median(err[ok])),
                         "geo_p95_deg": float(np.percentile(err[ok], 95)),
                         "geo_after_median_deg": float(np.median(err[after])) if after.any() else np.nan,
                         "geo_after_p95_deg": float(np.percentile(err[after], 95)) if after.any() else np.nan,
                         "start_median_deg": float(np.median(ref[ok])),
                         "start_p95_deg": float(np.percentile(ref[ok], 95)),
                         "start_end_deg": float(ref[ok][-1]), "first_info_s": r["t_first_info_s"],
                         "uncertain_frac": float(np.mean(r["uncertain"][ok])), "coverage95": float(cov95),
                         "seconds": r["seconds"]})  # fmt: skip
            log(f"  {rows[-1]['experiment']} {rows[-1]['joint']}: GEO-ID median {rows[-1]['geo_median_deg']:.1f}° "
                f"(after acquisition {rows[-1]['geo_after_median_deg']:.1f}°), start oracle median "
                f"{rows[-1]['start_median_deg']:.1f}°")
    return rows
