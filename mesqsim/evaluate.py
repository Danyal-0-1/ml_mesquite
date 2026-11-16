"""Baselines and error metrics against the labels (brief §4.7).

* **Primary:** relative-heading error |wrap(delta_hat - delta)| per joint and 32 Hz step, degrees.
* **Secondary:** parent-relative orientation error with the simulator's **true mountings**. At each step, every
  pod's latest valid received frame (by host time) is corrected outward from the pelvis; the joint rotation it
  implies is compared with the true joint rotation at the same sample instants:

      J_hat = (q_p m_p^-1)^-1 · Y(delta_hat_pc) · (q_c m_c^-1),   J = (P_p m_p^-1)^-1 · (P_c m_c^-1)
      error = geodesic angle of J^-1 · J_hat

  Comparing each frame with the truth at its own sample time removes timing error, so the secondary metric
  isolates heading, fusion and quantisation error.

Baselines:

* B0: delta_hat = 0, no correction after Step 0: the size of the problem.
* B1: box registration, delta_hat_pc = wrap(h_p - h_c) from Step 0's box headings, constant afterwards: the honest
  "no learning" baseline. (The T-pose snapshot removes mountings for display; with the true mountings used by the
  secondary metric it changes nothing, so B1's numbers come from the box alone.)
* O1: the true offsets wrap(a_c - a_p) of the current boot epochs (oracle): how much error is offset, not drift.
* O2: the true delta (oracle): primary error exactly 0.

Non-oracle baselines see only Step 0's output and session.json, never the labels.

Step 2 methods (brief §6.1), registered beside the baselines (:data:`GEO_METHODS`):

* GEO-K: the per-joint geometric estimator with known axes and mountings. **Uses simulator truth** (the true
  mountings from ``meta.json``, passed through its one ``truth`` argument).
* GEO-ID: the deployable estimator (axes identified from motion, T-pose frames, filter).
* Ablations of GEO-ID: ``GEO-ID-nofilter`` (windows only), ``GEO-ID-hinge`` (only the knees are solved; every other
  joint keeps its box estimate), ``GEO-ID-w4`` and ``GEO-ID-w16`` (4 s and 16 s windows).

``errors.parquet`` carries each method's ``kappa`` and ``uncertain`` flag; both are missing (NaN / null) for methods
without a confidence (the Step 1 baselines).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .body import JOINTS, N_JOINTS, N_PODS
from .conventions import geodesic_angle_rad, quat_conj, quat_mul, wrap_rad, yaw_quat

__all__ = ["BASELINES", "GEO_METHODS", "METHODS", "USES_TRUTH", "Grid", "grid_from_labels", "estimate",
           "method_estimate", "latest_frames", "evaluate"]  # fmt: skip

BASELINES = ("B0", "B1", "O1", "O2")
GEO_METHODS = ("GEO-K", "GEO-ID", "GEO-ID-nofilter", "GEO-ID-hinge", "GEO-ID-w4", "GEO-ID-w16")
METHODS = BASELINES + GEO_METHODS
USES_TRUTH = frozenset({"O1", "O2", "GEO-K"})
"""Methods that read simulator truth: every results table marks them."""


class Grid:
    """The 32 Hz evaluation steps and the true targets on them."""

    def __init__(self, labels_joints: pd.DataFrame):
        lj = labels_joints.sort_values(["step", "joint"], kind="mergesort")
        k = int(lj["step"].max()) + 1
        if len(lj) != k * N_JOINTS:
            raise ValueError("labels_joints must hold every joint at every step")
        first = lj[lj["joint"] == 0]
        self.step = first["step"].to_numpy()
        self.t_s = first["t_s"].to_numpy()
        self.phase = first["phase"].to_numpy()
        self.delta_rad = lj["delta_rad"].to_numpy().reshape(k, N_JOINTS)
        self.delta_offset_rad = lj["delta_offset_rad"].to_numpy().reshape(k, N_JOINTS)


def grid_from_labels(labels_joints: pd.DataFrame) -> Grid:
    return Grid(labels_joints)


def estimate(name: str, grid: Grid, series: pd.DataFrame, session: dict) -> np.ndarray:
    """delta_hat (K, 14) in radians for baseline ``name``."""
    if name == "B0":
        return np.zeros_like(grid.delta_rad)
    if name == "B1":
        from .step0 import box_headings

        h = np.full(N_PODS, np.nan)
        bh = box_headings(series, session)
        h[bh["pod_id"].to_numpy()] = bh["heading_rad"].to_numpy()
        d = np.array([wrap_rad(h[j.parent] - h[j.child]) for j in JOINTS])
        return np.broadcast_to(d, grid.delta_rad.shape).copy()
    if name == "O1":  # oracle: the true offsets of the current boot epochs
        return grid.delta_offset_rad.copy()
    if name == "O2":  # oracle: reads the labels
        return grid.delta_rad.copy()
    raise ValueError(f"unknown baseline {name!r}; known: {BASELINES}")


def geo_estimator(name: str, cfg, meta: dict | None):
    """The configured GEO estimator for method ``name``. Only GEO-K receives truth (the true mountings)."""
    from .estimators.contract import GeoTruth
    from .estimators.geo import GeoEstimator

    if name == "GEO-K":
        return GeoEstimator(cfg, GeoTruth.from_meta(meta))
    if name == "GEO-ID":
        return GeoEstimator(cfg)
    if name == "GEO-ID-nofilter":
        return GeoEstimator(cfg, use_filter=False)
    if name == "GEO-ID-hinge":
        return GeoEstimator(cfg, solve_types=("hinge",))
    if name in ("GEO-ID-w4", "GEO-ID-w16"):
        return GeoEstimator(cfg.with_overrides({"geometry.window_s": float(name.split("-w")[1])}))
    raise ValueError(f"unknown method {name!r}; known: {METHODS}")


def method_estimate(name: str, grid: Grid, series: pd.DataFrame, session: dict, meta: dict | None = None,
                    cfg=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """``(delta_hat, kappa, uncertain)`` (K, 14) for any method; kappa and uncertain are NaN for baselines.

    GEO methods are queried at the host times of the grid steps (the grid's true times plus the host epoch, which
    only the scorer knows); ``cfg`` must hold the ``geometry:`` section.
    """
    if name in BASELINES:
        nan = np.full(grid.delta_rad.shape, np.nan)
        return estimate(name, grid, series, session), nan, nan.copy()
    if cfg is None:
        raise ValueError(f"{name} needs the config (geometry section)")
    est = geo_estimator(name, cfg, meta).estimate(series, session, grid.t_s + meta["plan"]["host_epoch_s"])
    return est.delta_rad, est.kappa, est.uncertain.astype(np.float64)


def latest_frames(series: pd.DataFrame, host_times_s: np.ndarray) -> dict:
    """Per pod, the latest valid received frame at or before each host time.

    Returns arrays (K, 15): row index into ``series`` (-1 if none), and the decoded quaternions (K, 15, 4).
    """
    k = host_times_s.size
    rows = np.full((k, N_PODS), -1, np.int64)
    q = np.full((k, N_PODS, 4), np.nan)
    ok = series["received"].to_numpy() & series["valid"].to_numpy()
    for i in range(N_PODS):
        idx = np.flatnonzero(ok & (series["pod_id"].to_numpy() == i))
        if idx.size == 0:
            continue
        t = series["t_host_s"].to_numpy()[idx]
        order = np.argsort(t, kind="mergesort")
        idx, t = idx[order], t[order]
        pos = np.searchsorted(t, host_times_s, side="right") - 1
        have = pos >= 0
        rows[have, i] = idx[pos[have]]
        q[have, i] = series[["qx", "qy", "qz", "qw"]].to_numpy()[idx[pos[have]]]
    return {"row": rows, "q": q}


def _true_pod_q(series: pd.DataFrame, rows: np.ndarray, labels_ticks: pd.DataFrame) -> np.ndarray:
    """True pod orientation at the sample time of each selected frame, joined on (pod, epoch, tick)."""
    key = ["pod_id", "epoch", "tick"]
    lt = labels_ticks.set_index(key)[["pod_qx", "pod_qy", "pod_qz", "pod_qw"]]
    flat = rows.reshape(-1)
    out = np.full((flat.size, 4), np.nan)
    have = flat >= 0
    sel = series.iloc[flat[have]][key]
    out[have] = lt.loc[pd.MultiIndex.from_frame(sel)].to_numpy()
    return out.reshape(*rows.shape, 4)


def evaluate(data: dict, series: pd.DataFrame, baselines=BASELINES, cfg=None) -> pd.DataFrame:
    """Errors of each method at every step and joint. ``data`` is :func:`mesqsim.io.read_session` output.

    ``baselines`` may name any method of :data:`METHODS`; GEO methods need ``cfg`` (its ``geometry:`` section).
    """
    grid = Grid(data["labels_joints"])
    plan = data["meta"]["plan"]
    host = grid.t_s + plan["host_epoch_s"]
    latest = latest_frames(series, host)
    p_true = _true_pod_q(series, latest["row"], data["labels_ticks"])
    mount_inv = quat_conj(np.array([p["mount_q"] for p in plan["pods"]]))  # (15, 4)
    seg_hat = quat_mul(latest["q"], mount_inv)  # (K, 15, 4): q_i m_i^-1
    seg_true = quat_mul(p_true, mount_inv)
    parent = np.array([j.parent for j in JOINTS])
    child = np.array([j.child for j in JOINTS])
    j_true = quat_mul(quat_conj(seg_true[:, parent]), seg_true[:, child])  # (K, 14, 4)

    out = []
    for name in baselines:
        d_hat, kappa, uncertain = method_estimate(name, grid, series, data["session"], data.get("meta"), cfg)
        err_h = np.abs(wrap_rad(d_hat - grid.delta_rad))
        j_hat = quat_mul(quat_mul(quat_conj(seg_hat[:, parent]), yaw_quat(d_hat)), seg_hat[:, child])
        err_o = geodesic_angle_rad(quat_mul(quat_conj(j_true), j_hat))
        k = grid.step.size
        unc = uncertain.reshape(-1)
        out.append(pd.DataFrame({
            "baseline": name,
            "step": np.repeat(grid.step, N_JOINTS),
            "t_s": np.repeat(grid.t_s, N_JOINTS),
            "phase": np.repeat(grid.phase, N_JOINTS),
            "joint": np.tile(np.arange(N_JOINTS), k),
            "err_heading_deg": np.rad2deg(err_h).reshape(-1),
            "err_orient_deg": np.rad2deg(err_o).reshape(-1),
            "kappa": kappa.reshape(-1),
            "uncertain": pd.array(np.where(np.isnan(unc), None, unc > 0.5), dtype="boolean"),
        }))  # fmt: skip
    return pd.concat(out, ignore_index=True)
