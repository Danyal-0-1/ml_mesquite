"""GEO: the geometric relative-heading estimator, per joint, causal (Step 2 brief §5).

For every one of the 14 joints it estimates the relative heading delta (how far the child pod's north has slid
from its parent's) with its own confidence kappa, from Step 0's decoded orientations and ``session.json`` alone.

The estimator is a sequence of **events** in host time, each changing a von Mises belief per joint
(:class:`mesqsim.estimators.vmfilter.JointFilter`):

1. **init** at the end of the box phase: box registration (B1) with ``kappa0 = 1/(2·sigma_box²)``;
2. **reset** when Step 0 shows that a pod started a new boot epoch (it rebooted): its joints drop to kappa = 0;
3. **window** every ``step_s``: every joint is re-solved over the last ``window_s`` of paired samples by the solver
   of its joint type (hinge closed form, 2-DoF constraint, range of motion) and the result is multiplied in.

At a query time the answer is the latest event's belief predicted forward to it; before the box phase ends, a
running box registration with kappa = 0. Geometry only uses frames from the T-pose phase on (the pods are on the
body); a window only uses frames of each pod's current boot epoch (a reboot changes the north).

Two variants share this code: **GEO-K** (``truth`` given: the true mountings, so the anatomical axes are known in
pod frames; simulation only) and **GEO-ID** (``truth=None``: axes identified from motion, T-pose frames;
deployable). Ablations: ``use_filter=False`` (each window stands alone; without information, box registration),
``solve_types`` restricted (e.g. hinge only: every other joint keeps its box estimate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..body import JOINTS, N_JOINTS, N_PODS
from ..config import Config
from ..conventions import circ_mean_rad, heading_rad, wrap_rad
from .clock import PodSamples, epoch_starts, map_epoch, pair_samples, pod_samples
from .contract import Estimate, GeoTruth
from .vmfilter import JointFilter

__all__ = ["GeoParams", "GeoDiagnostics", "GeoEstimator", "estimate_geo_k", "estimate_geo_id", "JOINT_TYPES_ALL"]

JOINT_TYPES_ALL = ("hinge", "2dof", "3dof")
_PARENT = np.array([j.parent for j in JOINTS])
_CHILD = np.array([j.child for j in JOINTS])
_JTYPE = np.array([j.jtype for j in JOINTS])


class GeoParams:
    """The estimator's own beliefs: the ``geometry:`` section of the config, and nothing else (brief rule 8)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg.section("geometry")  # works on the full config and on the section alone

    def __getitem__(self, key: str):
        return self.cfg["geometry." + key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def kappa_scale(self, jtype: str) -> float:
        return float(self[f"kappa_scale.{jtype}"])


@dataclass
class GeoDiagnostics:
    """What the estimator did, for the scorer's checks (never needed to produce an estimate)."""

    windows: list = field(default_factory=list)  # per window and joint: T, joint, n_pairs, delta, kappa_raw, status
    pairs: list = field(default_factory=list)  # per window and joint: (T, joint, parent rows, child rows j, u)
    state: dict = field(default_factory=dict)  # estimator-specific state (identified axes, facing, ...)
    seconds: float = 0.0  # wall time of the whole estimate


@dataclass
class _Window:
    """Per-window context handed to the joint solvers."""

    t_s: float  # window end (host time)
    epochs: np.ndarray  # (15,) current boot epoch of every pod (-1: none yet)
    idx: list  # per pod: indices into PodSamples of the window's samples
    tau: list  # per pod: mapped host-clock times of those samples


def _phases(session: dict | None) -> dict[str, tuple[float, float]]:
    if not session:
        return {}
    return {p["name"]: (float(p["t_host_start_s"]), float(p["t_host_end_s"])) for p in session["phases"]}


class GeoEstimator:
    """GEO-K (with ``truth``) or GEO-ID (without), plus the ablation switches."""

    name = "GEO"

    def __init__(self, cfg: Config, truth: GeoTruth | None = None, *, use_filter: bool = True,
                 solve_types: tuple[str, ...] = JOINT_TYPES_ALL):  # fmt: skip
        self.p = GeoParams(cfg)
        self.truth = truth
        self.mode = "K" if truth is not None else "ID"
        self.use_filter = use_filter
        unknown = set(solve_types) - set(JOINT_TYPES_ALL)
        if unknown:
            raise ValueError(f"unknown joint types {sorted(unknown)}")
        self.solve_types = tuple(solve_types)

    # ================================================================== public
    def estimate(self, series: pd.DataFrame, session: dict, t_query_host_s) -> Estimate:
        return self.run(series, session, t_query_host_s)[0]

    def run(self, series: pd.DataFrame, session: dict, t_query_host_s,
            diagnostics: bool = False) -> tuple[Estimate, GeoDiagnostics]:  # fmt: skip
        t_wall = time.perf_counter()
        tq = np.asarray(t_query_host_s, dtype=np.float64)
        diag = GeoDiagnostics()
        self._diag = diag if diagnostics else None
        self.phases = _phases(session)
        self._unrecorded = {p["name"] for p in (session or {}).get("phases", []) if not p.get("recorded", True)}
        self.samples: list[PodSamples] = pod_samples(series)
        self.starts = epoch_starts(series)
        self._box_prepare(series)
        self._session_start()

        t_init = self.phases["box"][1] if "box" in self.phases else self._t_geom_start()
        filt = JointFilter(N_JOINTS, float(self.p["drift_q_rad2_per_s"]))
        events = self._events(t_init, tq)
        last_w = {"t": [], "delta": [], "kappa": []}  # window estimates, for the no-filter ablation
        for t, kind, arg in events:
            if kind == 0:
                mu0, k0 = self._box_init(t)
                filt.init(t, mu0, k0)
            elif kind == 1:
                filt.reset(t, arg)
            else:
                d_w, k_w = self._window(t)
                weight = self.p["step_s"] / self.p["window_s"] if self.use_filter else 1.0
                filt.update(t, d_w, k_w * weight)
                last_w["t"].append(t)
                last_w["delta"].append(d_w)
                last_w["kappa"].append(k_w)

        mu, kap, defined = filt.query(tq)
        if not self.use_filter:
            mu, kap = self._windows_only(tq, filt, last_w)
        # before initialisation: the running box registration, with no confidence
        pre = ~defined | (tq < t_init)
        if np.any(pre):
            mu[pre] = self._box_running(tq[pre])
            kap[pre] = 0.0
        uncertain = kap < float(self.p["kappa_hold"])
        diag.seconds = time.perf_counter() - t_wall
        diag.state.update(self._state_summary())
        return Estimate(wrap_rad(mu), kap, uncertain), diag

    # ================================================================== events
    def _t_geom_start(self) -> float:
        """Geometry uses frames from the T-pose on (the pods are strapped on); without one, from the motion on."""
        for name in ("tpose", "motion"):
            if name in self.phases:
                return self.phases[name][0]
        if "box" in self.phases:
            return self.phases["box"][1]
        firsts = [s[0][1] for s in self.starts if s]
        return min(firsts) if firsts else 0.0

    def _events(self, t_init: float, tq: np.ndarray) -> list[tuple[float, int, object]]:
        """Time-ordered (t, kind, arg): kind 0 init, 1 reset (arg: joints), 2 window."""
        events: list[tuple[float, int, object]] = [(t_init, 0, None)]
        for pod in range(N_PODS):
            for ep, t_first in self.starts[pod]:
                if ep >= 1 and t_first > t_init:
                    joints = np.flatnonzero((_PARENT == pod) | (_CHILD == pod))
                    events.append((t_first, 1, joints))
        if tq.size:
            t0, step = self._t_geom_start(), float(self.p["step_s"])
            k_first = max(1, int(np.ceil((t_init - t0) / step)))
            k_last = int(np.floor((tq.max() - t0) / step + 1e-9))
            t_win = t0 + step * np.arange(k_first, k_last + 1)
            # nothing is recorded inside an unrecorded phase (the simulated idle): no window ends there
            for name, (a, b) in self.phases.items():
                if name in self._unrecorded:
                    t_win = t_win[(t_win <= a) | (t_win > b)]
            events += [(float(t), 2, None) for t in t_win]
        return sorted(events, key=lambda e: (e[0], e[1]))

    def _epochs_at(self, t: float) -> np.ndarray:
        """Current boot epoch of every pod at host time t: the epoch of its latest first-frame at or before t."""
        out = np.full(N_PODS, -1, np.int64)
        for pod in range(N_PODS):
            for ep, t_first in self.starts[pod]:
                if t_first <= t:
                    out[pod] = ep
        return out

    # ================================================================== box registration
    def _box_prepare(self, series: pd.DataFrame) -> None:
        """Per pod: host times and headings of its valid box-phase frames of the first boot epoch (Step 0 §4.6)."""
        self._box = [(np.zeros(0), np.zeros(0)) for _ in range(N_PODS)]
        if "box" not in self.phases:
            return
        a, b = self.phases["box"]
        th = series["t_host_s"].to_numpy()
        sel = (series["received"].to_numpy() & series["valid"].to_numpy() & (series["epoch"].to_numpy() == 0)
               & (th >= a) & (th <= b))  # fmt: skip
        pod = series["pod_id"].to_numpy()
        q = series[["qx", "qy", "qz", "qw"]].to_numpy()
        for i in range(N_PODS):
            idx = np.flatnonzero(sel & (pod == i))
            order = np.argsort(th[idx], kind="mergesort")
            self._box[i] = (th[idx][order], heading_rad(q[idx][order]))

    def _box_headings_at(self, t: float) -> np.ndarray:
        """Circular-mean box heading of every pod over its box frames that arrived by t (NaN without any)."""
        h = np.full(N_PODS, np.nan)
        for i, (th, hd) in enumerate(self._box):
            sel = th <= t
            if np.any(sel):
                h[i] = circ_mean_rad(hd[sel])
        return h

    def _box_delta(self, h: np.ndarray) -> np.ndarray:
        d = wrap_rad(h[_PARENT] - h[_CHILD])
        return np.where(np.isfinite(d), d, 0.0)

    def _box_running(self, t_q: np.ndarray) -> np.ndarray:
        out = np.zeros((t_q.size, N_JOINTS))
        for k, t in enumerate(t_q):
            out[k] = self._box_delta(self._box_headings_at(float(t)))
        return out

    def _box_init(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """B1's estimate at the end of the box phase, with kappa0 = 1/sigma_pair², sigma_pair = √2·sigma_box."""
        h = self._box_headings_at(t)
        self.box_delta_rad = self._box_delta(h)
        sigma_pair = np.sqrt(2.0) * np.deg2rad(float(self.p["box_sigma_deg"]))
        k0 = np.full(N_JOINTS, 1.0 / sigma_pair**2)
        epochs = self._epochs_at(t)
        ok_pod = np.isfinite(h) & (epochs == 0)  # a pod that already rebooted has a new north: no prior
        k0[~(ok_pod[_PARENT] & ok_pod[_CHILD])] = 0.0
        return self.box_delta_rad, k0

    def _windows_only(self, tq: np.ndarray, filt: JointFilter, last_w: dict) -> tuple[np.ndarray, np.ndarray]:
        """No-filter ablation: the latest window's estimate where it carries information, else box registration."""
        mu = np.broadcast_to(getattr(self, "box_delta_rad", np.zeros(N_JOINTS)), (tq.size, N_JOINTS)).copy()
        kap = np.zeros((tq.size, N_JOINTS))
        if last_w["t"]:
            times = np.asarray(last_w["t"])
            idx = np.searchsorted(times, tq, side="right") - 1
            have = idx >= 0
            d = np.stack(last_w["delta"])[np.clip(idx, 0, None)]
            k = np.stack(last_w["kappa"])[np.clip(idx, 0, None)]
            use = have[:, None] & (k > 0)
            mu[use] = d[use]
            kap[use] = k[use]
        return mu, kap

    # ================================================================== windows
    def _window(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Solve every joint over the window ending at t; returns (delta_w, kappa_w) after calibration."""
        w = self._window_context(t)
        d_w, k_w = np.zeros(N_JOINTS), np.zeros(N_JOINTS)
        for j, jt in enumerate(JOINTS):
            if jt.jtype not in self.solve_types:
                continue
            p, c = jt.parent, jt.child
            if w.epochs[p] < 0 or w.epochs[c] < 0:
                continue
            sp, sc = self.samples[p], self.samples[c]
            pairs = pair_samples(w.tau[p], sp.q[w.idx[p]], w.tau[c], sc.q[w.idx[c]], float(self.p["max_pair_gap_s"]))
            delta, kappa_raw, status = self._solve(j, pairs, w)
            d_w[j] = delta
            k_w[j] = kappa_raw * self.p.kappa_scale(self._solver_type(j))
            if self._diag is not None:
                self._diag.windows.append({"t_s": t, "joint": j, "n_pairs": len(pairs), "delta_rad": delta,
                                           "kappa_raw": kappa_raw, "status": status})  # fmt: skip
                self._diag.pairs.append((t, j, sp.row[w.idx[p]][pairs.ip], sc.row[w.idx[c]][pairs.jc],
                                         sc.row[w.idx[c]][np.minimum(pairs.jc + 1, len(w.idx[c]) - 1)], pairs.u))
        return wrap_rad(d_w), k_w

    def _window_context(self, t: float) -> _Window:
        epochs = self._epochs_at(t)
        t0 = self._t_geom_start()
        idx, tau = [], []
        for pod in range(N_PODS):
            if epochs[pod] < 0:
                idx.append(np.zeros(0, np.int64))
                tau.append(np.zeros(0))
                continue
            i, ta = map_epoch(self.samples[pod], int(epochs[pod]), t, float(self.p["clock_window_s"]),
                              float(self.p["clock_block_s"]), t_min_host_s=t0)  # fmt: skip
            keep = ta > t - float(self.p["window_s"])
            idx.append(i[keep])
            tau.append(ta[keep])
        return _Window(t, epochs, idx, tau)

    # ================================================================== solvers (filled in per milestone)
    def _session_start(self) -> None:
        """Reset per-session state (identified axes, signs, facing)."""
        self.state: dict = {}

    def _state_summary(self) -> dict:
        return dict(self.state)

    def _solver_type(self, j: int) -> str:
        """The kappa-calibration class of joint j's solver: its joint type."""
        return str(_JTYPE[j])

    def _solve(self, j: int, pairs, w: _Window) -> tuple[float, float, str]:
        """(delta_w, kappa_raw, status) for joint j over one window. No solver yet: no information."""
        return 0.0, 0.0, "no_solver"


def estimate_geo_k(series: pd.DataFrame, session: dict, t_query_host_s, cfg: Config, *,
                   truth: GeoTruth) -> Estimate:  # fmt: skip
    """GEO-K: known axes and mountings. **Uses simulator truth** (``truth``); never a deployable result."""
    if truth is None:
        raise ValueError("GEO-K needs the true mountings (truth=GeoTruth)")
    return GeoEstimator(cfg, truth).estimate(series, session, t_query_host_s)


def estimate_geo_id(series: pd.DataFrame, session: dict, t_query_host_s, cfg: Config, *, truth=None) -> Estimate:
    """GEO-ID: the deployable estimator. Reads Step 0's series and session.json only."""
    if truth is not None:
        raise ValueError("GEO-ID must not receive simulator truth")
    return GeoEstimator(cfg, None).estimate(series, session, t_query_host_s)
