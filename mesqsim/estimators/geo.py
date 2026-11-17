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
from ..conventions import (
    circ_mean_rad,
    exp_map,
    heading_rad,
    log_map,
    quat_conj,
    quat_mean,
    quat_mul,
    quat_rotate,
    quat_to_matrix,
    wrap_rad,
    yaw_quat,
)
from ..correction import chain_corrections_rad
from .anatomy import CARDAN_SEQ, HINGE_AXES, TWODOF_AXES
from .clock import PodSamples, body_rates, epoch_starts, interp_rates, map_epoch, pair_samples, pod_samples
from .contract import Estimate, GeoTruth
from .hinge import (
    AxisFit,
    axis_in_world,
    hinge_closed_form,
    identify_hinge_axes,
    identify_hinge_axes_tpose,
    vote_tpose,
    vote_vertical,
)
from .rom import UNBOUNDED_DEG, rom_solve
from .twodof import twodof_solve
from .vmfilter import JointFilter

__all__ = ["GeoParams", "GeoDiagnostics", "GeoEvents", "GeoEstimator", "calibrated_kappa", "replay", "estimate_geo_k",
           "estimate_geo_id", "JOINT_TYPES_ALL"]

JOINT_TYPES_ALL = ("hinge", "2dof", "3dof")
_PARENT = np.array([j.parent for j in JOINTS])
_CHILD = np.array([j.child for j in JOINTS])
_JTYPE = np.array([j.jtype for j in JOINTS])
_UPRIGHT = exp_map([np.pi / 2, 0.0, 0.0])  # Rx(90°): the SMPL body frame upright (body y -> world +Z)
_KNEES = tuple(k for k, j in enumerate(JOINTS) if j.jtype == "hinge")


def calibrated_kappa(kappa_stat: np.ndarray, scale_by_type: dict, sigma_by_type_rad: dict) -> np.ndarray:
    """A window's kappa as the filter uses it: scale_type · kappa_stat / (1 + kappa_stat · sigma_m,type²).

    1/kappa_stat is the window's statistical variance, sigma_m its joint type's model error (variances add), and the
    scale corrects for residuals correlated in time; both are fitted per joint type on validation (brief §5.4).
    """
    k = np.asarray(kappa_stat, dtype=np.float64)
    s = np.array([float(scale_by_type[t]) for t in _JTYPE])
    sm2 = np.array([float(sigma_by_type_rad[t]) ** 2 for t in _JTYPE])
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(k > 0, s * k / (1.0 + k * sm2), 0.0)
    return out


@dataclass
class GeoEvents:
    """Everything the filter needs, independent of its own parameters: replaying it with other values of q, the
    kappa scales or the model errors reproduces exactly what the estimator would have output (used for fitting)."""

    t_init: float
    box_delta_rad: np.ndarray  # B1's estimate at the end of the box phase
    events: list  # (t, kind, payload): 0 init (mu0, kappa0); 1 reset (joints); 2 window (delta_w, kappa_stat)


def replay(ev: GeoEvents, t_query_s: np.ndarray, q_rad2_per_s: float, scale_by_type: dict, sigma_by_type_rad: dict,
           weight: float, use_filter: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Belief (mu, kappa) at the query times, and whether it is defined (after the box init)."""
    filt = JointFilter(N_JOINTS, q_rad2_per_s)
    last = {"t": [], "delta": [], "kappa": []}
    for t, kind, payload in ev.events:
        if kind == 0:
            filt.init(t, *payload)
        elif kind == 1:
            filt.reset(t, payload)
        else:
            d_w, k_stat = payload
            k_w = calibrated_kappa(k_stat, scale_by_type, sigma_by_type_rad)
            filt.update(t, d_w, k_w * weight)
            last["t"].append(t)
            last["delta"].append(d_w)
            last["kappa"].append(k_w)
    mu, kap, defined = filt.query(t_query_s)
    if not use_filter:  # each window stands alone; without information, box registration
        mu = np.broadcast_to(ev.box_delta_rad, (t_query_s.size, N_JOINTS)).copy()
        kap = np.zeros((t_query_s.size, N_JOINTS))
        if last["t"]:
            idx = np.searchsorted(np.asarray(last["t"]), t_query_s, side="right") - 1
            d = np.stack(last["delta"])[np.clip(idx, 0, None)]
            k = np.stack(last["kappa"])[np.clip(idx, 0, None)]
            use = (idx >= 0)[:, None] & (k > 0)
            mu[use] = d[use]
            kap[use] = k[use]
    return mu, kap, defined


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

    def scales(self) -> dict:
        return {t: self.kappa_scale(t) for t in JOINT_TYPES_ALL}

    def model_sigmas_rad(self) -> dict:
        return {t: float(np.deg2rad(self[f"model_sigma_deg.{t}"])) for t in JOINT_TYPES_ALL}

    def weight(self, use_filter: bool) -> float:
        return float(self["step_s"]) / float(self["window_s"]) if use_filter else 1.0


@dataclass
class GeoDiagnostics:
    """What the estimator did, for the scorer's checks (never needed to produce an estimate)."""

    windows: list = field(default_factory=list)  # per window and joint: T, joint, n_pairs, delta, kappa_raw, status
    pairs: list = field(default_factory=list)  # per window and joint: (T, joint, parent rows, child rows j, u)
    state: dict = field(default_factory=dict)  # estimator-specific state (identified axes, facing, ...)
    events: GeoEvents | None = None  # the filter's inputs, for replays with other parameters (fitting)
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
        self._filt = filt  # the live belief: read once, at the end of the T-pose (box registration's delta)
        scales, sigmas, weight = self.p.scales(), self.p.model_sigmas_rad(), self.p.weight(self.use_filter)
        log: list = []
        self.box_delta_rad = np.zeros(N_JOINTS)
        for t, kind, arg in self._events(t_init, tq):
            if kind == 0:
                mu0, k0 = self._box_init(t)
                filt.init(t, mu0, k0)
                log.append((t, 0, (mu0, k0)))
            elif kind == 1:
                filt.reset(t, arg)
                log.append((t, 1, arg))
            else:
                d_w, k_stat = self._window(t)
                filt.update(t, d_w, calibrated_kappa(k_stat, scales, sigmas) * weight)
                log.append((t, 2, (d_w, k_stat)))
        events = GeoEvents(t_init, self.box_delta_rad.copy(), log)
        mu, kap, defined = replay(events, tq, float(self.p["drift_q_rad2_per_s"]), scales, sigmas, weight,
                                  self.use_filter)  # fmt: skip
        # before initialisation: the running box registration, with no confidence
        pre = ~defined | (tq < t_init)
        if np.any(pre):
            mu[pre] = self._box_running(tq[pre])
            kap[pre] = 0.0
        uncertain = kap < float(self.p["kappa_hold"])
        diag.seconds = time.perf_counter() - t_wall
        diag.state.update(self._state_summary())
        diag.events = events
        return Estimate(wrap_rad(mu), kap, uncertain), diag

    # ================================================================== events
    def _t_geom_start(self) -> float:
        """Geometry windows use frames from the motion phase on; without one, from the T-pose on.

        The T-pose is a calibration snapshot, not joint mechanics: the wearer holds one posture (a static window
        repeats one configuration ~256 times and would claim a huge kappa), and that posture's error is unrelated to
        the joints' axes (in the simulator 5° per segment, which makes the knee non-hinge: GEO-K's knee was 3.3° off
        with kappa ~1e6 through the T-pose before this rule). The T-pose snapshot itself is still used.
        """
        for name in ("motion", "tpose"):
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

    # ================================================================== windows
    def _window(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Solve every joint over the window ending at t; returns (delta_w, kappa_stat): each solver's statistical
        concentration, before the model error and the scale (:func:`calibrated_kappa`)."""
        w = self._window_context(t)
        if self.state.get("tpose") is None and "tpose" in self.phases and t >= self.phases["tpose"][1]:
            self.state["tpose"] = self._tpose_snapshot()
        d_w, k_w = np.zeros(N_JOINTS), np.zeros(N_JOINTS)
        for j, jt in enumerate(JOINTS):
            if jt.jtype not in self.solve_types:
                continue
            p, c = jt.parent, jt.child
            if w.epochs[p] < 0 or w.epochs[c] < 0:
                continue
            sp, sc = self.samples[p], self.samples[c]
            pairs = pair_samples(w.tau[p], sp.q[w.idx[p]], w.tau[c], sc.q[w.idx[c]], float(self.p["max_pair_gap_s"]))
            self._detail = {}
            delta, kappa_stat, status = self._solve(j, pairs, w)
            d_w[j] = delta
            k_w[j] = kappa_stat
            if self._diag is not None:
                self._diag.windows.append({"t_s": t, "joint": j, "n_pairs": len(pairs), "delta_rad": delta,
                                           "kappa_stat": kappa_stat, "status": status, **self._detail})  # fmt: skip
                self._diag.pairs.append((t, j, sp.row[w.idx[p]][pairs.ip], sc.row[w.idx[c]][pairs.jc],
                                         sc.row[w.idx[c]][np.minimum(pairs.jc + 1, len(w.idx[c]) - 1)], pairs.u))
        return wrap_rad(d_w), k_w

    def _tpose_snapshot(self) -> dict:
        """q_i(T): mean of each pod's fresh T-pose samples of its epoch at the T-pose end (Step 0's snapshot, per
        epoch), and the filter's belief at that moment (box registration plus any T-pose windows)."""
        a, b = self.phases["tpose"]
        epochs = self._epochs_at(b)
        q = np.full((N_PODS, 4), np.nan)
        for i, s in enumerate(self.samples):
            sel = (s.t_host_s >= a) & (s.t_host_s <= b) & (s.epoch == epochs[i])
            if np.any(sel):
                q[i] = quat_mean(s.q[sel])
        return {"q": q, "valid": np.isfinite(q[:, 3]), "epochs": epochs, "mu": self._filt.mu.copy(),
                "kappa": self._filt.kappa.copy(), "t_s": b}  # fmt: skip

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

    # ================================================================== solvers
    def _session_start(self) -> None:
        """Reset per-session state (identified axes, signs, T-pose reference, facing)."""
        self.state: dict = {"axes": {}, "sign_tally": {}, "abs_tally": {}, "tpose": None, "facing_rad": None,
                            "m_hat": None}  # fmt: skip
        self._rate_cache: dict = {}

    def _state_summary(self) -> dict:
        return {"axes": dict(self.state["axes"]), "sign_tally": dict(self.state["sign_tally"]),
                "abs_tally": dict(self.state["abs_tally"]), "facing_rad": self.state["facing_rad"],
                "m_hat": self.state["m_hat"]}  # fmt: skip

    def _solver_type(self, j: int) -> str:
        """The kappa-calibration class of joint j's solver: its joint type."""
        return str(_JTYPE[j])

    def _solve(self, j: int, pairs, w: _Window) -> tuple[float, float, str]:
        """(delta_w, kappa_raw, status) for joint j over one window."""
        jtype = _JTYPE[j]
        if len(pairs) == 0:
            return 0.0, 0.0, "no_pairs"
        if jtype == "hinge":
            return self._hinge_k(j, pairs) if self.mode == "K" else self._hinge_id(j, pairs, w)
        if self.mode == "K":
            m = self.truth.mount_q
            if jtype == "2dof":
                return self._twodof_k(j, pairs)
            return self._rom(j, pairs, m[JOINTS[j].parent], m[JOINTS[j].child])
        # GEO-ID: range of motion at 2-DoF and 3-DoF joints, in T-pose frames (needs the facing)
        m_hat = self.state["m_hat"]
        p, c = JOINTS[j].parent, JOINTS[j].child
        if m_hat is None or not (np.isfinite(m_hat[p, 3]) and np.isfinite(m_hat[c, 3])):
            return 0.0, 0.0, "no_frames"
        return self._rom(j, pairs, m_hat[p], m_hat[c])

    # ------------------------------------------------------------------ 2-DoF (GEO-K) and range of motion
    def _twodof_k(self, j: int, pairs) -> tuple[float, float, str]:
        j1, j2, gamma = TWODOF_AXES[JOINTS[j].name]
        u1, u2 = self._true_axes_in_pods(j, (j1, j2))
        fit = twodof_solve(axis_in_world(pairs.q_p, u1), axis_in_world(pairs.q_c, u2), float(np.cos(gamma)),
                           float(np.sin(np.deg2rad(self.p["twodof_sigma_floor_deg"]))),
                           float(self.p["twodof_ambiguity_llr"]))  # fmt: skip
        self._detail = {"kappa_stat": fit.kappa, "cost": fit.cost, "second_cost": fit.second_cost}
        status = "ambiguous" if fit.ambiguous else ("ok" if fit.kappa > 0 else "flat")
        return fit.delta_rad, fit.kappa, status

    def _rom_limits_rad(self, j: int) -> tuple[np.ndarray, np.ndarray] | None:
        box = self.p.get(f"rom_limits_deg.{JOINTS[j].name}")
        if box is None or np.allclose(box, UNBOUNDED_DEG):
            return None
        b = np.deg2rad(np.asarray(box, dtype=np.float64))
        return b[:, 0], b[:, 1]

    def _rom(self, j: int, pairs, m_p: np.ndarray, m_c: np.ndarray) -> tuple[float, float, str]:
        """Range-of-motion window: J(delta) = (q_p·m_p^-1)^-1·Y(delta)·(q_c·m_c^-1) against the fitted box."""
        lim = self._rom_limits_rad(j)
        if lim is None:
            return 0.0, 0.0, "unbounded"
        a_q = quat_mul(pairs.q_p, quat_conj(m_p))
        b_q = quat_mul(pairs.q_c, quat_conj(m_c))
        fit = rom_solve(a_q, b_q, CARDAN_SEQ[JOINTS[j].name], lim[0], lim[1],
                        float(np.deg2rad(self.p["rom_sigma_deg"])))  # fmt: skip
        self._detail = {"kappa_stat": fit.kappa, "cost": fit.cost_min, "frac_violating": fit.frac_violating}
        return fit.delta_rad, fit.kappa, "ok" if fit.kappa > 0 else "flat"

    # ------------------------------------------------------------------ hinge
    def _hinge_floor_rad(self) -> float:
        return float(np.deg2rad(self.p["hinge_sigma_floor_deg"]))

    def _true_axes_in_pods(self, j: int, seg_axes: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """GEO-K: u = m^-1·u_seg in each pod's frame, with the true mountings m (pod -> segment)."""
        m = self.truth.mount_q
        jt = JOINTS[j]
        return quat_rotate(quat_conj(m[jt.parent]), seg_axes[0]), quat_rotate(quat_conj(m[jt.child]), seg_axes[1])

    def _hinge_k(self, j: int, pairs) -> tuple[float, float, str]:
        u_p, u_c = self._true_axes_in_pods(j, HINGE_AXES[JOINTS[j].name])
        fit = hinge_closed_form(axis_in_world(pairs.q_p, u_p)[:, :2], axis_in_world(pairs.q_c, u_c)[:, :2],
                                self._hinge_floor_rad())  # fmt: skip
        self._detail = {"info": fit.info, "kappa_stat": fit.kappa}
        return fit.delta_rad, fit.kappa, "ok" if fit.kappa > 0 else "blind"

    def _pod_rates(self, pod: int, w: _Window) -> tuple[np.ndarray, np.ndarray]:
        """Body rates of one pod over the identification span (current epoch, from the T-pose on), cached per window."""
        key = (pod, w.t_s)
        if key not in self._rate_cache:
            t_min = max(self._t_geom_start(), w.t_s - float(self.p["axis_span_s"]))
            idx, tau = map_epoch(self.samples[pod], int(w.epochs[pod]), w.t_s, float(self.p["clock_window_s"]),
                                 float(self.p["clock_block_s"]), t_min_host_s=t_min)  # fmt: skip
            self._rate_cache = {k: v for k, v in self._rate_cache.items() if k[1] == w.t_s}
            self._rate_cache[key] = body_rates(tau, self.samples[pod].q[idx], float(self.p["max_pair_gap_s"]))
        return self._rate_cache[key]

    def _identify(self, j: int, w: _Window) -> AxisFit | None:
        """GEO-ID: (re-)identify the hinge axes from the rates since the motion began; keep the last good fit otherwise.

        With a valid T-pose snapshot and box prior, the knee axis is tied to the T-pose (horizontal there, one line in
        both segments: :func:`identify_hinge_axes_tpose`, one unknown), then refined locally with all four angles and
        the refinement kept only if it stays within ``axis_tpose_tolerance_deg``. Without a T-pose (DIODEM), the free
        four-angle fit. Either way the fit must stand out from the alternatives (contrast <= ``axis_contrast_max``):
        when thigh and shin turn together, every axis fits and none is accepted.
        """
        jt = JOINTS[j]
        prev = self.state["axes"].get(j)
        tp, wp = self._pod_rates(jt.parent, w)
        tc, wc = self._pod_rates(jt.child, w)
        ok, wc_at_p = interp_rates(tc, wc, tp, float(self.p["max_pair_gap_s"]))
        wp = wp[ok]
        strong = np.linalg.norm(wp, axis=1) + np.linalg.norm(wc_at_p, axis=1) >= np.deg2rad(
            float(self.p["axis_min_rate_deg_s"]))
        if int(strong.sum()) < int(self.p["axis_min_pairs"]):
            return prev
        wp, wc_at_p = wp[strong], wc_at_p[strong]
        contrast_max = float(self.p["axis_contrast_max"])
        tie = self._tpose_tie(j)
        if tie is not None:
            tied = identify_hinge_axes_tpose(wp, wc_at_p, *tie)
            if tied.contrast > contrast_max:
                return prev
            ref = identify_hinge_axes(wp, wc_at_p, start=(tied.u_p, tied.u_c), n_starts=0)
            tol = np.deg2rad(float(self.p["axis_tpose_tolerance_deg"]))
            if max(_axis_angle(ref.u_p, tied.u_p), _axis_angle(ref.u_c, tied.u_c)) <= tol:
                fit = AxisFit(ref.u_p, ref.u_c, ref.rms_rad_s, ref.rel_residual, ref.n, tied.contrast, True)
            else:
                scale = float(np.sqrt(np.mean(0.5 * (np.sum(wp**2, axis=1) + np.sum(wc_at_p**2, axis=1)))))
                fit = AxisFit(tied.u_p, tied.u_c, tied.rms_rad_s, tied.rms_rad_s / scale if scale > 0 else np.inf,
                              tied.n, tied.contrast, True)  # fmt: skip
        else:
            start = (prev.u_p, prev.u_c) if prev is not None else None
            fit = identify_hinge_axes(wp, wc_at_p, start=start,
                                      n_starts=2 if prev is not None else int(self.p["axis_starts"]))  # fmt: skip
            if fit.contrast > contrast_max:
                return prev
        if fit.rel_residual > float(self.p["axis_max_residual"]):
            return prev
        if prev is not None:
            moved = max(_axis_angle(fit.u_p, prev.u_p), _axis_angle(fit.u_c, prev.u_c))
            if moved > np.deg2rad(30.0):  # a jump: the sign bookkeeping no longer refers to the same axes
                self.state["sign_tally"][j] = 0
            if not fit.tied:
                from .hinge import align_sign

                fit = AxisFit(align_sign(fit.u_p, prev.u_p), align_sign(fit.u_c, prev.u_c), fit.rms_rad_s,
                              fit.rel_residual, fit.n, fit.contrast, False)  # fmt: skip
        self.state["axes"][j] = fit
        return fit

    def _tpose_tie(self, j: int) -> tuple[np.ndarray, np.ndarray] | None:
        """(g_p, m_rel): world up in the parent pod frame at the T-pose, and the rotation taking the parent pod frame
        to the child's at the T-pose with box registration's delta_T, or None without a valid snapshot and prior."""
        tp = self.state.get("tpose")
        jt = JOINTS[j]
        if tp is None or not (tp["valid"][jt.parent] and tp["valid"][jt.child]) or tp["kappa"][j] <= 0:
            return None
        q_p, q_c = tp["q"][jt.parent], tp["q"][jt.child]
        g_p = quat_rotate(quat_conj(q_p), [0.0, 0.0, 1.0])
        rel = quat_mul(quat_mul(quat_conj(q_c), yaw_quat(-tp["mu"][j])), q_p)
        return g_p, quat_to_matrix(rel)

    def _tpose_ref(self, j: int) -> np.ndarray | None:
        """q_p(T)^-1·Y(mu_T)·q_c(T): the pod-to-pod rotation in the T-pose, if both pods have a snapshot and the
        joint had a box prior at the T-pose (its mu_T is box registration's delta)."""
        tp = self.state.get("tpose")
        jt = JOINTS[j]
        if tp is None or not (tp["valid"][jt.parent] and tp["valid"][jt.child]) or tp["kappa"][j] <= 0:
            return None
        return quat_mul(quat_mul(quat_conj(tp["q"][jt.parent]), yaw_quat(tp["mu"][j])), tp["q"][jt.child])

    def _hinge_id(self, j: int, pairs, w: _Window) -> tuple[float, float, str]:
        axes = self._identify(j, w)
        if axes is None:
            return 0.0, 0.0, "no_axes"
        a3, b3 = axis_in_world(pairs.q_p, axes.u_p), axis_in_world(pairs.q_c, axes.u_c)
        fit = hinge_closed_form(a3[:, :2], b3[:, :2], self._hinge_floor_rad())
        votes = vote_vertical(a3, b3, float(np.sin(np.deg2rad(self.p["sign_min_elev_deg"]))) ** 2)
        ref = self._tpose_ref(j)
        if ref is not None and fit.kappa > 0:
            votes += vote_tpose(pairs.q_p, pairs.q_c, fit.delta_rad, ref, np.deg2rad(self.p["tpose_sign_margin_deg"]))
        tally = self.state["sign_tally"].get(j, 0) + votes
        if axes.tied:  # u_c was derived from u_p through the T-pose: the relative sign is right by construction
            tally = max(tally, 1)
        self.state["sign_tally"][j] = tally
        self._detail = {"info": fit.info, "kappa_stat": fit.kappa, "u_p": axes.u_p, "u_c": axes.u_c,
                        "sign_tally": tally}  # fmt: skip
        if tally == 0:
            return fit.delta_rad, 0.0, "ambiguous"
        delta = fit.delta_rad if tally > 0 else float(wrap_rad(fit.delta_rad + np.pi))
        if ref is not None and fit.kappa > 0:
            self._vote_knee_direction(j, pairs, delta, axes.u_p, ref)
        return delta, fit.kappa, "ok" if fit.kappa > 0 else "blind"

    # ------------------------------------------------------------------ facing and T-pose frames (GEO-ID)
    def _vote_knee_direction(self, j: int, pairs, delta: float, u_p: np.ndarray, j_tpose: np.ndarray) -> None:
        """Which way u_p points: the knee only flexes one way (Step 2 brief §5.4 Should).

        K = q_p^-1·Y(delta)·q_c·j_tpose^-1 is the shin's rotation since the T-pose in the thigh pod's frame; for a
        hinge its rotation vector is theta·(the axis pointing to the body's left), theta > 0 in flexion. The window
        votes for the sign of the median Log(K)·u_p over samples flexed more than ``facing_flex_min_deg``.
        """
        jk = quat_mul(quat_mul(quat_conj(pairs.q_p), yaw_quat(np.full(len(pairs), delta))), pairs.q_c)
        rv = log_map(quat_mul(jk, quat_conj(j_tpose)))
        flexed = np.linalg.norm(rv, axis=1) >= np.deg2rad(float(self.p["facing_flex_min_deg"]))
        if np.any(flexed):
            vote = int(np.sign(np.median(rv[flexed] @ u_p)))
            self.state["abs_tally"][j] = self.state["abs_tally"].get(j, 0) + vote
        self._update_facing()

    def _update_facing(self) -> None:
        """Facing from the knees: in the T-pose, each knee axis (pointing left) is the body's left-right axis.

        With the corrections at the T-pose (box registration), Y(c_thigh(T))·q_thigh(T)·u_left is the body's left
        direction in the pelvis pod's north; its heading is the facing phi (the heading of the body's left axis).
        Then F_T = Y(phi)·Rx(90°) and every pod's segment frame from the T-pose is m_hat_i = F_T^-1·Y(c_i(T))·q_i(T).
        """
        tp = self.state["tpose"]
        if tp is None:
            return
        corr = chain_corrections_rad(tp["mu"])
        ok_chain = self._chain_valid(tp)
        phis = []
        for j in _KNEES:
            axes, tally = self.state["axes"].get(j), self.state["abs_tally"].get(j, 0)
            thigh = JOINTS[j].parent
            if axes is None or tally == 0 or not ok_chain[thigh]:
                continue
            left = axes.u_p * np.sign(tally)
            d = quat_rotate(quat_mul(yaw_quat(corr[thigh]), tp["q"][thigh]), left)
            phis.append(float(np.arctan2(d[1], d[0])))
        if not phis:
            return
        phi = float(circ_mean_rad(np.array(phis)))
        f_t = quat_mul(yaw_quat(phi), _UPRIGHT)
        m_hat = np.full((N_PODS, 4), np.nan)
        for i in range(N_PODS):
            if ok_chain[i]:
                m_hat[i] = quat_mul(quat_conj(f_t), quat_mul(yaw_quat(corr[i]), tp["q"][i]))
        self.state["facing_rad"] = phi
        self.state["m_hat"] = m_hat

    @staticmethod
    def _chain_valid(tp: dict) -> np.ndarray:
        """Pods whose T-pose snapshot exists and whose whole chain from the pelvis had a box prior at the T-pose."""
        ok = np.zeros(N_PODS, bool)
        root = [p for p in range(N_PODS) if all(jt.child != p for jt in JOINTS)][0]
        ok[root] = bool(tp["valid"][root])
        for k, jt in enumerate(JOINTS):  # parents come first
            ok[jt.child] = ok[jt.parent] and bool(tp["valid"][jt.child]) and tp["kappa"][k] > 0
        return ok


def _axis_angle(u: np.ndarray, v: np.ndarray) -> float:
    """Angle between two axes regardless of sign, in [0, pi/2]."""
    c = abs(float(np.dot(u, v)) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.arccos(np.clip(c, 0.0, 1.0)))


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
