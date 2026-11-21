"""The numbers to beat (Step 3 brief §8): the baselines, the oracles and RIGID on the twin's held-out sessions, the
ablations, and the page-level test. ``scripts/make_results_step3.py`` writes ``docs/RESULTS_STEP3.md`` from them.

**The sessions.** Each test session of a dataset (``twin_v1``) is simulated again from its seed and stored conditions
(:func:`mesqsim.learn.store.build_session`, which the dataset's own check proves gives the stored bytes), so the
methods that need the raw frames (B1, GEO-ID, RIGID) see what the stored inputs came from.

**The methods** (δ per joint; ε per pod, composed along the tree, ε_c = ε_p + δ_pc, :func:`mesqsim.learn.rigid.chain`):

=============  =====================================================================================  ===========
``B0``         δ̂ = 0; ε̂ = 0                                                                         no truth
``B1``         Step 0's box registration (constant); ε̂: the chain from ε̂_HipsAlt = 0                no truth
``O1``         the true offsets of the current boot epochs; ε̂: the chain from the true ε_HipsAlt     truth
``O2``         the true δ and ε                                                                       truth
``GEO-ID``     Step 2's deployable estimator (δ only)                                                 no truth
``RIGID+B1``   ε̂_HipsAlt by RIGID's causal track (phone and HipsAlt on the pelvis), then B1's δ̂      no truth
``RIGID+GEO``  the same root, then GEO-ID's δ̂, with κ along the chain                               no truth
=============  =====================================================================================  ===========

GEO-K is not run on the twin: its "true mounting" is a constant pod-to-segment rotation, and the twin's pods sit on
moving skin (§4.2), so the twin has no such constant to give it. Without HipsAlt (a tree that disables it) nothing
links the phone to the pelvis pods: RIGID cannot run, the chains start from 0, and O1's ε is not defined.

**The metrics** (§8.2): |wrap(θ̂ - θ)| in degrees on valid steps outside ``transition`` blends, pooled over steps and
units, as mergeable histograms (:class:`Agg`: 0.02° bins to 30°, 0.5° bins to 180°, so a median or P95 is exact to
that bin); per joint type or pod and per stratum. The unit of the statistics is the held-out subject (§8.5): each
subject's mean error per method, paired contrasts by a bootstrap over subjects, Holm across the contrasts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .body import JOINTS, PODS, ROOT_POD
from .conventions import vm_halfwidth_rad, wrap_rad

__all__ = ["EDGES", "Agg", "DELTA_METHODS", "EPS_METHODS", "ABLATIONS", "test_rows", "session_estimates",
           "session_result", "merge", "tables", "contrasts"]

EDGES = np.r_[np.arange(0.0, 30.0, 0.02), np.arange(30.0, 180.0 + 1e-9, 0.5)]
DELTA_METHODS = ("B0", "B1", "O1", "O2", "GEO-ID")
EPS_METHODS = ("B0", "B1", "O1", "O2", "RIGID+B1", "RIGID+GEO")
USES_TRUTH = frozenset({"O1", "O2"})
ABLATIONS = {"segment frames": {"twin.body.pod_frame": "segment"}, "analytic fusion": {"twin.fusion.mode": "analytic"}}
"""§8.1's twin ablations: Step 1's segment frames instead of surface frames; Step 1's analytic heading model instead
of VQF in the loop. The baselines and oracles only (GEO-ID is run on the twin as calibrated)."""
BOOT5_MIN = np.arange(0.0, 40.0, 5.0)
BOOT_STRATA = (0.0, 5.0, 15.0, 35.0, np.inf)
JTYPES = sorted({j.jtype for j in JOINTS})


class Agg:
    """A mergeable summary of absolute errors (degrees): histogram over :data:`EDGES`, sum and count."""

    __slots__ = ("h", "s", "n")

    def __init__(self):
        self.h = np.zeros(EDGES.size, np.int64)  # the last bin holds 180° exactly
        self.s, self.n = 0.0, 0

    def add(self, e: np.ndarray) -> None:
        e = np.asarray(e, dtype=np.float64).ravel()
        if e.size:
            self.h += np.bincount(np.clip(np.searchsorted(EDGES, e, side="right") - 1, 0, EDGES.size - 1),
                                  minlength=EDGES.size)  # fmt: skip
            self.s += float(e.sum())
            self.n += int(e.size)

    def merge(self, other: Agg) -> None:
        self.h += other.h
        self.s += other.s
        self.n += other.n

    def quantile(self, q: float) -> float:
        if self.n == 0:
            return float("nan")
        c = np.cumsum(self.h)
        k = int(np.searchsorted(c, q * self.n, side="left"))
        hi = EDGES[k + 1] if k + 1 < EDGES.size else 180.0
        return float(0.5 * (EDGES[k] + hi))

    def stats(self) -> dict:
        return {"n": self.n, "mean_deg": self.s / self.n if self.n else float("nan"),
                "median_deg": self.quantile(0.5), "p95_deg": self.quantile(0.95)}  # fmt: skip


# ------------------------------------------------------------------------------------------------ the sessions
def test_rows(root: Path, split: str = "test", max_sessions: int | None = None) -> list[tuple[dict, dict]]:
    """(index row, stored conditions) of each session of ``split``, in id order."""
    import h5py

    idx = pd.read_parquet(root / "index.parquet")
    idx = idx[idx["split"] == split].sort_values("id")
    if max_sessions:
        idx = idx.head(max_sessions)
    out = []
    for r in idx.to_dict("records"):
        with h5py.File(root / r["shard"], "r") as h5:
            cond = json.loads(h5[f"sessions/{r['id']}"].attrs["conditions"])
        lm = r.get("length_min")
        r["length_min"] = None if lm is None or (isinstance(lm, float) and math.isnan(lm)) else float(lm)
        out.append((r, {"twin": cond["twin"], "step1": cond["step1"], "suit": cond["twin"]["twin.suit.version"],
                        "hipsalt": cond["hipsalt"], "protocol": cond["program"].startswith("protocol")}))
    return out


def session_estimates(built: dict, cfg1, geo: bool = True) -> tuple[dict, dict]:
    """Every method's (θ̂, κ̂ or None) on one regenerated session: ``(delta, eps)``, each method -> (T, 14) / (T, 15)."""
    from .evaluate import grid_from_labels, method_estimate
    from .learn.rigid import chain, chain_kappa, rigid_track
    from .step0 import decode

    s, f, arr = built["session"], built["features"], built["arrays"]
    grid = grid_from_labels(s.labels_joints)
    if grid.t_s.size != arr["t_s"].size:
        raise ValueError("the label grid and the stored steps differ")
    series = decode(s.frames, s.session)
    delta = {m: (np.asarray(method_estimate(m, grid, series, s.session, s.meta, cfg1)[0], float), None)
             for m in ("B0", "B1", "O1", "O2")}  # fmt: skip
    if geo:
        dh, k, _ = method_estimate("GEO-ID", grid, series, s.session, s.meta, cfg1)
        delta["GEO-ID"] = (np.asarray(dh, float), np.asarray(k, float))
    has_alt = bool(np.isfinite(f.q_pod[:, ROOT_POD, 3]).any())
    if has_alt:
        eps_r, kap_r = rigid_track(f.q_phone, f.q_pod[:, ROOT_POD])
    else:
        eps_r, kap_r = np.zeros(grid.t_s.size), np.zeros(grid.t_s.size)
    y_eps = arr["y_eps"].astype(float)
    b1 = np.nan_to_num(delta["B1"][0])
    eps = {"B0": (np.zeros(y_eps.shape), None), "B1": (chain(0.0, b1), None), "O2": (y_eps, None),
           "RIGID+B1": (chain(eps_r, b1), None)}  # fmt: skip
    eps["O1"] = (chain(y_eps[:, ROOT_POD], np.nan_to_num(delta["O1"][0])) if has_alt
                 else np.full(y_eps.shape, np.nan), None)  # fmt: skip
    if geo:
        g, kg = np.nan_to_num(delta["GEO-ID"][0]), np.nan_to_num(delta["GEO-ID"][1])
        eps["RIGID+GEO"] = (chain(eps_r, g), chain_kappa(kap_r, kg))
    return delta, eps


def _strata(built: dict) -> dict:
    """Per step: the stratum value of each stratum kind (as the scorer's, plus 5-minute time-since-boot bins)."""
    from .learn.store import MOTION_CODES, SLIP_BITS

    arr, attrs = built["arrays"], built["attrs"]
    cond = json.loads(attrs["conditions"])
    t = arr["t_s"].size
    boot = arr["strata/time_since_boot_s"] / 60.0
    names = {v: k for k, v in MOTION_CODES.items()}
    lab = [f"{BOOT_STRATA[i]:g}-{BOOT_STRATA[i + 1]:g}" for i in range(len(BOOT_STRATA) - 1)]
    b5 = [f"{x:g}-{x + 5:g}" for x in BOOT5_MIN]
    return {"since_boot_min": np.array(lab)[np.clip(np.searchsorted(BOOT_STRATA, boot, side="right") - 1, 0,
                                                    len(lab) - 1)],
            "boot_5min": np.array(b5)[np.clip((boot // 5).astype(int), 0, len(b5) - 1)],
            "motion": np.array([names.get(int(m), "other") for m in arr["strata/motion"]]),
            "slip": np.array(["+".join(n for n, b in SLIP_BITS.items() if s & b) or "none"
                              for s in arr["strata/slip"]]),
            "suit": np.full(t, attrs["suit_version"]), "wire": np.full(t, cond["wire"]),
            "hipsalt": np.full(t, str(cond["hipsalt"]))}  # fmt: skip


def _accumulate(out: dict, target: str, method: str, err: np.ndarray, valid: np.ndarray, kappa, strata: dict,
                unit_names, subject: str) -> None:  # fmt: skip
    """Add one session's errors of one method to the aggregates."""

    def agg(key):
        if key not in out["aggs"]:
            out["aggs"][key] = Agg()
        return out["aggs"][key]

    e_all = err[valid]
    agg((target, method, "all", "all", "all")).add(e_all)
    for u, name in enumerate(unit_names):
        agg((target, method, "all", "all", name)).add(err[valid[:, u], u])
    if target == "delta":
        for jt in JTYPES:
            sel = np.array([j.jtype == jt for j in JOINTS])
            agg((target, method, "all", "all", f"type:{jt}")).add(err[:, sel][valid[:, sel]])
    rows_valid = valid.any(axis=1)
    for kind, vals in strata.items():
        for v in np.unique(vals[rows_valid]):
            m = (vals == v)[:, None] & valid
            agg((target, method, kind, str(v), "all")).add(err[m])
    key = (target, method)
    s, n = out["subject"].get(key, (0.0, 0))
    out["subject"][key] = (s + float(e_all.sum()), n + int(e_all.size))
    if kappa is not None:
        k = np.asarray(kappa, float)[valid]
        c68 = int(np.sum(e_all <= np.rad2deg(vm_halfwidth_rad(k, 0.68))))
        c95 = int(np.sum(e_all <= np.rad2deg(vm_halfwidth_rad(k, 0.95))))
        a, b, c = out["coverage"].get(key, (0, 0, 0))
        out["coverage"][key] = (a + c68, b + c95, c + int(e_all.size))


def session_result(args) -> dict:
    """One session (a process-pool task): regenerate it (with an ablation's overrides, if any) and aggregate every
    method's errors. ``args = (row, cond, ablation_name or None)``."""
    from .config import load_config
    from .learn.store import build_session
    from .paths import REPO_ROOT
    from .twin.config import load_twin_config

    row, cond, ablation = args
    cfgt, cfg1 = load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml")
    c = dict(cond)
    if ablation:
        c["twin"] = dict(cond["twin"]) | ABLATIONS[ablation]
    built = build_session(row, c, cfgt, cfg1, row.get("length_min"), keep_session=True)
    delta, eps = session_estimates(built, cfg1, geo=ablation is None)
    arr = built["arrays"]
    trans = arr["strata/transition"].astype(bool)[:, None]
    strata = _strata(built)
    subject = built["attrs"]["subject"]
    out = {"aggs": {}, "subject": {}, "coverage": {}, "id": row["id"], "subject_id": subject,
           "steps": int(arr["t_s"].size), "ablation": ablation}  # fmt: skip
    vd, ve = arr["valid_delta"] & ~trans, arr["valid_eps"] & ~trans
    for m, (th, ka) in delta.items():
        err = np.rad2deg(np.abs(wrap_rad(np.nan_to_num(th) - arr["y_delta"])))
        _accumulate(out, "delta", m, err, vd & np.isfinite(th), ka, strata, [j.name for j in JOINTS], subject)
    for m, (th, ka) in eps.items():
        ok = ve & np.isfinite(th)
        err = np.rad2deg(np.abs(wrap_rad(np.nan_to_num(th) - arr["y_eps"])))
        _accumulate(out, "eps", m, err, ok, ka, strata, [p.name for p in PODS], subject)
    return out


def merge(results: list[dict]) -> dict:
    """Pool the sessions' aggregates: per ablation (None = the calibrated twin)."""
    out: dict = {}
    for r in results:
        o = out.setdefault(r["ablation"], {"aggs": {}, "subject": {}, "coverage": {}, "sessions": 0, "steps": 0,
                                           "subjects": set()})  # fmt: skip
        o["sessions"] += 1
        o["steps"] += r["steps"]
        o["subjects"].add(r["subject_id"])
        for k, a in r["aggs"].items():
            if k in o["aggs"]:
                o["aggs"][k].merge(a)
            else:
                o["aggs"][k] = a
        for k, (s, n) in r["subject"].items():
            o["subject"].setdefault(k, {}).setdefault(r["subject_id"], [0.0, 0])
            o["subject"][k][r["subject_id"]][0] += s
            o["subject"][k][r["subject_id"]][1] += n
        for k, v in r["coverage"].items():
            a = o["coverage"].get(k, (0, 0, 0))
            o["coverage"][k] = tuple(x + y for x, y in zip(a, v, strict=True))
    return out


def tables(pooled: dict) -> pd.DataFrame:
    """One long table: target, method, stratum, value, unit, n, mean, median, P95."""
    rows = [{"target": k[0], "method": k[1], "stratum": k[2], "value": k[3], "unit": k[4]} | a.stats()
            for k, a in sorted(pooled["aggs"].items())]  # fmt: skip
    return pd.DataFrame(rows)


def contrasts(pooled: dict, pairs, n_boot: int, seed: int) -> pd.DataFrame:
    """Paired contrasts of the per-subject mean error (a - b, degrees): bootstrap over subjects, Holm across them."""
    from .results_step2 import bootstrap_contrast, holm

    rows = []
    for target, a, b in pairs:
        sa, sb = pooled["subject"].get((target, a), {}), pooled["subject"].get((target, b), {})
        subs = sorted(set(sa) & set(sb))
        d = np.array([sa[s][0] / sa[s][1] - sb[s][0] / sb[s][1] for s in subs if sa[s][1] and sb[s][1]])
        rows.append({"target": target, "a": a, "b": b} | bootstrap_contrast(d, n_boot, seed))
    df = pd.DataFrame(rows)
    if len(df):
        adj = holm({i: p for i, p in enumerate(df["p"]) if np.isfinite(p)})
        df["p_holm"] = [adj.get(i, np.nan) for i in range(len(df))]
    return df
