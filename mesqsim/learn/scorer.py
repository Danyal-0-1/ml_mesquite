"""The scorer (Step 3 brief §7.4, §8.2): ``python -m mesqsim score-model --model <module>:<Class> --dataset twin_v1``.

It loads any model that meets :mod:`.contract`, runs it on every session of a split, checks each prediction and
the model's causality, and writes the tables of §8.2:

* **Primary.** |wrap(θ̂ - θ)| in degrees per joint (δ) or pod (ε) and 32 Hz step, on valid steps, leaving out
  ``transition`` blends: mean, median and P95.
* **Derived.** A per-pod model is also scored on the joints, with δ̂_pc = ε̂_c - ε̂_p, so both kinds meet on one scale.
* **Confidence.** Coverage of the 68 % and 95 % von Mises intervals of κ̂ (:func:`mesqsim.conventions.vm_halfwidth_rad`);
  a reliability table (the share of errors inside the central interval at nominal levels 10 ... 95 %, the data of a
  reliability diagram); and selective risk: the median error of the steps kept when only the most confident 100,
  80, 50 and 20 % are kept.
* **Compute.** The model's time per 32 Hz step for all its joints or pods, on this machine.
* **Strata.** Per joint or pod and joint type; time since boot (0-5, 5-15, 15-35 min); motion type; slip; suit
  version; wire mode; HipsAlt present or not.

Built-in references on the dataset: ``B0`` (δ̂ = 0) and ``O2`` (the true targets, uses truth). The Step 1 and Step 2
baselines that need the raw frames (B1, GEO-ID, RIGID + B1) are scored on twin sessions by S3-M9's results script.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..body import JOINTS, PODS
from ..conventions import vm_halfwidth_rad, wrap_rad
from .contract import SessionInputs, causality_error, check_prediction, outputs_for
from .store import MOTION_CODES, SLIP_BITS

__all__ = ["load_model", "iterate_split", "score", "reliability", "cpu_name", "SELECTIVE", "RELIABILITY_LEVELS"]

SELECTIVE = (1.0, 0.8, 0.5, 0.2)
RELIABILITY_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.7, 0.8, 0.9, 0.95)
BOOT_EDGES_MIN = (0.0, 5.0, 15.0, 35.0, np.inf)


def cpu_name() -> str:
    """The CPU the timings were measured on (§7.4: "on a named CPU")."""
    import platform
    import subprocess

    try:
        name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        name = ""
    return name or platform.processor() or platform.machine()


def reliability(err_deg: np.ndarray, kappa: np.ndarray) -> pd.DataFrame:
    """The share of errors inside the central von Mises interval of κ̂, per nominal level: calibrated when equal."""
    rows = []
    for lev in RELIABILITY_LEVELS:
        hw = np.rad2deg(vm_halfwidth_rad(kappa, lev))
        rows.append({"nominal": lev, "empirical": float(np.mean(err_deg <= hw)), "n": int(err_deg.size)})
    return pd.DataFrame(rows)


def load_model(spec: str):
    """``module:Class`` -> an instance (constructed without arguments)."""
    mod, _, cls = spec.partition(":")
    return getattr(importlib.import_module(mod), cls)()


def iterate_split(root: Path, split: str, max_sessions: int | None = None):
    """Yield (index row, inputs, targets, strata) for each stored session of ``split``."""
    import h5py

    from .targets import read_targets

    idx = pd.read_parquet(root / "index.parquet")
    idx = idx[idx["split"] == split].sort_values("id")
    if max_sessions:
        idx = idx.head(max_sessions)
    for shard, g in idx.groupby("shard", sort=False):
        with h5py.File(root / shard, "r") as h5:
            for r in g.itertuples():
                grp = h5[f"sessions/{r.id}"]
                inputs = SessionInputs.from_group(grp, source=f"{root / shard}#{r.id}")
                strata = {k: np.asarray(grp[f"strata/{k}"]) for k in ("time_since_boot_s", "motion", "transition",
                                                                        "slip")}  # fmt: skip
                yield r, inputs, read_targets(grp), strata


def _rows(err_deg, valid, kappa, unit_names, strata_cols: dict) -> list[dict]:
    """Per stratum value and unit: n, mean, median, P95, coverage 68 / 95 %."""
    out = []
    hw68 = np.rad2deg(vm_halfwidth_rad(np.asarray(kappa, float), 0.68))
    hw95 = np.rad2deg(vm_halfwidth_rad(np.asarray(kappa, float), 0.95))
    for skind, svals in strata_cols.items():
        for sval in np.unique(svals):
            sel_t = svals == sval
            for u, name in enumerate(unit_names):
                m = valid[:, u] & sel_t
                if not m.any():
                    continue
                e = err_deg[m, u]
                out.append({"stratum": skind, "value": str(sval), "unit": name, "n": int(m.sum()),
                            "mean_deg": float(e.mean()), "median_deg": float(np.median(e)),
                            "p95_deg": float(np.percentile(e, 95)),
                            "cov68": float(np.mean(e <= hw68[m, u])), "cov95": float(np.mean(e <= hw95[m, u]))})
    return out


def _pooled(err, valid, kappa, unit: str) -> dict:
    e, k_ = err[valid], kappa[valid]
    return {"stratum": "all", "value": "all", "unit": unit, "n": int(e.size), "mean_deg": float(e.mean()),
            "median_deg": float(np.median(e)), "p95_deg": float(np.percentile(e, 95)),
            "cov68": float(np.mean(e <= np.rad2deg(vm_halfwidth_rad(k_, 0.68)))),
            "cov95": float(np.mean(e <= np.rad2deg(vm_halfwidth_rad(k_, 0.95))))}  # fmt: skip


def score(model, root: Path, split: str = "val", max_sessions: int | None = None, causal_steps: int = 3) -> dict:
    """Score ``model`` (or the built-in ``"B0"`` / ``"O2"``) on ``split``; returns the tables and a summary."""
    builtin = model if isinstance(model, str) else None
    target = "delta" if builtin else model.target
    k = outputs_for(target)
    units = [j.name for j in JOINTS] if target == "delta" else [p.name for p in PODS]
    jtype = np.array([j.jtype for j in JOINTS])
    motion_name = {v: n for n, v in MOTION_CODES.items()}
    errs, valids, kappas, strata_all, derived, t_pred, steps, causal = [], [], [], [], [], 0.0, 0, 0.0
    for _row, inputs, y, st in iterate_split(root, split, max_sessions):
        truth = y["y_delta"] if target == "delta" else y["y_eps"]
        valid = (y["valid_delta"] if target == "delta" else y["valid_eps"]) & ~st["transition"][:, None]
        if builtin == "B0":
            th, ka = np.zeros(truth.shape), np.zeros(truth.shape)
        elif builtin == "O2":
            th, ka = truth.astype(float), np.full(truth.shape, 1e3)
        else:
            t0 = time.perf_counter()
            pred = model.predict(inputs)
            t_pred += time.perf_counter() - t0
            check_prediction(model, inputs, pred)
            th, ka = np.asarray(pred.theta_rad, float), np.asarray(pred.kappa, float)
            if causal_steps:
                pick = np.linspace(inputs.steps // 4, inputs.steps - 1, causal_steps).astype(int)
                causal = max(causal, causality_error(model, inputs, pick))
        steps += inputs.steps
        errs.append(np.rad2deg(np.abs(wrap_rad(th - truth))))
        valids.append(valid)
        kappas.append(ka)
        cond = json.loads(inputs.attrs.get("conditions", "{}"))
        boot_min = st["time_since_boot_s"] / 60.0
        strata_all.append({"all": np.full(inputs.steps, "all"),
                           "since_boot_min": np.array([f"{BOOT_EDGES_MIN[i]:g}-{BOOT_EDGES_MIN[i + 1]:g}" for i in
                                                       np.searchsorted(BOOT_EDGES_MIN, boot_min, side="right") - 1]),
                           "motion": np.array([motion_name.get(int(m), "other") for m in st["motion"]]),
                           "slip": np.array(["+".join(n for n, b in SLIP_BITS.items() if s & b) or "none"
                                             for s in st["slip"]]),
                           "suit": np.full(inputs.steps, inputs.attrs.get("suit_version", "")),
                           "wire": np.full(inputs.steps, cond.get("wire", "")),
                           "hipsalt": np.full(inputs.steps, str(cond.get("hipsalt", True)))})  # fmt: skip
        if target == "eps":  # δ̂_pc = ε̂_c - ε̂_p on the joint metric
            d_hat = np.stack([wrap_rad(th[:, j.child] - th[:, j.parent]) for j in JOINTS], axis=1)
            d_err = np.rad2deg(np.abs(wrap_rad(d_hat - y["y_delta"])))
            derived.append((d_err, y["valid_delta"] & ~st["transition"][:, None]))
    if not errs:
        raise ValueError(f"no sessions of split {split!r} under {root}")
    err, valid, ka = np.vstack(errs), np.vstack(valids), np.vstack(kappas)
    strata = {s: np.concatenate([d[s] for d in strata_all]) for s in strata_all[0]}
    rows = _rows(err, valid, ka, units, strata)
    if target == "delta":  # joint types: every valid step of every joint of the type, pooled
        for t_ in np.unique(jtype):
            sel = jtype == t_
            rows.append(_pooled(err[:, sel], valid[:, sel], ka[:, sel], f"type:{t_}"))
    table = pd.DataFrame(rows)
    flat = err[valid]
    order = np.argsort(-ka[valid], kind="mergesort")
    selective = {f"{int(100 * f)}%": float(np.median(flat[order[: max(1, int(f * flat.size))]])) for f in SELECTIVE}
    summary = {"target": target, "split": split, "steps": int(steps), "units": k,
               "median_deg": float(np.median(flat)), "mean_deg": float(flat.mean()),
               "p95_deg": float(np.percentile(flat, 95)), "selective_median_deg": selective,
               "seconds_per_step": (t_pred / steps) if steps and not builtin else None,
               "causality_error_rad": causal if not builtin else None}  # fmt: skip
    out = {"summary": summary, "table": table, "reliability": reliability(flat, ka[valid])}
    if derived:
        d_err = np.vstack([d[0] for d in derived])
        d_val = np.vstack([d[1] for d in derived])
        out["derived_joint_table"] = pd.DataFrame(_rows(d_err, d_val, np.zeros_like(d_err),
                                                        [j.name for j in JOINTS], {"all": strata["all"]}))
        out["summary"]["derived_joint_median_deg"] = float(np.median(d_err[d_val]))
    return out
