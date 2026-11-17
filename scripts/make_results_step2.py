"""Reproduce docs/RESULTS_STEP2.md and its figures from one command (Step 2 brief §6, §7 S2-M6).

    python scripts/make_results_step2.py                      # AMASS test split if the split manifest exists
    python scripts/make_results_step2.py --source synthetic   # synthetic stand-ins: PROVISIONAL

1. The dataset's **test** sessions (windows x motion starts 1, 10, 30 min; the dataset's own seeds), every method:
   B0, B1, O1, O2, GEO-K (uses simulator truth), GEO-ID and its ablations, scored in memory.
2. Conditions: one-at-a-time sweeps around the default (motion at 10 min, one window per sequence; the same seeds for
   every value), for B1, O1, O2, GEO-K and GEO-ID; re-acquisition after reboots.
3. DIODEM: GEO-ID on every hinge of every experiment against the optical truth.
4. Tables, the subject-level paired bootstrap for H2 (Holm across C1-C3), coverage, reliability, selective risk and
   figures; the Markdown.

Nothing is fitted here: the estimator's fitted beliefs are those in configs/default.yaml (fitted on train and
validation by `python -m mesqsim fit-geometry`). The same config and seed give an identical RESULTS_STEP2.md apart
from its "Run:" line, which holds everything that depends on the machine and the clock (check 33.1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mesqsim import io, plotstyle  # noqa: E402
from mesqsim.body import JOINTS  # noqa: E402
from mesqsim.config import load_config  # noqa: E402
from mesqsim.dataset import amass_sources, synthetic_sources  # noqa: E402
from mesqsim.evaluate import USES_TRUTH  # noqa: E402
from mesqsim.geo_sessions import dataset_jobs, run_jobs  # noqa: E402
from mesqsim.paths import manifest_dir  # noqa: E402
from mesqsim.results_step2 import (  # noqa: E402
    METHODS_ALL,
    METHODS_SWEEP,
    bootstrap_contrast,
    coverage,
    holm,
    selective_risk,
    session_worker,
)
from mesqsim.runs import git_commit  # noqa: E402

SPLIT_MANIFEST = manifest_dir() / "split_manifest.csv"
JT = np.array([j.jtype for j in JOINTS])
TYPES = ("hinge", "2dof", "3dof")
TYPE_NAME = {"hinge": "hinge (knees)", "2dof": "2-DoF (elbows, wrists, ankles)",
             "3dof": "3-DoF (hips, shoulders, lower back, neck)"}  # fmt: skip
SWEEPS = (
    ("error_model.offset_mode", None),
    ("error_model.box_slot_deg", None),
    ("error_model.drift_sigma_deg_per_min", None),
    ("error_model.scale_factor_max", None),
    ("error_model.reboot_rate_per_h", None),
    ("radio.loss_mean", None),
    ("radio.burst_mean_frames", None),
    ("timing.stall_ms", [0.0, 100.0]),
)
MARGIN_DEG = 1.0  # the practical margin proposed in the brief (§3), to be frozen by Danyal in the analysis plan
KEY_PARAMS = ("error_model.offset_mode", "error_model.box_slot_deg", "error_model.drift_sigma_deg_per_min",
              "error_model.scale_factor_max", "error_model.tilt_sigma_deg", "error_model.reboot_rate_per_h",
              "radio.loss_mean", "dataset.window_s", "dataset.motion_start_min", "geometry.window_s",
              "geometry.step_s", "geometry.kappa_scale.hinge", "geometry.kappa_scale.2dof",
              "geometry.kappa_scale.3dof", "geometry.model_sigma_deg.hinge", "geometry.model_sigma_deg.2dof",
              "geometry.model_sigma_deg.3dof", "geometry.drift_q_rad2_per_s", "geometry.kappa_hold")  # fmt: skip


def cpu_name() -> str:
    try:
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              check=True).stdout.strip()  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        import platform

        return platform.processor() or platform.machine()


# ------------------------------------------------------------------------------------------ assembling the arrays

def stack(results: list, method: str, key: str = "err") -> np.ndarray:
    """(S, K, 14) array of one field across sessions (sessions of one dataset have equal motion lengths; padded)."""
    k = max(r[method][key].shape[0] for r in results)
    out = np.full((len(results), k, len(JOINTS)), np.nan, dtype=np.float32)
    for i, r in enumerate(results):
        a = r[method][key]
        out[i, : a.shape[0]] = a
    return out


def stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan}
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}


def fmt(v, nd=2) -> str:
    return "–" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def md(df: pd.DataFrame) -> str:
    return io.markdown_table(df, ".2f")


# ------------------------------------------------------------------------------------------ the tables

def table_overall(err: dict, starts: np.ndarray, sessions_by_start: dict) -> pd.DataFrame:
    rows = []
    for m in METHODS_ALL:
        for s in sorted(set(starts.tolist())):
            sel = starts == s
            st = stats(err[m][sel])
            rows.append({"method": m + (" *" if m in USES_TRUTH else ""), "min since boot": int(s), **st,
                         "sessions": int(sel.sum())})  # fmt: skip
    return pd.DataFrame(rows)


def table_by_type(err: dict, starts: np.ndarray, methods) -> pd.DataFrame:
    rows = []
    for t in TYPES:
        for m in methods:
            row = {"joint type": t, "method": m + (" *" if m in USES_TRUTH else "")}
            for s in sorted(set(starts.tolist())):
                st = stats(err[m][starts == s][:, :, JT == t])
                row[f"{int(s)} min"] = f"{st['mean']:.2f} / {st['p95']:.2f}"
            rows.append(row)
    return pd.DataFrame(rows)


def table_by_joint(err: dict, methods) -> pd.DataFrame:
    rows = []
    for k, j in enumerate(JOINTS):
        row = {"joint": j.name, "type": j.jtype}
        for m in methods:
            st = stats(err[m][:, :, k])
            row[m] = f"{st['median']:.2f} / {st['p95']:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)


def terciles(exc: np.ndarray) -> np.ndarray:
    """Per (session, joint) tercile 0/1/2 of the excitation, cut within each joint type over all sessions."""
    out = np.zeros(exc.shape, dtype=np.int64)
    for t in TYPES:
        cols = JT == t
        v = exc[:, cols]
        q1, q2 = np.nanpercentile(v, [100 / 3, 200 / 3])
        out[:, cols] = np.where(v <= q1, 0, np.where(v <= q2, 1, 2))
    return out


def table_excitation(err: dict, terc: np.ndarray, exc: np.ndarray, methods) -> pd.DataFrame:
    rows = []
    for t in TYPES:
        cols = np.flatnonzero(JT == t)
        for g, name in enumerate(("low", "middle", "high")):
            sel = terc[:, cols] == g  # (S, n_cols): which (session, joint) cells fall in this tercile
            ex = exc[:, cols][sel]
            row = {"joint type": t, "excitation tercile": name, "cells": int(sel.sum()),
                   "excitation range": f"{ex.min():.3g}–{ex.max():.3g}" if ex.size else "–"}
            for m in methods:
                parts = [err[m][i, :, cols[c]] for i, c in zip(*np.nonzero(sel), strict=True)]
                st = stats(np.concatenate(parts) if parts else np.zeros(0))
                row[m] = f"{st['median']:.2f} / {st['p95']:.2f}"
            rows.append(row)
    return pd.DataFrame(rows)


def per_subject_median(err_m: np.ndarray, subjects: np.ndarray, mask: np.ndarray) -> dict:
    """{subject: median error over all steps of that subject's sessions and the (session, joint) cells in mask}."""
    out = {}
    for s in sorted(set(subjects.tolist())):
        rows = subjects == s
        cells = [err_m[i, :, c] for i in np.flatnonzero(rows) for c in np.flatnonzero(mask[i])]
        if cells:
            v = np.concatenate(cells)
            v = v[np.isfinite(v)]
            if v.size:
                out[s] = float(np.median(v))
    return out


def contrasts(err: dict, subjects: np.ndarray, starts: np.ndarray, terc: np.ndarray, n_boot: int) -> pd.DataFrame:
    """H2's primary contrasts at 10 min since boot: GEO-ID against B1 per joint type, paired by subject."""
    at10 = starts == 10
    defs = (("C1", "hinge", (1, 2), "upper two excitation terciles"),
            ("C2", "2dof", (1, 2), "upper two excitation terciles"),
            ("C3", "3dof", (0, 1, 2), "all motion"))  # fmt: skip
    rows, pvals = [], {}
    for cid, t, tercs, stratum in defs:
        mask = (JT == t)[None, :] & np.isin(terc, tercs) & at10[:, None]
        geo = per_subject_median(err["GEO-ID"], subjects, mask)
        b1 = per_subject_median(err["B1"], subjects, mask)
        units = sorted(set(geo) & set(b1))
        diff = np.array([geo[u] - b1[u] for u in units])
        bs = bootstrap_contrast(diff, n_boot=n_boot)
        pvals[cid] = bs["p"]
        rows.append({"contrast": cid, "joint type": t, "stratum": stratum, "units": bs["n"],
                     "B1 median of units": float(np.median([b1[u] for u in units])) if units else np.nan,
                     "GEO-ID median of units": float(np.median([geo[u] for u in units])) if units else np.nan,
                     "GEO-ID − B1 (mean)": bs["mean"], "95 % CI low": bs["lo"], "95 % CI high": bs["hi"],
                     "p (bootstrap)": bs["p"]})  # fmt: skip
    adj = holm(pvals)
    for r in rows:
        r["p (Holm)"] = adj[r["contrast"]]
    return pd.DataFrame(rows)


def verdicts(c: pd.DataFrame) -> dict:
    """The plan's reading (descriptive while it is not frozen): C1, C2 'better' = interval below 0 and the mean
    improvement above the margin; C3 'not better' = the upper end of the improvement interval below the margin."""
    out = {}
    for _, r in c.iterrows():
        better = r["95 % CI high"] < 0 and -r["GEO-ID − B1 (mean)"] > MARGIN_DEG
        if r["contrast"] in ("C1", "C2"):
            out[r["contrast"]] = "supports H2 (GEO-ID better)" if better else "does not support H2 (not shown better)"
        else:
            not_better = -r["95 % CI low"] < MARGIN_DEG
            out[r["contrast"]] = ("supports H2 (an improvement above the margin is excluded)" if not_better
                                  else "does not support H2 (an improvement above the margin is not excluded)")
    return out


def table_identification(res: list, err: dict) -> pd.DataFrame:
    """How often GEO-ID identified the knee axes and the facing on test, and its error with and without them.

    Without accepted knee axes GEO-ID has no facing, hence no T-pose frames: its range-of-motion solvers cannot run
    and every joint keeps its box estimate (GEO-ID = B1 there, by construction)."""
    knees = np.stack([r["geo_id"]["knee_axes"] for r in res])
    facing = np.array([r["geo_id"]["facing"] for r in res])
    rows = []
    for name, sel in (("facing found", facing), ("no facing", ~facing), ("all", np.ones(len(res), bool))):
        row = {"sessions": name, "count": int(sel.sum()), "share": float(sel.mean()),
               "left knee axes accepted": float(knees[sel, 0].mean()) if sel.any() else np.nan,
               "right knee axes accepted": float(knees[sel, 1].mean()) if sel.any() else np.nan}  # fmt: skip
        for t in TYPES:
            b1, geo = (stats(err[m][sel][:, :, JT == t])["median"] for m in ("B1", "GEO-ID"))
            row[f"{t}: B1 → GEO-ID median"] = f"{b1:.2f} → {geo:.2f}" if sel.any() else "–"
        rows.append(row)
    return pd.DataFrame(rows)


def table_confidence(res: list, methods) -> pd.DataFrame:
    rows = []
    for m in methods:
        e, k, u = stack(res, m), stack(res, m, "kappa"), stack(res, m, "uncertain")
        for t in (*TYPES, "all"):
            cols = np.ones(len(JOINTS), bool) if t == "all" else JT == t
            cv = coverage(e[:, :, cols].ravel(), k[:, :, cols].ravel())
            unc = u[:, :, cols].ravel() > 0.5
            ee = e[:, :, cols].ravel()
            rows.append({"method": m + (" *" if m in USES_TRUTH else ""), "joint type": t,
                         "68 % coverage": cv[0.68], "95 % coverage": cv[0.95],
                         "uncertain share": float(np.mean(unc)),
                         "median error when certain": float(np.nanmedian(ee[~unc])) if (~unc).any() else np.nan,
                         "median error when uncertain": float(np.nanmedian(ee[unc])) if unc.any() else np.nan})
    return pd.DataFrame(rows)


LEVELS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.68, 0.8, 0.9, 0.95, 0.99])
COLOURS = {"B1": plotstyle.SERIES[0], "O1": plotstyle.MUTED, "GEO-K": plotstyle.SERIES[2],
           "GEO-ID": plotstyle.SERIES[1]}


def reliability(res: list, method: str, levels=LEVELS):
    e, k = stack(res, method), stack(res, method, "kappa")
    out = {}
    for t in TYPES:
        ee, kk = e[:, :, JT == t].ravel(), k[:, :, JT == t].ravel()
        cv = coverage(ee, kk, levels=tuple(levels))
        out[t] = np.array([cv[lv] for lv in levels])
    return levels, out


# ------------------------------------------------------------------------------------------ figures

def fig_growth(err: dict, starts: np.ndarray, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    col = COLOURS
    xs = sorted(set(starts.tolist()))
    for ax, t in zip(axes, TYPES, strict=True):
        for m in ("B1", "O1", "GEO-K", "GEO-ID"):
            y = [stats(err[m][starts == s][:, :, JT == t])["median"] for s in xs]
            ax.plot(xs, y, "-o", color=col[m], lw=2, ms=5, label=m + (" (truth)" if m in USES_TRUTH else ""))
        ax.set_title(TYPE_NAME[t], loc="left", fontsize=9)
        ax.set_xticks(xs)
        ax.set_xlabel("minutes since boot")
    axes[0].set_ylabel("median |δ̂ − δ| (deg)")
    axes[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


def fig_reliability(levels, rel: dict, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=plotstyle.AXIS, lw=1, ls="--", label="perfect calibration")
    for t, c in zip(TYPES, plotstyle.SERIES, strict=False):
        ax.plot(levels, rel[t], "-o", color=c, ms=4, lw=1.5, label=TYPE_NAME[t].split(" (")[0])
    ax.set_xlabel("nominal coverage of the von Mises interval")
    ax.set_ylabel("observed coverage on test")
    ax.set_title(title, loc="left", fontsize=10)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


def fig_selective(res: list, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    e, k = stack(res, "GEO-ID"), stack(res, "GEO-ID", "kappa")
    fr = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05)
    for t, c in zip(TYPES, plotstyle.SERIES, strict=False):
        rows = selective_risk(e[:, :, JT == t].ravel(), k[:, :, JT == t].ravel(), fractions=fr)
        ax.plot([r["fraction"] for r in rows], [r["median"] for r in rows], "-o", color=c, ms=4, lw=1.5,
                label=TYPE_NAME[t].split(" (")[0])  # fmt: skip
    ax.invert_xaxis()
    ax.set_xlabel("share of steps kept (highest κ first)")
    ax.set_ylabel("median error of the kept steps (deg)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


def fig_sweeps(sw: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    params = list(dict.fromkeys(sw["parameter"]))
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow), squeeze=False)
    col = COLOURS
    for ax, p in zip(axes.flat, params, strict=False):
        t = sw[sw["parameter"] == p]
        values = list(dict.fromkeys(t["value"]))
        x = np.arange(len(values))
        for m in ("B1", "O1", "GEO-K", "GEO-ID"):
            y = [t[(t["value"] == v) & (t["method"] == m)]["mean"].iloc[0] for v in values]
            ax.plot(x, y, "-o", color=col[m], lw=2, ms=4, label=m)
        ax.set_xticks(x, [str(v) for v in values], fontsize=8)
        ax.set_title(p.split(".")[-1], fontsize=9, loc="left")
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(params):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("mean error (deg)")
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=plotstyle.INK)
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


def fig_reboot(rb: pd.DataFrame, out: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    col = COLOURS
    x = np.arange(len(rb))
    for m in ("B1", "O1", "GEO-K", "GEO-ID"):
        ax.plot(x, rb[m], "-o", color=col[m], lw=2, ms=4, label=m)
    ax.set_xticks(x, rb["seconds since the reboot"], fontsize=8)
    ax.set_xlabel("seconds since the host learned of the reboot")
    ax.set_ylabel("median error, joints of the rebooted pod (deg)")
    ax.set_title(title, loc="left", fontsize=10)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)


# ------------------------------------------------------------------------------------------ DIODEM

def _diodem_exp(args):
    e, cfg_path, cfg_pre_text = args
    import tempfile

    from mesqsim.config import load_config as _load
    from mesqsim.data.diodem import load_trial
    from mesqsim.diodem_geo import hinge_joints, run_hinge

    cfgs = {"fitted": _load(cfg_path)}
    if cfg_pre_text is not None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "default.yaml"
            p.write_text(cfg_pre_text, encoding="utf-8")
            (Path(tmp) / "presets.yaml").write_text((Path(cfg_path).with_name("presets.yaml")).read_text())
            cfgs["assumed"] = _load(p)
    data = load_trial(e)
    rows = []
    from mesqsim.conventions import vm_halfwidth_rad, wrap_rad

    for parent, child in hinge_joints(e):
        row = {"experiment": f"exp{e:02d}", "joint": f"{parent}-{child}"}
        for tag, cfg in cfgs.items():
            r = run_hinge(data, parent, child, cfg)
            err = np.rad2deg(np.abs(wrap_rad(r["delta_rad"] - r["truth_rad"])))
            ref = np.rad2deg(np.abs(wrap_rad(r["start_oracle_rad"] - r["truth_rad"])))
            ok = np.isfinite(err)
            row["minutes"] = float(r["t_s"][-1] / 60)
            row[f"GEO-ID {tag} median"] = float(np.median(err[ok]))
            row[f"GEO-ID {tag} P95"] = float(np.percentile(err[ok], 95))
            if tag == "fitted":
                row["start median"] = float(np.median(ref[ok]))
                row["start end"] = float(ref[ok][-1])
                row["first info (s)"] = float(r["t_first_info_s"])
                row["95 % coverage"] = float(np.mean(err[ok] <= np.rad2deg(vm_halfwidth_rad(r["kappa"][ok], 0.95))))
        rows.append(row)
    return rows


def diodem_table(cfg_path: Path, workers: int | None) -> pd.DataFrame | None:
    from mesqsim.data import diodem

    if not diodem.is_available():
        return None
    try:  # the beliefs before the validation fit, for the transfer comparison
        pre = subprocess.run(["git", "-C", str(ROOT), "show", "944b822:configs/default.yaml"], capture_output=True,
                             text=True, check=True).stdout  # fmt: skip
    except (OSError, subprocess.CalledProcessError):
        pre = None
    exps = diodem.ARM_EXPS + diodem.GAIT_EXPS
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_diodem_exp, [(e, str(cfg_path), pre) for e in exps]))
    return pd.DataFrame([r for p in parts for r in p])


# ------------------------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["amass", "synthetic"], default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-sequences", type=int, default=None)
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--no-diodem", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default=str(ROOT / "docs" / "RESULTS_STEP2.md"))
    ap.add_argument("--figdir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    t_wall = time.perf_counter()
    source = args.source or ("amass" if SPLIT_MANIFEST.is_file() else "synthetic")
    sources = amass_sources(SPLIT_MANIFEST) if source == "amass" else synthetic_sources()
    cfg_path = ROOT / "configs" / "default.yaml"
    cfg = load_config(cfg_path)
    commit = git_commit()
    figdir = Path(args.figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    jobs = dataset_jobs(cfg, sources, args.seed, splits=("test",), max_sequences=args.max_sequences)
    print(f"source {source}: {len(jobs)} test sessions from {len({j['path'] for j in jobs})} sequences; "
          f"commit {commit[:12]}, config {cfg.hash()[:8]}, seed {args.seed}")
    if not jobs:
        raise SystemExit("no test sessions: no test sequence is as long as dataset.window_s")
    res = run_jobs(session_worker, jobs, cfg_path, workers=args.workers)
    starts = np.array([r["job"]["motion_start_min"] for r in res])
    subjects = np.array([f"{r['job']['subset']}/{r['job']['subject']}" for r in res])
    err = {m: stack(res, m) for m in METHODS_ALL}
    err_o = {m: stack(res, m, "err_orient") for m in ("B1", "O1", "O2", "GEO-K", "GEO-ID")}
    exc = np.stack([r["excitation"] for r in res])
    terc = terciles(exc)
    geo_ms = np.array([1e3 * r["geo_seconds"] / r["n_query"] for r in res])

    t1 = table_overall(err, starts, {})
    t2 = table_by_type(err, starts, ("B0", "B1", "O1", "GEO-K", "GEO-ID", "GEO-ID-nofilter", "GEO-ID-hinge",
                                     "GEO-ID-w4", "GEO-ID-w16"))  # fmt: skip
    t3 = table_by_joint(err, ("B1", "O1", "GEO-K", "GEO-ID"))
    t3b = table_identification(res, err)
    t4 = table_overall({m: err_o[m] for m in err_o} | {m: err[m] for m in METHODS_ALL if m not in err_o}, starts,
                       {})  # fmt: skip
    t4 = t4[t4["method"].str.replace(" *", "", regex=False).isin(list(err_o))]
    t5 = table_excitation(err, terc, exc, ("B1", "O1", "GEO-K", "GEO-ID"))
    t6 = table_confidence(res, ("GEO-K", "GEO-ID", "GEO-ID-nofilter", "GEO-ID-hinge", "GEO-ID-w4", "GEO-ID-w16"))
    c = contrasts(err, subjects, starts, terc, args.n_boot)
    v = verdicts(c)
    sel_rows = []
    e_id, k_id = stack(res, "GEO-ID"), stack(res, "GEO-ID", "kappa")
    for t in (*TYPES, "all"):
        cols = np.ones(len(JOINTS), bool) if t == "all" else JT == t
        for r in selective_risk(e_id[:, :, cols].ravel(), k_id[:, :, cols].ravel()):
            sel_rows.append({"joint type": t, **r})
    t7 = pd.DataFrame(sel_rows)

    label = "synthetic motion (PROVISIONAL)" if source == "synthetic" else "AMASS test subjects"
    fig_growth(err, starts, figdir / "results_step2_growth.png",
               f"Median relative-heading error by joint type: {label}")
    levels, rel = reliability(res, "GEO-ID")
    fig_reliability(levels, rel, figdir / "results_step2_reliability.png", f"GEO-ID calibration on test: {label}")
    fig_selective(res, figdir / "results_step2_selective.png", f"GEO-ID selective risk: {label}")

    sw_tab, rb_tab = None, None
    if not args.no_sweeps:
        base = cfg
        sw_rows, rb_cells = [], {m: {} for m in ("B1", "O1", "GEO-K", "GEO-ID")}
        sw_jobs = dataset_jobs(base, sources, args.seed, splits=("test",), windows_per_sequence=1,
                               motion_start_min=[10], max_sequences=args.max_sequences)  # fmt: skip
        bins = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 20)]
        for p, vals in SWEEPS:
            for val in vals or cfg.param(p)["sweep"]:
                ov = {p: val}
                for j in sw_jobs:
                    j["overrides"] = ov
                print(f"sweep {p} = {val}")
                part = run_jobs(_sweep_worker, sw_jobs, cfg_path, workers=args.workers, log=lambda s: None)
                for m in METHODS_SWEEP:
                    st = stats(stack(part, m))
                    sw_rows.append({"parameter": p, "value": val if isinstance(val, str) else f"{val:g}",
                                    "method": m, **st})  # fmt: skip
                if p == "error_model.reboot_rate_per_h" and float(val) > 0:
                    since = np.stack([r["since_reset_s"] for r in part])
                    for m in rb_cells:
                        e_m = stack(part, m)
                        for lo, hi in bins:
                            cell = (since >= lo) & (since < hi)
                            rb_cells[m][f"{lo}–{hi}"] = float(np.nanmedian(e_m[cell])) if cell.any() else np.nan
        sw_tab = pd.DataFrame(sw_rows)
        fig_sweeps(sw_tab, figdir / "results_step2_sweeps.png", f"One-at-a-time conditions, motion at 10 min: {label}")
        rb_tab = pd.DataFrame({"seconds since the reboot": [f"{lo}–{hi}" for lo, hi in bins],
                               **{m: [rb_cells[m].get(f"{lo}–{hi}", np.nan) for lo, hi in bins] for m in rb_cells}})
        fig_reboot(rb_tab, figdir / "results_step2_reboot.png", f"Re-acquisition after a reboot: {label}")
    dio = None if args.no_diodem else diodem_table(cfg_path, args.workers)

    wall = time.perf_counter() - t_wall
    write_markdown(Path(args.out), source, cfg, commit, args, jobs, res, starts, subjects, t1, t2, (t3, t3b), t4, t5,
                   t6, t7, c, v, sw_tab, rb_tab, dio, geo_ms, wall)  # fmt: skip
    print(f"wrote {args.out} in {wall / 60:.1f} min")
    return 0


def _sweep_worker(job, cfg):
    return session_worker(job, cfg, methods=METHODS_SWEEP)


def write_markdown(out: Path, source, cfg, commit, args, jobs, res, starts, subjects, t1, t2, t3, t4, t5, t6, t7, c,
                   v, sw, rb, dio, geo_ms, wall) -> None:  # fmt: skip
    synthetic = source == "synthetic"
    labels = Counter(node["label"] for _, node in cfg.leaves())
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = ["# Step 2 results: the geometric estimator (GEO-K, GEO-ID) against the Step 1 baselines", ""]
    if synthetic:
        lines += ["> **PROVISIONAL (SYNTHETIC MOTION).** Simple synthetic walking, turning and standing clips",
                  "> stand in for human motion. These numbers show that the pipeline runs; they are not results",
                  "> about people.", ""]
    else:
        sha_file = SPLIT_MANIFEST.with_suffix(".sha256")
        sha = sha_file.read_text(encoding="utf-8").split()[0] if sha_file.is_file() else "not written"
        lines += [
            f"> **AMASS motion, assumed suit parameters, descriptive.** {labels.get('assumption', 0)} of the "
            f"{sum(labels.values())} parameters are assumptions until the bench measures them and "
            f"{labels.get('fitted', 0)} are fitted (train / validation only). The split manifest "
            f"(SHA-256 `{sha[:12]}…`) and the analysis plan are **not frozen yet** (Danyal), so every comparison",
            "> below is **descriptive**: the intervals and Holm-adjusted p-values are computed as the draft plan says,",
            "> but no confirmatory claim is made. Simulated sessions: the numbers describe the configured model of the",
            "> suit.",
            "",
        ]
    flags = f"--source {source} --seed {args.seed}" + (f" --max-sequences {args.max_sequences}" if args.max_sequences
                                                       else "")  # fmt: skip
    lines += [
        f"*Generated by `python scripts/make_results_step2.py` (`{flags}`). Do not edit by hand. Same config and seed: "
        "identical file apart from the Run line (check 33.1).*",
        "",
        f"**Run:** {date} · {cpu_name()} · {wall / 60:.1f} min · GEO-ID compute: median {np.nanmedian(geo_ms):.3f} ms "
        f"per 32 Hz step for all 14 joints (P95 {np.nanpercentile(geo_ms, 95):.3f} ms, {len(geo_ms)} sessions).",
        "",
        f"**Code and data:** commit `{commit[:12]}`; config hash `{cfg.hash()[:12]}`; seed {args.seed}; "
        f"{len(res)} test sessions from {len({j['path'] for j in jobs})} sequences of "
        f"{len(set(subjects.tolist()))} subjects ({source}); motion starts "
        f"{', '.join(str(int(s)) for s in sorted(set(starts.tolist())))} min after boot.",
        "",
        "## Picture first",
        "",
        "Every joint's relative heading δ — how far the child pod's north has slid from its parent's — is estimated by",
        "each method at every 32 Hz step of the motion and compared with the simulator's truth. **B1** registers the",
        "pods in the box; **GEO-ID** is the deployable geometry: knee hinge axes identified from motion (tied to the",
        "T-pose), range of motion at the other joints in T-pose frames, a von Mises filter with its own confidence κ.",
        "**GEO-K** is the same geometry with the simulator's true axes and mountings: a ceiling, never deployable.",
        "",
        "| method | what it is | uses simulator truth |",
        "|---|---|---|",
        "| B0 | no correction | no |",
        "| B1 | box registration | no |",
        "| O1 | true offsets at boot (oracle) | **yes** |",
        "| O2 | true δ (oracle; its error is 0 by construction) | **yes** |",
        "| GEO-K | geometry with known axes and mountings | **yes** (axes, mountings) |",
        "| GEO-ID | deployable geometry: identified axes, T-pose frames, filter | no |",
        "| GEO-ID-nofilter | GEO-ID, each window alone (box registration without information) | no |",
        "| GEO-ID-hinge | GEO-ID solving the knees only; every other joint keeps its box estimate | no |",
        "| GEO-ID-w4, GEO-ID-w16 | GEO-ID with 4 s and 16 s windows (8 s default) | no |",
        "",
        "Methods marked `*` in the tables use simulator truth.",
        "",
        "## Parameter set",
        "",
        "| parameter | value | label |",
        "|---|---|---|",
    ]
    for k in KEY_PARAMS:
        prm = cfg.param(k)
        val = prm["value"]
        lines.append(f"| `{k}` | {val if not isinstance(val, float) else f'{val:.6g}'} | {prm['label']} |")
    lines += ["", "## Table 1: primary metric, |wrap(δ̂ − δ)| over all 14 joints (degrees)", "", md(t1),
              "## Table 2: by joint type (mean / P95, degrees)", "", md(t2),
              "## Table 3: by joint, all motion starts pooled (median / P95, degrees)", "", md(t3[0]),
              "## Table 3b: what GEO-ID could identify", "",
              "GEO-ID needs the knee axes first: the facing comes from them, the T-pose frames from the facing, "
              "and the range-of-motion solvers of the other joints from the T-pose frames. A session without "
              "accepted knee axes therefore has no facing, and every joint keeps its box estimate (GEO-ID = B1 by "
              "construction). Errors: median over all motion steps, all motion starts pooled (degrees).", "",
              md(t3[1]),
              "## Table 4: secondary metric, parent-relative orientation error with the true mountings (degrees)", "",
              "Includes fusion tilt error and quantisation, so even O2 is not 0.", "", md(t4),
              "## Table 5: by excitation tercile (median / P95, degrees)", "",
              "Excitation, from the truth per session and joint: for hinge and 2-DoF joints the spread (std, degrees) "
              "of the joint's main angle over the motion; for 3-DoF joints the share of frames within 5° of a "
              "range-of-motion limit or outside it. Terciles are cut within each joint type.", "", md(t5)]
    lines += ["## H2: what the data say", "",
              "**H2** (Step 2 brief §3): geometry recovers relative heading at hinge and 2-DoF joints during walking "
              "and arm motion, but not at 3-DoF joints in everyday motion. Contrasts at 10 min since boot, GEO-ID "
              "against B1, paired by subject (the unit): per subject the median error over all its steps and cells; "
              f"{args.n_boot:,} bootstrap resamples of subjects (fixed seed); Holm across C1–C3; margin "
              f"Δ = {MARGIN_DEG:g}° (proposed). Negative differences mean GEO-ID is better.", "", md(c)]
    for cid in ("C1", "C2", "C3"):
        lines.append(f"- **{cid}:** {v[cid]} — descriptive while the plan is not frozen.")
    lines += ["", "## Table 6: confidence — coverage of the von Mises intervals on test, and the hold flag", "",
              "H5's target: 68 % and 95 % intervals cover within ±5 points. `uncertain` = κ below "
              f"`geometry.kappa_hold` = {cfg['geometry.kappa_hold']:g} (fitted on validation).", "", md(t6),
              "![reliability](figures/results_step2_reliability.png)", "",
              "*Observed against nominal coverage of GEO-ID's intervals on test, per joint type.*", "",
              "## Table 7: selective risk of GEO-ID (steps kept from the highest κ down; errors in degrees)", "",
              md(t7), "![selective risk](figures/results_step2_selective.png)", "",
              "![growth](figures/results_step2_growth.png)", "",
              "*Median error against minutes since boot, per joint type. O1 and GEO-K use simulator truth.*", ""]
    if sw is not None:
        wide = sw.copy()
        wide["cell"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(wide["mean"], wide["p95"], strict=True)]
        wide = wide.pivot_table(index=["parameter", "value"], columns="method", values="cell", aggfunc="first",
                                sort=False).reset_index()  # fmt: skip
        lines += ["## Table 8: conditions — one-at-a-time sweeps (mean / P95, degrees; motion at 10 min)", "",
                  "Clean ≈ the smallest values; drift includes the 2.0 °/min stress value from the real suit's static "
                  "hold (brief §4.1); missing bursts = loss and burst length; resets = one reboot per pod per hour.",
                  "", md(wide), "![sweeps](figures/results_step2_sweeps.png)", ""]
    if rb is not None:
        lines += ["## Table 9: re-acquisition after a reboot (median error of the joints of the rebooted pod, degrees)",
                  "", md(rb), "![reboot](figures/results_step2_reboot.png)", ""]
    if dio is not None:
        lines += ["## Table 10: DIODEM — GEO-ID on real IMUs against optical truth (degrees)", "",
                  "Mechanical chains, out of distribution: the experiment is the unit and no claim about people rests "
                  "on these. Hinges only (saddle joints need 2-DoF identification, a Should; the spherical joint a "
                  "range of motion for these chains). `start` = a perfect registration at the first sample (its error "
                  "is the six-axis drift). `fitted` = the beliefs fitted on AMASS validation; `assumed` = the "
                  "beliefs before that fit (commit 944b822): the knee calibration from AMASS transfers poorly to a "
                  "mechanical hinge.", "", md(dio), "![DIODEM](figures/diodem_geo_hinge.png)", ""]
    lines += ["## How to read these numbers", "",
              "- B1 against GEO-ID is the question of H2; GEO-K bounds what geometry can do with perfect axes.",
              "- O1 is the error that is drift rather than offset; GEO methods can beat it, because they see the "
              "drift.",
              "- Every `assumption` in the parameter set is a guess until the bench measures it; the sweeps show how "
              "much each guess matters.",
              "- The unit of analysis is the subject; steps and sessions are not independent observations."]
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
