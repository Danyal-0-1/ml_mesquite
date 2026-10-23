"""Results tables and figures for docs/RESULTS_STEP1.md (brief §7 M5, §4.7).

Every table is computed from ``results/errors.parquet`` of a scored dataset run: motion-phase steps only, the
primary metric |wrap(delta_hat - delta)| and the secondary orientation error (true mountings), per baseline, at
1, 10 and 30 min since boot (the session's motion start).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import plotstyle
from .body import JOINTS

__all__ = ["SWEEP_PARAMS", "overall_table", "joint_type_table", "joint_table", "sweep_table", "fig_growth",
           "fig_sweeps"]  # fmt: skip

SWEEP_PARAMS = (
    "error_model.offset_mode",
    "error_model.box_slot_deg",
    "error_model.drift_sigma_deg_per_min",
    "error_model.scale_factor_max",
    "error_model.reboot_rate_per_h",
    "timing.stall_ms",
    "radio.loss_mean",
    "radio.burst_mean_frames",
)
"""Parameters swept one at a time around the default, over the values listed under ``sweep`` in the config."""

BASELINE_ORDER = ("B0", "B1", "O1", "O2")


def _stats(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if not x.size:
        return {"mean": np.nan, "median": np.nan, "p95": np.nan}
    return {"mean": float(np.mean(x)), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}


def overall_table(errors: pd.DataFrame, metric: str = "err_heading_deg") -> pd.DataFrame:
    """Rows: baseline x motion start (min); columns: mean, median, P95 of ``metric`` over all joints and steps."""
    rows = []
    for (b, m), g in errors.groupby(["baseline", "motion_start_min"], sort=True):
        rows.append({"baseline": b, "minutes_since_boot": int(m), **_stats(g[metric].to_numpy()),
                     "sessions": g["session"].nunique(), "steps": len(g) // 14})  # fmt: skip
    out = pd.DataFrame(rows)
    out["baseline"] = pd.Categorical(out["baseline"], BASELINE_ORDER, ordered=True)
    return out.sort_values(["baseline", "minutes_since_boot"], ignore_index=True)


def joint_type_table(errors: pd.DataFrame, metric: str = "err_heading_deg") -> pd.DataFrame:
    jt = np.array([j.jtype for j in JOINTS])
    e = errors.assign(joint_type=jt[errors["joint"].to_numpy()])
    rows = []
    for (t, b, m), g in e.groupby(["joint_type", "baseline", "motion_start_min"], sort=True):
        s = _stats(g[metric].to_numpy())
        rows.append({"joint_type": t, "baseline": b, "minutes_since_boot": int(m), "mean": s["mean"],
                     "p95": s["p95"]})  # fmt: skip
    return pd.DataFrame(rows)


def joint_table(errors: pd.DataFrame, metric: str = "err_heading_deg") -> pd.DataFrame:
    names = np.array([j.name for j in JOINTS])
    e = errors.assign(joint_name=names[errors["joint"].to_numpy()])
    rows = []
    for (j, b, m), g in e.groupby(["joint_name", "baseline", "motion_start_min"], sort=True):
        rows.append({"joint": j, "baseline": b, "minutes_since_boot": int(m), **_stats(g[metric].to_numpy()),
                     "sessions": g["session"].nunique()})  # fmt: skip
    return pd.DataFrame(rows)


def sweep_table(sweeps: dict) -> pd.DataFrame:
    """``{(param, value): errors}`` -> rows param, value, baseline, mean and P95 of the primary metric."""
    rows = []
    for (param, value), errors in sweeps.items():
        for b, g in errors.groupby("baseline"):
            s = _stats(g["err_heading_deg"].to_numpy())
            rows.append({"parameter": param, "value": value, "baseline": b, "mean": s["mean"], "p95": s["p95"]})
    return pd.DataFrame(rows)


def fig_growth(overall: pd.DataFrame, title: str, out: Path) -> Path:
    """Mean and P95 heading error versus minutes since boot, one line per baseline (log scale; O2 is 0)."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colours = dict(zip(("B0", "B1", "O1"), plotstyle.SERIES, strict=False))
    for b in ("B0", "B1", "O1"):
        g = overall[overall["baseline"] == b]
        if g.empty:
            continue
        ax.plot(g["minutes_since_boot"], g["mean"], "-o", color=colours[b], lw=2, ms=6, label=f"{b} mean")
        ax.plot(g["minutes_since_boot"], g["p95"], "--", color=colours[b], lw=1.2, alpha=0.8, label=f"{b} P95")
        ax.annotate(b, (g["minutes_since_boot"].iloc[-1], g["mean"].iloc[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=plotstyle.INK_2)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xticks(sorted(overall["minutes_since_boot"].unique()))
    ax.set_xlabel("minutes since boot (motion start)")
    ax.set_ylabel("relative-heading error (deg)")
    ax.set_title(title, loc="left")
    ax.legend(ncol=3, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)
    return out


def fig_sweeps(table: pd.DataFrame, title: str, out: Path) -> Path:
    """Small multiples: mean heading error of B1 and O1 for each value of each swept parameter."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    params = list(dict.fromkeys(table["parameter"]))
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.9 * nrow), squeeze=False)
    colours = {"B1": plotstyle.SERIES[1], "O1": plotstyle.SERIES[2]}
    for ax, p in zip(axes.flat, params, strict=False):
        t = table[table["parameter"] == p]
        values = list(dict.fromkeys(t["value"]))
        x = np.arange(len(values))
        for b in ("B1", "O1"):
            y = [t[(t["value"] == v) & (t["baseline"] == b)]["mean"].iloc[0] for v in values]
            ax.plot(x, y, "-o", color=colours[b], lw=2, ms=5, label=b)
        ax.set_xticks(x, [str(v) for v in values], fontsize=8)
        ax.set_title(p.split(".")[-1], fontsize=9, loc="left")
        ax.set_ylim(bottom=0)
    for ax in list(axes.flat)[len(params):]:
        ax.set_visible(False)
    axes.flat[0].set_ylabel("mean error (deg)")
    axes.flat[0].legend(loc="upper left")
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=plotstyle.INK)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)
    return out
