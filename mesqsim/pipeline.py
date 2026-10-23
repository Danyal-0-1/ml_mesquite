"""Runs on disk: simulate sessions into a run folder, then decode and score them.

    <output dir>/run_g<commit>_c<config hash>_s<seed>/
        sessions/<session id>/frames.parquet, session.json      what a real recording also has
                              labels_ticks.parquet, labels_joints.parquet, meta.json   simulation only
                              series.parquet                    Step 0 output (written by `baselines`)
        results/errors.parquet, summary.csv, summary.md        written by `baselines`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import evaluate as ev
from . import io, step0
from .body import JOINTS
from .config import Config
from .paths import data_dir
from .runs import git_commit, package_versions, run_folder
from .sim.motion import MotionSource
from .sim.simulate import simulate_session

__all__ = ["simulate_run", "session_meta", "score_run", "summarise"]


def _portable(spec: str) -> str:
    """Motion spec with AMASS paths written relative to the data directory (no machine-specific paths)."""
    try:
        return Path(spec).resolve().relative_to(data_dir().resolve()).as_posix()
    except (ValueError, OSError):
        return spec


def session_meta(cfg: Config, seed: int, commit: str, motion: MotionSource, spec: str, data, split=None) -> dict:
    return {
        "run": {"commit": commit, "config_hash": cfg.hash(), "seed": int(seed)},
        "config": cfg.values(),
        "config_parameters": [{"key": k, "value": v, "label": lab, "source": src} for k, v, lab, src in cfg.table()],
        "packages": package_versions(),
        "source": {"motion": _portable(spec), "fps": motion.fps, "duration_s": motion.duration_s},
        "split": split,
        **data.meta,
    }


def simulate_run(cfg: Config, seed: int, motion_spec: str, out_root: Path | None = None,
                 out_dir: Path | None = None, session_id: str = "s000") -> Path:  # fmt: skip
    """Simulate one session; return the run folder."""
    commit = git_commit()
    folder = Path(out_dir) if out_dir is not None else run_folder(commit, cfg.hash(), seed, out_root)
    motion = MotionSource.from_spec(motion_spec, spine_joint=int(cfg["motion.spine_joint"]))
    data = simulate_session(cfg, seed, motion)
    meta = session_meta(cfg, seed, commit, motion, motion_spec, data)
    io.write_session(folder / "sessions" / session_id, data, meta)
    return folder


def summarise(errors: pd.DataFrame, phase: str = "motion") -> pd.DataFrame:
    """Mean, median and P95 of both metrics per baseline and joint (and per joint type), in one phase."""
    e = errors[errors["phase"] == phase].copy()
    e["joint_name"] = [JOINTS[j].name for j in e["joint"]]
    e["joint_type"] = [JOINTS[j].jtype for j in e["joint"]]

    def agg(g):
        out = {}
        for col, tag in (("err_heading_deg", "heading"), ("err_orient_deg", "orient")):
            x = g[col].to_numpy()
            x = x[np.isfinite(x)]
            out[f"{tag}_mean_deg"] = float(np.mean(x)) if x.size else np.nan
            out[f"{tag}_median_deg"] = float(np.median(x)) if x.size else np.nan
            out[f"{tag}_p95_deg"] = float(np.percentile(x, 95)) if x.size else np.nan
        out["n"] = int(len(g))
        return pd.Series(out)

    by_joint = e.groupby(["baseline", "joint_name"], sort=False).apply(agg, include_groups=False).reset_index()
    by_type = e.groupby(["baseline", "joint_type"], sort=False).apply(agg, include_groups=False).reset_index()
    by_type = by_type.rename(columns={"joint_type": "joint_name"})
    by_type["joint_name"] = "type:" + by_type["joint_name"]
    every = e.groupby(["baseline"], sort=False).apply(agg, include_groups=False).reset_index()
    every["joint_name"] = "all"
    out = pd.concat([every, by_type, by_joint], ignore_index=True)
    out["n"] = out["n"].astype(int)
    first = ["baseline", "joint_name", "n"]
    return out[first + [c for c in out.columns if c not in first]]


def score_run(run: Path, baselines=ev.BASELINES) -> pd.DataFrame:
    """Decode every session with Step 0, score the baselines, write results; return the summary."""
    run = Path(run)
    errors = []
    for sess in sorted((run / "sessions").iterdir()):
        data = io.read_session(sess)
        series = step0.decode(data["frames"], data["session"])
        io.write_parquet(sess / "series.parquet", series)
        e = ev.evaluate(data, series, baselines)
        e.insert(0, "session", sess.name)
        errors.append(e)
    errors_df = pd.concat(errors, ignore_index=True)
    res = run / "results"
    res.mkdir(exist_ok=True)
    io.write_parquet(res / "errors.parquet", errors_df)
    summary = summarise(errors_df)
    summary.to_csv(res / "summary.csv", index=False, float_format="%.6g", lineterminator="\n")
    (res / "summary.md").write_text(io.markdown_table(summary), encoding="utf-8", newline="\n")
    return summary
