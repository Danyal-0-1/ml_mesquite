"""Simulated sessions in memory, for fitting (validation) and for the Step 2 results (test). Scorer side.

The sessions are exactly those :func:`mesqsim.dataset.generate` would make for a split: the same windows, motion
starts, variants and seeds (:func:`mesqsim.dataset.session_seed`). Each is simulated, decoded with Step 0 and
estimated inside a worker process, and only compact arrays come back, so thousands of sessions need neither the
~5 MB per session of a run folder on disk nor the memory of a full error table.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, load_config
from .dataset import pre_motion_s, session_seed

__all__ = ["dataset_jobs", "run_jobs", "simulate_job"]


def dataset_jobs(cfg: Config, sources: pd.DataFrame, run_seed: int, splits=("val",), windows_per_sequence=None,
                 motion_start_min=None, max_sequences: int | None = None) -> list[dict]:  # fmt: skip
    """One job per session of the dataset restricted to ``splits`` (the same seeds a dataset run would use)."""
    from .sim.motion import MotionSource

    window_s = float(cfg["dataset.window_s"])
    pre = pre_motion_s(cfg)
    starts = [float(m) for m in (motion_start_min or cfg["dataset.motion_start_min"])]
    per_seq = int(windows_per_sequence or cfg["dataset.windows_per_sequence"])
    chosen = sources[sources["split"].isin(splits)].sort_values("path", ignore_index=True)
    jobs, used = [], 0
    for _, src in chosen.iterrows():
        if max_sequences is not None and used >= max_sequences:  # the cap counts sequences that yield sessions
            break
        n_win = _n_windows(src["path"], window_s, int(cfg["motion.spine_joint"]), MotionSource)
        used += n_win > 0
        for w in range(min(n_win, per_seq)):
            for m in starts:
                for v in range(int(cfg["dataset.variants"])):
                    jobs.append({"path": src["path"], "subset": src["subset"], "subject": str(src["subject"]),
                                 "split": src["split"], "window": w, "window_start_s": w * window_s,
                                 "motion_start_min": m, "variant": v, "idle_s": 60.0 * m - pre,
                                 "seed": session_seed(run_seed, src["path"], w, m, v)})  # fmt: skip
    return jobs


def _n_windows(path: str, window_s: float, spine_joint: int, motion_cls) -> int:
    if path.startswith("synthetic:"):
        return motion_cls.from_spec(path, spine_joint=spine_joint).n_windows(window_s)
    with np.load(path, allow_pickle=False) as z:  # frame count without the forward kinematics
        n = int(z["poses"].shape[0])
        fps = float(z["mocap_framerate"] if "mocap_framerate" in z.files else z["mocap_frame_rate"])
    return max(0, (n - 1) // int(round(window_s * fps)))


def simulate_job(job: dict, cfg: Config, with_clip: bool = False):
    """Simulate one job's session in memory; returns (SessionData, series, query host times, Grid[, clip])."""
    from . import evaluate as ev
    from . import step0
    from .sim.motion import MotionSource
    from .sim.simulate import simulate_session

    cfg_s = cfg.with_overrides({"session.idle_s": float(job["idle_s"]), **job.get("overrides", {})})
    motion = MotionSource.from_spec(job["path"], spine_joint=int(cfg["motion.spine_joint"]))
    clip = motion.window(float(job["window_start_s"]), float(cfg["dataset.window_s"]))
    d = simulate_session(cfg_s, int(job["seed"]), clip)
    series = step0.decode(d.frames, d.session)
    grid = ev.Grid(d.labels_joints)
    host = grid.t_s + d.meta["plan"]["host_epoch_s"]
    return (d, series, host, grid, clip) if with_clip else (d, series, host, grid)


def _worker(args):
    fn, job, cfg_path, cfg_overrides = args
    cfg = load_config(cfg_path)
    if cfg_overrides:
        cfg = cfg.with_overrides(cfg_overrides)
    return fn(job, cfg)


def run_jobs(fn, jobs: list[dict], cfg_path: str | Path, cfg_overrides: dict | None = None,
             workers: int | None = None, log=print) -> list:  # fmt: skip
    """``fn(job, cfg)`` for every job, in worker processes, results in job order."""
    out = []
    args = [(fn, j, str(cfg_path), cfg_overrides or {}) for j in jobs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_worker, args, chunksize=1)):
            out.append(r)
            if (i + 1) % 100 == 0:
                log(f"  {i + 1}/{len(jobs)} sessions")
    return out
