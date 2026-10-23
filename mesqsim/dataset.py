"""Generate a simulated dataset: many sessions from many sequences, each in its sequence's split (brief §5.1, §7 M5).

For every source sequence (an AMASS file from the split manifest, or a synthetic stand-in), the generator cuts
whole windows of ``dataset.window_s`` and, for every window, simulates one session per value of
``dataset.motion_start_min`` (the motion starts that many minutes after boot; the unrecorded idle interval fills the
gap) and per variant. The session's seed is derived from (run seed, sequence, window, start, variant), so every
session can be regenerated on its own.

Output: a run folder (named after commit, config hash and seed) with ``sessions/<id>/`` for every session and
``dataset.csv`` describing them (id, source, subset, subject, split, window, motion start, variant, seed).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from . import io
from .config import Config
from .data import amass
from .pipeline import session_meta
from .runs import git_commit, run_folder
from .sim.motion import MotionSource
from .sim.simulate import simulate_session
from .splits import assign_splits

__all__ = ["synthetic_sources", "amass_sources", "session_seed", "pre_motion_s", "generate"]

SYNTH_KINDS = ("walk", "circle", "stand")


def synthetic_sources(subjects_per_kind: int = 4, duration_s: float = 60.0) -> pd.DataFrame:
    """Stand-in sequences when AMASS is absent: one 'subject' per (kind, k), split like AMASS subjects."""
    rows = [
        {"path": f"synthetic:{kind}:{duration_s:g}:{k}", "subset": "SYNTH", "subject": f"{kind}{k:02d}"}
        for kind in SYNTH_KINDS
        for k in range(subjects_per_kind)
    ]
    return assign_splits(pd.DataFrame(rows))


def amass_sources(manifest: Path) -> pd.DataFrame:
    """Rows of the split manifest, with paths made absolute under the AMASS folder."""
    df = pd.read_csv(manifest)
    df["path"] = [str(amass.amass_dir() / p) for p in df["path"]]
    return df


def session_seed(seed: int, source: str, window: int, start_min: float, variant: int) -> int:
    blob = f"{seed}|{source}|{window}|{start_min:g}|{variant}".encode()
    return int(hashlib.sha256(blob).hexdigest()[:8], 16)


def pre_motion_s(cfg: Config) -> float:
    return cfg["session.box_s"] + cfg["session.strapping_s"] + cfg["session.tpose_s"]


def _session_id(row, window: int, start_min: float, variant: int) -> str:
    path = str(row["path"])
    stem = path.replace(":", "-") if path.startswith("synthetic:") else Path(path).stem
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in f"{row['subset']}_{row['subject']}_{stem}")
    return f"{safe}_w{window}_m{start_min:g}_v{variant}"


def generate(cfg: Config, seed: int, sources: pd.DataFrame, splits=("train", "val", "test", "holdout"),
             out_dir: Path | None = None, max_sequences: int | None = None, log=print) -> Path:  # fmt: skip
    """Simulate every session of the dataset into a run folder; return the folder."""
    commit = git_commit()
    folder = Path(out_dir) if out_dir is not None else run_folder(commit, cfg.hash(), seed)
    window_s = float(cfg["dataset.window_s"])
    pre = pre_motion_s(cfg)
    starts = [float(m) for m in cfg["dataset.motion_start_min"]]
    for m in starts:
        if 60.0 * m < pre:
            raise ValueError(f"motion cannot start {m} min after boot: box + strapping + T-pose take {pre:.0f} s")
    chosen = sources[sources["split"].isin(splits)].sort_values("path", ignore_index=True)
    if max_sequences is not None:
        chosen = chosen.head(max_sequences)
    rows = []
    for _, src in chosen.iterrows():
        motion = MotionSource.from_spec(src["path"], spine_joint=int(cfg["motion.spine_joint"]))
        n_win = min(motion.n_windows(window_s), int(cfg["dataset.windows_per_sequence"]))
        for w in range(n_win):
            clip = motion.window(w * window_s, window_s)
            for m in starts:
                for v in range(int(cfg["dataset.variants"])):
                    sid = _session_id(src, w, m, v)
                    s_seed = session_seed(seed, src["path"], w, m, v)
                    cfg_s = cfg.with_overrides({"session.idle_s": 60.0 * m - pre})
                    data = simulate_session(cfg_s, s_seed, clip)
                    meta = session_meta(cfg_s, s_seed, commit, clip, src["path"], data, split=src["split"])
                    meta["dataset"] = {"run_seed": seed, "window": w, "window_start_s": w * window_s,
                                       "motion_start_min": m, "variant": v}  # fmt: skip
                    io.write_session(folder / "sessions" / sid, data, meta)
                    rows.append({"session": sid, "source": src["path"], "subset": src["subset"],
                                 "subject": src["subject"], "split": src["split"], "window": w,
                                 "window_start_s": w * window_s, "motion_start_min": m, "variant": v,
                                 "seed": s_seed})  # fmt: skip
        log(f"{src['path']}: {n_win} window(s) x {len(starts)} start(s) x {cfg['dataset.variants']} variant(s)")
    manifest = pd.DataFrame(rows)
    manifest["source"] = [s if s.startswith("synthetic:") else Path(s).name for s in manifest["source"]]
    manifest.to_csv(folder / "dataset.csv", index=False, lineterminator="\n")
    io.write_json(folder / "run.json", {"commit": commit, "config_hash": cfg.hash(), "seed": seed,
                                        "config": cfg.values(), "splits": list(splits),
                                        "sources": "synthetic" if chosen["path"].str.startswith("synthetic:").all()
                                        else "amass", "n_sessions": len(manifest)})  # fmt: skip
    return folder
