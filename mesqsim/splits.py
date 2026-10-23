"""Train / validation / test splits by subject (brief §5.1).

* Within each AMASS subset, whole **subjects** go to one split: 70 % train, 15 % val, 15 % test by count, with the
  largest-remainder rule (so a subset with 3 subjects gets 2 / 0 / 1 and nothing is lost to rounding).
* Subjects are ordered by SHA-256 of ``"<salt>:<subset>:<subject>"`` before they are cut, so the assignment does
  not depend on file order or on any random state, and adding a new subset does not move existing subjects.
* **TotalCapture** is never trained on: all its subjects go to the split ``holdout`` (a possible later test set).
* Every window and corruption variant of a sequence inherits the sequence's split: the dataset generator looks the
  split up here and never draws it.

The manifest (``manifests/split_manifest.csv``) lists file paths only (no data). Its SHA-256 goes into WORKLOG.md
and ``manifests/split_manifest.sha256``; once Danyal freezes it, check 9.2 fails if it ever changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["SALT", "FRACTIONS", "HOLDOUT_SUBSETS", "assign_splits", "write_manifest", "manifest_sha256",
           "check_manifest"]  # fmt: skip

SALT = "mesquite-step1-splits-v1"
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
HOLDOUT_SUBSETS = {"totalcapture"}  # lower-case subset names never used for training
COLUMNS = ["path", "subset", "subject", "split"]


def _order_key(subset: str, subject: str) -> str:
    return hashlib.sha256(f"{SALT}:{subset}:{subject}".encode()).hexdigest()


def _counts(n: int) -> dict[str, int]:
    """Largest-remainder apportionment of n subjects to the split fractions."""
    raw = {k: f * n for k, f in FRACTIONS.items()}
    out = {k: int(np.floor(v)) for k, v in raw.items()}
    rest = n - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - out[k]), list(FRACTIONS).index(k)))[:rest]:
        out[k] += 1
    return out


def assign_splits(files: pd.DataFrame) -> pd.DataFrame:
    """``files`` has columns path, subset, subject; returns the same rows plus ``split``, sorted by path."""
    rows = []
    for subset, g in files.groupby("subset", sort=True):
        subjects = sorted(g["subject"].unique(), key=lambda s: _order_key(subset, s))
        if subset.lower() in HOLDOUT_SUBSETS:
            which = {s: "holdout" for s in subjects}
        else:
            counts = _counts(len(subjects))
            which, k = {}, 0
            for split in FRACTIONS:
                for s in subjects[k : k + counts[split]]:
                    which[s] = split
                k += counts[split]
        gg = g.copy()
        gg["split"] = gg["subject"].map(which)
        rows.append(gg)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=COLUMNS)
    return out[COLUMNS].sort_values("path", ignore_index=True)


def write_manifest(df: pd.DataFrame, path: Path) -> str:
    """Write the manifest CSV (LF, sorted) and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].sort_values("path").to_csv(path, index=False, lineterminator="\n")
    return manifest_sha256(path)


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_manifest(df: pd.DataFrame) -> list[str]:
    """Problems with a split manifest (empty list = fine)."""
    problems = []
    per_subject = df.groupby(["subset", "subject"])["split"].nunique()
    for (subset, subject), n in per_subject[per_subject > 1].items():
        problems.append(f"{subset}/{subject} appears in {n} splits")
    tc = df[df["subset"].str.lower().isin(HOLDOUT_SUBSETS)]
    if (tc["split"] == "train").any():
        problems.append("a TotalCapture sequence is in the training split")
    if df["path"].duplicated().any():
        problems.append("a sequence is listed twice")
    return problems
