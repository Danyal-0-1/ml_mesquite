"""Run identity and run folders (brief §6, §8).

A run is identified by ``(git commit, config hash, seed)`` and its folder is named after it:
``<output dir>/run_g<commit 7>_c<config hash 8>_s<seed>``. The commit gets a ``+dirty`` mark (``-dirty`` in the
folder name) when the working tree had uncommitted changes, so a result can always be traced to code.
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .paths import REPO_ROOT, output_dir

__all__ = ["git_commit", "run_id", "run_folder", "package_versions", "list_runs", "remove_run"]

PACKAGES = ("numpy", "scipy", "pandas", "pyarrow", "PyYAML", "vqf", "imt-diodem")


def git_commit() -> str:
    """Full commit hash of HEAD, with ``+dirty`` if the working tree differs from it (untracked files too)."""
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        # untracked files count too: a new module can change results before it is committed
        status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True,
                                check=True)  # fmt: skip
        return sha.stdout.strip() + ("+dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_id(commit: str, config_hash: str, seed: int) -> str:
    short = commit.split("+")[0][:7] + ("-dirty" if "+dirty" in commit else "")
    return f"run_g{short}_c{config_hash[:8]}_s{seed}"


def run_folder(commit: str, config_hash: str, seed: int, root: Path | None = None) -> Path:
    return (output_dir() if root is None else root) / run_id(commit, config_hash, seed)


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in PACKAGES:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    return out


def list_runs(root: Path | None = None) -> list[Path]:
    root = output_dir() if root is None else root
    return sorted(p for p in root.glob("run_*") if p.is_dir()) if root.is_dir() else []


def remove_run(path: Path, root: Path | None = None) -> None:
    """Delete one run folder; refuses anything outside the output directory or not named ``run_*``."""
    root = (output_dir() if root is None else root).resolve()
    path = Path(path).resolve()
    if path.parent != root or not path.name.startswith("run_"):
        raise ValueError(f"refusing to delete {path}: not a run folder directly under {root}")
    shutil.rmtree(path)
