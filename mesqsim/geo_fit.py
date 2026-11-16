"""Fitting the estimator's data-dependent beliefs on the training and validation splits (Step 2 brief rule 6).

This module reads truth (source motion, simulated labels), so it lives outside :mod:`mesqsim.estimators`; the
estimator only ever sees the fitted *values*, through the ``geometry:`` section of the config, labelled ``fitted``
with a source note naming the command, the data and the split. Nothing here ever touches the test split.

* :func:`fit_rom` (train): the range-of-motion box Omega per joint - the 0.5th and 99.5th percentiles of the
  training subjects' true Cardan angles (Step 2 brief §5.6) - and two descriptive numbers the brief asks for: how
  hinge-like the knee is (RMS of its rotation about its two other axes) and how much each 2-DoF joint turns about
  its third axis.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .body import JOINTS
from .estimators.anatomy import CARDAN_SEQ
from .estimators.rom import joint_angles_rad

__all__ = ["ROM_JOINTS", "RomFitResult", "fit_rom", "rom_source_note"]

ROM_JOINTS = tuple(k for k, j in enumerate(JOINTS) if j.jtype != "hinge")
"""Joints with a range-of-motion box: GEO-K uses it at 3-DoF joints, GEO-ID at 2-DoF and 3-DoF joints."""
_BINS = np.linspace(-180.0, 180.0, 36001)  # 0.01° histogram bins: percentiles exact to 0.01°
_TARGET_HZ = 30.0  # frames are subsampled to about this rate (neighbouring frames carry no new posture)


@dataclass
class RomFitResult:
    limits_deg: dict  # joint name -> [[lo, hi] x 3]
    n_frames: int
    n_sequences: int
    knee_off_axis_rms_deg: dict  # knee name -> (RMS of Cardan angle 2, RMS of angle 3)
    third_axis_rms_deg: dict  # 2-DoF joint name -> RMS of its third Cardan angle


def _sequence_stats(args) -> tuple[np.ndarray, np.ndarray, int]:
    """Histogram counts (14, 3, bins) and sums of squares (14, 3) of one sequence's true joint angles."""
    path, spine_joint = args
    from .sim.motion import MotionSource

    m = MotionSource.from_spec(path, spine_joint=spine_joint)
    step = max(1, int(round(m.fps / _TARGET_HZ)))
    seg = m.seg_q[::step]
    counts = np.zeros((len(JOINTS), 3, _BINS.size - 1), np.int64)
    sumsq = np.zeros((len(JOINTS), 3))
    for k in range(len(JOINTS)):
        ang = np.rad2deg(joint_angles_rad(seg, k))
        for a in range(3):
            counts[k, a] += np.histogram(ang[:, a], bins=_BINS)[0]
        sumsq[k] += np.sum(ang * ang, axis=0)
    return counts, sumsq, seg.shape[0]


def _percentile_from_hist(counts: np.ndarray, q: float) -> float:
    c = np.cumsum(counts)
    target = q / 100.0 * c[-1]
    i = int(np.searchsorted(c, target))
    prev = c[i - 1] if i > 0 else 0
    frac = (target - prev) / max(counts[i], 1)
    return float(_BINS[i] + frac * (_BINS[i + 1] - _BINS[i]))


def fit_rom(sources: pd.DataFrame, spine_joint: int, percentiles=(0.5, 99.5), workers: int | None = None,
            log=print) -> RomFitResult:  # fmt: skip
    """Omega from the **training** rows of ``sources`` (columns path, split); raises if any other split is given."""
    train = sources[sources["split"] == "train"]
    if train.empty:
        raise ValueError("no training sequences")
    args = [(p, spine_joint) for p in sorted(train["path"])]
    counts = np.zeros((len(JOINTS), 3, _BINS.size - 1), np.int64)
    sumsq = np.zeros((len(JOINTS), 3))
    n = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (c, s, k) in enumerate(ex.map(_sequence_stats, args, chunksize=8)):
            counts += c
            sumsq += s
            n += k
            if (i + 1) % 1000 == 0:
                log(f"  {i + 1}/{len(args)} sequences")
    limits = {}
    for k in ROM_JOINTS:
        limits[JOINTS[k].name] = [[round(_percentile_from_hist(counts[k, a], percentiles[0]), 2),
                                   round(_percentile_from_hist(counts[k, a], percentiles[1]), 2)] for a in range(3)]
    rms = np.sqrt(sumsq / max(n, 1))
    knees = {JOINTS[k].name: (round(float(rms[k, 1]), 3), round(float(rms[k, 2]), 3))
             for k, j in enumerate(JOINTS) if j.jtype == "hinge"}  # fmt: skip
    third = {JOINTS[k].name: round(float(rms[k, 2]), 3) for k, j in enumerate(JOINTS) if j.jtype == "2dof"}
    return RomFitResult(limits, n, len(args), knees, third)


def rom_source_note(command: str, data: str, n_seq: int, n_frames: int, percentiles, commit: str, date: str) -> str:
    return (f"fitted by `{command}` on {data}, training split only: the {percentiles[0]:g}th and {percentiles[1]:g}th "
            f"percentiles of the true Cardan angles ({{seq}}) over {n_frames:,} frames of {n_seq:,} sequences "
            f"(subsampled to about {_TARGET_HZ:g} Hz); commit {commit[:12]}, {date}")


def cardan_seq_note(joint: str) -> str:
    return CARDAN_SEQ[joint]
