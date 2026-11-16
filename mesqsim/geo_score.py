"""Scorer-side checks of the Step 2 estimators. This module **may read the labels** (it is part of the scorer);
no estimator imports it.

* :func:`pair_timing_errors_ms`: how far apart in true time the two halves of each parent/child pair were
  (Step 2 brief §5.3): the child's true time at the slerp point minus the parent's true sample time.
* :func:`truncate_frames`: a recording cut at a host time, for the causality check.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["pair_timing_errors_ms", "truncate_frames", "TIMING_BOUND_MS"]

TIMING_BOUND_MS = 15.0
"""Bound for the differential timing error of a pair: the decode delay spread of 0-11 ms (Step 1 brief §3.3; a
sample's ``ms_lo`` is its decode time, not its sample time, and the two pods' delays are independent) plus 4 ms
for the clock fit."""


def pair_timing_errors_ms(diag, series: pd.DataFrame, labels_ticks: pd.DataFrame) -> np.ndarray:
    """True-time error (ms) of every pair the estimator formed: child time at the slerp point minus parent time.

    ``diag`` is the :class:`mesqsim.estimators.geo.GeoDiagnostics` of a run with ``diagnostics=True``. Rows are
    joined to the labels on (pod, epoch, tick), the labels' ``t_sample_s`` being the true sample instant.
    """
    key = ["pod_id", "epoch", "tick"]
    t_sample = labels_ticks.set_index(key)["t_sample_s"]
    ids = series[key]

    def true_t(rows: np.ndarray) -> np.ndarray:
        return t_sample.loc[pd.MultiIndex.from_frame(ids.iloc[rows])].to_numpy()

    out = []
    for _t, _j, prow, crow0, crow1, u in diag.pairs:
        if len(prow) == 0:
            continue
        tp, tc0, tc1 = true_t(prow), true_t(crow0), true_t(crow1)
        out.append(1e3 * (tc0 + u * (tc1 - tc0) - tp))
    return np.concatenate(out) if out else np.zeros(0)


def truncate_frames(frames: pd.DataFrame, t_cut_host_s: float) -> pd.DataFrame:
    """The frames table as a logger would have it at host time ``t_cut``: every later frame deleted."""
    return frames[frames["t_host_s"].to_numpy() <= t_cut_host_s].reset_index(drop=True)
