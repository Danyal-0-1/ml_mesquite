"""The Step 2 estimator contract (Step 2 brief §5.2).

Every estimator has one form::

    estimate(series, session, t_query_host_s, cfg, *, truth=None) -> Estimate

* ``series`` is Step 0's per-pod series and ``session`` the logger's ``session.json``: exactly what a real recording
  provides. An estimator never opens ``labels_*.parquet`` or ``meta.json`` (brief rule 7).
* ``t_query_host_s`` (K,) are host times at which the scorer wants an answer (its 32 Hz steps). They are times,
  not labels.
* ``cfg`` is the ``geometry:`` section of the config only (:meth:`mesqsim.config.Config.section`): the estimator's
  own beliefs, never the simulator's ``error_model:`` (rule 8).
* ``truth`` is ``None`` for every deployable estimator. Only GEO-K receives simulator truth, through this one
  argument (:class:`GeoTruth`), and every table marks it as using simulator truth.
* **Causal and deterministic:** the answer at a query time uses only frames with ``t_host_s`` at or before it, and
  the same inputs give the same bytes (tested).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from ..body import N_JOINTS, N_PODS

__all__ = ["Estimate", "GeoTruth", "empty_estimate"]


class Estimate(NamedTuple):
    """One estimator's answer at K query times for the 14 joints (columns in :data:`mesqsim.body.JOINTS` order)."""

    delta_rad: np.ndarray  # (K, 14) relative heading delta_hat in (-pi, pi]
    kappa: np.ndarray  # (K, 14) von Mises concentration >= 0; 0 means "no information"
    uncertain: np.ndarray  # (K, 14) bool: kappa below the hold threshold (the estimate still holds)


@dataclass(frozen=True)
class GeoTruth:
    """Simulator truth that GEO-K may use, and nothing else: every pod's true mounting m_i (pod -> segment).

    The scorer builds it from ``meta.json`` (``plan.pods[i].mount_q``). The anatomical joint axes GEO-K uses are
    fixed definitions in SMPL segment frames (:mod:`mesqsim.estimators.anatomy`), expressed in pod frames with
    these mountings.
    """

    mount_q: np.ndarray  # (15, 4) (x, y, z, w)

    def __post_init__(self):
        m = np.asarray(self.mount_q, dtype=np.float64)
        if m.shape != (N_PODS, 4):
            raise ValueError(f"mount_q must be ({N_PODS}, 4), got {m.shape}")
        object.__setattr__(self, "mount_q", m)

    @classmethod
    def from_meta(cls, meta: dict) -> GeoTruth:
        """Scorer side only: read the true mountings from a session's ``meta.json``."""
        return cls(np.array([p["mount_q"] for p in meta["plan"]["pods"]], dtype=np.float64))


def empty_estimate(k: int) -> Estimate:
    """No answer at all: delta 0, kappa 0, uncertain everywhere."""
    return Estimate(np.zeros((k, N_JOINTS)), np.zeros((k, N_JOINTS)), np.ones((k, N_JOINTS), bool))
