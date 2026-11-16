"""A von Mises filter per joint: the estimator's memory of the relative heading (Step 2 brief §5.7).

Each joint carries a belief VM(mu, kappa) about its relative heading delta:

* **Start** at the end of the box phase from box registration (B1), with ``kappa0 = 1/sigma_pair²``.
* **Predict** between events: the north of each pod slides, so the belief widens,
  ``A(kappa) <- A(kappa)·exp(-q·dt/2)`` (:func:`mesqsim.conventions.vm_predict_kappa`); for large kappa this is
  ``kappa <- 1/(1/kappa + q·dt)``. The mean does not move: drift has no known direction.
* **Update** with a window's estimate (delta_w, kappa_w): the product of the two von Mises densities,
  ``kappa·e^{i mu} + kappa_w·e^{i delta_w}`` (:func:`mesqsim.conventions.vm_product`).
* **Reset** a joint to kappa = 0 when either of its pods reboots: its new north is unknown.

The filter records its state after every event, so the answer at any query time is the latest event at or before
that time, predicted forward to it: causal by construction.
"""

from __future__ import annotations

import numpy as np

from ..conventions import vm_predict_kappa, vm_product, wrap_rad

__all__ = ["JointFilter"]


class JointFilter:
    """Beliefs VM(mu_j, kappa_j) for ``n`` joints, advanced through time-ordered events."""

    def __init__(self, n: int, q_rad2_per_s: float):
        self.n = n
        self.q = float(q_rad2_per_s)
        self.t = -np.inf
        self.mu = np.zeros(n)
        self.kappa = np.zeros(n)
        self._t: list[float] = []
        self._mu: list[np.ndarray] = []
        self._kappa: list[np.ndarray] = []

    # -------------------------------------------------------------------- events
    def _advance(self, t: float) -> None:
        if t < self.t:
            raise ValueError(f"events must be time-ordered: {t} < {self.t}")
        if np.isfinite(self.t):
            self.kappa = vm_predict_kappa(self.kappa, self.q, t - self.t)
        self.t = float(t)

    def _record(self) -> None:
        self._t.append(self.t)
        self._mu.append(self.mu.copy())
        self._kappa.append(self.kappa.copy())

    def init(self, t: float, mu_rad: np.ndarray, kappa: np.ndarray) -> None:
        """Set the belief at time t (the end of the box phase)."""
        self._advance(t)
        self.mu = wrap_rad(np.asarray(mu_rad, dtype=np.float64)).copy()
        self.kappa = np.asarray(kappa, dtype=np.float64).copy()
        self._record()

    def reset(self, t: float, joints) -> None:
        """kappa = 0 for ``joints`` at time t (a pod of theirs rebooted); the mean is kept but carries no weight."""
        self._advance(t)
        self.kappa[np.asarray(joints, dtype=np.int64)] = 0.0
        self._record()

    def update(self, t: float, delta_w_rad: np.ndarray, kappa_w: np.ndarray) -> None:
        """Multiply in one window's von Mises estimate for every joint (kappa_w = 0 leaves a joint unchanged)."""
        self._advance(t)
        self.mu, self.kappa = vm_product(self.mu, self.kappa, np.asarray(delta_w_rad), np.asarray(kappa_w))
        self._record()

    # -------------------------------------------------------------------- outputs
    def query(self, t_query_s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Belief at each query time: the latest event at or before it, predicted forward.

        Returns ``(mu (K, n), kappa (K, n), defined (K,))``; ``defined`` is False before the first event.
        """
        t_q = np.asarray(t_query_s, dtype=np.float64)
        k = t_q.size
        if not self._t:
            return np.zeros((k, self.n)), np.zeros((k, self.n)), np.zeros(k, bool)
        times = np.asarray(self._t)
        idx = np.searchsorted(times, t_q, side="right") - 1
        defined = idx >= 0
        ii = np.clip(idx, 0, None)
        mu = np.stack(self._mu)[ii]
        kap = np.stack(self._kappa)[ii]
        kap = vm_predict_kappa(kap, self.q, (t_q - times[ii])[:, None])
        mu[~defined] = 0.0
        kap[~defined] = 0.0
        return mu, kap, defined
