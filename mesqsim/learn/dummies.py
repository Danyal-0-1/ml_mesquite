"""Two dummy models (Step 3 brief §7.4) that prove the tests work and teach nothing about networks.

* :class:`Zeros` predicts 0 with κ = 0 (and so ``uncertain``) everywhere. It meets the contract and must score
  exactly as B0 (δ̂ = 0, Step 1's "size of the problem").
* :class:`Cheat` reads the labels of its own session through :mod:`mesqsim.learn.targets`. Its error is zero, and it
  must fail the label-blindness test: run on the inputs with the labels deleted, it cannot answer.
"""

from __future__ import annotations

import numpy as np

from .contract import Prediction, SessionInputs, outputs_for

__all__ = ["Zeros", "ZerosEps", "Cheat"]


class Zeros:
    target = "delta"

    def predict(self, inputs: SessionInputs) -> Prediction:
        k = outputs_for(self.target)
        z = np.zeros((inputs.steps, k))
        return Prediction(z, z.copy(), np.ones((inputs.steps, k), bool))


class ZerosEps(Zeros):
    target = "eps"


class Cheat:
    target = "delta"

    def predict(self, inputs: SessionInputs) -> Prediction:
        import h5py

        from .targets import read_targets

        shard, sid = inputs.source.split("#", 1)
        with h5py.File(shard, "r") as h5:
            y = read_targets(h5[f"sessions/{sid}"])
        k = y["y_delta"].shape[1]
        t = inputs.steps
        return Prediction(y["y_delta"][:t].astype(float), np.full((t, k), 1e3), np.zeros((t, k), bool))
