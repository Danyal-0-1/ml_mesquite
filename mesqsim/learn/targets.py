"""The targets (Step 3 brief §6.1): δ_pc per joint and ε_i per pod at every 32 Hz step, with validity masks.

This is the only module that reads the ``y_*`` arrays of the dataset (and the label tables they come from), so a
model, a loader or a feature can be checked for label-blindness by asking whether it imports it (``Cheat`` does).

* ``y_delta`` (T, 14) float32 rad: δ_pc = wrap(ψ_c - ψ_p), from ``labels_joints``; ``valid_delta`` where both pods are
  on and not inside a ``transition`` blend.
* ``y_eps`` (T, 15) float32 rad: ε_i = wrap(ψ_i - ψ_ph), from ``labels_steps``' ψ columns; ``valid_eps`` where the pod
  is on and the phone has been zeroed, and not inside a blend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..body import N_JOINTS, N_PODS
from ..conventions import wrap_rad

__all__ = ["session_targets", "read_targets"]


def session_targets(labels_joints: pd.DataFrame, labels_steps: pd.DataFrame) -> dict:
    """The four target arrays of one twin session, on ``labels_steps``' step grid."""
    lj = labels_joints.sort_values(["step", "joint"], kind="mergesort")
    t = len(labels_steps)
    delta = lj["delta_rad"].to_numpy(dtype=np.float64).reshape(t, N_JOINTS)
    psi = np.stack([labels_steps[f"psi{i}_rad"].to_numpy(dtype=np.float64) for i in range(N_PODS)], axis=1)
    psi_ph = labels_steps["psi_ph_rad"].to_numpy(dtype=np.float64)
    eps = wrap_rad(psi - psi_ph[:, None])
    blend = labels_steps["transition"].to_numpy(dtype=bool)[:, None]
    return {"y_delta": np.nan_to_num(delta).astype(np.float32), "valid_delta": np.isfinite(delta) & ~blend,
            "y_eps": np.nan_to_num(eps).astype(np.float32), "valid_eps": np.isfinite(eps) & ~blend}  # fmt: skip


def read_targets(group) -> dict:
    """The targets of one stored session (an ``h5py`` group of the dataset, ``/sessions/<id>/``)."""
    return {k: np.asarray(group[k]) for k in ("y_delta", "valid_delta", "y_eps", "valid_eps")}
