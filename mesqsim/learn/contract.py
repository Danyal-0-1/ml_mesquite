"""The model contract (Step 3 brief §7.4), built on Step 2's estimator contract.

A model is anything with a ``target`` and one method, :meth:`Model.predict`: given one session's stored inputs and
masks (:class:`SessionInputs`, the arrays of ``twin_v1`` minus every label), it returns :class:`Prediction`, the
estimate ``theta_rad`` (δ per joint, or ε per pod), its von Mises concentration ``kappa`` and an ``uncertain`` flag
at every 32 Hz step. A live stream is the same call on the inputs received so far: a model must be **causal**, its
output at step k unchanged by any input after k (:func:`causality_error` measures it, and the scorer reports it).

Shapes (T steps; K = 14 joints in the order of :data:`mesqsim.body.JOINTS` for ``target = "delta"``, K = 15 pods
in Step 1's id order for ``target = "eps"``):

=============  ==================  ===========================================================================
``t_s``        (T,) float64         step times, s
``x_pod``      (T, 15, 8) float32   tilt (3), rate between fresh samples rad/s (3), age s (1), held (1)
``x_joint``    (T, 14, 6) float32   the first two columns of R̂_pᵀ R̂_c
``x_phone``    (T, 8) float32       the phone's eight, "orientation only" for held
``x_pod_phone``(T, 15, 6) float32   the first two columns of R̂_phᵀ R̂_i
``state_pod``  (T, 15) uint8        0 lost, 1 held, 2 fresh
``state_phone``(T,) uint8           0 no message, 1 orientation only, 2 full
``theta_rad``  (T, K) float         radians, wrapped to (-π, π]
``kappa``      (T, K) float         >= 0; 0 means no information
``uncertain``  (T, K) bool          the model's own flag
=============  ==================  ===========================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

import numpy as np

from ..body import N_JOINTS, N_PODS
from ..conventions import wrap_rad

__all__ = ["INPUT_ARRAYS", "SessionInputs", "Prediction", "Model", "outputs_for", "check_prediction", "truncate",
           "causality_error"]

INPUT_ARRAYS = ("t_s", "x_pod", "x_joint", "x_phone", "x_pod_phone", "state_pod", "state_phone")


@dataclass(frozen=True)
class SessionInputs:
    t_s: np.ndarray
    x_pod: np.ndarray
    x_joint: np.ndarray
    x_phone: np.ndarray
    x_pod_phone: np.ndarray
    state_pod: np.ndarray
    state_phone: np.ndarray
    attrs: dict = field(default_factory=dict)  # subject, split, suit_version, conditions: never a label
    source: str | None = None  # "<shard>#<session id>", for diagnostics

    @property
    def steps(self) -> int:
        return int(self.t_s.size)

    @classmethod
    def from_group(cls, group, source: str | None = None) -> SessionInputs:
        """The inputs of one stored session (an ``h5py`` group): the arrays of :data:`INPUT_ARRAYS` only."""
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in group.attrs.items()}
        return cls(*(np.asarray(group[k]) for k in INPUT_ARRAYS), attrs=attrs, source=source)


@dataclass(frozen=True)
class Prediction:
    theta_rad: np.ndarray
    kappa: np.ndarray
    uncertain: np.ndarray


@runtime_checkable
class Model(Protocol):
    target: str  # "delta" or "eps"

    def predict(self, inputs: SessionInputs) -> Prediction: ...


def outputs_for(target: str) -> int:
    if target not in ("delta", "eps"):
        raise ValueError(f"target must be 'delta' or 'eps', got {target!r}")
    return N_JOINTS if target == "delta" else N_PODS


def check_prediction(model, inputs: SessionInputs, pred: Prediction) -> None:
    """Raise unless ``pred`` meets the contract for ``model`` on ``inputs``."""
    k = outputs_for(model.target)
    shape = (inputs.steps, k)
    for name in ("theta_rad", "kappa", "uncertain"):
        a = np.asarray(getattr(pred, name))
        if a.shape != shape:
            raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
    th, ka = np.asarray(pred.theta_rad, float), np.asarray(pred.kappa, float)
    if not np.all(np.isfinite(th)) or not np.all(np.isfinite(ka)):
        raise ValueError("theta_rad and kappa must be finite")
    if np.any(ka < 0):
        raise ValueError("kappa must be >= 0")
    if np.any(np.abs(wrap_rad(th) - th) > 1e-9):
        raise ValueError("theta_rad must be wrapped to (-pi, pi]")
    if np.asarray(pred.uncertain).dtype != bool:
        raise ValueError("uncertain must be boolean")


def truncate(inputs: SessionInputs, k: int) -> SessionInputs:
    """The inputs as a live stream holds them at step ``k``: steps 0 ... k."""
    return replace(inputs, **{n: getattr(inputs, n)[: k + 1] for n in INPUT_ARRAYS})


def causality_error(model, inputs: SessionInputs, steps) -> float:
    """The largest |Δθ| (rad) at the given steps between the whole session's prediction and the stream's."""
    full = model.predict(inputs)
    worst = 0.0
    for k in steps:
        part = model.predict(truncate(inputs, int(k)))
        worst = max(worst, float(np.max(np.abs(wrap_rad(np.asarray(full.theta_rad)[k]
                                                         - np.asarray(part.theta_rad)[k])))))  # fmt: skip
    return worst
