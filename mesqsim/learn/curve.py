"""How much training data is enough (Step 3 brief §6.3): the learning curve and the sufficiency rule.

Danyal trains a model on the nested parts of the training subjects (1/16, 1/8, 1/4, 1/2, all; ``index.parquet``),
for three seeds, and measures the validation median error at each training size n (hours). :func:`fit` fits

    err(n) = c + a · n^(-b)          (Hestness et al. 2017; a, b > 0, c >= 0)

by least squares, and :func:`sufficient` applies the rule: the data are **sufficient** when

1. one more doubling is predicted to improve the error by less than both ``gain_deg`` (0.1°) and ``gain_rel`` (2 %)
   of it, for the seeds' mean fit; and
2. the three seeds agree within their spread: each seed's own predicted gain is below both limits, and the seeds'
   errors at the largest size differ by more than the mean predicted gain (so the doubling would move the error by
   less than the seeds already disagree).

The two thresholds are ``twin_dataset.sufficient_gain_deg`` and ``twin_dataset.sufficient_gain_rel`` (`assumption`).
Part 2 is this module's reading of "three seeds agree within their spread"; Danyal may tighten it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit

__all__ = ["Curve", "power_law", "fit", "predicted_gain", "sufficient"]


@dataclass(frozen=True)
class Curve:
    c: float
    a: float
    b: float

    def __call__(self, n) -> np.ndarray:
        return power_law(np.asarray(n, dtype=float), self.c, self.a, self.b)


def power_law(n, c, a, b):
    return c + a * np.power(n, -b)


def fit(hours, err) -> Curve:
    """Least-squares fit of err(n) = c + a n^-b with a, b > 0 and c >= 0."""
    n = np.asarray(hours, dtype=float)
    e = np.asarray(err, dtype=float)
    if n.size < 3:
        raise ValueError("need at least three training sizes")
    p0 = (max(e.min() * 0.8, 0.0), max(e.max() - e.min(), 1e-3) * n.min() ** 0.5, 0.5)
    popt, _ = curve_fit(power_law, n, e, p0=p0, bounds=([0.0, 0.0, 1e-3], [np.inf, np.inf, 5.0]), maxfev=20000)
    return Curve(*map(float, popt))


def predicted_gain(curve: Curve, n_max: float) -> tuple[float, float]:
    """(degrees, relative): how much one more doubling beyond ``n_max`` is predicted to lower the error."""
    now, then = float(curve(n_max)), float(curve(2.0 * n_max))
    return now - then, (now - then) / now if now > 0 else 0.0


def sufficient(hours, errs_by_seed, gain_deg: float = 0.1, gain_rel: float = 0.02) -> dict:
    """The rule of the module docstring for the errors ``errs_by_seed`` (seeds x sizes) at sizes ``hours``."""
    n = np.asarray(hours, dtype=float)
    e = np.asarray(errs_by_seed, dtype=float)
    fits = [fit(n, row) for row in e]
    gains = [predicted_gain(f, n.max()) for f in fits]
    mean_fit = fit(n, e.mean(axis=0))
    g_deg, g_rel = predicted_gain(mean_fit, n.max())
    part1 = g_deg < gain_deg and g_rel < gain_rel
    each = all(gd < gain_deg and gr < gain_rel for gd, gr in gains)
    spread = float(e[:, int(np.argmax(n))].max() - e[:, int(np.argmax(n))].min())
    part2 = each and spread >= g_deg
    return {"sufficient": bool(part1 and part2), "gain_deg": g_deg, "gain_rel": g_rel,
            "seed_gains_deg": [g[0] for g in gains], "spread_deg": spread, "fit": mean_fit.__dict__,
            "part1": bool(part1), "part2": bool(part2)}
