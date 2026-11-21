"""The chip's six-axis fusion, in the loop (Step 3 brief §4.5): three modes, ``twin.fusion.mode``.

* ``vqf6d``: VQF (Laidig and Seel 2023) in six-axis mode, the pinned ``vqf`` package. Its first sample aligns the
  estimate with the measured gravity by the shortest arc, which has no twist about the vertical: yaw 0 in its own
  frame, the Game Rotation Vector's arbitrary north. ``updateBatch`` returns ``quat6D`` scalar first (w, x, y, z),
  sensor -> earth, +Z up; it is converted here to (x, y, z, w).
* ``mahony``: Mahony, Hamel and Pflimlin (2008), the explicit complementary filter, written out below with exact
  exponential integration, so that no result hangs on one filter.
* ``analytic``: Step 1's model, unchanged: q = Y(-ψ) · R · Exp(η), with ψ from Step 1's :class:`PodHeading`
  (offset, drift, scale × turning, angle random walk) and η Step 1's Gauss-Markov tilt error. It reads the truth;
  the other two read only the samples of :mod:`.sensors`.

A filter starts at its pod's boot and restarts at a reboot, so a new north is wherever the pod faces then. It runs
at ``twin.imu.oversample`` × the chip's rate and is read at the chip's sample times; reports are canonical (w >= 0).

**Mahony's step**, predict then correct, so the accelerometer is compared with the estimate at its own instant:
q ← q ⊗ Exp((ω̃ + b̂) Δt); then, with â the measured specific force made unit (up, when still) and v = R̂ᵀ ẑ the
estimate's up in the pod's frame, e = â × v, b̂ += k_I e Δt and q ← q ⊗ Exp(k_P e Δt). A rotation q ⊗ Exp(δ) moves v
by v × δ, so δ ∝ â × v turns v toward â; in the world, R̂ (â × v) = R̂ â × ẑ is horizontal, so the correction, like
VQF's, is a rotation about a horizontal axis (it leaves ψ alone to first order, :mod:`.labels`). With perfect
samples e = 0 and the estimate follows the truth exactly.

**Signs the checks pin** (q = Y(-ψ) · R, :mod:`.labels`). A gyroscope bias b on the pod axis that stays vertical
integrates into a report turned by +b t about +Z: ψ(t) = -b t. A scale error k on that axis over a turn Θ reports
(1 + k) Θ: ψ = -k Θ. Step 1's ψ = a + b_1 t + k_1 Θ therefore has b_1 = -b and k_1 = -k: Step 1 drew both
symmetric about 0, so its distributions are unchanged; only the sign convention differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import Config
from ..conventions import exp_map, quat_canonical, quat_mul, wxyz_to_xyzw, yaw_quat
from ..sim.heading import Epoch, PodHeading, random_walk_rad, theta_increments_rad
from ..sim.simulate import fusion_error

__all__ = ["MODES", "FusionResult", "vqf6d", "mahony", "fuse", "align_to_gravity", "fusion_times",
           "analytic_heading", "analytic_reports", "fusion_error"]

MODES = ("analytic", "vqf6d", "mahony")


@dataclass
class FusionResult:
    q: np.ndarray  # (N, 4) pod -> its private world, (x, y, z, w), canonical
    bias_rad_s: np.ndarray | None = None  # (N, 3) the filter's gyroscope-bias estimate
    rest: np.ndarray | None = None  # (N,) VQF's rest detection


def fusion_times(t_first_s: float, n_samples: int, chip_rate_hz: float, oversample: int = 1) -> np.ndarray:
    """The filter's step times for one epoch: ``oversample`` steps per chip sample; every ``oversample``-th step,
    from the first, is a chip sample t_first + n / f."""
    m = int(oversample)
    return t_first_s + np.arange((n_samples - 1) * m + 1) / (m * chip_rate_hz)


def vqf6d(gyr, acc, dt_s: float, tau_acc_s: float = 3.0, rest_bias: bool = True,
          motion_bias: bool = True) -> FusionResult:  # fmt: skip
    """VQF's six-axis orientation for samples ``gyr`` (rad/s) and ``acc`` (m/s²) ``(N, 3)``, every ``dt_s``."""
    from vqf import VQF

    v = VQF(float(dt_s), tauAcc=float(tau_acc_s), restBiasEstEnabled=bool(rest_bias),
            motionBiasEstEnabled=bool(motion_bias))  # fmt: skip
    out = v.updateBatch(np.ascontiguousarray(gyr, dtype=np.float64), np.ascontiguousarray(acc, dtype=np.float64))
    return FusionResult(quat_canonical(wxyz_to_xyzw(out["quat6D"])), np.asarray(out["bias"]),
                        np.asarray(out["restDetected"], bool))  # fmt: skip


def align_to_gravity(acc) -> np.ndarray:
    """The shortest-arc orientation taking a measured specific force ``acc`` (3,) to +Z: no twist about +Z."""
    a = np.asarray(acc, dtype=np.float64)
    a = a / np.linalg.norm(a)
    if a[2] < -1.0 + 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([a[1], -a[0], 0.0, 1.0 + a[2]])  # (a x z, 1 + a . z)
    return q / np.linalg.norm(q)


def _step(x, y, z, w, gx, gy, gz, dt_s):
    """(x, y, z, w) ⊗ Exp((gx, gy, gz) Δt), normalised."""
    rate = math.sqrt(gx * gx + gy * gy + gz * gz)
    half = 0.5 * rate * dt_s
    s = 0.5 * dt_s if half < 1e-12 else math.sin(half) / rate
    dx, dy, dz, dw = gx * s, gy * s, gz * s, math.cos(half)
    x, y, z, w = (w * dx + x * dw + y * dz - z * dy, w * dy - x * dz + y * dw + z * dx,
                  w * dz + x * dy - y * dx + z * dw, w * dw - x * dx - y * dy - z * dz)  # fmt: skip
    nq = math.sqrt(x * x + y * y + z * z + w * w)
    return x / nq, y / nq, z / nq, w / nq


def mahony(gyr, acc, dt_s: float, kp: float, ki: float, q0=None) -> FusionResult:
    """Mahony's complementary filter (module docstring) on samples ``(N, 3)``; starts aligned to ``acc[0]``."""
    gyr = np.asarray(gyr, dtype=np.float64)
    acc = np.asarray(acc, dtype=np.float64)
    n = gyr.shape[0]
    out = np.empty((n, 4))
    bias = np.empty((n, 3))
    x, y, z, w = (align_to_gravity(acc[0]) if q0 is None else np.asarray(q0, float))
    bx = by = bz = 0.0
    out[0] = (x, y, z, w)
    bias[0] = (0.0, 0.0, 0.0)
    for k in range(1, n):
        gx, gy, gz = gyr[k]
        x, y, z, w = _step(x, y, z, w, gx + bx, gy + by, gz + bz, dt_s)  # predict
        ax, ay, az = acc[k]
        an = math.sqrt(ax * ax + ay * ay + az * az)
        if an > 0.0:  # correct
            ax, ay, az = ax / an, ay / an, az / an
            vx, vy, vz = 2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)
            ex, ey, ez = ay * vz - az * vy, az * vx - ax * vz, ax * vy - ay * vx
            if ki > 0.0:
                bx, by, bz = bx + ki * ex * dt_s, by + ki * ey * dt_s, bz + ki * ez * dt_s
            if kp > 0.0:
                x, y, z, w = _step(x, y, z, w, kp * ex, kp * ey, kp * ez, dt_s)
        out[k] = (x, y, z, w)
        bias[k] = (-bx, -by, -bz)  # the integral cancels the bias: its estimate is minus the integral
    return FusionResult(quat_canonical(out), bias)


def fuse(cfg: Config, gyr, acc, dt_s: float, mode: str | None = None, bias_estimation: bool = True) -> FusionResult:
    """Run ``mode`` (default ``twin.fusion.mode``; not ``analytic``) on one epoch's samples.

    ``bias_estimation=False`` switches every filter's own bias estimate off (VQF's rest and motion estimation,
    Mahony's integral), as the brief's bias check asks."""
    mode = cfg["twin.fusion.mode"] if mode is None else mode
    f = "twin.fusion."
    if mode == "vqf6d":
        return vqf6d(gyr, acc, dt_s, cfg[f + "vqf_tau_acc_s"], bias_estimation and cfg[f + "vqf_rest_bias"],
                     bias_estimation and cfg[f + "vqf_motion_bias"])  # fmt: skip
    if mode == "mahony":
        return mahony(gyr, acc, dt_s, cfg[f + "mahony_kp"], cfg[f + "mahony_ki"] if bias_estimation else 0.0)
    if mode == "analytic":
        raise ValueError("analytic mode reads the truth, not samples: use analytic_heading and analytic_reports")
    raise ValueError(f"unknown fusion mode {mode!r}; known: {MODES}")


def analytic_heading(t_dense_s, r_dense, boot_s: float, end_s: float, first_sample_s: float, offset_rad: float,
                     drift_rad_per_s: float, scale_factor: float, arw_rad_per_sqrt_s: float,
                     rng: np.random.Generator) -> PodHeading:  # fmt: skip
    """Step 1's ψ_i(t) for one boot epoch of a twin trajectory: Θ from the true orientations ``r_dense`` on the dense
    grid ``t_dense_s`` (Step 1 uses 240 Hz), a random walk on the same grid, Step 1's :class:`PodHeading`."""
    t = np.asarray(t_dense_s, dtype=np.float64)
    theta = np.r_[0.0, np.cumsum(theta_increments_rad(np.asarray(r_dense, dtype=np.float64)))]
    epoch = Epoch(boot_s=boot_s, end_s=end_s, first_sample_s=first_sample_s, offset_rad=offset_rad)
    return PodHeading(t, theta, random_walk_rad(t, arw_rad_per_sqrt_s, rng), drift_rad_per_s, scale_factor, [epoch])


def analytic_reports(r_true, psi_rad, eta_rad=None) -> np.ndarray:
    """Step 1's report q = canonical(Y(-ψ) · R · Exp(η)) (``simulate.py``), η = 0 when ``eta_rad`` is None."""
    q = quat_mul(yaw_quat(-np.asarray(psi_rad, dtype=np.float64)), np.asarray(r_true, dtype=np.float64))
    if eta_rad is not None:
        q = quat_mul(q, exp_map(eta_rad))
    return quat_canonical(q)
