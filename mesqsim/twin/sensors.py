"""What the chip's gyroscope and accelerometer report: the truth of :mod:`.physics` with errors (Step 3 brief §4.4).

    ω̃ = (I + S_g) A_g ω + b_g(t) + n_g          f̃ = (I + S_a) A_a f + b_a + n_a
    b_g(t) = b_g,0 + (a random walk of density B_g)

* S: diagonal scale errors, each axis uniform within ±max; A: a small misalignment, a rotation Exp(a) of the
  sensing triad with each component of a uniform within ±max; b: biases.
* n_g, n_a: white, densities N_g (rad/s/√Hz) and N_a (m/s²/√Hz). Sampled every Δt, each sample has σ = N / √Δt, so
  the integrated gyroscope noise is an angle random walk N_g √t (Step 1's ``arw_deg_per_sqrt_s``).
* B_g: the bias moves by N(0, B_g² Δt) per sample.
* Every axis is then rounded to the step of a signed ``adc_bits`` sample over its full scale and clipped to it.

**The ideal gyroscope sample.** A rate gyroscope sampled every Δt behind its low-pass filter reports, at best, the
mean rate over the interval. The twin uses exactly that, ω_k = Log(R_{k-1}⁻¹ R_k) / Δt in the pod's frame, so a
filter that integrates q_k = q_{k-1} Exp(ω_k Δt), as VQF and :func:`.fusion.mahony` do, follows the truth exactly
when nothing else is wrong; every heading error then comes from an error term, not from the sampling. How the DMP
integrates within its cycle is not documented (``twin.imu.oversample`` sweeps it).

Values and labels: ``configs/twin.yaml`` ``twin.sensors``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..config import Config
from ..conventions import as_quat, exp_map, log_map, quat_conj, quat_mul, quat_rotate

__all__ = ["G0_M_S2", "SensorErrors", "draw_sensor_errors", "no_sensor_errors", "ideal_gyro_rad_s", "measure"]

G0_M_S2 = 9.80665
"""Standard gravity, for full scales given in g."""


@dataclass
class SensorErrors:
    """One pod's sensor errors (radians, seconds, metres). ``None`` ranges mean no clipping or rounding."""

    gyro_scale: np.ndarray  # (3,) the diagonal of S_g
    gyro_misalign: np.ndarray  # (4,) A_g as a quaternion
    gyro_bias0_rad_s: np.ndarray  # (3,) b_g,0
    gyro_bias_walk_rad_s_sqrt_s: float  # B_g
    gyro_noise_rad_s_sqrt_hz: float  # N_g
    accel_scale: np.ndarray
    accel_misalign: np.ndarray
    accel_bias_m_s2: np.ndarray
    accel_noise_m_s2_sqrt_hz: float
    gyro_range_rad_s: float | None = None
    accel_range_m_s2: float | None = None
    adc_bits: int | None = None

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self).items()}


def no_sensor_errors(ranges: bool = False, cfg: Config | None = None) -> SensorErrors:
    """Perfect sensors; with ``ranges`` (from ``cfg``), still clipped and rounded like the chip's."""
    z3, ident = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
    out = SensorErrors(z3.copy(), ident.copy(), z3.copy(), 0.0, 0.0, z3.copy(), ident.copy(), z3.copy(), 0.0)
    if ranges:
        if cfg is None:
            raise ValueError("ranges need the twin config")
        out.gyro_range_rad_s = np.deg2rad(cfg["twin.sensors.gyro_range_deg_s"])
        out.accel_range_m_s2 = cfg["twin.sensors.accel_range_g"] * G0_M_S2
        out.adc_bits = int(cfg["twin.sensors.adc_bits"])
    return out


def draw_sensor_errors(cfg: Config, rng: np.random.Generator) -> SensorErrors:
    """Draw one pod's errors from ``twin.sensors`` (the order of the draws is fixed, so a seed reproduces them)."""
    s = "twin.sensors."

    def misalign(max_deg):
        return exp_map(rng.uniform(-1.0, 1.0, 3) * np.deg2rad(max_deg))

    gyro_scale = rng.uniform(-1.0, 1.0, 3) * cfg[s + "gyro_scale_max"]
    gyro_mis = misalign(cfg[s + "gyro_misalign_max_deg"])
    bias0 = rng.normal(0.0, 1.0, 3) * np.deg2rad(cfg[s + "gyro_bias_sigma_deg_per_min"]) / 60.0
    accel_scale = rng.uniform(-1.0, 1.0, 3) * cfg[s + "accel_scale_max"]
    accel_mis = misalign(cfg[s + "accel_misalign_max_deg"])
    accel_bias = rng.uniform(-1.0, 1.0, 3) * cfg[s + "accel_bias_max_m_s2"]
    return SensorErrors(
        gyro_scale=gyro_scale, gyro_misalign=gyro_mis, gyro_bias0_rad_s=bias0,
        gyro_bias_walk_rad_s_sqrt_s=float(np.deg2rad(cfg[s + "gyro_bias_walk_deg_s_sqrt_s"])),
        gyro_noise_rad_s_sqrt_hz=float(np.deg2rad(cfg[s + "gyro_noise_deg_s_sqrt_hz"])),
        accel_scale=accel_scale, accel_misalign=accel_mis, accel_bias_m_s2=accel_bias,
        accel_noise_m_s2_sqrt_hz=float(cfg[s + "accel_noise_ug_sqrt_hz"]) * 1e-6 * G0_M_S2,
        gyro_range_rad_s=float(np.deg2rad(cfg[s + "gyro_range_deg_s"])),
        accel_range_m_s2=float(cfg[s + "accel_range_g"]) * G0_M_S2,
        adc_bits=int(cfg[s + "adc_bits"]),
    )  # fmt: skip


def ideal_gyro_rad_s(q, dt_s: float) -> np.ndarray:
    """Mean body-frame rates over each interval of a sampled orientation track ``q`` ``(N + 1, 4)`` -> ``(N, 3)``:
    Log(q_k⁻¹ q_{k+1}) / Δt."""
    q = as_quat(q)
    return log_map(quat_mul(quat_conj(q[:-1]), q[1:])) / dt_s


def _adc(x: np.ndarray, full_scale: float | None, bits: int | None) -> np.ndarray:
    if full_scale is None or bits is None:
        return x
    step = full_scale / 2.0 ** (bits - 1)
    return np.clip(np.round(x / step), -(2 ** (bits - 1)), 2 ** (bits - 1) - 1) * step


def measure(err: SensorErrors, omega_rad_s, f_m_s2, dt_s: float, rng: np.random.Generator | None = None,
            bias_start_rad_s=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """The chip's samples ``(gyr, acc, b_g)`` for true rates ``omega_rad_s`` and specific forces ``f_m_s2`` ``(N, 3)``
    taken every ``dt_s``. ``rng`` draws the noise and the bias walk (``None``: no noise and no walk);
    ``bias_start_rad_s`` continues a walk from an earlier epoch (default b_g,0)."""
    w = np.asarray(omega_rad_s, dtype=np.float64)
    f = np.asarray(f_m_s2, dtype=np.float64)
    n = w.shape[0]
    b0 = err.gyro_bias0_rad_s if bias_start_rad_s is None else np.asarray(bias_start_rad_s, float)
    bias = np.broadcast_to(b0, (n, 3)).copy()
    gyr = (1.0 + err.gyro_scale) * quat_rotate(np.broadcast_to(err.gyro_misalign, (n, 4)), w)
    acc = (1.0 + err.accel_scale) * quat_rotate(np.broadcast_to(err.accel_misalign, (n, 4)), f)
    if rng is not None:
        if err.gyro_bias_walk_rad_s_sqrt_s > 0 and n > 1:
            steps = rng.normal(0.0, 1.0, (n - 1, 3)) * err.gyro_bias_walk_rad_s_sqrt_s * np.sqrt(dt_s)
            bias[1:] += np.cumsum(steps, axis=0)
        gyr = gyr + rng.normal(0.0, 1.0, (n, 3)) * err.gyro_noise_rad_s_sqrt_hz / np.sqrt(dt_s)
        acc = acc + rng.normal(0.0, 1.0, (n, 3)) * err.accel_noise_m_s2_sqrt_hz / np.sqrt(dt_s)
    gyr = _adc(gyr + bias, err.gyro_range_rad_s, err.adc_bits)
    acc = _adc(acc + err.accel_bias_m_s2, err.accel_range_m_s2, err.adc_bits)
    return gyr, acc, bias
