"""Chip samples and the 32 Hz send grid of one pod (brief §3.3; §8 steps 5 and 8).

* The chip produces fused samples at its own rate f_i: t_n = t_first + n / f_i (true time).
* Each sample is decoded 0-11 ms later; ``ms_lo`` is the pod-clock millisecond of that decode, mod 65536.
* The pod sends on a fixed grid of its own clock: pod ms = grid_first_ms + j · 31.25.
* At each send the pod takes the newest decoded sample. Older unsent samples are discarded. If nothing new was
  decoded since the last send, it resends the last sample with the same ``ms_lo``: a held frame.
* ``count`` = j mod 65536.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config
from .session import PodParams

__all__ = ["PodTiming", "pod_ms", "true_time", "simulate_timing"]


@dataclass
class PodTiming:
    t_sample_s: np.ndarray  # (N,) true sample times
    t_decode_s: np.ndarray  # (N,) true decode times (non-decreasing)
    t_send_s: np.ndarray  # (J,) true send times
    sample_idx: np.ndarray  # (J,) index of the sample sent at each tick (-1: none yet)
    ms_lo: np.ndarray  # (J,) uint16 decode millisecond of that sample
    fresh: np.ndarray  # (J,) the sample differs from the previous tick's


def pod_ms(p: PodParams, t_s) -> np.ndarray:
    """The pod's millisecond clock at true time ``t_s`` (float; the firmware reads its floor)."""
    return (np.asarray(t_s, float) - p.boot_s) * 1000.0 * (1.0 + p.clock_ppm * 1e-6)


def true_time(p: PodParams, ms) -> np.ndarray:
    """Inverse of :func:`pod_ms`."""
    return p.boot_s + np.asarray(ms, float) / 1000.0 / (1.0 + p.clock_ppm * 1e-6)


def simulate_timing(cfg: Config, p: PodParams, t_end_s: float, rng: np.random.Generator) -> PodTiming:
    t_first = p.boot_s + p.chip_start_s
    n = int(np.floor((t_end_s - t_first) * p.chip_rate_hz)) + 1
    t_sample = t_first + np.arange(n) / p.chip_rate_hz
    d0, d1 = cfg["timing.decode_delay_ms"]
    t_decode = t_sample + rng.uniform(d0, d1, n) / 1000.0 if d1 > d0 else t_sample + d0 / 1000.0

    period_ms = cfg["timing.send_period_us"] / 1000.0
    j_max = int(np.floor((pod_ms(p, t_end_s) - p.grid_first_ms) / period_ms))
    t_send = true_time(p, p.grid_first_ms + np.arange(j_max + 1) * period_ms)
    idx = np.searchsorted(t_decode, t_send, side="right") - 1  # newest decoded sample at each send
    if np.any(idx < 0):
        raise ValueError("a send happened before the first decoded sample; grid_start_ms is too small")
    ms_lo = (np.floor(pod_ms(p, t_decode[idx])).astype(np.int64) % 65536).astype(np.uint16)
    fresh = np.r_[True, idx[1:] != idx[:-1]]
    return PodTiming(t_sample, t_decode, t_send, idx, ms_lo, fresh)
