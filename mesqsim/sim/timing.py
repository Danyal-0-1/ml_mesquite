"""Chip samples and the 32 Hz send grid of one pod during one boot epoch (brief §3.3; §8 steps 5 and 8).

* The chip produces fused samples at its own rate f_i: t_n = t_first + n / f_i (true time).
* Each sample is decoded 0-11 ms later; ``ms_lo`` is the pod-clock millisecond of that decode, mod 65536.
* Optional sample-task **stalls** (off by default): every ``stall_period_s`` of pod time (jittered), the task
  stops for ``stall_ms``. Nothing is decoded during a stall; at its end the newest waiting sample is decoded and
  the older ones are dropped. The task then needs ``decode_busy_ms`` before it can decode again, so two
  different samples never share a decode millisecond: a repeated ``ms_lo`` really means a held frame.
* The pod sends on a fixed grid of its own clock: pod ms = grid_first_ms + j · 31.25.
* At each send the pod takes the newest decoded sample. Older unsent samples are discarded. If nothing new was
  decoded since the last send, it resends the last sample with the same ``ms_lo``: a held frame.
* ``count`` = j mod 65536, j counted from the epoch's boot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config

__all__ = ["PodTiming", "pod_ms", "true_time", "stall_windows", "simulate_timing"]


@dataclass
class PodTiming:
    t_sample_s: np.ndarray  # (N,) true sample times
    t_decode_s: np.ndarray  # (N,) true decode times (non-decreasing; dropped samples share the stall end)
    t_send_s: np.ndarray  # (J,) true send times
    sample_idx: np.ndarray  # (J,) index of the sample sent at each tick
    ms_lo: np.ndarray  # (J,) uint16 decode millisecond of that sample
    fresh: np.ndarray  # (J,) the sample differs from the previous tick's
    n_stalls: int = 0


def pod_ms(p, t_s, boot_s: float | None = None) -> np.ndarray:
    """The pod's millisecond clock at true time ``t_s`` (float; the firmware reads its floor).

    ``p`` needs ``boot_s`` and ``clock_ppm``; ``boot_s`` overrides the boot (for later epochs).
    """
    boot = p.boot_s if boot_s is None else boot_s
    return (np.asarray(t_s, float) - boot) * 1000.0 * (1.0 + p.clock_ppm * 1e-6)


def true_time(p, ms, boot_s: float | None = None) -> np.ndarray:
    """Inverse of :func:`pod_ms`."""
    boot = p.boot_s if boot_s is None else boot_s
    return boot + np.asarray(ms, float) / 1000.0 / (1.0 + p.clock_ppm * 1e-6)


def stall_windows(cfg: Config, p, boot_s: float, t_end_s: float, rng: np.random.Generator) -> np.ndarray:
    """(M, 2) true-time [start, end) of each sample-task stall in one epoch; empty when stalls are off."""
    stall_ms = cfg["timing.stall_ms"]
    if stall_ms <= 0:
        return np.zeros((0, 2))
    period = cfg["timing.stall_period_s"] * 1000.0
    jitter = cfg["timing.stall_jitter_s"] * 1000.0
    total = float(pod_ms(p, t_end_s, boot_s))
    k = np.arange(1, int(total // period) + 1)
    start_ms = k * period + rng.uniform(-jitter, jitter, k.size)
    start_ms = start_ms[(start_ms > 0) & (start_ms + stall_ms < total)]
    return np.column_stack([true_time(p, start_ms, boot_s), true_time(p, start_ms + stall_ms, boot_s)])


def simulate_timing(cfg: Config, p, t_end_s: float, rng: np.random.Generator, boot_s: float | None = None,
                    chip_start_s: float | None = None) -> PodTiming:  # fmt: skip
    """Samples, decodes and sends of one epoch from ``boot_s`` (default: the first boot) to ``t_end_s``."""
    boot = p.boot_s if boot_s is None else boot_s
    t_first = boot + (p.chip_start_s if chip_start_s is None else chip_start_s)
    n = int(np.floor((t_end_s - t_first) * p.chip_rate_hz)) + 1
    t_sample = t_first + np.arange(n) / p.chip_rate_hz
    d0, d1 = cfg["timing.decode_delay_ms"]
    t_decode = t_sample + (rng.uniform(d0, d1, n) if d1 > d0 else np.full(n, d0)) / 1000.0

    stalls = stall_windows(cfg, p, boot, t_end_s, rng)
    busy = cfg["timing.decode_busy_ms"] / 1000.0
    for a, b in stalls:
        inside = (t_decode >= a) & (t_decode < b)
        t_decode[inside] = b  # the newest waiting sample is decoded at the end; older ones share b and lose
        after = np.flatnonzero(t_decode > b)
        if after.size:
            t_decode[after[0]] = max(t_decode[after[0]], b + busy)

    period_ms = cfg["timing.send_period_us"] / 1000.0
    j_max = int(np.floor((pod_ms(p, t_end_s, boot) - p.grid_first_ms) / period_ms))
    t_send = true_time(p, p.grid_first_ms + np.arange(j_max + 1) * period_ms, boot)
    idx = np.searchsorted(t_decode, t_send, side="right") - 1  # newest decoded sample at each send
    if np.any(idx < 0):
        raise ValueError("a send happened before the first decoded sample; grid_start_ms is too small")
    ms_lo = (np.floor(pod_ms(p, t_decode[idx], boot)).astype(np.int64) % 65536).astype(np.uint16)
    fresh = np.r_[True, idx[1:] != idx[:-1]]
    return PodTiming(t_sample, t_decode, t_send, idx, ms_lo, fresh, len(stalls))
