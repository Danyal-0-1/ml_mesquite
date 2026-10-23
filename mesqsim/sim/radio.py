"""From send to host: arrival times (brief §8 step 11). Radio loss is added in M4.

Arrival = send time + latency, latency = latency_min + Exp(mean latency_jitter). One pod's frames arrive in the
order they were sent (the radio link is first-in, first-out), so arrival times are made non-decreasing per pod.
"""

from __future__ import annotations

import numpy as np

from ..config import Config

__all__ = ["arrival_times"]


def arrival_times(cfg: Config, t_send_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter_ms = cfg["radio.latency_jitter_ms"]
    extra = rng.exponential(jitter_ms, t_send_s.shape) if jitter_ms > 0 else np.zeros(t_send_s.shape)
    t_arrive = t_send_s + (cfg["radio.latency_min_ms"] + extra) / 1000.0
    return np.maximum.accumulate(t_arrive)
