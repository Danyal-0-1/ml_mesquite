"""From send to host: radio loss and arrival times (brief §8 steps 10 and 11).

* **Loss** follows a two-state Gilbert-Elliott chain per pod, one step per send: in the good state frames
  arrive, in the bad state they are lost. With mean loss L and mean burst length B (frames):
  P(bad -> good) = 1/B and P(good -> bad) = L/(1-L) · 1/B, so the stationary loss is L and bad runs last B on
  average. B = 1 gives isolated losses.
* **Arrival** = send time + latency, latency = latency_min + Exp(mean latency_jitter). One pod's frames arrive in
  the order they were sent (the link is first-in, first-out), so arrival times are made non-decreasing per pod.
"""

from __future__ import annotations

import numpy as np

from ..config import Config

__all__ = ["gilbert_elliott_lost", "arrival_times"]


def gilbert_elliott_lost(n: int, loss_mean: float, burst_mean: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask of lost sends (length n) from the stationary two-state chain."""
    lost = np.zeros(n, bool)
    if loss_mean <= 0 or n == 0:
        return lost
    if not 0 < loss_mean < 1 or burst_mean < 1:
        raise ValueError("need 0 < loss_mean < 1 and burst_mean >= 1")
    p_bg = 1.0 / burst_mean
    p_gb = loss_mean / (1.0 - loss_mean) * p_bg
    bad = rng.random() < loss_mean  # start from the stationary distribution
    pos = 0
    while pos < n:  # alternate geometric sojourns (memoryless, so the first one needs no special case)
        run = int(rng.geometric(p_bg if bad else p_gb))
        if bad:
            lost[pos : pos + run] = True
        pos += run
        bad = not bad
    return lost


def arrival_times(cfg: Config, t_send_s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter_ms = cfg["radio.latency_jitter_ms"]
    extra = rng.exponential(jitter_ms, t_send_s.shape) if jitter_ms > 0 else np.zeros(t_send_s.shape)
    t_arrive = t_send_s + (cfg["radio.latency_min_ms"] + extra) / 1000.0
    return np.maximum.accumulate(t_arrive)
