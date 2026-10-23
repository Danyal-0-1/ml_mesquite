"""The heading of each pod's private frame, psi_i(t) (brief §4.2, §4.3).

    psi_i(t) = a_e + b_i·(t - boot_e) + k_i·(Theta_i(t) - Theta_i(boot_e)) + w_i(t)      in boot epoch e

* a_e: the north chosen at boot e (offset mode: arbitrary, aligned or none). A reboot starts a new epoch with a
  new north, a new drift start and a new turning reference.
* b_i: residual gyro-bias drift, rad/s (drawn per pod, kept across reboots).
* k_i: gyroscope scale-factor error; Theta_i(t): signed rotation of the pod about the world vertical,
  accumulated by integrating the vertical component of its angular velocity along the true trajectory.
* w_i(t): angle random walk, Brownian with variance N_g² t.

The trajectory is integrated on a dense 240 Hz grid. The idle interval (minutes, unrecorded) is one step of the
grid: its turning is the net twist about the vertical between the poses before and after it, plus whole turns
drawn from an assumed turning model; its random walk is one increment N(0, N_g² Δt). Between grid points Theta
and w are interpolated linearly. While a pod is switched off between a reboot and its next boot, psi is NaN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conventions import log_map, quat_conj, quat_mul, wrap_rad
from .motion import MotionSource
from .session import SessionPlan
from .truth import pod_orientation

__all__ = ["DENSE_HZ", "Epoch", "PodHeading", "twist_z_rad", "theta_increments_rad", "build_pod_heading",
           "random_walk_rad"]

DENSE_HZ = 240.0
IDLE_MARGIN_S = 1.0  # dense coverage on each side of the idle gap (frames sent just after it carry older samples)


@dataclass
class Epoch:
    boot_s: float  # the pod clock restarts here (true time)
    end_s: float  # reboot moment (sends stop) or session end
    first_sample_s: float  # the chip's first fused sample of this epoch
    offset_rad: float  # a_e


@dataclass
class PodHeading:
    t_s: np.ndarray  # dense times
    theta_rad: np.ndarray  # Theta at t_s (signed vertical rotation since the first boot)
    w_rad: np.ndarray  # angle random walk at t_s
    drift_rad_per_s: float
    scale_factor: float
    epochs: list

    def theta_at(self, t_s) -> np.ndarray:
        return np.interp(t_s, self.t_s, self.theta_rad)

    def epoch_index(self, t_s) -> np.ndarray:
        """Epoch whose boot precedes each time (the pod may be off: see :meth:`is_on`)."""
        boots = np.array([e.boot_s for e in self.epochs])
        return np.clip(np.searchsorted(boots, np.asarray(t_s, float), side="right") - 1, 0, len(self.epochs) - 1)

    def is_on(self, t_s) -> np.ndarray:
        t = np.asarray(t_s, float)
        e = self.epoch_index(t)
        ends = np.array([ep.end_s for ep in self.epochs])
        return (t < ends[e]) | ((e == len(self.epochs) - 1) & (t <= ends[e]))

    def psi(self, t_s) -> np.ndarray:
        """psi_i at true times (radians, unwrapped); NaN while the pod is switched off."""
        t = np.asarray(t_s, float)
        e = self.epoch_index(t)
        boot = np.array([ep.boot_s for ep in self.epochs])[e]
        a = np.array([ep.offset_rad for ep in self.epochs])[e]
        theta0 = self.theta_at(boot)
        out = (a + self.drift_rad_per_s * (t - boot) + self.scale_factor * (self.theta_at(t) - theta0)
               + np.interp(t, self.t_s, self.w_rad))  # fmt: skip
        return np.where(self.is_on(t) & (t >= boot), out, np.nan)

    def offset(self, t_s) -> np.ndarray:
        """a_e of the epoch at each time (NaN while off): the O1 oracle's ingredient."""
        t = np.asarray(t_s, float)
        e = self.epoch_index(t)
        a = np.array([ep.offset_rad for ep in self.epochs])[e]
        boot = np.array([ep.boot_s for ep in self.epochs])[e]
        return np.where(self.is_on(t) & (t >= boot), a, np.nan)


def twist_z_rad(q) -> np.ndarray:
    """Angle (radians, in (-pi, pi]) of the twist about world +Z in the swing-twist split of rotation q."""
    q = np.asarray(q, float)
    return wrap_rad(2.0 * np.arctan2(q[..., 2], q[..., 3]))  # twist part = (0, 0, z, w) normalised


def theta_increments_rad(p_world) -> np.ndarray:
    """Vertical rotation between consecutive orientations (N, 4) -> (N-1,): z of Log(P_{k+1} P_k^-1)."""
    return log_map(quat_mul(p_world[1:], quat_conj(p_world[:-1])))[:, 2]


def _dense_segments(plan: SessionPlan, t0: float, t_end: float) -> list[np.ndarray]:
    tpose_end, m0 = plan.phases["tpose"][1], plan.phases["motion"][0]
    dt = 1.0 / DENSE_HZ

    def grid(a, b):  # a, a + dt, ..., ending exactly at b
        n = int(np.ceil((b - a) * DENSE_HZ))
        return np.r_[a + np.arange(n) * dt, b] if b > a else np.array([a])

    def with_boundaries(ts):  # phase boundaries are grid points, so an instantaneous pose change sits between two
        inside = [b for b in (tpose_end, 0.5 * (tpose_end + m0), m0) if ts[0] < b < ts[-1]]
        return np.unique(np.r_[ts, inside])

    if m0 - tpose_end <= 2.0 * IDLE_MARGIN_S:
        return [with_boundaries(grid(t0, t_end))]
    return [with_boundaries(grid(t0, tpose_end + IDLE_MARGIN_S)), with_boundaries(grid(m0 - IDLE_MARGIN_S, t_end))]


def build_pod_heading(plan: SessionPlan, motion: MotionSource, pod: int, epochs: list, drift_rad_per_s: float,
                      scale_factor: float, arw_rad_per_sqrt_s: float, idle_turn_sigma_per_sqrt_min: float,
                      rng: np.random.Generator) -> PodHeading:  # fmt: skip
    """Integrate Theta and the random walk for one pod from its first boot to the end of the session."""
    t_end = plan.phases["motion"][1]
    segs = _dense_segments(plan, epochs[0].boot_s, t_end)
    t_all, theta_all = [], []
    theta_end = 0.0
    for k, ts in enumerate(segs):
        p = pod_orientation(plan, motion, pod, ts)
        inc = np.r_[0.0, np.cumsum(theta_increments_rad(p))]
        if k > 0:  # the idle gap: net twist about the vertical plus whole turns of the wearer
            jump = quat_mul(p[0], quat_conj(pod_orientation(plan, motion, pod, t_all[-1][-1:])[0]))
            gap_min = (ts[0] - t_all[-1][-1]) / 60.0
            turns = np.round(rng.normal(0.0, idle_turn_sigma_per_sqrt_min * np.sqrt(gap_min)))
            theta_end += float(twist_z_rad(jump)) + 2.0 * np.pi * turns
        t_all.append(ts)
        theta_all.append(theta_end + inc)
        theta_end = theta_all[-1][-1]
    t = np.concatenate(t_all)
    theta = np.concatenate(theta_all)
    return PodHeading(t, theta, random_walk_rad(t, arw_rad_per_sqrt_s, rng), drift_rad_per_s, scale_factor, epochs)


def random_walk_rad(t_s: np.ndarray, arw_rad_per_sqrt_s: float, rng: np.random.Generator) -> np.ndarray:
    """Angle random walk at sorted times, starting at 0: independent increments N(0, N_g² Δt)."""
    dt = np.diff(t_s)
    return np.r_[0.0, np.cumsum(rng.normal(0.0, 1.0, dt.size) * arw_rad_per_sqrt_s * np.sqrt(dt))]
