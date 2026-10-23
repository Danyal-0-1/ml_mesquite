"""The session plan: timeline and every pod's drawn parameters (brief §8 steps 1, 3, 4).

True time ``t`` is in seconds from the moment the first pod could be switched on in the box (t = 0). Phases::

    box [0, box_s) -> strapping -> tpose -> idle (not recorded) -> motion

Nothing here depends on the order in which other subsystems draw random numbers (see :mod:`.rng`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..body import N_PODS, PODS
from ..config import Config
from ..conventions import exp_map, heading_rad, quat_mul, wrap_rad, yaw_quat
from ..recording import interval_id, merge_recorded
from .motion import NOMINAL_MOUNT, UPRIGHT, MotionSource, facing_heading_rad
from .rng import stream

__all__ = ["PodParams", "SessionPlan", "PHASES", "plan_session"]

PHASES = ("box", "strapping", "tpose", "idle", "motion")
RECORDED = {"box": True, "strapping": True, "tpose": True, "idle": False, "motion": True}


@dataclass
class PodParams:
    pod_id: int
    boot_s: float  # true time the pod powered on
    clock_ppm: float  # pod millisecond clock rate error
    chip_rate_hz: float  # fused sample rate of the chip, true Hz
    chip_start_s: float  # boot -> first chip sample
    grid_first_ms: float  # pod ms of the first send: grid start + id x period / divisor
    mount_q: list  # m_i (pod -> segment), (x, y, z, w)
    box_slot_rad: float  # eps_i: heading error of the pod's slot in the box
    offset_rad: float  # a_i: heading of the pod's private frame at boot
    strap_window_s: tuple  # (w0, w1): the pod's own carry-and-strap movement
    strap_spins: int  # whole turns about the vertical while carried
    posture_q: list  # Exp(e_i): T-pose posture error of the pod's segment
    battery_start_pct: float
    drift_rad_per_s: float = 0.0  # b_i: residual gyro-bias drift
    scale_factor: float = 0.0  # k_i: gyroscope scale-factor error
    reboots_s: list = field(default_factory=list)  # true times at which the pod reboots


@dataclass
class SessionPlan:
    phases: dict  # name -> (t0, t1) true seconds
    host_epoch_s: float  # host clock = true time + host_epoch_s
    box_heading_rad: float  # heading of the box (pods flat, z up)
    facing_tpose_rad: float  # facing of the wearer in the T-pose (unknowable room heading)
    facing_motion_rad: float  # phi_m: rotation applied to the motion source
    motion_name: str
    pods: list  # PodParams by pod id

    def phase_of(self, t_s) -> np.ndarray:
        t = np.asarray(t_s, float)
        out = np.full(t.shape, "", dtype=object)
        for name in PHASES:
            t0, t1 = self.phases[name]
            out[(t >= t0) & (t < t1)] = name
        t0, t1 = self.phases["motion"]
        out[t == t1] = "motion"
        return out

    def recorded_intervals(self) -> list[tuple[float, float]]:
        """Recorded stretches of true time: contiguous recorded phases merged (closed intervals)."""
        return merge_recorded([(n, *self.phases[n], RECORDED[n]) for n in PHASES])

    def interval_id(self, t_s) -> np.ndarray:
        """Index of the recorded interval containing each time, -1 outside every one."""
        return interval_id(t_s, self.recorded_intervals())

    def recorded_mask(self, t_s) -> np.ndarray:
        return self.interval_id(t_s) >= 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pods"] = [asdict(p) for p in self.pods]
        return d


def _uniform(rng, lohi):
    lo, hi = lohi
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def _reboot_times(cfg: Config, rng: np.random.Generator, boot_s: float, t_end_s: float) -> list:
    """Poisson reboots at rate error_model.reboot_rate_per_h, at least a silence plus 2 s apart."""
    rate = cfg["error_model.reboot_rate_per_h"] / 3600.0
    silence = cfg["error_model.reboot_silence_s"]
    t0, t1 = boot_s + 5.0, t_end_s - silence - 2.0
    if rate <= 0 or t1 <= t0:
        return []
    times = np.sort(rng.uniform(t0, t1, rng.poisson(rate * (t1 - t0))))
    kept: list[float] = []
    for t in times:
        if not kept or t - kept[-1] >= silence + 2.0:
            kept.append(float(t))
    return kept


def plan_session(cfg: Config, seed: int, motion: MotionSource) -> SessionPlan:
    """Draw the timeline and every pod's parameters for one session."""
    r_time, r_pod, r_head = stream(seed, "timeline"), stream(seed, "pods"), stream(seed, "heading")
    r_drift, r_reboot = stream(seed, "drift"), stream(seed, "reboots")

    box_s, strap_s, tpose_s = cfg["session.box_s"], cfg["session.strapping_s"], cfg["session.tpose_s"]
    idle_s = float(cfg["session.idle_s"])
    t = 0.0
    phases = {}
    for name, dur in (("box", box_s), ("strapping", strap_s), ("tpose", tpose_s), ("idle", idle_s),
                      ("motion", motion.duration_s)):  # fmt: skip
        phases[name] = (t, t + dur)
        t += dur

    host_epoch_s = _uniform(r_time, cfg["timing.host_epoch_s"])
    box_heading = float(r_time.uniform(-np.pi, np.pi))
    facing_t = float(r_time.uniform(-np.pi, np.pi))
    fe = np.deg2rad(cfg["session.facing_error_deg"])
    facing_err = float(r_time.uniform(-fe, fe))
    facing_m = float(wrap_rad(facing_t + facing_err - facing_heading_rad(motion.seg_q[0, 2])))

    mode = cfg["error_model.offset_mode"]
    if mode not in ("aligned", "arbitrary", "none"):
        raise ValueError(f"error_model.offset_mode must be aligned, arbitrary or none, got {mode!r}")
    period_ms = cfg["timing.send_period_us"] / 1000.0
    mount_max = np.deg2rad(cfg["session.mounting_max_deg"])
    posture_sigma = np.deg2rad(cfg["session.tpose_posture_deg"]) / np.sqrt(3.0)  # RMS angle -> per axis
    slot_sigma = np.deg2rad(cfg["error_model.box_slot_deg"])
    s0, s1 = phases["strapping"]
    move_lo, move_hi = cfg["session.strap_move_s"]
    pods = []
    for pod in PODS:
        i = pod.id
        boot = _uniform(r_pod, (0.0, cfg["session.boot_spread_s"]))
        ppm = _uniform(r_pod, (-cfg["timing.pod_clock_ppm"], cfg["timing.pod_clock_ppm"]))
        rate = _uniform(r_pod, cfg["timing.chip_rate_hz"])
        chip_start = _uniform(r_pod, cfg["timing.chip_start_s"])
        mount = quat_mul(NOMINAL_MOUNT[i], exp_map(r_pod.uniform(-mount_max, mount_max, 3)))
        move = _uniform(r_pod, (move_lo, min(move_hi, strap_s)))
        w0 = s0 + _uniform(r_pod, (0.0, strap_s - move))
        spins = int(r_pod.integers(-cfg["session.strap_spins_max"], cfg["session.strap_spins_max"] + 1))
        posture = exp_map(r_pod.normal(0.0, posture_sigma, 3))
        battery = _uniform(r_pod, cfg["battery.start_pct"])

        slot = float(r_head.normal(0.0, slot_sigma)) if slot_sigma > 0 else 0.0
        if mode == "arbitrary":
            offset = float(r_head.uniform(-np.pi, np.pi))
        elif mode == "aligned":  # north = the chip's own heading at start-up, in its box slot
            offset = float(heading_rad(yaw_quat(box_heading + slot)))
        else:
            offset = 0.0
        drift = float(r_drift.normal(0.0, np.deg2rad(cfg["error_model.drift_sigma_deg_per_min"]) / 60.0))
        kmax = cfg["error_model.scale_factor_max"]
        scale = float(r_drift.uniform(-kmax, kmax)) if kmax > 0 else 0.0
        reboots = _reboot_times(cfg, r_reboot, boot, phases["motion"][1])
        pods.append(PodParams(
            pod_id=i, boot_s=boot, clock_ppm=ppm, chip_rate_hz=rate, chip_start_s=chip_start,
            grid_first_ms=cfg["timing.grid_start_ms"] + i * period_ms / cfg["timing.send_phase_divisor"],
            mount_q=mount.tolist(), box_slot_rad=slot, offset_rad=offset, strap_window_s=(w0, w0 + move),
            strap_spins=spins, posture_q=posture.tolist(), battery_start_pct=battery,
            drift_rad_per_s=drift, scale_factor=scale, reboots_s=reboots,
        ))  # fmt: skip
    assert len(pods) == N_PODS
    return SessionPlan(phases, host_epoch_s, box_heading, facing_t, facing_m, motion.name, pods)


def tpose_segment_q(plan: SessionPlan, pod: int) -> np.ndarray:
    """s_i during the T-pose: the SMPL rest pose turned to the T-pose facing, times the posture error."""
    return quat_mul(quat_mul(yaw_quat(plan.facing_tpose_rad), UPRIGHT), np.asarray(plan.pods[pod].posture_q))
