"""One simulated session, end to end (brief §8, steps 1-12).

Outputs (a :class:`SessionData`):

* ``frames``: one row per received frame, the schema a real logger writes (see docs/PIPELINE.md);
* ``labels_ticks``: one row per pod and knowable send tick: true sample time, received, fresh, true psi, the true
  pod orientation at the sample time;
* ``labels_joints``: one row per joint and 32 Hz step: the true relative heading delta and its offset part;
* ``session``: phases as host-time ranges (the logger's sidecar ``session.json``);
* ``meta``: every pod's drawn parameters, boot epochs, stall and loss counts, the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..body import JOINTS, N_PODS
from ..codec import quantize_component
from ..config import Config
from ..conventions import exp_map, heading_rad, quat_canonical, quat_mul, wrap_rad, yaw_quat
from ..recording import knowable_ticks
from .heading import Epoch, PodHeading, build_pod_heading
from .motion import MotionSource
from .radio import arrival_times, gilbert_elliott_lost
from .rng import stream
from .session import PHASES, RECORDED, PodParams, SessionPlan, plan_session
from .timing import simulate_timing
from .truth import pod_orientation

__all__ = ["SessionData", "simulate_session", "joint_grid_s", "pod_epochs", "fusion_error"]

GRID_MARGIN_S = 0.25  # start evaluating once every pod's first frame has had time to arrive


@dataclass
class SessionData:
    frames: pd.DataFrame
    labels_ticks: pd.DataFrame
    labels_joints: pd.DataFrame
    session: dict
    meta: dict
    plan: SessionPlan


def joint_grid_s(plan: SessionPlan, first_send_s: float, grid_hz: float) -> np.ndarray:
    """True times of the 32 Hz evaluation steps: from shortly after every pod has sent, through the end of
    motion, in recorded phases only. Steps sit on multiples of 1/grid_hz."""
    t0 = np.ceil((first_send_s + GRID_MARGIN_S) * grid_hz) / grid_hz
    t1 = plan.phases["motion"][1]
    t = np.arange(int(np.floor((t1 - t0) * grid_hz)) + 1) / grid_hz + t0
    return t[plan.recorded_mask(t)]


def pod_epochs(cfg: Config, plan: SessionPlan, motion: MotionSource, p: PodParams,
               rng: np.random.Generator) -> list:  # fmt: skip
    """Boot epochs of one pod: the first boot, then one per reboot (silent for reboot_silence_s, new north)."""
    t_end = plan.phases["motion"][1]
    silence = cfg["error_model.reboot_silence_s"]
    grid_start_s = cfg["timing.grid_start_ms"] / 1000.0
    mode = cfg["error_model.offset_mode"]
    starts = [(p.boot_s, p.chip_start_s, p.offset_rad)]
    for t_r in p.reboots_s:
        boot = t_r + max(0.0, silence - grid_start_s)  # first send about `silence` after the reboot
        lo, hi = cfg["timing.chip_start_s"]
        chip = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
        if mode == "arbitrary":
            a = float(rng.uniform(-np.pi, np.pi))
        elif mode == "aligned":  # the new north is the pod's own heading at the chip's start-up
            a = float(heading_rad(pod_orientation(plan, motion, p.pod_id, [boot + chip])[0]))
        else:
            a = 0.0
        starts.append((boot, chip, a))
    ends = list(p.reboots_s) + [t_end]
    return [Epoch(b, e, b + c, a) for (b, c, a), e in zip(starts, ends, strict=True)]


def fusion_error(t_s: np.ndarray, sigma_rad: float, tau_s: float, rng: np.random.Generator) -> np.ndarray:
    """eta (N, 3), a first-order Gauss-Markov process per body axis with stationary sigma, at sorted times."""
    n = t_s.size
    if sigma_rad <= 0 or n == 0:
        return np.zeros((n, 3))
    phi = np.exp(-np.diff(t_s) / tau_s)[:, None]
    eps = rng.normal(0.0, sigma_rad, (n, 3))
    out = np.empty((n, 3))
    out[0] = eps[0]
    for k in range(1, n):
        out[k] = phi[k - 1] * out[k - 1] + np.sqrt(1.0 - phi[k - 1] ** 2) * eps[k]
    return out


def simulate_session(cfg: Config, seed: int, motion: MotionSource, plan_hook=None) -> SessionData:
    """Simulate one session. ``plan_hook(plan)`` may modify the drawn plan first (used by the invariance checks,
    e.g. to turn every pod's north by the same angle); every random stream is unaffected by it."""
    plan = plan_session(cfg, seed, motion)
    if plan_hook is not None:
        plan_hook(plan)
    rngs = {k: stream(seed, k) for k in ("timing", "radio", "loss", "walk", "fusion", "epochs")}
    arw = np.deg2rad(cfg["error_model.arw_deg_per_sqrt_s"])
    tilt_sigma = np.deg2rad(cfg["error_model.tilt_sigma_deg"])

    frames, ticks, headings, epoch_meta = [], [], [], []
    first_send, n_stalls, n_lost_radio = [], 0, 0
    for i in range(N_PODS):
        p = plan.pods[i]
        epochs = pod_epochs(cfg, plan, motion, p, rngs["epochs"])
        hd: PodHeading = build_pod_heading(plan, motion, i, epochs, p.drift_rad_per_s, p.scale_factor, arw,
                                           cfg["error_model.idle_turns_sigma_per_sqrt_min"], rngs["walk"])  # fmt: skip
        headings.append(hd)
        epoch_meta.append([vars(e) for e in epochs])

        # one tick sequence across epochs, in send order
        parts = []
        for e, ep in enumerate(epochs):
            tm = simulate_timing(cfg, p, ep.end_s, rngs["timing"], boot_s=ep.boot_s,
                                 chip_start_s=ep.first_sample_s - ep.boot_s)  # fmt: skip
            n_stalls += tm.n_stalls
            j = np.arange(tm.t_send_s.size)
            parts.append(pd.DataFrame({
                "epoch": np.full(j.size, e, np.int32), "tick": j.astype(np.int64), "t_send_s": tm.t_send_s,
                "sample_index": tm.sample_idx.astype(np.int64), "t_sample_s": tm.t_sample_s[tm.sample_idx],
                "t_decode_s": tm.t_decode_s[tm.sample_idx], "ms_lo": tm.ms_lo, "fresh": tm.fresh,
            }))  # fmt: skip
        seq = pd.concat(parts, ignore_index=True)
        first_send.append(seq["t_send_s"].iloc[0])
        t_arrive = arrival_times(cfg, seq["t_send_s"].to_numpy(), rngs["radio"])
        lost = gilbert_elliott_lost(len(seq), cfg["radio.loss_mean"], cfg["radio.burst_mean_frames"], rngs["loss"])
        interval = plan.interval_id(t_arrive)
        received = ~lost & (interval >= 0)
        n_lost_radio += int(np.sum(lost & (interval >= 0)))
        epoch = seq["epoch"].to_numpy()
        keep_tick = knowable_ticks(received, epoch, interval)

        lab = seq[keep_tick].reset_index(drop=True)
        lab_received = received[keep_tick]
        t_sample = lab["t_sample_s"].to_numpy()
        psi = hd.psi(t_sample)
        p_true = pod_orientation(plan, motion, i, t_sample)
        # eta follows the chip's own samples: one value per distinct sample, in time order
        gid = lab["epoch"].to_numpy().astype(np.int64) * 10**9 + lab["sample_index"].to_numpy()
        uniq, inv = np.unique(gid, return_inverse=True)
        t_uniq = np.zeros(uniq.size)
        t_uniq[inv] = t_sample
        eta = fusion_error(t_uniq, tilt_sigma, cfg["error_model.tilt_tau_s"], rngs["fusion"])[inv]
        q = quat_canonical(quat_mul(quat_mul(yaw_quat(-psi), p_true), exp_map(eta)))  # q = Y(-psi)·P·Exp(eta)
        qi = quantize_component(q)
        batt = np.clip(np.floor(p.battery_start_pct - cfg["battery.drain_pct_per_h"]
                                * (lab["t_send_s"].to_numpy() - p.boot_s) / 3600.0), 0, 100)  # fmt: skip

        # what the host can tell: the sample differs from the last *received* one
        rec_idx = np.flatnonzero(lab_received)
        new_to_host = np.zeros(len(lab), bool)
        new_to_host[rec_idx] = np.r_[True, gid[rec_idx][1:] != gid[rec_idx][:-1]]

        ticks.append(pd.DataFrame({
            "pod_id": np.full(len(lab), i, np.uint8),
            "epoch": lab["epoch"].to_numpy(),
            "tick": lab["tick"].to_numpy(),
            "count": (lab["tick"].to_numpy() % 65536).astype(np.uint16),
            "phase": plan.phase_of(lab["t_send_s"].to_numpy()).astype(str),
            "t_send_s": lab["t_send_s"].to_numpy(),
            "t_arrive_s": t_arrive[keep_tick],
            "sample_index": lab["sample_index"].to_numpy(),
            "t_sample_s": t_sample,
            "t_decode_s": lab["t_decode_s"].to_numpy(),
            "ms_lo": lab["ms_lo"].to_numpy(),
            "received": lab_received,
            "fresh": lab["fresh"].to_numpy(),
            "new_to_host": new_to_host,
            "psi_rad": psi,
            "pod_qx": p_true[:, 0], "pod_qy": p_true[:, 1], "pod_qz": p_true[:, 2], "pod_qw": p_true[:, 3],
        }))  # fmt: skip
        r = lab_received
        frames.append(pd.DataFrame({
            "t_host_s": t_arrive[keep_tick][r] + plan.host_epoch_s,
            "pod_id": np.full(r.sum(), i, np.uint8),
            "batt": batt[r].astype(np.uint8),
            "qx": qi[r, 0], "qy": qi[r, 1], "qz": qi[r, 2], "qw": qi[r, 3],
            "count": (lab["tick"].to_numpy()[r] % 65536).astype(np.uint16),
            "ms_lo": lab["ms_lo"].to_numpy()[r],
        }))  # fmt: skip

    frames_df = pd.concat(frames, ignore_index=True)
    frames_df = frames_df.sort_values(["t_host_s", "pod_id"], kind="mergesort", ignore_index=True)
    ticks_df = pd.concat(ticks, ignore_index=True)

    grid = joint_grid_s(plan, max(first_send), cfg["eval.grid_hz"])
    psi_grid = np.stack([h.psi(grid) for h in headings], axis=1)  # (K, 15), NaN while a pod is off
    off_grid = np.stack([h.offset(grid) for h in headings], axis=1)
    rows = []
    for k, jt in enumerate(JOINTS):
        rows.append(pd.DataFrame({
            "step": np.arange(grid.size, dtype=np.int64),
            "t_s": grid,
            "phase": plan.phase_of(grid).astype(str),
            "joint": np.full(grid.size, k, np.uint8),
            "delta_rad": wrap_rad(psi_grid[:, jt.child] - psi_grid[:, jt.parent]),
            "delta_offset_rad": wrap_rad(off_grid[:, jt.child] - off_grid[:, jt.parent]),
        }))  # fmt: skip
    joints_df = pd.concat(rows, ignore_index=True)

    session = {
        "schema_version": 1,
        "pods": list(range(N_PODS)),
        "phases": [
            {"name": n, "t_host_start_s": plan.phases[n][0] + plan.host_epoch_s,
             "t_host_end_s": plan.phases[n][1] + plan.host_epoch_s, "recorded": RECORDED[n]}
            for n in PHASES
        ],
    }  # fmt: skip
    meta = {
        "plan": plan.to_dict(),
        "epochs": epoch_meta,
        "first_send_s": float(max(first_send)),
        "counts": {"stalls": n_stalls, "radio_lost_in_recording": n_lost_radio, "frames": len(frames_df)},
    }
    return SessionData(frames_df, ticks_df, joints_df, session, meta, plan)
