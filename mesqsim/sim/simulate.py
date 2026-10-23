"""One simulated session, end to end (brief §8, steps 1-12).

Outputs (a :class:`SessionData`):

* ``frames``: one row per received frame, the schema a real logger writes (see docs/PIPELINE.md);
* ``labels_ticks``: one row per pod and recorded send tick: true sample time, received, fresh, true psi, the true
  pod orientation at the sample time;
* ``labels_joints``: one row per joint and 32 Hz step: the true relative heading delta and the offset part of it;
* ``session``: phases as host-time ranges (the logger's sidecar ``session.json``);
* ``meta``: config, seed, every pod's drawn parameters, the timeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..body import JOINTS, N_PODS
from ..codec import quantize_component
from ..config import Config
from ..conventions import quat_canonical, quat_mul, wrap_rad, yaw_quat
from ..recording import knowable_ticks
from .heading import psi_rad
from .motion import MotionSource
from .radio import arrival_times
from .rng import stream
from .session import PHASES, RECORDED, SessionPlan, plan_session
from .timing import pod_ms, simulate_timing
from .truth import pod_orientation

__all__ = ["SessionData", "simulate_session", "joint_grid_s"]


@dataclass
class SessionData:
    frames: pd.DataFrame
    labels_ticks: pd.DataFrame
    labels_joints: pd.DataFrame
    session: dict
    meta: dict
    plan: SessionPlan


GRID_MARGIN_S = 0.25  # start evaluating once every pod's first frame has had time to arrive


def joint_grid_s(plan: SessionPlan, first_send_s: float, grid_hz: float) -> np.ndarray:
    """True times of the 32 Hz evaluation steps: from shortly after every pod has sent, through the end of
    motion, in recorded phases only. Steps sit on multiples of 1/grid_hz."""
    t0 = np.ceil((first_send_s + GRID_MARGIN_S) * grid_hz) / grid_hz
    t1 = plan.phases["motion"][1]
    t = np.arange(int(np.floor((t1 - t0) * grid_hz)) + 1) / grid_hz + t0
    return t[plan.recorded_mask(t)]


def simulate_session(cfg: Config, seed: int, motion: MotionSource) -> SessionData:
    plan = plan_session(cfg, seed, motion)
    t_end = plan.phases["motion"][1]
    r_timing, r_radio = stream(seed, "timing"), stream(seed, "radio")

    frames, ticks = [], []
    first_send = []
    for i in range(N_PODS):
        p = plan.pods[i]
        tm = simulate_timing(cfg, p, t_end, r_timing)
        t_arrive = arrival_times(cfg, tm.t_send_s, r_radio)
        interval = plan.interval_id(t_arrive)
        radio_ok = np.ones(tm.t_send_s.size, bool)  # radio loss arrives in M4
        epoch = np.zeros(tm.t_send_s.size, np.int32)  # reboots arrive in M4
        received = radio_ok & (interval >= 0)  # in the frames table
        first_send.append(tm.t_send_s[0])

        j = np.flatnonzero(knowable_ticks(received, epoch, interval))  # the ticks the labels describe
        idx = tm.sample_idx[j]
        t_sample = tm.t_sample_s[idx]
        psi = psi_rad(plan, i, t_sample)
        p_true = pod_orientation(plan, motion, i, t_sample)
        q = quat_canonical(quat_mul(yaw_quat(-psi), p_true))  # q_i = Y(-psi)·s·m (·Exp(eta) from M4)
        qi = quantize_component(q)
        batt = np.clip(np.floor(p.battery_start_pct - cfg["battery.drain_pct_per_h"]
                                * (tm.t_send_s[j] - p.boot_s) / 3600.0), 0, 100)  # fmt: skip

        # what the host can tell: the sample differs from the last *received* one
        new_to_host = np.zeros(j.size, bool)
        last = -1
        for k in range(j.size):
            if received[j[k]]:
                new_to_host[k] = idx[k] != last
                last = idx[k]

        ticks.append(pd.DataFrame({
            "pod_id": np.full(j.size, i, np.uint8),
            "epoch": epoch[j],
            "tick": j.astype(np.int64),
            "count": (j % 65536).astype(np.uint16),
            "phase": plan.phase_of(tm.t_send_s[j]).astype(str),
            "t_send_s": tm.t_send_s[j],
            "t_arrive_s": t_arrive[j],
            "sample_index": idx.astype(np.int64),
            "t_sample_s": t_sample,
            "t_decode_s": tm.t_decode_s[idx],
            "ms_lo": tm.ms_lo[j],
            "received": received[j],
            "fresh": tm.fresh[j],
            "new_to_host": new_to_host,
            "psi_rad": psi,
            "pod_qx": p_true[:, 0], "pod_qy": p_true[:, 1], "pod_qz": p_true[:, 2], "pod_qw": p_true[:, 3],
        }))  # fmt: skip
        keep = received[j]
        frames.append(pd.DataFrame({
            "t_host_s": t_arrive[j][keep] + plan.host_epoch_s,
            "pod_id": np.full(keep.sum(), i, np.uint8),
            "batt": batt[keep].astype(np.uint8),
            "qx": qi[keep, 0], "qy": qi[keep, 1], "qz": qi[keep, 2], "qw": qi[keep, 3],
            "count": (j[keep] % 65536).astype(np.uint16),
            "ms_lo": tm.ms_lo[j][keep],
        }))  # fmt: skip
        assert np.all(pod_ms(p, tm.t_decode_s) >= 0)

    frames_df = pd.concat(frames, ignore_index=True)
    frames_df = frames_df.sort_values(["t_host_s", "pod_id"], kind="mergesort", ignore_index=True)
    ticks_df = pd.concat(ticks, ignore_index=True)

    grid = joint_grid_s(plan, max(first_send), cfg["eval.grid_hz"])
    psi_grid = np.stack([psi_rad(plan, i, grid) for i in range(N_PODS)], axis=1)  # (K, 15)
    offsets = np.array([p.offset_rad for p in plan.pods])
    rows = []
    for k, jt in enumerate(JOINTS):
        rows.append(pd.DataFrame({
            "step": np.arange(grid.size, dtype=np.int64),
            "t_s": grid,
            "phase": plan.phase_of(grid).astype(str),
            "joint": np.full(grid.size, k, np.uint8),
            "delta_rad": wrap_rad(psi_grid[:, jt.child] - psi_grid[:, jt.parent]),
            "delta_offset_rad": np.full(grid.size, wrap_rad(offsets[jt.child] - offsets[jt.parent])),
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
    meta = {"plan": plan.to_dict(), "first_send_s": float(max(first_send))}
    return SessionData(frames_df, ticks_df, joints_df, session, meta, plan)
