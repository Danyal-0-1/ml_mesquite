"""The phone in the twin: ARKit's reports, the app, the link to the hub, and the phone's labels (Step 3 brief §4.7).

**ARKit-like reports.** For the camera's true pose R_C(t) (camera -> world, +Z up; the belt site's F · M_ph) and
p_C(t) at the ARKit frame times (60 Hz with jitter, ``twin.phone``)::

    R̂_C(t) = Y(-ψ_A(t)) · R_C(t) · Exp(η(t))            ψ_A(t) = a + w(t) + (jumps at the end of relocalisations)
    p̂_C(t) = Σ_k Y(-ψ_A(t_k)) (1 + σ_s) (p_C(t_k) - p_C(t_k-1)) + w_p(t)       p̂_C(t_0) = 0

in the Z-up version of ARKit's world. ``a`` is not drawn: ARKit's -Z is the camera's view direction at the session
start, made level, and its origin the camera's position then, so ``a`` puts that view direction on ARKit's -Z.
w is a random walk, η Step 1's Gauss-Markov error per camera axis. The pose crosses into ARKit's +Y-up frame in
:mod:`.page_frame`, as ``ARFrame.camera.transform`` does in float32, and reaches the app as the app reads it
(Shepperd's method on the float32 columns, ``App/PhoneController.swift`` ``input(from:)``).

**Tracking.** ``limited: initializing`` for ``init_s`` after the start; then Poisson episodes of ``limited: excessive
motion`` (orientation only) and ``limited: relocalizing`` (nothing), each of exponential length. A relocalisation
ends with a jump of ψ_A and of the position.

**The app** is :mod:`.phone_app` (the Swift core, ported and golden-tested): the automatic Zero on the first normal
frame, requested Zeros at the given times (they take effect on the next frame, as the app's countdown hands them to
its frame queue), the lever arm, the hip height, the tree turn, the 95-byte message, the rate gate. **The link**
(``Transport/HubLink.swift``): at most ``link_max_in_flight`` sends in flight, a pose that cannot go is dropped and
not marked sent, so the next frame is due again; each message reaches the page after the link's latency, in order.

**Labels** (the pods' twist rule, :mod:`.labels`). The message, converted to +Z up by conjugation (it is a turn
within the reported frame), is q_m(t) = Y(-ψ_ph(t)) · R_C(t) · Exp(η) · K: the app's Zero contributes the
constant K = R̂_C(T0)⁻¹ · Y(ψ0 - [π if the tree setting is thorny]), which the twin reads from the app's Zero, and
ψ_ph = ψ0 + ψ_A (- π for thorny). So R_ph^m(t) = R_C(t) · K is the true frame the message describes, and
E_ph = q_m · R_ph^m⁻¹ gives ψ_ph and the tilt. The app's Zero cancels ARKit's own north ``a``: ψ_ph depends on
where the wearer faced at the Zero and on ARKit's drift since its start, not on where the camera looked when ARKit
started.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import Config
from ..conventions import exp_map, quat_conj, quat_mul, quat_rotate, quat_to_matrix, yaw_quat
from ..sim.simulate import fusion_error
from . import phone_app as app
from .labels import report_labels
from .page_frame import R_THREE_TO_WORLD, arkit_to_world_rot, arkit_to_world_turn, world_to_arkit_vec

__all__ = ["UPTIME_OFFSET_S", "ArkitTrack", "frame_times", "tracking_states", "arkit_track", "AppRun",
           "run_app", "message_frame_offset", "phone_labels", "phone_frame_psi", "phone_tables"]

UPTIME_OFFSET_S = 21000.0
"""``ARFrame.timestamp`` runs on the device's uptime clock: true time + this (any value; the app uses differences).
The 2026-11-14 log's first Zero was at an uptime of 21005 s."""


@dataclass
class ArkitTrack:
    t_s: np.ndarray  # (N,) true frame times
    tracking: np.ndarray  # (N,) object: Tracking labels
    psi_a_rad: np.ndarray  # (N,) ψ_A: ARKit's world heading error against the twin's world
    q_report: np.ndarray  # (N, 4) R̂_C, camera -> ARKit's world in +Z-up axes
    p_report_m: np.ndarray  # (N, 3) p̂_C, the same frame
    cam_q: list  # (N,) tuples: the camera rotation as the app receives it (ARKit's +Y up)
    cam_p: list  # (N,) tuples: the camera position as the app receives it
    a_rad: float  # ψ_A at the start
    scale_error: float
    episodes: list = field(default_factory=list)  # (kind, start s, end s)


def frame_times(t0_s: float, t1_s: float, cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """ARKit frame times from ``t0_s`` to ``t1_s``: the frame grid plus uniform jitter (kept increasing)."""
    rate = cfg["twin.phone.frame_rate_hz"]
    jit = cfg["twin.phone.frame_jitter_ms"] / 1000.0
    t = t0_s + np.arange(int(np.floor((t1_s - t0_s) * rate)) + 1) / rate
    if jit > 0:
        t = t + rng.uniform(-jit, jit, t.size)
        t[0] = max(t[0], t0_s)
    return np.maximum.accumulate(t)


def _episodes(t0: float, t1: float, per_min: float, mean_s: float, rng) -> list[tuple[float, float]]:
    out, t = [], t0
    if per_min <= 0:
        return out
    while True:
        t += rng.exponential(60.0 / per_min)
        if t >= t1:
            return out
        d = rng.exponential(mean_s)
        out.append((t, min(t + d, t1)))
        t += d


def tracking_states(t_s: np.ndarray, start_s: float, cfg: Config, rng: np.random.Generator):
    """Per frame: the tracking label, and the episodes ``[(kind, start, end)]`` (relocalising wins over limited)."""
    p = "twin.phone."
    t_init = start_s + cfg[p + "init_s"]
    t_end = float(t_s[-1])
    lim = _episodes(t_init, t_end, cfg[p + "limited_per_min"], cfg[p + "limited_mean_s"], rng)
    rel = _episodes(t_init, t_end, cfg[p + "reloc_per_min"], cfg[p + "reloc_mean_s"], rng)
    state = np.full(t_s.size, "normal", dtype=object)
    for a, b in lim:
        state[(t_s >= a) & (t_s < b)] = "limited:excessiveMotion"
    for a, b in rel:
        state[(t_s >= a) & (t_s < b)] = "limited:relocalizing"
    state[t_s < t_init] = "limited:initializing"
    eps = [("limited:excessiveMotion", a, b) for a, b in lim] + [("limited:relocalizing", a, b) for a, b in rel]
    return state, sorted(eps, key=lambda e: e[1])


def _view_heading_offset(r_cam) -> float:
    """``a``: the turn that puts the camera's level view direction on ARKit's -Z (the Z-up frame's +Y)."""
    v = quat_rotate(r_cam, np.array([0.0, 0.0, -1.0]))
    return float(np.arctan2(v[1], v[0]) - np.pi / 2)


def arkit_track(cfg: Config, t_s, r_cam, p_cam, rng: np.random.Generator) -> ArkitTrack:
    """ARKit's reports for the camera's true orientations ``r_cam`` (N, 4) and positions ``p_cam`` (N, 3), world +Z up,
    at frame times ``t_s`` (the AR session starts at ``t_s[0]``)."""
    p = "twin.phone."
    t = np.asarray(t_s, dtype=np.float64)
    r_cam = np.asarray(r_cam, dtype=np.float64)
    p_cam = np.asarray(p_cam, dtype=np.float64)
    n = t.size
    state, eps = tracking_states(t, float(t[0]), cfg, rng)
    a = _view_heading_offset(r_cam[0])
    sw = np.deg2rad(cfg[p + "heading_walk_deg_sqrt_min"]) / np.sqrt(60.0)
    walk = np.r_[0.0, np.cumsum(rng.normal(0.0, 1.0, n - 1) * sw * np.sqrt(np.diff(t)))]
    jumps = np.zeros(n)
    pos_jumps = np.zeros((n, 3))
    for kind, _, end in eps:
        if kind == "limited:relocalizing":
            after = t >= end
            jumps[after] += rng.normal(0.0, np.deg2rad(cfg[p + "reloc_heading_jump_deg"]))
            pos_jumps[after] += rng.normal(0.0, cfg[p + "reloc_position_jump_cm"] / 100.0, 3)
    psi = a + walk + jumps
    eta = fusion_error(n, cfg[p + "frame_rate_hz"], np.deg2rad(cfg[p + "tilt_sigma_deg"]), cfg[p + "tilt_tau_s"], rng)
    q_rep = quat_mul(quat_mul(yaw_quat(-psi), r_cam), exp_map(eta))
    scale = float(rng.uniform(-1.0, 1.0) * cfg[p + "scale_error_max"])
    steps = quat_rotate(yaw_quat(-psi[1:]), (1.0 + scale) * np.diff(p_cam, axis=0))
    pw = cfg[p + "position_walk_cm_sqrt_min"] / 100.0 / np.sqrt(60.0)
    steps = steps + rng.normal(0.0, 1.0, (n - 1, 3)) * pw * np.sqrt(np.diff(t))[:, None]
    p_rep = np.vstack([np.zeros(3), np.cumsum(steps, axis=0)]) + pos_jumps
    # the boundary: camera -> ARKit (+Y up), float32 as in ARFrame.camera.transform, then the app's reading of it
    m = (R_THREE_TO_WORLD.T @ quat_to_matrix(q_rep)).astype(np.float32).astype(np.float64)
    pa = world_to_arkit_vec(p_rep).astype(np.float32).astype(np.float64)
    cam_q = [app.q_from_columns(tuple(mk[:, 0]), tuple(mk[:, 1]), tuple(mk[:, 2])) for mk in m]
    cam_p = [tuple(v) for v in pa]
    return ArkitTrack(t, state, psi, q_rep, p_rep, cam_q, cam_p, a, scale, eps)


@dataclass
class AppRun:
    t_frame_s: np.ndarray  # (N,) true frame times
    messages: pd.DataFrame  # one row per message the link took
    zeros: list  # dicts: frame, t_s, epoch, heading_rad, requested, cam_q (ARKit, as the app stored it)
    zero_errors: list  # (t_s, reason)
    skipped: int  # messages the link refused (two in flight)
    levels: dict
    frame_q: np.ndarray | None = None  # (N, 4) every frame's reported orientation after the tree turn; NaN before
    frame_epoch: np.ndarray | None = None  # (N,) the Zero epoch of each frame (0 before the first Zero)


def _settings(cfg: Config, overrides: dict | None):
    s = {"rate_hz": cfg["twin.app.rate_hz"], "tree": cfg["twin.app.tree"], "facing": cfg["twin.app.facing"],
         "hip_height_m": cfg["twin.app.hip_height_m"], "lever_m": tuple(cfg["twin.app.lever_m"])}
    return s | (overrides or {})


def run_app(cfg: Config, track: ArkitTrack, rng: np.random.Generator, zero_requests_s=(),
            settings: dict | None = None, turn_rad: float = 0.0) -> AppRun:  # fmt: skip
    """Run the app (``twin.app`` settings, overridable) and the link on an ARKit track.

    ``turn_rad`` turns every pose the app sends by Y(turn) about the reported frame's +Y, position and orientation
    alike, and re-encodes it (the invariance checks of Step 3 brief §4.12 only; 0 leaves the app untouched)."""
    s = _settings(cfg, settings)
    turn = app.q_yaw(turn_rad) if turn_rad else None
    pipe = app.PosePipeline(app.HipsFrame(lever=tuple(float(c) for c in s["lever_m"]),
                                          hip_height=float(s["hip_height_m"]), facing=s["facing"]),
                            tree=s["tree"], rate_hz=float(s["rate_hz"]))  # fmt: skip
    lat_min = cfg["twin.phone.link_latency_min_ms"] / 1000.0
    lat_jit = cfg["twin.phone.link_latency_jitter_ms"] / 1000.0
    max_flight = int(cfg["twin.phone.link_max_in_flight"])
    requests = sorted(float(r) for r in zero_requests_s)
    rows, zeros, errors, in_flight = [], [], [], []
    skipped, nxt, epoch = 0, 0, 0
    levels = {k: 0 for k in range(len(app.LEVELS))}
    frame_q = np.full((len(track.t_s), 4), np.nan)
    frame_epoch = np.zeros(len(track.t_s), dtype=np.int64)
    t_prev = -np.inf
    for k, tk in enumerate(track.t_s):
        while nxt < len(requests) and requests[nxt] <= t_prev:
            nxt += 1
        if nxt < len(requests) and t_prev < requests[nxt] <= tk:  # asked between the last frame and this one
            pipe.request_zero()
            nxt += 1
        uptime = float(tk) + UPTIME_OFFSET_S
        out = pipe.process(app.FrameInput(uptime, track.cam_p[k], track.cam_q[k], app.Tracking(track.tracking[k])))
        if out.zero is not None:
            epoch = out.zero.epoch
            zeros.append({"frame": k, "t_s": float(tk), "epoch": epoch, "heading_rad": out.zero.heading,
                          "requested": out.zero_was_requested, "cam_q": out.zero.cam_rotation})  # fmt: skip
        if out.zero_error is not None:
            errors.append((float(tk), out.zero_error))
        if out.reported is not None:
            wire_p, wire_q = app.tree_wire(pipe.tree, *out.reported)
            if turn is not None:
                wire_p, wire_q = app.q_rotate(turn, wire_p), app.q_mul(turn, wire_q)
                if out.message is not None:
                    out.message = app.encode(wire_q, wire_p if out.message.has_position else None)
            frame_q[k] = wire_q
            frame_epoch[k] = epoch
        if out.message is None:
            t_prev = tk
            continue
        in_flight = [c for c in in_flight if c > tk]
        if len(in_flight) >= max_flight:
            skipped += 1
            t_prev = tk
            continue
        lat = lat_min + (rng.exponential(lat_jit) if lat_jit > 0 else 0.0)
        in_flight.append(tk + lat)
        pipe.mark_sent(uptime)
        levels[out.message.level] += 1
        rows.append({"frame": k, "t_send_s": float(tk), "t_arrive_s": float(tk + lat), "epoch": epoch,
                     "line": out.message.text, "level": out.message.level, "len_bytes": out.message.bytes,
                     "wire_q": wire_q, "wire_p": wire_p, "tracking": track.tracking[k]})  # fmt: skip
        t_prev = tk
    msgs = pd.DataFrame(rows)
    if len(msgs):
        msgs["t_arrive_s"] = np.maximum.accumulate(msgs["t_arrive_s"].to_numpy())  # one TCP stream: in order
    return AppRun(np.asarray(track.t_s), msgs, zeros, errors, skipped, levels, frame_q, frame_epoch)


def phone_frame_psi(run: AppRun, r_cam, tree: str) -> np.ndarray:
    """ψ_ph (rad) at every ARKit frame from the first Zero on (NaN before): the twist rule on the frame's reported
    orientation, as for the messages (:func:`phone_labels`)."""
    psi = np.full(run.frame_epoch.size, np.nan)
    by_epoch = {z["epoch"]: z for z in run.zeros}
    ok = run.frame_epoch > 0
    if not ok.any():
        return psi
    k = np.stack([message_frame_offset(by_epoch[e], tree) for e in run.frame_epoch[ok]])
    q_m = arkit_to_world_turn(run.frame_q[ok])
    psi[ok] = report_labels(q_m, quat_mul(np.asarray(r_cam, dtype=np.float64)[ok], k))[0]
    return psi


def message_frame_offset(zero: dict, tree: str) -> np.ndarray:
    """K: the constant right factor between the camera's true frame and the frame a message describes."""
    k_turn = -np.pi if tree == "thorny" else 0.0
    return quat_mul(quat_conj(arkit_to_world_rot(np.asarray(zero["cam_q"]))), yaw_quat(zero["heading_rad"] + k_turn))


def phone_labels(run: AppRun, track: ArkitTrack, r_cam, tree: str) -> pd.DataFrame:
    """Per message: ψ_ph, the tilt, the true message-frame orientation R_C K and the camera's true orientation.

    ``r_cam`` are the camera's true orientations at the track's frames; ``tree`` the app's tree setting."""
    if not len(run.messages):
        return pd.DataFrame()
    m = run.messages
    by_epoch = {z["epoch"]: z for z in run.zeros}
    frames = m["frame"].to_numpy()
    r = np.asarray(r_cam, dtype=np.float64)[frames]
    k = np.stack([message_frame_offset(by_epoch[e], tree) for e in m["epoch"]])
    q_m = arkit_to_world_turn(np.stack([np.asarray(q, float) for q in m["wire_q"]]))
    r_msg = quat_mul(r, k)
    psi, tilt = report_labels(q_m, r_msg)
    return pd.DataFrame({
        "t_arrive_s": m["t_arrive_s"].to_numpy(), "epoch": m["epoch"].to_numpy(), "psi_ph_rad": psi,
        "tilt_ph_rad": tilt, "psi_arkit_rad": track.psi_a_rad[frames],
        "true_qx": r_msg[:, 0], "true_qy": r_msg[:, 1], "true_qz": r_msg[:, 2], "true_qw": r_msg[:, 3],
        "cam_qx": r[:, 0], "cam_qy": r[:, 1], "cam_qz": r[:, 2], "cam_qw": r[:, 3],
        "tracking": m["tracking"].to_numpy(),
    })  # fmt: skip


def phone_tables(run: AppRun, labels: pd.DataFrame, host_epoch_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``phone.parquet`` (one row per message as the hub would log it, the numbers as rounded on the wire, the
    position null when orientation only) and ``labels_phone.parquet``."""
    m = run.messages
    if not len(m):
        return pd.DataFrame(), pd.DataFrame()
    parsed = [app.HubMessage(t, 0).parsed() for t in m["line"]]
    get = lambda key: np.array([p.get(key, np.nan) for p in parsed], dtype=np.float64)  # noqa: E731
    t_host = m["t_arrive_s"].to_numpy() + host_epoch_s
    phone = pd.DataFrame({"t_host_s": t_host, "line": m["line"].to_numpy(), "qx": get("x"), "qy": get("y"),
                          "qz": get("z"), "qw": get("w"), "px": get("px"), "py": get("py"), "pz": get("pz"),
                          "len_bytes": m["len_bytes"].to_numpy()})  # fmt: skip
    lab = labels.copy()
    lab.insert(0, "t_host_s", t_host)
    return phone, lab.drop(columns=["t_arrive_s"])
