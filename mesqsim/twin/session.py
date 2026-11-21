"""One twin session end to end (Step 3 brief §4.8, §4.11): ``python -m mesqsim twin-session``.

**The procedure** (true time t, seconds from Start; the program's first piece is a T-pose, at t_prog0):

1. *Box* [0, box_s): the pods lie flat in the box (their z up), turned by the box's heading plus each slot's error,
   in a row on a table in front of the wearer, and boot inside it (Step 1's boot spread, clocks, chip rates).
2. *Strapping* [box_s, t_prog0): each pod is carried to its strap in its own window (Step 1's ``strap_path``: a
   smoothstep slerp from the box to the worn pose with whole spins; the position by smoothstep), then follows the
   body, which holds the program's first pose until the program starts.
3. *Slips* (``twin.procedure``): a USB restart of one pod or of the two upper pods of a limb after the box (the
   filter restarts where the pod faces); the phone's Zero only the automatic one at Start; the app's camera
   setting a quarter turn off; its tree setting the wrong one.
4. The phone app starts (``app_start_s``) during the strapping, Zeros automatically at its first normal frame and,
   unless slipped, again on request in the T-pose; then the page's **Set T-pose**; the wearer's T-pose is the
   SMPL rest pose with Step 1's posture error per body joint.
5. Optionally a forward raise of both arms after the T-pose (``arm_raise``); then the recording runs to the end.

**Every pod**: its true orientation R_i(t) = F_i(t) M_i E_i (sites, mountings, the strap error) and position on the
session grid; per boot epoch, Step 1's chip timing; the sensors and the fusion of ``twin.fusion.mode`` on the
chip's own sample grid; the wire of ``twin.wire.mode``; Step 1's radio. **The phone**: ARKit, the app, the link
(:mod:`.phone`). **The page twin** (:mod:`.page`) takes every message in arrival order, the Set T-pose and the
recording, and writes the twin BVH.

**Files** (§4.11): Step 1's ``frames.parquet``, ``session.json``, ``labels_ticks.parquet``, ``labels_joints.parquet``,
``meta.json``; ``labels_steps.parquet`` (per 32 Hz step: the true segment orientations, for the scorer only, the
motion type, the transition flag, the slips in force); ``phone.parquet`` and ``labels_phone.parquet``; ``page/``
with the twin BVH. Labels: per pod tick ψ_i, the tilt error and ε_i = wrap(ψ_i - ψ_ph); per joint and 32 Hz step
δ_pc; per phone message ψ_ph and the true phone pose.

Every random draw comes from ``stream(seed, name)``, so one seed gives identical files.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..body import JOINTS, N_PODS, PODS
from ..codec import Q_SCALE, quantize_component
from ..config import Config
from ..conventions import exp_map, interp_quat, quat_canonical, quat_mul, wrap_rad, yaw_quat
from ..io import FRAMES_SCHEMA, write_json, write_parquet
from ..recording import knowable_ticks
from ..sim.heading import DENSE_HZ, twist_z_rad
from ..sim.motion import strap_path
from ..sim.radio import arrival_times, gilbert_elliott_lost
from ..sim.rng import stream
from ..sim.simulate import fusion_error
from ..sim.timing import pod_ms, simulate_timing
from . import phone as ph
from .bodymotion import body_tracks, on_segments
from .fusion import analytic_heading, analytic_reports, fuse, fusion_times
from .labels import report_labels
from .physics import imu_truth
from .program import JoinedMotion
from .sensors import draw_sensor_errors, ideal_gyro_rad_s, measure
from .wire import BONE_NAMES, json_line, json_message

__all__ = ["LIMBS", "MARKER_SITES", "TwinPod", "TwinPlan", "TwinSession", "plan_twin", "simulate_twin_session",
           "write_twin_session", "page_assets", "old_recorder", "drive_page"]

LIMBS = {"left_arm": (3, 4), "right_arm": (6, 7), "left_leg": (9, 10), "right_leg": (12, 13)}
"""The two upper pods of each limb, which one USB slip restarts together."""
GRID_HZ = 32.0
GRID_MARGIN_S = 0.25


@dataclass
class TwinPod:
    pod_id: int
    boot_s: float
    clock_ppm: float
    chip_rate_hz: float
    chip_start_s: float
    grid_first_ms: float
    box_slot_rad: float
    box_pos_m: list
    strap_window_s: tuple
    strap_spins: int
    battery_start_pct: float
    strap_error_rotvec: list  # E_i
    sensors: dict  # SensorErrors.to_dict()
    reboots_s: list = field(default_factory=list)
    analytic: dict = field(default_factory=dict)  # Step 1's drift and scale for the analytic mode


@dataclass
class TwinPlan:
    seed: int
    program: str
    rate_hz: float
    t_prog0_s: float
    t_end_s: float
    phases: dict
    host_epoch_s: float
    box_heading_rad: float
    pods: list
    slips: dict
    phone: dict
    page: dict
    posture_rotvec: list  # per body joint 1-21, the T-pose's posture error

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pods"] = [asdict(p) for p in self.pods]
        return d


@dataclass
class TwinSession:
    frames: pd.DataFrame
    labels_ticks: pd.DataFrame
    labels_joints: pd.DataFrame
    labels_steps: pd.DataFrame
    phone: pd.DataFrame
    labels_phone: pd.DataFrame
    session: dict
    meta: dict
    bvh: str
    plan: TwinPlan
    truth: dict = field(default_factory=dict)  # t_s (session grid) and the pods' marker sites, for Φ (not written)
    page_events: list = field(default_factory=list, repr=False)  # (t_arrive_s, kind, id, message): the page's input


MARKER_SITES = {"Hips": "HipsAlt", "Spine": "Spine", "Head": "Head", "LeftArm": "LeftArm", "RightArm": "RightArm",
                "LeftForeArm": "LeftForeArm", "RightForeArm": "RightForeArm", "LeftUpLeg": "LeftUpLeg",
                "RightUpLeg": "RightUpLeg", "LeftLeg": "LeftLeg", "RightLeg": "RightLeg"}
"""The twin's markers, named as Φ names the sites (``mesqsim.real.bvh.SITES``): on Nov 13 the markers sat on the
pods, the pelvis marker on the HipsAlt pod on the lower back (Step 3 brief §5.1), so the twin's are its pod sites."""


def old_recorder(bvh: str) -> str:
    """The older page's recorder (Nov 13): the frame time asserted as 1/30 s and no ';' provenance lines, as the
    Nov 13 files have them (measured: their headers)."""
    from .three_math import js_number

    out = []
    for line in bvh.split("\n"):
        if line.startswith(";"):
            continue
        out.append(f"Frame Time: {js_number(1 / 30)}" if line.startswith("Frame Time:") else line)
    return "\n".join(out)


def _u(rng, lohi) -> float:
    lo, hi = lohi
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def plan_twin(cfg1: Config, cfgt: Config, seed: int, joined: JoinedMotion, program: str) -> TwinPlan:
    """Draw the timeline, every pod's parameters, the slips, the phone's and the page's settings."""
    pr, sl = "twin.procedure.", "twin.program."
    r_time, r_pod, r_sens = stream(seed, "twin.timeline"), stream(seed, "twin.pods"), stream(seed, "twin.sensors")
    r_slip, r_post = stream(seed, "twin.slips"), stream(seed, "twin.posture")
    rate = float(cfgt[sl + "rate_hz"])
    box_s, strap_s = float(cfgt[pr + "box_s"]), float(cfgt[pr + "strapping_s"])
    t_prog0 = np.round((box_s + strap_s) * rate) / rate  # on the session grid
    n = joined.seq.poses.shape[0]
    t_end = t_prog0 + (n - 1) / rate
    first = joined.pieces[0]
    t_tpose_end = t_prog0 + (first["last"] + 1) / rate
    phases = {"box": (0.0, box_s), "strapping": (box_s, t_prog0), "tpose": (t_prog0, t_tpose_end),
              "motion": (t_tpose_end, t_end)}
    host_epoch = _u(r_time, cfg1["timing.host_epoch_s"])
    box_heading = float(r_time.uniform(-np.pi, np.pi))
    period_ms = cfg1["timing.send_period_us"] / 1000.0
    slot_sigma = np.deg2rad(cfg1["error_model.box_slot_deg"])
    strap_max = np.deg2rad(cfgt["twin.body.strap_error_max_deg"])
    move_lo, move_hi = cfg1["session.strap_move_s"]
    box_pos = np.asarray(cfgt[pr + "box_position_m"], dtype=float)
    pods = []
    for pod in PODS:
        i = pod.id
        boot = _u(r_pod, (0.0, cfg1["session.boot_spread_s"]))
        ppm = _u(r_pod, (-cfg1["timing.pod_clock_ppm"], cfg1["timing.pod_clock_ppm"]))
        chip_rate = _u(r_pod, cfg1["timing.chip_rate_hz"])
        chip_start = _u(r_pod, cfg1["timing.chip_start_s"])
        move = _u(r_pod, (move_lo, min(move_hi, strap_s)))
        w0 = box_s + _u(r_pod, (0.0, strap_s - move))
        spins = int(r_pod.integers(-cfg1["session.strap_spins_max"], cfg1["session.strap_spins_max"] + 1))
        battery = _u(r_pod, cfg1["battery.start_pct"])
        slot = float(r_pod.normal(0.0, slot_sigma)) if slot_sigma > 0 else 0.0
        strap_err = r_pod.uniform(-strap_max, strap_max, 3)
        kmax = cfg1["error_model.scale_factor_max"]
        analytic = {"drift_rad_per_s": float(r_pod.normal(0.0, np.deg2rad(cfg1["error_model.drift_sigma_deg_per_min"])
                                                          / 60.0)),
                    "scale_factor": float(r_pod.uniform(-kmax, kmax)) if kmax > 0 else 0.0}  # fmt: skip
        pods.append(TwinPod(
            pod_id=i, boot_s=boot, clock_ppm=ppm, chip_rate_hz=chip_rate, chip_start_s=chip_start,
            grid_first_ms=cfg1["timing.grid_start_ms"] + i * period_ms / cfg1["timing.send_phase_divisor"],
            box_slot_rad=slot, box_pos_m=(box_pos + [0.06 * (i - 7), 0.0, 0.0]).tolist(),
            strap_window_s=(w0, w0 + move),
            strap_spins=spins, battery_start_pct=battery, strap_error_rotvec=strap_err.tolist(),
            sensors=draw_sensor_errors(cfgt, r_sens).to_dict(), analytic=analytic))  # fmt: skip
    # slips, each with its own probability, in a fixed order of draws
    slips = {"limb_restart": None, "zero_at_start": False, "camera_mismatch_deg": 0.0, "tree_mismatch": False}
    if r_slip.random() < cfgt[pr + "p_limb_restart"]:
        if r_slip.random() < 0.5:
            ids = [int(r_slip.integers(N_PODS))]
            limb = PODS[ids[0]].name
        else:
            limb = list(LIMBS)[int(r_slip.integers(len(LIMBS)))]
            ids = list(LIMBS[limb])
        forced = cfgt[pr + "force_limb"]
        if forced:  # a scenario (Step 3 brief §8.3): this limb, the draws above still made
            limb, ids = forced, list(LIMBS[forced])
        latest = max(pods[i].strap_window_s[1] for i in ids)
        t_r = float(r_slip.uniform(latest + 0.5, max(latest + 0.6, t_prog0 - 3.5)))
        for i in ids:
            pods[i].reboots_s = [t_r + float(r_slip.uniform(0.0, 0.5))]
        slips["limb_restart"] = {"what": limb, "pods": ids, "t_s": t_r}
    slips["zero_at_start"] = bool(r_slip.random() < cfgt[pr + "p_zero_at_start"])
    if r_slip.random() < cfgt[pr + "p_camera_mismatch"]:
        slips["camera_mismatch_deg"] = float(r_slip.choice([-90.0, 90.0]))
    slips["tree_mismatch"] = bool(r_slip.random() < cfgt[pr + "p_tree_mismatch"])
    facing = cfgt["twin.app.facing"]
    if slips["camera_mismatch_deg"]:
        order = ["forward", "right", "backward", "left"]  # a quarter turn to the right per step
        facing = order[(order.index(facing) + (1 if slips["camera_mismatch_deg"] > 0 else -1)) % 4]
    tree = "matched" if slips["tree_mismatch"] else cfgt["twin.app.tree"]
    r_hand = stream(seed, "twin.hand")  # its own stream: adding the hand left every other draw as it was
    hand = {"belt_s": float(t_prog0 + cfgt[pr + "phone_belt_at_s"]),
            "height_m": _u(r_hand, cfgt[pr + "hand_height_m"]),
            "yaw_rad": float(r_hand.uniform(-1.0, 1.0) * np.deg2rad(cfgt[pr + "hand_yaw_deg"])),
            "pitch_rad": float(r_hand.uniform(-1.0, 1.0) * np.deg2rad(cfgt[pr + "hand_pitch_deg"]))}
    phone = {"app_start_s": float(t_prog0 + cfgt[pr + "app_start_s"]),
             "zero_request_s": None if slips["zero_at_start"] else float(t_prog0 + cfgt[pr + "zero_at_s"]),
             "settings": {"facing": facing, "tree": tree}, "hand": hand}  # fmt: skip
    page = {"tree": cfgt["twin.page.tree"], "mode": cfgt["twin.page.mode"],
            "set_tpose_s": float(t_prog0 + cfgt[pr + "set_tpose_at_s"]),
            "record_on_s": float(t_prog0 + cfgt[pr + "record_at_s"]), "record_off_s": float(t_end),
            "frame_interval_ms": float(cfgt["twin.page.frame_interval_ms"])}  # fmt: skip
    sigma = np.deg2rad(cfgt[pr + "posture_deg"]) / np.sqrt(3.0)
    posture = r_post.normal(0.0, sigma, (21, 3)).tolist()
    return TwinPlan(seed, program, rate, float(t_prog0), float(t_end), phases, host_epoch, box_heading, pods, slips,
                    phone, page, posture)  # fmt: skip


def _with_posture(joined: JoinedMotion, plan: TwinPlan):
    """The program's first T-pose with the wearer's posture error on every body joint (a copy of the poses), faded
    out over the blend that follows it (smoothstep), so the body leaves the T-pose as smoothly as the program does.
    (Applied to the T-pose alone, the error vanished in one frame: up to 15° along the arm's chain.)"""
    from scipy.spatial.transform import Rotation

    seq = joined.seq
    poses = seq.poses.copy()
    last = joined.pieces[0]["last"]
    n_fade = 0
    while last + 1 + n_fade < poses.shape[0] and joined.transition[last + 1 + n_fade]:
        n_fade += 1
    u = np.arange(1, n_fade + 1) / (n_fade + 1)
    weight = np.r_[np.ones(last + 1), 1.0 - u * u * (3.0 - 2.0 * u)]  # 1 through the T-pose, then down to 0
    err = np.asarray(plan.posture_rotvec)
    for j in range(1, 22):
        r = Rotation.from_rotvec(poses[: weight.size, 3 * j : 3 * j + 3])
        e = Rotation.from_rotvec(weight[:, None] * err[j - 1][None, :])
        poses[: weight.size, 3 * j : 3 * j + 3] = (e * r).as_rotvec()
    return type(seq)(poses, seq.trans, seq.betas, seq.gender, seq.fps, seq.source)


def _phase_of(plan: TwinPlan, t) -> np.ndarray:
    t = np.asarray(t, float)
    out = np.full(t.shape, "motion", dtype=object)
    for name, (a, b) in plan.phases.items():
        out[(t >= a) & (t < b)] = name
    return out


def page_assets(tree_id: str):
    """The page twin's rig, tree and slerp table: from the mesquite-mocap clone if present, else from the page golden
    fixtures (which hold the velvet and smooth trees as the page read them)."""
    from ..paths import data_dir
    from . import golden
    from .page import load_rig, load_slerp, load_tree

    mocap = data_dir() / "mesquite-mocap"
    if (mocap / "trees" / tree_id / "meta.json").is_file() and (mocap / "glbs" / "scene2.glb").is_file():
        return load_rig(mocap / "glbs" / "scene2.glb"), load_tree(tree_id, mocap), load_slerp(mocap)
    case = golden.load_case(f"tpose_leg_swing_{tree_id}")
    return golden.rig(), case["provenance"]["tree"], case["provenance"]["slerp"]


HAND_MOVE_S = 2.0
HAND_FORWARD_M = 0.3


def _phone_in_hand(t, r_belt, p_belt, hand: dict):
    """Before the phone goes on the belt it is in the hand: turned by ``yaw_rad`` about the vertical and tilted by
    ``pitch_rad`` about the camera's own horizontal axis (its +Y), ``height_m`` above the belt and 0.3 m in front of
    it along the camera's view. Over the last 2 s before ``belt_s`` it moves to the belt (smoothstep). ARKit starts
    in the hand, so its world's origin and north are the hand's; a Zero taken there keeps them (Step 3 brief §2.4)."""
    from ..conventions import quat_slerp

    t = np.asarray(t, dtype=np.float64)
    u = np.clip((t - (hand["belt_s"] - HAND_MOVE_S)) / HAND_MOVE_S, 0.0, 1.0)
    w = u * u * (3.0 - 2.0 * u)
    if np.all(w >= 1.0):
        return r_belt, p_belt
    tilt = np.array([0.0, np.sin(hand["pitch_rad"] / 2), 0.0, np.cos(hand["pitch_rad"] / 2)])
    r_hand = quat_mul(quat_mul(yaw_quat(np.full(t.size, hand["yaw_rad"])), r_belt), tilt)
    from ..conventions import quat_rotate

    view = quat_rotate(r_hand, np.array([0.0, 0.0, -1.0]))
    view[:, 2] = 0.0
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-9)
    p_hand = p_belt + HAND_FORWARD_M * view + np.array([0.0, 0.0, hand["height_m"]])
    return quat_slerp(r_hand, r_belt, w), p_hand + w[:, None] * (p_belt - p_hand)


def _epochs(cfg1: Config, p: TwinPod, t_end: float, rng) -> list[dict]:
    silence = cfg1["error_model.reboot_silence_s"]
    grid_start = cfg1["timing.grid_start_ms"] / 1000.0
    starts = [(p.boot_s, p.chip_start_s)]
    for t_r in p.reboots_s:
        starts.append((t_r + max(0.0, silence - grid_start), _u(rng, cfg1["timing.chip_start_s"])))
    ends = list(p.reboots_s) + [t_end]
    return [{"boot_s": b, "chip_start_s": c, "end_s": e} for (b, c), e in zip(starts, ends, strict=True)]


def simulate_twin_session(cfg1: Config, cfgt: Config, seed: int, joined: JoinedMotion, model, program: str,
                          pod_turn_rad=None, phone_turn_rad: float = 0.0, room_shift_m=(0.0, 0.0, 0.0),
                          page: bool = True) -> TwinSession:  # fmt: skip
    """Simulate one session. The invariance checks (§4.12) may turn every pod's report by ``pod_turn_rad`` (a number,
    or a dict pod id -> angle) and the phone's messages by ``phone_turn_rad`` about the vertical, or move the room's
    origin by ``room_shift_m``; every random stream is unaffected by them. ``page=False`` skips the page twin (no
    BVH, no page events): the training data stores none, and every other output is the same."""
    plan = plan_twin(cfg1, cfgt, seed, joined, program)
    rate = plan.rate_hz
    seq = _with_posture(joined, plan)
    tracks = body_tracks(seq, model)
    frame = cfgt["twin.body.pod_frame"]
    if frame == "segment":  # the ablation: pods and phone rigid on their segments (Step 3 brief §4.2, §8.1)
        tracks = on_segments(tracks)
    elif frame != "surface":
        raise ValueError(f"twin.body.pod_frame must be surface or segment, not {frame!r}")
    shift = np.asarray(room_shift_m, dtype=float)
    n_prog = seq.poses.shape[0]
    j0 = int(round(plan.t_prog0_s * rate))
    t_grid = np.arange(j0 + n_prog) / rate
    k_prog = np.clip(np.arange(t_grid.size) - j0, 0, n_prog - 1)  # the body holds its first pose before the program
    site = {name: k for k, name in enumerate(tracks.names)}
    mode = cfgt["twin.fusion.mode"]
    wire_mode = cfgt["twin.wire.mode"]
    m_over = int(cfgt["twin.imu.oversample"])
    cutoff = float(cfgt["twin.physics.position_cutoff_hz"])
    cfg_time = cfg1.with_overrides({"timing.send_period_us": int(cfgt["twin.wire.json_send_min_ms"]) * 1000}) \
        if wire_mode == "json" else cfg1
    rngs = {k: stream(seed, "twin." + k) for k in ("timing", "radio", "loss", "noise", "epochs", "analytic")}
    turns = pod_turn_rad if isinstance(pod_turn_rad, dict) else {i: float(pod_turn_rad or 0.0) for i in range(N_PODS)}

    # ------------------------------------------------------------------------------------------- the phone
    t_ph = ph.frame_times(plan.phone["app_start_s"], plan.t_end_s, cfgt, stream(seed, "twin.phone_frames"))
    kph = site["Phone"]
    r_cam = interp_quat(t_grid, tracks.q[k_prog, kph], t_ph)
    p_cam = np.stack([np.interp(t_ph, t_grid, tracks.p[k_prog, kph, a]) for a in range(3)], axis=1) + shift
    r_cam, p_cam = _phone_in_hand(t_ph, r_cam, p_cam, plan.phone["hand"])
    track = ph.arkit_track(cfgt, t_ph, r_cam, p_cam, stream(seed, "twin.arkit"))
    zero_req = [] if plan.phone["zero_request_s"] is None else [plan.phone["zero_request_s"]]
    run = ph.run_app(cfgt, track, stream(seed, "twin.app"), zero_requests_s=zero_req,
                     settings=plan.phone["settings"], turn_rad=phone_turn_rad)  # fmt: skip
    app_tree = plan.phone["settings"]["tree"]
    psi_frame = ph.phone_frame_psi(run, r_cam, app_tree)
    lab_phone = ph.phone_labels(run, track, r_cam, app_tree)

    def psi_phone_at(t):
        out = np.full(np.shape(t), np.nan)
        for e in np.unique(run.frame_epoch[run.frame_epoch > 0]):
            sel = run.frame_epoch == e
            ts, ps = t_ph[sel], np.unwrap(psi_frame[sel])
            inside = (t >= ts[0]) & (t <= (t_ph[np.flatnonzero(sel)[-1] + 1] if np.flatnonzero(sel)[-1] + 1 < t_ph.size
                                           else plan.t_end_s))  # fmt: skip
            out[inside] = wrap_rad(np.interp(t[inside], ts, ps))
        return out

    # ------------------------------------------------------------------------------------------- the pods
    frames, ticks, page_events, first_send = [], [], [], []
    sample_psi = []  # per pod: (t, psi unwrapped per epoch, epoch) of every chip sample
    n_lost = 0
    active = {int(i) for i in cfgt["twin.suit.pods"]}
    for i in range(N_PODS):
        if i not in active:  # a suit without this pod: it sends nothing, and the page leaves its bone at rest
            sample_psi.append([])
            continue
        p = plan.pods[i]
        name = PODS[i].name
        q_worn = quat_mul(tracks.q[k_prog, site[name]], exp_map(np.asarray(p.strap_error_rotvec)))
        p_worn = tracks.p[k_prog, site[name]] + shift
        w0, w1 = p.strap_window_s
        q_box = yaw_quat(plan.box_heading_rad + p.box_slot_rad)
        k1 = min(int(np.ceil(w1 * rate)), t_grid.size - 1)
        q_i = q_worn.copy()
        pre = t_grid < w1
        q_i[pre] = strap_path(t_grid[pre], w0, w1, q_box, q_worn[k1], p.strap_spins)
        u = np.clip((t_grid - w0) / max(w1 - w0, 1e-9), 0.0, 1.0)
        g = (u * u * (3.0 - 2.0 * u))[:, None]
        box_p = np.asarray(p.box_pos_m) + shift
        p_i = p_worn.copy()
        p_i[pre] = (box_p + g * (p_worn[k1] - box_p))[pre]
        epochs = _epochs(cfg1, p, plan.t_end_s, rngs["epochs"])
        bias = None
        parts, psi_parts = [], []
        for e, ep in enumerate(epochs):
            tm = simulate_timing(cfg_time, p, ep["end_s"], rngs["timing"], boot_s=ep["boot_s"],
                                 chip_start_s=ep["chip_start_s"])  # fmt: skip
            ts = tm.t_sample_s
            if mode == "analytic":
                dense = np.arange(ts[0], ts[-1] + 1.0 / DENSE_HZ, 1.0 / DENSE_HZ)
                r_dense = interp_quat(t_grid, q_i, dense)
                r_s = interp_quat(t_grid, q_i, ts)
                hd = analytic_heading(dense, r_dense, ep["boot_s"], ep["end_s"], ts[0], float(twist_z_rad(r_s[0])),
                                      p.analytic["drift_rad_per_s"], p.analytic["scale_factor"],
                                      np.deg2rad(cfg1["error_model.arw_deg_per_sqrt_s"]), rngs["analytic"])
                eta = fusion_error(ts.size, p.chip_rate_hz, np.deg2rad(cfg1["error_model.tilt_sigma_deg"]),
                                   cfg1["error_model.tilt_tau_s"], rngs["analytic"])  # fmt: skip
                q_rep = analytic_reports(r_s, hd.psi(ts), eta)
            else:
                tf = fusion_times(ts[0], ts.size, p.chip_rate_hz, m_over)
                dt = 1.0 / (m_over * p.chip_rate_hz)
                imu = imu_truth(t_grid, q_i, p_i, rate, cutoff_hz=cutoff, t_eval=np.r_[tf[0] - dt, tf])
                from .sensors import SensorErrors

                err = SensorErrors(**{k: (np.asarray(v) if isinstance(v, list) else v) for k, v in p.sensors.items()})
                gyr, acc, b = measure(err, ideal_gyro_rad_s(imu.q, dt), imu.f_m_s2[1:], dt, rngs["noise"], bias)
                bias = b[-1]
                q_rep = fuse(cfgt, gyr, acc, dt, mode=mode).q[::m_over]
                r_s = imu.q[1:][::m_over]
            q_rep = quat_canonical(quat_mul(yaw_quat(np.full(ts.size, turns.get(i, 0.0))), q_rep))
            psi_s, tilt_s = report_labels(q_rep, r_s)
            psi_parts.append((ts, np.unwrap(psi_s), np.full(ts.size, e)))
            j = np.arange(tm.t_send_s.size)
            idx = tm.sample_idx
            parts.append(pd.DataFrame({
                "epoch": np.full(j.size, e, np.int32), "tick": j.astype(np.int64), "t_send_s": tm.t_send_s,
                "sample_index": idx.astype(np.int64), "t_sample_s": ts[idx], "t_decode_s": tm.t_decode_s[idx],
                "ms_lo": tm.ms_lo, "fresh": tm.fresh, "psi_rad": psi_s[idx], "tilt_rad": tilt_s[idx],
                "qx_rep": q_rep[idx, 0], "qy_rep": q_rep[idx, 1], "qz_rep": q_rep[idx, 2], "qw_rep": q_rep[idx, 3],
                "pod_qx": r_s[idx, 0], "pod_qy": r_s[idx, 1], "pod_qz": r_s[idx, 2], "pod_qw": r_s[idx, 3],
            }))  # fmt: skip
        sample_psi.append(psi_parts)
        seq_i = pd.concat(parts, ignore_index=True)
        first_send.append(seq_i["t_send_s"].iloc[0])
        t_arr = arrival_times(cfg1, seq_i["t_send_s"].to_numpy(), rngs["radio"])
        lost = gilbert_elliott_lost(len(seq_i), cfg1["radio.loss_mean"], cfg1["radio.burst_mean_frames"], rngs["loss"])
        interval = np.where(t_arr <= plan.t_end_s, 0, -1)
        received = ~lost & (interval >= 0)
        n_lost += int(np.sum(lost & (interval >= 0)))
        keep = knowable_ticks(received, seq_i["epoch"].to_numpy(), interval)
        lab = seq_i[keep].reset_index(drop=True)
        rec = received[keep]
        t_arr = t_arr[keep]
        q_rep = lab[["qx_rep", "qy_rep", "qz_rep", "qw_rep"]].to_numpy()
        count = (lab["tick"].to_numpy() % 65536).astype(np.uint16)
        drain = cfg1["battery.drain_pct_per_h"] * (lab["t_send_s"].to_numpy() - p.boot_s) / 3600.0
        batt = np.clip(np.floor(p.battery_start_pct - drain), 0, 100).astype(np.uint8)
        if wire_mode == "json":
            millis = np.floor(pod_ms(p, lab["t_send_s"].to_numpy(), p.boot_s)).astype(np.int64)
            lines = [json_line(i, q, c, ms, b, int(cfgt["twin.wire.json_decimals"]))
                     for q, c, ms, b in zip(q_rep, count, millis, batt, strict=True)]  # fmt: skip
            msgs = [json_message(s) for s in lines]
            qi = quantize_component(np.array([[mm[k] for k in "xyzw"] for mm in msgs]))
        else:
            qi = quantize_component(q_rep)
            msgs = None
        r = rec
        frames.append(pd.DataFrame({
            "t_host_s": t_arr[r] + plan.host_epoch_s, "pod_id": np.full(r.sum(), i, np.uint8), "batt": batt[r],
            "qx": qi[r, 0], "qy": qi[r, 1], "qz": qi[r, 2], "qw": qi[r, 3], "count": count[r],
            "ms_lo": lab["ms_lo"].to_numpy()[r].astype(np.uint16)}))  # fmt: skip
        ms_lo = lab["ms_lo"].to_numpy()
        for k in np.flatnonzero(r) if page else ():
            if msgs is not None:
                m = msgs[k]
            else:
                m = {"bone": BONE_NAMES[i], "id": i, "batt": int(batt[k]) / 100, "x": int(qi[k, 0]) / Q_SCALE,
                     "y": int(qi[k, 1]) / Q_SCALE, "z": int(qi[k, 2]) / Q_SCALE, "w": int(qi[k, 3]) / Q_SCALE,
                     "count": int(count[k]), "millis": int(ms_lo[k])}  # fmt: skip
            page_events.append((float(t_arr[k]), 1, i, m))
        lab = lab.drop(columns=["qx_rep", "qy_rep", "qz_rep", "qw_rep"])
        lab.insert(0, "pod_id", np.full(len(lab), i, np.uint8))
        lab["count"] = count
        lab["t_arrive_s"] = t_arr
        lab["received"] = rec
        lab["phase"] = _phase_of(plan, lab["t_send_s"].to_numpy()).astype(str)
        lab["psi_ph_rad"] = psi_phone_at(lab["t_sample_s"].to_numpy())
        lab["eps_rad"] = wrap_rad(lab["psi_rad"].to_numpy() - lab["psi_ph_rad"].to_numpy())
        ticks.append(lab)

    frames_df = pd.concat(frames, ignore_index=True).sort_values(["t_host_s", "pod_id"], kind="mergesort",
                                                                  ignore_index=True)  # fmt: skip
    ticks_df = pd.concat(ticks, ignore_index=True)

    # ------------------------------------------------------------------------------------------- the page twin
    bvh = ""
    if page:
        for row in run.messages.itertuples():
            page_events.append((float(row.t_arrive_s), 0, 99, json.loads(row.line)))
        page_events.sort(key=lambda e: (e[0], e[1], e[2]))
        bvh = _drive_page(plan, page_events, cfgt["twin.page.recorder"])

    # ------------------------------------------------------------------------------------------- 32 Hz labels
    t0 = np.ceil((max(first_send) + GRID_MARGIN_S) * GRID_HZ) / GRID_HZ
    grid = np.arange(int(np.floor((plan.t_end_s - t0) * GRID_HZ)) + 1) / GRID_HZ + t0
    psi_grid = np.full((grid.size, N_PODS), np.nan)
    off_grid = np.full((grid.size, N_PODS), np.nan)  # ψ at each epoch's first sample: its north (the O1 oracle)
    for i, parts in enumerate(sample_psi):
        for ts, ps, _ in parts:
            inside = (grid >= ts[0]) & (grid <= ts[-1])
            psi_grid[inside, i] = wrap_rad(np.interp(grid[inside], ts, ps))
            off_grid[inside, i] = wrap_rad(ps[0])
    rows = []
    for k, jt in enumerate(JOINTS):
        rows.append(pd.DataFrame({"step": np.arange(grid.size, dtype=np.int64), "t_s": grid,
                                  "phase": _phase_of(plan, grid).astype(str), "joint": np.full(grid.size, k, np.uint8),
                                  "delta_rad": wrap_rad(psi_grid[:, jt.child] - psi_grid[:, jt.parent]),
                                  "delta_offset_rad": wrap_rad(off_grid[:, jt.child] - off_grid[:, jt.parent])}))
    joints_df = pd.concat(rows, ignore_index=True)
    kg = np.clip(np.round((grid - plan.t_prog0_s) * rate).astype(int), 0, n_prog - 1)
    in_prog = grid >= plan.t_prog0_s
    steps = {"step": np.arange(grid.size, dtype=np.int64), "t_s": grid, "phase": _phase_of(plan, grid).astype(str),
             "motion": np.where(in_prog, joined.kind[kg], "preparation").astype(str),
             "transition": in_prog & joined.transition[kg]}  # fmt: skip
    lr = plan.slips["limb_restart"]
    steps["slip_restarted"] = np.full(grid.size, bool(lr)) & (grid >= (lr["t_s"] if lr else np.inf))
    steps["slip_zero_at_start"] = np.full(grid.size, plan.slips["zero_at_start"])
    steps["slip_camera_mismatch_deg"] = np.full(grid.size, plan.slips["camera_mismatch_deg"])
    steps["slip_tree_mismatch"] = np.full(grid.size, plan.slips["tree_mismatch"])
    for i in range(N_PODS):
        steps[f"psi{i}_rad"] = psi_grid[:, i]
    steps["psi_ph_rad"] = psi_phone_at(grid)
    for i in range(N_PODS):
        seg = tracks.q_segment[kg, site[PODS[i].name]]
        for a, c in enumerate("xyzw"):
            steps[f"seg{i}_q{c}"] = seg[:, a]
    steps_df = pd.DataFrame(steps)

    # ------------------------------------------------------------------------------------------- tables, files
    phone_df, labels_phone = ph.phone_tables(run, lab_phone, plan.host_epoch_s)
    session = {"schema_version": 1, "pods": list(range(N_PODS)),
               "phases": [{"name": nm, "t_host_start_s": a + plan.host_epoch_s, "t_host_end_s": b + plan.host_epoch_s,
                           "recorded": True} for nm, (a, b) in plan.phases.items()]}  # fmt: skip
    meta = {"plan": plan.to_dict(), "program": {"name": program, "pieces": joined.pieces, "source": joined.seq.source,
                                                "gender": joined.seq.gender, "betas": joined.seq.betas},
            "config_hash": {"step1": cfg1.hash(), "twin": cfgt.hash()},
            "fusion_mode": mode, "wire_mode": wire_mode,
            "counts": {"frames": len(frames_df), "radio_lost": n_lost, "phone_messages": len(phone_df),
                       "phone_skipped": run.skipped, "phone_levels": run.levels, "zeros": len(run.zeros),
                       "zero_errors": len(run.zero_errors), "page_events": len(page_events)},
            "zeros": [{k: v for k, v in z.items() if k != "cam_q"} for z in run.zeros],
            "arkit": {"a_rad": track.a_rad, "scale_error": track.scale_error, "episodes": track.episodes}}  # fmt: skip
    truth = {"t_s": t_grid, "record_on_s": plan.page["record_on_s"],
             "frame_interval_s": plan.page["frame_interval_ms"] / 1000.0,
             "sites": {m: tracks.p[k_prog, site[n]] + shift for m, n in MARKER_SITES.items()}}  # fmt: skip
    return TwinSession(frames_df, ticks_df, joints_df, steps_df, phone_df, labels_phone, session, meta, bvh, plan,
                       truth, page_events)  # fmt: skip


def _drive_page(plan: TwinPlan, page_events: list, recorder: str, mode: str | None = None,
                heading_correction=None) -> str:  # fmt: skip
    """Every message to the page in arrival order, with the Set T-pose and the recorder's ticks between them: the
    page twin's BVH."""
    from .page import PageTwin

    rig, tree, slerp = page_assets(plan.page["tree"])
    page = PageTwin(rig, tree, slerp, mode=mode or plan.page["mode"], heading_correction=heading_correction)
    ctl = [(plan.page["set_tpose_s"], "calibrate"), (plan.page["record_on_s"], "record_start")]
    t_tick = plan.page["record_on_s"] + plan.page["frame_interval_ms"] / 1000.0
    while t_tick < plan.page["record_off_s"]:
        ctl.append((t_tick, "record_tick"))
        t_tick += plan.page["frame_interval_ms"] / 1000.0
    ctl.sort()
    c = 0
    for t_ev, _, _, msg in page_events:
        while c < len(ctl) and ctl[c][0] <= t_ev:
            getattr(page, ctl[c][1])() if ctl[c][1] == "calibrate" else getattr(page, ctl[c][1])(ctl[c][0] * 1000.0)
            c += 1
        page.handle(dict(msg), t_ev * 1000.0)
    while c < len(ctl):
        getattr(page, ctl[c][1])() if ctl[c][1] == "calibrate" else getattr(page, ctl[c][1])(ctl[c][0] * 1000.0)
        c += 1
    bvh = page.record_stop(plan.page["record_off_s"] * 1000.0)
    return old_recorder(bvh) if recorder == "old" else bvh


def drive_page(s: TwinSession, cfgt: Config, mode: str | None = None, heading_correction=None) -> str:
    """The page twin's BVH of session ``s`` again, in ``mode`` (default the session's), every pod reading first turned
    by ``heading_correction(bone, t_ms)`` radians about its world's +Z (the phone never): the page-level test (Step 3
    brief §8.3). With neither given it is ``s.bvh`` exactly."""
    return _drive_page(s.plan, s.page_events, cfgt["twin.page.recorder"], mode, heading_correction)


def write_twin_session(folder: Path, s: TwinSession) -> Path:
    """Write the session's files (§4.11) into ``folder``; byte for byte the same for the same session."""
    folder = Path(folder)
    (folder / "page").mkdir(parents=True, exist_ok=True)
    write_parquet(folder / "frames.parquet", s.frames, FRAMES_SCHEMA)
    write_parquet(folder / "labels_ticks.parquet", s.labels_ticks)
    write_parquet(folder / "labels_joints.parquet", s.labels_joints)
    write_parquet(folder / "labels_steps.parquet", s.labels_steps)
    write_parquet(folder / "phone.parquet", s.phone)
    write_parquet(folder / "labels_phone.parquet", s.labels_phone)
    write_json(folder / "session.json", s.session)
    write_json(folder / "meta.json", s.meta)
    (folder / "page" / "twin.bvh").write_text(s.bvh, encoding="utf-8", newline="\n")
    return folder
