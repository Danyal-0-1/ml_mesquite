"""Motion programs, and long sessions joined from short AMASS clips (Step 3 brief §4.8, §4.9).

**One subject per session.** A session is built from one subject's clips, joined end to end; clips stay in their
subject's split (``manifests/split_manifest.csv``). This settles ``docs/DATA.md`` §1.2 for Step 3.

**Joining.** Every piece is resampled to ``twin.program.rate_hz`` (slerp per joint, lerp of the translation). Each
next piece is turned about the vertical through its first pelvis position, and shifted, so that its first root
pose continues the previous piece's last: the same heading (the root's forward, canonical +z, made horizontal) and
the same horizontal pelvis position; it keeps its own height. Between the two, a blend of ``twin.program.blend_s``
slerps every joint's local rotation and lerps the root from the one pose to the other (smoothstep weights, so it
starts and ends at rest); its frames are labelled ``transition``.

In AMASS's SMPL+H convention the model frame is the world frame (+Z up); ``root_orient`` turns the canonical body
(x the subject's left, y up, z forward) about the rest pelvis J0, and the pelvis sits at J0 + trans. Turning a piece
by α about the vertical through its first pelvis point p0: root' = Z(α) root; pelvis' = Z(α)(pelvis - p0) + target.

**Synthetic pieces** stand in where the subject has no clip, built on the subject's own rest pose (a T-pose) or a
standing frame of its own: a T-pose hold, hip twists, leg swings (forward and back, to the side), arm raises
(forward, to the side), a still stand. Their ``source`` says so. Local joint rotations are relative to the parent,
whose axes are the canonical ones at rest, so a rotation multiplied on the left turns the limb about its joint in
the parent's axes.

**The join check** (§4.9): no joint's angular speed at a join exceeds the P99 of the pieces' own speeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from ..conventions import quat_canonical, quat_mul
from ..data.amass import amass_dir
from ..paths import REPO_ROOT
from .smplh import SmplhModel, SmplhSequence, load_smplh_sequence, rest_pose

__all__ = ["N_JOINTS", "Piece", "JoinedMotion", "resample", "clip_piece", "hold_piece", "tpose_piece",
           "synthetic_piece", "join", "join_check", "subject_clips", "random_program", "PROTOCOL_SUBJECT",
           "protocol_program", "training_subjects", "make_program"]

N_JOINTS = 52
BODY = 22  # the root and the 21 body joints; the hands are kept as they are
L_HIP, R_HIP, SPINE1, SPINE2, SPINE3 = 1, 2, 3, 6, 9
L_SHOULDER, R_SHOULDER = 16, 17
_UPRIGHT = Rotation.from_rotvec([np.pi / 2, 0.0, 0.0])  # canonical y (up) -> world +Z; canonical z -> world -Y


@dataclass
class Piece:
    kind: str  # motion type: tpose, still, walk, turn, squat, leg_swing, leg_side, arm_forward, arm_side, twist, ...
    source: str  # "clip:<path>[@start-end s]" or "synthetic:<name>"
    rot: np.ndarray  # (n, 52, 4) local joint rotations, quaternions (x, y, z, w); joint 0 is the root orient
    trans: np.ndarray  # (n, 3) metres
    travel_turn_rad: float | None = None  # when set: turn so the piece travels at the previous travel + this


def _travel(p: Piece) -> float:
    d = p.trans[-1, :2] - p.trans[0, :2]
    return float(np.arctan2(d[1], d[0]))


@dataclass
class JoinedMotion:
    seq: SmplhSequence
    kind: np.ndarray  # (N,) object: the motion type per frame ("transition" on blends)
    transition: np.ndarray  # (N,) bool
    piece: np.ndarray  # (N,) int: the piece index (-1 on blends)
    pieces: list = field(default_factory=list)  # dicts: kind, source, first, last (frames)


def _to_rot(poses: np.ndarray) -> np.ndarray:
    n = poses.shape[0]
    return Rotation.from_rotvec(poses.reshape(n * N_JOINTS, 3)).as_quat().reshape(n, N_JOINTS, 4)


def _to_poses(rot: np.ndarray) -> np.ndarray:
    n = rot.shape[0]
    return Rotation.from_quat(rot.reshape(n * N_JOINTS, 4)).as_rotvec().reshape(n, N_JOINTS * 3)


def _slerp(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Per-element slerp of quaternion arrays ``a``, ``b`` (..., 4) with weights ``w`` (...), the short way."""
    b = np.where(np.sum(a * b, axis=-1, keepdims=True) < 0, -b, b)
    d = np.clip(np.sum(a * b, axis=-1, keepdims=True), -1.0, 1.0)
    th = np.arccos(d)
    s = np.sin(th)
    w = np.asarray(w, float)[..., None]
    small = s < 1e-9
    ka = np.where(small, 1.0 - w, np.sin((1.0 - w) * th) / np.where(small, 1.0, s))
    kb = np.where(small, w, np.sin(w * th) / np.where(small, 1.0, s))
    q = ka * a + kb * b
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def resample(rot: np.ndarray, trans: np.ndarray, fps_in: float, fps_out: float) -> tuple[np.ndarray, np.ndarray]:
    """Joint rotations (n, 52, 4) and translations (n, 3) from ``fps_in`` to ``fps_out``, both ends kept."""
    n = rot.shape[0]
    t_out = np.arange(int(np.floor((n - 1) / fps_in * fps_out + 1e-9)) + 1) / fps_out
    x = t_out * fps_in
    k = np.clip(np.floor(x).astype(int), 0, n - 2) if n > 1 else np.zeros(t_out.size, int)
    if n == 1:
        return rot[[0] * t_out.size].copy(), trans[[0] * t_out.size].copy()
    u = x - k
    r = _slerp(rot[k], rot[k + 1], np.broadcast_to(u[:, None], (u.size, N_JOINTS)))
    tr = trans[k] + u[:, None] * (trans[k + 1] - trans[k])
    return r, tr


def clip_piece(path: str, kind: str, rate_hz: float, start_s: float | None = None,
               end_s: float | None = None) -> Piece:  # fmt: skip
    """A piece from an AMASS file (relative to the AMASS folder), optionally cut to [start_s, end_s]."""
    seq = load_smplh_sequence(amass_dir() / path)
    a = 0 if start_s is None else int(round(start_s * seq.fps))
    b = seq.poses.shape[0] if end_s is None else min(seq.poses.shape[0], int(round(end_s * seq.fps)) + 1)
    rot, tr = resample(_to_rot(seq.poses[a:b]), seq.trans[a:b], seq.fps, rate_hz)
    cut = "" if start_s is None and end_s is None else f"@{(start_s or 0):g}-{(end_s or b / seq.fps):g}s"
    return Piece(kind, f"clip:{path}{cut}", rot, tr)


def hold_piece(rot0: np.ndarray, trans0: np.ndarray, seconds: float, rate_hz: float, kind: str = "still",
               source: str = "synthetic:hold") -> Piece:  # fmt: skip
    n = max(1, int(round(seconds * rate_hz)))
    return Piece(kind, source, np.repeat(rot0[None], n, axis=0), np.repeat(trans0[None], n, axis=0))


def standing_trans(model: SmplhModel, betas, rot: np.ndarray) -> np.ndarray:
    """The translation that stands a pose (52, 4) with its lowest rest vertex on the floor (z = 0), the pelvis
    over the world origin (only the height is used: pieces are shifted horizontally when joined)."""
    v, j = rest_pose(model, betas)
    root = Rotation.from_quat(rot[0])
    low = (root.apply(v - j[0]) + j[0])[:, 2].min()
    return np.array([-j[0, 0], -j[0, 1], -low])


def tpose_piece(model: SmplhModel, betas, seconds: float, rate_hz: float) -> Piece:
    """The SMPL rest pose (a T-pose), upright, feet on the floor, held."""
    rot = np.tile([0.0, 0.0, 0.0, 1.0], (N_JOINTS, 1))
    rot[0] = _UPRIGHT.as_quat()
    return hold_piece(rot, standing_trans(model, betas, rot), seconds, rate_hz, "tpose", "synthetic:tpose")


def _about(axis: int, angle) -> np.ndarray:
    v = np.zeros((np.size(angle), 3))
    v[:, axis] = angle
    return Rotation.from_rotvec(v).as_quat()


def synthetic_piece(name: str, base_rot: np.ndarray, base_trans: np.ndarray, rate_hz: float, cycles: int,
                    freq_hz: float, amp_deg: float) -> Piece:  # fmt: skip
    """``cycles`` repetitions at ``freq_hz`` of a simple movement from the standing pose ``base_rot`` (52, 4).

    * ``twist``: the pelvis turns ±amp about the vertical; spine1-3 and both hips turn back, so the chest and the
      feet stay where they were;
    * ``leg_swing_left|right``: the hip swings forward and back by ±amp (about canonical x);
    * ``leg_side_left|right``: the hip swings out to the side by 0 -> amp -> 0 (about canonical z);
    * ``arm_forward_left|right`` / ``arm_side_left|right``: the shoulder raises the arm forward (about x) or out to
      the side (about z) by 0 -> amp -> 0;
    * ``turn_spot_left|right``: the whole body turns by amp about the vertical through the pelvis (smoothstep, from
      rest to rest), the feet turning with it rather than stepping.
    """
    n = int(round(cycles / freq_hz * rate_hz))
    t = np.arange(n) / rate_hz
    amp = np.deg2rad(amp_deg)
    sym = amp * np.sin(2 * np.pi * freq_hz * t)  # ±amp
    up = amp * 0.5 * (1.0 - np.cos(2 * np.pi * freq_hz * t))  # 0 -> amp -> 0
    rot = np.repeat(base_rot[None], n, axis=0)

    def turn(j, q):
        rot[:, j] = quat_mul(q, rot[:, j])

    kind, _, side = name.rpartition("_") if name != "twist" else ("twist", "", "")
    sgn = 1.0 if side == "left" else -1.0
    if name == "twist":
        up_world = _about(2, sym)  # the root orient is in the world: +Z is up
        rot[:, 0] = quat_mul(up_world, rot[:, 0])
        for j in (SPINE1, SPINE2, SPINE3):
            turn(j, _about(1, -sym / 3.0))
        for j in (L_HIP, R_HIP):
            turn(j, _about(1, -sym))
    elif kind == "leg_swing":
        turn(L_HIP if side == "left" else R_HIP, _about(0, -sym))
    elif kind == "leg_side":
        turn(L_HIP if side == "left" else R_HIP, _about(2, sgn * up))
    elif kind == "arm_forward":
        turn(L_SHOULDER if side == "left" else R_SHOULDER, _about(0, -up))
    elif kind == "arm_side":
        turn(L_SHOULDER if side == "left" else R_SHOULDER, _about(2, sgn * up))
    elif kind == "turn_spot":
        u = t / (cycles / freq_hz)
        rot[:, 0] = quat_mul(_about(2, sgn * amp * u * u * (3.0 - 2.0 * u)), rot[:, 0])  # a left turn is positive
    else:
        raise ValueError(f"unknown synthetic movement {name!r}")
    return Piece(name.rsplit("_", 1)[0] if side in ("left", "right") else name, f"synthetic:{name}",
                 quat_canonical(rot), np.repeat(base_trans[None], n, axis=0))  # fmt: skip


def _heading(root_q: np.ndarray) -> float:
    """The facing of root orientations (n, 4) or (4,): the circular mean of the forward (canonical +z) made
    horizontal."""
    f = Rotation.from_quat(np.atleast_2d(root_q)).apply([0.0, 0.0, 1.0])
    return float(np.arctan2(np.mean(f[:, 1]), np.mean(f[:, 0])))


def _align(piece: Piece, j0: np.ndarray, last_root: np.ndarray, last_pelvis: np.ndarray, window: int,
           last_travel: float | None = None) -> Piece:  # fmt: skip
    """Turn ``piece`` about the vertical through its first pelvis point and shift it so that its first root pose
    continues the previous last one: the facing averaged over ``window`` frames at each end (about one gait cycle,
    since the pelvis yaws by some 15° within a stride), and the horizontal position; its own height is kept.

    A piece with ``travel_turn_rad`` is turned instead so that it travels at ``last_travel`` plus that angle (a walk
    there and back stays on one line even when the clip walks off its own facing); the blend absorbs the facing."""
    if piece.travel_turn_rad is not None and last_travel is not None:
        alpha = last_travel + piece.travel_turn_rad - _travel(piece)
    else:
        alpha = _heading(last_root[-window:]) - _heading(piece.rot[:window, 0])
    z = Rotation.from_rotvec([0.0, 0.0, alpha])
    pelvis = piece.trans + j0
    p0 = pelvis[0]
    target = np.array([last_pelvis[0], last_pelvis[1], p0[2]])
    rot = piece.rot.copy()
    rot[:, 0] = (z * Rotation.from_quat(piece.rot[:, 0])).as_quat()
    new_pelvis = z.apply(pelvis - p0) + target
    return Piece(piece.kind, piece.source, rot, new_pelvis - j0, piece.travel_turn_rad)


def join(pieces: list[Piece], model: SmplhModel, betas, gender: str, rate_hz: float, blend_s: float,
         heading_window_s: float = 1.0, source: str = "program") -> JoinedMotion:  # fmt: skip
    """Join ``pieces`` (one subject's) into one sequence, with a ``blend_s`` transition between each two."""
    _, j_rest = rest_pose(model, betas)
    j0 = j_rest[0]
    n_blend = max(0, int(round(blend_s * rate_hz)))
    window = max(1, int(round(heading_window_s * rate_hz)))
    rots, trs, kinds, trans_flag, idx, info = [], [], [], [], [], []
    frame = 0
    last_travel = None  # the travel direction of the last piece that set one, or of the first facing
    for k, p in enumerate(pieces):
        if k == 0:
            last_travel = _heading(p.rot[:window, 0])
        else:
            p = _align(p, j0, rots[-1][:, 0], trs[-1][-1] + j0, window, last_travel)
            if p.travel_turn_rad is not None:
                last_travel = _travel(p)
            if n_blend:
                u = np.arange(1, n_blend + 1) / (n_blend + 1)
                w = u * u * (3.0 - 2.0 * u)
                shape = (n_blend, N_JOINTS, 4)
                a, b = np.broadcast_to(rots[-1][-1], shape), np.broadcast_to(p.rot[0], shape)
                rots.append(_slerp(a, b, np.broadcast_to(w[:, None], (n_blend, N_JOINTS))))
                trs.append(trs[-1][-1] + w[:, None] * (p.trans[0] - trs[-1][-1]))
                kinds += ["transition"] * n_blend
                trans_flag += [True] * n_blend
                idx += [-1] * n_blend
                frame += n_blend
        rots.append(p.rot)
        trs.append(p.trans)
        n = p.rot.shape[0]
        kinds += [p.kind] * n
        trans_flag += [False] * n
        idx += [k] * n
        info.append({"kind": p.kind, "source": p.source, "first": frame, "last": frame + n - 1})
        frame += n
    rot = np.concatenate(rots)
    seq = SmplhSequence(_to_poses(rot), np.concatenate(trs), np.asarray(betas, float), gender, float(rate_hz), source)
    return JoinedMotion(seq, np.array(kinds, dtype=object), np.array(trans_flag), np.array(idx), info)


def _joint_speeds(rot: np.ndarray, rate_hz: float) -> np.ndarray:
    """(n - 1, 22) angular speeds (rad/s) of the root and the body joints' local rotations."""
    r = rot[:, :BODY]
    d = np.abs(np.sum(r[1:] * r[:-1], axis=-1)).clip(0.0, 1.0)
    return 2.0 * np.arccos(d) * rate_hz


def join_check(m: JoinedMotion) -> dict:
    """The largest joint angular speed in the transitions against the P99 of the pieces' own speeds."""
    rot = _to_rot(m.seq.poses)
    sp = _joint_speeds(rot, m.seq.fps)  # step k is frames k -> k + 1
    trans = m.transition[1:] | m.transition[:-1]  # a step that touches a blend frame
    inside = (~trans) & (m.piece[1:] == m.piece[:-1])
    p99 = float(np.percentile(sp[inside], 99)) if inside.any() else float("nan")
    worst = float(sp[trans].max()) if trans.any() else 0.0
    return {"p99_rad_s": p99, "transition_max_rad_s": worst, "ok": bool(worst <= p99), "joins": len(m.pieces) - 1}


def subject_clips(subject: str, subset: str | None = None) -> pd.DataFrame:
    """One subject's files on disk from the split manifest (path, subset, subject, split, action)."""
    m = pd.read_csv(REPO_ROOT / "manifests" / "split_manifest.csv", dtype={"subject": str})
    m = m[m["subject"] == str(subject)]
    if subset is not None:
        m = m[m["subset"] == subset]
    m = m[[(amass_dir() / p).is_file() for p in m["path"]]].copy()
    m["action"] = m["path"].str.split("/").str[-1].str.replace(r"_?\d*_poses\.npz$", "", regex=True).str.lower()
    if m["split"].nunique() > 1:
        raise ValueError(f"subject {subject} spans splits {sorted(m['split'].unique())}")
    return m.reset_index(drop=True)


def _kind_of(action: str) -> str:
    for key, kind in (("turn", "turn"), ("walk", "walk"), ("squat", "squat"), ("kick", "kick"), ("wave", "arm"),
                      ("jump", "jump"), ("bend", "bend"), ("stairs", "stairs"), ("kneel", "kneel")):  # fmt: skip
        if key in action:
            return kind
    return "other"


def random_program(model: SmplhModel, subject: str, length_s: float, rng: np.random.Generator, rate_hz: float,
                   still_s=(5.0, 30.0), p_still: float = 0.15, p_tpose: float = 0.05,
                   subset: str | None = None) -> list[Piece]:  # fmt: skip
    """A program of ``length_s`` from one subject's clips, drawn with replacement, starting and ending in a T-pose,
    with a still stand (``p_still``) or a T-pose (``p_tpose``) inserted after a clip now and then."""
    clips = subject_clips(subject, subset)
    if clips.empty:
        raise ValueError(f"no clips on disk for subject {subject}")
    first = load_smplh_sequence(amass_dir() / clips["path"].iloc[0])
    betas = first.betas
    pieces = [tpose_piece(model, betas, 5.0, rate_hz)]
    total = pieces[0].rot.shape[0] / rate_hz
    while total < length_s - 5.0:
        row = clips.iloc[int(rng.integers(len(clips)))]
        p = clip_piece(row["path"], _kind_of(row["action"]), rate_hz)
        pieces.append(p)
        total += p.rot.shape[0] / rate_hz
        r = rng.random()
        if r < p_still:
            pieces.append(hold_piece(p.rot[-1], p.trans[-1], float(rng.uniform(*still_s)), rate_hz))
        elif r < p_still + p_tpose:
            pieces.append(tpose_piece(model, betas, 5.0, rate_hz))
        total += pieces[-1].rot.shape[0] / rate_hz if pieces[-1] is not p else 0.0
    pieces.append(tpose_piece(model, betas, 5.0, rate_hz))
    return pieces


PROTOCOL_SUBJECT = ("KIT", "3")
"""KIT subject 3 (training split, male, 100 Hz): the one subject whose clips cover most of the protocol."""


def protocol_program(model: SmplhModel, rate_hz: float) -> list[Piece]:
    """The script of Danyal's next OptiTrack session (Step 3 brief §4.8) as one program of KIT subject 3: his own
    clips for the walks, the turnarounds and the squats; synthetic pieces on his own poses for the rest (the
    T-poses, the hip twists, the leg swings, the arm raises, the 360° turns on the spot, the still stand)."""
    clips = subject_clips(PROTOCOL_SUBJECT[1], PROTOCOL_SUBJECT[0])
    path = {a: sorted(clips.loc[clips["action"] == a, "path"]) for a in ("walk_6m_straight_line", "turn_left",
                                                                         "squat")}  # fmt: skip
    if any(not v for v in path.values()):
        raise ValueError(f"KIT subject 3 lacks a protocol clip: {[k for k, v in path.items() if not v]}")
    betas = load_smplh_sequence(amass_dir() / path["squat"][0]).betas
    stand = clip_piece(path["squat"][0], "still", rate_hz, 0.0, 0.0)  # his standing pose before a squat
    s_rot, s_tr = stand.rot[0], stand.trans[0]
    walk = clip_piece(path["walk_6m_straight_line"][0], "walk", rate_hz)
    p = walk.trans + rest_pose(model, betas)[1][0]
    gone = np.linalg.norm(p[:, :2] - p[0, :2], axis=1)
    k4 = int(np.argmax(gone >= 4.0)) if gone.max() >= 4.0 else walk.rot.shape[0] - 1
    src = walk.source + f"@0-{k4 / rate_hz:.2f}s"
    out = Piece("walk", src, walk.rot[: k4 + 1], walk.trans[: k4 + 1], travel_turn_rad=0.0)  # along the facing
    back = Piece("walk", src, out.rot, out.trans, travel_turn_rad=np.pi)  # every later leg reverses the last
    turn_l = clip_piece(path["turn_left"][0], "turn", rate_hz)
    pieces = [tpose_piece(model, betas, 5.0, rate_hz), synthetic_piece("twist", s_rot, s_tr, rate_hz, 3, 1.2, 30.0)]
    for leg in range(6):  # forward and back, three times; his turn_left clips turn round in a few steps
        pieces += [out if leg == 0 else back, turn_l]
    for side in ("left", "right"):  # 360° each way, on the spot: his turn clips wander 1.2 m, so synthetic
        pieces.append(synthetic_piece(f"turn_spot_{side}", s_rot, s_tr, rate_hz, 1, 0.25, 360.0))
    for side in ("left", "right"):
        pieces.append(synthetic_piece(f"leg_swing_{side}", s_rot, s_tr, rate_hz, 5, 0.8, 30.0))
        pieces.append(synthetic_piece(f"leg_side_{side}", s_rot, s_tr, rate_hz, 5, 0.8, 30.0))
    for side in ("left", "right"):
        pieces.append(synthetic_piece(f"arm_forward_{side}", s_rot, s_tr, rate_hz, 3, 0.5, 90.0))
        pieces.append(synthetic_piece(f"arm_side_{side}", s_rot, s_tr, rate_hz, 3, 0.5, 90.0))
    for path_sq in (path["squat"] * 3)[:3]:
        pieces.append(clip_piece(path_sq, "squat", rate_hz))
    pieces += [hold_piece(s_rot, s_tr, 30.0, rate_hz), tpose_piece(model, betas, 5.0, rate_hz)]
    return pieces


def training_subjects(split: str = "train") -> list[tuple[str, str]]:
    """(subset, subject) pairs of ``split`` with at least one file on disk, in manifest order."""
    m = pd.read_csv(REPO_ROOT / "manifests" / "split_manifest.csv", dtype={"subject": str})
    m = m[m["split"] == split]
    out = []
    for (subset, subject), g in m.groupby(["subset", "subject"], sort=False):
        if any((amass_dir() / p).is_file() for p in g["path"]):
            out.append((subset, subject))
    return out


def make_program(cfgt, name: str, seed: int, subject: str | None = None, length_min: float | None = None):
    """``(joined, model, label)`` for program ``name``: ``protocol`` (KIT subject 3) or ``random`` (one subject's
    clips; the subject ``SUBSET/ID`` and the length are drawn from the seed unless given)."""
    from ..sim.rng import stream
    from .smplh import load_model

    sl = "twin.program."
    rate, blend, window = cfgt[sl + "rate_hz"], cfgt[sl + "blend_s"], cfgt[sl + "heading_window_s"]
    if name == "protocol":
        subset, subj = PROTOCOL_SUBJECT
        clips = subject_clips(subj, subset)
        first = load_smplh_sequence(amass_dir() / clips["path"].iloc[0])
        model = load_model(first.gender)
        pieces = protocol_program(model, rate)
        label = f"protocol:{subset}/{subj}"
    elif name == "random":
        rng = stream(seed, "twin.program")
        if subject is None:
            subs = training_subjects()
            subset, subj = subs[int(rng.integers(len(subs)))]
        else:
            subset, subj = subject.split("/", 1)
        lo, hi = cfgt[sl + "length_min"]
        minutes = float(rng.uniform(lo, hi)) if length_min is None else float(length_min)
        clips = subject_clips(subj, subset)
        first = load_smplh_sequence(amass_dir() / clips["path"].iloc[0])
        model = load_model(first.gender)
        pieces = random_program(model, subj, 60.0 * minutes, rng, rate, tuple(cfgt[sl + "still_s"]),
                                cfgt[sl + "p_still"], cfgt[sl + "p_tpose"], subset=subset)  # fmt: skip
        label = f"random:{subset}/{subj}:{minutes:.2f}min"
    else:
        raise ValueError(f"unknown program {name!r}: protocol or random")
    return join(pieces, model, first.betas, first.gender, rate, blend, window, label), model, label
