"""Read the page's BVH files and compute their forward kinematics (Step 3 brief §5.1).

A BVH file has a HIERARCHY (joints; each one's OFFSET from its parent, in the parent's axes; its CHANNELS) and a
MOTION section (``Frames``, ``Frame Time``, then one line of numbers per frame). The page's recorder adds comment
lines that start with ``;`` after the frame time: ``MESQ_CAPTURE_START`` and ``MESQ_CAPTURE_END`` in UTC and
``MESQ_FRAME_TIME_SOURCE`` (mesquite-mocap ``87d7c3b``, ``js/bvh_converter.js:155-197``). The older page of
2026-11-13 wrote none and asserted a frame time of exactly 1/30 s.

Forward kinematics runs in the file's own frame (three.js, +Y up, centimetres). :func:`bvh_sites` and
:func:`bvh_facing` convert at the boundary, through :mod:`mesqsim.twin.page_frame`, to the repository's world
(+Z up) in metres.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..conventions import quat_identity, quat_mul, quat_rotate
from ..twin.page_frame import bvh_euler_to_quat, three_to_world_rot, three_to_world_vec

__all__ = [
    "POD_BONES",
    "SITES",
    "BvhJoint",
    "Bvh",
    "read_bvh",
    "forward_kinematics",
    "pod_bones",
    "bvh_sites",
    "bvh_facing",
]

POD_BONES = ("Hips", "Spine", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand",
             "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg", "LeftFoot", "RightFoot")
"""The BVH joints a pod can drive (the page's bone names). Hips is driven by HipsAlt, or by the phone without it."""

SITES = {
    # site (named after the pod's bone): (joint, second joint for a midpoint, or "up" for 10 cm up the bone)
    "Hips": ("Hips", None),             # the pelvis, at Hips
    "Spine": ("Spine2", None),          # the chest pod, at Spine2
    "Head": ("Head", "up"),             # the head pod, 10 cm up the head
    "LeftArm": ("LeftArm", "LeftForeArm"),          # mid upper arm
    "RightArm": ("RightArm", "RightForeArm"),
    "LeftForeArm": ("LeftForeArm", "LeftHand"),     # mid forearm
    "RightForeArm": ("RightForeArm", "RightHand"),
    "LeftUpLeg": ("LeftUpLeg", "LeftLeg"),          # mid thigh
    "RightUpLeg": ("RightUpLeg", "RightLeg"),
    "LeftLeg": ("LeftLeg", "LeftFoot"),             # mid shin
    "RightLeg": ("RightLeg", "RightFoot"),
}
"""Where each pod's marker sits on the drawn skeleton (Step 3 brief §5.1)."""

HEAD_UP_CM = 10.0
"""The head pod sits 10 cm up the head, along the Head bone's own +Y (three.js bones point along +Y; the page
writes End Site offsets as 0 0 0, so the end site cannot give the direction)."""


@dataclass(frozen=True)
class BvhJoint:
    name: str
    parent: int  # index of the parent joint; -1 for the root
    offset: np.ndarray  # (3,) from the parent, in the parent's axes, file units (cm)
    channels: tuple[str, ...]
    first_channel: int  # column of this joint's first channel in the motion array
    end_site: np.ndarray | None = None


@dataclass
class Bvh:
    path: str
    joints: list[BvhJoint]
    frame_time_s: float
    frame_time_source: str  # "measured" when the file says so, otherwise "asserted"
    capture_start_utc: dt.datetime | None
    capture_end_utc: dt.datetime | None
    comments: dict[str, str]  # the ";" lines, KEY -> value
    motion: np.ndarray  # (F, C), one row per frame

    @property
    def n_frames(self) -> int:
        return int(self.motion.shape[0])

    @property
    def names(self) -> list[str]:
        return [j.name for j in self.joints]

    @property
    def duration_s(self) -> float:
        return self.n_frames * self.frame_time_s

    @property
    def t_s(self) -> np.ndarray:
        """Frame times from the first frame, from the file's frame time."""
        return np.arange(self.n_frames) * self.frame_time_s

    def index(self, name: str) -> int:
        return self.names.index(name)

    def rotation_channels(self, name: str) -> np.ndarray:
        """``(F, 3)`` rotation channels of one joint, in degrees, in the joint's channel order."""
        j = self.joints[self.index(name)]
        cols = [j.first_channel + k for k, c in enumerate(j.channels) if c.lower().endswith("rotation")]
        return self.motion[:, cols]


def _parse_utc(text: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def read_bvh(path) -> Bvh:
    """Parse a BVH file. Raises ``ValueError`` when the motion does not match the hierarchy."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    spec: list[dict] = []  # one dict per joint, in file order (parents before children)
    stack: list[tuple[str, int]] = []  # open blocks: ("joint", index) or ("end", index of the owning joint)
    opening: tuple[str, object] | None = None  # what the next "{" opens
    n_channels = 0
    i = 0
    while i < len(lines):
        tok = lines[i].split()
        i += 1
        if not tok or tok[0] == "HIERARCHY":
            continue
        head = tok[0]
        if head in ("ROOT", "JOINT"):
            opening = ("joint", tok[1])
        elif head == "End":  # "End Site"
            opening = ("end", None)
        elif head == "{":
            if opening is None:
                raise ValueError(f"{path.name}:{i}: '{{' without ROOT, JOINT or End Site")
            owner = next((k for kind, k in reversed(stack) if kind == "joint"), -1)
            if opening[0] == "joint":
                spec.append({"name": opening[1], "parent": owner, "offset": np.zeros(3), "channels": (),
                             "first": n_channels, "end": None})
                stack.append(("joint", len(spec) - 1))
            else:
                stack.append(("end", owner))
            opening = None
        elif head == "}":
            stack.pop()
        elif head == "OFFSET":
            kind, k = stack[-1]
            spec[k]["end" if kind == "end" else "offset"] = np.array([float(v) for v in tok[1:4]])
        elif head == "CHANNELS":
            kind, k = stack[-1]
            n = int(tok[1])
            spec[k]["channels"], spec[k]["first"] = tuple(tok[2:2 + n]), n_channels
            n_channels += n
        elif head == "MOTION":
            break
    joints = [BvhJoint(s["name"], s["parent"], s["offset"], s["channels"], s["first"], s["end"]) for s in spec]
    frames = frame_time = None
    comments: dict[str, str] = {}
    data: list[str] = []
    for line in lines[i:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith(";"):
            parts = s.lstrip(";").split(None, 1)
            if parts:
                comments[parts[0]] = parts[1].strip() if len(parts) > 1 else ""
        elif s.startswith("Frames:"):
            frames = int(s.split(":", 1)[1])
        elif s.startswith("Frame Time:"):
            frame_time = float(s.split(":", 1)[1])
        else:
            data.append(s)
    if frames is None or frame_time is None:
        raise ValueError(f"{path.name}: no 'Frames:' or 'Frame Time:' line")
    values = np.array(" ".join(data).split(), dtype=np.float64)
    if values.size != frames * n_channels:
        raise ValueError(f"{path.name}: {values.size} numbers, expected {frames} frames x {n_channels} channels")
    # the page writes "measured", or "ASSUMED_1_30_UNTRUSTWORTHY" (js/bvh_converter.js:197); older pages nothing
    source = "measured" if comments.get("MESQ_FRAME_TIME_SOURCE", "").lower() == "measured" else "asserted"
    return Bvh(str(path), joints, frame_time, source, _parse_utc(comments.get("MESQ_CAPTURE_START", "")),
               _parse_utc(comments.get("MESQ_CAPTURE_END", "")), comments, values.reshape(frames, n_channels))


def forward_kinematics(bvh: Bvh) -> tuple[np.ndarray, np.ndarray]:
    """Global joint positions ``(F, J, 3)`` (file units) and orientations ``(F, J, 4)`` in the file's frame.

    A joint's local rotation is the product of its rotation channels in channel order; its local translation is its
    OFFSET, or its position channels when it has them (the root: three.js's BVHLoader replaces the rest position by
    the channel values). Global: R_j = R_parent · R_local, p_j = p_parent + R_parent · t_local.
    """
    f = bvh.n_frames
    pos = np.empty((f, len(bvh.joints), 3))
    rot = np.empty((f, len(bvh.joints), 4))
    for j, jt in enumerate(bvh.joints):
        cols = {c.lower(): jt.first_channel + k for k, c in enumerate(jt.channels)}
        rch = [c for c in jt.channels if c.lower().endswith("rotation")]
        if rch:
            angles = bvh.motion[:, [cols[c.lower()] for c in rch]]
            local = bvh_euler_to_quat(angles, "".join(c[0] for c in rch))
        else:
            local = quat_identity((f,))
        if all(f"{a}position" in cols for a in "xyz"):
            t_local = bvh.motion[:, [cols[f"{a}position"] for a in "xyz"]]
        else:
            t_local = np.broadcast_to(jt.offset, (f, 3))
        if jt.parent < 0:
            rot[:, j] = local
            pos[:, j] = t_local
        else:
            rot[:, j] = quat_mul(rot[:, jt.parent], local)
            pos[:, j] = pos[:, jt.parent] + quat_rotate(rot[:, jt.parent], t_local)
    return pos, rot


def pod_bones(bvh: Bvh) -> list[str]:
    """The pod-carrying bones of this recording: those of :data:`POD_BONES` whose rotation channels are not all 0."""
    return [b for b in POD_BONES if b in bvh.names and np.any(bvh.rotation_channels(b) != 0.0)]


def bvh_sites(bvh: Bvh, fk: tuple[np.ndarray, np.ndarray] | None = None) -> dict[str, np.ndarray]:
    """Each site of :data:`SITES` as ``(F, 3)`` world positions in metres (+Z up)."""
    pos, rot = fk if fk is not None else forward_kinematics(bvh)
    out = {}
    for site, (a, b) in SITES.items():
        if a not in bvh.names or (b not in (None, "up") and b not in bvh.names):
            continue
        p = pos[:, bvh.index(a)]
        if b == "up":
            p = p + quat_rotate(rot[:, bvh.index(a)], np.array([0.0, HEAD_UP_CM, 0.0]))
        elif b is not None:
            p = 0.5 * (p + pos[:, bvh.index(b)])
        out[site] = three_to_world_vec(p) / 100.0
    return out


def bvh_facing(bvh: Bvh, fk: tuple[np.ndarray, np.ndarray] | None = None) -> np.ndarray:
    """``(F, 2)`` unit horizontal world vectors: the root's image of the avatar's forward (+Z in three.js)."""
    _, rot = fk if fk is not None else forward_kinematics(bvh)
    fwd = quat_rotate(three_to_world_rot(rot[:, 0]), np.array([0.0, 0.0, 1.0]))[:, :2]
    n = np.linalg.norm(fwd, axis=1, keepdims=True)
    return np.where(n > 1e-9, fwd / np.where(n > 1e-9, n, 1.0), np.nan)
