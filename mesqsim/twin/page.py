"""The page twin: the page's pose chain, smoothing, root and recorder, ported from mesquite-mocap ``87d7c3b``
(Step 3 brief §4.10). Every rule is kept, defects included; messages are handled in arrival order, as the page does.

Sources (line numbers at ``87d7c3b``): ``js/custom_icm.js`` (``handleWSMessage`` :500-972, ``calibrate`` :227-372,
``initGlobalLocalLast`` :95-223, ``applyMountingOffset`` :37-50, ``getTransformedQuaternion`` :439-476,
``getTransformedPosition`` :478-498), ``js/bvh_converter.js`` (``updateMotionData``, ``traverseHierarchy``,
``generateBVH`` :35-211), ``index.html`` (``slerpDict`` :138-157, the recorder :373-431) and ``js/threejsscene.js``
(the avatar ``glbs/scene2.glb``, :433-470). The arithmetic is three.js r157's (:mod:`mesqsim.twin.three_math`).

Per message ``{bone, x, y, z, w}`` (and ``px, py, pz`` from the phone, bone ``Hips``):

* world delta: raw · calibration⁻¹ (the third argument ``bc`` of ``multiplyQuaternions`` is ignored by three.js);
* tree map (:func:`tree_map`): reorder and sign the components, ``normalize``; then the tree's ``tposeOffset`` and
  the mounting offset, both multiplied on the right;
* a child's local target = (its parent pod's stored *target* global, normalised)⁻¹ · its own target, normalised
  (Spine's parent is Hips, or HipsAlt while Hips' stored global is still exactly the identity); the rig's bones in
  between (Spine1, Spine2, Neck, the shoulders) are ignored;
* ``bone.quaternion.slerp(target, slerpDict[bone] || 0.24)``, once per message;
* the Hips orientation comes from HipsAlt unless HipsAlt has been silent for over 1 s, then from the phone;
* the phone's position: ``posswap`` if the tree has one for Hips, x 100, minus ``initialPosition``, + 100 cm up,
  then a lerp of 0.1; a coordinate that is exactly 0 drops the message (a truthiness test). ``initialPosition`` is
  {0, 0, 0} until the first Set T-pose, then *the hips' own position object*, so the hips settle at
  x* = (100 p + (0, 100, 0)) / 2.

A tree without an entry for a bone (``_HipsAlt_DISABLED``) makes the page throw inside ``getTransformedQuaternion``;
the twin stops the message there too, after ``last`` was stored.

Mode ``ideal_root_box`` is a counterfactual, not the page: ``initialPosition`` is copied at the T-pose instead of
aliased, so 1 m is drawn as 1 m, and every pod reading (not the phone's) is first turned by a heading correction
(Step 0's box registration B1 in the brief) about its own world's +Z: q̃ = Y(δ) · q, so at t and at the T-pose alike.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from .three_math import (
    IDENTITY,
    euler_xyz,
    js_number,
    q_conj,
    q_from_axis_angle,
    q_mul,
    q_normalize,
    q_slerp,
    v_lerp,
)

__all__ = ["RIG_PREFIX", "POD_BONES", "PARENT", "JOINT_NAMES", "Rig", "load_rig", "load_tree", "load_slerp",
           "tree_map", "PageTwin", "PAGE_EPOCH_MS"]

RIG_PREFIX = "mm"
DEFAULT_SLERP = 0.24  # custom_icm.js:578
POSITION_SENSITIVITY = 100.0  # custom_icm.js:10
ROOT_LERP = 0.1  # custom_icm.js:956
HIPSALT_TIMEOUT_MS = 1000.0  # custom_icm.js:582
PAGE_EPOCH_MS = 1794703756277.0  # wall clock of the twin's t = 0 (2026-11-15T00:49:16.277Z, Nov 14's take start)

POD_BONES = ("Hips", "HipsAlt", "Spine", "Head", "LeftArm", "LeftForeArm", "RightArm", "RightForeArm", "LeftUpLeg",
             "LeftLeg", "RightUpLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftHand", "RightHand")
"""``initGlobalLocalLast``'s bones, in its order (which is also ``Object.keys(mac2Bones)`` in ``calibrate``)."""
TPOSE_OFFSET_BONES = ("Hips", "HipsAlt", "Spine", "Head", "LeftArm", "LeftForeArm", "LeftHand", "RightArm",
                      "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg",
                      "RightFoot")
PARENT = {"LeftUpLeg": "Hips", "RightUpLeg": "Hips", "Spine": "Hips", "Head": "Spine", "LeftArm": "Spine",
          "RightArm": "Spine", "LeftForeArm": "LeftArm", "LeftHand": "LeftForeArm", "RightForeArm": "RightArm",
          "RightHand": "RightForeArm", "LeftLeg": "LeftUpLeg", "RightLeg": "RightUpLeg", "LeftFoot": "LeftLeg",
          "RightFoot": "RightLeg"}
"""The pod whose stored global each branch of ``handleWSMessage`` takes as the parent."""
NO_SET_GLOBAL = ("LeftFoot", "RightFoot")  # their setGlobal is commented out (:891, :908)
JOINT_NAMES = ("Hips", "LeftUpLeg", "RightUpLeg", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder",
               "LeftArm", "LeftForeArm", "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
               "LeftLeg", "LeftFoot", "LeftToeBase", "RightLeg", "RightFoot", "RightToeBase")
"""The joints the recorder writes (``bvh_converter.js:52-75``)."""
_RAD_TO_DEG = 180 / math.pi


# ---------------------------------------------------------------------------------------------- the avatar

@dataclass
class Rig:
    """The avatar's node tree as three.js builds it from ``glbs/scene2.glb``: names, children (in order), rest
    translations (cm) and rotations, and which nodes are bones (joints of a skin)."""

    names: list[str | None]
    children: list[list[int]]
    translation: list[tuple[float, float, float]]
    rotation: list[tuple[float, float, float, float]]
    is_bone: list[bool]
    roots: list[int]
    _first: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        order = []
        stack = list(reversed(self.roots))
        while stack:  # depth-first, pre-order: Object3D.getObjectByName
            i = stack.pop()
            order.append(i)
            stack.extend(reversed(self.children[i]))
        for i in order:
            if self.names[i] is not None and self.names[i] not in self._first:
                self._first[self.names[i]] = i
        self.parent = {c: i for i, ch in enumerate(self.children) for c in ch}

    def find(self, name: str) -> int | None:
        """``model.getObjectByName(name)``: the first node of that name, depth first."""
        return self._first.get(name)

    def to_json(self) -> dict:
        return {"names": self.names, "children": self.children, "translation": self.translation,
                "rotation": self.rotation, "is_bone": self.is_bone, "roots": self.roots}

    @classmethod
    def from_json(cls, d: dict) -> Rig:
        return cls(d["names"], d["children"], [tuple(t) for t in d["translation"]],
                   [tuple(r) for r in d["rotation"]], d["is_bone"], d["roots"])


def load_rig(glb_path) -> Rig:
    """Read the node tree of a binary glTF file (its JSON chunk) as three.js r157's GLTFLoader builds it.

    The loader gives each node a unique name, reserving names depth first from the scene's roots
    (``GLTFParser._loadNodeShallow``, ``createUniqueName``): a second ``mmHips`` becomes ``mmHips_1``. The avatar
    has such duplicates (a zero-offset copy of each bone, hung under the bone); the page's bones are the first ones.
    """
    b = Path(glb_path).read_bytes()
    n = struct.unpack_from("<I", b, 12)[0]
    g = json.loads(b[20:20 + n])
    nodes = g["nodes"]
    joints = {j for s in g.get("skins", []) for j in s["joints"]}
    roots = list(g["scenes"][g.get("scene", 0)]["nodes"])
    names: list[str | None] = [nd.get("name") for nd in nodes]
    used: dict[str, int] = {}
    stack = list(reversed(roots))
    while stack:
        i = stack.pop()
        if names[i]:
            s = re.sub(r"[\[\].:/]", "", re.sub(r"\s", "_", names[i]))  # PropertyBinding.sanitizeNodeName
            if s in used:
                used[s] += 1
                names[i] = f"{s}_{used[s]}"
            else:
                used[s] = 0
                names[i] = s
        stack.extend(reversed(nodes[i].get("children", [])))
    return Rig(names, [list(nd.get("children", [])) for nd in nodes],
               [tuple(float(v) for v in nd.get("translation", (0.0, 0.0, 0.0))) for nd in nodes],
               [tuple(float(v) for v in nd.get("rotation", IDENTITY)) for nd in nodes],
               [i in joints for i in range(len(nodes))], roots)


def load_tree(tree_id: str, mocap_dir) -> dict:
    return json.loads((Path(mocap_dir) / "trees" / tree_id / "meta.json").read_text(encoding="utf-8"))


def load_slerp(mocap_dir) -> dict[str, float]:
    """``var slerpDict = {...}`` parsed from the page's ``index.html``."""
    text = (Path(mocap_dir) / "index.html").read_text(encoding="utf-8")
    block = re.search(r"var slerpDict = \{(.*?)\};", text, re.S).group(1)
    return {k: float(v) for k, v in re.findall(r'"(\w+)"\s*:\s*([0-9.]+)', block)}


# ---------------------------------------------------------------------------------------------- the page

class PageThrew(Exception):
    """The page would have thrown inside ``handleWSMessage`` (it then stops handling that message)."""


def tree_map(q, mapping: dict) -> tuple[float, float, float, float]:
    """``getTransformedQuaternion``: (q[order[0]]·sign[0], q[order[1]]·sign[1], q[order[2]]·sign[2], w), normalised."""
    order = mapping["axis"]["order"].lower()
    sign = mapping["axis"]["sign"]
    comp = {"x": q[0], "y": q[1], "z": q[2]}
    return q_normalize(tuple(comp[order[k]] * (1 if sign[k] == "+" else -1) for k in range(3)) + (q[3],))


def _truthy(v) -> bool:
    """JavaScript truthiness of a message field (undefined, 0 and NaN are false)."""
    return v is not None and v != 0 and not (isinstance(v, float) and math.isnan(v))


def _js_acos(x: float) -> float:
    """``Math.acos``: NaN outside [-1, 1] (rounding can put |w| a hair above 1), where Python would raise."""
    return math.acos(x) if -1.0 <= x <= 1.0 else float("nan")


def _wall_ms(t_ms: float) -> int:
    """``new Date().getTime()`` at twin time ``t_ms``: a Date holds whole milliseconds (TimeClip truncates)."""
    return int(PAGE_EPOCH_MS + t_ms)


class PageTwin:
    """One page session: feed messages in arrival order, set the T-pose, record a BVH.

    ``tree`` is a parsed ``trees/<id>/meta.json``; ``slerp`` the page's ``slerpDict``; times are milliseconds on
    the twin's clock (the page's wall clock is ``PAGE_EPOCH_MS`` + t).
    """

    def __init__(self, rig: Rig, tree: dict, slerp: dict[str, float], mode: str = "as_is",
                 heading_correction=None):
        if mode not in ("as_is", "ideal_root_box"):
            raise ValueError(f"page mode {mode!r}")
        self.rig, self.tree, self.slerp, self.mode = rig, tree, dict(slerp), mode
        self.heading_correction = heading_correction  # ideal_root_box: callable (bone, t_ms) -> radians
        self.quat = {i: tuple(rig.rotation[i]) for i in range(len(rig.names))}  # drawn local rotation per node
        self.pos = {i: tuple(rig.translation[i]) for i in range(len(rig.names))}  # local position per node
        self.hips = rig.find(RIG_PREFIX + "Hips")
        self.mac2bones: dict[str, dict] = {}
        self.tpose_offsets: dict[str, tuple] = {}
        self.mounting_offsets: dict[str, tuple] = {}
        self.mounting_done = False
        self.flag = True
        self.hips_alt_last = 0.0  # wall ms, as `hipsAltLast = 0` (custom_icm.js:5)
        self.hips_last = 0.0
        self.initial_position = (0.0, 0.0, 0.0)  # custom_icm.js:9
        self.initial_is_alias = False
        self.calibrated = False
        self.recording = False
        self.recorded: list[list[dict]] = []
        self.first_frame: list[dict] | None = None
        self.timing: dict | None = None

    # -- helpers -------------------------------------------------------------------------------------------

    def bone_quaternion(self, name: str) -> tuple[float, float, float, float]:
        """The drawn local rotation of the rig bone ``mm<name>``."""
        return self.quat[self.rig.find(RIG_PREFIX + name)]

    def hips_position(self) -> tuple[float, float, float]:
        return self.pos[self.hips]

    def _world_quat(self, i: int) -> tuple[float, float, float, float]:
        chain = []
        while i is not None:
            chain.append(i)
            i = self.rig.parent.get(i)
        q = IDENTITY
        for j in reversed(chain):
            q = q_mul(q, self.quat[j])
        return q

    def _init(self):
        """``initGlobalLocalLast``: the tree's T-pose offsets, and every pod's stored rotations from the rig."""
        self.flag = False
        for name in TPOSE_OFFSET_BONES:
            cfg = self.tree.get(name)
            comp = IDENTITY
            if cfg and cfg.get("tposeOffset"):
                for rot in cfg["tposeOffset"]:
                    axis = {"X": (1.0, 0.0, 0.0), "Y": (0.0, 1.0, 0.0)}.get(rot["axis"], (0.0, 0.0, 1.0))
                    comp = q_mul(comp, q_from_axis_angle(axis, rot["degrees"] * math.pi / 180))
            self.tpose_offsets[name] = comp
        for name in POD_BONES:
            q = self.quat[self.rig.find(RIG_PREFIX + name.replace("Alt", "", 1))]
            self.mac2bones[name] = {"calibration": IDENTITY, "last": IDENTITY, "global": q, "local": q,
                                    "initial": q}

    def _mount(self, q, bone: str):
        """``applyMountingOffset``: the tree's T-pose offset, then the mounting offset, on the right."""
        if bone in self.tpose_offsets:
            q = q_mul(q, self.tpose_offsets[bone])
        if bone in self.mounting_offsets:
            q = q_mul(q, self.mounting_offsets[bone])
        return q

    def _map(self, q, bone: str):
        if bone not in self.tree:
            raise PageThrew(f"tree has no entry for {bone}")
        return tree_map(q, self.tree[bone])

    def _slerp_node(self, node: int, target, bone: str):
        self.quat[node] = q_slerp(self.quat[node], target, self.slerp.get(bone) or DEFAULT_SLERP)

    # -- the page ------------------------------------------------------------------------------------------

    def handle(self, msg: dict, t_ms: float) -> None:
        """``handleWSMessage(msg)`` at time ``t_ms``."""
        try:
            self._handle(msg, _wall_ms(t_ms), t_ms)
        except PageThrew:
            pass

    def _handle(self, msg: dict, now: float, t_ms: float) -> None:
        if self.flag:
            self._init()
        bone = msg.get("bone")
        if not bone:
            return
        node = self.rig.find(RIG_PREFIX + bone.replace("Alt", "", 1))
        if node is None:
            return
        raw = (float(msg["x"]), float(msg["y"]), float(msg["z"]), float(msg["w"]))
        if bone != "Hips" and self.heading_correction is not None:  # any mode: the page-level test (§8.3)
            d = self.heading_correction(bone, t_ms)
            raw = q_mul((0.0, 0.0, math.sin(d / 2), math.cos(d / 2)), raw)
        if bone not in self.mac2bones:
            raise PageThrew(f"no mac2Bones entry for {bone}")
        mb = self.mac2bones[bone]
        mb["last"] = raw
        transformed = q_mul(raw, q_conj(mb.get("calibration", IDENTITY)))
        if bone == "Hips" and self.hips_alt_last + HIPSALT_TIMEOUT_MS < now:
            alt = self._mount(q_normalize(self._map(transformed, bone)), bone)
            hip = self._mount(q_normalize(self._map(transformed, bone)), bone)
            self._slerp_node(node, hip, bone)
            if self.hips_alt_last + HIPSALT_TIMEOUT_MS < now:
                self.mac2bones["Hips"]["local"] = alt
                self.mac2bones[bone]["global"] = hip
            self.hips_last = now
        if bone == "HipsAlt":
            hip = self._mount(q_normalize(self._map(transformed, bone)), bone)
            self._slerp_node(node, hip, bone)
            self.mac2bones["Hips"]["local"] = hip
            self.mac2bones["Hips"]["global"] = hip
            self.hips_alt_last = now
        if bone in PARENT:
            q = self._mount(self._map(transformed, bone), bone)
            parent = PARENT[bone]
            if bone == "Spine":
                g = self.mac2bones["Hips"]["global"]
                if g == (0, 0, 0, 1):
                    parent = "HipsAlt"
            pq = q_normalize(self.mac2bones[parent]["global"])
            corr = q_normalize(q_mul(q_conj(pq), q))
            self._slerp_node(node, corr, bone)
            mb["local"] = corr
            if bone not in NO_SET_GLOBAL:
                mb["global"] = q
        if bone == "Hips" and _truthy(msg.get("px")) and _truthy(msg.get("py")) and _truthy(msg.get("pz")):
            p = (msg["px"], msg["py"], msg["pz"])
            swap = self.tree.get("Hips", {}).get("posswap")
            if swap:
                comp = {"x": p[0], "y": p[1], "z": p[2]}
                order, sign = swap["order"].lower(), swap["sign"]
                p = tuple(comp[order[k]] * (1 if sign[k] == "+" else -1) for k in range(3))
            init = self.pos[self.hips] if self.initial_is_alias else self.initial_position
            target = (p[0] * POSITION_SENSITIVITY - init[0], p[1] * POSITION_SENSITIVITY - init[1] + 100,
                      p[2] * POSITION_SENSITIVITY - init[2])
            self.pos[self.hips] = v_lerp(self.pos[self.hips], target, ROOT_LERP)

    def calibrate(self) -> None:
        """``calibrate()``: the T-pose snapshot of every pod, the mounting offsets (first time only) and the root."""
        if self.mode == "as_is":
            self.initial_is_alias = True  # initialPosition = model.getObjectByName("mmHips").position
        else:
            self.initial_position = tuple(self.pos[self.hips])
        for key in list(self.mac2bones):
            mb = self.mac2bones[key]
            if not mb.get("last"):
                continue
            mb["calibration"] = tuple(mb["last"])
            if self.mounting_done:
                continue
            node = self.rig.find(RIG_PREFIX + key.replace("Alt", "", 1))
            if node is None or key not in self.tree:
                continue
            expected = q_normalize(self._world_quat(node))
            t = tree_map(q_normalize(mb["last"]), self.tree[key])
            if key in self.tpose_offsets:
                t = q_mul(t, self.tpose_offsets[key])
            normal = q_mul(expected, q_conj(t))
            if any(s in key for s in ("Arm", "Hand", "Leg", "Foot")):
                axis = (0.0, 1.0, 0.0)
            elif key in ("Head", "Spine") or "Hips" in key:
                axis = (0.0, 0.0, 1.0)
            else:
                axis = None
            offset = normal
            if axis is not None:
                flipped = q_mul(expected, q_conj(q_mul(t, q_from_axis_angle(axis, math.pi))))
                if 2 * _js_acos(abs(flipped[3])) < 2 * _js_acos(abs(normal[3])):  # NaN compares false, as in JS
                    offset = flipped
            self.mounting_offsets[key] = offset
        self.mounting_done = True
        self.calibrated = True

    # -- the recorder (index.html:373-431, bvh_converter.js) -----------------------------------------------

    def _motion(self) -> list[dict]:
        """``updateMotionData``: every recorded joint's local position and Euler XYZ rotation in degrees."""
        info: list[dict] = []

        def visit(i: int, level: int):
            name = (self.rig.names[i] or "").replace("mm", "", 1)
            exists = any(j["name"] == name for j in info)
            if name in JOINT_NAMES and not exists:
                e = euler_xyz(self.quat[i])
                info.append({"name": name, "position": list(self.pos[i]),
                             "rotation": [e[0] * _RAD_TO_DEG, e[1] * _RAD_TO_DEG, e[2] * _RAD_TO_DEG],
                             "level": level})
            for c in self.rig.children[i]:
                if self.rig.is_bone[c]:
                    visit(c, level + 1 if not exists else level)

        visit(self.hips, 0)
        if self.recording:
            self.recorded.append(info)
        return info

    def record_start(self, t_ms: float) -> None:
        self.recording = True
        self.recorded = []
        self.timing = {"frames": 0, "startMs": t_ms, "endMs": t_ms, "startISO": _iso(t_ms), "endISO": None}
        self.first_frame = self._motion()

    def record_tick(self, t_ms: float) -> None:
        """One ``setInterval(..., 1000/30)`` tick of the recorder."""
        if self.recording:
            self._motion()
            self.timing["frames"] += 1
            self.timing["endMs"] = t_ms

    def record_stop(self, t_ms: float) -> str:
        """Stop recording and return the BVH text ``generateBVH`` writes."""
        self.recording = False
        self.timing["endISO"] = _iso(t_ms)
        self.timing["frames"] = len(self.recorded)
        return generate_bvh(self.first_frame, self.recorded, self.timing)


def _iso(t_ms: float) -> str:
    """``new Date().toISOString()`` at twin time ``t_ms`` (whole milliseconds)."""
    ms = _wall_ms(t_ms)
    t = dt.datetime.fromtimestamp(ms // 1000, tz=dt.UTC)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


def generate_bvh(joint_info: list[dict], motion: list[list[dict]], timing: dict | None) -> str:
    """``generateBVH`` (``bvh_converter.js:113-211``), numbers printed as JavaScript prints them."""
    out = ["HIERARCHY\n"]
    for index, joint in enumerate(joint_info):
        ind = "  " * joint["level"]
        out.append(f"{ind}{'ROOT' if joint['level'] == 0 else 'JOINT'} {joint['name']}\n")
        out.append(f"{ind}{{\n")
        out.append(f"{ind}  OFFSET {' '.join(js_number(v) for v in joint['position'])}\n")
        if joint["name"] == "Hips":
            out.append(f"{ind}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation\n")
        else:
            out.append(f"{ind}  CHANNELS 3 Xrotation Yrotation Zrotation\n")
        if index == len(joint_info) - 1 or joint_info[index + 1]["level"] <= joint["level"]:
            out.append(f"{ind}  End Site\n{ind}  {{\n{ind}    OFFSET 0 0 0\n{ind}  }}\n")
        if index < len(joint_info) - 1:
            nxt = joint_info[index + 1]
            if nxt["level"] <= joint["level"]:
                for i in range(joint["level"] - nxt["level"] + 1):
                    out.append(f"{'  ' * (joint['level'] - i)}}}\n")
        if index == len(joint_info) - 1:
            for i in range(joint["level"], -1, -1):
                out.append(f"{'  ' * i}}}\n")
    measured = (timing is not None and timing.get("frames", 0) > 1
                and timing["endMs"] > timing["startMs"])
    frame_time = ((timing["endMs"] - timing["startMs"]) / 1000) / (timing["frames"] - 1) if measured else 1 / 30
    out.append("MOTION\n")
    out.append(f"Frames: {len(motion)}\n")
    out.append(f"Frame Time: {js_number(frame_time)}\n")
    if timing is not None and timing.get("startISO"):
        out.append(f"; MESQ_CAPTURE_START {timing['startISO']}\n")
        out.append(f"; MESQ_CAPTURE_END   {timing.get('endISO') or ''}\n")
    out.append(f"; MESQ_FRAME_TIME_SOURCE {'measured' if measured else 'ASSUMED_1_30_UNTRUSTWORTHY'}\n")
    for frame in motion:
        line = []
        for joint in frame:
            if joint["name"] == "Hips":
                line.append(" ".join(js_number(v) for v in joint["position"]) + " "
                            + " ".join(js_number(v) for v in joint["rotation"]) + " ")
            else:
                line.append(" ".join(js_number(v) for v in joint["rotation"]) + " ")
        out.append("".join(line) + "\n")
    return "".join(out)
