"""The phone app, ported line by line from its Swift core (Step 3 brief §4.7).

Source: ``new_device_code/phone_app/MesquiteHips/Sources/MesquiteHips/Core`` at mesquite-mocap ``87d7c3b``
(``Geometry.swift``, ``HipsFrame.swift``, ``HubMessage.swift``, ``PosePipeline.swift``, ``LeverArmFit.swift``,
``Settings.swift``), with ``MATH.md`` §3-§8 as its explanation. Where the two disagree the Swift code wins, as the
brief says. Every operation keeps Swift's order on IEEE doubles, so the port reproduces the app bit for bit where
libm agrees; ``tools/app_golden`` compiles the unmodified Swift files and the golden test compares.

Everything here lives in ARKit's frames: +Y up, metres, quaternions (x, y, z, w), Hamilton products. The app:

1. **Zero** (``HipsFrame.setZero``): the camera's view direction f = q_C (0, 0, -1) q_C⁻¹ must keep at least 0.3 of
   horizontal length; ψ0 = remainder(heading(f) + β, 2π), β from the camera-facing setting; A = Y(-ψ0). Automatic on
   the first frame with normal tracking, and again whenever requested (``PosePipeline.process``).
2. **Report** (``HipsFrame.report``): h = p_C + q_C l (lever l in the camera frame); p_R = A (h - h_0) + (0, H, 0);
   q_R = A q_C q_C0⁻¹ A⁻¹, normalised. The orientation does not depend on how the phone is worn.
3. **Sign**: q_R is negated when its dot product with the previous one is negative.
4. **Tree turn** (``PageTree.wire``): velvet family turns the position by Y(180°); thorny turns both; matched none.
5. **Message** (``HubMessage.encode``): the first precision level that fits 95 bytes, (q, p) decimals (4, 3),
   (3, 3), (3, 2), (4, none); numbers by integer arithmetic, rounded half away from zero (Swift's ``rounded()``),
   trailing zeros dropped, never "-0", and a position that would round to 0 sent as one step of its sign.
6. **Policy and rate**: normal tracking sends both, limited by excessive motion or insufficient features sends the
   orientation only, anything else nothing; a frame is due once 1/rate - 1/240 s has passed since the last *sent*
   one (``isDue``, ``markSent``).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

__all__ = ["Vec3", "Quat", "CameraFacing", "ZeroState", "ZeroError", "HipsFrame", "PAGE_TREES", "tree_wire",
           "HubMessage", "MAX_BYTES", "LEVELS", "fixed", "Tracking", "FrameInput", "FrameOutput", "PosePipeline",
           "LeverArmFit", "LeverFitResult", "LeverFitFailure", "swift_rounded"]

Vec3 = tuple  # (x, y, z)
Quat = tuple  # (x, y, z, w)
IDENTITY: Quat = (0.0, 0.0, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- Geometry.swift
def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(s, a):
    return (s * a[0], s * a[1], s * a[2])


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def v_length(a):
    return math.sqrt(v_dot(a, a))


def v_finite(a):
    return all(math.isfinite(c) for c in a)


def v_heading(a, min_horizontal: float = 1e-9) -> float | None:
    """``Vec3.heading``: atan2(-x, -z), 0 for -Z, +π/2 for -X; None when the horizontal part is shorter than given."""
    h = (a[0], 0.0, a[2])
    if not v_length(h) >= min_horizontal:
        return None
    return math.atan2(-a[0], -a[2])


def q_axis_angle(axis, angle: float) -> Quat:
    s = math.sin(angle / 2) / v_length(axis)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(angle / 2))


def q_yaw(angle: float) -> Quat:
    """``Quat.yaw``: a turn about +Y; positive turns -Z toward -X (a left turn for someone facing -Z)."""
    return q_axis_angle((0.0, 1.0, 0.0), angle)


def q_mul(a, b) -> Quat:
    return (a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
            a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
            a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
            a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2])  # fmt: skip


def q_conj(q) -> Quat:
    return (-q[0], -q[1], -q[2], q[3])


def q_neg(q) -> Quat:
    return (-q[0], -q[1], -q[2], -q[3])


def q_norm(q) -> float:
    return math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])


def q_finite(q) -> bool:
    return all(math.isfinite(c) for c in q)


def q_normalized(q) -> Quat:
    n = q_norm(q)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n) if n > 0 else IDENTITY


def q_dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


def q_rotate(q, v) -> Vec3:
    """v' = v + w t + u × t with t = 2 (u × v), in Swift's order."""
    u = (q[0], q[1], q[2])
    t = v_scale(2.0, v_cross(u, v))
    return v_add(v_add(v, v_scale(q[3], t)), v_cross(u, t))


def q_angle(a, b) -> float:
    d = q_mul(q_conj(q_normalized(a)), q_normalized(b))
    return 2 * math.atan2(v_length((d[0], d[1], d[2])), abs(d[3]))


def q_from_columns(c0, c1, c2) -> Quat:
    """``Quat(columns:)``: Shepperd's method with three.js's branches, then normalised."""
    m00, m10, m20 = c0
    m01, m11, m21 = c1
    m02, m12, m22 = c2
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1)
        q = ((m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s, 0.25 / s)
    elif m00 > m11 and m00 > m22:
        s = 2 * math.sqrt(1 + m00 - m11 - m22)
        q = (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    elif m11 > m22:
        s = 2 * math.sqrt(1 + m11 - m00 - m22)
        q = ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    else:
        s = 2 * math.sqrt(1 + m22 - m00 - m11)
        q = ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    return q_normalized(q)


def q_columns(q) -> tuple[Vec3, Vec3, Vec3]:
    return (q_rotate(q, (1.0, 0.0, 0.0)), q_rotate(q, (0.0, 1.0, 0.0)), q_rotate(q, (0.0, 0.0, 1.0)))


# ----------------------------------------------------------------------------------------------- HipsFrame.swift
CameraFacing = {"forward": 0.0, "backward": math.pi, "left": -math.pi / 2, "right": math.pi / 2}
"""``CameraFacing.toWearerForward``: the turn from the camera's horizontal view direction to the wearer's forward."""


@dataclass(frozen=True)
class ZeroState:
    epoch: int
    cam_position: Vec3
    cam_rotation: Quat
    align: Quat  # A: ARKit world -> reported frame, a turn about +Y
    heading: float  # ψ0


class ZeroError(Exception):
    """``cameraNotLevel`` (the camera looks nearly straight up or down) or ``notFinite``."""


MIN_HORIZONTAL = 0.3


@dataclass
class HipsFrame:
    lever: Vec3 = (0.0, 0.0, 0.0)
    hip_height: float = 1.0
    facing: str = "forward"
    zero: ZeroState | None = None

    def hip(self, p, q) -> Vec3:
        return v_add(p, q_rotate(q, self.lever))

    def set_zero(self, p, q) -> ZeroState:
        if not (v_finite(p) and q_finite(q) and q_norm(q) > 0):
            raise ZeroError("notFinite")
        r = q_normalized(q)
        view = q_rotate(r, (0.0, 0.0, -1.0))
        view_heading = v_heading(view, MIN_HORIZONTAL)
        if view_heading is None:
            raise ZeroError("cameraNotLevel")
        psi0 = math.remainder(view_heading + CameraFacing[self.facing], 2 * math.pi)
        state = ZeroState((self.zero.epoch if self.zero else 0) + 1, tuple(p), r, q_yaw(-psi0), psi0)
        self.zero = state
        return state

    def clear_zero(self) -> None:
        self.zero = None

    def report(self, p, q) -> tuple[Vec3, Quat] | None:
        z = self.zero
        if z is None:
            return None
        r = q_normalized(q)
        origin = self.hip(z.cam_position, z.cam_rotation)
        position = v_add(q_rotate(z.align, v_sub(self.hip(p, r), origin)), (0.0, self.hip_height, 0.0))
        orientation = q_normalized(q_mul(q_mul(q_mul(z.align, r), q_conj(z.cam_rotation)), q_conj(z.align)))
        return position, orientation


PAGE_TREES = ("velvetFamily", "matched", "thorny")
"""``PageTree``: velvet, spiny, honey, velvetGreen (and bc); smooth (a posswap equal to its axis map); thorny."""


def tree_wire(tree: str, p, q) -> tuple[Vec3, Quat]:
    """``PageTree.wire``: what goes on the wire for a reported pose."""
    if tree not in PAGE_TREES:
        raise ValueError(f"unknown tree setting {tree!r}")
    pos = (-p[0], p[1], -p[2]) if tree != "matched" else p
    ori = (-q[0], q[1], -q[2], q[3]) if tree == "thorny" else q
    return pos, ori


# ---------------------------------------------------------------------------------------------- HubMessage.swift
MAX_BYTES = 95
LEVELS = ((4, 3), (3, 3), (3, 2), (4, None))
_POW10 = (1, 10, 100, 1_000, 10_000, 100_000, 1_000_000)


def swift_rounded(x: float) -> float:
    """``Double.rounded()``: to the nearest integer, ties away from zero (x - trunc(x) is exact in doubles)."""
    t = float(math.trunc(x))
    if abs(x - t) >= 0.5:
        t += math.copysign(1.0, x)
    return t


def fixed(v: float, decimals: int, avoid_zero: bool = False) -> str | None:
    """``HubMessage.fixed``: at most ``decimals`` decimals, trailing zeros removed, never "-0"."""
    if not math.isfinite(v) or not 0 <= decimals < len(_POW10):
        return None
    scale = _POW10[decimals]
    scaled = swift_rounded(v * float(scale))
    if not abs(scaled) < 9e15:
        return None
    n = int(scaled)
    if n == 0 and avoid_zero:
        n = -1 if v < 0 else 1
    negative = n < 0
    a = -n if negative else n
    s = ("-" if negative else "") + str(a // scale)
    fraction = a % scale
    if fraction != 0:
        digits = decimals
        while fraction % 10 == 0:
            fraction //= 10
            digits -= 1
        f = str(fraction)
        s += "." + "0" * (digits - len(f)) + f
    return s


@dataclass(frozen=True)
class HubMessage:
    text: str
    level: int

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))

    @property
    def has_position(self) -> bool:
        return LEVELS[self.level][1] is not None

    def parsed(self) -> dict:
        return json.loads(self.text)


def _format(q, p, q_dec: int, p_dec: int | None) -> str | None:
    x, y, z, w = (fixed(c, q_dec) for c in q)
    if None in (x, y, z, w):
        return None
    s = f'{{"bone":"Hips","x":{x},"y":{y},"z":{z},"w":{w}'
    if p is not None and p_dec is not None:
        px, py, pz = (fixed(c, p_dec, avoid_zero=True) for c in p)
        if None in (px, py, pz):
            return None
        s += f',"px":{px},"py":{py},"pz":{pz}'
    return s + "}"


def encode(q, p) -> HubMessage | None:
    """``HubMessage.encode``: ``p`` None means orientation only. None only for a non-finite input."""
    if not (q_finite(q) and q_norm(q) > 0):
        return None
    u = q_normalized(q)
    pos = p if (p is not None and v_finite(p)) else None
    for index, (q_dec, p_dec) in enumerate(LEVELS):
        if pos is None and p_dec is not None:
            continue
        text = _format(u, pos, q_dec, p_dec)
        if text is None:
            continue
        if len(text.encode("utf-8")) <= MAX_BYTES:
            return HubMessage(text, index)
    return None


HubMessage.encode = staticmethod(encode)  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------- PosePipeline.swift
@dataclass(frozen=True)
class Tracking:
    """``Tracking``: ``normal``, ``limited:<reason>`` or ``not_available`` (its ``label``)."""

    label: str = "normal"

    @property
    def allows(self) -> tuple[bool, bool]:
        if self.label == "normal":
            return True, True
        if self.label in ("limited:excessiveMotion", "limited:insufficientFeatures"):
            return True, False
        return False, False


NORMAL = Tracking("normal")


@dataclass(frozen=True)
class FrameInput:
    time: float  # ARFrame.timestamp, s
    cam_position: Vec3
    cam_rotation: Quat
    tracking: Tracking = NORMAL


@dataclass
class FrameOutput:
    reported: tuple[Vec3, Quat] | None = None
    message: HubMessage | None = None
    zero: ZeroState | None = None
    zero_was_requested: bool = False
    zero_error: str | None = None


@dataclass
class PosePipeline:
    frame: HipsFrame = field(default_factory=HipsFrame)
    tree: str = "velvetFamily"
    rate_hz: float = 30.0
    last_sent: float = -math.inf
    last_orientation: Quat | None = None
    zero_requested: bool = False

    def request_zero(self) -> None:
        self.zero_requested = True

    def reset_world(self) -> None:
        self.frame.clear_zero()
        self.last_orientation = None
        self.last_sent = -math.inf

    def is_due(self, time: float) -> bool:
        return time - self.last_sent >= 1 / max(1.0, self.rate_hz) - 0.25 / 60

    def mark_sent(self, time: float) -> None:
        self.last_sent = time

    def process(self, f: FrameInput) -> FrameOutput:
        out = FrameOutput()
        if f.tracking == NORMAL and (self.zero_requested or self.frame.zero is None):
            try:
                out.zero = self.frame.set_zero(f.cam_position, f.cam_rotation)
                out.zero_was_requested = self.zero_requested
                self.zero_requested = False
                self.last_orientation = None
            except ZeroError as e:
                if self.zero_requested:
                    self.zero_requested = False
                    out.zero_error = str(e)
        reported = self.frame.report(f.cam_position, f.cam_rotation)
        if reported is None:
            return out
        position, orientation = reported
        if self.last_orientation is not None and q_dot(orientation, self.last_orientation) < 0:
            orientation = q_neg(orientation)
        self.last_orientation = orientation
        out.reported = (position, orientation)
        allow_q, allow_p = f.tracking.allows
        if not (allow_q and self.is_due(f.time)):
            return out
        wpos, wori = tree_wire(self.tree, position, orientation)
        out.message = encode(wori, wpos if allow_p else None)
        return out


# ---------------------------------------------------------------------------------------------- LeverArmFit.swift
@dataclass(frozen=True)
class LeverFitResult:
    lever: Vec3
    pivot: Vec3
    rms: float
    heading_range: float
    samples: int


class LeverFitFailure(Exception):
    """``tooFewSamples``, ``notEnoughTurn``, ``implausibleLever`` or ``poorFit``; ``args`` = (kind, value)."""


@dataclass
class LeverArmFit:
    """Pivot calibration from running sums (``MATH.md`` §4): l and c from p_i + R_i l = c, with a ridge to the prior."""

    prior: Vec3 = (0.0, 0.0, 0.0)
    count: int = 0
    sum_p: Vec3 = (0.0, 0.0, 0.0)
    sum_r: list = field(default_factory=lambda: [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    sum_rtp: Vec3 = (0.0, 0.0, 0.0)
    sum_pp: float = 0.0
    unwrapped: float | None = None
    last_heading: float | None = None
    min_heading: float = 0.0
    max_heading: float = 0.0

    @property
    def heading_range(self) -> float:
        return self.max_heading - self.min_heading

    def add(self, p, q) -> None:
        if not (v_finite(p) and q_finite(q)):
            return
        r = q_normalized(q)
        cols = q_columns(r)
        for row in range(3):
            for col in range(3):
                self.sum_r[row][col] += cols[col][row]
        self.sum_p = v_add(self.sum_p, p)
        self.sum_rtp = v_add(self.sum_rtp, q_rotate(q_conj(r), p))
        self.sum_pp += v_dot(p, p)
        self.count += 1
        h = v_heading(q_rotate(r, (0.0, 0.0, -1.0)), MIN_HORIZONTAL)
        if h is not None:
            if self.last_heading is not None and self.unwrapped is not None:
                nxt = self.unwrapped + math.remainder(h - self.last_heading, 2 * math.pi)
                self.unwrapped = nxt
                self.min_heading = min(self.min_heading, nxt)
                self.max_heading = max(self.max_heading, nxt)
            else:
                self.unwrapped = 0.0
                self.min_heading = 0.0
                self.max_heading = 0.0
            self.last_heading = h

    def solve(self, min_samples: int = 60, min_heading_range: float = math.pi, ridge: float = 1e-3,
              max_lever: float = 0.5, max_rms: float = 0.03) -> LeverFitResult:  # fmt: skip
        if self.count < min_samples:
            raise LeverFitFailure("tooFewSamples", self.count)
        if not self.heading_range >= min_heading_range:
            raise LeverFitFailure("notEnoughTurn", self.heading_range * 180 / math.pi)
        n = float(self.count)
        p_bar = v_scale(1 / n, self.sum_p)
        r_bar = [[self.sum_r[i][j] / n for j in range(3)] for i in range(3)]
        m = [[0.0] * 3 for _ in range(3)]
        for i in range(3):
            for j in range(3):
                s = 0.0
                for k in range(3):
                    s += r_bar[k][i] * r_bar[k][j]
                m[i][j] = (1 + ridge if i == j else 0) - s
        rt_pbar = (r_bar[0][0] * p_bar[0] + r_bar[1][0] * p_bar[1] + r_bar[2][0] * p_bar[2],
                   r_bar[0][1] * p_bar[0] + r_bar[1][1] * p_bar[1] + r_bar[2][1] * p_bar[2],
                   r_bar[0][2] * p_bar[0] + r_bar[1][2] * p_bar[1] + r_bar[2][2] * p_bar[2])  # fmt: skip
        rhs = v_add(v_sub(v_scale(ridge, self.prior), v_scale(1 / n, self.sum_rtp)), rt_pbar)
        lever = _solve3(m, rhs)
        if not (v_finite(lever) and v_length(lever) <= max_lever):
            raise LeverFitFailure("implausibleLever", v_length(lever))
        rl = (r_bar[0][0] * lever[0] + r_bar[0][1] * lever[1] + r_bar[0][2] * lever[2],
              r_bar[1][0] * lever[0] + r_bar[1][1] * lever[1] + r_bar[1][2] * lever[2],
              r_bar[2][0] * lever[0] + r_bar[2][1] * lever[1] + r_bar[2][2] * lever[2])  # fmt: skip
        pivot = v_add(p_bar, rl)
        sq = (self.sum_pp + n * v_dot(lever, lever) + n * v_dot(pivot, pivot) + 2 * v_dot(self.sum_rtp, lever)
              - 2 * v_dot(self.sum_p, pivot) - 2 * n * v_dot(rl, pivot))  # fmt: skip
        rms = math.sqrt(max(0.0, sq) / n)
        if not rms <= max_rms:
            raise LeverFitFailure("poorFit", rms)
        return LeverFitResult(lever, pivot, rms, self.heading_range, self.count)


def _det3(a) -> float:
    return (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))  # fmt: skip


def _solve3(m, b) -> Vec3:
    d = _det3(m)
    bv = (b[0], b[1], b[2])
    out = [0.0, 0.0, 0.0]
    for c in range(3):
        a = [row[:] for row in m]
        for r in range(3):
            a[r][c] = bv[r]
        out[c] = _det3(a) / d
    return (out[0], out[1], out[2])
