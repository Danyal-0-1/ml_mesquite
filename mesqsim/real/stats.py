"""The statistics Φ1-Φ12 of a recording (Step 3 brief §5.2). One code path for real files and twin files.

Everything works on a :class:`Series`: sites in world coordinates (+Z up, metres), a facing (a horizontal unit
vector averaged over 1.5 s) and, for a BVH file, the drawn root. A series comes from a BVH file
(:func:`series_from_bvh`, the page's drawing) or from an OptiTrack take (:func:`series_from_track`, the truth).

Conventions. Body coordinates are (forward, right) of the facing, right = facing x up. An angle "travel minus
facing" is positive when the travel is turned anticlockwise seen from above (to the left of the facing), the sense of
:func:`mesqsim.conventions.yaw_quat`. A swing direction is an axis: atan2(right, forward) of the major axis, wrapped
into (-90°, 90°], positive forward-right, compared modulo 180°.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..conventions import geodesic_angle_rad, heading_rad, quat_conj, quat_mul, wrap_deg
from ..twin.page_frame import bvh_euler_to_quat, three_to_world_rot, three_to_world_vec
from .bvh import Bvh, bvh_facing, bvh_sites, forward_kinematics
from .markers import TrackResult, smooth_facing

__all__ = [
    "WALK_STD_M", "WALK_WINDOW_S", "STILL_DEG", "EXACT_DEG", "JUMP_DEG", "SNAP_BACK_S", "SLOW_M_S",
    "HOLD_WITHIN_DEG", "Series", "series_from_bvh", "series_from_track", "resample", "body_xy", "walking_mask",
    "axial_deg", "swing_axis_deg", "swing_roundness", "swing_windows", "umeyama_2d", "root_speed",
    "travel_minus_facing", "frame_angles_deg", "recorded_rotations", "exact_repeats", "jumps", "static_hold",
]

WALK_STD_M = 0.03  # Φ1: both shins' displacement from the pelvis has a standard deviation above 3 cm ...
WALK_WINDOW_S = 1.0  # ... over 1 s
STILL_DEG = 0.2  # Φ8: a still frame turns less than 0.2° from the previous one
EXACT_DEG = 1e-6  # Φ8: an exact repeat turns less than 1e-6°
JUMP_DEG = 15.0  # Φ9: a jump turns more than 15° in one frame
SNAP_BACK_S = 0.5  # Φ9: a jump snaps back if the bone returns within 0.5 s
SNAP_BACK_DEG = 5.0  # ... to within 5° of where it was before the jump (this module's choice)
SLOW_M_S = 0.05  # Φ6, Φ7: 5 cm/s


@dataclass
class Series:
    t_s: np.ndarray  # (T,) seconds from the first frame
    dt_s: float
    sites: dict[str, np.ndarray]  # site -> (T, 3) world metres, NaN where unknown
    facing: np.ndarray  # (T, 2) unit horizontal vectors, averaged over 1.5 s; NaN where unknown
    root: np.ndarray | None  # (T, 3) the drawn root (BVH only), world metres
    source: str  # "bvh" or "optitrack"


def series_from_bvh(bvh: Bvh, fk=None) -> Series:
    fk = fk if fk is not None else forward_kinematics(bvh)
    facing = smooth_facing(bvh_facing(bvh, fk), bvh.frame_time_s)
    root = three_to_world_vec(fk[0][:, 0]) / 100.0
    return Series(bvh.t_s, bvh.frame_time_s, bvh_sites(bvh, fk), facing, root, "bvh")


def series_from_track(track: TrackResult, t_s: np.ndarray, dt_s: float) -> Series:
    return Series(t_s, dt_s, track.sites, track.facing, None, "optitrack")


def resample(series: Series, t_query_s: np.ndarray, dt_s: float) -> Series:
    """The series looked up at ``t_query_s`` (nearest sample of its own uniform grid; NaN outside)."""
    k = np.rint((np.asarray(t_query_s) - series.t_s[0]) / series.dt_s).astype(np.int64)
    ok = (k >= 0) & (k < len(series.t_s))

    def take(a):
        out = np.full((len(k),) + a.shape[1:], np.nan)
        out[ok] = a[k[ok]]
        return out

    return Series(np.asarray(t_query_s, dtype=np.float64), dt_s, {s: take(p) for s, p in series.sites.items()},
                  take(series.facing), None if series.root is None else take(series.root), series.source)


def body_xy(series: Series, site: str, origin: str = "Hips") -> np.ndarray:
    """``(T, 2)`` horizontal displacement of ``site`` from ``origin`` in (forward, right) of the facing."""
    d = series.sites[site][:, :2] - series.sites[origin][:, :2]
    f = series.facing
    right = np.stack([f[:, 1], -f[:, 0]], axis=1)  # facing x up
    return np.stack([np.sum(d * f, axis=1), np.sum(d * right, axis=1)], axis=1)


def _moving_std(x: np.ndarray, half: int) -> np.ndarray:
    """Centred moving standard deviation (population) of ``x`` (T, k) summed over its k columns; NaN-aware."""
    ok = np.all(np.isfinite(x), axis=1)
    v = np.where(ok[:, None], x, 0.0)
    c1 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(v, axis=0)])
    c2 = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(v * v, axis=0)])
    cn = np.concatenate([[0], np.cumsum(ok)])
    i = np.arange(len(x))
    lo, hi = np.clip(i - half, 0, len(x)), np.clip(i + half + 1, 0, len(x))
    n = cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (c1[hi] - c1[lo]) / n[:, None]
        var = (c2[hi] - c2[lo]) / n[:, None] - mean ** 2
        out = np.sqrt(np.clip(var, 0.0, None).sum(axis=1))
    return np.where(n >= max(half, 2), out, np.nan)


def walking_mask(series: Series) -> np.ndarray:
    """Φ1: frames where both shins' (forward, right) displacement from the pelvis has a standard deviation (both
    components together) above 3 cm over a centred 1 s window."""
    half = int(round(0.5 * WALK_WINDOW_S / series.dt_s))
    s_l = _moving_std(body_xy(series, "LeftLeg"), half)
    s_r = _moving_std(body_xy(series, "RightLeg"), half)
    with np.errstate(invalid="ignore"):
        return (s_l > WALK_STD_M) & (s_r > WALK_STD_M)


def axial_deg(a) -> np.ndarray:
    """Wrap axis angles into (-90°, 90°]."""
    a = np.mod(np.asarray(a, dtype=np.float64) + 90.0, 180.0) - 90.0
    return np.where(a <= -90.0, a + 180.0, a)


def swing_axis_deg(series: Series, site: str, mask: np.ndarray, min_frames: int = 30) -> float:
    """Φ3: the major axis of the site's (forward, right) position relative to the pelvis over the masked frames,
    as atan2(right, forward) in (-90°, 90°]."""
    xy = body_xy(series, site)[mask]
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < min_frames:
        return float("nan")
    w, v = np.linalg.eigh(np.cov(xy.T))
    u = v[:, np.argmax(w)]
    return float(axial_deg(np.rad2deg(np.arctan2(u[1], u[0]))))


def swing_roundness(series: Series, site: str, mask: np.ndarray, min_frames: int = 30) -> float:
    """Minor over major standard deviation of the swing (0: a line, 1: a circle, whose axis means nothing)."""
    xy = body_xy(series, site)[mask]
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < min_frames:
        return float("nan")
    w = np.linalg.eigvalsh(np.cov(xy.T))
    return float(np.sqrt(max(w[0], 0.0) / w[1])) if w[1] > 0 else float("nan")


def swing_windows(series: Series, site: str, walking: np.ndarray, window_s: float = 30.0,
                  min_walk_s: float = 3.0) -> tuple[np.ndarray, np.ndarray, float]:
    """Φ4: Φ3 in consecutive windows (start times, angles) and the slope in degrees per minute.

    Windows with less than ``min_walk_s`` of walking give NaN. The angles are unwrapped as axes (period 180°)
    before the least-squares slope against the window centres.
    """
    starts = np.arange(0.0, series.t_s[-1] + 1e-9, window_s)
    ang = np.full(len(starts), np.nan)
    for i, t0 in enumerate(starts):
        sel = walking & (series.t_s >= t0) & (series.t_s < t0 + window_s)
        if sel.sum() * series.dt_s >= min_walk_s:
            ang[i] = swing_axis_deg(series, site, sel)
    ok = np.isfinite(ang)
    slope = float("nan")
    if ok.sum() >= 2:
        un = np.rad2deg(np.unwrap(np.deg2rad(ang[ok]) * 2.0) / 2.0)
        centres_min = (starts[ok] + 0.5 * window_s) / 60.0
        slope = float(np.polyfit(centres_min, un, 1)[0])
    return starts, ang, slope


def umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Φ5: the similarity dst ≈ s·R(θ)·src + t of least squares (Umeyama 1991) for (n, 2) point sets.

    Returns (s, θ in degrees, RMS residual, t). Rows with a NaN in either set are left out.
    """
    ok = np.all(np.isfinite(src), axis=1) & np.all(np.isfinite(dst), axis=1)
    a, b = src[ok], dst[ok]
    ma, mb = a.mean(axis=0), b.mean(axis=0)
    xa, xb = a - ma, b - mb
    u, d, vt = np.linalg.svd(xb.T @ xa / len(a))
    sgn = np.diag([1.0, np.sign(np.linalg.det(u) * np.linalg.det(vt))])
    r = u @ sgn @ vt
    s = float(np.trace(np.diag(d) @ sgn) / np.mean(np.sum(xa * xa, axis=1)))
    t = mb - s * r @ ma
    res = b - (s * a @ r.T + t)
    return s, float(np.rad2deg(np.arctan2(r[1, 0], r[0, 0]))), float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))), t


def _velocity_xy(p: np.ndarray, dt_s: float) -> np.ndarray:
    return np.gradient(p[:, :2], dt_s, axis=0)


def root_speed(series: Series, walking: np.ndarray) -> tuple[float, float]:
    """Φ6: median horizontal root speed while walking (m/s) and the share of walking frames below 5 cm/s."""
    sp = np.linalg.norm(_velocity_xy(series.root, series.dt_s), axis=1)[walking]
    sp = sp[np.isfinite(sp)]
    return (float(np.median(sp)), float(np.mean(sp < SLOW_M_S))) if sp.size else (float("nan"), float("nan"))


def travel_minus_facing(series: Series, walking: np.ndarray) -> float:
    """Φ7: median of wrap(direction of the root's horizontal velocity - facing), walking frames at 5 cm/s or more."""
    v = _velocity_xy(series.root, series.dt_s)
    sel = walking & (np.linalg.norm(v, axis=1) >= SLOW_M_S) & np.all(np.isfinite(series.facing), axis=1)
    d = np.arctan2(v[sel, 1], v[sel, 0]) - np.arctan2(series.facing[sel, 1], series.facing[sel, 0])
    return float(np.median(wrap_deg(np.rad2deg(d)))) if sel.any() else float("nan")


def frame_angles_deg(q: np.ndarray) -> np.ndarray:
    """Frame-to-frame rotation angles (T-1,) in degrees of a quaternion series (T, 4)."""
    return np.rad2deg(geodesic_angle_rad(quat_mul(quat_conj(q[:-1]), q[1:])))


def recorded_rotations(bvh: Bvh, bone: str) -> np.ndarray:
    """``(F, 4)`` the bone's rotation as recorded: its BVH channels, i.e. its local rotation."""
    j = bvh.joints[bvh.index(bone)]
    order = "".join(c[0] for c in j.channels if c.lower().endswith("rotation"))
    return bvh_euler_to_quat(bvh.rotation_channels(bone), order)


def exact_repeats(bvh: Bvh, bones: list[str]) -> dict[str, float]:
    """Φ8: per bone, among still frames of its recorded (local) rotation, the share that repeat exactly."""
    out = {}
    for b in bones:
        a = frame_angles_deg(recorded_rotations(bvh, b))
        still = a < STILL_DEG
        out[b] = float(np.mean(a[still] < EXACT_DEG)) if still.any() else float("nan")
    return out


def _global_world(bvh: Bvh, fk) -> np.ndarray:
    return three_to_world_rot(fk[1])  # (F, J, 4): each bone's drawn orientation in the world


def jumps(bvh: Bvh, bones: list[str]) -> dict[str, list[tuple[float, float, bool]]]:
    """Φ9: per bone, every frame whose recorded (local) rotation turns more than 15°, as (time, angle, snapped
    back), where snapped back means the bone comes within 5° of its pre-jump rotation within 0.5 s. Recorded
    rotations, like Φ8: a pod's own glitch shows there, a parent's does not."""
    n_back = int(round(SNAP_BACK_S / bvh.frame_time_s))
    out = {}
    for b in bones:
        q = recorded_rotations(bvh, b)
        a = frame_angles_deg(q)
        ev = []
        for k in np.flatnonzero(a > JUMP_DEG):
            after = q[k + 1: k + 2 + n_back]
            back = np.rad2deg(geodesic_angle_rad(quat_mul(quat_conj(np.broadcast_to(q[k], after.shape)), after)))
            ev.append((float(bvh.t_s[k + 1]), float(a[k]), bool(np.any(back < SNAP_BACK_DEG))))
        out[b] = ev
    return out


HOLD_WITHIN_DEG = 1.8  # Φ12: a pod holds if its heading drift over the window stays within 1.8°


def static_hold(bvh: Bvh, bones: list[str], fk=None) -> dict:
    """Φ12 for a static hold, from each pod's drawn (global) orientation in the world.

    Restart events are frames turning more than 15°. The window runs from the start to the first restart of any
    pod (the whole file without one). Per pod over the window: the heading drift (degrees per minute, least
    squares), the drift over the window (|drift| x window), the largest rotation away from the first frame and the
    frame-to-frame noise (standard deviation of the frame-to-frame angle). A pod *holds* when its drift over the
    window is within 1.8°; pods that restart are counted apart.
    """
    fk = fk if fk is not None else forward_kinematics(bvh)
    g = _global_world(bvh, fk)
    restarts = {b: [float(bvh.t_s[k + 1]) for k in np.flatnonzero(frame_angles_deg(g[:, bvh.index(b)]) > JUMP_DEG)]
                for b in bones}
    first = min((v[0] for v in restarts.values() if v), default=float(bvh.t_s[-1]) + bvh.frame_time_s)
    w = bvh.t_s < first
    t_min = bvh.t_s[w] / 60.0
    window_min = float(w.sum() * bvh.frame_time_s / 60.0)
    pods = {}
    for b in bones:
        q = g[w, bvh.index(b)]
        if q.shape[0] < 2:  # a restart in the first frames leaves nothing to fit (twin sessions can do this)
            pods[b] = {"drift_deg_per_min": float("nan"), "drift_over_window_deg": float("nan"),
                       "max_dev_deg": float("nan"), "noise_deg": float("nan"), "restarts_s": restarts[b]}
            continue
        h = np.rad2deg(np.unwrap(heading_rad(q)))
        drift = float(np.polyfit(t_min, h - h[0], 1)[0])
        dev = np.rad2deg(geodesic_angle_rad(quat_mul(quat_conj(np.broadcast_to(q[0], q.shape)), q)))
        pods[b] = {"drift_deg_per_min": drift, "drift_over_window_deg": abs(drift) * window_min,
                   "max_dev_deg": float(dev.max()), "noise_deg": float(np.std(frame_angles_deg(q))),
                   "restarts_s": restarts[b]}
    steady = [b for b in bones if not restarts[b]]
    holding = [b for b in steady if pods[b]["drift_over_window_deg"] <= HOLD_WITHIN_DEG]
    return {"window_min": window_min, "pods": pods, "steady": steady, "holding": holding,
            "restarted": [b for b in bones if restarts[b]]}
