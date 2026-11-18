"""Name the OptiTrack markers of 2026-11-13 and follow them through the take (Step 3 brief §5.1).

One reflective marker sat on each of 11 pods. Motive exported them unlabeled: each column is a *fragment*, one marker
for as long as Motive kept it, and a lost marker comes back as a new fragment.

1. **Naming, in the T-pose that starts every take** (:func:`name_tpose`). The 11 visible markers are sorted by
   height: the highest is the head; the next four are the arms, the two nearer the midline the upper arms and the
   two farther out the forearms; then the chest, then the pelvis (the HipsAlt pod, on the lower back), then the two
   thighs, then the two shins. Forward is the horizontal direction from the pelvis marker (back) to the chest marker
   (front), left = up x forward, and each pair is split by its side of the midline.
2. **Following** (:func:`track_sites`), in two kinds of step that alternate:
   - *continuity*: a fragment that starts at most 0.5 s after a site's marker was lost, where that marker's last
     velocity predicts (world coordinates), is the same marker. Most losses are this short, and no facing is needed;
   - *templates*: every other fragment, longest first, goes to the free site whose usual body coordinates (forward,
     left, height about the head marker, the best-tracked one) its median matches, keeping each left site to the
     left of its right twin and each thigh above its own shin.

   The body frame's facing needs names and the names need the facing, so the template step runs three times, each
   from the T-pose names with the facing and templates of the previous run.

All positions are world coordinates (+Z up, metres; :func:`mesqsim.twin.page_frame.motive_to_world_vec`). The
thresholds below are this module's choices (`assumption`), set from tracking diagnostics (coverage, the left/right
checks, the leg-height correlation with the drawing), not by fitting any reference value of Step 3 brief §5.3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..twin.page_frame import motive_to_world_vec
from .motive import MotiveTake

__all__ = ["MARKER_SITES", "TrackResult", "name_tpose", "track_sites", "facing_from_sites", "smooth_facing"]

MARKER_SITES = ("Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "Spine", "Hips",
                "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg")
"""The 11 marker sites of 2026-11-13, named after the pod's bone (Spine is the chest pod, Hips the HipsAlt pod)."""

TPOSE_BANDS_M = {"Head": (1.71, 1.74), "arms": (1.45, 1.52), "Spine": (1.28, 1.29), "Hips": (1.04, 1.05),
                 "thighs": (0.58, 0.69), "shins": (0.34, 0.39)}
"""Heights in the T-pose (Step 3 brief §5.1, `measured` on 2026-11-13); reported against, never used to assign."""

FACING_WINDOW_S = 1.5


@dataclass
class TrackResult:
    sites: dict[str, np.ndarray]  # site -> (N, 3) world metres, NaN where not seen
    fragments: dict[str, list[int]]  # site -> marker columns, in time order
    tpose_frame: int
    tpose_heights_m: dict[str, float]
    facing: np.ndarray  # (N, 2) facing (pelvis -> head) averaged over 1.5 s, NaN where unknown
    unnamed: list[int] = field(default_factory=list)  # fragments left unnamed (strays or unmatched)

    def coverage(self) -> dict[str, float]:
        return {s: float(np.isfinite(p[:, 0]).mean()) for s, p in self.sites.items()}


def name_tpose(points_world: np.ndarray) -> dict[str, int]:
    """Name 11 marker positions ``(11, 3)`` (world, metres) taken in a T-pose; returns site -> row."""
    p = np.asarray(points_world, dtype=np.float64)
    if p.shape != (11, 3) or not np.all(np.isfinite(p)):
        raise ValueError(f"need 11 visible markers in the T-pose, got {p.shape}")
    order = list(np.argsort(-p[:, 2]))
    head, arms, chest, pelvis, thighs, shins = order[0], order[1:5], order[5], order[6], order[7:9], order[9:11]
    fwd = p[chest, :2] - p[pelvis, :2]
    if np.linalg.norm(fwd) < 0.05:
        raise ValueError("chest and pelvis markers are not apart horizontally; cannot tell forward")
    fwd = fwd / np.linalg.norm(fwd)
    left = np.array([-fwd[1], fwd[0]])  # up x forward
    mid = np.mean(p[[head, chest, pelvis], :2], axis=0)
    lat = (p[:, :2] - mid) @ left
    out = {"Head": head, "Spine": chest, "Hips": pelvis}
    arms = sorted(arms, key=lambda k: abs(lat[k]))
    for pair, (inner, outer) in (("Arm", (arms[0:2], arms[2:4])),):
        for k in inner:
            out[("Left" if lat[k] > 0 else "Right") + pair] = k
        for k in outer:
            out[("Left" if lat[k] > 0 else "Right") + "Fore" + pair] = k
    for pair, ks in (("UpLeg", thighs), ("Leg", shins)):
        for k in ks:
            out[("Left" if lat[k] > 0 else "Right") + pair] = k
    if len(out) != 11:
        raise ValueError(f"left/right split failed: {sorted(out)}")
    return out


def smooth_facing(raw: np.ndarray, dt_s: float, window_s: float = FACING_WINDOW_S) -> np.ndarray:
    """Average unit vectors ``(N, 2)`` over a centred window (NaN rows ignored), renormalised; NaN if none."""
    raw = np.asarray(raw, dtype=np.float64)
    ok = np.all(np.isfinite(raw), axis=1)
    v = np.where(ok[:, None], raw, 0.0)
    half = max(int(round(0.5 * window_s / dt_s)), 0)
    c = np.concatenate([np.zeros((1, 2)), np.cumsum(v, axis=0)])
    n = np.concatenate([[0], np.cumsum(ok)])
    lo = np.clip(np.arange(len(v)) - half, 0, len(v))
    hi = np.clip(np.arange(len(v)) + half + 1, 0, len(v))
    s = c[hi] - c[lo]
    cnt = n[hi] - n[lo]
    norm = np.linalg.norm(s, axis=1)
    good = (cnt > 0) & (norm > 1e-9)
    return np.where(good[:, None], s / np.where(good, norm, 1.0)[:, None], np.nan)


def facing_from_sites(sites: dict[str, np.ndarray]) -> np.ndarray:
    """Raw OptiTrack facing ``(N, 2)``: the horizontal direction from the pelvis marker to the head marker."""
    d = sites["Head"][:, :2] - sites["Hips"][:, :2]
    n = np.linalg.norm(d, axis=1, keepdims=True)
    return np.where(n > 1e-6, d / np.where(n > 1e-6, n, 1.0), np.nan)


def _fill_facing(f: np.ndarray) -> np.ndarray:
    """Facing at every frame for the body frame: interpolate the heading over gaps, hold it at the ends."""
    ok = np.all(np.isfinite(f), axis=1)
    if not ok.any():
        raise ValueError("no frame with both the head and the pelvis marker")
    ang = np.unwrap(np.arctan2(f[ok, 1], f[ok, 0]))
    a = np.interp(np.arange(len(f)), np.flatnonzero(ok), ang)
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def _body(p: np.ndarray, origin: np.ndarray, fwd: np.ndarray) -> np.ndarray:
    """World points ``(..., 3)`` -> body coordinates (forward, left, height) about ``origin`` (..., 2)."""
    d = p[..., :2] - origin
    left = np.stack([-fwd[..., 1], fwd[..., 0]], axis=-1)
    return np.stack([np.sum(d * fwd, axis=-1), np.sum(d * left, axis=-1), p[..., 2]], axis=-1)


SIDE = {"LeftArm": 1, "RightArm": -1, "LeftForeArm": 1, "RightForeArm": -1,
        "LeftUpLeg": 1, "RightUpLeg": -1, "LeftLeg": 1, "RightLeg": -1}
"""+1 for a site on the body's left, -1 on its right."""
TWIN = {"LeftArm": "RightArm", "RightArm": "LeftArm", "LeftForeArm": "RightForeArm", "RightForeArm": "LeftForeArm",
        "LeftUpLeg": "RightUpLeg", "RightUpLeg": "LeftUpLeg", "LeftLeg": "RightLeg", "RightLeg": "LeftLeg"}
"""The same site on the other side."""
PARTNER = {"LeftUpLeg": "LeftLeg", "LeftLeg": "LeftUpLeg", "RightUpLeg": "RightLeg", "RightLeg": "RightUpLeg"}
"""A thigh's marker stands above its own shin's (horizontally within about 20 cm), and the reverse."""
PARTNER_SLACK_M = 0.25  # a thigh marker more than 25 cm (median, horizontally) from its shin's costs the excess
MAX_COST_M = 0.25  # a fragment is named only if its body coordinates lie within 25 cm of the site's template
MIN_FREE = 0.95  # ... and the site is unseen during at least 95 % of the fragment
CONT_GAP_S = 0.5  # continuity: a fragment starting at most 0.5 s after a site's marker was lost ...
CONT_TAIL_S = 0.1  # ... where the velocity over that marker's last 0.1 s predicts ...
CONT_TOL_M = 0.08  # ... within 8 cm ...
CONT_TOL_PER_S = 0.5  # ... plus 50 cm per second of gap ...
CONT_MAX_SPEED = 3.0  # ... with the predicting speed capped at 3 m/s, is the same marker


FACING_CUES = (("Hips", "Head", "fwd"), ("Hips", "Spine", "fwd"), ("RightUpLeg", "LeftUpLeg", "left"),
               ("RightLeg", "LeftLeg", "left"), ("RightArm", "LeftArm", "left"))
"""Pairs (a, b) whose horizontal direction a -> b points forward (pelvis on the back to head or chest on the front)
or to the left (right to left twin). The pelvis marker is often hidden for tens of seconds, so the tracker's facing
takes every cue it has."""


def _tracking_facing(sites: dict[str, np.ndarray], dt: float) -> np.ndarray:
    """Facing for the tracker's body frame (not Φ2, which is pelvis -> head only): the sum of the unit cues of
    :data:`FACING_CUES` seen at each frame (left cues turned 90° clockwise), averaged over 1.5 s, interpolated over
    gaps."""
    total = np.zeros((len(next(iter(sites.values()))), 2))
    for a, b, kind in FACING_CUES:
        d = sites[b][:, :2] - sites[a][:, :2]
        if kind == "left":
            d = np.stack([d[:, 1], -d[:, 0]], axis=1)  # left -> forward
        n = np.linalg.norm(d, axis=1, keepdims=True)
        u = d / np.where(n > 0.05, n, np.inf)
        total += np.where(np.isfinite(u), u, 0.0)
    norm = np.linalg.norm(total, axis=1, keepdims=True)
    raw = np.where(norm > 1e-9, total / np.where(norm > 1e-9, norm, 1.0), np.nan)
    return _fill_facing(smooth_facing(raw, dt))


def _anchor(sites: dict[str, np.ndarray], xyz: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Body-frame origin ``(N, 2)``: the head marker where seen; elsewhere the horizontal median of the visible
    markers, shifted by the median offset from that median to the head."""
    med = np.nanmedian(np.where(vis[..., None], xyz[..., :2], np.nan), axis=1)
    head = sites["Head"][:, :2]
    ok = np.all(np.isfinite(head), axis=1) & np.all(np.isfinite(med), axis=1)
    off = np.median(head[ok] - med[ok], axis=0) if ok.any() else np.zeros(2)
    return np.where(ok[:, None], head, med + off)


def track_sites(take: MotiveTake, iterations: int = 3, max_cost_m: float = MAX_COST_M,
                min_free: float = MIN_FREE) -> TrackResult:
    """Name the T-pose markers, then every other fragment, and return each site's trajectory.

    Body coordinates (forward, left, height) are taken about the head marker (:func:`_anchor`) and the tracker's
    facing (:func:`_tracking_facing`). Each fragment is described by its median body coordinates over its whole
    life; each site by a template, the median over its named fragments. Fragments are named longest first, each
    after the site with the nearest template (within ``max_cost_m``) that is free (unseen during ``min_free`` of the
    fragment), that keeps a left site to the left of its right twin wherever both are seen, and with a cost raised
    when a thigh is not above its own shin. Templates and facing come from the previous naming, so the naming runs
    ``iterations`` times, starting from the T-pose names each time.
    """
    xyz = motive_to_world_vec(take.xyz)  # (N, M, 3)
    n, m = xyz.shape[:2]
    dt = 1.0 / take.frame_rate_hz
    vis = np.isfinite(xyz[..., 0])
    alive = vis.any(axis=0)
    counts = vis.sum(axis=1)
    window = np.flatnonzero(counts[: int(2.0 / dt) + 1] == 11)
    if window.size == 0:
        raise ValueError(f"{take.take_name}: no frame with exactly 11 markers in the first 2 s (the T-pose)")
    k0 = int(window[0])
    cols0 = np.flatnonzero(vis[k0])
    named = name_tpose(xyz[k0, cols0])
    seed = {int(cols0[row]): site for site, row in named.items()}
    rows = {c: np.flatnonzero(vis[:, c]) for c in range(m) if alive[c]}

    def arrays(owner) -> dict[str, np.ndarray]:
        out = {s: np.full((n, 3), np.nan) for s in MARKER_SITES}
        for c, s in owner.items():
            out[s][rows[c]] = xyz[rows[c], c]
        return out

    first = {c: r[0] for c, r in rows.items()}
    last = {c: r[-1] for c, r in rows.items()}
    n_tail = max(int(round(CONT_TAIL_S / dt)), 1)

    def continuity(owner: dict[int, str]) -> dict[int, str]:
        """Link, in time order, each unnamed fragment to the site whose last marker ended at most ``CONT_GAP_S``
        earlier where its velocity predicts (world coordinates; no facing needed)."""
        owner = dict(owner)
        for c in sorted((c for c in rows if c not in owner), key=lambda c: first[c]):
            t0, p0 = first[c], xyz[first[c], c]
            best, best_d = None, np.inf
            for s in MARKER_SITES:
                mine = [f for f, o in owner.items() if o == s]
                if any(first[f] <= t0 <= last[f] for f in mine):
                    continue
                recent = [f for f in mine if last[f] < t0 and (t0 - last[f]) * dt <= CONT_GAP_S]
                if not recent:
                    continue
                f = max(recent, key=lambda f: last[f])
                k = last[f]
                tail = rows[f][rows[f] >= k - n_tail]
                v = (xyz[k, f] - xyz[tail[0], f]) / ((k - tail[0]) * dt) if k > tail[0] else np.zeros(3)
                v = v * min(1.0, CONT_MAX_SPEED / max(np.linalg.norm(v), 1e-9))
                gap = (t0 - k) * dt
                d = float(np.linalg.norm(p0 - (xyz[k, f] + v * gap)))
                if d < CONT_TOL_M + CONT_TOL_PER_S * gap and d < best_d:
                    best, best_d = s, d
            if best is not None:
                owner[c] = best
        return owner

    owner = continuity(seed)
    for _ in range(iterations):
        sites = arrays(owner)
        fwd = _tracking_facing(sites, dt)
        anchor = _anchor(sites, xyz, vis)
        body = {c: _body(xyz[r, c], anchor[r], fwd[r]) for c, r in rows.items()}
        feat = {c: np.median(b, axis=0) for c, b in body.items()}
        templ = {}
        for s in MARKER_SITES:
            mine = [c for c, o in owner.items() if o == s]
            templ[s] = np.median(np.concatenate([body[c] for c in mine]), axis=0)
        owner = continuity(seed)
        linked = set(owner)
        cur = arrays(owner)  # the sites as named so far in this pass
        for c in sorted((c for c in rows if c not in linked), key=lambda c: -len(rows[c])):
            r = rows[c]
            best, best_cost = None, max_cost_m
            for s in MARKER_SITES:
                if np.isfinite(cur[s][r, 0]).mean() > 1.0 - min_free:
                    continue  # not free
                cost = float(np.linalg.norm(feat[c] - templ[s]))
                if s in TWIN:  # left of the right twin (and the reverse) wherever both are seen
                    tw = cur[TWIN[s]][r]
                    ok = np.isfinite(tw[:, 0])
                    if ok.sum() > 0.2 * len(r):
                        lat = body[c][ok, 1]
                        lat_tw = _body(tw[ok], anchor[r[ok]], fwd[r[ok]])[:, 1]
                        if SIDE[s] * np.median(lat - lat_tw) < 0:
                            continue
                if s in PARTNER:  # a thigh above its own shin
                    pa = cur[PARTNER[s]][r]
                    ok = np.isfinite(pa[:, 0])
                    if ok.sum() > 0.2 * len(r):
                        cost += max(0.0, float(np.median(np.linalg.norm(xyz[r[ok], c, :2] - pa[ok, :2], axis=1)))
                                    - PARTNER_SLACK_M)
                if cost < best_cost:
                    best, best_cost = s, cost
            if best is not None:
                owner[c] = best
                cur[best][r] = xyz[r, c]
        owner = continuity(owner)  # successors of fragments named by template
    sites = arrays(owner)
    facing = smooth_facing(facing_from_sites(sites), dt)
    frags = {s: sorted((c for c, o in owner.items() if o == s), key=lambda c: rows[c][0]) for s in MARKER_SITES}
    heights = {s: float(xyz[k0, cols0[row], 2]) for s, row in named.items()}
    return TrackResult(sites, frags, k0, heights, facing, [c for c in rows if c not in owner])
