"""Calibrating the twin against the recordings (Step 3 brief §5.4-5.7): ``python -m mesqsim twin-calibrate``.

**The split** (§5.4, proposed; Danyal freezes it): calibrate on Nov 4, Nov 13 13-41-17 and 13-59-24, and Nov 14;
check on Nov 13 13-42-39, 13-44-44 and 13-47-28. Nothing here ever fits on a check recording.

**Matching sessions** (§5.6 step 1). For each recording: its suit version (:mod:`.suits`) and a program from one
training subject whose walking clips match the recording's walking speed within 20 % (the true speed: the drawn root
speed divided by Φ5's scale on Nov 13, by the page's 1/2 on Nov 14, whose root path has no truth), filled to the
recording's walking time, with the subject's turning clips spread to match its turning rate (from the BVH root path;
the brief prefers the OptiTrack pelvis path where there is one, which turns with the root up to the phone's drift),
and still stands for the rest. Every program starts with a 5 s T-pose, as the takes do. Nov 4 is the static hold:
a T-pose, then the pods still for its whole length.

**The fit** (§5.6): approximate Bayesian computation by rejection (Beaumont et al. 2002). θ is drawn from
:data:`PRIORS`; for each θ one session per calibration recording is simulated, run through the page twin, and
measured by :func:`.twin_stats.analyse_twin`; the distance is the root mean square over the summary statistics of
:func:`summary`, each difference divided by that statistic's spread across the draws (its MAD) but never by less
than its reproduction tolerance (§5.3, :data:`TOLERANCE`), each squared term capped at :data:`Z2_CAP` (so that one
statistic the twin cannot reach does not decide the ranking alone); the closest ``keep`` share of draws is the
posterior. Nov 4 is its own suit version (:mod:`.suits`): its wire precision is fitted apart from Nov 13's.

**The gap** (§5.7): for each recording and core statistic, where the real value falls in the posterior predictive,
for the twin and for Step 1's simulator (the twin's ``analytic`` mode with Step 1's own error model, through the same
page twin); a classifier two-sample test on 10 s windows of the drawn rotations (Lopez-Paz and Oquab 2017),
leaving one recording out; and gate G-T.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..body import POD_BY_NAME
from ..config import load_config
from ..data.amass import amass_dir
from ..paths import REPO_ROOT
from ..real.recordings import RECORDINGS, real_suit_dir
from ..sim.rng import stream
from .config import load_twin_config
from .suits import suit_overrides

__all__ = ["CALIBRATE", "CHECK", "PRIORS", "Prior", "real_profiles", "walking_clip_table", "matching_program",
           "draw_theta", "summary", "simulate_recording", "distances", "posterior", "STEP1_THETA"]

CALIBRATE = ("nov04", "nov13_1341", "nov13_1359", "nov14")
CHECK = ("nov13_1342", "nov13_1344", "nov13_1347")
REC = {r.id: r for r in RECORDINGS}
CORE = ("phi3_LeftLeg", "phi3_RightLeg", "phi4_LeftLeg", "phi4_RightLeg", "phi5_scale", "phi6", "phi7", "phi8")
"""The core statistics of the predictive checks (§5.7: Φ3, Φ4, Φ5, Φ6, Φ7, Φ8), as named by :func:`summary`."""


@dataclass(frozen=True)
class Prior:
    key: str  # a twin config key, or "step1:<key>" for configs/default.yaml
    kind: str  # "uniform", "log_uniform" or "choice"
    lo: float = 0.0
    hi: float = 1.0
    choices: tuple = ()
    suits: tuple = ("suit_2026_11_04", "suit_2026_11_13", "suit_2026_11_14")  # the suits it applies to

    def draw(self, rng: np.random.Generator):
        if self.kind == "uniform":
            return float(rng.uniform(self.lo, self.hi))
        if self.kind == "log_uniform":
            return float(math.exp(rng.uniform(math.log(self.lo), math.log(self.hi))))
        return self.choices[int(rng.integers(len(self.choices)))]


PRIORS = (
    Prior("twin.sensors.gyro_bias_sigma_deg_per_min", "log_uniform", 0.02, 2.0),
    Prior("twin.sensors.gyro_bias_walk_deg_s_sqrt_s", "log_uniform", 1e-4, 0.1),
    Prior("twin.sensors.gyro_scale_max", "uniform", 0.0, 0.03),
    Prior("twin.fusion.vqf_tau_acc_s", "log_uniform", 0.1, 10.0),  # round 3: round 2's floor of 1 s excluded the data
    Prior("step1:error_model.box_slot_deg", "log_uniform", 0.5, 30.0),
    Prior("twin.wire.json_decimals", "choice", choices=(2, 3, 4, 5, 6), suits=("suit_2026_11_13",)),
    Prior("twin.wire.json_decimals@suit_2026_11_04", "choice", choices=(2, 3, 4, 5, 6), suits=("suit_2026_11_04",)),
    Prior("twin.procedure.p_limb_restart", "uniform", 0.0, 1.0),
    Prior("twin.procedure.p_zero_at_start", "uniform", 0.0, 1.0),
    Prior("twin.procedure.p_camera_mismatch", "uniform", 0.0, 1.0, suits=("suit_2026_11_14",)),
)
"""The priors of §5.5's fitted parameters (`assumption`): the residual gyroscope bias's spread, the gyroscope scale
error (bounded by twice the datasheet's ±1.5 %), VQF's tilt time constant, the box-slot error (up to 30°: Nov 13's
legs start 34° apart), the old wire's decimals, and the slips' probabilities (fitted weakly: one session each)."""

STEP1_THETA = {"twin.fusion.mode": "analytic"}
"""Step 1's simulator in the twin: its analytic heading model with configs/default.yaml's error model, unchanged."""


# ----------------------------------------------------------------------------------------------- the recordings
TURN_SMOOTH_S = 2.0


def net_turn_rate_deg_s(root_xy: np.ndarray, dt_s: float, walking: np.ndarray) -> float:
    """The path's net turning while walking, °/s: the heading of the horizontal velocity averaged over 2 s (so the
    gait's sway does not count), its total absolute change over walking frames divided by the walking time."""
    v = np.gradient(root_xy, dt_s, axis=0)
    k = max(1, int(round(TURN_SMOOTH_S / dt_s)))
    kern = np.ones(k) / k
    vs = np.stack([np.convolve(v[:, a], kern, mode="same") for a in range(2)], axis=1)
    fast = walking & (np.linalg.norm(vs, axis=1) > 0.05)
    if fast.sum() < 10:
        return 0.0
    h = np.unwrap(np.arctan2(vs[:, 1], vs[:, 0]))
    dh = np.abs(np.diff(h))[fast[1:] & fast[:-1]]
    return float(np.rad2deg(dh.sum()) / (fast.sum() * dt_s))


def _turn_rate_deg_s(rec_id: str) -> float:
    """The drawn root path's net turning rate while walking (the root turns with the phone's path; the brief prefers
    the OptiTrack pelvis path where there is one, which differs from it only by the phone's drift)."""
    from ..real import stats as st
    from ..real.bvh import forward_kinematics, read_bvh

    bvh = read_bvh(real_suit_dir() / REC[rec_id].bvh)
    d = st.series_from_bvh(bvh, forward_kinematics(bvh))
    return net_turn_rate_deg_s(d.root[:, :2], d.dt_s, st.walking_mask(d))


def real_profiles(summary: dict) -> dict:
    """Per recording, from ``real-stats``' summary: length, walking time, true speed, turning rate, kind, suit."""
    out = {}
    for rid, r in summary["results"].items():
        rec = REC[rid]
        scale = (r.get("phi5") or {}).get("scale") or 0.5  # Nov 14 has no truth: the page's 1/2 (Step 3 brief §2.4)
        drawn = r.get("phi6_speed_cm_s")
        pods = sorted({2 if b == "Hips" else POD_BY_NAME[b].id for b in r["pod_bones"]})
        out[rid] = {"duration_s": r["duration_s"], "walking_s": r["phi1_walking_s"], "kind": rec.kind,
                    "suit": rec.suit_version, "role": rec.role, "pods": pods,
                    "speed_m_s": None if drawn is None else drawn / 100.0 / scale,
                    "turn_deg_s": _turn_rate_deg_s(rid) if r["phi1_walking_s"] > 5 else 0.0}  # fmt: skip
    return out


@functools.lru_cache(maxsize=1)
def walking_clip_table() -> pd.DataFrame:
    """Every training-split walking clip on disk: subject, path, length, path speed (m/s), turning rate (°/s)."""
    from .program import _to_rot, resample
    from .smplh import load_smplh_sequence

    m = pd.read_csv(REPO_ROOT / "manifests" / "split_manifest.csv", dtype={"subject": str})
    m = m[(m["split"] == "train") & m["path"].str.contains("walk", case=False)]
    rows = []
    for _, r in m.iterrows():
        f = amass_dir() / r["path"]
        if not f.is_file():
            continue
        seq = load_smplh_sequence(f)
        _, tr = resample(_to_rot(seq.poses[:, :]), seq.trans, seq.fps, 30.0)
        dur = (tr.shape[0] - 1) / 30.0
        if dur < 2.0:
            continue
        step = np.diff(tr[:, :2], axis=0)
        v = np.linalg.norm(step, axis=1) * 30.0
        h = np.unwrap(np.arctan2(step[:, 1], step[:, 0]))
        ok = v[1:] > 0.05
        turn = float(np.rad2deg(np.median(np.abs(np.diff(h) * 30.0)[ok]))) if ok.sum() > 10 else 0.0
        rows.append({"subset": r["subset"], "subject": r["subject"], "path": r["path"], "seconds": dur,
                     "speed_m_s": float(v.mean()), "turn_deg_s": turn})  # fmt: skip
    return pd.DataFrame(rows)


def matching_program(cfgt, profile: dict, seed: int):
    """``(joined, model, label, match)`` for a recording's profile (module docstring). ``match`` reports how close
    the chosen clips came: their mean speed and turning against the recording's."""
    from .program import Piece, clip_piece, hold_piece, join, load_smplh_sequence, synthetic_piece, tpose_piece
    from .smplh import load_model

    rate = cfgt["twin.program.rate_hz"]
    blend, window = cfgt["twin.program.blend_s"], cfgt["twin.program.heading_window_s"]
    rng = stream(seed, "twin.matching")
    if profile["kind"] == "static" or not profile["speed_m_s"]:
        subset, subject = "KIT", "3"
        clips = pd.DataFrame()
    else:
        t = walking_clip_table()
        v, w = profile["speed_m_s"], profile["turn_deg_s"]
        near = t[(t["speed_m_s"] - v).abs() <= 0.2 * v]
        if near.empty:  # nobody walks that fast or slow: take the closest clips
            near = t.iloc[(t["speed_m_s"] - v).abs().argsort()[:20]]
        score = near.groupby(["subset", "subject"]).agg(n=("path", "size"), secs=("seconds", "sum"))
        subset, subject = score.sort_values(["secs", "n"], ascending=False).index[0]
        clips = near[(near["subset"] == subset) & (near["subject"] == subject)]
        # within the subject: prefer clips whose turning is near the recording's
        clips = clips.assign(dt=(clips["turn_deg_s"] - w).abs()).sort_values("dt")
    first_path = (clips["path"].iloc[0] if len(clips) else
                  sorted(walking_clip_table().query("subset == 'KIT' and subject == '3'")["path"])[0])  # fmt: skip
    first = load_smplh_sequence(amass_dir() / first_path)
    model = load_model(first.gender)
    pieces: list[Piece] = [tpose_piece(model, first.betas, 5.0, rate)]
    walked, turns = 0.0, 0
    if len(clips):
        # legs of walking, each followed by a 180° turn, at the recording's net turning rate (walks there and back)
        w = max(profile["turn_deg_s"], 1.0)
        leg_s = float(np.clip(180.0 / w, 2.0, 60.0))
        travel = 0.0
        turn_s = 2.0  # each turnaround: 180° on the spot at 90°/s; the real walking time includes its turnarounds
        while walked + turn_s * turns < profile["walking_s"]:
            row = clips.iloc[int(rng.integers(min(len(clips), 8)))]
            p = clip_piece(row["path"], "walk", rate)
            left = profile["walking_s"] - walked - turn_s * turns
            n = min(p.rot.shape[0], int(round(min(leg_s, left + 1.0) * rate)) + 1)
            pieces.append(Piece("walk", p.source + f"@0-{(n - 1) / rate:.2f}s", p.rot[:n], p.trans[:n],
                                travel_turn_rad=travel))  # fmt: skip
            travel = np.pi
            walked += (n - 1) / rate
            if walked + turn_s * (turns + 1) < profile["walking_s"]:
                side = "left" if rng.random() < 0.5 else "right"
                pieces.append(synthetic_piece(f"turn_spot_{side}", p.rot[n - 1], p.trans[n - 1], rate, 1, 0.5, 180.0))
                turns += 1
    used = 5.0 + walked + 2.0 * turns + 0.5 * len(pieces)  # T-pose, walking, turns, blends (and one more)
    rest = max(1.0, profile["duration_s"] + cfgt["twin.procedure.record_at_s"] - used)  # the BVH starts at record-on
    base = pieces[-1]
    pieces.append(hold_piece(base.rot[-1], base.trans[-1], rest, rate))
    joined = join(pieces, model, first.betas, first.gender, rate, blend, window, f"match:{subset}/{subject}")
    chosen = [p.source for p in pieces if p.source.startswith("clip:")]
    match = {"subject": f"{subset}/{subject}", "clips": len(chosen), "walking_s": walked, "turns": turns,
             "speed_m_s": float(clips["speed_m_s"].mean()) if len(clips) else None,
             "turn_deg_s": float(clips["turn_deg_s"].mean()) if len(clips) else None}  # fmt: skip
    return joined, model, joined.seq.source, match


# ----------------------------------------------------------------------------------------------- one simulation
def draw_theta(rng: np.random.Generator) -> dict:
    return {p.key: p.draw(rng) for p in PRIORS}


def split_theta(theta: dict, suit: str) -> tuple[dict, dict]:
    """(twin overrides, Step 1 overrides) of a draw for one suit version: only the priors that apply to it, and a
    prior keyed ``<key>@<suit>`` (one suit's own value of a shared parameter) under its bare key."""
    tw, s1 = {}, {}
    applies = {p.key for p in PRIORS if suit in p.suits}
    for k, v in theta.items():
        if k not in applies and k in {p.key for p in PRIORS}:
            continue
        key = k.split("@", 1)[0]
        if key.startswith("step1:"):
            s1[key[6:]] = v
        else:
            tw[key] = v
    return tw, s1


def _axial(a) -> float | None:
    return None if a is None else float((a + 90.0) % 180.0 - 90.0)


def summary(phi: dict) -> dict:
    """Named scalar statistics of a Φ dict (real or twin) for the distance and the predictive checks."""
    out = {"phi8": phi.get("phi8_pct")}
    p3 = phi.get("phi3_drawn_deg") or {}
    for s in ("LeftLeg", "RightLeg", "LeftUpLeg", "RightUpLeg"):
        out[f"phi3_{s}"] = _axial(p3.get(s))
    for s in ("LeftLeg", "RightLeg"):
        out[f"phi4_{s}"] = ((phi.get("phi4") or {}).get(s) or {}).get("slope_deg_per_min")
    out["phi5_scale"] = (phi.get("phi5") or {}).get("scale")
    out["phi6"] = phi.get("phi6_speed_cm_s")
    out["phi7"] = phi.get("phi7_deg")
    out["root_height"] = phi.get("root_height_cm")
    if phi.get("phi12"):
        pods = phi["phi12"]["pods"]
        drift = np.array([abs(v["drift_over_window_deg"]) for v in pods.values()])
        out["phi12_within"] = float(np.sum(drift <= 1.8))
        out["phi12_max_drift"] = float(drift.max())
    return out


AXIAL = {"phi3_LeftLeg", "phi3_RightLeg", "phi3_LeftUpLeg", "phi3_RightUpLeg"}


@functools.lru_cache(maxsize=2)
def _model(gender: str):
    from .smplh import load_model

    return load_model(gender)


def simulate_recording(rec_id: str, theta: dict, seed: int, profile: dict, program, step1: bool = False,
                       keep_bvh: bool = False) -> dict:  # fmt: skip
    """One matching session of ``rec_id`` under ``theta`` with the matching ``program`` (joined, label): its Φ
    summary, its slips, and with ``keep_bvh`` the twin BVH (for the classifier test)."""
    from .session import simulate_twin_session
    from .twin_stats import analyse_twin

    tw, s1 = split_theta(theta, profile["suit"])
    over = suit_overrides(profile["suit"]) | {"twin.suit.pods": profile["pods"]} | tw | (STEP1_THETA if step1 else {})
    cfgt = load_twin_config(overrides=over)
    cfg1 = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides(s1)
    joined, label = program
    s = simulate_twin_session(cfg1, cfgt, seed, joined, _model(joined.seq.gender), label)
    phi = analyse_twin(s, profile["kind"])
    return {"rec": rec_id, "seed": seed, "summary": summary(phi), "slips": s.plan.slips,
            "windows": window_features(_bvh_from_text(s.bvh)) if keep_bvh else None}


def matching_programs(profiles: dict) -> dict:
    """rec -> ((joined, label), match), built once (the seeds are fixed per recording)."""
    cfgt = load_twin_config()
    out = {}
    for k, rid in enumerate(sorted(profiles)):
        joined, _, label, match = matching_program(cfgt, profiles[rid], 1000 + k)
        out[rid] = ((joined, label), match)
    return out


# ----------------------------------------------------------------------------------------------- the fit
def _diff(k: str, a: float, b: float) -> float:
    d = a - b
    return (d + 90.0) % 180.0 - 90.0 if k in AXIAL else d


def _centre(k: str, vals: np.ndarray) -> float:
    """The median, or for an axial statistic (an axis, modulo 180°) half the angle of the mean doubled direction:
    values clustered at ±89° centre on the ±90° axis, not on 0°."""
    vals = vals[np.isfinite(vals)]
    if k not in AXIAL:
        return float(np.median(vals))
    a = np.deg2rad(2.0 * vals)
    return float(np.rad2deg(0.5 * np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))))


Z2_CAP = 25.0
"""The largest squared scaled difference one statistic adds to the distance (5 scale units; `assumption`)."""
TOLERANCE = {"phi3": 5.0, "phi4": 2.0, "phi5_scale": 0.03, "phi7": 5.0, "phi8": 3.0, "root_height": 3.0,
             "phi12_within": 1.0, "phi12_max_drift": 1.8}  # fmt: skip
"""The smallest scale of a statistic in the distance: the reproduction tolerances of Step 3 brief §5.3 (Φ3 ±5°, Φ5
±0.03, Φ7 ±5°, Φ8 ±3 points, heights ±3 cm, Φ12 one pod and its 1.8° hold; Φ4's ±7° over the 4 min of 13-41-17 as
±2°/min; Φ6 ±10 % of the recording's value). A twin cannot be asked to match a recording more closely than the
recordings' own analysis is reproduced (`assumption`)."""


def _tolerance(k: str, real_value: float) -> float:
    if k == "phi6":
        return 0.1 * abs(real_value)
    return TOLERANCE.get(k, TOLERANCE.get(k.split("_")[0], 0.0))


def distances(sims: pd.DataFrame, real: dict) -> pd.DataFrame:
    """Per draw, the RMS over statistics and recordings of (sim - real) / scale, §5.6 step 3: the scale is the
    statistic's MAD across the draws (Beaumont et al. 2002), but never below its reproduction tolerance
    (:data:`TOLERANCE`), and each squared term is capped at :data:`Z2_CAP`.

    ``sims``: one row per (draw, rec, statistic) with ``value``; ``real``: rec -> summary."""
    rows = []
    for (rec, k), g in sims.groupby(["rec", "stat"]):
        r = real.get(rec, {}).get(k)
        if r is None:
            continue
        vals = g["value"].to_numpy(dtype=float)
        med = _centre(k, vals)
        dev = np.array([_diff(k, v, med) for v in vals])
        mad = 1.4826 * np.nanmedian(np.abs(dev)) if np.isfinite(dev).any() else np.nan
        scale = max(mad if np.isfinite(mad) else 0.0, _tolerance(k, r))
        if scale <= 1e-9:
            continue
        z = np.array([_diff(k, v, r) for v in vals]) / scale
        rows.append(pd.DataFrame({"draw": g["draw"].to_numpy(), "rec": rec, "stat": k, "z": z}))
    z = pd.concat(rows, ignore_index=True)
    # a statistic counts at most Z2_CAP (5 MAD), as one the draw could not produce does: otherwise one the twin
    # cannot reach at all (round 1: Nov 4's 14° drift of one pod at rest, 500 MAD away) decides the ranking alone
    z["z2"] = np.where(np.isfinite(z["z"]), np.minimum(z["z"] ** 2, Z2_CAP), Z2_CAP)
    return z.groupby("draw")["z2"].mean().pow(0.5).rename("distance").reset_index()


def posterior(thetas: pd.DataFrame, dist: pd.DataFrame, keep: float) -> tuple[pd.DataFrame, dict]:
    """The closest ``keep`` share of draws, and per parameter its prior range, median, 90 % range and narrowing."""
    d = thetas.merge(dist, on="draw").sort_values("distance")
    n = max(5, int(round(keep * len(d))))
    post = d.head(n)
    report = {}
    for p in PRIORS:
        if p.key not in post:
            continue
        x = post[p.key].to_numpy(dtype=float)
        if p.kind == "choice":
            vals, counts = np.unique(x, return_counts=True)
            report[p.key] = {"kind": "choice", "posterior": {int(v): int(c) for v, c in zip(vals, counts, strict=True)},
                             "prior": list(p.choices)}  # fmt: skip
            continue
        lo, hi = (math.log(p.lo), math.log(p.hi)) if p.kind == "log_uniform" else (p.lo, p.hi)
        xs = np.log(x) if p.kind == "log_uniform" else x
        q5, q50, q95 = np.percentile(xs, [5, 50, 95])
        back = (lambda v: float(math.exp(v))) if p.kind == "log_uniform" else float  # noqa: E731
        report[p.key] = {"kind": p.kind, "prior": [p.lo, p.hi], "median": back(q50), "p5": back(q5),
                         "p95": back(q95), "width_share": float((q95 - q5) / (0.9 * (hi - lo)))}  # fmt: skip
    return post, report


@dataclass
class CalibrationRun:
    profiles: dict
    real: dict
    thetas: pd.DataFrame = field(default_factory=pd.DataFrame)
    sims: pd.DataFrame = field(default_factory=pd.DataFrame)
    matches: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------- parallel runs
_PROGRAMS: dict = {}


def _init(programs: dict) -> None:
    _PROGRAMS.update(programs)


def _task(args):
    """One simulation; a failure is recorded (and scored as far from everything), never raised, so that one odd
    session cannot end a run of hundreds."""
    rec, theta, seed, profile, step1, keep_bvh, draw = args
    try:
        out = simulate_recording(rec, theta, seed, profile, _PROGRAMS[rec], step1, keep_bvh)
    except Exception as e:  # noqa: BLE001 - recorded in the report
        out = {"rec": rec, "seed": seed, "summary": {}, "slips": None, "windows": None,
               "error": f"{type(e).__name__}: {e}"}
    out["draw"] = draw
    return out


TASKS_PER_WORKER = 25
"""A worker is replaced after this many sessions: arrays freed in a process stay in its memory on macOS, and round
1's eight workers grew to 1-2 GB each and swapped. (``multiprocessing.Pool``'s ``maxtasksperchild``: Python 3.11's
``ProcessPoolExecutor(max_tasks_per_child=...)`` did not replace its workers, and round 2's first start ran down to
one.)"""


def run_many(tasks: list, programs: dict, workers: int) -> list:
    """Run (rec, theta, seed, profile, step1, keep_bvh, draw) tasks over ``workers`` processes, in order."""
    import multiprocessing as mp

    if workers <= 1:
        _init(programs)
        return [_task(t) for t in tasks]
    with mp.get_context("spawn").Pool(processes=workers, initializer=_init, initargs=(programs,),
                                      maxtasksperchild=TASKS_PER_WORKER) as pool:  # fmt: skip
        return list(pool.imap(_task, tasks, chunksize=1))


def run_checkpointed(tasks: list, programs: dict, workers: int, path: Path | None, log=print,
                     batch: int | None = None) -> list:  # fmt: skip
    """:func:`run_many` in batches (a fresh pool each), the results so far pickled to ``path`` after every batch; run
    again with the same tasks, it starts after the last finished batch. An interrupted run loses one batch."""
    import pickle

    key = [(t[0], int(t[2]), int(t[6]), bool(t[4])) for t in tasks]  # rec, seed, draw, Step 1: the task's identity
    done: list = []
    if path is not None and path.is_file():
        saved = pickle.loads(path.read_bytes())
        if saved.get("key") == key:
            done = saved["results"]
            log(f"  resuming after {len(done)} of {len(tasks)} sessions ({path.name})")
    batch = batch or max(1, 25 * max(workers, 1))
    while len(done) < len(tasks):
        part = tasks[len(done) : len(done) + batch]
        done += run_many(part, programs, workers)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(pickle.dumps({"key": key, "results": done}))
            tmp.replace(path)
        log(f"  {len(done)} of {len(tasks)} sessions")
    return done


def seed_of(draw: int, rec: str, salt: int = 0) -> int:
    return int(stream(draw * 7919 + salt, "twin.calibration." + rec).integers(2**31))


def abc_tasks(profiles: dict, draws: int, seed: int) -> tuple[pd.DataFrame, list]:
    rng = stream(seed, "twin.calibration.prior")
    thetas, tasks = [], []
    for j in range(draws):
        th = draw_theta(rng)
        thetas.append({"draw": j, **th})
        for rec in CALIBRATE:
            tasks.append((rec, th, seed_of(j, rec), profiles[rec], False, False, j))
    return pd.DataFrame(thetas), tasks


def results_table(results: list) -> pd.DataFrame:
    rows = []
    for r in results:
        for k, v in r["summary"].items():
            rows.append({"draw": r["draw"], "rec": r["rec"], "stat": k, "value": np.nan if v is None else float(v)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- the gap
def coverage(pred: pd.DataFrame, real: dict) -> pd.DataFrame:
    """Per recording and core statistic: the real value's quantile in the predictive values, and inside the central
    90 % interval or not (axial statistics compared modulo 180°, about the predictive median)."""
    rows = []
    for (rec, k), g in pred[pred["stat"].isin(CORE)].groupby(["rec", "stat"]):
        r = real.get(rec, {}).get(k)
        vals = g["value"].dropna().to_numpy(dtype=float)
        if r is None or vals.size < 5:
            continue
        if k in AXIAL:
            med = _centre(k, vals)
            vals = np.array([_diff(k, v, med) for v in vals])
            r = _diff(k, r, med)
        q = float(np.mean(vals < r) + 0.5 * np.mean(vals == r))
        lo, hi = np.percentile(vals, [5, 95])
        rows.append({"rec": rec, "stat": k, "real": float(r), "p5": float(lo), "p95": float(hi), "quantile": q,
                     "inside": bool(lo <= r <= hi), "n": int(vals.size)})  # fmt: skip
    return pd.DataFrame(rows, columns=["rec", "stat", "real", "p5", "p95", "quantile", "inside", "n"])


WINDOW_S = 10.0
FEATURE_BONES = ("Hips", "Spine", "Head", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftUpLeg",
                 "RightUpLeg", "LeftLeg", "RightLeg")
"""The pod-carrying bones every recording has (the old suit had no hand or foot pods)."""


def window_features(bvh) -> np.ndarray:
    """(windows, 39) features of 10 s windows of a BVH's drawn rotations (never the truth): per bone of
    :data:`FEATURE_BONES`, the 10th, 50th and 90th percentiles of log10(1e-4 + its frame-to-frame rotation in
    degrees); the root's median horizontal speed and its share of frames moving under 1 mm; and per shin the swing
    axis over the window's walking frames as (cos 2a, sin 2a), zeros when it does not walk."""
    from ..real import stats as st
    from ..real.bvh import forward_kinematics

    fk = forward_kinematics(bvh)
    d = st.series_from_bvh(bvh, fk)
    walk = st.walking_mask(d)
    n = int(round(WINDOW_S / bvh.frame_time_s))
    ang = {b: np.r_[0.0, st.frame_angles_deg(st.recorded_rotations(bvh, b))] for b in FEATURE_BONES}
    step = np.r_[0.0, np.linalg.norm(np.diff(d.root[:, :2], axis=0), axis=1)]
    out = []
    for a in range(0, bvh.n_frames - n + 1, n):
        sl = slice(a, a + n)
        f = []
        for b in FEATURE_BONES:
            f += list(np.percentile(np.log10(1e-4 + ang[b][sl]), [10, 50, 90]))
        f += [float(np.median(step[sl]) / bvh.frame_time_s), float(np.mean(step[sl] < 1e-3))]
        w = np.zeros(bvh.n_frames, bool)
        w[sl] = walk[sl]
        for s in ("LeftLeg", "RightLeg"):
            ax = st.swing_axis_deg(d, s, w) if w.sum() >= 30 else np.nan
            f += [0.0, 0.0] if not np.isfinite(ax) else [math.cos(math.radians(2 * ax)), math.sin(math.radians(2 * ax))]
        out.append(f)
    return np.asarray(out, dtype=float).reshape(-1, 3 * len(FEATURE_BONES) + 6)


def _bvh_from_text(text: str):
    import tempfile

    from ..real.bvh import read_bvh

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.bvh"
        p.write_text(text, encoding="utf-8", newline="\n")
        return read_bvh(p)


def logistic_fit(x: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 50) -> np.ndarray:
    """L2-regularised logistic regression by Newton's method (IRLS); returns weights with the intercept last."""
    xb = np.c_[x, np.ones(len(x))]
    w = np.zeros(xb.shape[1])
    reg = l2 * np.r_[np.ones(x.shape[1]), 0.0]
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-xb @ w))
        g = xb.T @ (p - y) + reg * w
        h = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(reg + 1e-9)
        step = np.linalg.solve(h, g)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """The area under the ROC curve: P(score of a 1 > score of a 0), ties counting a half (Mann-Whitney)."""
    from scipy.stats import rankdata

    r = rankdata(score)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def classifier_test(real: dict, sim: dict, boot: int = 1000, seed: int = 0) -> dict:
    """Leave one recording out: train on the other recordings' real windows (0) and the twin windows matched to them
    (1), score the held-out recording's; the pooled out-of-fold AUC and a bootstrap 90 % interval."""
    scores, labels, recs = [], [], []
    real = {r: v for r, v in real.items() if r in sim and len(sim[r])}
    for held in real:
        tr_x = np.vstack([real[r] for r in real if r != held] + [sim[r] for r in sim if r != held])
        tr_y = np.r_[np.zeros(sum(len(real[r]) for r in real if r != held)),
                     np.ones(sum(len(sim[r]) for r in sim if r != held))]  # fmt: skip
        mu, sd = tr_x.mean(axis=0), tr_x.std(axis=0) + 1e-9
        w = logistic_fit((tr_x - mu) / sd, tr_y)
        te = np.vstack([real[held], sim[held]])
        s_ = np.c_[(te - mu) / sd, np.ones(len(te))] @ w
        scores.append(s_)
        labels.append(np.r_[np.zeros(len(real[held])), np.ones(len(sim[held]))])
        recs += [held] * len(te)
    sc, y = np.concatenate(scores), np.concatenate(labels)
    rng = np.random.default_rng(seed)
    i0, i1 = np.flatnonzero(y == 0), np.flatnonzero(y == 1)
    bs = []
    for _ in range(boot):  # resample the windows within each class
        idx = np.r_[rng.choice(i0, i0.size), rng.choice(i1, i1.size)]
        bs.append(auc(sc[idx], y[idx]))
    lo, hi = np.percentile(bs, [5, 95])
    return {"auc": auc(sc, y), "p5": float(lo), "p95": float(hi), "windows_real": int(i0.size),
            "windows_sim": int(i1.size)}


def gate_gt(cov_twin: pd.DataFrame, cov_step1: pd.DataFrame, auc_twin: dict, auc_step1: dict) -> dict:
    """Gate G-T (§5.7): the twin covers at least as many core statistics as Step 1's simulator and its AUC is not
    higher."""
    n_t, n_s = int(cov_twin["inside"].sum()), int(cov_step1["inside"].sum())
    return {"twin_inside": n_t, "step1_inside": n_s, "of": int(len(cov_twin)), "twin_auc": auc_twin["auc"],
            "step1_auc": auc_step1["auc"], "passes": bool(n_t >= n_s and auc_twin["auc"] <= auc_step1["auc"])}


def run_calibration(real_summary: dict, draws: int, keep: float, n_pred: int, workers: int, seed: int = 1,
                    log=print, out_dir: Path | None = None, resume: Path | None = None) -> dict:  # fmt: skip
    """The whole of §5.6-5.7: the fit, the predictive checks for the twin and Step 1, the classifier test, G-T.
    ``resume``: a folder holding an earlier run's ``thetas.parquet`` and ``sims.parquet`` (the fit's simulations,
    same priors): the distances, the posterior and everything after are computed again from them."""
    import time

    from ..real.bvh import read_bvh

    t0 = time.perf_counter()
    profiles = real_profiles(real_summary)
    real = {rid: summary(r) for rid, r in real_summary["results"].items()}
    built = matching_programs(profiles)
    programs = {k: v[0] for k, v in built.items()}
    log(f"programs built ({time.perf_counter() - t0:.0f} s)")
    if resume is not None:
        thetas = pd.read_parquet(Path(resume) / "thetas.parquet")
        sims = pd.read_parquet(Path(resume) / "sims.parquet")
        if sorted(c for c in thetas.columns if c != "draw") != sorted(p.key for p in PRIORS):
            raise ValueError(f"{resume}: its draws are not of these priors")
        draws, res = len(thetas), []
        log(f"resumed {draws} draws from {resume}")
    else:
        thetas, tasks = abc_tasks(profiles, draws, seed)
        res = run_checkpointed(tasks, programs, workers, out_dir / "fit_partial.pkl" if out_dir else None, log)
        sims = results_table(res)
    dist = distances(sims, real)
    post, report = posterior(thetas, dist, keep)
    log(f"fit: {draws} draws x {len(CALIBRATE)} recordings ({time.perf_counter() - t0:.0f} s)")
    if out_dir is not None:  # checkpoint the expensive part
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in (("thetas", thetas), ("sims", sims), ("distances", dist)):
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
    # the posterior predictive, every recording (calibration and check), and Step 1's simulator beside it
    rng = stream(seed, "twin.calibration.predictive")
    post_th = post[[p.key for p in PRIORS]].to_dict("records")
    pred_tasks, s1_tasks = [], []
    for j in range(n_pred):
        th = post_th[int(rng.integers(len(post_th)))]
        for rec in sorted(profiles):
            pred_tasks.append((rec, th, seed_of(10_000 + j, rec), profiles[rec], False, True, j))
            s1_tasks.append((rec, {}, seed_of(20_000 + j, rec), profiles[rec], True, True, j))
    pred = run_checkpointed(pred_tasks, programs, workers, out_dir / "pred_partial.pkl" if out_dir else None, log)
    step1 = run_checkpointed(s1_tasks, programs, workers, out_dir / "step1_partial.pkl" if out_dir else None, log)
    log(f"predictive: {len(pred)} twin and {len(step1)} Step 1 sessions ({time.perf_counter() - t0:.0f} s)")
    pred_t, step1_t = results_table(pred), results_table(step1)
    if out_dir is not None:
        pred_t.to_parquet(out_dir / "predictive_twin.parquet", index=False)
        step1_t.to_parquet(out_dir / "predictive_step1.parquet", index=False)
    cov_t, cov_s = coverage(pred_t, real), coverage(step1_t, real)
    real_w = {rid: window_features(read_bvh(real_suit_dir() / REC[rid].bvh)) for rid in profiles}

    def windows(rs):
        out: dict = {}
        for r in rs:
            if r.get("windows") is not None and len(r["windows"]):
                out.setdefault(r["rec"], []).append(r["windows"])
        return {k: np.vstack(v) for k, v in out.items()}

    auc_t, auc_s = classifier_test(real_w, windows(pred)), classifier_test(real_w, windows(step1))
    gate = gate_gt(cov_t, cov_s, auc_t, auc_s)
    log(f"classifier and gate ({time.perf_counter() - t0:.0f} s)")
    failed = [{"rec": r["rec"], "draw": r["draw"], "seed": r["seed"], "error": r["error"]}
              for r in res + pred + step1 if r.get("error")]  # fmt: skip
    return {"failed": failed, "profiles": profiles, "matches": {k: v[1] for k, v in built.items()}, "real": real,
            "priors": [p.__dict__ | {"choices": list(p.choices), "suits": list(p.suits)} for p in PRIORS],
            "posterior": report, "kept": int(len(post)), "draws": draws, "keep": keep,
            "best_distance": float(post["distance"].min()), "median_distance": float(dist["distance"].median()),
            "coverage_twin": cov_t.to_dict("records"), "coverage_step1": cov_s.to_dict("records"),
            "auc_twin": auc_t, "auc_step1": auc_s, "gate": gate, "seconds": time.perf_counter() - t0,
            "tables": {"thetas": thetas, "sims": sims, "distances": dist}}  # fmt: skip
