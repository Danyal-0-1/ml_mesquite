"""The training data, ``twin_v1`` (Step 3 brief §6): ``python -m mesqsim twin-dataset`` and ``verify-dataset``.

**What one session is** (§6.1): a twin session of one subject's clips, stored whole on its 32 Hz step grid under
``/sessions/<id>/`` of an HDF5 shard: the inputs of :mod:`.features` (``x_pod``, ``x_joint``, ``x_phone``,
``x_pod_phone``, ``state_pod``, ``state_phone``), the targets of :mod:`.targets`, and ``strata/`` (time since boot,
motion type, ``transition``, slips). The group's attributes are ``subject``, ``split``, ``suit_version`` and
``conditions``. Cutting windows is the loader's job, which Danyal writes.

**What varies** (§6.2): θ is drawn from the calibration's posterior, widened by ``twin_dataset.widen`` in log scale
(domain randomisation, Tobin et al. 2017), and every twin parameter with a sweep that the fit did not touch is drawn
from its sweep; each session's conditions are mixed: both suit versions, HipsAlt present or absent, the slips at their
drawn probabilities, the protocol program for KIT subject 3 now and then.

**How much** (§6.3): every subject of a split, one composition of their clips (the program) with ``variants``
corruption draws of it; the nested parts 1/16 ... 1 of the training subjects are marked in ``index.parquet``;
``--extend <factor>`` multiplies the training sessions deterministically (more corruption draws of the same
composition first, up to :data:`DRAWS_PER_COMPOSITION`, then new compositions) and never changes a stored one. The
validation and test sessions never change, so scores before and after an extension compare.

**Storage** (§6.4): ``$MESQ_OUTPUT_DIR/datasets/<name>/``: ``shards/<split>_<nnnnn>.h5`` (gzip, chunked in time),
``index.parquet``, ``meta.json``, ``MANIFEST.csv`` (every file's name, size and SHA-256; a copy goes to
``manifests/<name>.csv``) and ``README.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..sim.rng import stream

__all__ = ["MOTION_CODES", "SLIP_BITS", "PARTS", "DRAWS_PER_COMPOSITION", "plan_dataset", "draw_conditions",
           "build_session", "write_session_group", "manifest_rows", "verify", "windows", "coverage", "generate"]

MOTION_CODES = {"preparation": 0, "tpose": 1, "still": 2, "transition": 3, "walk": 4, "turn": 5, "squat": 6,
                "kick": 7, "arm": 8, "jump": 9, "bend": 10, "stairs": 11, "kneel": 12, "other": 13, "twist": 14,
                "leg_swing": 15, "leg_side": 16, "arm_forward": 17, "arm_side": 18, "turn_spot": 19}
"""strata/motion: the program piece's kind at each step (``preparation`` before the program starts)."""
SLIP_BITS = {"restarted": 1, "zero_at_start": 2, "camera_mismatch": 4, "tree_mismatch": 8}
PARTS = (16, 8, 4, 2, 1)
"""Nested parts of the training subjects: a subject in part 1/16 is in every larger part."""
DRAWS_PER_COMPOSITION = 8
"""``twin_dataset.draws_per_composition`` (`assumption`): corruption draws of one composition of a subject's clips
before the next composition (§6.3: "more corruption draws first, then new compositions of the clips")."""


def plan_dataset(split: str, variants: int, seed: int, first_variant: int = 0, max_subjects: int | None = None,
                 per_composition: int = DRAWS_PER_COMPOSITION) -> pd.DataFrame:  # fmt: skip
    """One row per (subject, variant) of ``split``: its id, the seed of its composition (the program: which clips,
    in which order, how long) and the seed of its corruption draw (everything else). Variant v is draw
    v mod ``per_composition`` of composition v // ``per_composition``. Deterministic, so ``--extend`` only appends.
    ``max_subjects`` keeps the first subjects only (a pilot or a test)."""
    from ..twin.program import training_subjects

    rows = []
    for subset, subject in training_subjects(split)[:max_subjects]:
        base = f"{split}_{subset}_{subject}".replace("/", "-")
        for v in range(first_variant, first_variant + variants):
            c = v // per_composition
            sid = f"{base}_v{v}"
            rows.append({"id": sid, "split": split, "subset": subset, "subject": subject, "variant": v,
                         "composition": c,
                         "program_seed": int(stream(seed, f"twin_dataset.program.{base}_c{c}").integers(2**31)),
                         "seed": int(stream(seed, "twin_dataset." + sid).integers(2**31))})  # fmt: skip
    return pd.DataFrame(rows)


def nested_parts(subjects: list[tuple[str, str]], seed: int) -> dict:
    """subject -> the smallest part (16, 8, 4, 2 or 1 meaning 1/16 ... all) it belongs to."""
    rng = stream(seed, "twin_dataset.parts")
    order = list(rng.permutation(len(subjects)))
    rank = {subjects[k]: r for r, k in enumerate(order)}
    out = {}
    for s in subjects:
        frac = (rank[s] + 1) / len(subjects)
        out[s] = next(p for p in PARTS if frac <= 1.0 / p) if frac <= 1.0 else 1
    return out


def draw_conditions(cfgt, posterior: pd.DataFrame, priors, seed: int, widen: float, shares: dict) -> dict:
    """The session's twin overrides: θ from the posterior (a random kept draw, each continuous parameter moved in log
    scale, or linear for a uniform prior, by N(0, (widen · its posterior spread)²) and clipped to the prior), every
    unfitted swept twin parameter from its sweep, and the session's conditions."""
    from ..twin.calibration import split_theta
    from ..twin.suits import OLD_SUIT_PODS, suit_overrides

    rng = stream(seed, "twin_dataset.conditions")
    row = posterior.iloc[int(rng.integers(len(posterior)))]
    theta, over = {}, {}
    for p in priors:
        x = row[p.key]
        if p.kind == "choice":
            v = int(x)
        else:
            col = posterior[p.key].to_numpy(dtype=float)
            if p.kind == "log_uniform":
                sd = float(np.std(np.log(col))) * widen
                v = float(np.clip(math.exp(math.log(x) + rng.normal(0.0, sd)), p.lo, p.hi))
            else:
                sd = float(np.std(col)) * widen
                v = float(np.clip(x + rng.normal(0.0, sd), p.lo, p.hi))
        theta[p.key] = v
    fitted = {p.key.split("@", 1)[0] for p in priors}
    for key, leaf in cfgt.leaves():
        if "sweep" in leaf and key not in fitted and not key.startswith("twin.physics"):
            over[key] = leaf["sweep"][int(rng.integers(len(leaf["sweep"])))]
    suit = "suit_2026_11_13" if rng.random() < shares["old_suit"] else "suit_2026_11_14"
    sov = suit_overrides(suit)
    pods = list(sov.get("twin.suit.pods", range(15)))
    hipsalt = not (rng.random() < shares["no_hipsalt"])
    if not hipsalt:
        pods = [i for i in pods if i != 2]
    tw, step1 = split_theta(theta, suit)  # as the calibration applies them: only the suit's own priors
    over = sov | over | tw | {"twin.suit.pods": pods}
    protocol = bool(rng.random() < shares["protocol"])
    return {"twin": over, "step1": step1, "suit": suit, "hipsalt": hipsalt, "protocol": protocol,
            "old_pods": pods == OLD_SUIT_PODS}  # fmt: skip


def build_session(row: dict, cond: dict, cfgt_base, cfg1_base, length_min: float | None = None,
                  keep_session: bool = False, page: bool = False) -> dict:  # fmt: skip
    """Simulate one dataset session and compute its stored arrays (and keep the simulated session if asked). The page
    twin runs only with ``page`` (the store holds no BVH; the page-level test needs it)."""
    from ..learn.features import features
    from ..learn.targets import session_targets
    from ..step0 import decode
    from ..twin.program import PROTOCOL_SUBJECT, make_program
    from ..twin.session import simulate_twin_session

    cfgt = cfgt_base.with_overrides(cond["twin"])
    cfg1 = cfg1_base.with_overrides(cond["step1"]) if cond["step1"] else cfg1_base
    proto = cond["protocol"] and (row["subset"], row["subject"]) == PROTOCOL_SUBJECT
    joined, model, label = make_program(cfgt, "protocol" if proto else "random", row["program_seed"],
                                        None if proto else f"{row['subset']}/{row['subject']}", length_min)
    s = simulate_twin_session(cfg1, cfgt, row["seed"], joined, model, label, page=page)
    st = s.labels_steps
    t = st["t_s"].to_numpy()
    f = features(decode(s.frames, s.session), s.phone, t + s.meta["plan"]["host_epoch_s"],
                 cfg1["geometry.clock_window_s"], cfg1["geometry.clock_block_s"])  # fmt: skip
    y = session_targets(s.labels_joints, st)
    slip = np.zeros(t.size, np.uint8)
    slip |= np.where(st["slip_restarted"].to_numpy(), SLIP_BITS["restarted"], 0).astype(np.uint8)
    slip |= np.where(st["slip_zero_at_start"].to_numpy(), SLIP_BITS["zero_at_start"], 0).astype(np.uint8)
    slip |= np.where(st["slip_camera_mismatch_deg"].to_numpy() != 0, SLIP_BITS["camera_mismatch"], 0).astype(np.uint8)
    slip |= np.where(st["slip_tree_mismatch"].to_numpy(), SLIP_BITS["tree_mismatch"], 0).astype(np.uint8)
    boot = min(p.boot_s for p in s.plan.pods)
    arrays = {"t_s": t, "x_pod": f.x_pod, "x_joint": f.x_joint, "x_phone": f.x_phone, "x_pod_phone": f.x_pod_phone,
              "state_pod": f.state_pod, "state_phone": f.state_phone, **y,
              "strata/time_since_boot_s": (t - boot).astype(np.float32),
              "strata/motion": np.array([MOTION_CODES.get(m, MOTION_CODES["other"]) for m in st["motion"]], np.uint8),
              "strata/transition": st["transition"].to_numpy(dtype=bool), "strata/slip": slip}  # fmt: skip
    attrs = {"subject": f"{row['subset']}/{row['subject']}", "split": row["split"], "suit_version": cond["suit"],
             "conditions": json.dumps({"program": label, "hipsalt": cond["hipsalt"], "slips": s.plan.slips,
                                       "wire": cfgt["twin.wire.mode"], "fusion": cfgt["twin.fusion.mode"],
                                       "twin": cond["twin"], "step1": cond["step1"]}, sort_keys=True, default=str)}
    info = {"steps": int(t.size), "length_s": float(t[-1] - t[0]) if t.size else 0.0, "program": label,
            "suit_version": cond["suit"], "hipsalt": cond["hipsalt"], "wire": cfgt["twin.wire.mode"],
            "slips": json.dumps(s.plan.slips, sort_keys=True, default=str)}  # fmt: skip
    out = {"arrays": arrays, "attrs": attrs, "info": info}
    if keep_session:
        out["session"], out["features"] = s, f
    return out


def write_session_group(h5, sid: str, built: dict, chunk_steps: int = 2048) -> None:
    g = h5.create_group(f"sessions/{sid}")
    for k, v in built["arrays"].items():
        v = np.asarray(v)
        chunks = (min(chunk_steps, max(1, v.shape[0])),) + v.shape[1:] if v.ndim and v.shape[0] else None
        g.create_dataset(k, data=v, chunks=chunks, compression="gzip", compression_opts=4, shuffle=True)
    for k, v in built["attrs"].items():
        g.attrs[k] = v


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_rows(root: Path) -> pd.DataFrame:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "MANIFEST.csv")
    return pd.DataFrame({"file": [str(p.relative_to(root)) for p in files],
                         "size_bytes": [p.stat().st_size for p in files], "sha256": [_sha256(p) for p in files]})


PER_STEP_ARRAYS = ("t_s", "x_pod", "x_joint", "x_phone", "x_pod_phone", "state_pod", "state_phone", "y_delta",
                   "valid_delta", "y_eps", "valid_eps", "strata/time_since_boot_s", "strata/motion",
                   "strata/transition", "strata/slip")  # fmt: skip


def verify(root: Path) -> dict:
    """Re-hash every file against MANIFEST.csv (and no file missing from it); no subject (its subset and id) in two
    splits; session ids unique; the nested parts only on training subjects; every session inside one shard of its own
    split, its steps contiguous on the 32 Hz grid and every per-step array of that length, so a window cut by step
    index (see :func:`windows`) never crosses a session or a split and its inputs end with its loss region."""
    import h5py

    man = pd.read_csv(root / "MANIFEST.csv")
    bad = [r.file for r in man.itertuples() if not (root / r.file).is_file() or _sha256(root / r.file) != r.sha256]
    listed = set(man["file"])
    unlisted = sorted(str(p.relative_to(root)) for p in root.rglob("*")
                      if p.is_file() and p.name != "MANIFEST.csv" and str(p.relative_to(root)) not in listed)
    idx = pd.read_parquet(root / "index.parquet")
    who = idx["subset"].astype(str) + "/" + idx["subject"].astype(str)  # ids repeat across subsets: CMU and KIT 10
    two = idx.assign(who=who).groupby("who")["split"].nunique()
    mixed = sorted(two[two > 1].index)
    problems = [f"{k}: id twice" for k, n in idx["id"].value_counts().items() if n > 1]
    if not idx.loc[idx["split"] == "train", "part"].isin(PARTS).all() or (idx.loc[idx["split"] != "train", "part"]
                                                                          != 0).any():  # fmt: skip
        problems.append("parts")
    for shard, g in idx.groupby("shard"):
        with h5py.File(root / shard, "r") as h5:
            if set(h5["sessions"]) != set(g["id"]):
                problems.append(f"{shard}: sessions differ from the index")
            for r in g.itertuples():
                grp = h5[f"sessions/{r.id}"]
                t = np.asarray(grp["t_s"])
                if grp.attrs["split"] != r.split or not shard.split("/")[-1].startswith(r.split + "_"):
                    problems.append(f"{r.id}: split")
                if t.size != r.steps or (t.size > 1 and not np.allclose(np.diff(t), 1.0 / 32.0, atol=1e-9)):
                    problems.append(f"{r.id}: steps")
                for k in PER_STEP_ARRAYS:
                    if grp[k].shape[0] != t.size:
                        problems.append(f"{r.id}: {k} length")
    return {"files": len(man), "bad_hashes": bad + unlisted, "subjects_in_two_splits": mixed, "problems": problems,
            "sessions": len(idx), "ok": not bad and not unlisted and not mixed and not problems}  # fmt: skip


# ----------------------------------------------------------------------------------------------- generation
DATASET_DEFAULTS = {"variants": {"train": 4, "val": 2, "test": 2}, "sessions_per_shard": 8, "widen": 1.5,
                    "shares": {"old_suit": 0.5, "no_hipsalt": 0.2, "protocol": 1.0}, "max_gb": 20.0,
                    "min_windows": 1000}  # fmt: skip
"""``twin_dataset`` (`assumption`): corruption draws per subject and split (§6.3 starts at 4), the widening of the
posterior in log scale (§6.2), the shares of the old suit, of sessions without HipsAlt and of the protocol program
among its subject's sessions (all of them: KIT subject 3 is the only subject whose clips make the protocol, and its
twists, swings, raises and turns on the spot occur nowhere else), the disk budget (Step 3 brief rule 5: ask before
more than 20 GB) and the coverage rule's least number of training windows per stratum (§6.2)."""
BYTES_PER_STEP = 800
"""Stored bytes per 32 Hz step (`measured`, the pilot of 2026-11-21: 624 for the old suit's 11 pods, 885 to 978 for
the new suit's 14 or 15; about 790 at the default shares), for the projection made before anything is written."""
OVERHEAD_S = 110.0
"""A session's length beyond its program's drawn length (`measured`, the same pilot: 63 to 149 s; the procedure
before the program, and the last clip running past the drawn length)."""


def planned_steps(plan: pd.DataFrame, cfgt, length_min: float | None = None) -> float:
    """The 32 Hz steps a plan will hold, from each composition's drawn length (as
    :func:`mesqsim.twin.program.make_program` draws it) and :data:`OVERHEAD_S`."""
    lo, hi = cfgt["twin.program.length_min"]
    minutes = [float(stream(int(s), "twin.program").uniform(lo, hi)) if length_min is None else float(length_min)
               for s in plan["program_seed"]]  # fmt: skip
    return float(sum((60.0 * m + OVERHEAD_S) * 32.0 for m in minutes))


def projected_gb(plan: pd.DataFrame, cfgt, length_min: float | None = None) -> float:
    """The disk a plan will take: :func:`planned_steps` at :data:`BYTES_PER_STEP`."""
    return planned_steps(plan, cfgt, length_min) * BYTES_PER_STEP / 1e9


def _shard_task(args):
    import h5py

    from ..config import load_config
    from ..paths import REPO_ROOT
    from ..twin.config import load_twin_config

    shard_path, rows, posterior, priors, widen, shares, length_min = args
    cfgt, cfg1 = load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml")
    infos = []
    tmp = Path(str(shard_path) + ".partial")
    with h5py.File(tmp, "w") as h5:
        for row in rows:
            cond = draw_conditions(cfgt, posterior, priors, row["seed"], widen, shares)
            built = build_session(row, cond, cfgt, cfg1, length_min)
            write_session_group(h5, row["id"], built)
            infos.append(row | built["info"] | {"shard": f"shards/{Path(shard_path).name}"})
        h5.attrs["index_rows"] = json.dumps(infos, sort_keys=True, default=_json_default)  # a restart reads them back
    tmp.rename(shard_path)
    return infos


def _json_default(x):
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def _finished_shard(path: Path) -> list | None:
    """The index rows of a shard that a previous run finished (its ``.partial`` was renamed), else None."""
    import h5py

    if not path.is_file():
        return None
    with h5py.File(path, "r") as h5:
        rows = h5.attrs.get("index_rows")
    return json.loads(rows) if rows is not None else None


def generate(root: Path, posterior: pd.DataFrame, priors, seed: int, workers: int, opts: dict | None = None,
             extend: float = 0.0, log=print) -> pd.DataFrame:  # fmt: skip
    """Generate the dataset at ``root``, or, with ``extend`` (a factor > 1) and a dataset already there, multiply its
    training sessions by that factor (the validation and test sessions are never extended). ``opts`` may set
    ``max_subjects`` and ``length_min`` for a pilot or a test. Refuses, before writing, a plan projected over
    ``max_gb``, and stops if the shards written so far project over it."""
    from ..twin.config import load_twin_config
    from ..twin.program import training_subjects

    o = DATASET_DEFAULTS | (opts or {})
    old = pd.read_parquet(root / "index.parquet") if (root / "index.parquet").is_file() else pd.DataFrame()
    if not old.empty and extend <= 1.0:
        log(f"{root} exists ({len(old)} sessions); --extend <factor> > 1 adds to it")
        return old
    plans = []
    for split, n in o["variants"].items():
        if old.empty:
            first, count = 0, n
        elif split == "train":
            have = int(old.loc[old["split"] == split, "variant"].max() + 1)
            first, count = have, int(math.ceil((extend - 1.0) * have))
        else:
            continue
        if count:
            plans.append(plan_dataset(split, count, seed, first_variant=first, max_subjects=o.get("max_subjects")))
    plan = pd.concat(plans, ignore_index=True)
    if o.get("length_min") is not None:
        plan["length_min"] = float(o["length_min"])
    have_gb = sum(p.stat().st_size for p in (root / "shards").glob("*.h5")) / 1e9 if not old.empty else 0.0
    cfgt = load_twin_config()
    steps_planned = planned_steps(plan, cfgt, o.get("length_min"))
    proj = steps_planned * BYTES_PER_STEP / 1e9
    log(f"{len(plan)} new sessions, about {proj:.1f} GB (with {have_gb:.1f} GB already there)")
    if have_gb + proj > o["max_gb"]:
        raise RuntimeError(f"the dataset would take about {have_gb + proj:.1f} GB, over the {o['max_gb']} GB budget "
                           "(Step 3 brief rule 5: ask first, then raise --max-gb)")  # fmt: skip
    (root / "shards").mkdir(parents=True, exist_ok=True)
    tasks, k0 = [], 0 if old.empty else len(list((root / "shards").glob("*.h5")))
    for split, g in plan.groupby("split", sort=False):
        rows = g.to_dict("records")
        for j in range(0, len(rows), o["sessions_per_shard"]):
            path = root / "shards" / f"{split}_{k0 + len(tasks):05d}.h5"
            tasks.append((path, rows[j : j + o["sessions_per_shard"]], posterior, priors, o["widen"], o["shares"],
                          o.get("length_min")))  # fmt: skip
    log(f"{len(plan)} sessions in {len(tasks)} shards")
    infos, new = [], {Path(t[0]).name for t in tasks}
    import multiprocessing as mp

    # a fresh process per shard (multiprocessing.Pool's maxtasksperchild: Python 3.11's ProcessPoolExecutor did not
    # replace workers retired by max_tasks_per_child), so no worker carries one long session's memory into the next
    finished = {Path(t[0]).name: _finished_shard(Path(t[0])) for t in tasks}  # an interrupted run's shards stay
    todo = [t for t in tasks if finished[Path(t[0]).name] is None]
    if len(todo) < len(tasks):
        log(f"  {len(tasks) - len(todo)} shards already written by an interrupted run: kept")
    by_name: dict = {k: v for k, v in finished.items() if v is not None}
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        for k, _ in enumerate(pool.imap(_shard_task, todo, chunksize=1)):
            # read back from the shard, as a restart does, so the index is the same bytes however it was made
            by_name[Path(todo[k][0]).name] = _finished_shard(Path(todo[k][0]))
            infos = [r for t in tasks if Path(t[0]).name in by_name for r in by_name[Path(t[0]).name]]
            if k + 1 in (2, 10) or (k + 1) % 20 == 0 or k + 1 == len(todo):
                done = sum(p.stat().st_size for p in (root / "shards").glob("*.h5") if p.name in new) / 1e9
                steps = sum(i["steps"] for i in infos)
                made = [r for r in plan.to_dict("records") if r["id"] in {i["id"] for i in infos}]
                ratio = steps / max(planned_steps(pd.DataFrame(made), cfgt, o.get("length_min")), 1.0)
                proj = done / max(steps, 1) * steps_planned * ratio  # the bytes per step so far, on every planned step
                log(f"  {len(by_name)} of {len(tasks)} shards: {done:.2f} GB so far, about {proj:.1f} GB projected")
                if have_gb + proj > o["max_gb"] and k + 1 >= 2:
                    pool.terminate()
                    raise RuntimeError(f"the dataset would take about {have_gb + proj:.1f} GB, over the "
                                       f"{o['max_gb']} GB budget")  # fmt: skip
    infos = [r for t in tasks for r in by_name[Path(t[0]).name]]  # in plan order, however they were made
    idx = pd.concat([old, pd.DataFrame(infos)], ignore_index=True)
    parts = nested_parts(training_subjects("train"), seed)
    idx["part"] = [parts.get((r.subset, r.subject), 0) if r.split == "train" else 0 for r in idx.itertuples()]
    idx.to_parquet(root / "index.parquet", index=False)
    return idx


WINDOW_STEPS, LOSS_STEPS = 576, 320
"""The guide's windows (§6.1): 18 s at 32 Hz, the loss on the last 10 s."""
BOOT_BINS_MIN = (0.0, 5.0, 15.0, 35.0)


def windows(steps: int) -> np.ndarray:
    """The reference windows of one session of ``steps`` steps, as (start, end) step indices, end exclusive: 18 s
    windows whose 10 s loss regions (their last :data:`LOSS_STEPS` steps) tile the session from its first full
    window on, so every step after the first 8 s is in exactly one loss region. A window never leaves its session,
    and its inputs end where its loss region ends (causal)."""
    ends = np.arange(WINDOW_STEPS, steps + 1, LOSS_STEPS)
    return np.stack([ends - WINDOW_STEPS, ends], axis=1) if ends.size else np.zeros((0, 2), int)


def _boot_bin(minutes: np.ndarray) -> np.ndarray:
    b = np.clip(np.searchsorted(BOOT_BINS_MIN, minutes, side="right") - 1, 0, len(BOOT_BINS_MIN) - 2)
    return np.array([f"{BOOT_BINS_MIN[i]:g}-{BOOT_BINS_MIN[i + 1]:g} min" for i in b])


def coverage(root: Path, split: str = "train") -> pd.DataFrame:
    """Training windows per stratum (§6.2): (pod, or joint type) × motion type × time since boot. A window's motion
    type is the most frequent over its loss region, its time since boot that of its last step; it counts for a pod
    whose target is valid on at least half its loss steps, and for a joint type when any joint of the type is."""
    import h5py

    from ..body import JOINTS, PODS

    idx = pd.read_parquet(root / "index.parquet")
    names = {v: k for k, v in MOTION_CODES.items()}
    jtypes = sorted({j.jtype for j in JOINTS})
    of_type = {t: np.array([j.jtype == t for j in JOINTS]) for t in jtypes}
    units = [f"pod:{p.name}" for p in PODS] + [f"type:{t}" for t in jtypes]
    counts: dict = {}
    for shard, g in idx[idx["split"] == split].groupby("shard"):
        with h5py.File(root / shard, "r") as h5:
            for sid in g["id"]:
                grp = h5[f"sessions/{sid}"]
                motion = np.asarray(grp["strata/motion"])
                w = windows(motion.size)
                if not len(w):
                    continue
                loss = w[:, 1:2] - LOSS_STEPS + np.arange(LOSS_STEPS)[None, :]  # (W, 320) step indices
                major = np.array([np.bincount(m, minlength=len(MOTION_CODES)).argmax() for m in motion[loss]])
                boot = _boot_bin(np.asarray(grp["strata/time_since_boot_s"])[w[:, 1] - 1] / 60.0)
                pod_ok = np.asarray(grp["valid_eps"])[loss].mean(axis=1) >= 0.5  # (W, 15)
                joint_ok = np.asarray(grp["valid_delta"])[loss].mean(axis=1) >= 0.5  # (W, 14)
                ok = np.concatenate([pod_ok] + [joint_ok[:, of_type[t]].any(axis=1, keepdims=True) for t in jtypes],
                                    axis=1)  # fmt: skip
                for k in range(len(w)):
                    for u in np.flatnonzero(ok[k]):
                        key = (units[u], names[int(major[k])], boot[k])
                        counts[key] = counts.get(key, 0) + 1
    present = sorted({k[1] for k in counts})
    rows = [{"unit": u, "motion": m, "since_boot": b, "windows": counts.get((u, m, b), 0)}
            for u in units for m in present for b in _boot_bin(np.array([2.5, 10.0, 25.0]))]  # fmt: skip
    return pd.DataFrame(rows)


def write_meta(root: Path, meta: dict) -> None:
    from ..io import write_json

    write_json(root / "meta.json", meta)
    readme = f"""# {root.name}

The Mesquite twin's training data (Step 3 brief §6). Generated by `python -m mesqsim twin-dataset`; never edited in
place (a new version is a new name).

- `shards/<split>_<nnnnn>.h5`: HDF5, one group per session under `/sessions/<id>/`, gzip, chunked in time.
  Arrays: `t_s` (T,), `x_pod` (T, 15, 8), `x_joint` (T, 14, 6), `x_phone` (T, 8), `x_pod_phone` (T, 15, 6),
  `state_pod` (T, 15), `state_phone` (T,), `y_delta`/`valid_delta` (T, 14), `y_eps`/`valid_eps` (T, 15),
  `strata/time_since_boot_s`, `strata/motion`, `strata/transition`, `strata/slip`. Attributes: `subject`, `split`,
  `suit_version`, `conditions` (JSON).
- `index.parquet`: one row per session (its shard, subject, split, length, conditions, nested `part`).
- `meta.json`: commit, config hashes, seeds, the feature version, the calibration it was drawn from.
- `MANIFEST.csv`: every file's name, size and SHA-256; `python -m mesqsim verify-dataset --name {root.name}` checks it.

Read it with `h5py`: `h5py.File(shard)["sessions"][id]["x_pod"][...]`. Only `mesqsim/learn/targets.py` reads
the `y_*` arrays; a loader cuts the windows (18 s, the loss on the last 10 s).
"""
    (root / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def write_manifest(root: Path, copy_to: Path | None = None) -> pd.DataFrame:
    man = manifest_rows(root)
    man.to_csv(root / "MANIFEST.csv", index=False)
    if copy_to is not None:
        man.to_csv(copy_to, index=False)
    return man


def recompute_matches(root: Path, n: int = 1, streaming: tuple = ()) -> dict:
    """Simulate ``n`` stored sessions again from their seeds and conditions: are the stored arrays the same bytes?
    With ``streaming`` (fractions of the session, for example (0.25, 0.5, 1.0)), also recompute the features from only
    the frames and phone messages that had arrived by each such step, as live inference would, and compare that
    step's stored row (§6.1, §10 item 11)."""
    import h5py

    from ..config import load_config
    from ..paths import REPO_ROOT
    from ..twin.config import load_twin_config

    idx = pd.read_parquet(root / "index.parquet").sort_values("id").head(n)
    cfgt, cfg1 = load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml")
    out = {}
    for r in idx.to_dict("records"):
        lm = r.get("length_min")
        lm = None if lm is None or (isinstance(lm, float) and math.isnan(lm)) else float(lm)
        with h5py.File(root / r["shard"], "r") as h5:
            grp = h5[f"sessions/{r['id']}"]
            stored = {k: np.asarray(grp[k]) for k in ("x_pod", "x_joint", "x_phone", "x_pod_phone", "y_delta")}
            cond = json.loads(grp.attrs["conditions"])
        c = {"twin": cond["twin"], "step1": cond["step1"], "suit": cond["twin"]["twin.suit.version"],
             "hipsalt": cond["hipsalt"], "protocol": cond["program"].startswith("protocol")}  # fmt: skip
        built = build_session(r, c, cfgt, cfg1, lm, keep_session=bool(streaming))
        same = all(np.array_equal(stored[k], np.asarray(built["arrays"][k])) for k in stored)
        if streaming:
            same &= _streaming_equal(built["session"], stored, streaming, cfg1)
        out[r["id"]] = bool(same)
    return out


def _streaming_equal(s, stored: dict, fractions, cfg1) -> bool:
    from ..learn.features import features
    from ..step0 import decode

    t = s.labels_steps["t_s"].to_numpy() + s.meta["plan"]["host_epoch_s"]
    for frac in fractions:
        k = min(t.size - 1, int(frac * (t.size - 1)))
        g = features(decode(s.frames[s.frames["t_host_s"] <= t[k]], s.session), s.phone[s.phone["t_host_s"] <= t[k]],
                     t[: k + 1], cfg1["geometry.clock_window_s"], cfg1["geometry.clock_block_s"])  # fmt: skip
        if not all(np.array_equal(stored[a][k], getattr(g, a)[k]) for a in ("x_pod", "x_joint", "x_phone",
                                                                            "x_pod_phone")):  # fmt: skip
            return False
    return True
