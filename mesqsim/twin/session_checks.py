"""The numbers behind checks 40.2, 43.2 and 46.1-48.1 (Step 3 brief §10 items 1, 4, 7, 8, 9): the twin's sessions.

They need the licensed AMASS files and SMPL+H model (and Node for 40.2); without them each returns ``None`` and the
check skips. Sessions are cached, so one validation run builds each only once.

* ``short_session``: a one-minute random program of KIT subject 3 (training split) after the 35 s procedure.
"""

from __future__ import annotations

import functools
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from ..body import JOINTS, N_PODS
from ..conventions import wrap_rad

__all__ = ["available", "short_session", "join_numbers", "table_numbers", "invariance_numbers",
           "determinism_numbers", "bvh_numbers"]

SEED, MINUTES, SUBJECT = 3, 1.0, "KIT/3"


def available() -> bool:
    from ..data.amass import amass_dir
    from ..paths import data_dir

    return (data_dir() / "body_models" / "smplh").is_dir() and (amass_dir() / "KIT" / "3").is_dir()


@functools.lru_cache(maxsize=4)
def _program(name: str, seed: int, minutes: float | None):
    from .config import load_twin_config
    from .program import make_program

    cfgt = load_twin_config()
    return make_program(cfgt, name, seed, SUBJECT if name == "random" else None, minutes)


@functools.lru_cache(maxsize=8)
def short_session(seed: int = SEED, pod_turn: tuple = (), phone_turn: float = 0.0, shift: tuple = (0.0, 0.0, 0.0)):
    """The cached short session; ``pod_turn`` is a tuple of (pod id, angle) pairs."""
    from ..config import load_config
    from ..paths import REPO_ROOT
    from .config import load_twin_config
    from .session import simulate_twin_session

    joined, model, label = _program("random", SEED, MINUTES)
    cfg1, cfgt = load_config(REPO_ROOT / "configs" / "default.yaml"), load_twin_config()
    turns = dict(pod_turn) if pod_turn else None
    return simulate_twin_session(cfg1, cfgt, seed, joined, model, label, pod_turn_rad=turns,
                                 phone_turn_rad=phone_turn, room_shift_m=shift)  # fmt: skip


def join_numbers() -> dict | None:
    """The join check (§4.9) on the protocol program and on the short random program."""
    if not available():
        return None
    from .program import join_check

    out = {}
    for name, minutes in (("protocol", None), ("random", MINUTES)):
        joined = _program(name, SEED, minutes)[0]
        out[name] = join_check(joined) | {"seconds": joined.seq.poses.shape[0] / joined.seq.fps,
                                          "pieces": len(joined.pieces)}  # fmt: skip
    return out


def table_numbers() -> dict | None:
    """δ and ε recomputed from the stored ψ columns against the stored δ and ε (largest |difference|, rad)."""
    if not available():
        return None
    s = short_session()
    t = s.labels_ticks
    eps = wrap_rad(t["psi_rad"].to_numpy() - t["psi_ph_rad"].to_numpy())
    ok = ~np.isnan(eps)
    d_eps = float(np.max(np.abs(wrap_rad(eps[ok] - t["eps_rad"].to_numpy()[ok]))))
    st, lj = s.labels_steps, s.labels_joints
    worst = 0.0
    for k, jt in enumerate(JOINTS):
        dj = wrap_rad(st[f"psi{jt.child}_rad"].to_numpy() - st[f"psi{jt.parent}_rad"].to_numpy())
        stored = lj.loc[lj["joint"] == k, "delta_rad"].to_numpy()
        m = ~np.isnan(dj)
        worst = max(worst, float(np.max(np.abs(wrap_rad(dj[m] - stored[m])))) if m.any() else 0.0)
    return {"eps_max_rad": d_eps, "delta_max_rad": worst, "ticks": int(ok.sum()), "steps": len(st)}


def _labels(s):
    t = s.labels_ticks.sort_values(["pod_id", "epoch", "tick"], kind="mergesort")
    return t, s.labels_joints.sort_values(["step", "joint"], kind="mergesort")


def invariance_numbers(beta: float = 0.7, gamma: float = -0.4, pod: int = 10) -> dict | None:
    """§4.12: (a) every pod report and the phone turned by one Y(β); (b) pod ``pod`` alone by Y(γ); (c) the room's
    origin moved. Largest label changes against what the twist rule says they must be."""
    if not available():
        return None
    base = short_session()
    tb, jb = _labels(base)
    common = short_session(pod_turn=tuple((i, beta) for i in range(N_PODS)), phone_turn=beta)
    tc, jc = _labels(common)
    one = short_session(pod_turn=((pod, gamma),))
    to, jo = _labels(one)
    moved = short_session(shift=(3.0, -2.0, 0.0))
    tm, jm = _labels(moved)

    def dmax(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        return float(np.max(np.abs(wrap_rad(a[m] - b[m])))) if m.any() else 0.0

    psi_b, eps_b, del_b = tb["psi_rad"].to_numpy(), tb["eps_rad"].to_numpy(), jb["delta_rad"].to_numpy()
    out = {"common": {"psi_shift_err": dmax(tc["psi_rad"].to_numpy(), psi_b - beta),
                      "psi_ph_shift_err": dmax(tc["psi_ph_rad"].to_numpy(), tb["psi_ph_rad"].to_numpy() - beta),
                      "eps": dmax(tc["eps_rad"].to_numpy(), eps_b), "delta": dmax(jc["delta_rad"].to_numpy(), del_b)}}
    is_pod = to["pod_id"].to_numpy() == pod
    sign = np.zeros(len(jb))
    joint = jb["joint"].to_numpy()
    for k, jt in enumerate(JOINTS):  # δ_pc = ψ_c - ψ_p: ψ_i - γ moves δ by -γ where i is the child, +γ the parent
        sign[joint == k] = -1.0 if jt.child == pod else (1.0 if jt.parent == pod else 0.0)
    out["one"] = {"psi_pod_err": dmax(to["psi_rad"].to_numpy()[is_pod], psi_b[is_pod] - gamma),
                  "psi_others": dmax(to["psi_rad"].to_numpy()[~is_pod], psi_b[~is_pod]),
                  "eps_pod_err": dmax(to["eps_rad"].to_numpy()[is_pod], eps_b[is_pod] - gamma),
                  "eps_others": dmax(to["eps_rad"].to_numpy()[~is_pod], eps_b[~is_pod]),
                  "delta_err": dmax(jo["delta_rad"].to_numpy(), del_b + sign * gamma),
                  "joints_moved": sorted({JOINTS[k].name for k in np.unique(joint[sign != 0])})}  # fmt: skip
    out["room"] = {"psi": dmax(tm["psi_rad"].to_numpy(), psi_b), "psi_ph": dmax(tm["psi_ph_rad"].to_numpy(),
                                                                                tb["psi_ph_rad"].to_numpy()),
                   "delta": dmax(jm["delta_rad"].to_numpy(), del_b),
                   "phone_lines_equal": bool(moved.phone["line"].equals(base.phone["line"]))}  # fmt: skip
    return out


def estimator_invariance(beta: float = 0.7) -> dict | None:
    """B1 and GEO-ID on the short session and on its common-turn twin: the largest |Δδ_hat| (degrees)."""
    if not available():
        return None
    from ..config import load_config
    from ..evaluate import grid_from_labels, method_estimate
    from ..paths import REPO_ROOT
    from ..step0 import decode

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    est = []
    for s in (short_session(), short_session(pod_turn=tuple((i, beta) for i in range(N_PODS)), phone_turn=beta)):
        grid = grid_from_labels(s.labels_joints)
        series = decode(s.frames, s.session)
        est.append({m: method_estimate(m, grid, series, s.session, s.meta, cfg)[0] for m in ("B1", "GEO-ID")})
    return {m: float(np.rad2deg(np.nanmax(np.abs(wrap_rad(est[0][m] - est[1][m]))))) for m in ("B1", "GEO-ID")}


def _hash_folder(folder: Path) -> dict:
    return {str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob("*")) if p.is_file()}


def determinism_numbers() -> dict | None:
    """The same seed twice, written to disk: identical files?"""
    if not available():
        return None
    from ..config import load_config
    from ..paths import REPO_ROOT
    from .config import load_twin_config
    from .session import simulate_twin_session, write_twin_session

    joined, model, label = _program("random", SEED, MINUTES)
    cfg1, cfgt = load_config(REPO_ROOT / "configs" / "default.yaml"), load_twin_config()
    with tempfile.TemporaryDirectory() as tmp:
        hashes = []
        for k in range(2):
            s = short_session() if k == 0 else simulate_twin_session(cfg1, cfgt, SEED, joined, model, label)
            hashes.append(_hash_folder(write_twin_session(Path(tmp) / f"run{k}", s)))
    return {"files": len(hashes[0]), "identical": hashes[0] == hashes[1],
            "differ": sorted(k for k in hashes[0] if hashes[0][k] != hashes[1].get(k))}


def bvh_numbers() -> dict | None:
    """The twin BVH, and Danyal's Nov 14 BVH when present, through the page's own BVHLoader (Node)."""
    from ..paths import REPO_ROOT, data_dir

    mocap = data_dir() / "mesquite-mocap"
    node = shutil.which("node")
    if not available() or node is None or not (mocap / "build-static" / "BVHLoader.js").is_file():
        return None
    script = REPO_ROOT / "tools" / "page_golden" / "load_bvh.mjs"

    def load(path: Path) -> dict:
        r = subprocess.run([node, str(script), str(path), str(mocap)], capture_output=True, text=True, check=True,
                           env={"NODE_NO_WARNINGS": "1", "PATH": str(Path(node).parent)})  # fmt: skip
        return json.loads(r.stdout)

    s = short_session()
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "twin.bvh"
        p.write_text(s.bvh, encoding="utf-8", newline="\n")
        twin = load(p)
    frames = int(s.bvh.split("Frames:")[1].split()[0])
    real_path = data_dir() / "real_suit" / "Test_Nov14_new_code" / "MMcap_bvh_2026-11-14-17-52-25.bvh"
    real = load(real_path) if real_path.is_file() else None
    return {"twin": twin, "frames": frames, "real": real}
