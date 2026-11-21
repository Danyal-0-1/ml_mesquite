"""The numbers behind the dataset's checks (Step 3 brief §10 items 9 and 11): a tiny dataset made twice, extended,
and verified; and the rules on ``twin_v1`` itself when it has been generated."""

from __future__ import annotations

import atexit
import functools
import math
import shutil
import tempfile
from pathlib import Path

import pandas as pd

__all__ = ["available", "stand_in_posterior", "tiny_dataset", "manifest_numbers", "dataset_numbers", "c5_numbers",
           "kit_numbers", "rigid_numbers", "VM_CASES", "vm_nll_reference"]

TINY = {"variants": {"train": 2, "val": 1, "test": 1}, "max_subjects": 1, "length_min": 0.25, "sessions_per_shard": 2}
"""One subject per split, two training draws and one of each other split, quarter-minute programs."""


def available() -> bool:
    from ..twin import session_checks

    return session_checks.available()


def stand_in_posterior(n: int = 3) -> pd.DataFrame:
    """A posterior of ``n`` equal rows at the middle of every prior (the geometric middle of a log-uniform one, the
    first choice of a choice): the machinery's tests need a posterior, not the fit's."""
    from ..twin.calibration import PRIORS

    row = {p.key: (p.choices[0] if p.kind == "choice" else math.sqrt(p.lo * p.hi) if p.kind == "log_uniform"
                   else 0.5 * (p.lo + p.hi)) for p in PRIORS}  # fmt: skip
    return pd.DataFrame([row] * n)


@functools.cache
def _scratch() -> Path:
    d = Path(tempfile.mkdtemp(prefix="mesq_tiny_dataset_"))
    atexit.register(shutil.rmtree, d, True)
    return d


@functools.cache
def tiny_dataset(name: str, seed: int = 5) -> Path:
    """Generate the tiny dataset ``name`` (with ``seed``) once per process, its manifest written."""
    from ..twin.calibration import PRIORS
    from .store import generate, write_manifest

    root = _scratch() / name
    generate(root, stand_in_posterior(), PRIORS, seed, 2, TINY, log=lambda *_: None)
    write_manifest(root)
    return root


def manifest_numbers() -> dict | None:
    """§10 item 9: the tiny dataset twice with one seed (identical manifests?), then one of them extended by a factor
    of 2 (the stored files unchanged? the new training draws of the same composition?)."""
    if not available():
        return None
    from ..twin.calibration import PRIORS
    from .store import generate, verify, write_manifest

    a, b = tiny_dataset("a"), tiny_dataset("b")
    ma, mb = pd.read_csv(a / "MANIFEST.csv"), pd.read_csv(b / "MANIFEST.csv")
    ext = _scratch() / "extended"
    shutil.rmtree(ext, ignore_errors=True)  # a second call in the same process starts again
    shutil.copytree(a, ext)
    before = pd.read_csv(ext / "MANIFEST.csv").set_index("file")["sha256"]
    idx = generate(ext, stand_in_posterior(), PRIORS, 5, 2, TINY, extend=2.0, log=lambda *_: None)
    me = write_manifest(ext).set_index("file")["sha256"]
    kept = [f for f in before.index if f.startswith("shards/")]
    train = idx[idx["split"] == "train"]
    return {"files": len(ma), "identical": ma.equals(mb), "sessions": len(pd.read_parquet(a / "index.parquet")),
            "extended_sessions": len(idx), "train_variants": sorted(train["variant"].tolist()),
            "compositions": int(train["program_seed"].nunique()),
            "stored_unchanged": all(me.get(f) == before[f] for f in kept), "kept_shards": len(kept),
            "extended_ok": verify(ext)["ok"]}  # fmt: skip


def dataset_numbers(name: str = "twin_v1") -> dict:
    """§10 item 11 on the tiny dataset (always) and on ``name`` under $MESQ_OUTPUT_DIR/datasets when it exists: the
    manifest, the splits, the window rules (:func:`mesqsim.learn.store.verify`), and one session simulated again from
    its seed, its features also recomputed as a stream at a quarter, a half and the end."""
    from ..paths import output_dir
    from .store import recompute_matches, verify, windows

    out = {}
    if available():
        tiny = tiny_dataset("a")
        idx = pd.read_parquet(tiny / "index.parquet")
        out["tiny"] = verify(tiny) | {"recomputed": recompute_matches(tiny, 1, streaming=(0.25, 0.5, 1.0)),
                                      "windows": int(sum(len(windows(int(n))) for n in idx["steps"]))}  # fmt: skip
    root = output_dir() / "datasets" / name
    if (root / "MANIFEST.csv").is_file():
        out[name] = verify(root) | {"recomputed": recompute_matches(root, 1, streaming=(0.25, 0.5, 1.0))}
    return out


# ------------------------------------------------------------------------------------------------- the model kit
VM_CASES = ((0.0, 0.0), (0.3, 1.0), (-2.0, 1.0), (1.0, 50.0), (0.01, 1e4), (3.1, 1e5), (0.2, 1e7))
"""(θ - μ, κ) of the loss references: they include large κ, where a naive log I₀(κ) overflows (I₀(1e4) ≈ 10^4340)."""


def vm_nll_reference(d: float, kappa: float) -> float:
    """-log of the von Mises density at θ - μ = d, by SciPy (log 2π at κ = 0)."""
    from scipy.stats import vonmises

    return float(-vonmises.logpdf(d, kappa)) if kappa > 0 else math.log(2 * math.pi)


def c5_numbers() -> dict | None:
    """C5 (§7.1): the short twin session's inputs, and the same with every pod and the phone turned by one heading
    (0.9 rad, applied to the decoded reports, as a turned world would deliver them; the wire's int16 would round a
    turned report differently). The largest difference per input array."""
    if not available():
        return None
    import numpy as np

    from ..config import load_config
    from ..conventions import quat_mul, yaw_quat
    from ..paths import REPO_ROOT
    from ..step0 import decode
    from ..twin import session_checks
    from .features import features

    s = session_checks.short_session()
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    t = s.labels_steps["t_s"].to_numpy() + s.meta["plan"]["host_epoch_s"]
    series = decode(s.frames, s.session)
    turned = series.copy()
    q = series[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    ok = np.isfinite(q[:, 3])
    q[ok] = quat_mul(yaw_quat(np.full(ok.sum(), 0.9)), q[ok])
    turned[["qx", "qy", "qz", "qw"]] = q
    phone = s.phone.copy()
    h = 0.9 / 2  # the same turn about the reported frame's +Y (ARKit's up), as a rotation within that frame
    qp = phone[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    phone[["qx", "qy", "qz", "qw"]] = np.stack([np.cos(h) * qp[:, 0] + np.sin(h) * qp[:, 2],
                                                np.cos(h) * qp[:, 1] + np.sin(h) * qp[:, 3],
                                                np.cos(h) * qp[:, 2] - np.sin(h) * qp[:, 0],
                                                np.cos(h) * qp[:, 3] - np.sin(h) * qp[:, 1]], axis=1)  # fmt: skip
    cw, bs = cfg["geometry.clock_window_s"], cfg["geometry.clock_block_s"]
    a, b = features(series, s.phone, t, cw, bs), features(turned, phone, t, cw, bs)
    out = {k: float(np.abs(getattr(a, k).astype(float) - getattr(b, k).astype(float)).max())
           for k in ("x_pod", "x_joint", "x_phone", "x_pod_phone")}  # fmt: skip
    out["states_equal"] = bool(np.array_equal(a.state_pod, b.state_pod) and np.array_equal(a.state_phone,
                                                                                            b.state_phone))
    out["steps"] = int(t.size)
    return out


def kit_numbers() -> dict | None:
    """§10 item 12 on the tiny dataset: Zeros against B0 (every summary number and table cell), Cheat against
    label-free inputs, the loss references against a stable closed form, and the tests of Danyal's code run in a
    subprocess (they skip while their modules do not exist)."""
    import importlib.util
    import re
    import subprocess
    import sys

    import h5py
    import numpy as np
    from scipy.special import i0e

    from ..paths import REPO_ROOT
    from .contract import Model, SessionInputs, check_prediction
    from .dummies import Cheat, Zeros
    from .scorer import score

    out = {}
    worst = 0.0
    for d, k in VM_CASES:
        stable = -k * math.cos(d) + math.log(2 * math.pi) + math.log(i0e(k)) + k
        worst = max(worst, abs(vm_nll_reference(d, k) - stable) / max(1.0, abs(stable)))
    out["loss_rel_err"] = worst
    out["danyal_modules"] = {m: importlib.util.find_spec(f"mesqsim.learn.{m}") is not None
                             for m in ("loader", "models", "loss", "train")}  # fmt: skip
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_model_kit.py",
                          "-k", "danyals or parameter or receptive or overfitted or loader or tcn_p"],
                         cwd=REPO_ROOT, capture_output=True, text=True)  # fmt: skip
    tail = (run.stdout.strip().splitlines() or [""])[-1]
    out["danyal_tests"] = {w: int(n) for n, w in re.findall(r"(\d+) (passed|failed|skipped|error)", tail)}
    out["danyal_tests_rc"] = run.returncode
    if not available():
        return out
    tiny = tiny_dataset("a")
    z = Zeros()
    a, b = score(z, tiny, "test"), score("B0", tiny, "test")
    cols = ["stratum", "value", "unit", "n", "mean_deg", "median_deg", "p95_deg"]
    out["zeros_is_model"] = isinstance(z, Model)
    out["zeros_equals_b0"] = bool(all(a["summary"][k] == b["summary"][k] for k in ("median_deg", "mean_deg",
                                                                                   "p95_deg", "steps"))
                                  and a["table"][cols].equals(b["table"][cols]))  # fmt: skip
    out["zeros_causality_rad"] = a["summary"]["causality_error_rad"]
    out["b0_median_deg"] = b["summary"]["median_deg"]
    idx = pd.read_parquet(tiny / "index.parquet")
    r = idx[idx["split"] == "test"].iloc[0]
    blind = _scratch() / "blind.h5"
    shutil.copy(tiny / r["shard"], blind)
    with h5py.File(blind, "a") as h5:
        for k in ("y_delta", "valid_delta", "y_eps", "valid_eps"):
            del h5[f"sessions/{r['id']}/{k}"]
    with h5py.File(blind, "r") as h5:
        inputs = SessionInputs.from_group(h5[f"sessions/{r['id']}"], source=f"{blind}#{r['id']}")
    check_prediction(z, inputs, z.predict(inputs))
    with h5py.File(tiny / r["shard"], "r") as h5:
        full = SessionInputs.from_group(h5[f"sessions/{r['id']}"], source=f"{tiny / r['shard']}#{r['id']}")
    out["zeros_label_free"] = bool(np.array_equal(z.predict(inputs).theta_rad, z.predict(full).theta_rad))
    out["cheat_error_deg"] = score(Cheat(), tiny, "test", causal_steps=0)["summary"]["median_deg"]
    try:
        Cheat().predict(inputs)
        out["cheat_label_free"] = True
    except KeyError:
        out["cheat_label_free"] = False
    return out


def rigid_numbers() -> dict:
    """§10 item 13: the 7° example; 20 random noise-free rigid motions; a rotation about the vertical only."""
    import numpy as np

    from ..conventions import exp_map, quat_mul, wrap_rad, yaw_quat
    from .rigid import rigid

    def stream(t, w, psi, r0=None):
        r = exp_map(np.outer(t, w))
        if r0 is not None:
            r = quat_mul(r, np.broadcast_to(r0, r.shape))
        return quat_mul(yaw_quat(np.full(t.size, -psi)), r)

    t = np.linspace(0.0, 1.0, 33)
    w = np.array([1.0, 0.0, 0.0])
    seven = rigid(t, stream(t, w, np.deg2rad(5.0)), t, stream(t, w, np.deg2rad(12.0)))
    a = np.array([np.cos(np.deg2rad(-5.0)), np.sin(np.deg2rad(-5.0))])
    b = np.array([np.cos(np.deg2rad(-12.0)), np.sin(np.deg2rad(-12.0))])
    rng = np.random.default_rng(0)
    t4 = np.linspace(0.0, 4.0, 129)
    worst, vert = 0.0, 0.0
    for _ in range(20):
        w = rng.normal(size=3)
        r0 = exp_map(rng.normal(size=3))
        psi_h, psi_ph = rng.uniform(-np.pi, np.pi, 2)
        f = rigid(t4, stream(t4, w, psi_ph, r0), t4, stream(t4, w, psi_h, r0))
        worst = max(worst, abs(float(wrap_rad(f.eps_rad - (psi_h - psi_ph)))))
        vert = max(vert, f.vertical_rms)
    v = rigid(t4, stream(t4, [0, 0, 1.0], 0.3), t4, stream(t4, [0, 0, 1.0], 1.1))
    return {"seven_deg": float(np.rad2deg(seven.eps_rad)), "C": float(a @ b), "S": float(b[0] * a[1] - b[1] * a[0]),
            "random_max_err_rad": worst, "random_vertical_rms": vert, "vertical_only_kappa": float(v.kappa)}


def results_repro_numbers() -> dict | None:
    """§10 item 14: ``scripts/make_results_step3.py`` twice on the tiny dataset's test split (no ablations, no page
    test, no figures, 200 resamples): the two documents identical apart from their "Run:" line?"""
    if not available():
        return None
    import subprocess
    import sys

    from ..paths import REPO_ROOT

    tiny = tiny_dataset("a")
    outs = []
    for k in range(2):
        out = _scratch() / f"results_{k}.md"
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "make_results_step3.py"), "--dataset-root",
                        str(tiny), "--max-sessions", "1", "--no-ablations", "--no-page", "--no-figures", "--workers",
                        "1", "--n-boot", "200", "--out", str(out)], check=True, capture_output=True)  # fmt: skip
        outs.append([ln for ln in out.read_text(encoding="utf-8").splitlines() if not ln.startswith("Run: ")])
    return {"identical": outs[0] == outs[1], "lines": len(outs[0]),
            "differ": [a for a, b in zip(outs[0], outs[1], strict=False) if a != b][:3]}  # fmt: skip
