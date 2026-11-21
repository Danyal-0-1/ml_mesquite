"""Through the page (Step 3 brief §8.3): the symptoms of 2026-11-14, and what a heading correction does to them.

**The set-up.** Held-out test sessions (the first draw of each test subject) simulated again with Nov 14's slips
forced: both left-leg pods restarted after the box, the phone zeroed at Start, the velvet tree, HipsAlt present, the
new suit (:data:`NOV14`).

**The correction.** Every pod reading is turned by its estimate, q̃_i(t) = Y(ε̂_i(t) + c)·q_i(t), at its arrival; so
the T-pose snapshot the page keeps is q̃_i(T) = Y(ε̂_i(T) + c)·q_i(T), as the world-delta chain requires. c is the
page's constant turn between the phone's frame and a pod's (:data:`PHONE_TO_POD_RAD`). The phone is the anchor and is
never turned. ε̂ is held from the latest 32 Hz step at or before the reading (causal), and is 0 before
the first step. Then the page twin draws, and its BVH is measured by the recordings' own code
(:func:`mesqsim.twin.twin_stats.analyse_twin`).

**Reported** per variant: the drawn shins' Φ3 against the truth's (target: within ±15°) and Φ7, travel minus facing,
against the truth's (target: within ±20°). Variants: ``as_is`` alone; ``as_is`` + RIGID+B1; ``as_is`` + O2 (the true
ε: what a perfect TCN-P would reach); ``ideal_root_box`` + O2. A model of :mod:`.contract` slots in through
``python -m mesqsim score-model --through-page``: a per-pod model's ε̂ as it is, a per-joint model's δ̂ composed from
RIGID's root link along the tree (TCN-P-chain).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..body import PODS

__all__ = ["NOV14", "VARIANTS", "page_rows", "corrected_bvh", "page_session", "page_test", "page_section",
           "score_through_page"]

NOV14 = {"twin.procedure.p_limb_restart": 1.0, "twin.procedure.force_limb": "left_leg",
         "twin.procedure.p_zero_at_start": 1.0, "twin.procedure.p_camera_mismatch": 0.0,
         "twin.procedure.p_tree_mismatch": 0.0, "twin.page.tree": "velvet"}  # fmt: skip
"""Nov 14's slips (Step 3 brief §2.3, §8.3), forced; the suit and HipsAlt are set by :func:`page_rows`."""
PHONE_TO_POD_RAD = -np.pi / 2
"""The turn about the vertical between the phone's frame and a pod's in the page's drawing (velvet tree): the page
maps a pod's world (+Z up) to three.js by (x, y, z) -> (y, z, x) and the phone's ARKit frame by (x, y, z) ->
(-x, y, -z), and the repository reads ARKit as (x, y, z) -> (x, -z, y); so world +X is drawn at three.js (0, 0, 1)
for a pod and at (-1, 0, 0) for the phone, and a pod turned to the phone's north ε is drawn 90° off the phone's root
path unless it is turned by this as well. A constant map, so it is absorbed into the correction (§8.4: "a constant
map turned about the vertical is absorbed"); measured on the twin: with the phone zeroed in the T-pose, ε alone puts
both shins' Φ3 about 90° off, ε + this within about 20°. Φ3 and Φ7 cannot tell +90° from -90°; the sign follows from
the maps."""
VARIANTS = (("as_is", None), ("as_is", "RIGID+B1"), ("as_is", "O2"), ("ideal_root_box", "O2"))
SHINS = ("LeftLeg", "RightLeg")
TARGET_PHI3_DEG, TARGET_PHI7_DEG = 15.0, 20.0
BONE_POD = {p.name: p.id for p in PODS}


def page_rows(root, split: str = "test", n: int = 12, zero_at_start: bool = True) -> list[tuple[dict, dict]]:
    """The first draw of the first ``n`` subjects of ``split``, with Nov 14's conditions; ``zero_at_start=False`` keeps
    every other slip but has the phone zeroed in the T-pose, on the belt, as the procedure asks."""
    from ..results_step3 import test_rows
    from ..twin.suits import suit_overrides

    rows = [(r, c) for r, c in test_rows(root, split) if int(r["variant"]) == 0][:n]
    out = []
    for r, c in rows:
        twin = (dict(c["twin"]) | suit_overrides("suit_2026_11_14") | NOV14 | {"twin.suit.pods": list(range(15))}
                | {"twin.procedure.p_zero_at_start": 1.0 if zero_at_start else 0.0})
        twin.pop("twin.wire.json_decimals", None)
        out.append((r, c | {"twin": twin, "suit": "suit_2026_11_14", "hipsalt": True, "protocol": False}))
    return out


def _correction(t_grid: np.ndarray, eps: np.ndarray):
    """(bone, t_ms) -> ε̂ of the bone's pod at the latest step at or before t (0 before the first step)."""
    eps = np.nan_to_num(np.asarray(eps, dtype=np.float64))

    def corr(bone: str, t_ms: float) -> float:
        i = BONE_POD.get(bone)
        if i is None:
            return 0.0
        k = int(np.searchsorted(t_grid, t_ms / 1000.0, side="right")) - 1
        return float(eps[k, i]) if k >= 0 else 0.0

    return corr


def corrected_bvh(s, cfgt, mode: str, eps: np.ndarray | None, t_grid: np.ndarray) -> str:
    """The page twin's BVH of session ``s`` in ``mode``, every pod reading turned by ``eps`` + :data:`PHONE_TO_POD_RAD`
    (None: as recorded)."""
    from ..twin.session import drive_page

    if eps is None:
        return drive_page(s, cfgt, mode=mode)
    return drive_page(s, cfgt, mode=mode, heading_correction=_correction(t_grid, np.nan_to_num(np.asarray(eps, float))
                                                                         + PHONE_TO_POD_RAD))  # fmt: skip


class _Drawn:
    def __init__(self, bvh: str, truth: dict):
        self.bvh, self.truth = bvh, truth


def _measures(bvh: str, truth: dict) -> dict:
    from ..twin.twin_stats import analyse_twin

    a = analyse_twin(_Drawn(bvh, truth), "optitrack")
    d3, t3 = a.get("phi3_drawn_deg", {}), a.get("phi3_truth_deg", {})
    out = {f"phi3_{s}_err_deg": (d3[s] - t3[s]) if d3.get(s) is not None and t3.get(s) is not None else np.nan
           for s in SHINS}  # fmt: skip
    p7, p7t = a.get("phi7_deg"), a.get("phi7_truth_deg")
    out["phi7_err_deg"] = float(np.rad2deg(np.angle(np.exp(1j * np.deg2rad(p7 - p7t))))) if (
        p7 is not None and p7t is not None) else np.nan
    out["phi7_drawn_deg"], out["phi7_truth_deg"] = p7, p7t
    out["root_height_cm"] = a.get("root_height_cm")
    return out


def _model_eps(model_spec: str, built: dict, rigid_root: np.ndarray) -> np.ndarray:
    """A contract model's ε̂ on one regenerated session: as it is for a per-pod model; for a per-joint model, its δ̂
    composed from RIGID's root link along the tree (TCN-P-chain)."""
    from .contract import SessionInputs
    from .rigid import chain
    from .scorer import load_model

    model = load_model(model_spec)
    a = built["arrays"]
    inputs = SessionInputs(*(np.asarray(a[k]) for k in ("t_s", "x_pod", "x_joint", "x_phone", "x_pod_phone",
                                                        "state_pod", "state_phone")), attrs=dict(built["attrs"]))
    th = np.asarray(model.predict(inputs).theta_rad, float)
    return chain(rigid_root, th) if model.target == "delta" else th


def page_session(args) -> list[dict]:
    """One page-test session (a process-pool task): ``args = (row, cond, model_spec or None)``; a model adds the
    variant ``as_is`` + its ε̂."""
    from ..config import load_config
    from ..paths import REPO_ROOT
    from ..results_step3 import session_estimates
    from ..twin.config import load_twin_config
    from .rigid import rigid_track
    from .store import build_session

    row, cond, model_spec = args
    cfgt0, cfg1 = load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml")
    built = build_session(row, cond, cfgt0, cfg1, row.get("length_min"), keep_session=True, page=True)
    s = built["session"]
    cfgt = cfgt0.with_overrides(cond["twin"])
    t_grid = built["arrays"]["t_s"]
    _, eps = session_estimates(built, cfg1, geo=False)
    est = {"RIGID+B1": eps["RIGID+B1"][0], "O2": eps["O2"][0]}
    variants = list(VARIANTS)
    if model_spec:
        f = built["features"]
        est[model_spec] = _model_eps(model_spec, built, rigid_track(f.q_phone, f.q_pod[:, 2])[0])
        variants.append(("as_is", model_spec))
    out = []
    for mode, method in variants:
        bvh = corrected_bvh(s, cfgt, mode, None if method is None else est[method], t_grid)
        out.append({"id": row["id"], "subject": built["attrs"]["subject"], "mode": mode, "method": method or "none",
                    "slips": built["info"]["slips"]} | _measures(bvh, s.truth))  # fmt: skip
    return out


def page_test(root, split: str = "test", n: int = 12, workers: int = 8, model_spec: str | None = None,
              zero_at_start: bool = True) -> pd.DataFrame:  # fmt: skip
    import multiprocessing as mp

    tasks = [(r, c, model_spec) for r, c in page_rows(root, split, n, zero_at_start)]
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=2) as pool:  # memory: fresh processes
        res = list(pool.imap(page_session, tasks, chunksize=1))
    return pd.DataFrame([x for r in res for x in r])


def _md_table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _f1(x, nd: int = 1) -> str:
    return "–" if not np.isfinite(x) else f"{x:.{nd}f}"


def _axial(d) -> np.ndarray:
    return (np.asarray(d, dtype=float) + 90.0) % 180.0 - 90.0


def page_table(df: pd.DataFrame) -> str:
    """Per variant: the median |error| of the shins' Φ3 and of Φ7 against the truth, the share of sessions within the
    targets, and how far the two shins disagree with each other (|Φ3 left − Φ3 right| of the errors, axial)."""
    rows = []
    for (mode, method), g in df.groupby(["mode", "method"], sort=False):
        e3s = _axial(g[[f"phi3_{s}_err_deg" for s in SHINS]].to_numpy(dtype=float))  # Φ3 is an axis: modulo 180°
        e3 = np.abs(e3s)
        e7 = np.abs(g["phi7_err_deg"].to_numpy(dtype=float))
        has3, has7 = np.all(np.isfinite(e3), axis=1), np.isfinite(e7)  # a session that never walks has no Φ3, Φ7
        ok3 = np.all(e3[has3] <= TARGET_PHI3_DEG, axis=1)
        ok7 = e7[has7] <= TARGET_PHI7_DEG
        rows.append([f"`{mode}` + {method}" if method != "none" else f"`{mode}` alone",
                     _f1(np.nanmedian(e3[:, 0]) if has3.any() else np.nan),
                     _f1(np.nanmedian(e3[:, 1]) if has3.any() else np.nan),
                     f"{int(ok3.sum())} of {int(has3.sum())}",
                     _f1(np.nanmedian(np.abs(_axial(e3s[:, 0] - e3s[:, 1]))) if has3.any() else np.nan),
                     _f1(np.nanmedian(e7) if has7.any() else np.nan), f"{int(ok7.sum())} of {int(has7.sum())}",
                     _f1(np.nanmedian(g["root_height_cm"].to_numpy(dtype=float)), 0)])  # fmt: skip
    return _md_table(["variant", "|Φ3| left shin", "|Φ3| right shin", f"both shins within ±{TARGET_PHI3_DEG:g}°",
                      "shins disagree", "|Φ7|", f"Φ7 within ±{TARGET_PHI7_DEG:g}°", "drawn root height (cm)"], rows)


def page_section(df: pd.DataFrame, df_tpose: pd.DataFrame | None = None) -> str:
    """§8.3: the table for Nov 14's slips, and (if given) for the same slips with the phone zeroed in the T-pose."""
    if df.empty:
        return "## 4. Through the page (§8.3)\n\nNo sessions.\n"
    n = df["id"].nunique()
    out = ["## 4. Through the page (§8.3)", "",
           f"{n} held-out test sessions (the first draw of each of the first {n} test subjects) with Nov 14's slips "
           "forced: both left-leg pods restarted after the box, the phone zeroed at Start, the velvet tree, HipsAlt "
           "present, the new suit. Each pod reading turned by its estimate at its arrival (so at the T-pose too); the "
           "phone never turned; drawn by the page twin; measured by the recordings' own code. Errors are drawn minus "
           "truth, degrees (median over the sessions that walk of the absolute error; a session that never walks has "
           "no Φ3 or Φ7). Φ3 is the shin's swing axis against the drawn hips' facing.", "", page_table(df), ""]
    if df_tpose is not None and not df_tpose.empty:
        out += ["The same sessions and slips, but the phone zeroed in the T-pose on the belt, as the procedure asks:",
                "", page_table(df_tpose), ""]
    out += ["**Reading it.** Every correction adds the page's constant turn between the phone's frame and a pod's "
            "(-90° for velvet's maps, `PHONE_TO_POD_RAD`). Turned to the phone's north (O2), all pods share one "
            "world, so the two shins agree with each other ('shins disagree'). The drawing is right when, in that "
            "world, the wearer faced the page's forward at the T-pose, which the phone's Zero in the T-pose arranges: "
            "then the true ε brings the shins close to the truth. Zeroed at Start, in the hand, the phone's north is "
            "wherever the hand pointed, and every pod corrected to the phone inherits it (Φ7 too): no correction "
            "relative to the phone fixes a phone zeroed away from the wearing pose. RIGID+B1 turns every pod by the "
            "root link's estimate (B1's box registration carries no information when the pods boot in the box): the "
            "legs keep their own norths, the drawing turns as a whole, and Φ3 and Φ7 both get worse.", ""]
    return "\n".join(out)


def score_through_page(model_spec: str, split: str = "test", max_sessions: int | None = None,
                       dataset: str = "twin_v1", workers: int = 8) -> int:  # fmt: skip
    """``score-model --through-page``: the page test with ``model_spec``'s ε̂ added as a variant."""
    from ..paths import output_dir

    df = page_test(output_dir() / "datasets" / dataset, split, max_sessions or 12, workers, model_spec)
    out = output_dir() / "scores" / f"{model_spec.replace(':', '-')}__{dataset}__through_page"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "page.csv", index=False)
    print(page_section(df))
    print(f"wrote {out}")
    return 0
