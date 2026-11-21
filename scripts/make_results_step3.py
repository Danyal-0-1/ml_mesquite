"""Reproduce docs/RESULTS_STEP3.md and its figures from one command (Step 3 brief §8, S3-M9).

    python scripts/make_results_step3.py            # twin_v1's test sessions: every part (about an hour on 8 workers)
    python scripts/make_results_step3.py --dataset-root DIR --max-sessions 1 --no-ablations --no-page --no-figures
                                                    # the reproducibility check's reduced run (check 53.1)

1. The calibration (S3-M6): the posterior, the predictive checks, the classifier test and gate G-T, read from the
   ``twin-calibrate`` folder the dataset was drawn from (its ``calibration.json``).
2. The data (S3-M7): hours, sessions, subjects, disk, the manifest's SHA-256, the coverage of the strata.
3. The numbers to beat (§8.1-8.2): B0, B1, O1, O2 and GEO-ID on δ; B0, B1, O1, O2, RIGID+B1 and RIGID+GEO on ε; every
   test session simulated again from its seed (mesqsim.results_step3); the strata; paired contrasts over subjects
   (§8.5). Then the twin ablations: segment frames, analytic fusion.
4. Through the page (§8.3): held-out sessions with Nov 14's slips, corrected by no one, RIGID+B1 and the truth, drawn
   by the page twin.

Nothing is fitted here. The same dataset, config and seed give an identical RESULTS_STEP3.md apart from its "Run:"
line, which holds everything that depends on the machine and the clock (check 53.1).
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import platform
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from mesqsim import plotstyle  # noqa: E402
from mesqsim import results_step3 as r3  # noqa: E402
from mesqsim.body import PODS  # noqa: E402
from mesqsim.paths import output_dir  # noqa: E402
from mesqsim.runs import git_commit  # noqa: E402
from mesqsim.twin.config import load_twin_config  # noqa: E402

CONTRASTS = (("delta", "B1", "B0"), ("delta", "GEO-ID", "B1"), ("eps", "RIGID+B1", "B1"),
             ("eps", "RIGID+GEO", "RIGID+B1"))  # fmt: skip
"""The primary contrasts (a - b of the per-subject mean error): does the box registration help, does Step 2's
estimator beat it, does RIGID's phone link help, does GEO-ID's δ beat the box's along the chain."""
TYPE_NAME = {"hinge": "hinge", "2dof": "2-DoF", "3dof": "3-DoF"}


def f2(x) -> str:
    return "–" if x is None or not np.isfinite(x) else f"{x:.2f}"


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def banner(cfgt) -> str:
    labels = Counter(leaf.get("label") for _, leaf in cfgt.leaves())
    parts = ", ".join(f"{labels.get(k, 0)} {k}" for k in ("fitted", "measured", "datasheet", "firmware",
                                                          "assumption"))  # fmt: skip
    return (f"> **Twin `{cfgt.hash()[:12]}`, AMASS motion.** Twin parameters: {parts}. Every number below is a twin "
            "number unless it says otherwise; the seven recordings are described, not tested.")


# ------------------------------------------------------------------------------------------------ 1. calibration
def section_calibration(cal: dict | None) -> str:
    if cal is None:
        return "## 1. The calibration (S3-M6)\n\nNo `twin-calibrate` folder was given or found.\n"
    rows = []
    for k, v in cal["posterior"].items():
        if v["kind"] == "choice":
            rows.append([f"`{k}`", "choice", v["posterior"], "–", "–"])
        else:
            rows.append([f"`{k}`", f"{v['prior']}", f"{v['median']:.4g}", f"{v['p5']:.4g} – {v['p95']:.4g}",
                         f"{100 * v['width_share']:.0f} %"])
    g = cal["gate"]
    a_t, a_s = cal["auc_twin"], cal["auc_step1"]
    txt = [
        "## 1. The calibration (S3-M6)",
        "",
        f"ABC by rejection (Beaumont et al. 2002) on the calibration recordings: {cal['draws']} prior draws, the "
        f"closest {cal['kept']} kept (distance: MAD-scaled RMS over the core statistics; best "
        f"{cal['best_distance']:.2f}, median {cal['median_distance']:.2f}).",
        "",
        md_table(["parameter", "prior", "posterior median", "90 % interval", "width vs prior"], rows),
        "",
        f"**Predictive checks** (every recording, core statistics inside the 90 % posterior predictive interval): "
        f"twin {g['twin_inside']} of {g['of']}, Step 1's simulator {g['step1_inside']} of {g['of']}.",
        "",
        f"**Classifier two-sample test** (real 10 s BVH windows against simulated ones, leave one recording out): "
        f"AUC twin {g['twin_auc']:.3f} ({a_t['p5']:.3f} – {a_t['p95']:.3f}), Step 1 {g['step1_auc']:.3f} "
        f"({a_s['p5']:.3f} – {a_s['p95']:.3f}).",
        "",
        f"**Gate G-T** {'passes' if g['passes'] else 'fails'}: the twin covers "
        f"{'at least as many' if g['twin_inside'] >= g['step1_inside'] else 'fewer'} core statistics as Step 1's "
        f"simulator, and its AUC is {'not higher' if g['twin_auc'] <= g['step1_auc'] else 'higher'}.",
        "",
    ]
    return "\n".join(txt)


# ------------------------------------------------------------------------------------------------ 2. the data
def section_data(root: Path) -> str:
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8")) if (root / "meta.json").is_file() else {}
    man = pd.read_csv(root / "MANIFEST.csv") if (root / "MANIFEST.csv").is_file() else None
    idx = pd.read_parquet(root / "index.parquet")
    rows = []
    for split in ("train", "val", "test"):
        g = idx[idx["split"] == split]
        if len(g):
            rows.append([split, (g["subset"] + "/" + g["subject"].astype(str)).nunique(), len(g),
                         f"{g['length_s'].sum() / 3600:.1f}",
                         int(g["variant"].max() + 1), f"{100 * (g['suit_version'] == 'suit_2026_11_13').mean():.0f} %",
                         f"{100 * (~g['hipsalt'].astype(bool)).mean():.0f} %"])  # fmt: skip
    txt = ["## 2. The data (S3-M7)", ""]
    txt.append(md_table(["split", "subjects", "sessions", "hours", "draws per subject", "old suit", "no HipsAlt"],
                        rows))  # fmt: skip
    if man is not None:
        digest = hashlib.sha256((root / "MANIFEST.csv").read_bytes()).hexdigest()
        txt += ["", f"{len(man)} files, {man['size_bytes'].sum() / 1e9:.2f} GB; `MANIFEST.csv` SHA-256 "
                f"`{digest[:16]}…`; feature version {meta.get('feature_version')}; drawn from the calibration "
                f"`{meta.get('calibration', {}).get('folder')}`."]
    cov_path = root / "coverage_train.csv"
    if cov_path.is_file():
        cov = pd.read_csv(cov_path)
        min_w = 1000
        under = cov[cov["windows"] < min_w]
        piv = cov.groupby(["motion", "since_boot"])["windows"].min().unstack("since_boot")
        txt += ["", f"**Coverage** (training windows of 18 s whose 10 s loss regions tile each session, per pod or "
                f"joint type × motion type × time since boot): {len(cov)} strata, {len(under)} under {min_w:,}. "
                "The least-covered unit per cell:", ""]
        cols = list(piv.columns)
        txt.append(md_table(["motion"] + cols, [[m] + [int(piv.loc[m, c]) if np.isfinite(piv.loc[m, c]) else 0
                                                        for c in cols] for m in piv.index]))  # fmt: skip
    txt.append("")
    return "\n".join(txt)


# ------------------------------------------------------------------------------------------------ 3. baselines
def _get(tb: pd.DataFrame, target, method, stratum="all", value="all", unit="all", col="median_deg"):
    m = tb[(tb["target"] == target) & (tb["method"] == method) & (tb["stratum"] == stratum)
           & (tb["value"] == value) & (tb["unit"] == unit)]  # fmt: skip
    return float(m[col].iloc[0]) if len(m) else float("nan")


def section_baselines(pooled: dict, tb: pd.DataFrame, con: pd.DataFrame, n_boot: int) -> str:
    cov = pooled["coverage"]
    rows = []
    for m in r3.DELTA_METHODS:
        c = cov.get(("delta", m))
        rows.append([m + (" ¹" if m in r3.USES_TRUTH else ""), f2(_get(tb, "delta", m)),
                     f2(_get(tb, "delta", m, col="mean_deg")), f2(_get(tb, "delta", m, col="p95_deg"))]
                    + [f2(_get(tb, "delta", m, unit=f"type:{t}")) for t in r3.JTYPES]
                    + ([f"{100 * c[0] / c[2]:.0f} % / {100 * c[1] / c[2]:.0f} %"] if c else ["–"]))  # fmt: skip
    txt = ["## 3. The numbers to beat (§8.1, §8.2)", "",
           f"Every test session of the dataset ({pooled['sessions']} sessions of {len(pooled['subjects'])} held-out "
           f"subjects, {pooled['steps']:,} steps of 32 Hz) simulated again from its seed and conditions. "
           "|wrap(θ̂ − θ)| in degrees on valid steps outside the `transition` blends, pooled over steps and units "
           "(medians exact to the 0.02° histogram bin). ¹ uses simulator truth.", "",
           "### 3.1 δ per joint", "",
           md_table(["method", "median", "mean", "P95"] + [f"median {TYPE_NAME[t]}" for t in r3.JTYPES]
                    + ["coverage 68 % / 95 %"], rows), ""]  # fmt: skip
    rows = []
    for m in r3.EPS_METHODS:
        c = cov.get(("eps", m))
        rows.append([m + (" ¹" if m in r3.USES_TRUTH else ""), f2(_get(tb, "eps", m)),
                     f2(_get(tb, "eps", m, col="mean_deg")), f2(_get(tb, "eps", m, col="p95_deg")),
                     f2(_get(tb, "eps", m, unit="HipsAlt"))]
                    + ([f"{100 * c[0] / c[2]:.0f} % / {100 * c[1] / c[2]:.0f} %"] if c else ["–"]))  # fmt: skip
    txt += ["### 3.2 ε per pod (relative to the phone)", "",
            md_table(["method", "median", "mean", "P95", "median HipsAlt (the root link)",
                      "coverage 68 % / 95 %"], rows), "",
            "Per pod (median):", ""]  # fmt: skip
    rows = [[m] + [f2(_get(tb, "eps", m, unit=p.name)) for p in PODS] for m in ("B1", "RIGID+B1", "RIGID+GEO")]
    txt += [md_table(["method"] + [p.name for p in PODS], rows), ""]
    txt += ["### 3.3 Strata (median)", ""]
    for kind, title in (("since_boot_min", "time since boot (min)"), ("motion", "motion type"), ("slip", "slip"),
                        ("suit", "suit version"), ("wire", "wire mode"), ("hipsalt", "HipsAlt present")):
        vals = sorted(set(tb.loc[(tb["stratum"] == kind), "value"]))
        keys = [("delta", "B1"), ("delta", "GEO-ID"), ("eps", "RIGID+B1"), ("eps", "RIGID+GEO")]
        rows = [[v] + [f2(_get(tb, t, m, kind, v)) for t, m in keys]
                + [str(int(_get(tb, "delta", "B1", kind, v, col="n")) if np.isfinite(_get(tb, "delta", "B1", kind,
                                                                                         v, col="n")) else "–")]
                for v in vals]  # fmt: skip
        txt += [f"**{title}**", "", md_table([kind, "δ B1", "δ GEO-ID", "ε RIGID+B1", "ε RIGID+GEO", "n (δ B1)"],
                                             rows), ""]  # fmt: skip
    rows = [[f"{r.target}: {r.a} − {r.b}", r.n, f"{r.mean:+.2f}", f"{r.lo:+.2f} – {r.hi:+.2f}", f"{r.p:.4f}",
             f"{r.p_holm:.4f}"] for r in con.itertuples()]  # fmt: skip
    txt += ["### 3.4 Paired contrasts over subjects (§8.5)", "",
            f"Each held-out subject's mean error per method; the difference a − b; {n_boot:,} bootstrap resamples of "
            "the subjects; Holm across the four contrasts.", "",
            md_table(["contrast (degrees)", "subjects", "mean", "95 % interval", "p", "p (Holm)"], rows), ""]
    return "\n".join(txt)


def section_ablations(pooled_all: dict) -> str:
    rows = []
    for abl in [None, *r3.ABLATIONS]:
        p = pooled_all.get(abl)
        if p is None:
            continue
        tb = r3.tables(p)
        rows.append([abl or "the twin (surface frames, VQF)"] + [f2(_get(tb, "delta", m)) for m in ("B0", "B1",
                                                                                                   "O1")]
                    + [f2(_get(tb, "eps", m)) for m in ("B1", "RIGID+B1", "O1")])  # fmt: skip
    return "\n".join(["### 3.5 The twin ablations (§8.1)", "",
                      "The same sessions built with Step 1's segment frames instead of surface frames, and with "
                      "Step 1's analytic heading model instead of VQF in the loop (median, degrees):", "",
                      md_table(["twin", "δ B0", "δ B1", "δ O1 ¹", "ε B1", "ε RIGID+B1", "ε O1 ¹"], rows), ""])


def section_retro(real: dict | None) -> str:
    """§8.4's "before" measures on the recordings (descriptive): Φ3 of the shins drawn against the OptiTrack truth on
    Nov 13; Φ3 of both legs and Φ7 on Nov 14. The converter and twin_v1_bvh (a Should) are not built."""
    head = ["## 6. The retro path's \"before\" measures (§8.4, on the recordings: descriptive)", ""]
    if real is None:
        return "\n".join(head + ["No `real-stats` summary was given or found.", ""])
    rows = []
    for rid in sorted(real):
        r = real[rid]
        d, t = r.get("phi3_drawn_deg") or {}, r.get("phi3_truth_deg") or {}
        if not d:
            continue
        rows.append([rid] + [f2(d.get(b)) for b in ("LeftLeg", "RightLeg")]
                    + [f2(t.get(b)) if t else "–" for b in ("LeftLeg", "RightLeg")]
                    + [f2(r.get("phi7_deg")), f2(r.get("root_height_cm"))])  # fmt: skip
    return "\n".join(head + [
        "Φ3 is the drawn shin's swing direction against the travel (axial, degrees); the truth is OptiTrack's on "
        "Nov 13 and absent on Nov 14. These are what a corrected redraw would be measured against; the "
        "BVH-to-virtual-pod converter, `twin_v1_bvh` and the before-and-after command (a Should) are not built yet.",
        "",
        md_table(["recording", "Φ3 left shin", "Φ3 right shin", "truth left", "truth right", "Φ7", "root height (cm)"],
                 rows), ""])


def fig_growth(tb: pd.DataFrame, out: Path) -> Path:
    import matplotlib.pyplot as plt

    plotstyle.apply()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=False)
    x = [f"{b:g}-{b + 5:g}" for b in r3.BOOT5_MIN]
    for ax, target, methods in ((axes[0], "delta", ("B0", "B1", "O1", "GEO-ID")),
                                (axes[1], "eps", ("B1", "RIGID+B1", "RIGID+GEO", "O1"))):  # fmt: skip
        for k, m in enumerate(methods):
            y = [_get(tb, target, m, "boot_5min", v) for v in x]
            ax.plot(np.arange(len(x)), y, marker="o", ms=3, lw=1.4, color=plotstyle.SERIES[k], label=m)
        ax.set_xticks(np.arange(len(x)), x)
        ax.set_xlabel("time since boot (min)")
        ax.set_ylabel("median |error| (°)")
        ax.set_title("δ per joint" if target == "delta" else "ε per pod")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="twin_v1")
    ap.add_argument("--dataset-root", help="a dataset folder (default $MESQ_OUTPUT_DIR/datasets/<dataset>)")
    ap.add_argument("--calibration", help="the twin-calibrate folder (default: the one the dataset names)")
    ap.add_argument("--real-summary", help="a real-stats summary.json (default: the newest under $MESQ_OUTPUT_DIR)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-sessions", type=int)
    ap.add_argument("--no-ablations", action="store_true")
    ap.add_argument("--no-page", action="store_true")
    ap.add_argument("--page-sessions", type=int, default=12)
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--out", default=str(ROOT / "docs" / "RESULTS_STEP3.md"))
    ap.add_argument("--figdir", default=str(ROOT / "docs" / "figures"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    root = Path(args.dataset_root) if args.dataset_root else output_dir() / "datasets" / args.dataset
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8")) if (root / "meta.json").is_file() else {}
    cal_dir = Path(args.calibration) if args.calibration else (
        output_dir() / meta["calibration"]["folder"] if meta.get("calibration", {}).get("folder") else None)
    cal = json.loads((cal_dir / "calibration.json").read_text(encoding="utf-8")) if cal_dir and (
        cal_dir / "calibration.json").is_file() else None
    cfgt = load_twin_config()
    found = sorted(output_dir().glob("real_stats_g*/summary.json"), key=lambda p: p.stat().st_mtime)
    real_path = Path(args.real_summary) if args.real_summary else (found[-1] if found else None)
    real = json.loads(real_path.read_text(encoding="utf-8"))["results"] if real_path and real_path.is_file() else None

    rows = r3.test_rows(root, args.split, args.max_sessions)
    tasks = [(row, cond, None) for row, cond in rows]
    if not args.no_ablations:
        tasks += [(row, cond, abl) for abl in r3.ABLATIONS for row, cond in rows]
    with mp.get_context("spawn").Pool(processes=args.workers, maxtasksperchild=4) as pool:  # memory: recycle
        results = list(pool.imap(r3.session_result, tasks, chunksize=1))
    pooled_all = r3.merge(results)
    pooled = pooled_all[None]
    tb = r3.tables(pooled)
    con = r3.contrasts(pooled, CONTRASTS, args.n_boot, args.seed)
    out_tables = output_dir() / "results_step3"
    out_tables.mkdir(parents=True, exist_ok=True)
    tb.to_csv(out_tables / f"baselines_{args.split}.csv", index=False)
    page_txt = ""
    if not args.no_page:
        from mesqsim.learn.through_page import page_section, page_test

        page = page_test(root, args.split, args.page_sessions, args.workers)
        page.to_csv(out_tables / f"page_{args.split}.csv", index=False)
        page_txt = page_section(page)
    figs = []
    if not args.no_figures:
        figdir = Path(args.figdir)
        figdir.mkdir(parents=True, exist_ok=True)
        figs.append(fig_growth(tb, figdir / "results_step3_growth.png"))

    run = (f"Run: {dt.datetime.now().isoformat(timespec='seconds')}; commit {git_commit()}; {platform.node()} "
           f"({platform.machine()}, Python {platform.python_version()}); {args.workers} workers; "
           f"{time.perf_counter() - t0:.0f} s.")
    doc = [f"# Step 3 results: the twin, its data, and the numbers to beat ({args.split} split)", "", banner(cfgt),
           "", run, "", section_calibration(cal), section_data(root), section_baselines(pooled, tb, con, args.n_boot)]
    if not args.no_ablations:
        doc.append(section_ablations(pooled_all))
    if figs:
        doc += [f"![median error against time since boot](figures/{figs[0].name})", ""]
    doc.append(page_txt or "## 4. Through the page (§8.3)\n\nNot run (`--no-page`).\n")
    doc += ["## 5. What no heading correction fixes", "",
            "A heading correction turns each pod about the vertical. It cannot undo the page's halved root (the hips "
            "settle at x* = (100·p + c)/2 because `initialPosition` aliases the hips' own position: a page defect, "
            "drawn the same with every method above), nor a mirrored tree map (a reflection is not a rotation about "
            "the vertical). Both need the page fixed, not a better estimate.", "", section_retro(real)]
    Path(args.out).write_text("\n".join(doc), encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
