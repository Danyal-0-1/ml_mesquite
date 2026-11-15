"""Reproduce docs/RESULTS_STEP1.md and its figures from one command (brief §7 M5).

    python scripts/make_results.py                 # AMASS if manifests/split_manifest.csv exists, else synthetic
    python scripts/make_results.py --source synthetic --subjects-per-kind 4
    python scripts/make_results.py --source amass --max-sequences 60

Steps:
1. the default dataset: every source sequence's windows, motion starting 1, 10 and 30 min after boot;
2. one-at-a-time sweeps around the default (motion at 10 min, one window per sequence) over the values listed
   under ``sweep`` in configs/default.yaml. Every sweep value uses the same session seeds (common random numbers),
   so the differences come from the swept parameter alone;
3. decode every session with Step 0, score B0, B1, O1 and O2, write the tables, figures and the Markdown.

Run folders are reused when they already hold scored results for the same (commit, config, seed); --fresh forces
regeneration. Every number in the Markdown comes from these runs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from mesqsim import io, results  # noqa: E402
from mesqsim.config import load_config  # noqa: E402
from mesqsim.dataset import amass_sources, generate, synthetic_sources  # noqa: E402
from mesqsim.paths import manifest_dir  # noqa: E402
from mesqsim.pipeline import score_run  # noqa: E402
from mesqsim.runs import git_commit, run_folder  # noqa: E402

SPLIT_MANIFEST = manifest_dir() / "split_manifest.csv"
KEY_PARAMS = (
    "error_model.offset_mode", "error_model.box_slot_deg", "error_model.drift_sigma_deg_per_min",
    "error_model.scale_factor_max", "error_model.arw_deg_per_sqrt_s", "error_model.tilt_sigma_deg",
    "error_model.reboot_rate_per_h", "error_model.idle_turns_sigma_per_sqrt_min", "timing.stall_ms",
    "radio.loss_mean", "radio.burst_mean_frames", "dataset.window_s", "dataset.motion_start_min",
)  # fmt: skip


def runs_commit(arg: str | None) -> str:
    """The commit whose run folders hold the simulated and scored sessions: HEAD (with +dirty) by default.

    ``--runs-commit C`` reuses the runs made at an earlier commit C. It is allowed only when nothing that simulates
    or scores (``mesqsim/``, ``configs/``) differs between C and the working tree, so C's runs are exactly what the
    current code would produce (check 8.1) and only the reporting changed. It never simulates under C's name.
    """
    if arg is None:
        return git_commit()
    git = ["git", "-C", str(ROOT)]
    full = subprocess.run([*git, "rev-parse", "--verify", f"{arg}^{{commit}}"], capture_output=True, text=True,
                          check=True).stdout.strip()  # fmt: skip
    changed = subprocess.run([*git, "diff", "--quiet", full, "--", "mesqsim", "configs"]).returncode != 0
    untracked = subprocess.run([*git, "ls-files", "--others", "--exclude-standard", "--", "mesqsim", "configs"],
                               capture_output=True, text=True, check=True).stdout.strip()  # fmt: skip
    if changed or untracked:
        raise SystemExit(f"--runs-commit {arg}: mesqsim/ or configs/ differ from that commit; run without it")
    return full


def scored_run(cfg, seed, sources, fresh: bool, log, run_commit: str, reuse_only: bool) -> tuple[Path, pd.DataFrame]:
    folder = run_folder(run_commit, cfg.hash(), seed)
    done = folder / "results" / "errors.parquet"
    if fresh or not done.is_file():
        if reuse_only:
            raise SystemExit(f"--runs-commit: no scored run {folder.name}; run without --runs-commit")
        generate(cfg, seed, sources, out_dir=folder, log=lambda s: None)
        score_run(folder)
        log(f"  simulated and scored {folder.name}")
    else:
        log(f"  reused {folder.name}")
    return folder, io.read_parquet(done)


def md_table(df: pd.DataFrame, floatfmt=".2f") -> str:
    return io.markdown_table(df, floatfmt)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["amass", "synthetic"], default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--subjects-per-kind", type=int, default=4, help="synthetic: pseudo-subjects per motion kind")
    ap.add_argument("--max-sequences", type=int, default=None, help="AMASS: cap on the number of sequences")
    ap.add_argument("--no-sweeps", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="regenerate even if scored runs exist")
    ap.add_argument("--runs-commit", default=None,
                    help="reuse the scored runs made at this earlier commit (only if mesqsim/ and configs/ are "
                         "unchanged since); simulates nothing")  # fmt: skip
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if args.runs_commit and args.fresh:
        ap.error("--runs-commit reuses runs; it cannot be combined with --fresh")

    source = args.source or ("amass" if SPLIT_MANIFEST.is_file() else "synthetic")
    sources = amass_sources(SPLIT_MANIFEST) if source == "amass" else synthetic_sources(args.subjects_per_kind)
    if args.max_sequences:
        sources = sources.sort_values("path").head(args.max_sequences)
    cfg = load_config(ROOT / "configs" / "default.yaml")
    commit = git_commit()
    run_commit = runs_commit(args.runs_commit)
    reuse_only = args.runs_commit is not None
    print(f"source: {source}, {len(sources)} sequences; commit {commit[:12]}; runs from {run_commit[:12]}; "
          f"config {cfg.hash()[:8]}; seed {args.seed}")

    def scored(c):
        return scored_run(c, args.seed, sources, args.fresh, print, run_commit, reuse_only)

    print("default dataset")
    folder, errors = scored(cfg)
    overall = results.overall_table(errors)
    secondary = results.overall_table(errors, "err_orient_deg")
    by_type = results.joint_type_table(errors)
    by_joint = results.joint_table(errors)
    out_res = folder / "results"
    for name, df in (("overall", overall), ("secondary", secondary), ("joint_type", by_type), ("joint", by_joint)):
        df.to_csv(out_res / f"table_{name}.csv", index=False, float_format="%.6g", lineterminator="\n")
    dataset = pd.read_csv(folder / "dataset.csv", dtype={"subject": str})
    # On AMASS an error table holds about 10^8 rows (125.7 M for the default dataset, 27.5 M per sweep value).
    # Keep only the summaries: holding every sweep's full table ran a 64 GB machine out of memory on 2026-11-14.
    del errors

    sweep_parts, sweep_runs = [], []
    if not args.no_sweeps:
        base = cfg.with_overrides({"dataset.motion_start_min": [10], "dataset.windows_per_sequence": 1})
        for p in results.SWEEP_PARAMS:
            for v in cfg.param(p)["sweep"]:
                print(f"sweep {p} = {v}")
                f, e = scored(base.with_overrides({p: v}))
                sweep_parts.append(results.sweep_table({(p, v): e}))
                del e
                sweep_runs.append(f.name)
        sweeps = pd.concat(sweep_parts, ignore_index=True)
        sweeps.to_csv(out_res / "table_sweeps.csv", index=False, float_format="%.6g", lineterminator="\n")

    figs = ROOT / "docs" / "figures"
    label = "SYNTHETIC motion (not AMASS)" if source == "synthetic" else "AMASS motion"
    results.fig_growth(overall, f"Relative-heading error vs time since boot: {label}, default parameters",
                       figs / "results_growth.png")  # fmt: skip
    if sweep_parts:
        results.fig_sweeps(sweeps, f"One-at-a-time sweeps, motion at 10 min since boot: {label}",
                           figs / "results_sweeps.png")  # fmt: skip

    write_markdown(source, cfg, commit, run_commit, args, folder, dataset, overall, secondary, by_type,
                   sweeps if sweep_parts else None, sweep_runs)  # fmt: skip
    print("wrote docs/RESULTS_STEP1.md")
    return 0


def write_markdown(source, cfg, commit, run_commit, args, folder, dataset, overall, secondary, by_type, sweeps,
                   sweep_runs):  # fmt: skip
    date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC")
    synthetic = source == "synthetic"
    labels = Counter(node["label"] for _, node in cfg.leaves())
    lines = ["# Step 1 results: baselines B0, B1, O1, O2", ""]
    if synthetic:
        lines += [
            "> **PROVISIONAL: SYNTHETIC MOTION, NOT AMASS.** These numbers come from the simulator driven by simple",
            "> synthetic walking, turning-in-a-circle and standing clips (`mesqsim/synth.py`), because AMASS was not",
            "> downloaded yet. They show that the pipeline runs end to end and how large each baseline's error is",
            "> under the **assumed** parameters. They are not results about human motion. Rerun",
            "> `python scripts/make_results.py` after the AMASS download to replace them.",
            "",
        ]
    else:
        sha = SPLIT_MANIFEST.with_suffix(".sha256")
        digest = sha.read_text(encoding="utf-8").split()[0] if sha.is_file() else "not written"
        lines += [
            f"> **AMASS motion, assumed parameters.** {labels.get('assumption', 0)} of the {sum(labels.values())} "
            "parameters are assumptions until the bench measures them",
            "> (`python -m mesqsim params`), so these numbers describe the configured model of the suit, not the",
            f"> real suit. Split manifest SHA-256 `{digest}`.",
            "",
        ]
    flags = f"--source {source} --seed {args.seed}" + (f" --runs-commit {run_commit[:7]}" if args.runs_commit else "")
    run_line = f"commit `{commit[:12]}`"
    if run_commit != commit:
        run_line = (f"runs simulated and scored at commit `{run_commit[:12]}` (mesqsim/ and configs/ unchanged since), "
                    f"report written at commit `{commit[:12]}`")  # fmt: skip
    lines += [
        f"*Generated by `python scripts/make_results.py` (`{flags}`). Do not edit by hand.*",
        "",
        f"**Run:** {date}; {run_line}; config hash `{cfg.hash()[:12]}`; seed {args.seed}; default run "
        f"folder `{folder.name}`.",
        "",
        "## Picture first",
        "",
        "Every joint's relative heading δ (how far the child pod's north has slid from its parent's) is estimated by",
        "four baselines and compared with the truth at every 32 Hz step of the motion:",
        "",
        "- **B0** does nothing;",
        "- **B1** registers the pods in the box;",
        "- **O1** knows the true initial offsets;",
        "- **O2** knows the truth.",
        "",
        "The motion starts 1, 10 or 30 minutes after the pods were switched on, so the tables show how the error",
        "grows with time since boot.",
        "",
        "## Parameter set",
        "",
        "Default configuration ([configs/default.yaml](../configs/default.yaml)); all 40+ parameters with labels:",
        "`python -m mesqsim params`. The ones that drive these numbers:",
        "",
        "| parameter | value | label |",
        "|---|---|---|",
    ]
    for k in KEY_PARAMS:
        prm = cfg.param(k)
        lines.append(f"| `{k}` | {prm['value']} | {prm['label']} |")
    # A sequence is (subset, subject, file name): KIT reuses file names such as walking_slow01_poses.npz across
    # subjects. A subject is (subset, subject id): CMU and KIT reuse ids for different people.
    ds = dataset.astype({"subject": str})
    seqs = ds.drop_duplicates(["subset", "subject", "source"])
    subjects = ds.drop_duplicates(["subset", "subject"])
    per_split = ", ".join(f"{k} {n} / {subjects['split'].value_counts().get(k, 0)}"
                          for k, n in seqs["split"].value_counts().sort_index().items())  # fmt: skip
    if synthetic:
        unit = ("Steps within a session are strongly correlated, so the independent unit is the **session** (column "
                "`sessions`): differences between baselines smaller than a few tenths of a degree are within sampling "
                "noise.")  # fmt: skip
    else:
        unit = ("Steps within a session are strongly correlated, and sessions of one subject share that person's "
                "motion, so the independent unit is the **subject**; the `sessions` column counts sessions, not "
                "independent observations. Step 1 computes no intervals; the Step 2 analysis plan bootstraps over "
                "subjects.")  # fmt: skip
    lines += [
        "",
        f"**Data:** {len(ds)} sessions from {len(seqs)} source sequences of {len(subjects)} subjects ({source}); "
        f"per split, sequences / subjects: {per_split}. Baselines have no fitted parameters, so every split is "
        "scored; nothing is trained in Step 1. Errors are over the motion phase, all 14 joints, every 32 Hz step. "
        + unit,
        "",
        "## Table 1: primary metric, relative-heading error |wrap(δ̂ − δ)| (degrees)",
        "",
        md_table(overall.rename(columns={"minutes_since_boot": "min since boot"})),
        "## Table 2: by joint type (mean / P95, degrees)",
        "",
    ]
    pivot = by_type[by_type["baseline"].isin(["B0", "B1", "O1"])].copy()
    pivot["cell"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(pivot["mean"], pivot["p95"], strict=True)]
    wide = pivot.pivot_table(index=["joint_type", "baseline"], columns="minutes_since_boot", values="cell",
                             aggfunc="first").reset_index()  # fmt: skip
    wide.columns = [f"{c} min" if isinstance(c, int | float) else str(c) for c in wide.columns]
    lines += [md_table(wide), "O2 is 0 everywhere by construction (check 12.1).", ""]
    lines += [
        "## Table 3: secondary metric, parent-relative orientation error (degrees, true mountings)",
        "",
        "Includes fusion tilt error and quantisation, so even O2 is not zero here.",
        "",
        md_table(secondary.rename(columns={"minutes_since_boot": "min since boot"})),
    ]
    if sweeps is not None:
        sw = sweeps[sweeps["baseline"].isin(["B0", "B1", "O1"])].copy()
        sw["cell"] = [f"{a:.2f} / {b:.2f}" for a, b in zip(sw["mean"], sw["p95"], strict=True)]
        wide = sw.pivot_table(index=["parameter", "value"], columns="baseline", values="cell", aggfunc="first",
                              sort=False).reset_index()  # fmt: skip
        lines += [
            "## Table 4: one-at-a-time sweeps (mean / P95 of the primary metric, degrees)",
            "",
            "Motion at 10 min since boot, one window per sequence. Every other parameter at its default; the same",
            "session seeds for every value. Run folders: " + ", ".join(f"`{r}`" for r in sweep_runs[:3]) + ", ….",
            "",
            md_table(wide),
        ]
    lines += [
        "## Figures",
        "",
        "![error growth](figures/results_growth.png)",
        "",
        "*Mean (solid) and P95 (dashed) relative-heading error of B0, B1 and O1 against minutes since boot. Log "
        "scale; O2 is 0.*",
        "",
    ]
    if sweeps is not None:
        lines += ["![sweeps](figures/results_sweeps.png)", "",
                  "*Mean error of B1 and O1 for each value of each swept parameter.*", ""]  # fmt: skip
    lines += [
        "## How to read these numbers",
        "",
        "- B0 against B1 shows what box registration buys: in the arbitrary offset mode almost everything; in the "
        "aligned mode almost nothing (Table 4, `offset_mode`).",
        "- O1 against B1 is the box registration's own error (slot error, plus what accrued before the box).",
        "- O1 against O2 is the drift, turning and noise that a Step 2 estimator must track. It grows with time since "
        "boot.",
        "- Every *assumption* row in the parameter table is a guess until Danyal's bench measurements replace it. The "
        "sweeps show how much each guess matters.",
    ]
    (ROOT / "docs" / "RESULTS_STEP1.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    sys.exit(main())
