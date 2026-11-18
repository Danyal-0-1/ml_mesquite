"""Command line: ``python -m mesqsim <command>``.

    simulate   --config configs/default.yaml --seed 1 [--motion synthetic:walk:20 | path.npz] [--preset NAME]
               [--set key=value ...] [--out DIR]
    baselines  --run DIR [--methods B0 B1 GEO-K GEO-ID ...]   decode with Step 0, score the methods, write results/
    params     [--config FILE] [--label assumption]   every parameter with its value, label and source
    split      [--write] [--force]              split AMASS subjects 70/15/15 -> manifests/split_manifest.csv
    dataset    --seed 1 [--source synthetic|amass] [--set ...]   simulate a dataset into a run folder and score it
    fit-geometry rom [--source amass|synthetic] [--write]   fit the range-of-motion boxes on the training split
    fit-geometry calibration [--source ...] [--write]       fit kappa scales, model errors, q, kappa_hold on val
    list-runs                                   runs under the output directory
    clean      --run DIR | --all --yes          delete run folders (only run_* folders in the output directory)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, parse_override
from .paths import REPO_ROOT, output_dir
from .runs import list_runs, remove_run

DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _config(args):
    cfg = load_config(args.config)
    for name in args.preset or []:
        cfg = cfg.with_preset(name)
    if args.set:
        cfg = cfg.with_overrides(dict(parse_override(s) for s in args.set))
    return cfg


def cmd_simulate(args) -> int:
    from .pipeline import simulate_run

    cfg = _config(args)
    folder = simulate_run(cfg, args.seed, args.motion, out_dir=Path(args.out) if args.out else None)
    print(folder)
    return 0


def cmd_baselines(args) -> int:
    from .evaluate import BASELINES, METHODS
    from .pipeline import score_run

    methods = BASELINES if not args.methods else (METHODS if args.methods == ["all"] else tuple(args.methods))
    unknown = set(methods) - set(METHODS)
    if unknown:
        print(f"unknown methods {sorted(unknown)}; known: {', '.join(METHODS)}")
        return 1
    summary = score_run(Path(args.run), baselines=methods)
    rows = summary[summary["joint_name"] == "all"]
    print(rows.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"results in {Path(args.run) / 'results'}")
    return 0


def cmd_params(args) -> int:
    cfg = load_config(args.config)
    rows = [r for r in cfg.table() if args.label is None or r[2] == args.label]
    width = max(len(k) for k, *_ in rows)
    for key, value, label, source in rows:
        print(f"{key:<{width}}  {str(value):<18} {label:<10} {source}")
    from .config import LABELS as labels

    counts = {lab: sum(1 for r in cfg.table() if r[2] == lab) for lab in labels}
    print(f"\n{len(cfg.table())} parameters: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
    return 0


def cmd_split(args) -> int:
    import pandas as pd

    from .data import amass
    from .paths import manifest_dir
    from .splits import assign_splits, check_manifest, manifest_sha256, write_manifest

    files = amass.motion_files()
    if not files:
        print(f"no AMASS motion files under {amass.amass_dir()}; see docs/DATA.md")
        return 1
    rows = []
    for f in files:
        subset, subject = amass.subset_subject(f)
        rows.append({"path": f.relative_to(amass.amass_dir()).as_posix(), "subset": subset, "subject": subject})
    df = assign_splits(pd.DataFrame(rows))
    problems = check_manifest(df)
    for p in problems:
        print("PROBLEM:", p)
    print(df.groupby(["subset", "split"])["subject"].nunique().unstack(fill_value=0).to_string())
    if not args.write:
        print("dry run: add --write to write manifests/split_manifest.csv")
        return 1 if problems else 0
    path, frozen = manifest_dir() / "split_manifest.csv", manifest_dir() / "split_manifest.sha256"
    if frozen.is_file() and not args.force:
        tmp = path.with_suffix(".new.csv")
        new_hash = write_manifest(df, tmp)
        tmp.unlink()
        if new_hash != frozen.read_text().split()[0]:
            print("REFUSED: the split manifest is frozen and the new one differs; --force to replace (ask Danyal)")
            return 1
    digest = write_manifest(df, path)
    if not frozen.is_file():
        frozen.write_text(f"{digest}  split_manifest.csv\n", encoding="utf-8", newline="\n")
    print(f"wrote {path} (SHA-256 {digest}, verified: {manifest_sha256(path) == digest})")
    return 1 if problems else 0


def cmd_dataset(args) -> int:
    from .dataset import amass_sources, generate, synthetic_sources
    from .paths import manifest_dir
    from .pipeline import score_run

    cfg = _config(args)
    manifest = manifest_dir() / "split_manifest.csv"
    if args.source == "amass":
        sources = amass_sources(manifest)
    else:
        sources = synthetic_sources(args.subjects_per_kind)
    folder = generate(cfg, args.seed, sources, max_sequences=args.max_sequences)
    summary = score_run(folder)
    print(summary[summary["joint_name"] == "all"].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(folder)
    return 0


def cmd_fit_geometry(args) -> int:
    """Fit the estimator's data-dependent beliefs on train (Omega); ``--write`` puts them into the config file."""
    import datetime as dt
    import json

    from .config import set_leaf_in_file
    from .dataset import amass_sources, synthetic_sources
    from .geo_fit import fit_rom, rom_source_note
    from .paths import manifest_dir
    from .runs import git_commit

    cfg = load_config(args.config)
    manifest = manifest_dir() / "split_manifest.csv"
    if args.what == "calibration":
        return _fit_calibration(args, cfg, manifest)
    if args.source == "amass":
        sources = amass_sources(manifest)
        sha = (manifest_dir() / "split_manifest.sha256").read_text(encoding="utf-8").split()[0]
        data = f"AMASS (split manifest SHA-256 {sha[:12]})"
    else:
        sources = synthetic_sources()
        data = "SYNTHETIC stand-in motion (provisional)"
    res = fit_rom(sources, int(cfg["motion.spine_joint"]), tuple(cfg["geometry.rom_percentiles"]))
    print(f"{res.n_sequences} training sequences, {res.n_frames:,} frames")
    print(json.dumps({"limits_deg": res.limits_deg, "knee_off_axis_rms_deg": res.knee_off_axis_rms_deg,
                      "third_axis_rms_deg": res.third_axis_rms_deg}, indent=1))  # fmt: skip
    if args.write:
        command = f"python -m mesqsim fit-geometry rom --source {args.source} --write"
        date = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d")
        note = rom_source_note(command, data, res.n_sequences, res.n_frames, cfg["geometry.rom_percentiles"],
                               git_commit(), date)  # fmt: skip
        from .estimators.anatomy import CARDAN_SEQ

        for joint, box in res.limits_deg.items():
            set_leaf_in_file(args.config, f"geometry.rom_limits_deg.{joint}", box, "fitted",
                             note.replace("{seq}", CARDAN_SEQ[joint]))  # fmt: skip
        print(f"wrote {len(res.limits_deg)} boxes into {args.config} (label fitted)")
    return 0


def _fit_calibration(args, cfg, manifest) -> int:
    """GEO-ID on the validation sessions of the dataset; kappa scales, model errors, q and kappa_hold fitted."""
    import datetime as dt
    import json

    from .config import set_leaf_in_file
    from .dataset import amass_sources, synthetic_sources
    from .geo_fit import calibration_worker, fit_calibration
    from .geo_sessions import dataset_jobs, run_jobs
    from .paths import manifest_dir, output_dir
    from .runs import git_commit

    if args.source == "amass":
        sources = amass_sources(manifest)
        sha = (manifest_dir() / "split_manifest.sha256").read_text(encoding="utf-8").split()[0]
        data = f"AMASS (split manifest SHA-256 {sha[:12]})"
    else:
        sources = synthetic_sources()
        data = "SYNTHETIC stand-in motion (provisional)"
    jobs = dataset_jobs(cfg, sources, args.seed, splits=("val",), max_sequences=args.max_sequences)
    for j in jobs:
        j["estimator"] = args.estimator
    print(f"{len(jobs)} validation sessions from {len({j['path'] for j in jobs})} sequences; seed {args.seed}; "
          f"GEO-{args.estimator}")
    results = run_jobs(calibration_worker, jobs, args.config)
    w = float(cfg["geometry.step_s"]) / float(cfg["geometry.window_s"])
    suffix = "_k" if args.estimator == "K" else ""
    sig0 = {t: float(cfg[f"geometry.model_sigma_deg{suffix}.{t}"]) for t in ("hinge", "2dof", "3dof")}
    fit = fit_calibration(results, w, sig0, hold_target_p95_deg=float(cfg["geometry.hold_target_p95_deg"]))
    commit = git_commit()
    out = output_dir() / f"fit_calibration_g{commit.split('+')[0][:7]}_{args.source}_s{args.seed}_{args.estimator}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(fit, indent=1), encoding="utf-8")
    print(json.dumps({k: fit[k] for k in ("kappa_scale", "model_sigma_deg", "fitted_how", "drift_q_rad2_per_s",
                                          "kappa_hold", "hold_rule", "hold_p95_deg",
                                          "certain_fraction", "coverage", "n_sessions", "n_steps")}, indent=1))
    print(f"report: {out / 'report.json'}")
    if args.write:
        command = (f"python -m mesqsim fit-geometry calibration --source {args.source} --seed {args.seed} "
                   f"--estimator {args.estimator} --write")
        n_subj = len({(j["subset"], j["subject"]) for j in jobs})
        note = (f"fitted by `{command}` on {data}, validation split only: {fit['n_sessions']} sessions of "
                f"{len({j['path'] for j in jobs})} sequences, {n_subj} subjects, GEO-{args.estimator}; "
                f"commit {commit[:12]}, "
                f"{dt.datetime.now(dt.UTC).strftime('%Y-%m-%d')}")
        for t in ("hinge", "2dof", "3dof"):
            set_leaf_in_file(args.config, f"geometry.kappa_scale{suffix}.{t}", round(fit["kappa_scale"][t], 6),
                             "fitted", f"{note}: maximum likelihood of the window errors under "
                                       "VM(0, s·k/(1 + k·sigma_m^2))")
            if fit["fitted_how"][t] == "scale+sigma":
                set_leaf_in_file(args.config, f"geometry.model_sigma_deg{suffix}.{t}",
                                 round(fit["model_sigma_deg"][t], 4), "fitted",
                                 f"{note}: maximum likelihood, jointly with kappa_scale{suffix}.{t}")
        if args.estimator == "K":  # q and kappa_hold are GEO-ID's (the deployable estimator), shared by GEO-K
            print(f"wrote GEO-K's calibration into {args.config} (label fitted)")
            return 0
        set_leaf_in_file(args.config, "geometry.drift_q_rad2_per_s", float(f"{fit['drift_q_rad2_per_s']:.4g}"),
                         "fitted", f"{note}: the log-grid value with the largest von Mises log-likelihood of the "
                                   "filter's outputs at every motion-phase 32 Hz step")  # fmt: skip
        target = cfg["geometry.hold_target_p95_deg"]
        rule = (f"the smallest threshold whose retained steps have P95 error <= {target:g} deg"
                if fit["hold_rule"] == "target" else
                f"no threshold reached P95 <= {target:g} deg, so the one with the smallest retained P95 "
                f"({fit['hold_p95_deg']:.2f} deg) keeping at least 10 % of steps")  # fmt: skip
        set_leaf_in_file(args.config, "geometry.kappa_hold", round(fit["kappa_hold"], 4), "fitted",
                         f"{note}: {rule}; {100 * fit['certain_fraction']:.1f} % of validation steps certain")
        print(f"wrote the calibration into {args.config} (label fitted)")
    return 0


def cmd_list_runs(args) -> int:
    runs = list_runs()
    print(f"{len(runs)} run(s) in {output_dir()}")
    for r in runs:
        n = len(list((r / "sessions").glob("*"))) if (r / "sessions").is_dir() else 0
        scored = "scored" if (r / "results" / "summary.csv").is_file() else "not scored"
        print(f"  {r.name}  sessions: {n}  {scored}")
    return 0


def cmd_clean(args) -> int:
    if args.run:
        remove_run(Path(args.run))
        print(f"deleted {args.run}")
        return 0
    if args.all:
        if not args.yes:
            print("refusing to delete every run without --yes")
            return 1
        for r in list_runs():
            remove_run(r)
            print(f"deleted {r}")
        return 0
    print("nothing to do: give --run DIR or --all --yes")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mesqsim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="simulate one session into a run folder")
    s.add_argument("--config", default=str(DEFAULT_CONFIG))
    s.add_argument("--preset", action="append", help="named override set from configs/presets.yaml (repeatable)")
    s.add_argument("--set", action="append", help="override one parameter: key=value (repeatable)")
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--motion", default="synthetic:walk:20", help="synthetic:<kind>[:<seconds>] or an AMASS .npz")
    s.add_argument("--out", help="run folder (default: named after commit, config hash and seed)")
    s.set_defaults(fn=cmd_simulate)

    b = sub.add_parser("baselines", help="decode and score a run (Step 1 baselines and Step 2 estimators)")
    b.add_argument("--run", required=True)
    b.add_argument("--methods", nargs="+", help="methods to score (default B0 B1 O1 O2; 'all' for every method)")
    b.set_defaults(fn=cmd_baselines)

    pa = sub.add_parser("params", help="list every parameter with its label and source")
    pa.add_argument("--config", default=str(DEFAULT_CONFIG))
    pa.add_argument("--label", choices=["datasheet", "firmware", "assumption", "measured", "fitted"])
    pa.set_defaults(fn=cmd_params)

    sp = sub.add_parser("split", help="assign AMASS subjects to train/val/test")
    sp.add_argument("--write", action="store_true")
    sp.add_argument("--force", action="store_true", help="replace a frozen manifest (only with Danyal's approval)")
    sp.set_defaults(fn=cmd_split)

    d = sub.add_parser("dataset", help="simulate a dataset and score it")
    d.add_argument("--config", default=str(DEFAULT_CONFIG))
    d.add_argument("--preset", action="append")
    d.add_argument("--set", action="append")
    d.add_argument("--seed", type=int, required=True)
    d.add_argument("--source", choices=["synthetic", "amass"], default="synthetic")
    d.add_argument("--subjects-per-kind", type=int, default=4)
    d.add_argument("--max-sequences", type=int)
    d.set_defaults(fn=cmd_dataset)

    fg = sub.add_parser("fit-geometry", help="fit the estimator's beliefs on train/validation (never test)")
    fg.add_argument("what", choices=["rom", "calibration"])
    fg.add_argument("--seed", type=int, default=1, help="run seed of the dataset sessions (calibration)")
    fg.add_argument("--max-sequences", type=int, default=None, help="cap on validation sequences (calibration)")
    fg.add_argument("--estimator", choices=["ID", "K"], default="ID",
                    help="calibrate GEO-ID (default; also q and kappa_hold) or GEO-K's own window scales")
    fg.add_argument("--config", default=str(DEFAULT_CONFIG))
    fg.add_argument("--source", choices=["amass", "synthetic"], default="amass")
    fg.add_argument("--write", action="store_true", help="write the fitted values into the config (label fitted)")
    fg.set_defaults(fn=cmd_fit_geometry)

    lr = sub.add_parser("list-runs", help="list run folders")
    lr.set_defaults(fn=cmd_list_runs)

    c = sub.add_parser("clean", help="delete run folders")
    c.add_argument("--run")
    c.add_argument("--all", action="store_true")
    c.add_argument("--yes", action="store_true")
    c.set_defaults(fn=cmd_clean)

    from .cli_step3 import add_parsers  # Step 3's commands live there, so the lines above stay where docs point
    add_parsers(sub)

    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return args.fn(args)
