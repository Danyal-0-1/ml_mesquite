"""Command line: ``python -m mesqsim <command>``.

    simulate   --config configs/default.yaml --seed 1 [--motion synthetic:walk:20 | path.npz] [--preset NAME]
               [--set key=value ...] [--out DIR]
    baselines  --run DIR                        decode with Step 0, score B0, B1, O1, O2, write results/
    params     [--config FILE] [--label assumption]   every parameter with its value, label and source
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
    from .pipeline import score_run

    summary = score_run(Path(args.run))
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
    labels = ("datasheet", "firmware", "assumption", "measured")
    counts = {lab: sum(1 for r in cfg.table() if r[2] == lab) for lab in labels}
    print(f"\n{len(cfg.table())} parameters: " + ", ".join(f"{v} {k}" for k, v in counts.items()))
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

    b = sub.add_parser("baselines", help="decode and score a run")
    b.add_argument("--run", required=True)
    b.set_defaults(fn=cmd_baselines)

    pa = sub.add_parser("params", help="list every parameter with its label and source")
    pa.add_argument("--config", default=str(DEFAULT_CONFIG))
    pa.add_argument("--label", choices=["datasheet", "firmware", "assumption", "measured"])
    pa.set_defaults(fn=cmd_params)

    lr = sub.add_parser("list-runs", help="list run folders")
    lr.set_defaults(fn=cmd_list_runs)

    c = sub.add_parser("clean", help="delete run folders")
    c.add_argument("--run")
    c.add_argument("--all", action="store_true")
    c.add_argument("--yes", action="store_true")
    c.set_defaults(fn=cmd_clean)

    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    return args.fn(args)
