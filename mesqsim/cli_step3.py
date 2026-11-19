"""Step 3's commands for ``python -m mesqsim`` (registered by :func:`add_parsers` from :mod:`mesqsim.cli`).

    real-stats [--recording ID ...]   Φ1-Φ12 of Danyal's recordings against the analysis of Step 3 brief §5.3;
                                      prints the comparison, writes $MESQ_OUTPUT_DIR/real_stats_g<commit>/summary.json
    twin-body [--files N]             skin against bone, specific force under each smoothing, sensor saturation,
                                      on AMASS walking of the training split; writes twin_body_g<commit>/report.json

They live in their own module so that adding one never moves a line of ``mesqsim/cli.py``, to which the Step 2
explanations link by line number.
"""

from __future__ import annotations

import json

from .paths import output_dir

__all__ = ["add_parsers", "cmd_real_stats"]


def _fmt(v) -> str:
    if isinstance(v, (tuple, list)):
        return "(" + ", ".join(_fmt(x) for x in v) + ")"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 1 else f"{v:.1f}"
    return str(v)


def cmd_real_stats(args) -> int:
    from .real import analysis, recordings
    from .runs import git_commit

    if not recordings.available():
        print(f"the recordings are not under {recordings.real_suit_dir()} (see manifests/real_suit.csv)")
        return 1
    recs = [r for r in recordings.RECORDINGS if not args.recording or r.id in args.recording]
    results = {}
    for r in recs:
        print(f"analysing {r.id} ...", flush=True)
        results[r.id] = analysis.analyse(r)
    rows = analysis.compare(results)
    print(f"\n{'statistic':30s} {'recording':11s} {'value':>26s} {'reference':>22s}  tolerance       ok")
    for x in rows:
        flag = {True: "yes", False: "NO", None: "-"}[x["ok"]]
        print(f"{x['statistic']:30s} {x['recording']:11s} {_fmt(x['value']):>26s} {_fmt(x['reference']):>22s}  "
              f"{x['tolerance']:15s} {flag}")
    n_ok = sum(1 for x in rows if x["ok"])
    print(f"\n{n_ok} of {len(rows)} reference values reproduced within tolerance")
    commit = git_commit()
    out = output_dir() / f"real_stats_g{commit[:7]}{'-dirty' if commit.endswith('+dirty') else ''}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"commit": commit, "results": results, "comparison": rows}, f, indent=1, default=float)
    print(f"wrote {out / 'summary.json'}")
    return 0


def cmd_twin_body(args) -> int:
    from .runs import git_commit
    from .twin.body_report import body_report

    rep = body_report(n_files=args.files)
    print(f"{len(rep['files'])} training-split walking files, {rep['seconds']:.0f} s, IMU at {rep['rate_hz']:.0f} Hz")
    cuts = list(next(iter(rep["sites"].values()))["force_m_s2"])
    head = "  ".join(f"|f| {c:>4s} Hz med/P99" for c in cuts)
    print(f"{'site':13s} skin-bone med/P95/max deg   {head}   saturated (8 Hz): accel ±4 g, gyro ±2000 °/s")
    for name, s in rep["sites"].items():
        a = s["skin_against_bone_deg"]
        f = "  ".join(f"{s['force_m_s2'][c]['median']:7.2f}/{s['force_m_s2'][c]['p99']:6.2f}   " for c in cuts)
        z = s["saturated_share_at_8hz"]
        print(f"{name:13s} {a['median']:5.2f}/{a['p95']:5.2f}/{a['max']:5.2f}            {f}  "
              f"{100 * z['accel_4g']:6.3f} %  {100 * z['gyro_2000dps']:6.3f} %")
    commit = git_commit()
    out = output_dir() / f"twin_body_g{commit[:7]}{'-dirty' if commit.endswith('+dirty') else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep | {"commit": commit}, indent=1), encoding="utf-8")
    print(f"wrote {out / 'report.json'}")
    return 0


def add_parsers(sub) -> None:
    """Add Step 3's subcommands to the ``python -m mesqsim`` parser."""
    rs = sub.add_parser("real-stats", help="Φ1-Φ12 of the recordings against Step 3 brief §5.3")
    rs.add_argument("--recording", nargs="+", help="recording ids (default: all seven)")
    rs.set_defaults(fn=cmd_real_stats)
    tb = sub.add_parser("twin-body", help="skin against bone and smoothed forces on AMASS walking (S3-M3)")
    tb.add_argument("--files", type=int, default=12, help="training-split walking files (default 12)")
    tb.set_defaults(fn=cmd_twin_body)
