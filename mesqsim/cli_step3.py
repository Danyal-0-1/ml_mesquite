"""Step 3's commands for ``python -m mesqsim`` (registered by :func:`add_parsers` from :mod:`mesqsim.cli`).

    real-stats [--recording ID ...]   Φ1-Φ12 of Danyal's recordings against the analysis of Step 3 brief §5.3;
                                      prints the comparison, writes $MESQ_OUTPUT_DIR/real_stats_g<commit>/summary.json
    twin-body [--files N]             skin against bone, specific force under each smoothing, sensor saturation,
                                      on AMASS walking of the training split; writes twin_body_g<commit>/report.json
    twin-fusion [--files N]           the fusion's own tilt and heading error on the same walking, with perfect and
                                      with drawn sensors, and how much gyroscope bias each filter removes; writes
                                      twin_fusion_g<commit>/report.json
    twin-calibrate [--draws N] [--keep Q] [--predictive N] [--workers W] [--real-summary FILE]
                                      the fit to the recordings (Step 3 brief §5.4-5.7): rejection ABC, posterior
                                      predictive checks for the twin and Step 1, the classifier test, gate G-T;
                                      writes twin_calibration_g<commit>/
    twin-dataset --calibration DIR [--name twin_v1] [--extend N] [--workers W]
                                      the training data (Step 3 brief §6) under $MESQ_OUTPUT_DIR/datasets/<name>/
    verify-dataset [--name twin_v1] [--recompute N]
                                      re-hash it against MANIFEST.csv, check its rules, re-simulate N sessions
    twin-session --seed S [--program protocol|random] [--subject SUBSET/ID] [--length-min M]
                 [--set twin.key=value ...] [--out DIR]
                                      one twin session end to end (Step 3 brief §4.8-4.11): the procedure and its
                                      slips, the pods, the phone, the page twin; writes the session's files and
                                      page/twin.bvh to DIR (default $MESQ_OUTPUT_DIR/twin_session_<program>_s<S>)

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


def cmd_twin_fusion(args) -> int:
    from .runs import git_commit
    from .twin.fusion_report import fusion_report

    rep = fusion_report(n_files=args.files)
    w = rep["walking"]
    print(f"{len(w['files'])} training-split walking files, {w['seconds']:.0f} s per site, filters at "
          f"{w['rate_hz']:.0f} Hz, positions smoothed at {w['cutoff_hz']:.0f} Hz")
    for mode, cases in w["modes"].items():
        for case, sites in cases.items():
            print(f"\n{mode}, {case} sensors: site, turns med, tilt error med/P95/max deg, |Δψ| P95/max deg, "
                  f"|Δψ end| per turn med/max deg")
            for name, s in sites.items():
                a, d, r = s["tilt_deg"], s["abs_dpsi_deg"], s["dpsi_per_turn_deg"]
                print(f"  {name:13s} {w['turns'][name]['median']:4.2f}  {a['median']:6.3f}/{a['p95']:6.3f}/"
                      f"{a['max']:6.3f}   {d['p95']:7.4f}/{d['max']:7.4f}   {r['median']:6.3f}/{r['max']:6.3f}")
    b = rep["bias_removal"]
    print(f"\nbias {b['bias_deg_s']} °/s; share of the vertical component estimated at {b['marks_s']} s:")
    for case in ("still", "turning"):
        for mode, r in b[case].items():
            print(f"  {case:8s} {mode:7s} {[round(x, 3) for x in r['vertical_share']]}  ψ end {r['psi_end_deg']:8.3f}° "
                  f"(without: {r['psi_end_without_estimate_deg']:.1f}°)")
    commit = git_commit()
    out = output_dir() / f"twin_fusion_g{commit[:7]}{'-dirty' if commit.endswith('+dirty') else ''}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep | {"commit": commit}, indent=1), encoding="utf-8")
    print(f"wrote {out / 'report.json'}")
    return 0


def cmd_twin_session(args) -> int:
    import time

    from .config import load_config, parse_override
    from .paths import REPO_ROOT
    from .twin.config import load_twin_config
    from .twin.program import join_check, make_program
    from .twin.session import simulate_twin_session, write_twin_session

    cfg1 = load_config(REPO_ROOT / "configs" / "default.yaml")
    cfgt = load_twin_config(overrides=dict(parse_override(s) for s in (args.set or [])))
    t0 = time.perf_counter()
    joined, model, label = make_program(cfgt, args.program, args.seed, args.subject, args.length_min)
    jc = join_check(joined)
    print(f"{label}: {len(joined.pieces)} pieces, {joined.seq.poses.shape[0] / joined.seq.fps:.1f} s; join check "
          f"{'passes' if jc['ok'] else 'FAILS'} ({jc['transition_max_rad_s']:.2f} rad/s at the joins, P99 of the "
          f"pieces {jc['p99_rad_s']:.2f})")
    s = simulate_twin_session(cfg1, cfgt, args.seed, joined, model, label)
    out = args.out or (output_dir() / f"twin_session_{args.program}_s{args.seed}")
    write_twin_session(out, s)
    c = s.meta["counts"]
    print(f"slips {s.plan.slips}; phone {s.plan.phone['settings']}, Zero request {s.plan.phone['zero_request_s']}")
    print(f"{c['frames']} frames ({c['radio_lost']} lost), {c['phone_messages']} phone messages, {c['zeros']} Zeros, "
          f"{len(s.labels_ticks)} tick labels; {time.perf_counter() - t0:.1f} s")
    print(f"wrote {out}")
    return 0 if jc["ok"] else 1


def cmd_twin_calibrate(args) -> int:
    from .real import analysis, recordings
    from .runs import git_commit
    from .twin.calibration import run_calibration

    if args.real_summary:
        real_summary = json.loads(open(args.real_summary, encoding="utf-8").read())
    else:
        if not recordings.available():
            print(f"the recordings are not under {recordings.real_suit_dir()}")
            return 1
        real_summary = {"results": {r.id: analysis.analyse(r) for r in recordings.RECORDINGS}}
    commit = git_commit()
    out = output_dir() / f"twin_calibration_g{commit[:7]}{'-dirty' if commit.endswith('+dirty') else ''}"
    out.mkdir(parents=True, exist_ok=True)
    rep = run_calibration(real_summary, args.draws, args.keep, args.predictive, args.workers, args.seed,
                          out_dir=out, resume=args.resume)  # fmt: skip
    tables = rep.pop("tables")
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    (out / "calibration.json").write_text(json.dumps(rep | {"commit": commit}, indent=1, default=float),
                                          encoding="utf-8")  # fmt: skip
    print(f"\nposterior ({rep['kept']} of {rep['draws']} draws; best distance {rep['best_distance']:.2f}, median "
          f"{rep['median_distance']:.2f}):")
    for k, v in rep["posterior"].items():
        if v["kind"] == "choice":
            print(f"  {k:46s} {v['posterior']}")
        else:
            print(f"  {k:46s} median {v['median']:.4g}, 90 % {v['p5']:.4g}-{v['p95']:.4g} (prior {v['prior']}, "
                  f"width {100 * v['width_share']:.0f} % of the prior's)")
    g = rep["gate"]
    a_t, a_s = rep["auc_twin"], rep["auc_step1"]
    print(f"coverage: twin {g['twin_inside']} of {g['of']}, Step 1 {g['step1_inside']} of {g['of']}; AUC twin "
          f"{g['twin_auc']:.3f} ({a_t['p5']:.3f}-{a_t['p95']:.3f}), Step 1 {g['step1_auc']:.3f} "
          f"({a_s['p5']:.3f}-{a_s['p95']:.3f}); gate G-T {'passes' if g['passes'] else 'FAILS'}")
    print(f"{rep['seconds']:.0f} s; wrote {out}")
    return 0


def add_parsers(sub) -> None:
    """Add Step 3's subcommands to the ``python -m mesqsim`` parser."""
    rs = sub.add_parser("real-stats", help="Φ1-Φ12 of the recordings against Step 3 brief §5.3")
    rs.add_argument("--recording", nargs="+", help="recording ids (default: all seven)")
    rs.set_defaults(fn=cmd_real_stats)
    tb = sub.add_parser("twin-body", help="skin against bone and smoothed forces on AMASS walking (S3-M3)")
    tb.add_argument("--files", type=int, default=12, help="training-split walking files (default 12)")
    tb.set_defaults(fn=cmd_twin_body)
    tf = sub.add_parser("twin-fusion", help="the fusion's own error on AMASS walking, and bias removal (S3-M4)")
    tf.add_argument("--files", type=int, default=12, help="training-split walking files (default 12)")
    tf.set_defaults(fn=cmd_twin_fusion)
    ts = sub.add_parser("twin-session", help="one twin session end to end (S3-M5)")
    ts.add_argument("--seed", type=int, required=True)
    ts.add_argument("--program", choices=("protocol", "random"), default="protocol")
    ts.add_argument("--subject", help="SUBSET/ID of a training-split subject (random program; default from the seed)")
    ts.add_argument("--length-min", type=float, help="the random program's length (default from the seed)")
    ts.add_argument("--set", action="append", help="override one twin parameter: twin.key=value (repeatable)")
    ts.add_argument("--out", help="the session folder")
    ts.set_defaults(fn=cmd_twin_session)
    tc = sub.add_parser("twin-calibrate", help="fit the twin to the recordings, predictive checks, gate G-T (S3-M6)")
    tc.add_argument("--draws", type=int, default=200, help="prior draws (default 200)")
    tc.add_argument("--keep", type=float, default=0.1, help="the share of draws kept as the posterior (default 0.1)")
    tc.add_argument("--predictive", type=int, default=10, help="posterior predictive sessions per recording")
    tc.add_argument("--workers", type=int, default=8)
    tc.add_argument("--seed", type=int, default=1)
    tc.add_argument("--real-summary", help="a real-stats summary.json (default: analyse the recordings now)")
    tc.add_argument("--resume", help="a twin-calibrate folder: reuse its fit's simulations, redo everything after")
    tc.set_defaults(fn=cmd_twin_calibrate)
