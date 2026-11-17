# Pipeline

*Status: S2-M6. Every stage, from one session to the Step 1 and Step 2 results: Step 2's estimator stage (§7), its fitted beliefs (§8), DIODEM (§9) and the results (§10).*

## Picture first

A simulated session goes from human motion to a table of postcards, exactly as a real recording would. The labels
travel separately and are only opened by the scorer. Step 0 cannot tell whether it is reading the simulator or the
real logger.

```text
            ┌─────────────── simulation only ───────────────┐
AMASS .npz ─► segment orientations s_i(t) ─► session plan ─► chip samples, send grid, quantisation, radio
(or synthetic)                                                   │
                                                                 ├─► frames.parquet + session.json   (= real logger)
                                                                 └─► labels_ticks / labels_joints / meta.json
frames.parquet + session.json ─► STEP 0 ─► series.parquet ─► baselines (B0 B1 O1 O2) ─► errors ─► summary
                                                    labels ─────────────────────────────┘ (scorer only)
```

## Commands

| stage | command | output |
|---|---|---|
| simulate one session | `python -m mesqsim simulate --seed 1 [--motion synthetic:walk:20 \| path/to/seq_poses.npz] [--preset NAME] [--set key=value]` | `$MESQ_OUTPUT_DIR/run_g<commit>_c<config hash>_s<seed>/sessions/s000/` |
| decode and score | `python -m mesqsim baselines --run <run folder> [--methods B0 B1 GEO-K GEO-ID ... \| all]` | `series.parquet` per session; `results/errors.parquet`, `summary.csv`, `summary.md` |
| split AMASS subjects | `python -m mesqsim split --write` | `manifests/split_manifest.csv` and `.sha256` |
| simulate and score a dataset | `python -m mesqsim dataset --seed 1 --source amass\|synthetic` | run folder with `sessions/`, `dataset.csv`, `results/` |
| results tables and figures | `python scripts/make_results.py` | `docs/RESULTS_STEP1.md`, `docs/figures/results_*.png` |
| fit the range-of-motion boxes (train only) | `python -m mesqsim fit-geometry rom --source amass --write` | `geometry.rom_limits_deg.<joint>` in `configs/default.yaml`, label `fitted` |
| Step 2 results tables and figures | `python scripts/make_results_step2.py` | `docs/RESULTS_STEP2.md`, `docs/figures/results_step2_*.png` |
| fit GEO-K's own window calibration (validation only) | `python -m mesqsim fit-geometry calibration --source amass --seed 1 --estimator K --write` | `geometry.kappa_scale_k.*`, `geometry.model_sigma_deg_k.*` |
| fit the confidence, q and κ_hold (validation only) | `python -m mesqsim fit-geometry calibration --source amass --seed 1 --write` | `geometry.kappa_scale.*`, `geometry.model_sigma_deg.*`, `geometry.drift_q_rad2_per_s`, `geometry.kappa_hold` (label `fitted`); `$MESQ_OUTPUT_DIR/fit_calibration_g<commit>_amass_s1/report.json` |
| list runs | `python -m mesqsim list-runs` | |
| delete runs | `python -m mesqsim clean --run <folder>` or `clean --all --yes` | |

## Stage by stage

### 1. Motion source

- **In:** an AMASS `.npz` (`poses` (N, ≥66) axis-angle radians, frame-rate key) or `synthetic:<kind>[:<seconds>]`
  (`tpose`, `stand`, `walk`, `circle`; synthetic motion is for tests and examples only).
- **Out:** `MotionSource`: segment orientations S_i (N, 15, 4) `(x, y, z, w)`, segment → world, +Z up, at the source
  frame rate. Code: `mesqsim/sim/motion.py`, `mesqsim/data/amass.py`.

### 2. Session plan

- **In:** config (`configs/default.yaml` plus presets and overrides), seed, motion source.
- **Out:** `SessionPlan`: phase times (true seconds from the first power-on), host-clock epoch, box heading, T-pose
  and motion facing, and every pod's drawn parameters: boot time, clock ppm, chip rate, chip start, first-send
  ms, mounting m_i, box slot ε_i, offset a_i, strapping window and turns, posture error, battery. Stored in
  `meta.json → plan`. Code: `mesqsim/sim/session.py`.

### 3. Truth, timing, reports, radio

- True pod orientation P_i(t) = s_i(t)·m_i (box: Y(box heading + ε_i), flat). Code: `sim/truth.py`.
- Chip samples at the pod's own rate, decode delay, the 32 Hz send grid, newest sample, held repeats, `count`,
  `ms_lo`. Code: `sim/timing.py`.
- ψ_i(t) for every pod and boot epoch: offset, drift, scale factor × turning Θ, random walk.
  Code: `sim/heading.py`.
- Report q_i = Y(−ψ_i)·P_i·Exp(η_i), canonical (w ≥ 0), quantised as the firmware does. Code: `sim/simulate.py`,
  `codec.quantize_component`.
- Reboots split a pod's sends into boot epochs: count, clock and north restart. Code: `sim/simulate.py`.
- Stalls create held frames. Code: `sim/timing.py`.
- Gilbert–Elliott bursts drop frames. Arrival = send + latency, first-in first-out per pod; host time = arrival +
  host epoch. Code: `sim/radio.py`.
- A frame enters the frames table if it arrived inside a recorded phase (the idle interval is not recorded).

### 4. Files of a session (`sessions/<id>/`)

**`frames.parquet`** (the contract with Danyal's logger; checked by `io.validate_frames`): one row per received
frame, sorted by `t_host_s`.

| column | type | meaning |
|---|---|---|
| `t_host_s` | float64 | host arrival time, seconds, arbitrary epoch; **never** used as sample time |
| `pod_id` | uint8 | 0–14 (15, 16 exist in firmware, not simulated) |
| `batt` | uint8 | battery percent, 0–100 |
| `qx`, `qy`, `qz`, `qw` | int16 | raw wire values, trunc(float32(v)·32767), qw ≥ 0 |
| `count` | uint16 | send counter, wraps 65535 → 0 |
| `ms_lo` | uint16 | pod milliseconds at the sample's decode, mod 65,536 |

**`session.json`** (the logger's sidecar): `{"schema_version": 1, "pods": [...], "phases": [{"name",
"t_host_start_s", "t_host_end_s", "recorded"}, ...]}` with phases `box`, `strapping`, `tpose`, `idle`
(`recorded: false` in simulation), `motion`. A real logger that records continuously marks every phase recorded.

**`labels_ticks.parquet`** (simulation only; never read by Step 0 or an estimator): one row per pod and send tick
the host can know about (see "knowable ticks" in [MATH.md](MATH.md)).

| column | meaning |
|---|---|
| `pod_id`, `epoch`, `tick`, `count` | identity of the send: **observed** boot epoch (boots that left at least one received frame, numbered as Step 0 sees them), sends since boot, wire counter |
| `boot_epoch` | the true boot index (two reboots inside the unrecorded idle look like one to any logger) |
| `phase` | phase at the send time |
| `t_send_s`, `t_arrive_s`, `t_sample_s`, `t_decode_s` | true times, seconds since the first power-on |
| `sample_index` | index of the chip sample sent |
| `ms_lo` | the wire value |
| `received` | in the frames table |
| `fresh` | the sample differs from the previous tick's (sender's view) |
| `new_to_host` | received and different from the last received sample (what the host can tell) |
| `psi_rad` | true heading ψ_i of the pod's private frame at the sample time (unwrapped radians) |
| `pod_qx..pod_qw` | true pod orientation P_i (pod → world) at the sample time |

**`labels_joints.parquet`** (simulation only): one row per joint and 32 Hz step.

| column | meaning |
|---|---|
| `step`, `t_s`, `phase` | step index, true time, phase |
| `joint` | 0–13 in `mesqsim.body.JOINTS` order |
| `delta_rad` | true relative heading δ_pc = wrap(ψ_c − ψ_p) at `t_s`; NaN while either pod is off (rebooting) |
| `delta_offset_rad` | the offset part wrap(a_c − a_p) of the current boot epochs (the O1 oracle) |

**`meta.json`** (simulation only): `run` (commit, config hash, seed), `config` (resolved values),
`config_parameters` (value, label, source of each), `packages`, `source` (motion, fps, duration), `split`, and:

- `plan`: the timeline and every pod's drawn parameters (boot, clock, chip rate, mounting, slot error, offset,
  drift, scale factor, reboot times);
- `epochs`: per pod, the boot, end, first sample and north of each boot epoch;
- `first_send_s`;
- `counts`: stalls, radio losses inside recordings, frames.

### 5. Step 0 → `series.parquet`

- **In:** `frames.parquet` and `session.json` only.
- **Out:** one row per (pod, epoch, tick).

| column | meaning |
|---|---|
| `pod_id`, `epoch`, `tick`, `count` | `epoch` counts reboots; `tick` = sends since that boot |
| `status` | `fresh` (new sample), `held` (repeated `ms_lo`) or `lost` (a known tick that never arrived) |
| `received`, `valid` | arrived; passed the norm check (\|‖q‖ − 1\| ≤ 0.01) |
| `t_host_s` / `t_host_est_s` | arrival time / interpolated arrival time of a lost tick |
| `ms_lo`, `ms_unwrapped`, `t_pod_s` | wire value; unwrapped pod milliseconds; the same in seconds |
| `batt`, `norm_err` | battery; ‖q‖ − 1 |
| `qx..qw` | decoded, normalised, canonical (w ≥ 0); NaN when lost or invalid |

Code: `mesqsim/step0.py`. Rules: [MATH.md §5](MATH.md).

### 6. Baselines and errors → `results/`

- **In:** series, session, and (scorer and oracles only) labels and `meta.json` (host epoch, true mountings).
- **Out:**
  - `errors.parquet`: one row per session, baseline (B0, B1, O1, O2), step and joint, with `err_heading_deg`
    (primary) and `err_orient_deg` (secondary, true mountings).
  - `summary.csv` and `summary.md`: mean, median and P95 per baseline, overall, per joint type and per joint
    (motion phase).

Code: `mesqsim/evaluate.py`, `mesqsim/pipeline.py`.

### 7. Step 2 estimators (GEO-K, GEO-ID) → `kappa`, `uncertain` in `results/errors.parquet`

- **In:** Step 0's `series.parquet` and `session.json` only, plus the `geometry:` section of the config (the
  estimator's own beliefs). GEO-K also receives the true mountings from `meta.json` through its one `truth`
  argument (simulation only). The scorer passes the host times of its 32 Hz steps (grid time + host epoch) as the
  query times.
- **Out:** for every query time and joint, δ̂ in radians in (−π, π], a concentration κ ≥ 0 (0 = no information)
  and an `uncertain` flag (κ below `geometry.kappa_hold`; the estimate still holds). `errors.parquet` gains the
  columns:

| column | meaning |
|---|---|
| `kappa` | float32; the method's von Mises concentration at that step; NaN for methods without a confidence (B0, B1, O1, O2) |
| `uncertain` | nullable boolean; κ < `geometry.kappa_hold`; null for methods without a confidence |

- **What runs** ([mesqsim/estimators/geo.py](../mesqsim/estimators/geo.py)):
  1. fresh, valid samples per pod; for each pod and boot epoch a causal clock map t_host ≈ α + β·t_pod (lower
     envelope; [MATH.md §9](MATH.md));
  2. events in host time: the box-registration start at the end of the box phase; a reset of a joint whenever
     one of its pods starts a new boot epoch; a window every `geometry.step_s` (from the motion phase on, never
     inside an unrecorded phase) solving each joint over the last `geometry.window_s` of parent/child pairs with the
     solver of its type: the hinge closed form (knees), the 2-DoF constraint (GEO-K) or the range-of-motion cost
     (GEO-K at 3-DoF joints; GEO-ID at 2-DoF and 3-DoF joints, in T-pose frames once the facing is known);
  3. the per-joint von Mises filter, read out at the query times.
- **Methods** (`evaluate.METHODS`): B0, B1, O1, O2; GEO-K (uses simulator truth), GEO-ID, and the ablations
  GEO-ID-nofilter, GEO-ID-hinge, GEO-ID-w4, GEO-ID-w16.
- **Checks:** 21.1 causality, 21.2 determinism, 21.3 label blindness, 31.1 pair timing
  ([VALIDATION.md](VALIDATION.md)).

### 8. Fitting the estimator's beliefs (`mesqsim/geo_fit.py`, `mesqsim/geo_sessions.py`)

- **In:** the split manifest; AMASS train sequences (Ω) or the validation sessions of the dataset (calibration),
  simulated in memory in worker processes with the dataset's own seeds (nothing is written per session).
- **Out:** `fitted` values in `configs/default.yaml` (in place, comments kept; the source note names the command, the
  data, the split hash, the counts and the commit) and a JSON report under `$MESQ_OUTPUT_DIR`. Run it from a clean
  checkout: the commit in the note must contain the code that made the fit. Rules: [MATH.md §14, §16](MATH.md).

### 9. DIODEM (`mesqsim/diodem_geo.py`)

- **In:** the verified DIODEM download (`$MESQ_DATA_DIR/diodem`), loaded with the authors' loader at 40 Hz.
- **What runs:** six-axis VQF per rigid IMU; GEO-ID unchanged on a Step 0-shaped series of the two IMUs of each hinge;
  truth from the optical frames ([MATH.md §17](MATH.md)).
- **Out:** per experiment and hinge, the error summaries (in RESULTS_STEP2), and the figure
  `docs/figures/diodem_geo_hinge.png` (check 32.1).

### 10. The Step 2 results (`scripts/make_results_step2.py`, `mesqsim/results_step2.py`)

- **In:** the split manifest (AMASS, else synthetic stand-ins, labelled PROVISIONAL), `configs/default.yaml` (with its
  fitted beliefs), the DIODEM download.
- **What runs:** the dataset's **test** sessions (windows × motion starts 1, 10, 30 min; the dataset's own seeds),
  simulated, decoded and scored in memory in worker processes: B0, B1, O1, O2, GEO-K, GEO-ID, GEO-ID-nofilter,
  GEO-ID-hinge, GEO-ID-w4, GEO-ID-w16. Then one-at-a-time sweeps at 10 min (B1, O1, O2, GEO-K, GEO-ID), the
  re-acquisition after reboots, and GEO-ID on every DIODEM hinge with the fitted and the assumed beliefs.
- **Out:** `docs/RESULTS_STEP2.md` (tables 1–10, H2's contrasts) and `docs/figures/results_step2_{growth,reliability,
  selective,sweeps,reboot}.png`. Nothing is written per session. The same config and seed give an identical file
  apart from its **Run** line, which holds the date, CPU, wall time and compute time (check 33.1).

