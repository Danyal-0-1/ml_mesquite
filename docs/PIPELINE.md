# Pipeline

*Status: M3. The stages exist end to end; M4 adds the full error model (drift, scale factor, random walk, tilt
error, radio loss, stalls, reboots), box registration, B1 and O1.*

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
| decode and score | `python -m mesqsim baselines --run <run folder>` | `series.parquet` per session; `results/errors.parquet`, `summary.csv`, `summary.md` |
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
- Report q_i = Y(−ψ_i)·P_i, canonical (w ≥ 0), quantised as the firmware does. Code: `sim/simulate.py`,
  `codec.quantize_component`.
- Arrival = send + latency, first-in first-out per pod; host time = arrival + host epoch. Code: `sim/radio.py`.
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
| `pod_id`, `epoch`, `tick`, `count` | identity of the send: boot epoch, sends since boot, wire counter |
| `phase` | phase at the send time |
| `t_send_s`, `t_arrive_s`, `t_sample_s`, `t_decode_s` | true times, seconds since the first power-on |
| `sample_index` | index of the chip sample sent |
| `ms_lo` | the wire value |
| `received` | in the frames table |
| `fresh` | the sample differs from the previous tick's (sender's view) |
| `new_to_host` | received and different from the last received sample (what the host can tell) |
| `psi_rad` | true heading ψ_i of the pod's private frame at the sample time |
| `pod_qx..pod_qw` | true pod orientation P_i (pod → world) at the sample time |

**`labels_joints.parquet`** (simulation only): one row per joint and 32 Hz step.

| column | meaning |
|---|---|
| `step`, `t_s`, `phase` | step index, true time, phase |
| `joint` | 0–13 in `mesqsim.body.JOINTS` order |
| `delta_rad` | true relative heading δ_pc = wrap(ψ_c − ψ_p) at `t_s` |
| `delta_offset_rad` | the offset part wrap(a_c − a_p) (the O1 oracle) |

**`meta.json`** (simulation only): `run` (commit, config hash, seed), `config` (resolved values),
`config_parameters` (value, label, source of each), `packages`, `source` (motion, fps, duration), `split`, `plan`
(timeline and every pod's drawn parameters), `first_send_s`.

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
  - `errors.parquet`: one row per session, baseline, step and joint, with `err_heading_deg` (primary) and
    `err_orient_deg` (secondary, true mountings).
  - `summary.csv` and `summary.md`: mean, median and P95 per baseline, overall, per joint type and per joint
    (motion phase).

Code: `mesqsim/evaluate.py`, `mesqsim/pipeline.py`.
