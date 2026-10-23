# The simulator

*Status: M5: the full error model, datasets, sweeps and the results command.*

## Picture first

You give the simulator a motion clip, a config and a seed. It builds a session as the real suit would live it, in
this order: pods switched on in a box, strapped on, a T-pose, an idle wait, the motion. For each pod it writes the
16-byte frames the hub would have logged, and separately the truth: every pod's true heading error at every send
and every joint's true relative heading on a 32 Hz grid. The same seed and config always give byte-identical files.

## Run it

```bash
# one session from a synthetic walk (20 s), default parameters, seed 1
python -m mesqsim simulate --seed 1
# an AMASS clip, a named preset, and one overridden parameter
python -m mesqsim simulate --seed 3 --motion "$MESQ_DATA_DIR/amass/CMU/01/01_01_poses.npz" \
    --preset offsets_only --set session.idle_s=600
# decode with Step 0 and score the baselines
python -m mesqsim baselines --run "$MESQ_OUTPUT_DIR/run_g<commit>_c<hash>_s1"
```

`--motion` takes `synthetic:<kind>[:<seconds>[:<seed>]]` (`tpose`, `stand`, `walk`, `circle`) or a path to an
AMASS file.

## Datasets, sweeps and results

```bash
python -m mesqsim split --write                  # AMASS: subjects -> train/val/test (docs/DATA.md §3)
python -m mesqsim dataset --seed 1 --source amass     # or --source synthetic; simulate + score
python scripts/make_results.py                    # the default dataset + every sweep -> docs/RESULTS_STEP1.md
```

- **A dataset** cuts every source sequence into windows of `dataset.window_s` (at most
  `dataset.windows_per_sequence` per sequence).
  - For each window it simulates one session per `dataset.motion_start_min` value (1, 10, 30). The idle interval is
    60·m minus box, strapping and T-pose, so the motion starts m minutes after boot. It also simulates one session
    per `dataset.variants`.
  - Each session's seed comes from (run seed, sequence, window, start, variant).
  - The run folder holds `sessions/<id>/` and `dataset.csv` (id, source, subset, subject, split, window, motion start,
    variant, seed). `results/errors.parquet` keeps the motion-phase steps with each session's split and motion start.
- **Sweeps** (`scripts/make_results.py`) vary one parameter at a time around the default, over the values under
  `sweep` in the config (offset mode, box slot, drift, scale factor, reboots, stalls, loss, burst). Motion starts at
  10 min, one window per sequence. Every value reuses the same session seeds (common random numbers), so a
  difference between values comes from the swept parameter alone.
- **Results**: `scripts/make_results.py` writes `docs/RESULTS_STEP1.md` (tables 1–4 and the parameter set),
  `docs/figures/results_growth.png` and `results_sweeps.png`, plus CSV tables in the default run's `results/`. It
  reuses scored runs of the same (commit, config, seed); `--fresh` regenerates. Without AMASS it uses synthetic
  stand-ins and labels every table "PROVISIONAL: SYNTHETIC MOTION".

## Configure it

- One file holds every parameter: [configs/default.yaml](../configs/default.yaml). Each one has a `value`, a `label`
  (`datasheet`, `firmware`, `assumption` or `measured`) and a `source` note. A `sweep` list gives the values a
  sweep visits.
- `mesqsim.config.Config.validate` refuses a parameter without a valid label or a source.
- `python -m mesqsim params` prints every parameter with its value, label and source. `--label assumption` lists
  only the assumptions: 34 of 40 today, all to be replaced by bench measurements.
- **Presets** in [configs/presets.yaml](../configs/presets.yaml) are named override sets, applied in order with
  `--preset` (repeatable):
  - `zero`: every error off, no loss, exact timing (check 5.1);
  - `offsets_only`: arbitrary offsets and nothing else (checks 5.2, 5.3);
  - `stress`: the brief's stress settings (3 % scale factor, 1 reboot per pod per hour, 5 % loss in bursts of
    10, 100 ms stalls).
- **Overrides:** `--set key=value` with a dotted key. The value is parsed as YAML: `--set timing.decode_delay_ms=[0,0]`.
- Units are in the parameter names. A range written `[a, b]` is a uniform draw per pod.

## Runs on disk

- A run is identified by `(git commit, config hash, seed)`. Its folder is
  `$MESQ_OUTPUT_DIR/run_g<commit 7>_c<config hash 8>_s<seed>`.
  - `-dirty` is added when the working tree had uncommitted or untracked changes.
  - The config hash covers resolved values only, so editing a source note keeps the identity.
- Contents: `sessions/<id>/` (frames, session.json, labels, meta.json, and series.parquet after scoring) and
  `results/` (errors.parquet, summary.csv, summary.md). File formats: [PIPELINE.md](PIPELINE.md).
- `python -m mesqsim list-runs` lists runs. `python -m mesqsim clean --run <folder>` deletes one;
  `clean --all --yes` deletes all. `clean` refuses anything that is not a `run_*` folder directly inside the
  output directory.
- Outputs never go into git (`/outputs/` is ignored, and `$MESQ_OUTPUT_DIR` usually lives outside the repository).

## Randomness

- Every draw comes from the run's one seed.
- Each subsystem (timeline, pods, heading, timing, radio, …) draws from its own stream, seeded by
  `(seed, subsystem name)`. Adding a new error term with its own stream leaves every other draw unchanged, so old
  runs stay comparable.

## Validate after a change

1. `bash scripts/check_all.sh`: ruff, every test, and the validation script, which rewrites
   [VALIDATION.md](VALIDATION.md).
2. If a check fails, do not loosen its tolerance. Find out which convention or bookkeeping broke: most bugs in this
   project are sign, order or frame errors.

## The error terms

- **Modelled:** offsets, box-slot error, drift, scale factor × turning, angle random walk, fusion tilt-and-noise
  error, reboots, sample-task stalls, burst radio loss, decode delay, clock rate error, arrival latency and
  quantisation. Each one, with its model, value and label, is in [MATH.md §5](MATH.md).
- **Random streams:** each term draws from its own stream: `pods`, `heading`, `drift`, `reboots`, `epochs`, `walk`,
  `fusion`, `timing`, `radio`, `loss`, `timeline`.
- **Not modelled** (each needs a measurement first): the chip rate's ±1 % drift over temperature, per-axis scale
  factors, cross-axis sensitivity, and radio loss correlated across pods.

## Add a new error term

1. Add its parameters to `configs/default.yaml`, each with a label and a source.
2. Give it its own random stream (`mesqsim.sim.rng.stream(seed, "<name>")`).
3. Add it to `presets.yaml: zero` with its "off" value, so check 5.1 keeps proving that everything off means zero
   error.
4. Write its analytic expectation as a validation check (like the random-walk and drift checks of brief §9.6).
5. Document it in [MATH.md](MATH.md) with its value and label.

## Add a new motion dataset

Write a loader in `mesqsim/data/` that returns segment orientations (N, 15, 4) in the +Z-up world, with a test of
the world conversion. Then accept its files in `MotionSource.from_spec`. See [CODE.md](CODE.md), "Where to extend".
