# The simulator

*Status: M3 (the thin slice: offsets only). M4 adds the full error model, sweeps and dataset generation.*

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

`--motion` takes `synthetic:<kind>[:<seconds>]` (`tpose`, `stand`, `walk`, `circle`) or a path to an AMASS file.

## Configure it

- One file holds every parameter: [configs/default.yaml](../configs/default.yaml). Each one has a `value`, a `label`
  (`datasheet`, `firmware`, `assumption` or `measured`) and a `source` note. A `sweep` list gives the values a
  sweep visits.
- `mesqsim.config.Config.validate` refuses a parameter without a valid label or a source.
- **Presets** in [configs/presets.yaml](../configs/presets.yaml) are named override sets used by the checks:
  - `zero`: every error off, exact timing;
  - `offsets_only`: arbitrary offsets and nothing else.
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

## Add a new error term (from M4 on)

1. Add its parameters to `configs/default.yaml`, each with a label and a source.
2. Give it its own random stream (`mesqsim.sim.rng.stream(seed, "<name>")`).
3. Add it to `presets.yaml: zero` with its "off" value, so check 5.1 keeps proving that everything off means zero
   error.
4. Write its analytic expectation as a validation check (like the random-walk and drift checks of brief §9.6).
5. Document it in [MATH.md](MATH.md) with its value and label.

## Add a new motion dataset

Write a loader in `mesqsim/data/` that returns segment orientations (N, 15, 4) in the +Z-up world, with a test of
the world conversion. Then accept its files in `MotionSource.from_spec`. See [CODE.md](CODE.md), "Where to extend".
