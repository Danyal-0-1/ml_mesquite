# Code index

*Status: M3. Updated at every milestone.*

## How the modules connect

```text
conventions ◄── everything (the only place rotations are defined)
body        ◄── correction, sim, evaluate
codec       ◄── sim (quantises reports), step0 (decodes int16)
config      ◄── sim, cli, validation (one labelled YAML file + presets)
recording   ◄── sim (which ticks the labels describe), step0 (which missing ticks are lost)
sim/*       ◄── pipeline: motion → session plan → truth, timing, radio → simulate (frames + labels)
step0       ◄── pipeline, evaluate (decoded series); reads frames + session.json only
evaluate    ◄── pipeline (baselines, primary and secondary metrics)
io, runs    ◄── pipeline, cli (session folders, run identity)
manifest    ◄── scripts/check_env.py, data/diodem (download), scripts/verify_amass.py
data/net    ◄── data/diodem (HTTP session: OS trust store, User-Agent, MD5-verified download)
data/amass  ◄── synth (writes AMASS-format files), figures, (later) simulator
data/diodem ◄── scripts/download_diodem.py, scripts/make_figures.py, validation 11.x
validation  ◄── scripts/validate.py, tests/test_validation.py
```

## Files

| file | purpose |
|---|---|
| [mesqsim/conventions.py](../mesqsim/conventions.py) | quaternion order, Hamilton product, Y(α), heading, wrap, Exp/Log, slerp, circular statistics |
| [mesqsim/codec.py](../mesqsim/codec.py) | the 16-byte wire frame: firmware quantisation, encode, decode, stream resync |
| [mesqsim/body.py](../mesqsim/body.py) | the 15 rendered pods and the 14-joint tree (parent before child) |
| [mesqsim/correction.py](../mesqsim/correction.py) | targets δ = wrap(ψ_c − ψ_p); outward correction ĉ; q̃ = Y(ĉ)·q |
| [mesqsim/manifest.py](../mesqsim/manifest.py) | data manifests (path, size, SHA-256): read, write, verify, record |
| [mesqsim/paths.py](../mesqsim/paths.py) | data and output folders (`MESQ_DATA_DIR`, `MESQ_OUTPUT_DIR`) |
| [mesqsim/data/net.py](../mesqsim/data/net.py) | HTTP session (OS trust store, User-Agent) and an atomic, size- and MD5-checked download |
| [mesqsim/data/diodem.py](../mesqsim/data/diodem.py) | DIODEM: Dataverse file list, download into the official loader's layout, manifest, chain adjacency, rate/norm scan, `load_trial` (official loader), six-axis relative-heading analysis |
| [mesqsim/data/amass.py](../mesqsim/data/amass.py) | AMASS: `.npz` loader (SMPL+H and SMPL-X keys), forward kinematics, pod → SMPL joint mapping, up-axis estimate, hip and knee flexion, file discovery |
| [mesqsim/synth.py](../mesqsim/synth.py) | synthetic AMASS-format motion (T-pose, stand, walk, circle) so tests run without licensed data |
| [mesqsim/figures.py](../mesqsim/figures.py), [mesqsim/plotstyle.py](../mesqsim/plotstyle.py) | stick figure (generic bone lengths) and the DIODEM figure; one plot style |
| [mesqsim/config.py](../mesqsim/config.py) | labelled config: load, validate labels and sources, overrides, presets, hash of values |
| [mesqsim/recording.py](../mesqsim/recording.py) | recorded intervals and the shared "knowable tick" rule for lost frames |
| [mesqsim/sim/rng.py](../mesqsim/sim/rng.py) | one random stream per subsystem, all from the run seed |
| [mesqsim/sim/motion.py](../mesqsim/sim/motion.py) | motion source (AMASS or synthetic), nominal mountings, strapping path, facing |
| [mesqsim/sim/session.py](../mesqsim/sim/session.py) | session plan: phases, host epoch, facings, every pod's drawn parameters |
| [mesqsim/sim/truth.py](../mesqsim/sim/truth.py) | true pod orientation P_i(t) in every phase |
| [mesqsim/sim/heading.py](../mesqsim/sim/heading.py) | ψ_i(t), the heading of each pod's private frame (offsets in M3) |
| [mesqsim/sim/timing.py](../mesqsim/sim/timing.py) | chip samples, decode times, send grid, newest sample, ms_lo |
| [mesqsim/sim/radio.py](../mesqsim/sim/radio.py) | arrival times (loss from M4) |
| [mesqsim/sim/simulate.py](../mesqsim/sim/simulate.py) | one session end to end: frames, labels, session.json, meta |
| [mesqsim/step0.py](../mesqsim/step0.py) | Step 0: unwrap count and ms_lo, reboots, fresh/held/lost, integrity |
| [mesqsim/evaluate.py](../mesqsim/evaluate.py) | baselines and the primary and secondary metrics |
| [mesqsim/io.py](../mesqsim/io.py) | deterministic Parquet and JSON; the frames-table schema; session folders |
| [mesqsim/runs.py](../mesqsim/runs.py) | run identity (commit, config hash, seed), run folders, listing and deleting |
| [mesqsim/pipeline.py](../mesqsim/pipeline.py) | simulate a run, score a run, summaries |
| [mesqsim/cli.py](../mesqsim/cli.py), [mesqsim/__main__.py](../mesqsim/__main__.py) | `python -m mesqsim simulate / baselines / list-runs / clean` |
| [configs/default.yaml](../configs/default.yaml), [configs/presets.yaml](../configs/presets.yaml) | every parameter with label and source; presets for the checks |
| [mesqsim/validation.py](../mesqsim/validation.py) | registry of the brief §9 checks: what, why, expected, command, and the computation |
| [scripts/check_env.py](../scripts/check_env.py) | prints versions, OS, CPU; checks imports and data checksums |
| [scripts/download_diodem.py](../scripts/download_diodem.py) | downloads and MD5-verifies DIODEM, writes `manifests/diodem.csv` |
| [scripts/verify_amass.py](../scripts/verify_amass.py) | records, extracts and indexes hand-downloaded AMASS archives and the SMPL+H model |
| [scripts/make_figures.py](../scripts/make_figures.py) | regenerates `docs/figures/*.png` |
| [scripts/diodem_inspect.py](../scripts/diodem_inspect.py) | reproduces the DIODEM frame, adjacency and gyroscope-offset facts in DATA.md |
| [manifests/](../manifests/) | committed manifests: path, size, SHA-256 of every data file (data itself never committed) |
| [scripts/validate.py](../scripts/validate.py) | runs every check and rewrites [VALIDATION.md](VALIDATION.md) |
| [scripts/check_all.sh](../scripts/check_all.sh) | the full check: ruff, pytest, validation |
| [tests/](../tests/) | `test_codec.py` (check 1), `test_conventions.py` (2), `test_correction.py` (3, 4), `test_amass.py` (10), `test_diodem.py` (11), `test_sim_slice.py` (5, 7, 8: the M3 slice), `test_step0.py` (Step 0 on hand-built frames), `test_repo_hygiene.py` (rule 3), `test_validation.py` (all registered checks); `conftest.py` skips data tests when data are absent |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | CI: ruff, the synthetic tests and quick validation on every push and pull request |

## Where to extend

- A new rotation helper goes in `conventions.py` with a test in `test_conventions.py`, never inline elsewhere.
- A new validation number: add a `@register(...)` function in `validation.py`; it appears in VALIDATION.md and is
  asserted by `tests/test_validation.py` automatically.
- A new dataset: a module in `mesqsim/data/` that downloads (or verifies a hand download) into `$MESQ_DATA_DIR/<name>/`,
  writes `manifests/<name>.csv`, converts to the +Z-up world at load time with a test of that conversion, and a
  `conftest.py` fixture that skips its tests when the data are absent.
