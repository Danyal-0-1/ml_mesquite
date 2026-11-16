# Code index

*Status: S2-M3 (Step 2: contract, clock, GEO framework, hinge, 2-DoF and range-of-motion solvers). Updated at every milestone.*

## How the modules connect

```text
conventions ◄── everything (the only place rotations are defined)
body        ◄── correction, sim, evaluate
codec       ◄── sim (quantises reports), step0 (decodes int16)
config      ◄── sim, cli, validation (one labelled YAML file + presets)
recording   ◄── sim (which ticks the labels describe), step0 (which missing ticks are lost)
sim/*       ◄── pipeline: motion → session plan → truth, timing, radio → simulate (frames + labels)
step0       ◄── pipeline, evaluate (decoded series); reads frames + session.json only
evaluate    ◄── pipeline (baselines B0-O2 and the Step 2 methods; primary and secondary metrics, kappa, uncertain)
estimators/ ◄── evaluate (GEO-K, GEO-ID and ablations; reads Step 0's series + session.json only)
   contract   Estimate(delta_rad, kappa, uncertain); GeoTruth (GEO-K's one truth argument)
   clock      fresh samples, causal pod->host clock map (lower envelope), pairs, rotation rates
   vmfilter   per-joint von Mises filter: init (box), predict, update (window), reset (reboot)
   geo        events (box init, reboot resets, windows every step_s), solvers per joint type, query outputs,
              GEO-ID's identified axes, sign votes, facing from the knees and T-pose frames
   anatomy    knee hinge axis, 2-DoF axis pairs, one Cardan sequence per joint (SMPL segment frames)
   hinge      closed form and its von Mises kappa; axis identification from rates; relative-sign votes
   twodof     the 2-DoF constraint as a trigonometric polynomial; Laplace kappa; two-minima rule
   rom        range-of-motion cost in fast form; moment-matched kappa; Omega percentiles
geo_score   ◄── validation, scripts (scorer-side Step 2 diagnostics; may read labels): pair timing, truncation
geo_fit     ◄── cli fit-geometry (fits on train/validation only; reads truth; writes `fitted` values into the config)
io, runs    ◄── pipeline, cli (session folders, run identity)
splits      ◄── dataset, cli split, validation 9.x
dataset     ◄── cli dataset, scripts/make_results.py (many sessions, each in its sequence's split)
results     ◄── scripts/make_results.py (tables, figures)
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
| [mesqsim/sim/heading.py](../mesqsim/sim/heading.py) | ψ_i(t) per boot epoch: offset, drift, scale factor × Θ (integrated from the trajectory), random walk; the idle bridge |
| [mesqsim/sim/timing.py](../mesqsim/sim/timing.py) | chip samples, decode times, sample-task stalls, send grid, newest sample, ms_lo, per boot epoch |
| [mesqsim/sim/radio.py](../mesqsim/sim/radio.py) | Gilbert–Elliott burst loss and arrival times |
| [mesqsim/sim/simulate.py](../mesqsim/sim/simulate.py) | one session end to end: boot epochs, fusion error η, frames, labels, session.json, meta; `plan_hook` for the invariance checks |
| [mesqsim/step0.py](../mesqsim/step0.py) | Step 0: unwrap count and ms_lo, reboots, fresh/held/lost, integrity, box registration, T-pose snapshot |
| [mesqsim/evaluate.py](../mesqsim/evaluate.py) | the method registry (B0, B1, O1, O2; GEO-K, GEO-ID and its ablations), the primary and secondary metrics, `kappa` and `uncertain` columns |
| [mesqsim/estimators/contract.py](../mesqsim/estimators/contract.py) | the Step 2 estimator contract: `Estimate(delta_rad, kappa, uncertain)`, `GeoTruth` (GEO-K only) |
| [mesqsim/estimators/clock.py](../mesqsim/estimators/clock.py) | fresh samples per pod, the causal pod-to-host clock map (lower envelope), parent/child pairs (slerp), body rotation rates |
| [mesqsim/estimators/vmfilter.py](../mesqsim/estimators/vmfilter.py) | the per-joint von Mises filter (init, predict, update, reset; outputs at query times) |
| [mesqsim/estimators/geo.py](../mesqsim/estimators/geo.py) | GEO-K and GEO-ID: box initialisation, reboot resets, windows every `step_s`, solver dispatch per joint type, hold flag, ablation switches |
| [mesqsim/estimators/anatomy.py](../mesqsim/estimators/anatomy.py) | anatomical joint models in SMPL segment frames: knee hinge axis, 2-DoF axis pairs, one Cardan sequence per joint |
| [mesqsim/estimators/hinge.py](../mesqsim/estimators/hinge.py) | hinge closed form and von Mises kappa, blind spot; axis identification from body rates (Seel et al. 2014); vertical and T-pose sign votes |
| [mesqsim/estimators/twodof.py](../mesqsim/estimators/twodof.py) | the 2-DoF constraint (Laidig et al. 2019) as a degree-2 trigonometric polynomial; Laplace kappa; two-minima ambiguity rule |
| [mesqsim/estimators/rom.py](../mesqsim/estimators/rom.py) | range-of-motion cost (Lehmann et al. 2020) in fast form over a 1° grid; moment-matched kappa; true joint angles; Omega percentiles |
| [mesqsim/geo_score.py](../mesqsim/geo_score.py) | scorer-side Step 2 diagnostics (reads labels): pair timing errors, recordings cut at a host time |
| [mesqsim/geo_fit.py](../mesqsim/geo_fit.py) | fitting on train/validation only (reads truth): Omega per joint from the training subjects' true angles; knee and third-axis RMS |
| [mesqsim/io.py](../mesqsim/io.py) | deterministic Parquet and JSON; the frames-table schema; session folders |
| [mesqsim/runs.py](../mesqsim/runs.py) | run identity (commit, config hash, seed), run folders, listing and deleting |
| [mesqsim/pipeline.py](../mesqsim/pipeline.py) | simulate a run, score a run (motion-phase errors, split and motion start attached), summaries |
| [mesqsim/splits.py](../mesqsim/splits.py) | subject splits 70/15/15 by stable hash, TotalCapture holdout, manifest write and check |
| [mesqsim/dataset.py](../mesqsim/dataset.py) | dataset generation: windows × motion starts × variants, session seeds, synthetic stand-in sources |
| [mesqsim/results.py](../mesqsim/results.py) | results tables (overall, joint type, joint, sweeps) and the two results figures |
| [mesqsim/cli.py](../mesqsim/cli.py), [mesqsim/__main__.py](../mesqsim/__main__.py) | `python -m mesqsim simulate / baselines / params / split / dataset / list-runs / clean` |
| [configs/default.yaml](../configs/default.yaml), [configs/presets.yaml](../configs/presets.yaml) | every parameter with label and source; presets for the checks |
| [mesqsim/validation.py](../mesqsim/validation.py) | registry of the brief §9 checks: what, why, expected, command, and the computation |
| [scripts/check_env.py](../scripts/check_env.py) | prints versions, OS, CPU; checks imports and data checksums |
| [scripts/download_diodem.py](../scripts/download_diodem.py) | downloads and MD5-verifies DIODEM, writes `manifests/diodem.csv` |
| [scripts/verify_amass.py](../scripts/verify_amass.py) | records, extracts and indexes hand-downloaded AMASS archives and the SMPL+H model |
| [scripts/make_figures.py](../scripts/make_figures.py) | regenerates `docs/figures/*.png` |
| [scripts/make_results.py](../scripts/make_results.py) | one command: default dataset + sweeps → docs/RESULTS_STEP1.md and figures |
| [scripts/diodem_inspect.py](../scripts/diodem_inspect.py) | reproduces the DIODEM frame, adjacency and gyroscope-offset facts in DATA.md |
| [manifests/](../manifests/) | committed manifests: path, size, SHA-256 of every data file (data itself never committed) |
| [scripts/validate.py](../scripts/validate.py) | runs every check and rewrites [VALIDATION.md](VALIDATION.md) |
| [scripts/check_all.sh](../scripts/check_all.sh) | the full check: ruff, pytest, validation |
| [tests/](../tests/) | `test_codec.py` (check 1), `test_conventions.py` (2), `test_correction.py` (3, 4), `test_amass.py` (10), `test_diodem.py` (11), `test_sim_slice.py` (5, 7, 8: the M3 slice), `test_sim_full.py` (error-model units: Θ, η, loss, reboots, box registration, T-pose snapshot), `test_step0.py` (Step 0 on hand-built frames), `test_splits.py` (9: splits, manifest, dataset), `test_repo_hygiene.py` (rule 3), `test_validation.py` (every registered check, ids `check_<n>_<m>`); Step 2: `test_conventions_step2.py` (Cardan angles, von Mises helpers), `test_clock.py` (clock map, pairs, rates), `test_geo_contract.py` (causality, determinism, label blindness, own beliefs, pair timing), `test_hinge.py` (closed form, worked example, blind spot, likelihood, identification, sign votes), `test_twodof_rom.py` (2-DoF exact recovery and ambiguity, range-of-motion fast form, limits, flat inside), `test_config_write.py` (fitted values written in place); `conftest.py` skips data tests when data are absent |
| [.github/workflows/ci.yml](../.github/workflows/ci.yml) | CI: ruff, the synthetic tests and quick validation on every push and pull request |

## Where to extend

- A new rotation helper goes in `conventions.py` with a test in `test_conventions.py` (Step 2 additions:
  `test_conventions_step2.py`), never inline elsewhere. Circular statistics (von Mises) live there too.
- A new Step 2 estimator implements the contract of `estimators/contract.py`, receives `cfg.section("geometry")`
  (never the simulator's `error_model`), is registered in `evaluate.METHODS` (and `USES_TRUTH` if it reads truth),
  and gets the causality, determinism and label-blindness tests of `test_geo_contract.py`.
- A new validation number: add a `@register(...)` function in `validation.py`; it appears in VALIDATION.md and is
  asserted by `tests/test_validation.py` automatically.
- A new dataset: a module in `mesqsim/data/` that downloads (or verifies a hand download) into `$MESQ_DATA_DIR/<name>/`,
  writes `manifests/<name>.csv`, converts to the +Z-up world at load time with a test of that conversion, and a
  `conftest.py` fixture that skips its tests when the data are absent.
