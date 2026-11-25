# Repository guide

This page shows where everything in the repository is and which step of the project built it. The project is built
in steps, and each step is one release on the repository's only branch, `main`.

| release | step | what it delivers | tags | specification | results |
|---|---|---|---|---|---|
| **v1.0.0** (2026-11-16) | **Step 1**, the ground | the suit simulator on AMASS motion; Step 0, the decoding shared with real recordings; the baselines B0, B1, O1, O2; DIODEM as a reality check | `step1` | [STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md) | [RESULTS_STEP1.md](docs/RESULTS_STEP1.md) |
| **v2.0.0** (2026-11-18) | **Step 2**, the geometry | GEO, the geometric estimator of each joint's relative heading δ (GEO-K uses simulator truth, GEO-ID is deployable); the explanation pipeline | `step2`, `step2-m0` … `step2-m8` | [STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md) | [RESULTS_STEP2.md](docs/RESULTS_STEP2.md) |
| **v3.0.0** (2026-11-21) | **Step 3**, the twin and the model kit | the Mesquite twin, calibrated on Danyal's recordings; the training data `twin_v1`; the model kit; the numbers to beat | `step3`, `step3-m0` … `step3-m10` | [STEP3_BRIEF.md](docs/brief/STEP3_BRIEF.md) | [RESULTS_STEP3.md](docs/RESULTS_STEP3.md) |
| next | Step 4, fusion | not started. Danyal's learned models come first; the plan is item 10 of the Step 3 report (the last entry of [WORKLOG.md](WORKLOG.md)) | | | |

[CHANGELOG.md](CHANGELOG.md) lists each release with its milestones. [WORKLOG.md](WORKLOG.md) has one entry per
milestone: what changed, the commands that were run, and the decisions. Everything, Step 3 included, has been on
GitHub since 2026-11-21 ([CLAUDE.md](CLAUDE.md) rule 2).

## 1. The layout

The code is organised by component, as in most Python projects. Each line ends with the step or steps that built it:
`[1]` for Step 1, `[1, 2, 3]` for all three.

```text
ml_mesquite/
├── README.md               overview, the project in one picture, quickstart
├── REPOSITORY_GUIDE.md     this map
├── CHANGELOG.md            the releases, one per step
├── CLAUDE.md               the rules for every work session
├── WORKLOG.md              one entry per milestone, Step 1 to Step 3; the Step 3 report is the last entry
├── pyproject.toml          package metadata (mesqsim 3.0.0); ruff and pytest settings
├── requirements.txt        pinned packages (requirements-lock.txt: the full freeze of the reference machine)
├── configs/                every parameter: value, source, label             [1: default, presets; 3: twin]
├── mesqsim/                the Python package
│   ├── *.py                the core: conventions, codec, body tree, Step 0, runs, results, checks   [1, 2, 3]
│   ├── sim/                the suit simulator                                           [1]
│   ├── data/               AMASS and DIODEM: download, verification, loading            [1]
│   ├── estimators/         GEO, the geometric estimator                                 [2]
│   ├── real/               Danyal's recordings: readers, alignment, the statistics Φ    [3]
│   ├── twin/               the Mesquite twin                                            [3]
│   └── learn/              the model kit and the training data twin_v1                 [3]
├── scripts/                one-command entry points: checks, downloads, results        [1, 2, 3]
├── tools/                  golden-fixture makers: the page (Node) and the phone app (Swift)   [3]
├── tests/                  pytest suite, no licensed data needed; fixtures/ = golden outputs   [1, 2, 3]
├── manifests/              path, size and SHA-256 of every data file (the data stay outside git)   [1, 3]
├── docs/                   documents, results and validation                            [1, 2, 3]
│   ├── brief/              the three briefs, the specification (never edited)          [1, 2, 3]
│   ├── explain/            the explanation pipeline                                     [2]
│   └── figures/            the figures of the results documents                         [1, 2, 3]
├── project_explanations/   the two datasets explained (Danyal's request)                [2]
└── .github/workflows/      CI on every push: ruff, the tests, the quick validation      [1]
```

Data and outputs never enter git; §6 lists them.

## 2. Step 1: the simulator, Step 0 and the baselines (v1.0.0, tag `step1`)

Specification: [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md). Its suit facts, conventions, measurement model,
data rules and environment still hold wherever the later briefs are silent.

| path | what it is |
|---|---|
| `mesqsim/conventions.py` | the one place for rotation conventions: quaternion order (x, y, z, w), the Hamilton product, `Y(α)`, heading, `wrap`, circular statistics (CLAUDE.md rule 7) |
| `mesqsim/codec.py` | the 16-byte wire frame, encoded and decoded exactly as the pod firmware does |
| `mesqsim/body.py` | the pods, the segments and the 14-joint tree |
| `mesqsim/recording.py` | what one recording is, and which missing frames count as lost |
| `mesqsim/sim/` | the simulator: `motion.py` and `truth.py` (true pod orientations), `heading.py` (each pod's private north ψ), `session.py` (timeline and drawn parameters), `timing.py` (chip samples, the 32 Hz send grid), `radio.py` (loss, arrival times), `rng.py` (seeded streams), `simulate.py` (one session end to end) |
| `mesqsim/step0.py` | Step 0: the deterministic repairs shared by simulated and real recordings |
| `mesqsim/correction.py`, `mesqsim/evaluate.py` | relative-heading targets and the heading correction; the baselines B0, B1, O1, O2 and the error metrics |
| `mesqsim/data/` | `amass.py` (AMASS to the 15 segment orientations), `diodem.py` (DIODEM's mechanical chains), `net.py` (downloads) |
| `mesqsim/synth.py` | synthetic motion in AMASS format, so that every test runs without licensed data |
| `mesqsim/splits.py`, `mesqsim/dataset.py` | the train, validation and test split by subject; many simulated sessions as one dataset |
| `mesqsim/config.py`, `mesqsim/paths.py`, `mesqsim/io.py`, `mesqsim/runs.py`, `mesqsim/manifest.py`, `mesqsim/pipeline.py` | the configuration, where data and outputs live, session folders, run identity, manifests, runs on disk |
| `mesqsim/results.py`, `mesqsim/figures.py`, `mesqsim/plotstyle.py` | the tables and figures of `docs/RESULTS_STEP1.md`, the figures in `docs/figures/`, one plot style |
| `mesqsim/cli.py`, `mesqsim/__main__.py`, `mesqsim/validation.py` | `python -m mesqsim`; the numeric checks (Step 1's are numbered 1.1 to 12.1) |
| `configs/default.yaml`, `configs/presets.yaml` | every simulator parameter with its source and label (Step 2 added the `geometry:` section); named presets |
| `scripts/` | `check_env.py`, `check_all.sh` (the full check), `validate.py`, `download_diodem.py`, `verify_amass.py`, `diodem_inspect.py`, `make_figures.py`, `make_results.py` |
| `manifests/` | `amass_archives.csv`, `amass_files.csv`, `smplh.csv`, `diodem.csv`, `split_manifest.csv` and its `split_manifest.sha256` |
| `docs/` | `MUST_KNOW.md` (read first), `PIPELINE.md`, `MATH.md`, `CODE.md`, `DATA.md`, `SIMULATOR.md`, `analysis_plan_DRAFT.md`, `RESULTS_STEP1.md` |
| `tests/` | `test_conventions.py`, `test_codec.py`, `test_step0.py`, `test_correction.py`, `test_sim_slice.py`, `test_sim_full.py`, `test_splits.py`, `test_amass.py`, `test_diodem.py`, `test_validation.py`, `test_repo_hygiene.py` |

Commands: `simulate`, `baselines`, `params`, `split`, `dataset`, `list-runs`, `clean` (`python -m mesqsim --help`).
The milestones M1 to M5 come before the tag scheme; their commits are named `M1: …` to `M5: …`
(`git log --oneline step1`).

## 3. Step 2: the geometric estimator (v2.0.0, tag `step2`)

Specification: [docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md).

| path | what it is |
|---|---|
| `mesqsim/estimators/contract.py` | the estimator contract: `Estimate(delta_rad, kappa, uncertain)`; `GeoTruth`, for GEO-K only |
| `mesqsim/estimators/clock.py` | one clock for parent and child: fresh samples, the causal pod-to-host time map, pairs, rotation rates |
| `mesqsim/estimators/hinge.py`, `twodof.py`, `rom.py`, `anatomy.py` | the solvers per joint type: hinge (the knees), 2-DoF (elbows, wrists, ankles), range of motion (hips, shoulders), the joint models in SMPL segment frames |
| `mesqsim/estimators/vmfilter.py`, `geo.py` | the von Mises filter per joint; GEO-K and GEO-ID themselves |
| `mesqsim/geo_fit.py` | fits GEO's beliefs on the training and validation splits, never on test (`fit-geometry`) |
| `mesqsim/geo_score.py`, `mesqsim/geo_sessions.py` | the scorer side: checks that may read the labels; sessions in memory for fitting and results |
| `mesqsim/diodem_geo.py` | GEO-ID on DIODEM's real IMUs against optical truth |
| `mesqsim/results_step2.py`, `scripts/make_results_step2.py` | `docs/RESULTS_STEP2.md` and its figures, from one command |
| `configs/default.yaml`, section `geometry:` | GEO's own beliefs, labelled like every parameter (it never reads the simulator's `error_model:`) |
| `docs/explain/` | the explanation pipeline: [README](docs/explain/README.md), then MENTAL_PICTURE, PIPELINE_EXPLAINED, MATH_EXPLAINED, CODE_EXPLAINED, THEORY |
| `project_explanations/DATASETS.md` | AMASS and DIODEM: what they are, how they were recorded, why they matter here |
| `tests/` | `test_clock.py`, `test_hinge.py`, `test_twodof_rom.py`, `test_filter.py`, `test_geo_contract.py`, `test_conventions_step2.py`, `test_config_write.py`, `test_results_step2.py`, `test_explain_links.py`, `test_explain_examples.py` |

Commands: `fit-geometry rom`, `fit-geometry calibration`, and `baselines --methods GEO-K GEO-ID`. Validation checks
21.1 to 33.1.

## 4. Step 3: the Mesquite twin, `twin_v1` and the model kit (v3.0.0, tag `step3`)

Specification: [docs/brief/STEP3_BRIEF.md](docs/brief/STEP3_BRIEF.md). The report is the last entry of
[WORKLOG.md](WORKLOG.md) (S3-M10).

| path | what it is |
|---|---|
| `mesqsim/real/` | Danyal's seven recordings: `recordings.py` (the list and its pairing with OptiTrack), `bvh.py` (the page's BVH and its kinematics), `motive.py` (Motive CSV), `markers.py` (marker naming), `align.py` (BVH and OptiTrack on one clock), `stats.py` (the statistics Φ1 to Φ12, one code path for real and twin files), `analysis.py` (the analysis of 2026-11-14 reproduced) |
| `mesqsim/twin/`, body | `smplh.py` (SMPL+H for a handful of vertices), `sites.py` (where pods and phone sit on the skin), `bodymotion.py` (their true poses), `body_report.py` |
| `mesqsim/twin/`, sensors and fusion | `physics.py` (what the gyroscope and accelerometer feel), `sensors.py` (what they report), `fusion.py` (the chip's six-axis fusion in the loop), `fusion_report.py` |
| `mesqsim/twin/`, wire, phone and app | `wire.py`, `phone.py` (ARKit, the app, the link, the phone's labels), `phone_app.py` (the app's Swift core ported line by line), `app_golden.py` |
| `mesqsim/twin/`, page | `page.py` (the page's pose chain, smoothing, root and recorder, ported from `mesquite-mocap` at `87d7c3b`), `page_frame.py` (the only frame conversion), `three_math.py`, `page_scenarios.py`, `golden.py` |
| `mesqsim/twin/`, sessions | `program.py` (motion programs, long sessions), `session.py` (one twin session end to end), `labels.py`, `suits.py` (the suit versions), `config.py` |
| `mesqsim/twin/`, calibration | `calibration.py` (the twin fitted to the recordings, gate G-T), `twin_stats.py`, `checks.py`, `session_checks.py` |
| `mesqsim/learn/`, data | `features.py` (inputs from deployable data only), `targets.py` (the only way to the targets), `store.py` (`twin_v1`: generate, extend, verify), `checks.py`, `curve.py` (how much data is enough) |
| `mesqsim/learn/`, the kit | `contract.py` (the model contract), `scorer.py` (`score-model`), `dummies.py` (Zeros, Cheat), `rigid.py` (RIGID and the chain along the tree), `through_page.py` (the page-level test) |
| `mesqsim/cli_step3.py`, `mesqsim/results_step3.py`, `scripts/make_results_step3.py` | Step 3's commands; `docs/RESULTS_STEP3.md` and its figure, from one command |
| `configs/twin.yaml` | every twin parameter with its source and label, including the values `twin-calibrate` fitted |
| `manifests/real_suit.csv`, `manifests/twin_v1.csv` | the recordings' and the dataset's files: path, size, SHA-256 |
| `tools/page_golden/`, `tools/app_golden/` | run the real page code (Node) and the real app core (Swift) on scripted inputs, writing `tests/fixtures/` |
| `tests/` | `test_real_readers.py`, `test_real_markers.py`, `test_real_recordings.py`, `test_real_stats.py`, `test_page_frame.py`, `test_page_twin.py`, `test_twin_body.py`, `test_twin_fusion.py`, `test_twin_wire.py`, `test_twin_phone.py`, `test_twin_session.py`, `test_twin_calibration.py`, `test_learn_features.py`, `test_learn_store.py`, `test_learn_curve.py`, `test_learn_rigid.py`, `test_model_kit.py`, `test_results_step3.py`, `test_through_page.py`, `test_step3_guide_examples.py` (the worked examples of `docs/step_3/`) |
| `docs/step_3/` | the explanations: [README.md](docs/step_3/README.md) (where to start), [DATASET_CARD.md](docs/step_3/DATASET_CARD.md) (`twin_v1` and what a loader can rely on), [LEARN_THE_MODEL.md](docs/step_3/LEARN_THE_MODEL.md) (the course: a two-day track to a trained, scored TCN-P) |

Validation checks 40.1 to 53.1. Danyal writes the learned models in `mesqsim/learn/{loader,models,loss,train}.py`
(not written yet), following [LEARN_THE_MODEL.md](docs/step_3/LEARN_THE_MODEL.md). `tests/test_model_kit.py` tests
them, and skips until they exist. The main commands:

```bash
python -m mesqsim real-stats                       # Φ1-Φ12 of the recordings ($MESQ_DATA_DIR/real_suit)
python -m mesqsim twin-session --seed 1            # one twin session end to end
python -m mesqsim twin-calibrate --draws 200 --keep 0.1 --predictive 10 --workers 5 \
    --real-summary <real_stats>/summary.json       # the calibration: 3,149 s on 5 workers (round 3)
python -m mesqsim twin-dataset --name twin_v1 --calibration <twin_calibration run> \
    --variants-train 3 --variants-eval 2 --workers 5 --seed 1 --max-gb 20   # the data: 19.42 GB, about 1.4 h
python -m mesqsim verify-dataset --name twin_v1    # re-hash against manifests/twin_v1.csv
python -m mesqsim score-model --model <module>:<Class> --dataset twin_v1 --split val   # or --model B0
python scripts/make_results_step3.py --workers 5 --page-sessions 12                    # docs/RESULTS_STEP3.md
```

Written on 2026-11-21, at Danyal's request: `docs/step_3/`'s `README.md`, `DATASET_CARD.md` and `LEARN_THE_MODEL.md`.
Still not written, at Danyal's earlier request (ask for them by name):
- in `docs/step_3/`: `MENTAL_PICTURE.md`, `PIPELINE_EXPLAINED.md`, `MATH_EXPLAINED.md`, `CODE_EXPLAINED.md`,
  `THEORY.md`, `REFERENCES.md` with `references.bib`, `FIDELITY.md` and `MODEL_CARD_TEMPLATE.md`;
- the Step 3 updates of `README.md`, `docs/SIMULATOR.md`, `docs/DATA.md`, `docs/PIPELINE.md`, `docs/MATH.md`,
  `docs/CODE.md`, `docs/MUST_KNOW.md` and `docs/analysis_plan_DRAFT.md`.

## 5. Files that every step extends

| path | how the steps share it |
|---|---|
| `CLAUDE.md` | the rules; each brief adds its own |
| `WORKLOG.md` | one entry per milestone, appended in order |
| `mesqsim/cli.py` | one `python -m mesqsim` for every step (Step 3's commands are registered from `mesqsim/cli_step3.py`) |
| `mesqsim/validation.py`, `scripts/validate.py` | the numeric checks, numbered by step: 1.1-12.1 (Step 1), 21.1-33.1 (Step 2), 40.1-53.1 (Step 3) |
| `docs/VALIDATION.md` | written by `bash scripts/check_all.sh` (ruff, every test, every check), with the commit it ran at |
| `tests/conftest.py`, `pyproject.toml` | shared fixtures and the pytest markers (`amass`, `diodem`, `real`, `node`, `swift`, `slow`) |
| `docs/figures/` | each step's figures, named after its results document (`results_*`, `results_step2_*`, `results_step3_*`) |

## 6. Data and outputs (outside git)

| where | what |
|---|---|
| `$MESQ_DATA_DIR` (default `data/`; on Danyal's Mac `../mesquite_work/data`) | `amass/`, `body_models/`, `diodem/`, `real_suit/` (Danyal's recordings), `mesquite-mocap/` (the suit's code, read-only, pinned at `87d7c3b`) |
| `$MESQ_OUTPUT_DIR` (default `outputs/`; on Danyal's Mac `../mesquite_work/outputs`) | simulator runs (`run_*`), fits (`fit_calibration_*`), the twin's calibrations (`twin_calibration_*`), `datasets/twin_v1/`, `results_step3/`, `scores/` |

The manifests in `manifests/` record every data file's path, size and SHA-256. `tests/test_repo_hygiene.py` fails if a
data file or any file over 5 MB is committed.

## 7. Branches, tags, versions and commits

- **One long-lived branch: `main`.** Work on a milestone happens on a short-lived branch (`s<step>-m<n>`). The branch
  is merged into `main` and then deleted, and a tag keeps the milestone.
- **Tags.** `v<step>.0.0` is a release (Semantic Versioning: a new step is a new major version). `step<k>` marks the
  end of step k. `step<k>-m<n>` marks milestone n of step k. `v1.0.0` = `step1` and `v2.0.0` = `step2`, both placed
  on 2026-11-21 on the commits where those steps ended. `v3.0.0` = `step3` plus this organisation of the repository.
- **Version.** `pyproject.toml` and `mesqsim.__version__` read `3.0.0`.
- **Commits.** A milestone's commits start with the milestone (`S3-M7: …`). Other commits use the Conventional
  Commits prefixes (`docs:`, `fix:`, `chore:`, `test:`).
- **Looking around.** `git log --oneline step2..step3` lists Step 3's work. `git show step3-m7` shows one milestone.
  `git switch --detach v2.0.0` checks out the tree as it was at the end of Step 2 (`git switch main` returns).
- **GitHub** (`origin`, public): one branch, `main` (the default branch), and every tag. The merged branches
  `m5-results` and `s2-m0` … `s2-m8` were deleted on 2026-11-21; pull requests #1 to #7 still show what each merged.

## 8. Where to start

| you want to | read or run |
|---|---|
| understand the project | [docs/MUST_KNOW.md](docs/MUST_KNOW.md), then [docs/explain/README.md](docs/explain/README.md) |
| write and train the models | [docs/step_3/DATASET_CARD.md](docs/step_3/DATASET_CARD.md), then [docs/step_3/LEARN_THE_MODEL.md](docs/step_3/LEARN_THE_MODEL.md) |
| run it from a fresh clone | the quickstart in [README.md](README.md) (Steps 1 and 2); Step 3's commands are in §4 above |
| see the numbers | [RESULTS_STEP1.md](docs/RESULTS_STEP1.md), [RESULTS_STEP2.md](docs/RESULTS_STEP2.md), [RESULTS_STEP3.md](docs/RESULTS_STEP3.md), [VALIDATION.md](docs/VALIDATION.md) |
| know where a parameter's value came from | `python -m mesqsim params`, and the source and label next to each value in `configs/` |
| check that everything still works | `bash scripts/check_all.sh` (about 13 minutes on an M2 Pro with the data present) |

The requested tutoring notes are in `docs/step3_simulation_explanation/`:
[MOTION_SIMULATION_TUTORIAL.md](docs/step3_simulation_explanation/MOTION_SIMULATION_TUTORIAL.md) brings together
Modules 0 and 1 (the project map and rotation mathematics), expanded derivations, code and test references,
worked examples, runnable snippets, and self-check answers. Its nine-module reading plan describes the continuation.
[LESSON_2_THE_LOADER.md](docs/step3_simulation_explanation/LESSON_2_THE_LOADER.md) is the guided build of the
course's Lesson 2: the loader's contract, mathematics, pseudocode, tests and drills, without the finished code.
