# Changelog

Every release of this repository. There is one release per project step. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the versions follow
[Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html), with a new step as a new major version. Each
milestone has a tag; [WORKLOG.md](WORKLOG.md) has the full entry for each one.
[REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) maps every file to its step.

## [Unreleased]

### Changed
- Step 3 is published. On 2026-11-21 the person running the session had `main` and every tag pushed to GitHub.
  GitHub's default branch is now `main`, and its merged branches (`m5-results`, `s2-m0` … `s2-m8`) were deleted.

Step 4, the fusion of geometry and learning, has not started. Danyal's learned models
(`mesqsim/learn/{loader,models,loss,train}.py`) come first.

## [3.0.0] - 2026-11-21 - Step 3: the Mesquite twin, `twin_v1` and the model kit

Tags: `v3.0.0` (this release, with the repository organised), `step3` (the end of the Step 3 work), `step3-m0` …
`step3-m10`. Specification: [docs/brief/STEP3_BRIEF.md](docs/brief/STEP3_BRIEF.md). Results:
[docs/RESULTS_STEP3.md](docs/RESULTS_STEP3.md). The report is WORKLOG entry S3-M10. **Local only at release:**
Danyal's answer of 2026-11-17 kept Step 3 off the public repository ([CLAUDE.md](CLAUDE.md) rule 2). It was published
later the same day (see Unreleased).

### Added
- **S3-M0** (`step3-m0`): Step 3 starts. The brief is saved, `CLAUDE.md` gets the Step 3 rules, and the recordings'
  manifest (`manifests/real_suit.csv`) and the answers to the brief's questions are recorded.
- **S3-M1** (`step3-m1`): the recordings measured (`mesqsim/real/`). The readers cover the page's BVH and Motive CSV.
  OptiTrack markers are named, BVH and OptiTrack share one clock, the statistics Φ1-Φ12 are computed, and the
  analysis of 2026-11-14 is reproduced (`real-stats`).
- **S3-M2** (`step3-m2`): the page twin (`mesqsim/twin/page.py`, ported from `mesquite-mocap` at `87d7c3b`). It has
  one frame-conversion module and a golden test against the real page code run in Node (`tools/page_golden/`).
- **S3-M3** (`step3-m3`): bodies, sites and sensor physics. SMPL+H vertices carry the pods and the phone on the skin;
  the module computes what each gyroscope and accelerometer feels (`twin-body`).
- **S3-M4** (`step3-m4`): sensor errors, the chip's fusion in the loop, the wire, and the phone with its app. The
  app's Swift core is ported line by line and checked against the real one (`tools/app_golden/`, `twin-fusion`).
- **S3-M5** (`step3-m5`): the procedure, the slips, the motion programs, long sessions, and one twin session end to
  end (`twin-session`).
- **S3-M6** (`step3-m6`): the calibration. The twin is fitted to the recordings by approximate Bayesian computation,
  with posterior predictive checks; gate G-T passes (`twin-calibrate`). The fitted values are in
  `configs/twin.yaml`.
- **S3-M7** (`step3-m7`): the training data `twin_v1`, with 665 sessions, 200.4 h and 19.42 GB outside git. Its
  manifest is `manifests/twin_v1.csv` (`twin-dataset`, `verify-dataset`).
- **S3-M8** (`step3-m8`): the model kit (`mesqsim/learn/`): the contract, the scorer (`score-model`), RIGID and the
  chain along the tree, two dummy models, and the tests that Danyal's loader, networks, loss and training loop must
  pass.
- **S3-M9** (`step3-m9`): the numbers to beat on `twin_v1`'s held-out subjects, the twin ablations, and the test
  through the page (`scripts/make_results_step3.py`, `docs/RESULTS_STEP3.md`).
- **S3-M10** (`step3-m10` = `step3`): the Step 3 report, the full check, and the tag `step3`.
- Validation checks 40.1-53.1.

### Changed (the repository, 2026-11-21)
- One branch. Step 3's stacked milestone branches were fast-forwarded into `main`, and every other local branch and
  worktree was removed. Each milestone keeps a tag (`step3-m0` … `step3-m10`, and the missing `step2-m0` and
  `step2-m8`). The releases are tagged `v1.0.0`, `v2.0.0` and `v3.0.0`.
- Added `CHANGELOG.md`, `REPOSITORY_GUIDE.md` and `tests/test_repo_guide.py`, which checks that every link and path
  in the README, the changelog and the guide resolves.
- `README.md` now covers all three steps, with a table of releases and the layout. `CLAUDE.md` records that Step 3
  is done, that the repository has one branch, and the tag scheme.
- The package version went from 0.1.0 to 3.0.0 (`pyproject.toml`, `mesqsim.__version__`), and each subpackage's
  docstring names its step.

## [2.0.0] - 2026-11-18 - Step 2: the geometric estimator GEO

Tags: `v2.0.0` = `step2`, `step2-m0` … `step2-m8`. Specification:
[docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md). Results: [docs/RESULTS_STEP2.md](docs/RESULTS_STEP2.md).
The report is WORKLOG entry S2-M8. Pushed to GitHub and merged through pull requests #1 to #7.

### Added
- **S2-M0** (`step2-m0` = `step1`): Step 1 closed. This adds the M5 work log, the M6 documents, the AMASS download
  and split, and the results on AMASS; `main` was created (pull request #1).
- **S2-M1** (`step2-m1`): the estimator contract, one clock for parent and child, pairs, rotation rates, the
  `geometry:` config section, and the scorer integration (#2).
- **S2-M2, S2-M3** (`step2-m2`, `step2-m3`): the hinge solver (GEO-K, axis identification, GEO-ID), the 2-DoF
  constraint, and the range of motion with Ω fitted on AMASS train; GEO-ID's facing comes from the knees (#3).
- **S2-M4, S2-M5** (`step2-m4`, `step2-m5`): the filter replay and its calibration on validation, the knee axes tied
  to the T-pose, and GEO-ID on DIODEM (#4).
- **S2-M6** (`step2-m6`): the Step 2 results on AMASS test subjects and DIODEM (#5).
- **S2-M7** (`step2-m7`): the explanation pipeline, `docs/explain/`, with its two tests (#6).
- **S2-M8** (`step2-m8` = `step2`): the documentation completed, `project_explanations/DATASETS.md`, and the
  Step 2 report (#7).
- Validation checks 21.1-33.1.

## [1.0.0] - 2026-11-16 - Step 1: the simulator, Step 0 and the baselines

Tags: `v1.0.0` = `step1`. Specification: [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md). Results:
[docs/RESULTS_STEP1.md](docs/RESULTS_STEP1.md). Built 2026-10-22 to 2026-10-23 on the collaborator's machine; the
step was closed on Danyal's Mac in S2-M0. The milestones have no tags of their own, and their commits are named
`M1: …` to `M5: …`.

### Added
- **M1**: the package skeleton, the conventions, the wire codec, the checks and CI.
- **M2**: data access. This covers the AMASS and SMPL+H instructions, the DIODEM download, the manifests, the loaders
  and the mapping from AMASS to the suit.
- **M3**: a thin end-to-end slice: offsets, then frames, then Step 0, then B0 and O2.
- **M4**: the full simulator, with its error model, timing, loss, reboots, the baselines B1 and O1, the configs and
  the command line.
- **M5**: the splits, dataset generation, the sweeps, and the results command (`scripts/make_results.py`).
- **M6** (done in S2-M0): `docs/MUST_KNOW.md`, the analysis-plan draft, and the README quickstart.
- Validation checks 1.1-12.1.
