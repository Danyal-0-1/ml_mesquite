# Mesquite ML: Step 1 (data, environment, simulator) and Step 2 (geometric estimator)

Research repository for Danyal Khorami's Mesquite project. It is public by Danyal's choice (2026-11-12); licensed
data never enter it. The specifications are [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md) (done) and
[docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md) (done, tag `step2`); the rules for every work session are in
[CLAUDE.md](CLAUDE.md). **New here? Read [docs/MUST_KNOW.md](docs/MUST_KNOW.md) first, then learn the project from
[docs/explain/README.md](docs/explain/README.md)** (five files, from a mental picture to the theory). The two
datasets are explained in [project_explanations/DATASETS.md](project_explanations/DATASETS.md).

## The whole project in one picture

**The suit.** A person wears fifteen watches ("pods"): pelvis, chest, head, both upper arms, forearms, hands,
thighs, shins and feet. In each pod a motion chip fuses its gyroscope and accelerometer into an orientation.
Gravity tells every chip which way is down, so **tilt is right**. There is no compass, so each chip invents its
own private **"north"** when it switches on. That north slowly slides: a little every minute (drift and noise),
and a little more every time the pod spins around (gyroscope scale error).

**The data path.** Thirty-two times a second each pod radios a 16-byte postcard: its orientation, a postcard
number (`count`) and the pod-clock time its picture was taken (`ms_lo`). A hub passes the postcards to a laptop,
which turns each pod into a bone of an avatar.

**The problem.** The norths disagree, so a joint computed from two pods shows a false twist about the vertical:
the thigh twists against the shin when the knee only bent. Step 2 estimates, for each of the 14 joints, how
far the child pod's north has slid relative to its parent's: the **relative heading δ**. Only relative headings
can ever be estimated. Turn the whole room about the vertical and every pod sends the same numbers, so north
and the body's facing in the room are unknowable.

**Step 1 builds the ground Step 2 stands on:**

1. Take real human motion from AMASS and compute how each of the 15 body segments was oriented at every instant.
2. Put virtual pods on the segments. Give each one its own sliding north, its own clocks and its own radio
   losses, as the real hardware has.
3. Write out the postcards the real suit would have sent. We add the errors ourselves, so we know the true δ
   for every joint at every instant (the **labels**).
4. Decode the postcards with **Step 0**, the same code that will decode real suit recordings.
5. Measure how wrong the joints are with no correction (B0), with the simplest real correction (B1, box
   registration) and with two oracles (O1, O2).
6. As a reality check, use DIODEM: real IMUs on mechanical chains, with cameras recording the truth.

```text
AMASS sequence (SMPL+H poses) ──► 15 segment orientations s_i(t)
   + random facing, mounting m_i, session timeline (box → strapping → T-pose → idle → motion)
   ──► each pod's chip samples at its own rate f_i (50.1–60 Hz)         [labels: true ψ_i(t)]
   ──► reported q_i = Y(−ψ_i)·s_i·m_i·Exp(η_i), with w ≥ 0
   ──► 32 Hz send grid: newest sample, held repeats, count, ms_lo, int16
   ──► radio loss in bursts; host arrival times
   ──► frames.parquet (the same schema a real logger writes)   +   labels.parquet (kept separate)
STEP 0, shared with real recordings: unwrap ms_lo; fresh, held or lost; norm check; box registration; T-pose
   ──► per-pod series ──► baselines B0, B1, O1, O2 ──► error against the labels ──► tables and figures
DIODEM: download → load → drop the magnetometer → six-axis VQF → relative heading against optical truth
```

**Step 2 adds the geometry.** The body itself constrains δ: a knee is a hinge, so the thigh pod and the shin pod
see the same hinge pin, and the difference between the pin's two shadows on the floor is δ; an elbow, wrist or
ankle has two such pins; a hip or shoulder only has walls (its range of motion), which say something only when the
motion reaches them. Per joint, every second, the estimator solves the last 8 s of motion and multiplies the answer
into a von Mises belief (a mean and a confidence κ) that fades with time and restarts at a reboot. **GEO-K** knows
the true pins and straps (simulator truth: a ceiling); **GEO-ID** finds them from the motion and the T-pose (the
deployable estimator). Both are scored against B1 on held-out AMASS subjects, and GEO-ID on DIODEM's real IMUs
([docs/RESULTS_STEP2.md](docs/RESULTS_STEP2.md)).

```text
STEP 0 series + session.json ──► one clock for parent and child (lower envelope), pairs, body rates
   ──► every 1 s, over the last 8 s, per joint type: hinge closed form (knees) │ 2-DoF constraint (GEO-K) │
       range of motion in T-pose frames (GEO-ID's other joints)  ──►  (δ̂_w, κ_w), calibrated on validation
   ──► von Mises filter per joint: start at B1, widen with time, multiply in each window, reset at a reboot
   ──► δ̂, κ and an "uncertain" flag at every 32 Hz step ──► scored like the baselines (subject = unit)
```

**What goes in:** AMASS motion files and the SMPL+H body model (licensed; you download them by hand), the DIODEM
dataset (open; scripted download) and one YAML config. **What comes out:** for each simulated session, a frames
table identical in schema to a real recording, a separate labels table, the decoded per-pod series, error tables
for the baselines and the Step 2 estimators, and the two results documents.

## Status

| milestone | content | state |
|---|---|---|
| M1 | skeleton, environment, conventions, wire codec, CI | done on branch `m1-skeleton`; on GitHub through `s2-m0` |
| M2 | data: AMASS/SMPL+H instructions, DIODEM download, manifests, loaders, mapping | done on branch `m2-data` except the AMASS-dependent checks, which wait for the AMASS download |
| M3 | thin end-to-end slice (offsets only → frames → Step 0 → B0, O2) | done on branch `m3-slice` (synthetic motion; AMASS when downloaded) |
| M4 | full simulator, configs, CLI | done on branch `m4-simulator` |
| M5 | splits, dataset, baselines, first results | done on branch `m5-results`; work log written in S2-M0 |
| M6 | complete documentation, report, tag `step1` | done in S2-M0 (branch `s2-m0`); **tag `step1`** at `9b06673` (2026-11-16) |
| S2-M0 | close Step 1: M5 work log, M6, AMASS download and split, results on AMASS | done; merged by Danyal (pull request #1); `main` = `step1`; the split and analysis plan wait for his freeze |
| S2-M1 | Step 2 estimator contract, common clock, pairs, rates, `geometry:` config, scorer integration | done on branch `s2-m1`; merged into `main` (pull request #2), tag `step2-m1` |
| S2-M2, S2-M3 | hinge (GEO-K, axis identification, GEO-ID); 2-DoF constraint; range of motion with Ω fitted on AMASS train; GEO-ID facing from the knees | done on branch `s2-m2`; merged into `main` (one pull request for both), tags `step2-m2`, `step2-m3` |
| S2-M4, S2-M5 | filter replay and the validation calibration (κ scale, model error, q, κ_hold); T-pose-tied knee identification; GEO-ID on DIODEM | done on branch `s2-m4`; merged into `main` (one pull request), tags `step2-m4`, `step2-m5` |
| S2-M6 | the Step 2 results on AMASS test subjects and DIODEM ([docs/RESULTS_STEP2.md](docs/RESULTS_STEP2.md)); GEO-ID's tied knee-axis test relaxed and the calibration refitted on validation; the simulator's idle turns made the wearer's (shared by every pod) | done on branch `s2-m6`; merged into `main`, tag `step2-m6` |
| S2-M7 | the explanation pipeline ([docs/explain/](docs/explain/README.md)) with its two tests | done on branch `s2-m7`; merged into `main`, tag `step2-m7` |
| S2-M8 | documentation complete, the report (last [WORKLOG.md](WORKLOG.md) entry) | done on branch `s2-m8`; merged into `main`, **tag `step2`** |

**Tags.** `step1` marks the end of Step 1 (simulator, Step 0, baselines). Each Step 2 milestone is merged into
`main` and tagged `step2-m1` … `step2-m7`; `step2` marks the end of Step 2. `git log --oneline step1..step2` lists the
Step 2 work.

## Quickstart: from a fresh clone to the Step 1 and Step 2 results

Commands are for bash (on Windows, Git Bash). Each step says what it produces and how long it took on the
collaborator's machine where that was measured.

```bash
# 1. Python 3.11 environment (on the collaborator's machine: D:\mesquite_work\venv)
git clone https://github.com/Danyal-0-1/ml_mesquite && cd ml_mesquite
python3.11 -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
#    macOS alternative (used on Danyal's Mac, 2026-11-16): uv fetches Python 3.11 itself
#    uv venv --python 3.11 .venv && uv pip install --python .venv/bin/python \
#        --index-strategy unsafe-best-match -r requirements.txt

# 2. Optional: keep data and outputs outside the repository (defaults: ./data and ./outputs, both git-ignored)
export MESQ_DATA_DIR=/path/to/data MESQ_OUTPUT_DIR=/path/to/outputs

# 3. Data (docs/DATA.md)
python scripts/download_diodem.py     # DIODEM: open licence, scripted
#    AMASS and SMPL+H: register yourself at amass.is.tue.mpg.de and mano.is.tue.mpg.de, download the
#    "SMPL+H G" archives into $MESQ_DATA_DIR/amass/archives/ (docs/DATA.md §1.1), then:
python scripts/verify_amass.py        # hashes, extracts, indexes -> manifests/amass_*.csv

# 4. Check the environment, then run the full check (ruff, tests, validation -> docs/VALIDATION.md)
python scripts/check_env.py
bash scripts/check_all.sh             # 5 min 19 s on 2026-11-13 (12 logical cores); 3 min 46 s on an M2 Pro, 2026-11-17
python scripts/make_figures.py        # regenerates docs/figures

# 5. Simulate one session and score it (docs/SIMULATOR.md)
python -m mesqsim simulate --seed 1
python -m mesqsim baselines --run "$MESQ_OUTPUT_DIR"/run_g*_s1

# 6. The results: the split (once, after AMASS; Danyal freezes it), then every table and figure
python -m mesqsim split --write       # -> manifests/split_manifest.csv and its SHA-256
python scripts/make_results.py        # -> docs/RESULTS_STEP1.md (Step 1 baselines; about 12 h on AMASS)

# 7. Step 2: the estimator's fitted beliefs (train / validation only; already in configs/default.yaml), then the
#    results. Run in a clean checkout: the commit is written into the results.
python -m mesqsim fit-geometry rom --source amass --write                  # Ω on train
python -m mesqsim fit-geometry calibration --source amass --seed 1 --write # GEO-ID on validation (about 3 min)
python -m mesqsim fit-geometry calibration --source amass --seed 1 --estimator K --write   # GEO-K's own
python scripts/make_results_step2.py --source amass --seed 1                # -> docs/RESULTS_STEP2.md (about 20 min)

# 8. One session through GEO-K and GEO-ID (docs/explain/PIPELINE_EXPLAINED.md follows this one, stage by stage)
python -m mesqsim simulate --seed 1 --motion "$MESQ_DATA_DIR/amass/CMU/13/13_02_poses.npz" --set session.idle_s=562
python -m mesqsim baselines --run <printed folder> --methods B0 B1 O1 O2 GEO-K GEO-ID
```

**Where to read next:** [docs/MUST_KNOW.md](docs/MUST_KNOW.md) (what to understand before trusting a number),
[docs/explain/README.md](docs/explain/README.md) (the explanation pipeline: start here to learn the project),
[docs/PIPELINE.md](docs/PIPELINE.md) (every stage and file format), [docs/MATH.md](docs/MATH.md),
[docs/RESULTS_STEP2.md](docs/RESULTS_STEP2.md), [docs/analysis_plan_DRAFT.md](docs/analysis_plan_DRAFT.md) (what
will be tested, and how; for Danyal to freeze).

## Where things are

| path | what |
|---|---|
| [mesqsim/](mesqsim/) | the Python package: conventions, codec, body tree, correction, data loaders, simulator (`sim/`), Step 0, baselines, the Step 2 estimators (`estimators/`), fits, results, validation checks |
| [tests/](tests/) | pytest suite; runs without licensed data |
| [scripts/](scripts/) | `check_env.py`, `check_all.sh`, `validate.py`, `download_diodem.py`, `verify_amass.py`, `make_figures.py`, `make_results.py`, `make_results_step2.py` |
| [manifests/](manifests/) | name, size and SHA-256 of every data file |
| [docs/](docs/) | MUST_KNOW, PIPELINE, MATH, CODE, DATA, SIMULATOR, VALIDATION, RESULTS_STEP1, RESULTS_STEP2, analysis_plan_DRAFT, figures; `brief/` holds both briefs |
| [docs/explain/](docs/explain/README.md) | the explanation pipeline: MENTAL_PICTURE → PIPELINE_EXPLAINED → MATH_EXPLAINED → CODE_EXPLAINED → THEORY |
| [project_explanations/](project_explanations/DATASETS.md) | the two datasets (AMASS, DIODEM): what they are, how they were recorded, why they matter here |
| [configs/](configs/) | `default.yaml` (every parameter, labelled), `presets.yaml` |
| [WORKLOG.md](WORKLOG.md) | one entry per session or milestone: what changed, commands run, decisions |
| `data/`, `outputs/` | git-ignored; licensed data and generated runs never enter git |
