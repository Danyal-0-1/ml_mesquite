# Mesquite ML: the suit simulator (Step 1), the geometric estimator (Step 2), the Mesquite twin and model kit (Step 3)

Research repository for Danyal Khorami's Mesquite project. It is public by Danyal's choice (2026-11-12); licensed
data never enter it. The specifications are the three briefs: [docs/brief/STEP1_BRIEF.md](docs/brief/STEP1_BRIEF.md)
(done, release `v1.0.0`), [docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md) (done, `v2.0.0`) and
[docs/brief/STEP3_BRIEF.md](docs/brief/STEP3_BRIEF.md) (done, `v3.0.0`, kept local). The rules for every work
session are in [CLAUDE.md](CLAUDE.md). **New here? Read [docs/MUST_KNOW.md](docs/MUST_KNOW.md) first, then learn the
project from [docs/explain/README.md](docs/explain/README.md)** (five files, from a mental picture to the theory). The
two datasets are explained in [project_explanations/DATASETS.md](project_explanations/DATASETS.md). **Where is
everything?** [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) maps every folder and file to the step that built it, and
[CHANGELOG.md](CHANGELOG.md) lists the releases and their milestones.

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

## Releases and status

One branch, `main`; one release per step ([CHANGELOG.md](CHANGELOG.md) has every milestone, [WORKLOG.md](WORKLOG.md)
every entry).

| release | step | state | tags | results |
|---|---|---|---|---|
| `v1.0.0` | Step 1: data, environment, simulator, Step 0, baselines B0, B1, O1, O2 | done 2026-11-16; on GitHub | `step1` | [docs/RESULTS_STEP1.md](docs/RESULTS_STEP1.md) |
| `v2.0.0` | Step 2: the geometric estimator (GEO-K, GEO-ID) and the explanation pipeline | done 2026-11-18; on GitHub (pull requests #1 to #7) | `step2`, `step2-m0` … `step2-m8` | [docs/RESULTS_STEP2.md](docs/RESULTS_STEP2.md) |
| `v3.0.0` | Step 3: the Mesquite twin calibrated on Danyal's recordings, the training data `twin_v1`, the model kit, the numbers to beat | done 2026-11-21; **local only** (Danyal's answer of 2026-11-17) | `step3`, `step3-m0` … `step3-m10` | [docs/RESULTS_STEP3.md](docs/RESULTS_STEP3.md) |
| next | Step 4: fusion of geometry and learning | not started; Danyal's learned models first | | |

**Tags.** `v<step>.0.0` is a release; `step<k>` marks the end of step k; `step<k>-m<n>` marks one milestone.
`git log --oneline step2..step3` lists Step 3's work; `git show step3-m7` shows one milestone.

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
will be tested, and how; for Danyal to freeze). Step 3's commands are in [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md) §4.

## Where things are

The full map, file by file and step by step, is [REPOSITORY_GUIDE.md](REPOSITORY_GUIDE.md).

| path | what | step |
|---|---|---|
| [mesqsim/](mesqsim/) | the Python package: conventions, codec, body tree, Step 0, baselines, runs, results, validation checks | 1, 2, 3 |
| [mesqsim/sim/](mesqsim/sim/), [mesqsim/data/](mesqsim/data/) | the suit simulator; AMASS and DIODEM access | 1 |
| [mesqsim/estimators/](mesqsim/estimators/) | GEO, the geometric estimator | 2 |
| [mesqsim/real/](mesqsim/real/), [mesqsim/twin/](mesqsim/twin/), [mesqsim/learn/](mesqsim/learn/) | Danyal's recordings; the Mesquite twin; the model kit and `twin_v1` | 3 |
| [tests/](tests/) | the pytest suite; runs without licensed data | 1, 2, 3 |
| [scripts/](scripts/), [tools/](tools/) | one-command entry points (checks, downloads, results); the golden-fixture makers | 1, 2, 3 |
| [configs/](configs/) | `default.yaml` (simulator and GEO, every parameter labelled), `presets.yaml`, `twin.yaml` (the twin) | 1, 2, 3 |
| [manifests/](manifests/) | name, size and SHA-256 of every data file | 1, 3 |
| [docs/](docs/) | MUST_KNOW, PIPELINE, MATH, CODE, DATA, SIMULATOR, VALIDATION, RESULTS_STEP1-3, analysis_plan_DRAFT, figures; `brief/` holds the three briefs | 1, 2, 3 |
| [docs/explain/](docs/explain/README.md) | the explanation pipeline: MENTAL_PICTURE → PIPELINE_EXPLAINED → MATH_EXPLAINED → CODE_EXPLAINED → THEORY | 2 |
| [project_explanations/](project_explanations/DATASETS.md) | the two datasets (AMASS, DIODEM): what they are, how they were recorded, why they matter here | 2 |
| [CHANGELOG.md](CHANGELOG.md), [WORKLOG.md](WORKLOG.md) | the releases; one entry per session or milestone (what changed, commands run, decisions) | all |
| `data/`, `outputs/` | git-ignored; licensed data, recordings and generated runs never enter git | all |
