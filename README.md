# Mesquite ML, Step 1: data, environment, simulator


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
the thigh twists against the shin when the knee only bent. Step 2 will estimate, for each of the 14 joints, how
far the child pod's north has slid relative to its parent's: the **relative heading δ**. Only relative headings
can ever be estimated. Turn the whole room about the vertical and every pod sends the same numbers, so north
and the body's facing in the room are unknowable.

**Step 1 (this repository) builds the ground Step 2 stands on:**

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

**What goes in:** AMASS motion files and the SMPL+H body model (licensed; you download them by hand), the DIODEM
dataset (open; scripted download) and one YAML config. **What comes out:** for each simulated session, a frames
table identical in schema to a real recording, a separate labels table, the decoded per-pod series, and error
tables and figures for the four baselines.

## Status

| milestone | content | state |
|---|---|---|
| M1 | skeleton, environment, conventions, wire codec, CI | done on branch `m1-skeleton` (not pushed: repository is public) |
| M2 | data: AMASS/SMPL+H instructions, DIODEM download, manifests, loaders, mapping | done on branch `m2-data` except the AMASS-dependent checks, which wait for the AMASS download |
| M3 | thin end-to-end slice (offsets only → frames → Step 0 → B0, O2) | done on branch `m3-slice` (synthetic motion; AMASS when downloaded) |
| M4 | full simulator, configs, CLI | done on branch `m4-simulator` |
| M5 | splits, dataset, baselines, first results | done on branch `m5-results`: machinery complete; results provisional on synthetic motion until AMASS |
| M6 | complete documentation, report, tag `step1` | not started |

## Quickstart (so far)

```bash
# 1. Python 3.11 environment (on the collaborator's machine: D:\mesquite_work\venv)
python3.11 -m venv .venv
source .venv/bin/activate            # Windows Git Bash: source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 2. Optional: keep data and outputs elsewhere (defaults: ./data and ./outputs, both git-ignored)
export MESQ_DATA_DIR=/path/to/data MESQ_OUTPUT_DIR=/path/to/outputs

# 3. Data: DIODEM is scripted; AMASS and SMPL+H need your own registration (docs/DATA.md)
python scripts/download_diodem.py
python scripts/verify_amass.py        # after downloading AMASS archives by hand

# 4. Check the environment, then run the full check (ruff, tests, validation → docs/VALIDATION.md)
python scripts/check_env.py
bash scripts/check_all.sh
python scripts/make_figures.py        # regenerates docs/figures

# 5. Simulate one session and score it (see docs/SIMULATOR.md)
python -m mesqsim simulate --seed 1
python -m mesqsim baselines --run "$MESQ_OUTPUT_DIR"/run_g*_s1

# 6. The results: splits (after AMASS), then every table and figure from one command
python -m mesqsim split --write
python scripts/make_results.py        # -> docs/RESULTS_STEP1.md
```

## Where things are

| path | what |
|---|---|
| [mesqsim/](mesqsim/) | the Python package: conventions, codec, body tree, correction, data loaders, synthetic motion, figures, validation checks |
| [tests/](tests/) | pytest suite; runs without licensed data |
| [scripts/](scripts/) | `check_env.py`, `check_all.sh`, `validate.py`, `download_diodem.py`, `verify_amass.py`, `make_figures.py` |
| [manifests/](manifests/) | name, size and SHA-256 of every data file |
| [docs/](docs/) | PIPELINE, MATH, CODE, DATA, SIMULATOR, VALIDATION, RESULTS_STEP1, figures; later MUST_KNOW |
| [configs/](configs/) | `default.yaml` (every parameter, labelled), `presets.yaml` |
| [WORKLOG.md](WORKLOG.md) | one entry per session or milestone: what changed, commands run, decisions |
| `data/`, `outputs/` | git-ignored; licensed data and generated runs never enter git |
