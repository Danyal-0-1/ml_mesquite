# Work log

One entry per work session or milestone, newest last. A commit cannot contain its own hash, so each entry lists
the hashes of the previous entry's commits at its start.

---

## 2026-10-22 · M1 · branches `main`, `m1-skeleton`

**Machine.** Windows 11 Home (10.0.22000), Intel64 Family 6 Model 165 (12 logical cores). Python 3.11.9 in
`D:\mesquite_work\venv`.

**Before starting.**
- The repository was empty (no commits). Its remote `github.com/Danyal-0-1/ml_mesquite` reported
  `"private": false, "visibility": "public"` (`curl -s https://api.github.com/repos/Danyal-0-1/ml_mesquite`).
  The brief requires private. **Decision (collaborator):** commit each milestone on local branches; push and
  open pull requests only after the collaborator confirms that Danyal has made the repository private.
- Only Python 3.14 was installed. **Decision (collaborator):** installed Python 3.11.9 with the Python install
  manager (`py install 3.11`).
- The repository is inside OneDrive. **Decision (collaborator):** data, outputs and the virtual environment live
  outside it, in `D:\mesquite_work\{data,outputs,venv}`, through user environment variables `MESQ_DATA_DIR` and
  `MESQ_OUTPUT_DIR` (set with `setx`). Elsewhere the defaults are `data/` and `outputs/` in the repo.
- AMASS and SMPL+H are not downloaded yet; the collaborator will register. Until then, synthetic motion only.
- `gh` (GitHub CLI) is not installed, so pull requests will need `gh` or the GitHub web page once pushing is allowed.

**Files added.**
- `docs/brief/STEP1_BRIEF.md` (the brief, unchanged; SHA-256
  `c20ceab7870efbf809329c05f96114b47f114e341dbd22b52d33977649647ab1`, 32,375 bytes), `CLAUDE.md`, `.gitignore`,
  `.gitattributes` (first commit on `main`).
- `mesqsim/`: `conventions.py`, `codec.py`, `body.py`, `correction.py`, `manifest.py`, `paths.py`,
  `validation.py`, `__init__.py`.
- `tests/`: `test_codec.py`, `test_conventions.py`, `test_correction.py`, `test_validation.py`.
- `scripts/`: `check_env.py`, `validate.py`, `check_all.sh`.
- `requirements.txt`, `requirements-lock.txt`, `pyproject.toml`, `.github/workflows/ci.yml`.
- `README.md`, `docs/MATH.md`, `docs/CODE.md`, `docs/VALIDATION.md` (generated), `WORKLOG.md`.

**Commands and results.**
- `py install 3.11` → Python 3.11.9. `py -V:3.11 -m venv D:\mesquite_work\venv`.
- The first `pip install` failed on every download with `SSL: CERTIFICATE_VERIFY_FAILED` (pip 24.0 bundled with
  3.11.9 uses its own CA bundle, and this machine's HTTPS traffic presents a certificate only the Windows store
  trusts). `python -m pip install --use-feature=truststore --upgrade pip` → pip 26.2.2, which uses the system
  trust store by default; every later install worked.
- `pip install torch --index-url https://download.pytorch.org/whl/cpu` → torch 2.14.1+cpu; then the other
  packages from PyPI. Installed versions are pinned in `requirements.txt`; `pip freeze` is in
  `requirements-lock.txt`.
- `python scripts/check_env.py --fast` → all 15 packages import, versions equal the pins; no manifests yet; PASS.
- `bash scripts/check_all.sh` (on commit `8bf4044`) → ruff: all checks passed; pytest: **47 passed**;
  validation: **11 of 11 checks pass**, including the golden vector `aa550257ff3f01c0ff1fff5f3412cdab` both ways,
  heading(Y(α)·R) = wrap(heading(R) + α) to 1.78e-15 rad over 100,000 orientations, and the §4.4 example
  (δ = −7°, residual 4.8e-15° with the right sign, 14.000000° with the wrong sign).

**Data downloaded.** None.

**Decisions.**
- Package layout: one flat package `mesqsim/` at the repository root, so `python -m mesqsim …` (from M4) runs
  from a fresh clone without installation; pytest finds it through `pythonpath = ["."]`.
- `encode_frames` refuses a quaternion whose quantised `qw` is negative: the pod can never send one, so a sign
  bug upstream fails loudly instead of being silently encoded.
- Battery values are floored and then clamped to 0–100 (the firmware's rounding is not documented in the brief).
- `docs/VALIDATION.md` is generated entirely by `scripts/validate.py` from the check registry in
  `mesqsim/validation.py` (what, why, expected, command, result). Its commit stamp is `HEAD` at run time, marked
  `+dirty` if the tree had uncommitted changes. So each milestone commits code first, then runs the check, then
  commits the regenerated VALIDATION.md.
- `requirements.txt` carries `--extra-index-url https://download.pytorch.org/whl/cpu` so that Linux (CI)
  installs the CPU torch build; `torch==2.14.1` matches `2.14.1+cpu`.
- CI runs ruff, pytest and `validate.py --quick` (which does not rewrite the doc).

**Open questions.**
- Repository visibility (Danyal): it must be private before anything is pushed.
- Install `gh`, or open pull requests by hand on GitHub once pushing is allowed?

**Commits.** `13ea832` (main: brief and rules), `8bf4044` (m1-skeleton: code). The commit holding this entry is
listed at the start of the next entry. Nothing pushed (see above).

---

## 2026-10-22 · M2 · branch `m2-data` (stacked on `m1-skeleton`)

**Previous entry's commits.** `6872dc8` (M1 validation and work log), `ba0bea7` (fix: validation summary count, LF
endings), `3585d9c` (M1 regenerated VALIDATION.md), all on `m1-skeleton`. Nothing pushed (repository still public).

**Plan stated to the collaborator.** DIODEM download, verification and sanity figure; AMASS and SMPL+H instructions,
expected layout and verification script; AMASS loader and mapping with up-axis and left/right checks, tested on
synthetic AMASS-format files until the collaborator downloads AMASS.

**Files added.** `mesqsim/data/{__init__,net,diodem,amass}.py`, `mesqsim/synth.py`, `mesqsim/figures.py`,
`mesqsim/plotstyle.py`, `scripts/{download_diodem,verify_amass,make_figures,diodem_inspect}.py`,
`tests/{conftest,test_amass,test_diodem,test_repo_hygiene}.py`, `manifests/diodem.csv`, `docs/DATA.md`,
`docs/figures/{stick_figure_synthetic,diodem_relative_heading}.png`.
**Changed.** `.gitignore`, `requirements.txt`, `requirements-lock.txt`, `scripts/check_env.py`,
`scripts/validate.py`, `mesqsim/validation.py` (checks 10.1–10.5, 11.1–11.3), `README.md`, `docs/CODE.md`.

**Data downloaded.** DIODEM v3.0 from Harvard Dataverse (doi:10.7910/DVN/SGJLZA, CC0 1.0) with
`python scripts/download_diodem.py`. The API listed 485 files (2,738,355,798 bytes). Downloaded 267 files,
525,817,065 bytes: `dataset/` (264 files, 525,812,448 bytes) and three notes files from `make_dataset/`. Every file was
MD5-checked against Dataverse. Names, sizes and SHA-256 are in `manifests/diodem.csv`
(SHA-256 of the manifest: `414d4ddf7f96e2fc49188b3497c0199a85e35cc7be1088b7a1a4e6f7605c6814`). Stored in
`D:\mesquite_work\data\diodem`. AMASS and SMPL+H: not downloaded (collaborator to register).

**Commands and results.**
- First Python download attempts failed: `SSLCertVerificationError`, because the certificate chain is issued by
  "Avast Web/Mail Shield Root" (`openssl s_client -connect dataverse.harvard.edu:443`); then `403 Forbidden` from
  Dataverse for the default `python-requests` User-Agent. Fixed with `truststore` (OS trust store) and an explicit
  User-Agent (`mesqsim/data/net.py`).
- Official loader found: PyPI `imt-diodem` 1.1.11 (authors' package). Downloaded with `pip download --no-deps` and
  read in full before installing. It pulls in `qmt`, `dm-tree`, `imt-tree-utils`, `jax`/`jaxlib` and `aiohttp`; numpy,
  scipy and pandas stayed at their pins.
- `python scripts/diodem_inspect.py`:
  - exp01: gravity cosine between six-axis VQF and the optical frames is 1.000 for all five segments, with no
    mounting rotation. Gyroscope–optical rate correlation 0.976–0.994. Adjacent marker gaps 109–134 mm, others
    ≥ 372 mm. Hinge pairs: PC1 share 0.968, 0.997, 0.996.
  - exp06: same gravity agreement. Adjacent gaps 110–133 mm; the hinges seg5–seg1 and seg3–seg4 have PC1 0.998.
  - Gyroscope offsets against optical rates reach about 0.6 °/s (exp01 seg2: −0.54, −0.20, −0.64 °/s).
- Header scan over all 88 optical files: 30 Hz in exp01–03 and 120 Hz in exp04–11. IMU files are 40 Hz everywhere.
- `python scripts/make_figures.py` → `stick_figure_synthetic.png`; `stick_figure_amass.png` skipped (no AMASS);
  `diodem_relative_heading.png`. The figure's final VQF−optical difference is −80.3°, max 91.4°, over 6.13 min.
- `python scripts/verify_amass.py` → 0 archives, 0 motion files, 0 model files ("Nothing to index yet").
- `python scripts/check_env.py --require diodem` → all 17 packages import at their pins; diodem 267 of 267 files ok;
  PASS.
- `bash scripts/check_all.sh` (on commit `39b9cd7`) → ruff clean; pytest **76 passed, 8 skipped** (all skips: AMASS
  not downloaded); validation **15 passed, 0 failed, 4 skipped** (10.2–10.5 need AMASS).

**Decisions.**
- Use the authors' DIODEM loader (brief §5.2), but never let it download. Our script fetches from Dataverse with
  MD5 verification into the loader's cache layout, and the loader runs with `backend="dataverse"` on that copy.
- Download only `dataset/` and the authors' notes (526 MB). The videos, raw logs, CAD files and images (about
  2.2 GB) are not needed in Step 1; `--all` fetches them.
- The DIODEM sanity pair is arm exp01, hinge seg2 → seg3, rigid IMUs, all 13 motions concatenated (6.13 min).
  The rigid IMUs' axes coincide with the segment axes in the processed data (verified), so no mounting estimate is
  needed. The drift criterion of check 11.3: the end difference must exceed twice the 95th percentile of the first
  5 s, and the optical relative heading must span less than 180°.
- AMASS world taken as +Z up (`AMASS_WORLD_UP`), to be verified on the data by check 10.3 rather than assumed.
- Spine pod → SMPL spine3 (joint 9) by default, spine2 available through `spine_joint=6`.
- Stick figures use generic bone lengths (an assumption for drawing only), so they work without the SMPL+H body
  model. Body-model positions are not needed in Step 1 computations.
- Synthetic motion (`mesqsim/synth.py`) writes real AMASS-format `.npz` files in both SMPL+H and SMPL-X layouts, so
  the loader is tested end to end without licensed data.
- `.gitignore` rules for data are anchored (`/data/`, `/outputs/`). A bare `data/` hid `mesqsim/data/`; caught
  before the commit, and now guarded by `tests/test_repo_hygiene.py`.

**Open questions.**
- AMASS download (collaborator): which small subset for the M3 slice, and confirmation of CMU, KIT and BMLmovi
  for M5 after checking sizes on the download page (Danyal).
- Spine joint: spine3 (default) or spine2 (Danyal).
- The AMASS key table in DATA.md §1.3 is written from the brief and the loader's tests. It must be confirmed
  against the real files when they arrive.

**M2 "done when" status.** `check_env.py` passes. The DIODEM sanity figure is in `docs/figures/`. **Not yet met, blocked
on the AMASS download:** one AMASS sequence giving 15 orientations, the up-axis and left/right checks on real data,
and an AMASS stick figure. The code and tests for them exist and skip cleanly. Only the synthetic stick figure is in
`docs/figures/` so far.

---

## 2026-10-22 · M3 · branch `m3-slice` (stacked on `m2-data`)

**Previous entry's commits.** `39b9cd7` (M2 code), `783d489` (M2 inspection script, validation, work log), on
`m2-data`. Nothing pushed (repository still public).

**Plan stated to the collaborator.** A thin end-to-end slice on one motion sequence, synthetic until AMASS is
downloaded:

1. Give every pod only a constant heading offset.
2. Simulate chip samples and the 32 Hz send grid; write the frames table, labels and meta.
3. Decode with Step 0.
4. Score B0 and O2. Done when B0's error equals the offset differences up to quantisation and O2's error is 0.

**Files added.**
- `configs/{default,presets}.yaml`.
- `mesqsim/{config,recording,step0,evaluate,io,runs,pipeline,cli,__main__}.py`.
- `mesqsim/sim/{__init__,rng,motion,session,truth,heading,timing,radio,simulate}.py`.
- `tests/{test_sim_slice,test_step0}.py`, `docs/{PIPELINE,SIMULATOR}.md`.

**Changed.** `mesqsim/validation.py` (checks 5.1, 5.2, 7.3, 7.4, 8.1), `docs/{MATH,CODE}.md`, `README.md`.

**Commands and results.**
- First end-to-end run (scratch script; synthetic walk, 20 s, preset `offsets_only`, seed 1): 27,205 frames.
  - Step 0 rebuilt (pod, epoch, tick) for every frame; fresh flags and ms_lo equal the labels.
  - B0 orientation error minus |a_c − a_p| was at most 0.0081°.
  - O2 heading error 0; O2 orientation error at most 0.0085°.
- The one NaN was a grid step a few ms before a pod's first frame arrived. Fixed by starting the grid 0.25 s after
  the last pod's first send.
- `test_unrecorded_idle_gap_is_not_counted_as_loss` failed twice, and both failures were real:
  1. a sample taken just before motion start, inside the idle interval, had no defined orientation (fixed: the
     wearer holds the T-pose, then the first motion pose);
  2. 5 ticks next to the 600 s idle gap were marked lost: Step 0 classified missing ticks by an arrival time
     extrapolated over 600 s with up to ±50 ppm pod-clock error (fixed by the shared knowable-tick rule, below).
- `python -m mesqsim simulate --seed 1 --preset offsets_only` then `python -m mesqsim baselines --run …` →
  motion-phase summary: B0 heading and orientation mean 90.80°; O2 heading 0.0000°, orientation mean 0.0034°,
  P95 0.0058°.
- `bash scripts/check_all.sh` on commit `a90ab07` → ruff clean; pytest **95 passed, 8 skipped** (AMASS);
  validation **20 passed, 0 failed, 4 skipped**. Of note:
  - 5.1: B0 orientation P99 0.0082°, max 0.0098°;
  - 5.2: max deviation 0.0081° with offset differences up to 179.5°;
  - 7.3: 113,605 frames over 3.63 ms_lo wraps, 0 mismatches;
  - 7.4: 27,205 ticks, 0 mismatches;
  - 8.1: identical bytes for the same seed.

**Decisions.**
- Labels are two files, `labels_ticks.parquet` (per pod and send tick) and `labels_joints.parquet` (per joint
  and 32 Hz step), instead of the brief's single `labels.parquet`, because the two have different rows.
  `labels_ticks` also carries the true pod orientation at each sample time, which the secondary metric needs.
- **Knowable ticks** (`mesqsim/recording.py`): a missing tick is lost only when received frames of the same
  pod bracket it in one recorded interval and the later one is in its own boot epoch. Both the labels and Step 0
  apply the same rule, which makes the flag comparison exact. Edge ticks are unknowable to any logger.
- The idle interval is not recorded (`session.json` marks it `recorded: false`), so a 30-minute idle costs no
  frames. Step 0 bridges it with host time: up to 4 + 200 ppm of the expected sends before declaring a reboot.
- Secondary metric: each pod's latest valid frame by host time, compared with the truth at that frame's own sample
  time, using the true mountings (brief §4.7). This isolates heading error from timing error.
- One random stream per subsystem, seeded by (seed, name): adding a term later does not reshuffle the others.
- Run folder `run_g<commit7>_c<config hash8>_s<seed>`. Untracked files also mark the commit dirty, because the
  first trial run used new untracked modules but was named as clean HEAD; caught and fixed.
- Default `error_model.offset_mode` is `arbitrary` (the larger problem); `aligned` is in the sweep. Which mode the
  real chips follow is unmeasured.
- T-pose posture error "5° per segment" is read as an RMS rotation angle of 5° (per-axis σ = 5°/√3).
- Nominal pod placements (forehead, chest front, back of pelvis, outer upper arm, …) are an assumption, recorded
  in `mesqsim/sim/motion.py`.

**Open questions.**
- Does the chip's north start at its own heading (`aligned`) or anywhere (`arbitrary`)? The 70-minute box
  evening will tell (Danyal).
- AMASS still not downloaded; the slice runs on synthetic motion.

**Not done in M3, by design.** Drift, scale factor, random walk, tilt error, loss, stalls, reboots, box
registration, B1, O1, sweeps and dataset generation are M4 and M5.

---

## 2026-10-23 · M4 · branch `m4-simulator` (stacked on `m3-slice`)

**Previous entry's commits.** `a90ab07` (M3 code), `d4c330f` (M3 validation and work log), on `m3-slice`. Nothing
pushed (repository still public).

**Plan stated to the collaborator.**
- The full heading model ψ = a + b·t + k·Θ + w with reboots, plus the fusion error η.
- Optional sample-task stalls and Gilbert–Elliott burst loss.
- Box registration and the T-pose snapshot in Step 0; baselines B1 and O1.
- Every remaining §9 check.

**Files added.** `tests/test_sim_full.py`.
**Changed.** `mesqsim/sim/{heading,timing,radio,session,simulate,truth}.py`, `mesqsim/{step0,evaluate,validation,cli}.py`,
`configs/{default,presets}.yaml`, `tests/test_validation.py` (readable ids `check_<n>_<m>`),
`docs/{MATH,SIMULATOR,PIPELINE,CODE}.md`, `README.md`.

**Commands and results.**
- Trial runs (scratch script; synthetic walk 30 s, seed 3, 10-minute idle; mean heading error over the motion phase):

  | setting | B0 | B1 | O1 | O2 |
  |---|---|---|---|---|
  | default | 99.7° | 7.3° | 6.6° | 0 |
  | stress | 100.2° | 13.2° | 12.3° | 0 |
  | aligned | 6.9° | 6.7° | 6.6° | 0 |

  Step 0 flags matched the labels exactly in every case (752–1,919 lost ticks, 765 held frames under stress).
- Seeds 3 and 4 drew no reboots under stress. Over 300 seeds the mean is 2.63 reboots per session against 2.74
  expected (the spacing rule removes a few), so they were unlucky draws. Seed 5 (3 reboots) is used by check 7.4:
  Step 0 found the same epochs, 0 flag mismatches. After a reboot, B1's error rises to 30° (stale box
  registration) while O1 follows the new north.
- Check 6.1 in quick mode first showed σ = 0.321° against 0.367°. Investigated with a 2,250-pod run
  (σ = 0.361°): the first 300 pods (seeds 0–19) were a low block (0.321°), and the other blocks of 300 gave
  0.355–0.393°. Quick mode now uses 600 pods; the full run uses 4,500 after one 2,250-pod draw put the pair
  statistic at −4.0 % against the 5 % tolerance.
- Check 6.3 showed Θ = 359.436° for a 360° turn: with no idle, the T-pose → motion pose change fell between two
  240 Hz grid points and Θ at motion start was interpolated halfway through it. Phase boundaries are now exact grid
  points: 360.000°, ψ change 5.4000°.
- `bash scripts/check_all.sh` on commit `e644460` → ruff clean; pytest **114 passed, 8 skipped** (AMASS);
  validation **30 passed, 0 failed, 4 skipped** (10.2–10.5 need AMASS). Every number is in docs/VALIDATION.md.
  Highlights:
  - 4.2: a common 73° turn of every north moves the labels by 1.5e-13° and B1 by 0.0038°;
  - 4.3: one pod's 23° turn moves l_hip by +23.000000° and l_knee by −23.000000°, nothing else, and leaves the other
    pods' frames identical;
  - 5.3: B1 0.0034°, O1 0 (offsets only);
  - 6.1: σ 0.3619° (expected 0.3674°), pair 0.5073° (0.5196°), 4,500 pods;
  - 6.2: drift exact to 2.5e-14°;
  - 6.3: 5.4000°;
  - 7.1: discards 0.3620, 0.4188, 0.4673 (expected 0.3613, 0.4182, 0.4667);
  - 7.2: held rate 0.02607 with stalls (expected 0.02613), 0 without;
  - 7.4: 0 mismatches with 3 reboots under stress;
  - 7.5: loss model matches at all three corners;
  - 12.1: O2 0 ≤ O1 6.485° ≤ B1 6.898° ≪ B0 93.879°.

**Decisions.**
- Θ_i is integrated from the true trajectory, as the z component of Log(P_{k+1}·P_k⁻¹), on a 240 Hz grid with the
  phase boundaries as exact points. The unrecorded idle is one step: its net twist about the vertical plus whole
  turns ~ round(N(0, (0.5·√min)²)), an assumed turning model (brief §4.3: "an assumed amount of turning").
- Reboots are a Poisson process per pod, at least silence + 2 s apart. A reboot restarts count, pod clock, drift
  origin and Θ origin, and draws a new north. In the aligned mode the new north is the pod's own heading at its
  chip's start-up. The random walk continues.
- Stalls: at the end of a stall only the newest waiting sample is decoded, and the decode task then needs 1 ms
  (`decode_busy_ms`, assumption). This guarantees distinct samples have distinct `ms_lo`, so "repeated `ms_lo`"
  means held, as the brief states.
- Stall starts are jittered ±0.5 s, so their phase on the 32 Hz grid varies (3 s is exactly 96 sends: without
  jitter every stall would hit the same grid phase).
- B1 = box registration only. The T-pose snapshot is implemented and tested (Δ_i equals the true world rotation to
  0.02° with no errors), but it removes mountings, which the secondary metric already takes as known.
- The fusion error η is Gauss–Markov per pod axis, σ 0.5° and τ 1 s (τ is an assumption).
- Default drift σ_b 0.2 °/min, loss 2 % in bursts of 3, no stalls, no reboots: the middle of the brief's sweeps.
  Stress settings are in the `stress` preset.
- Datasheet source notes cite only what the brief states ("Tables 1 and 5"); I have not read the datasheet itself.

**Not modelled** (in docs/SIMULATOR.md): the chip rate's ±1 % drift over temperature, per-axis scale factors,
cross-axis sensitivity, and radio loss correlated across pods.

**Open questions.** The same as M3 (offset mode; AMASS download), and every `assumption` row awaits Danyal's bench
measurements: `python -m mesqsim params --label assumption` lists all 34.

## 2026-10-23 · M5 · branch `m5-results` (stacked on `m4-simulator`); entry written on 2026-11-13 in S2-M0

*The M5 session ended without a work-log entry. This entry was written afterwards from the M5 commits, their
messages and diffs, and a fresh run of the full check on the same code. Numbers quoted from a commit message are
marked as such; they were not re-run.*

**Previous entry's commits.** `e4b4667`, `67c5ac4`, `e644460` (M4 code), `3a88b80` (M4 validation and work log),
`f826716` (work-log fix), on `m4-simulator`.

**M5 commits.** `5f494f4` (speed-up), `a53c510` (M5), `441be49` (label epochs), on `m5-results`. GitHub's
`m5-results` holds `a53c510`; `441be49` was local only. The regenerated `docs/RESULTS_STEP1.md` and its two
figures were left staged but uncommitted; they are committed with this entry.

**Files added.** `mesqsim/{splits,dataset,results}.py`, `scripts/make_results.py`, `tests/test_splits.py`,
`docs/RESULTS_STEP1.md`, `docs/figures/{results_growth,results_sweeps,sim_random_walk}.png`.
**Changed.** `mesqsim/{cli,figures,paths,pipeline,step0,validation}.py`, `mesqsim/sim/{motion,simulate}.py`,
`configs/default.yaml` (`dataset:` section), `scripts/{make_figures,validate}.py`, `tests/test_sim_full.py`,
`docs/{CODE,DATA,MATH,PIPELINE,SIMULATOR}.md`, `README.md`.

**What was built.**
- Splits ([mesqsim/splits.py](mesqsim/splits.py)): subjects → 70/15/15 % per subset by a stable SHA-256 order and
  largest remainder; TotalCapture to `holdout`; `python -m mesqsim split --write` writes the manifest and its
  SHA-256 and refuses to replace a frozen one without `--force`. Checks 9.1–9.3.
- Dataset generation ([mesqsim/dataset.py](mesqsim/dataset.py), `python -m mesqsim dataset`): windows of 20 s
  (at most 3 per sequence) × motion start 1, 10, 30 min since boot × variants, stable per-session seeds, every
  session in its sequence's split. Synthetic stand-in sources until AMASS.
- Results ([mesqsim/results.py](mesqsim/results.py), [scripts/make_results.py](scripts/make_results.py)): the
  default dataset plus one-at-a-time sweeps with common random numbers → Tables 1–4 and two figures in
  `docs/RESULTS_STEP1.md`, labelled provisional while the motion is synthetic.
- Speed-up (`5f494f4`, from its message): Step 0's unwrapping and lost-tick insertion are vectorised per boot epoch
  (0.55 s → 0.07 s per session); η is generated on each epoch's chip grid with one linear filter. One session
  (30-minute idle, 20 s motion) takes 0.54 s to simulate, decode and score, down from about 1.5 s.
- Label epochs (`441be49`, from its message): a pod that rebooted twice inside the unrecorded idle showed Step 0 one
  count jump, while the labels counted two, so the reboot sweep crashed. Labels' `epoch` is now the observed boot
  epoch and `boot_epoch` the true index. Regression test
  `test_two_reboots_in_the_unrecorded_idle_look_like_one`; 30 stress sessions with 834 reboots gave 0 unmatched
  ticks and 0 flag mismatches.

**Commands and results.**
- `python scripts/make_results.py` (`--source synthetic --seed 1`) at `441be49`, 2026-10-23 07:52 UTC (the run line
  of `docs/RESULTS_STEP1.md`; not re-run in this session): 72 sessions from 12 synthetic sequences; primary error at
  10 min since boot, mean / P95: B0 91.96° / 173.58°, B1 6.70° / 17.16°, O1 6.25° / 16.07°, O2 0 / 0. Provisional.
- `bash scripts/check_all.sh` on 2026-11-14 01:52 UTC (2026-11-13 18:52 local), commit `af0b0cd` + the staged
  results files (code identical to `441be49`), 5 min 19 s:
  - ruff: all checks passed;
  - pytest: **125 passed, 9 skipped** (185 s; the 9 skips need AMASS);
  - validation: **32 passed, 0 failed, 5 skipped, of 37 checks** (9.2 and 10.2–10.5 need AMASS). The committed
    `docs/VALIDATION.md` still showed the M4 run (30 of 34), so M5 had not re-run the full check after adding 9.1–9.3.
  - Changes against M4: 4.2 B1 max change 0.0039° (was 0.0038°) and 12.1 B1 6.900° (was 6.898°), from the new η
    generation in `5f494f4`.

**Decisions** (read from the M5 code and docs).
- The split order is SHA-256 of `mesquite-step1-splits-v1:<subset>:<subject>`, so shuffling the file list or adding
  a subset moves no existing subject (tested).
- Sweeps vary one parameter at a time at 10 min since boot, with the same session seeds for every value.
- `dataset.window_s` 20 s, `windows_per_sequence` 3 and `motion_start_min` [1, 10, 30] are `assumption`
  (experiment design).

**Open questions** (at the end of M5). AMASS and SMPL+H not downloaded, so no split manifest and provisional results;
M6 not started; no `main` on GitHub and no pull request.

---

## 2026-11-13 to 2026-11-15 · S2-M0 (close Step 1) · branch `s2-m0` (from `m5-results`)

**Previous entry's commits.** The M5 entry above was written in this milestone, in `e56954d`.
**This milestone's commits** (local only; nothing pushed): `af0b0cd` (Step 2 brief, CLAUDE.md), `e56954d` (M5 entry,
regenerated synthetic results), `7fb2ae2` (AMASS manifests, split, check 9.2 count), `b44a8af` (MUST_KNOW, analysis
plan, README), `2432062` (make_results memory), `5b1b83e` (SMPL+H manifest, check_env), `fcb637c` (make_results counts,
unit, `--runs-commit`), `1b400e3` (sweep values), `c83d768` (RESULTS_STEP1 on AMASS), and the commit of this entry.

**Plan stated to the collaborator.** Save the Step 2 brief and update CLAUDE.md; write the missing M5 entry; download
and verify AMASS and SMPL+H with the collaborator's own accounts; run the AMASS checks, write the split, regenerate
the Step 1 results on AMASS; write MUST_KNOW, the analysis-plan draft, the README quickstart and the Step 1 report;
ask before creating `main` on GitHub and tagging `step1`.

**Data downloaded** (by the session, with the collaborator's own AMASS and MANO accounts, which the collaborator gave
for this purpose; the password was never written to any file, and the collaborator was asked to change it).

| file | bytes | SHA-256 |
|---|---|---|
| `amass/archives/HumanEva.tar.bz2` (SMPL+H G) | 33,071,034 | `9e1d6fb2bda95360d0b3f8175bbfe7ca69620db6ba637505e903c4e44667121a` |
| `amass/archives/BMLmovi.tar.bz2` | 748,793,904 | `15a771effd6694383aa8f26b03d37a5b9a8a9ee0bbab702172db8cc5b3b98cf4` |
| `amass/archives/CMU.tar.bz2` | 1,993,171,135 | `c360b8296a1a11bbd35d709815e1fd704162259c50ab3198c381a9e8c499003a` |
| `amass/archives/KIT.tar.bz2` | 2,196,810,275 | `2f259ea449392d4e3f7003a0912f80daef1d9f63bcf43878fbc89d9f852c091d` |
| `body_models/archives/smplh.tar.xz` (extended SMPL+H) | 113,231,292 | `46d5b8687be48c91181fa88271feff3a5e83aa62a481fad8a0bcb9254b2a74f1` |

- Sizes were read from the server (`Content-Range`) before downloading, and the files match them. Not downloaded:
  TotalCapture (83,967,231 bytes), SFU (62,318,066 bytes).
- The MPI download server needs a session cookie, and curl's resume (`-C`) combined with the login POST fails with
  curl error 2. A byte-range request, appended only when the server answers 206 with a `Content-Range` starting at the
  current size, worked. BMLmovi arrived in two pieces (601,750,994 + 147,042,910 bytes) after a connection reset; the
  full bzip2 decompression during extraction passed.

**Files added.** `docs/brief/STEP2_BRIEF.md`, `docs/MUST_KNOW.md`, `docs/analysis_plan_DRAFT.md`,
`manifests/{amass_archives,amass_files,smplh,split_manifest}.csv`, `manifests/split_manifest.sha256`.
**Changed.** `CLAUDE.md`, `README.md`, `WORKLOG.md`, `docs/{DATA,RESULTS_STEP1,VALIDATION}.md`,
`docs/figures/results_{growth,sweeps}.png`, `mesqsim/validation.py`, `scripts/{check_env,make_results}.py`.

**Commands and results.**
- `python scripts/verify_amass.py`: 4 archives, 4.97 GB; 8,143 motion files, 11.71 GB (BMLmovi 1,801, CMU 2,082,
  HumanEva 28, KIT 4,232); frame rates 60 Hz (495, all CMU), 100 Hz (4,232, KIT), 120 Hz (3,416); **0 load
  failures**. With `--no-extract` after the SMPL+H download: 3 body-model files, 0.26 GB; the AMASS manifests came out
  byte-identical.
- `python -m mesqsim split --write`: 246 subjects, train / val / test 173 / 37 / 36 (BMLmovi 60/13/13, CMU 72/15/15,
  HumanEva 2/1/0, KIT 39/8/8); sequences 5,974 / 1,104 / 1,065; SHA-256
  `d36448adb76762251668b9940933e8a14df664ad6d3277fe3a930f7d55049fb3` (also `sha256sum`).
- Scratch count (`count_windows.py`, not committed: `trans` frames / `mocap_framerate` per file): 23.2 h of motion;
  **767 of 8,143 sequences are at least 20 s long**; 20 s windows (at most 3 per sequence) train / val / test
  884 / 110 / 173; BMLmovi 2, KIT 397, CMU 756, HumanEva 12; median lengths BMLmovi 4.6 s, KIT 6.1 s, CMU 9.6 s,
  HumanEva 17.5 s.
- `python scripts/check_env.py --require amass_archives amass_files smplh diodem`: PASS (4, 8,143, 3 and 267 files ok).
- `bash scripts/check_all.sh`:
  - before AMASS (`af0b0cd`, 2026-11-13): ruff clean; pytest 125 passed, 9 skipped; validation 32 passed, 0 failed,
    5 skipped of 37;
  - **final (`c83d768`, 2026-11-15 08:12 UTC, 3 min 1 s): ruff clean; pytest 134 passed; validation 37 passed,
    0 failed, 0 skipped of 37.** The AMASS checks: 9.2 hash unchanged, no subject in two splits; 10.2 (473, 15, 4)
    unit orientations; 10.3 +Z up, mean cosine 0.962; 10.4 thighs alternate, correlation −0.94; 10.5 knee flexion
    within [−8.9°, 156.1°].
- `python scripts/make_results.py --seed 1` on AMASS:
  - at `b44a8af`: the default dataset (3,501 sessions) took 4.3 h; after 15 of 22 sweep values the process ran out of
    memory (`ArrowMemoryError: malloc of size 894169408 failed`) after 1,036 min, about 10 h of which the machine was
    asleep. Cause: every sweep's full error table (27.5 M rows each; 125.7 M for the default) was kept until the end.
  - at `2432062` (the fix): completed in 725 min, again with idle gaps.
  - `--runs-commit 2432062` at `fcb637c` and `1b400e3`: the report rebuilt from those runs in 7 min 46 s. Table 1 is
    identical to the default tables the `b44a8af` run had written.

**Decisions.**
- Subsets: the DATA.md proposal (CMU, KIT, BMLmovi) plus HumanEva as the small walking subset.
- The split was written only after all four subsets were in (adding a subset changes the hash). Writing it makes the
  tool treat it as frozen; **Danyal's approval is still pending.**
- A subject is (subset, subject id): CMU and KIT reuse the ids 10, 11, 12 and 63 for different people. Check 9.2's
  displayed count used the id alone (172 train subjects instead of 173) and is fixed; its leakage test
  (`splits.check_manifest`) already grouped by (subset, subject). The analysis plan states the rule for Step 2.
- `check_env.py` skips manifests without a `sha256` column (it reported the split manifest's 8,143 rows as missing).
- `make_results.py`: summarises each sweep as soon as it is scored; counts sequences as (subset, subject, file) and
  subjects as (subset, id) (it had counted 721 sequences by file name, which KIT reuses); on AMASS names the subject as
  the unit; an AMASS banner counts the assumptions from the config; swept values print as configured (0.005 had shown
  as 0.01); `--runs-commit C` reuses runs only when `mesqsim/` and `configs/` are unchanged since C, and never
  simulates under C's name (tested: it refuses `441be49`).
- `git_commit()` is read again for every sweep, so during the long run the session's own edits were stashed to keep
  the run folders labelled with a clean commit.
- **No data on GitHub.** The collaborator asked whether data should be separated, chunked and pushed. Original data
  stay in `D:\mesquite_work\data`, generated runs in `D:\mesquite_work\outputs` (about 100 GB per full results run),
  both outside the repository. They are not pushed: the AMASS and SMPL+H licences forbid redistribution, the
  AMASS-derived sessions fall under them, the repository is public, and both briefs forbid data in git. The manifests
  and the (commit, config, seed) run identity make everything reproducible for anyone who registers.
- The analysis-plan draft marks every choice not in a brief as a proposal.

**Open questions.** In the report below (items 5 and 6).

### Step 1 report (Step 1 brief §11)

**1. What was built** (89 files in git; data and outputs outside it).
- `mesqsim/`: `body`, `cli`, `codec`, `config`, `conventions`, `correction`, `dataset`, `evaluate`, `figures`, `io`,
  `manifest`, `paths`, `pipeline`, `plotstyle`, `recording`, `results`, `runs`, `splits`, `step0`, `synth`,
  `validation`; `data/{amass,diodem,net}`; `sim/{heading,motion,radio,rng,session,simulate,timing,truth}`.
- `scripts/`: `check_all.sh`, `check_env.py`, `diodem_inspect.py`, `download_diodem.py`, `make_figures.py`,
  `make_results.py`, `validate.py`, `verify_amass.py`.
- `tests/`: `test_{amass,codec,conventions,correction,diodem,repo_hygiene,sim_full,sim_slice,splits,step0,validation}`
  and `conftest.py`.
- `configs/{default,presets}.yaml`; `manifests/{amass_archives,amass_files,diodem,smplh,split_manifest}.csv`,
  `split_manifest.sha256`; `.github/workflows/ci.yml`.
- `docs/`: `CODE`, `DATA`, `MATH`, `MUST_KNOW`, `PIPELINE`, `RESULTS_STEP1`, `SIMULATOR`, `VALIDATION`,
  `analysis_plan_DRAFT`; `brief/STEP{1,2}_BRIEF`; five figures. Root: `README.md`, `CLAUDE.md`, `WORKLOG.md`,
  `pyproject.toml`, `requirements.txt`, `requirements-lock.txt`.

**2. `bash scripts/check_all.sh`** at `c83d768`: `ruff: All checks passed!`; `pytest: 134 passed in 78.43s`;
`VALIDATION: PASS` (37 passed, 0 failed, 0 skipped of 37); `== ALL CHECKS PASSED`.

**3. The results table** ([docs/RESULTS_STEP1.md](docs/RESULTS_STEP1.md), Table 1). AMASS, default parameters
(offsets arbitrary, box slot 2°, drift σ 0.2 °/min, scale factor ±1.5 %, no reboots, loss 2 % in bursts of 3,
20 s windows), seed 1, config `b083b87e8724`; 3,501 sessions from 767 sequences of 81 subjects; 748,047 steps per
row. Primary error |wrap(δ̂ − δ)|, mean / P95 in degrees:

| baseline | 1 min since boot | 10 min | 30 min |
|---|---|---|---|
| B0 | 89.83 / 170.94 | 90.21 / 171.06 | 90.74 / 171.70 |
| B1 | 4.03 / 10.16 | 6.99 / 18.30 | 12.06 / 31.49 |
| O1 | 3.26 / 8.46 | 6.55 / 17.63 | 11.81 / 30.86 |
| O2 | 0 / 0 | 0 / 0 | 0 / 0 |

With one reboot per pod per hour, B1's mean at 10 min is 29.33° (P95 146.72°) against O1's 6.06° (Table 4).

**4. Every assumption still in the config** (38 of 44; `python -m mesqsim params --label assumption`):
`motion.spine_joint`; `session.{boot_spread_s, box_s, strapping_s, strap_move_s, strap_spins_max, tpose_s, idle_s,
tpose_posture_deg, facing_error_deg, mounting_max_deg}`; `timing.{chip_start_s, decode_delay_ms, grid_start_ms,
stall_ms, stall_period_s, stall_jitter_s, decode_busy_ms, pod_clock_ppm, host_epoch_s}`; `radio.{loss_mean,
burst_mean_frames, latency_min_ms, latency_jitter_ms}`; `error_model.{offset_mode, box_slot_deg,
drift_sigma_deg_per_min, tilt_sigma_deg, tilt_tau_s, reboot_rate_per_h, reboot_silence_s,
idle_turns_sigma_per_sqrt_min}`; `battery.{start_pct, drain_pct_per_h}`; `dataset.{window_s, windows_per_sequence,
motion_start_min, variants}`. The other six: 3 datasheet, 3 firmware, 0 measured.

**5. What is blocked, and on whom.**
- Pushing `s2-m0`, creating `main` on GitHub and the tag `step1`: **the collaborator's approval** (asked). GitHub's
  default branch is `m5-results`; making `main` the default is Danyal's (owner) setting.
- The split freeze and the analysis-plan freeze: **Danyal**.
- Replacing assumptions with measurements: **Danyal's bench** (the 70-minute box evening, the taped-board session).

**6. What Danyal must decide.**
- Freeze the split (`d36448ad…`) and the analysis plan ([docs/analysis_plan_DRAFT.md](docs/analysis_plan_DRAFT.md),
  §12 checklist).
- **The window length:** at 20 s only 81 of 246 subjects and 767 of 8,143 sequences take part, the test set has 15
  subjects, and BMLmovi's everyday actions are almost absent; this bears on H2. Proposal: 8 s windows, or one
  subject's sequences concatenated.
- The AMASS subsets, and whether to add TotalCapture as a holdout.
- The Spine joint: SMPL spine3 (default) or spine2.
- Before the first push of Step 2 estimator code: public, or kept local.

**7. Plan for Step 2** (specified in [docs/brief/STEP2_BRIEF.md](docs/brief/STEP2_BRIEF.md) §7; not started).
S2-M1 the estimator contract, the common clock, pairing and rotation rates, the `geometry:` config; S2-M2 the hinge
estimator (GEO-K, then axis identification and GEO-ID); S2-M3 the 2-DoF constraint and the range-of-motion cost with
Ω fitted on train; S2-M4 windows, the von Mises filter, reboot resets, the hold rule and κ recalibration; S2-M5
DIODEM; S2-M6 `make_results_step2.py` and `docs/RESULTS_STEP2.md`; S2-M7 the explanation pipeline; S2-M8 the
documentation, report and tag `step2`. The window-length decision should come before S2-M6's results are read; the
AMASS-based fits (Ω, κ scale, q, hold threshold) use the frozen split.

## 2026-11-15: S2-M0 pushed

**What happened.** The collaborator approved pushing the code without the data ("push the code without uploading the
data"). Before pushing I checked the history of `s2-m0`: no data file (only the `manifests/*.csv` of names, sizes and
SHA-256s), no blob over 5 MB (largest: `manifests/amass_files.csv`, 942,523 bytes), and no password or cookie.
`git log -p s2-m0 | grep` hits only the commit author line. `data/` and `outputs/` are git-ignored, and the data live
in `D:\mesquite_work`, outside the repository.

- **Danyal's commit on GitHub.** GitHub's `m5-results` had Danyal's 4ffa9bc (2026-11-12, README: remove the private
  notice), which was not in `s2-m0`. Our README already removes the notice, so I merged it with `git merge -s ours`
  (the files are unchanged against 746733c; `git diff --stat` is empty). `s2-m0` now contains everything on GitHub.
- **Pushed.** `main` (13ea832, the first commit) as GitHub's `main`, and `s2-m0`. `gh` is not installed, so the
  collaborator opens the pull request `s2-m0` → `main` by hand. The merge and the tag `step1` wait for Danyal's
  approval, and making `main` GitHub's default branch is Danyal's setting.
- **No Step 2 estimator code in this push.** S2-M0 only closes Step 1. The question in CLAUDE.md rule 2 (public, or
  kept local?) will be asked before the first push of S2-M1.
- **The check.** The last full check ran at c83d768: ruff clean, pytest 134 passed, validation 37 passed, 0 failed,
  0 skipped. Every later commit changes only README.md, WORKLOG.md and the header line of docs/VALIDATION.md
  (`git diff --stat c83d768 HEAD`).

## 2026-11-16 · Step 2 starts on Danyal's Mac: environment, data, `main` and `step1`

**Previous entry's commits.** `b143815` (the "S2-M0 pushed" entry), merged by Danyal into `m5-results` through pull
request #1 as `9b06673` (2026-11-16).

**Machine.** macOS (Darwin 25.6.0), Apple M2 Pro (10 cores), 16 GB RAM. Python 3.11.16 in `.venv`, made with
`uv venv --python 3.11` and filled with `uv pip install --index-strategy unsafe-best-match -r requirements.txt`
(under uv's default first-index strategy the CPU-torch extra index would shadow PyPI for every package);
`python scripts/check_env.py --fast`: every pin installed, PASS. Data and outputs live next to the repository, outside
git: `../mesquite_work/data` and `../mesquite_work/outputs` (`MESQ_DATA_DIR`, `MESQ_OUTPUT_DIR`), like the
collaborator's `D:\mesquite_work`. The Desktop is not synced to iCloud (`defaults read com.apple.finder
FXICloudDriveDesktop` = 0).

**Decisions asked and answered (Danyal, who ran this session).**
- Step 2 brief rule 3, asked once: *"Should the Step 2 estimator code be pushed to the public repository, or kept
  local?"* → **push to the public repository.**
- `main` and tags: "push and merge all the code on the main branch, and separate Step 1 and Step 2 with tags".
  GitHub's `main` still held only `13ea832` (pull request #1 had been merged into `m5-results`). `main` was
  fast-forwarded to `9b06673` (no force) and the annotated tag **`step1`** pushed there. Each Step 2 milestone: a
  branch from `main`, the full check, a pull request, the merge into `main` (approved in advance by Danyal), and a
  tag `s2-m<n>`; `step2` at the end.
- AMASS on this Mac: Danyal gave his own AMASS / MANO account for the download. The password went only into the
  download requests and was never written to any file; it is in the session transcript, so **Danyal should change
  it**.

**Data downloaded** (sizes read from the server's `Content-Range` before downloading; every SHA-256 equal to the
committed manifests).

| file | bytes | SHA-256 |
|---|---|---|
| `amass/archives/HumanEva.tar.bz2` | 33,071,034 | `9e1d6fb2bda95360d0b3f8175bbfe7ca69620db6ba637505e903c4e44667121a` |
| `amass/archives/BMLmovi.tar.bz2` | 748,793,904 | `15a771effd6694383aa8f26b03d37a5b9a8a9ee0bbab702172db8cc5b3b98cf4` |
| `amass/archives/CMU.tar.bz2` | 1,993,171,135 | `c360b8296a1a11bbd35d709815e1fd704162259c50ab3198c381a9e8c499003a` |
| `amass/archives/KIT.tar.bz2` | 2,196,810,275 | `2f259ea449392d4e3f7003a0912f80daef1d9f63bcf43878fbc89d9f852c091d` |
| `body_models/archives/smplh.tar.xz` | 113,231,292 | `46d5b8687be48c91181fa88271feff3a5e83aa62a481fad8a0bcb9254b2a74f1` |

- AMASS from the MPI download server (domain `amass`, `amass_per_dataset/smplh/gender_specific/mosh_results/
  <Subset>.tar.bz2`); SMPL+H from domain `mano`, `smplh.tar.xz`, extracted to `body_models/smplh/`. KIT stopped at
  1,152,385,024 bytes when a Claude session ended and was resumed with an HTTP range request (appended only on a `206`
  whose `Content-Range` started at the partial size).
- DIODEM: `python scripts/download_diodem.py`, 267 files, each MD5-checked; `manifests/diodem.csv` unchanged.
- Requests to outside servers used a generic User-Agent (`Mozilla/5.0 (compatible; data-download)`) instead of the
  repository's `mesqsim-download/0.1 (Mesquite ML research; ...)`, so no project name was sent. **Open question:**
  make the repository's default generic too.
- `python scripts/verify_amass.py` (in a clean worktree), 4 min 52 s: 8,143 motion files, 11.71 GB (BMLmovi 1,801,
  CMU 2,082, HumanEva 28, KIT 4,232); frame rates 60 Hz 495, 100 Hz 4,232, 120 Hz 3,416; **0 load failures**; SMPL+H 3
  files, 0.26 GB. `manifests/amass_archives.csv`, `amass_files.csv` and `smplh.csv` came out **byte-identical** to the
  collaborator's (`git status` clean afterwards).

**Baseline before any Step 2 code** (main at `9b06673`, AMASS not yet extracted): `ruff` clean; `pytest` 126 passed,
8 skipped (AMASS) in 25 s; `python scripts/validate.py --quick` PASS. DIODEM check 11.3 gives the collaborator's
numbers on this machine (end −80.3°, max 91.4°).

---

## 2026-11-16 · S2-M1 (estimator contract, common clock, scorer) · branch `s2-m1` (from `main` = `9b06673`)

**Plan stated.** The estimator contract and its registration with the scorer; one clock for parent and child (a
causal lower-envelope pod-to-host map), pairs and rotation rates; the `geometry:` config section; the causality,
label-deletion and rate-invariance tests; the timing check.

**Files added.** `mesqsim/estimators/{__init__,contract,clock,vmfilter,geo}.py`, `mesqsim/geo_score.py`,
`tests/test_{conventions_step2,clock,geo_contract}.py`.
**Changed.** `mesqsim/{conventions,config,evaluate,pipeline,cli,validation}.py`, `configs/default.yaml`,
`docs/{CODE,PIPELINE,MATH,VALIDATION}.md`, `README.md`, `CLAUDE.md`.

**What was built.**
- `contract`: `Estimate(delta_rad, kappa, uncertain)`; `GeoTruth`, the one argument through which GEO-K receives
  simulator truth (the true mountings; the scorer builds it from `meta.json`).
- Estimators receive `cfg.section("geometry")` only: reading `error_model.*` from inside raises `KeyError` (tested).
- `clock`: fresh valid samples; per pod and boot epoch the lower envelope t_host ≈ α + β·t_pod (minimum per 2 s block,
  a line through the minima); pairs by slerp within 0.1 s; body rates (a constant heading offset cancels, tested).
- `vmfilter`: init, predict (A(κ′) = A(κ)·e^{−qΔt/2}), product update, reset; the state after every event is
  recorded, so outputs at query times are causal by construction.
- `geo`: events — box-registration start at the box end with κ₀ = 1/(2σ_box²), a reset when a pod's new boot epoch
  appears, a window every 1 s over the last 8 s (none inside unrecorded phases). No joint solvers yet, so GEO-K and
  GEO-ID equal B1 with a decaying κ (smoke test: identical to B1 over the motion; κ from 410 to about 330).
- `evaluate`: methods B0, B1, O1, O2, GEO-K, GEO-ID, GEO-ID-nofilter, GEO-ID-hinge, GEO-ID-w4, GEO-ID-w16;
  `USES_TRUTH` = {O1, O2, GEO-K}; `errors.parquet` gains `kappa` (float32, NaN for the baselines) and `uncertain`
  (nullable boolean). `python -m mesqsim baselines --run DIR --methods ...`.
- `conventions`: Cardan angles (pinned against scipy); von Mises A(κ), A⁻¹, product, prediction, interval
  half-widths, log density.
- `config`: the label `fitted`; the `geometry:` section (11 parameters, all `assumption` with sources); the drift
  stress value 2.0 °/min in the sweep (Step 2 brief §4.1).

**Commands and results.**
- The first run of check 31.1 failed: max 41.25 ms. Three seconds after a 10-minute idle the clock slope had been
  fitted through two block minima, +14,888 ppm for one pod. Fix: a slope only through at least 5 block minima
  (10 s), clipped to ±200 ppm (Step 0's clock tolerance). Then max 12.73 ms.
- `bash scripts/check_all.sh` in a clean worktree at `a3167cc`, 2026-11-16 23:13 UTC, 1 min 19 s: `ruff` clean;
  `pytest` **171 passed**; validation **41 passed, 0 failed, 0 skipped of 41**. Step 2 checks: 21.1 11,052 query
  steps compared over 5 cuts × 2 estimators, 0 differ; 21.2 identical bytes; 21.3 identical bytes without labels;
  31.1 default 268,654 pairs, P50 2.91 ms, P95 7.80 ms, max 12.73 ms (bound 15 ms); stress P95 7.96 ms, max 19.10
  ms (reported). Step 1's 12.1 gives the collaborator's numbers exactly (O1 6.485°, B1 6.900°, B0 93.879°).

**Decisions.**
- Timing bound 15 ms = the 11 ms decode-delay spread + 4 ms for the clock fit; stalls exceed it by design.
- Each window enters the filter with weight `step_s/window_s` (consecutive windows share 7/8 of their samples).
- The errors column `baseline` keeps its name and now holds any method's name.
- Check numbering: Step 2 brief §8 item n is validation check 20 + n.
- Checks run in a separate git worktree at the milestone commit, so `VALIDATION.md` is stamped with a clean commit
  while work continues in the main tree.

**Open questions.** The repository's download User-Agent (above).

---

## 2026-11-16 · S2-M2 and S2-M3 (hinge; 2-DoF; range of motion) · branch `s2-m2` (from `main` = `3e6e59d`)

**Previous entry's commits.** `a3167cc`, `63adaba` (S2-M1), merged into `main` by pull request #2 as `3e6e59d`; tag
`step2-m1` (first named `s2-m1`, which collided with the branch name, so milestone tags are `step2-m<n>`).

**Plan stated.** S2-M2: the hinge estimator, GEO-K, then axis identification and GEO-ID. S2-M3: the 2-DoF constraint
with known axes and the range-of-motion cost with Ω fitted on train. **One branch and one pull request for both**:
the solvers share the dispatch in `geo.py`, and GEO-ID's range of motion needs the facing, which comes from the knee.
Both milestones' "done when" criteria are checks below.

**Files added.** `mesqsim/estimators/{anatomy,hinge,twodof,rom}.py`, `mesqsim/geo_fit.py`,
`tests/test_{hinge,twodof_rom,config_write}.py`.
**Changed.** `mesqsim/estimators/geo.py`, `mesqsim/{config,cli,validation}.py`, `configs/default.yaml`,
`docs/{MATH,CODE,PIPELINE,VALIDATION}.md`, `README.md`, `CLAUDE.md`.

**What was built.**
- Hinge (`hinge.py`): δ̂ = atan2(S, C), κ = I/σ̂² (the exact von Mises likelihood), the blind spot; axis
  identification from body rates (Seel et al. 2014); relative-sign votes by the vertical rule and a T-pose rule (the
  flipped sign would turn the knee 180° since the T-pose), which realises the brief's one-way-flexion *Should*.
- 2-DoF (`twodof.py`): the Laidig et al. 2019 constraint as a degree-2 trigonometric polynomial (five sums), Laplace κ,
  the two-minima rule (ambiguous when the likelihood ratio is below e³).
- Range of motion (`rom.py`): J(δ) = M0 + cos δ·M1 + sin δ·M2 on a 1° grid; κ by matching the circular mean of
  exp(−cost/2σ²) (the cost is flat inside the box, where Laplace is degenerate); Ω percentiles.
- `anatomy.py`: the knee axis (SMPL x), the 2-DoF axis pairs (elbow y/x, wrist z/y, ankle x/z, γ = 90°), one Cardan
  sequence per joint (ISB Z-X-Y = SMPL `XZY` for the trunk, neck and hips; `ZYX` for the shoulders).
- `geo.py`: GEO-K (true axes and mountings: hinge, 2-DoF, range of motion) and GEO-ID (identified knee axes, sign
  votes, the knee axis's direction from one-way flexion, **facing from the knees**, T-pose frames m̂ for the range of
  motion at 2-DoF and 3-DoF joints).
- `config.set_leaf_in_file`, `python -m mesqsim fit-geometry rom [--source amass] [--write]`, `geo_fit.fit_rom`.

**Commands and results.**
- Scratch run, synthetic walk and circle, 10-minute idle, two seeds each, knees only: mean error GEO-K 0.51–2.31°,
  GEO-ID 1.53–3.68°, B1 6.80–11.74°; GEO-ID's axes 0.02–0.98° from the truth (fusion noise on); every window voted
  for the right relative sign.
- The first run of check 22.1(d) failed: GEO-K's knee 5.4° off on a clean session. Cause: T-pose windows — the
  simulator's 5° posture error per segment makes the knee non-hinge there, and a static window repeats one posture
  ~256 times, claiming κ ≈ 10⁶ (−3.3° biased). Fixes: geometry windows start at the motion phase; each window's
  variance gets a model-error term (1/κ_w = 1/κ_stat + σ_m²). (d) then tests the window estimates through frames,
  Step 0, clock and pairs: max 0.0144°.
- `python -m mesqsim fit-geometry rom --source amass --write` at `894ba79` (clean worktree), 1 min 36 s: 5,974
  training sequences, 1,919,013 frames (about 30 Hz). Ω (degrees, Cardan angles 1 / 2 / 3):

  | joint | angle 1 | angle 2 | angle 3 |
  |---|---|---|---|
  | lower_back | −15.44, 69.20 | −22.90, 25.83 | −29.46, 24.47 |
  | neck | −49.96, 45.94 | −26.73, 28.04 | −56.61, 57.69 |
  | l_shoulder | −126.29, 87.86 | −78.78, 30.38 | −90.26, 90.59 |
  | r_shoulder | −102.05, 126.91 | −30.20, 81.52 | −104.54, 107.83 |
  | l_elbow | −129.07, 0.87 | −40.43, 45.72 | −30.43, 48.31 |
  | r_elbow | −5.05, 130.13 | −42.53, 45.84 | −52.63, 27.28 |
  | l_wrist | −43.04, 61.46 | −37.24, 15.75 | −71.23, 41.06 |
  | r_wrist | −64.26, 52.69 | −18.14, 36.61 | −68.11, 41.76 |
  | l_hip | −84.03, 23.71 | −10.03, 36.34 | −18.13, 14.30 |
  | r_hip | −83.26, 19.63 | −32.45, 10.90 | −14.60, 15.96 |
  | l_ankle | −39.03, 27.01 | −17.44, 19.02 | −12.47, 22.45 |
  | r_ankle | −39.69, 29.06 | −12.12, 21.09 | −23.29, 13.49 |

  **How hinge-like the AMASS knee is** (brief §5.4; RMS on the training split of the knee's rotation about its two
  other axes, Cardan `XZY` angles 2 and 3): left 6.99° and 11.37°, right 5.65° and 7.44°. **Third-axis rotation of
  the 2-DoF joints** (brief §5.5): elbows 13.71° / 16.17°, wrists 20.42° / 20.11°, ankles 9.52° / 9.06°.
- Scratch run, all joints with synthetic-fitted boxes: GEO-ID's facing from the knees; m̂ within a median 5.4–5.7° of
  the true mountings (the T-pose posture error is 5° RMS).
- `bash scripts/check_all.sh` at `05dc726` (clean worktree), 2026-11-16 23:34 UTC, 2 min 16 s: `ruff` clean; `pytest`
  **198 passed**; validation **47 passed, 0 failed, 0 skipped of 47**. 22.1: 8.9e-16 rad, −7.0000°, I = 0 and κ = 0,
  GEO-K's 38 knee windows within 0.0144°; 23.1: spread 0.790° against 0.799° predicted (ratio 0.989); 24.1: axes within
  0.001° noise-free, sign votes as expected, GEO-ID's knee axes within 0.55° on a simulated walk; 25.1: 5.0e-13 rad over
  120 windows, the built twin ambiguous with κ = 0; 26.1: 0.369° when the limits are reached, κ = 1.97 inside them
  (95 % half-width 104°), 0 without a box; 29.1: O2 0 ≤ GEO-K 3.85° ≤ B1 19.87–19.90° at both knees.

**Decisions.**
- The anatomical axes and Cardan sequences above ([MATH.md](docs/MATH.md) §12–14).
- GEO-ID's facing from the identified knee axes (the knee's flexion axis is the body's left-right axis; its direction
  from the one-way flexion) instead of the brief's optional forward arm raise (§5.8, a *Should*): no extra protocol
  step and no simulator change. Until both are known, GEO-ID's 2-DoF and 3-DoF joints hold their box estimate.
- GEO-ID uses the range of motion at the 2-DoF joints (the brief's rule until j1, j2 are identified from motion, a
  *Should* not built).
- Range-of-motion κ by moment matching, not Laplace (flat cost). Windows from the motion phase. Model-error floors.
- Fitted values are written into `configs/default.yaml` in place (comments kept) with a source naming the command,
  the data and split hash, the counts and the commit; fits run in the clean worktree.

**Open questions.** On AMASS the knee is far from an exact hinge in SMPL's frames (off-axis RMS up to 11°) and the
2-DoF joints turn 9–20° about their third axis, so the assumed model errors (2°, 3°) are too small: S2-M4 fits them
on validation.

---

## 2026-11-17 · S2-M4 and S2-M5 (filter calibration; DIODEM) · branch `s2-m4` (from `main` = `6db73cc`)

**Previous entry's commits.** `894ba79`, `05dc726`, `72f3eb3` (S2-M2/M3), merged into `main` by pull request #3 as
`6db73cc`; tags `step2-m2`, `step2-m3`.

**Plan stated.** S2-M4: windows, the filter, reboot resets, the hold rule and the κ recalibration on validation
(facing, a *Should*, came with S2-M3 from the knees). S2-M5: GEO-ID on DIODEM against the optical truth, the DIODEM
checks and a one-hinge figure. One pull request for both (the DIODEM figure depends on the calibration).

**Files added.** `mesqsim/{geo_sessions,diodem_geo}.py`, `tests/test_filter.py`, `docs/figures/diodem_geo_hinge.png`.
**Changed.** `mesqsim/estimators/{geo,hinge}.py`, `mesqsim/{geo_fit,cli,evaluate,figures,validation}.py`,
`scripts/validate.py`, `configs/default.yaml`, `tests/{test_hinge,test_config_write}.py`,
`docs/{MATH,PIPELINE,CODE,VALIDATION}.md`, `README.md`.

**What was built.**
- Solvers return their statistical κ; the filter receives κ_w = s·κ_stat/(1 + κ_stat·σ_m²) per joint type (the model
  errors are now `geometry.model_sigma_deg.{hinge,2dof,3dof}`). The estimator records its events (`GeoEvents`) and its
  outputs come from `replay`, which the fits reuse; a test shows replay reproduces GEO-ID, GEO-K and the no-filter
  ablation bit for bit.
- `geo_sessions`: the dataset's sessions of a split, simulated, decoded and estimated in memory, in worker processes,
  with the dataset's own seeds.
- `geo_fit.fit_calibration` and `python -m mesqsim fit-geometry calibration`: (s, σ_m) per joint type by maximum
  likelihood of the window errors (the scale alone when the joint optimum lies on a bound), q on a log grid by the
  filter's log-likelihood at every motion step, κ_hold as the smallest threshold whose retained steps have P95 ≤ 10°.
- **T-pose-tied knee identification** (below). Checks 27.1 (filter), 28.1 (GEO invariances), 30.1 (coverage), 32.1
  (DIODEM).
- `diodem_geo`: GEO-ID, unchanged, on the six-axis VQF orientations of the rigid IMUs at 40 Hz, presented as a Step
  0-shaped series of two pods (a hinge) with one motion phase; truth ψ = −heading(q_VQF·q_opt⁻¹).

**Commands and results.**
- First calibration (`fit-geometry calibration --source amass --seed 1 --write` at `862e352`, clean worktree): 330
  validation sessions of 78 sequences, 2 min 52 s; the hinge scale came out 0.0004, i.e. knee windows worth only
  σ ≈ 20°. **Diagnosis** (scratch script, 60 validation sessions at 10 min): GEO-K's knee windows had median error 5.2°
  (P95 14.1°; the SMPL knee is not an exact hinge), but GEO-ID's 11.6° (P95 90°, 5 % beyond 90°), and its identified
  axes lay a median 19.5° (P90 60°) from the truth: when thigh and shin turn together, every pair of axes with matching
  angles satisfies Seel's condition, so the free fit wandered. **Fix:** in the T-pose the knee axis is horizontal and
  one line in both segments, so with box registration only its azimuth is unknown (a 1-D search), a contrast test
  rejects motion that does not identify it, and a local refinement within 15° corrects posture error; the tie fixes
  the relative sign. Same sessions: GEO-ID's knee windows median 6.6°, P95 17.2°, none beyond 90°; axes a median 5.8°
  (max 19.1°) from the truth; 804 of 2,400 knee windows identifiable (the rest hold the box estimate). The first
  calibration was discarded.
- Second calibration at `944b822` (clean worktree), 330 sessions, 2 min 9 s, **written with label `fitted`**: s hinge
  60.15, 2dof 0.752 (scale only), 3dof 4.577; σ_m hinge 66.04°, 3dof 20.85° (2dof: 3° kept); so the largest window κ is
  45.2 (σ ≈ 8.5°) at the knee and 34.6 at 3-DoF joints. q = 5.623e-05 rad²/s; κ_hold = 89.13 (35.1 % of validation
  steps certain). Validation coverage 68 % / 95 %: hinge 0.715 / 0.945, 2dof 0.702 / 0.949, 3dof 0.728 / 0.949, all
  0.715 / 0.948. Report: `$MESQ_OUTPUT_DIR/fit_calibration_g944b822_amass_s1/report.json`.
- DIODEM arm exp01, hinge seg2 → seg3 (6.1 min, six-axis drift −80.7°): GEO-ID's median error 7.8° (P95 24.5°) with the
  beliefs before the fit, **15.0° (P95 29.5°) with the AMASS-fitted beliefs**, against 80.7° for a perfect registration
  at the start; first geometric information after 3.0 s. The AMASS knee calibration (σ ≈ 8.5° per window) is
  pessimistic for a near-perfect mechanical hinge, so the filter follows the fast six-axis drift more slowly: a
  transfer effect, reported in RESULTS_STEP2 with both settings.
- `bash scripts/check_all.sh` at `6b6b270` (clean worktree), 2026-11-17 22:39 UTC, 2 min 54 s: `ruff` clean; `pytest`
  **213 passed**; validation **51 passed, 0 failed, 0 skipped of 51**. 27.1: prediction exact to 4.4e-16, large-κ limit
  within 0.30 %, product identical, 3 of 3 reboots reset κ to 0, GEO-ID never worse than B1 without information;
  28.1: a common 73° yaw moves GEO outputs by at most 0.032°, one pod's 23° lands on its two joints within 0.0051°,
  tilt 1e-17; 29.1: O2 0 ≤ GEO-K 8.20–8.21° ≤ B1 19.87–19.90°; 30.1 (synthetic validation, AMASS-fitted beliefs):
  hinge 0.750 / 1.000, 2dof 0.638 / 0.807, 3dof 0.682 / 0.976; 32.1: 267 files ok, ψ 30.000°, δ −7.000°, GEO-ID 15.01°
  against 80.71°.
- A first full-check run at `9abaa8a` failed one test only (`test_config_write` expected `kappa_hold`'s label line to
  change, but it was already `fitted`); fixed so the test checks that only that leaf's lines change.

**Decisions.**
- The calibration model adds a model-error term to the brief's single scale, and falls back to the scale alone where
  the data cannot tell the two apart (2-DoF here). The knee's "σ_m = 66°" is not physical on its own: with s = 60.15
  it caps a knee window at κ = 45 (σ ≈ 8.5°), matching its windows' 6.6° median error.
- κ_hold is fitted as "the smallest threshold whose kept validation steps have P95 ≤ 10°" (`hold_target_p95_deg`, an
  assumption): "uncertain" means not shown on validation to be within that bound.
- GEO-ID's knee axes are tied to the T-pose whenever its snapshot and the box prior are valid; DIODEM (no T-pose) uses
  the free fit with the same contrast test.
- DIODEM's estimator is GEO-ID unchanged (two IMUs as pods 9/10), not a separate pipeline. Only hinges are covered.

**Open questions.** Whether the calibration should differ for the real suit's knees (closer to SMPL's or to a
mechanical hinge?) is for the bench data to answer.

---

## 2026-11-17 · S2-M6 (the Step 2 results) · branch `s2-m6` (from `main` = `88763e5`)

**Previous entry's commits.** `1ab3c52` (S2-M4/M5 results, work log, README), `f8a1911` (WORKLOG hash fix), merged
into `main` as `88763e5` (pull request #4), tags `step2-m4`, `step2-m5`.

**Files changed.** `mesqsim/results_step2.py` (new: `session_worker`, `excitation`, `bootstrap_contrast`, `holm`,
`coverage`, `selective_risk`), `scripts/make_results_step2.py` (new: the one command), `mesqsim/cli.py`
(`fit-geometry calibration --estimator K`), `mesqsim/geo_fit.py` (GEO-K's calibration; `fit_hold`),
`mesqsim/estimators/geo.py` (GEO-K's own calibration keys; the tied knee-axis test), `mesqsim/estimators/hinge.py`
(`spread`), `mesqsim/sim/session.py`, `mesqsim/sim/heading.py`, `mesqsim/sim/simulate.py` (the wearer's idle turns),
`mesqsim/validation.py` (check 33.1; helper signature), `configs/default.yaml` (new keys `kappa_scale_k.*`,
`model_sigma_deg_k.*`, `axis_tpose_contrast_max`, `axis_min_spread`; every calibration value refitted), tests
(`test_results_step2.py` new; `test_hinge.py`, `test_filter.py`, `test_sim_full.py` extended), docs (`MATH.md` §6,
§16, the error-model table; `PIPELINE.md` §10; `CODE.md`), and the generated `docs/RESULTS_STEP2.md`,
`docs/VALIDATION.md` and `docs/figures/results_step2_*.png`.

**What was built.** `python scripts/make_results_step2.py --source amass --seed 1`: the dataset's 519 **test**
sessions (117 sequences long enough for a 20 s window, 15 subjects; motion at 1, 10 and 30 min since boot; the
dataset's own seeds), simulated, decoded and scored in memory in worker processes, for B0, B1, O1, O2, GEO-K, GEO-ID
and four GEO-ID ablations; one-at-a-time sweeps at 10 min; the re-acquisition after reboots; GEO-ID on every DIODEM
hinge with the fitted and the assumed beliefs. Tables 1–10 (and 3b), H2's contrasts C1–C3 with subject-level paired
bootstraps (10,000 resamples, seed 20261117) and Holm's correction, coverage, selective risk and five figures. Check
33.1: the same config and seed give the same file apart from its Run line.

**Commands and results, in order.**
- GEO-K's own calibration (`fit-geometry calibration --source amass --seed 1 --estimator K --write`, at `03d4a18`,
  clean worktree): GEO-K had been using GEO-ID's window scales, and its 2-DoF windows (a different solver) came out
  over-confident (P95 89° at 68 %/95 % coverage 0.25/0.45 on a trial). Its own keys fix it (values below).
- **First full results run** at `f988391` (13.7 min on an Apple M2 Pro). Three findings, all investigated on
  **validation** data before anything changed:
  1. GEO-ID was almost B1: the T-pose-tied knee axes were accepted in only 44 % (left) and 47 % (right) of 78
     validation sessions at 10 min, and without knee axes GEO-ID has no facing and runs no other solver. Yet the
     tied fits with contrast 0.25–0.8 were as accurate as those below 0.25 (median thigh-axis error 6.9–8.1° and
     5.2–8.9° by bin, left/right; 13–24° above 0.8). A threshold scan on 156 validation sessions (10 and 30 min),
     mean session-median error, hinge / 2-DoF / 3-DoF: 0.25 → 7.62 / 7.61 / 8.21°; 0.6 → 6.89 / 7.13 / 7.45°;
     0.7 → 6.55 / 7.08 / 7.34°; 0.8 → 6.50 / 7.05 / 7.31°; 0.9 → 6.52 / 7.01 / 7.26°; 1.0 → 6.51 / 7.00 / 7.26°
     (B1 8.90 / 8.33 / 9.57°); the facing was found in 56 %, 82 %, 90 %, 91 %, 93 %, 93 % of sessions. Dropping the
     local four-angle refinement changed little (0.8: 6.61 / 7.04 / 7.33°), so it stays. **Change:**
     `geometry.axis_tpose_contrast_max` = 0.8 for the tied fit (the free fit keeps 0.25). The rigid-motion test then
     showed the contrast of a rigid motion to be 0.79 — a ratio of two near-zero costs — so a second test was added:
     `geometry.axis_min_spread` = 0.1 (the residual at the median azimuth relative to the rates: below 0.01 for rigid
     motion, above 0.3 for every one of 408 validation knee fits).
  2. C3's verdict read "GEO-ID better by more than the margin" when its interval only failed to exclude that: the
     wording is now "an improvement above the margin is not excluded".
  3. Table 9 showed GEO-ID never re-acquiring after reboots (≈ 90°). A first diagnosis (a forced reboot *during* the
     motion of an identifiable clip re-acquired, ankle 29° → 6–7°) blamed finding 1. It was wrong: the same reboot in
     the *idle* never re-acquired (the T-pose snapshot, below).
- While writing PIPELINE_EXPLAINED, a **simulator bug**: the idle's whole turns, documented as the wearer's, were
  drawn per pod. Pods on one body cannot make different whole turns; the per-pod draws added k·2π·(n_c − n_p) of
  relative error (up to ≈ 10° for a two-turn difference at k ≈ 1.4 %). **Fixed** (`045b343`): one
  draw per session in `plan_session` (own stream `idle_turns`), shared by all pods; test
  `test_idle_whole_turns_belong_to_the_wearer_and_every_pod_shares_them`. The explanation session's left-knee B1 error
  fell from 27.7° to 19.0°. `docs/RESULTS_STEP1.md` (tag `step1`) was made with the per-pod draws; regenerating it on
  AMASS takes about 12 h and 100 GB of runs, so it stays as the `step1` record, and RESULTS_STEP2 (which has B0–O2 on
  the test subjects with the fix) supersedes its idle-dependent numbers.
- Calibrations refitted, each in the clean worktree at the named commit, validation only (330 sessions of 78
  sequences, 9 subjects): at `fab6e3d` (tied threshold), superseded by the simulator fix; at `045b343`, where κ_hold
  came out 10⁶ ("no step is ever certain": the best retained P95 was 10.25°, just above the 10° target, because the
  box prior's own error tail at 1 min sits near 10°). **Change** (`f85ec19`): `fit_hold` keeps the rule, and when no
  threshold reaches the target takes the one with the smallest retained P95 that keeps at least 10 % of the steps
  (test `test_kappa_hold_meets_the_target_or_falls_back_to_the_smallest_tail`); the config source says which rule
  applied. **Final values** (`f85ec19`, written with label `fitted`; committed as `317323c`):
  - GEO-ID: hinge s = 0.011776, σ_m = 0.9385° (cap 43.9, ≈ 8.6° per window; 9,536 informative knee windows, was
    3,821); 2-DoF s = 0.556852 (σ_m 3° kept); 3-DoF s = 3.728253, σ_m = 20.68°; q = 5.623e-5 rad²/s;
    κ_hold = 158.49 (fallback rule, retained P95 10.16°, 32.5 % of steps certain). Validation coverage 68 % / 95 %:
    hinge 0.692 / 0.926, 2-DoF 0.704 / 0.927, 3-DoF 0.730 / 0.949, all 0.713 / 0.936.
  - GEO-K: hinge s = 0.013994, σ_m = 0.7313°; 2-DoF s = 0.003916, σ_m = 2.2575°; 3-DoF s = 3.496256,
    σ_m = 11.2873° (unchanged by the fix: windows never see the idle). Coverage 0.734 / 0.948.
- **The reason for finding 3, an estimator bug:** the T-pose snapshot (q_i(T) and the box belief, which tie the knee axes and give
  every pod its T-pose frame) was taken at the first geometry window, i.e. after the idle. A pod that rebooted in the
  idle had its joints reset at the motion's first frame, before the snapshot, so it lost the tie and its T-pose frame
  (and its descendants theirs) for good. **Fixed** (`d6c0487`): the snapshot is an event of its own at the end of the
  T-pose; test `test_a_reboot_in_the_idle_keeps_the_tpose_frames` (fails before the fix). Sessions without such a
  reboot get the same snapshot, so the calibration and every table without reboots are unchanged (confirmed below).
- `bash scripts/check_all.sh` at `317323c` (clean worktree), 3 min 46 s: ruff clean; pytest 219 passed; validation
  52 of 52. **At `d6c0487`** (clean worktree), 2026-11-18 05:10 UTC, 3 min 47 s: ruff clean; pytest **220 passed**;
  validation **52 passed, 0 failed, 0 skipped of 52**. Step 2 checks: 21.1–21.3 identical bytes; 22.1 GEO-K knee
  windows max 0.0144°; 23.1 spread ratio 0.989; 24.1 axes within 0.001° noise-free, 0.56° on a walk; 26.1 0.369° /
  κ 1.97 / 0; 27.1 exact; 28.1 common yaw 0.127°, one pod 0.0043°; 29.1 GEO-K 6.35° / 5.47° against B1 21.30° /
  18.21°; 30.1 hinge 0.833 / 1.000, 2-DoF 0.721 / 0.843, 3-DoF 0.716 / 0.981 (12 synthetic sessions); 31.1 P50 2.91
  ms, P95 7.79 ms, max 12.73 ms; 32.1 GEO-ID 15.18° against 80.71°; 33.1 identical apart from the Run line.
- **The results:** `python scripts/make_results_step2.py --source amass --seed 1` at `d6c0487` (clean
  worktree), 18.5 min of compute (the Mac slept through most of its 8 h wall time; apart from the Run line the file
  differs from the `317323c` run only in the reboot rows, as the snapshot fix predicts). 519 test sessions. All 14
  joints, mean error at 1 / 10 / 30 min since boot: B1 3.90 / 7.08 / 11.91°; GEO-ID 3.95 / 6.69 / 9.96°; GEO-K
  (truth) 3.75 / 6.30 / 9.08°; O1 (truth) 3.17 / 6.67 / 11.65°. At 30 min by joint type: hinge B1 12.15° → GEO-ID
  9.25° (GEO-K 6.37°); 2-DoF 11.76° → 9.75° (10.84°); 3-DoF 11.97° → 10.41° (8.23°). GEO-ID found the facing in 358
  of 519 sessions (69 %; with the knee axes in 92–93 % of those); in the other 161 it equals B1 by construction.
  **H2** at 10 min (GEO-ID − B1 per subject, 95 % bootstrap interval, Holm): C1 hinge −0.98° [−2.02, −0.07], p 0.03
  (0.09); C2 2-DoF −0.78° [−1.79, 0.04], p 0.06 (0.13); C3 3-DoF −0.08° [−0.66, 0.44], p 0.81 (0.81). By the draft
  plan C1 and C2 are "not shown better" (C1's mean gain sits just under the 1° margin) and C3 "supports H2";
  descriptive. Coverage of the 68 % / 95 % intervals on test: GEO-ID 0.71 / 0.94 at every joint type; GEO-K's knees
  0.58 / 0.89. Sweeps (mean at 10 min): GEO-ID below B1 in every condition (drift 2.0 °/min 19.49° against 23.74°;
  one reboot per pod per hour 17.62° against 29.10°). Re-acquisition after a reboot (median of the rebooted pod's
  joints): GEO-ID 78.19° at 2–5 s, 12.65° at 5–10 s, 9.37° at 10–20 s (GEO-K 10.81°, 8.50°, 7.83°; B1 about 88°).
  DIODEM: GEO-ID's median error below the start registration's in 21 of 27 hinge runs; arm exp01 seg2–seg3 15.18°
  (P95 29.63°) against 80.71°, and 7.79° with the beliefs assumed before the AMASS fit.

**Decisions.**
- Both estimator changes (the tied knee-axis threshold with its spread test, and the κ_hold fallback) were chosen on
  **validation** data only. The first came from a look at one test-split clip while preparing the explanation (its
  identified knee axis was 12° off); it was then measured, and decided, on validation sessions alone. The analysis
  plan is not frozen, so the results stay descriptive either way.
- The simulator fix changes every session with an idle; Step 1's results file is kept as the `step1` record (above).
- GEO-K keeps its own calibration keys: its solvers differ from GEO-ID's.

**Open questions.**
- **Correlated window errors.** Range-of-motion windows (and identified-knee windows) carry errors that persist over
  many consecutive windows, while the filter treats windows as independent beyond the 1/8 overlap weight. Over many
  sessions the calibration absorbs this (coverage 0.71 / 0.94 on test), but within one session GEO-ID can make a
  joint whose box registration happened to be good worse (PIPELINE_EXPLAINED's session: the hips). A per-session
  bias term, or a calibration that models the windows' correlation, is a question for Step 4's fusion as much as for
  Step 2.
- **Identification in short motion.** The test sessions have 20 s of motion; the knee axes were refused in 31 % of
  them. Longer windows (`dataset.window_s`) or concatenated sequences would test whether more motion helps.
- **GEO-K's 2-DoF constraint** does worse than GEO-ID's range of motion at the 2-DoF joints (its raw κ needs a scale
  of 0.0039): fixed axes at 90° fit the SMPL elbows, wrists and ankles poorly.
- **Transfer to real hinges:** the AMASS knee calibration is pessimistic for DIODEM's mechanical hinges (the assumed
  beliefs did better there); the bench must say which the real suit's knees resemble.
- Whether `docs/RESULTS_STEP1.md` should be regenerated with the shared idle turns (about 12 h, 100 GB).

---

## 2026-11-18 · S2-M7 (the explanation pipeline) · branch `s2-m7` (from `main` = `2f14e9c`)

**Previous entry's commits.** S2-M6: `03d4a18`, `f988391`, `fab6e3d`, `e1f8c3b`, `045b343`, `f85ec19`, `317323c`, `d6c0487`,
`de719c2`, merged into `main` as `2f14e9c` (pull request #5), tag
`step2-m6`.

**Files added.** `docs/explain/README.md` (the index: the pipeline, what each file answers and how long it takes,
which file for which question), `MENTAL_PICTURE.md` (no equations; the suit, the room, the simulator, Step 0, the
baselines, the geometry, confidence and memory, one diagram and one in/out table), `PIPELINE_EXPLAINED.md` (one real
AMASS test-split session through the ten stages, with its numbers, and the DIODEM path), `MATH_EXPLAINED.md` (every
formula derived, each naming its function and its test; worked examples E1–E29), `CODE_EXPLAINED.md` (the module
map, `baselines --methods GEO-ID` traced call by call, the Step 2 modules block by block, the Step 1 modules Step 2
stands on, the tests, how to extend, how to debug), `THEORY.md` (rotations, inertial sensing, gauge and
identifiability, kinematic constraints, directional statistics, estimation, evaluation, a bridge to Steps 3–4, full
references); every file ends with self-check questions and answers. `tests/test_explain_examples.py` recomputes every
worked example (27 tests); `tests/test_explain_links.py` checks every link, anchor, `path:line` reference, linked
symbol and named test (the link text's backticked symbol must stand on the linked line, so a reference cannot drift
away from the code it names).

**Commands and results.**
- The session PIPELINE_EXPLAINED follows: `python -m mesqsim simulate --seed 1 --motion
  "$MESQ_DATA_DIR/amass/CMU/13/13_02_poses.npz" --set session.idle_s=562` and `python -m mesqsim baselines --run
  <folder> --methods B0 B1 O1 O2 GEO-K GEO-ID` at `d6c0487` (clean worktree; run folder
  `run_gd6c0487_c5ce0b3f7_s1`; identical bytes at `317323c`). All joints, mean / median / P95: B1 7.02 / 4.23 /
  18.86°, GEO-ID 7.58 / 7.00 / 17.65°, GEO-K 5.93 / 4.71 / 11.71°. The per-stage numbers (plan, ψ, Step 0 counts,
  clock maps, pair timing, window solutions, filter states) were read from that folder and from
  `GeoEstimator.run(..., diagnostics=True)` with short scratch scripts.
- `pytest tests/test_explain_examples.py tests/test_explain_links.py`: 46 passed (27 worked examples; 19 link checks over the six files).
- `bash scripts/check_all.sh` at `822a171` (clean worktree), 2026-11-18 16:21 UTC: ruff clean; pytest **266 passed**
  (the 46 new ones included); validation **52 passed, 0 failed, 0 skipped of 52**. Its wall time was 2 h 12 min for
  4 min 59 s of compute: the Mac, on battery, slept between steps.

**Fixes found while explaining** (rule: where the explanation and the code or MATH.md disagree, fix the wrong one).
- MATH_EXPLAINED cited a test that does not exist (`test_targets_are_child_minus_parent`); it now cites
  `test_worked_example_section_4_4` and `test_same_example_through_the_tree_functions`.
- THEORY §6.4 overstated what recalibration shows; it now gives the measured coverage and names its command.
- Following one session through the pipeline exposed the low knee-axis acceptance (GEO-ID equal to B1 at every joint
  of the first session tried) and the per-pod idle turns; the late T-pose snapshot came out of the reboot table. All
  three were fixed in S2-M6, before its final results (that entry).
- E26 (the knee window cap) was rewritten for the refitted calibration (s = 0.011776, σ_m = 0.9385°: cap 43.89,
  8.65°); its test computes it through `calibrated_kappa`.

**Decisions.** The worked session is a test-split clip (`CMU/13/13_02`): nothing is fitted on it, and it is the
split the results describe. It was chosen for its knee motion, and because it shows GEO-ID's successes (the right
knee, the lower back, the right ankle) and failures (a left-knee axis 24.5° off; the hips) side by side.

---

## 2026-11-18 · S2-M8 (documentation complete) and the Step 2 report · branch `s2-m8` (from `main` = `a8642d0`)

**Previous entry's commits.** S2-M7: `822a171`, `1f461f1`, merged into `main` as `a8642d0` (pull request #6), tag
`step2-m7`.

**Files changed.** `README.md` (the picture with Step 2, status, the quickstart to the Step 2 results, links to
`docs/explain/` and `project_explanations/`), `docs/MUST_KNOW.md` (§10: the Step 2 section and ten self-check
questions), `docs/SIMULATOR.md` (the drift stress value 2.0 °/min; the wearer's idle turns; the `idle_turns`
stream), `project_explanations/DATASETS.md` (new, Danyal's request: AMASS and DIODEM, what they are, how they were
recorded and on what hardware, what matters about them here; facts cited to the papers and to our own inspection
commands), `WORKLOG.md` (this report), `docs/VALIDATION.md` (regenerated).

### The Step 2 report (brief §11)

**1. What was built** (Step 2, S2-M1 … S2-M8):
- estimators: `mesqsim/estimators/{contract,clock,anatomy,hinge,twodof,rom,vmfilter,geo}.py`;
- scorer side: `mesqsim/{evaluate,geo_score,geo_sessions,geo_fit,results_step2,diodem_geo}.py`, `mesqsim/cli.py`
  (`baselines --methods`, `params --label fitted`, `fit-geometry rom|calibration [--estimator ID|K]`),
  `mesqsim/config.py` (sections, in-place writes), `mesqsim/conventions.py` (Cardan angles, von Mises helpers),
  `mesqsim/validation.py` (checks 21.1–33.1), `mesqsim/figures.py`;
- simulator fix: `mesqsim/sim/{session,heading,simulate}.py` (the wearer's idle turns);
- scripts: `scripts/make_results_step2.py`;
- config: `configs/default.yaml` (`geometry:`: 25 assumption and 25 fitted values);
- tests: `tests/test_{conventions_step2,clock,geo_contract,hinge,twodof_rom,config_write,filter,results_step2,
  explain_examples,explain_links}.py`, and additions to `test_sim_full.py`;
- docs: `docs/RESULTS_STEP2.md` and its figures (generated), `docs/VALIDATION.md` (generated), `docs/MATH.md` §9–§18,
  `docs/PIPELINE.md` §7–§10, `docs/CODE.md`, `docs/MUST_KNOW.md` §10, `docs/SIMULATOR.md`, `README.md`, the six files
  of `docs/explain/`, `project_explanations/DATASETS.md`, this work log.

**2. `bash scripts/check_all.sh`** at `e05570c` (clean worktree; the documentation commit of this milestone), 2026-11-18 17:07 UTC, 3 min 56 s:

```text
== environment
Python 3.11.16
== ruff
All checks passed!
== pytest
266 passed in 115.30s (0:01:55)
== validation
wrote docs/VALIDATION.md
VALIDATION: PASS
== ALL CHECKS PASSED
```

Validation: **52 passed, 0 failed, 0 skipped of 52** (`docs/VALIDATION.md`, commit `e05570c`).

**3. The Step 2 results** (`python scripts/make_results_step2.py --source amass --seed 1` at `d6c0487`; AMASS test
subjects; parameter set: offsets arbitrary, box slot 2°, drift 0.2 °/min, scale factor ±1.5 % (datasheet), tilt
0.5°, no reboots, loss 2 %, the fitted geometry beliefs above; banner: assumed suit, descriptive). Mean error over
all 14 joints, degrees (`*` = uses simulator truth):

| method | 1 min | 10 min | 30 min |
|---|---|---|---|
| B1 | 3.90 | 7.08 | 11.91 |
| O1 * | 3.17 | 6.67 | 11.65 |
| GEO-K * | 3.75 | 6.30 | 9.08 |
| GEO-ID | 3.95 | 6.69 | 9.96 |

By joint type at 30 min (mean / P95): hinge B1 12.15 / 35.84, GEO-ID 9.25 / 27.53, GEO-K 6.37 / 14.76; 2-DoF
11.76 / 32.20, 9.75 / 27.68, 10.84 / 29.81; 3-DoF 11.97 / 33.89, 10.41 / 28.94, 8.23 / 24.12. GEO-ID's facing (hence
its non-knee solvers) was found in 358 of 519 sessions (69 %); in the other 161 it equals B1 by construction.
Coverage of GEO-ID's 68 % / 95 % intervals on test: 0.71 / 0.94 at every joint type (H5's ±5-point target met);
GEO-K's knees under-cover (0.58 / 0.89). DIODEM: GEO-ID's median error below the start registration's on 21 of 27
hinge runs; arm exp01 seg2–seg3 15.18° against 80.71°. With one reboot per pod per hour GEO-ID's mean error is 17.62° against B1's 29.10°, and after a reboot it re-acquires the rebooted pod's joints within 5–10 s (median 12.65°, then 9.37° at 10–20 s; GEO-K 8.50° and 7.83°).

**4. H2** (at 10 min, GEO-ID − B1 per subject, 10,000 bootstrap resamples, Holm across C1–C3, margin 1°,
**descriptive** because the plan is not frozen):
- **Hinge (C1):** −0.98° [−2.02, −0.07], p = 0.03 (Holm 0.09). GEO-ID is better than B1 at the knees, but the mean
  gain sits just under the proposed 1° margin, so by the draft plan's two-part definition it is "not shown better".
- **2-DoF (C2):** −0.78° [−1.79, 0.04], p = 0.06 (Holm 0.13): probably better, not shown.
- **3-DoF (C3):** −0.08° [−0.66, 0.44], p = 0.81: an improvement above the margin is excluded, as H2 predicts.
- **How firm:** the direction is the one H2 predicts at every joint type, and the gains grow with time since boot
  (at 30 min the hinge mean falls from 12.15° to 9.25°); but nothing is confirmatory, the suit's error model is
  assumed, and the knee identification (69 % of sessions) limits GEO-ID: GEO-K, with perfect axes, shows the
  ceiling (hinge 6.37° at 30 min).

**5. Every assumption still in the config** (63; `python -m mesqsim params --label assumption`):
  - `motion`: `spine_joint` 9.
  - `session`: `boot_spread_s` 0.5; `box_s` 5.0; `strapping_s` 30.0; `strap_move_s` [2.0, 6.0]; `strap_spins_max` 1; `tpose_s` 3.0; `idle_s` 0.0; `tpose_posture_deg` 5.0; `facing_error_deg` 15.0; `mounting_max_deg` 20.0.
  - `timing`: `chip_start_s` [0.2, 0.5]; `decode_delay_ms` [0.0, 11.0]; `grid_start_ms` 1000.0; `stall_ms` 0.0; `stall_period_s` 3.0; `stall_jitter_s` 0.5; `decode_busy_ms` 1.0; `pod_clock_ppm` 50.0; `host_epoch_s` [100.0, 1000.0].
  - `radio`: `loss_mean` 0.02; `burst_mean_frames` 3.0; `latency_min_ms` 2.0; `latency_jitter_ms` 3.0.
  - `error_model`: `offset_mode` arbitrary; `box_slot_deg` 2.0; `drift_sigma_deg_per_min` 0.2; `tilt_sigma_deg` 0.5; `tilt_tau_s` 1.0; `reboot_rate_per_h` 0.0; `reboot_silence_s` 3.0; `idle_turns_sigma_per_sqrt_min` 0.5.
  - `battery`: `start_pct` [70.0, 100.0]; `drain_pct_per_h` 10.0.
  - `dataset`: `window_s` 20.0; `windows_per_sequence` 3; `motion_start_min` [1, 10, 30]; `variants` 1.
  - `geometry`: `window_s` 8.0; `step_s` 1.0; `max_pair_gap_s` 0.1; `clock_window_s` 30.0; `clock_block_s` 2.0; `box_sigma_deg` 2.0; `hold_target_p95_deg` 10.0; `model_sigma_deg.2dof` 3.0; `hinge_sigma_floor_deg` 0.5; `axis_span_s` 120.0; `axis_min_rate_deg_s` 20.0; `axis_min_pairs` 64; `axis_max_residual` 0.5; `axis_contrast_max` 0.25; `axis_tpose_contrast_max` 0.8; `axis_min_spread` 0.1; `axis_tpose_tolerance_deg` 15.0; `axis_starts` 8; `sign_min_elev_deg` 5.0; `tpose_sign_margin_deg` 60.0; `facing_flex_min_deg` 20.0; `twodof_sigma_floor_deg` 0.5; `twodof_ambiguity_llr` 3.0; `rom_sigma_deg` 5.0; `rom_percentiles` [0.5, 99.5].

**Every fitted value and its command** (25; validation or train only):
- `python -m mesqsim fit-geometry rom --source amass --write` (train): `geometry.rom_limits_deg.*` for the 12 non-knee
  joints (lower back, neck, shoulders, elbows, wrists, hips, ankles).
- `python -m mesqsim fit-geometry calibration --source amass --seed 1 --estimator ID --write` (validation, at
  `f85ec19`): `geometry.kappa_scale.{hinge 0.011776, 2dof 0.556852, 3dof 3.728253}`,
  `geometry.model_sigma_deg.{hinge 0.9385, 3dof 20.6758}`, `geometry.drift_q_rad2_per_s` 5.623e-05,
  `geometry.kappa_hold` 158.4893.
- `python -m mesqsim fit-geometry calibration --source amass --seed 1 --estimator K --write` (validation, at
  `f85ec19`): `geometry.kappa_scale_k.{hinge 0.013994, 2dof 0.003916, 3dof 3.496256}`,
  `geometry.model_sigma_deg_k.{hinge 0.7313, 2dof 2.2575, 3dof 11.2873}`.

**6. What is blocked, and on whom.**
- **AMASS:** not blocked (downloaded and verified 2026-11-13; the four subsets above).
- **The plan freeze (Danyal):** the split manifest (SHA-256 `d36448ad…`) and `docs/analysis_plan_DRAFT.md`, including
  the 1° margin. Until then every Step 2 comparison is descriptive.
- **The bench measurements (Danyal):** every `assumption` of the error model (drift, scale factor, slot error, tilt,
  loss, stalls, reboots, idle turning) waits for the 70-minute box evening and the taped-board session; after them the
  calibration must be refitted (`fit-geometry calibration`, both estimators) and the results regenerated.
- **Real recordings:** none exists in the frames-table format yet.

**7. What Danyal must decide.**
- Freeze the analysis plan and the split (and the margin Δ = 1°).
- Pushing the estimator code (rule 3): **decided** — public repository (2026-11-16).
- Switching to the DIODEM-only project if AMASS were missing: **not needed** (AMASS is here).
- The Spine joint: spine3 (default) or spine2.
- New in Step 2: whether Step 1's `docs/RESULTS_STEP1.md` should be regenerated with the fixed idle turns (about 12 h
  and 100 GB of runs on AMASS); whether the knee identification thresholds and the κ_hold fallback, chosen on
  validation, should be frozen in the plan.

**8. Where the explanations are:** `docs/explain/` — README.md (the index), MENTAL_PICTURE.md, PIPELINE_EXPLAINED.md,
MATH_EXPLAINED.md, CODE_EXPLAINED.md, THEORY.md — and `project_explanations/DATASETS.md` for the two datasets.

**9. A proposed plan for Step 3, the learned estimators (not started).**
- **Contract:** the same `estimate(series, session, t_query, cfg) -> Estimate(δ̂, κ, uncertain)`, causal, label-blind,
  deterministic at inference; the same scorer, strata and subject-level statistics, so geometry and learning are
  compared on identical sessions.
- **Inputs:** gauge-invariant features only (a common yaw of the room must change nothing): per pod, gravity in its
  own frame and body rates; per joint, the parent-to-child relative rotation q_p⁻¹·q_c; whole-body models see every pod
  relative to the pelvis pod. Trained on simulated **train** sessions, early-stopped and calibrated on **validation**.
- **Outputs:** per joint a von Mises (μ as a unit vector (cos μ, sin μ), κ ≥ 0 through a softplus), trained with the
  von Mises negative log-likelihood −κ·cos(δ − μ) + log I₀(κ) (computed with `i0e`), so κ is learned as a confidence.
- **Models:** **TCN-J** — a causal temporal convolutional network per joint on that joint's two pods (weights shared
  within a joint type), receptive field ≈ 8 s like the geometry windows; **TCN-B** — one causal TCN on the whole body,
  14 heads; **MLP-B** — the whole body over a fixed recent window without temporal structure (does time help?);
  **GRU-B** — a recurrent whole-body baseline (does an unbounded memory help?).
- **Tests before results:** the invariance checks of §5.9 (a common yaw, one pod's yaw), causality and label-blindness
  as in checks 21.1–21.3, and coverage on validation; then H3 (TCN's advantage larger at 3-DoF joints than at hinge
  and 2-DoF joints) on test, and in Step 4 the fusion of GEO-ID and the best network by the product of their von Mises
  densities (H4).

## 2026-11-17 to 2026-11-18 · S3-M0 (start Step 3) · branch `s3-m0` (from `main` = `88763e5`, rebased onto `step2` = `fc89dbb`), worktree `../ml_mesquite_s3`

**State (Step 3 brief §2.1).** No `step2` tag at the start: Step 2 was at S2-M6 (committed locally on `s2-m6`), and a
Step 2 session was busy in the main folder. This milestone therefore ran in its own git worktree and changed nothing
there. Step 2 was finished and tagged `step2` (`main` = `fc89dbb`, pull request #7) before this milestone was
committed, so the branch was rebased onto it, and S3-M1 follows at once (option (a)'s condition is met).

**Plan stated.** Save the brief and update `CLAUDE.md`; ask the S3-M0 questions once; copy and hash the recordings;
clone `mesquite-mocap` at `87d7c3b`; take the suit facts from it, where the person pointed; record every answer.

**Answers** (the person running the session; asked 2026-11-17, follow-ups 2026-11-18).
1. Step 2 gate: **option (a), wait.** S3-M1 onward starts once `step2` exists.
2. Disclosure (rule 3): **keep the Step 3 code local.** Nothing of Step 3 is pushed until the person says otherwise.
3. Recordings: copy them; no phone-app log of the Nov 14 take exists. Done (Commands).
4. Suit facts: "the positions are in the mesquite-mocap repo; the hips pod is on the back and the phone on the front;
   the T-pose is done in front of the laptop". What `mesquite-mocap@87d7c3b` says, and the follow-ups:
   - Pods: the velvet wear guide, `trees/velvet/wearguide/threejsscene.js:334-404`, gives 14 positions (cm, avatar
     frame: +Y up, facing +Z, left at +X) and rotations of `pod.glb`, whose case is long along its X (49–53 mm), thin
     along Y (7–18.5 mm, the face normal) and wide along Z. Converted with SciPy (Euler XYZ, intrinsic): head (−2, 168,
     8) and chest (−2, 137, 14) on the front, case X up (tilted 20°), face forward; upper arms (∓30, 148), forearms
     (∓56/58, 147) and hands (∓77, 146) on top of the arm in the T-pose, case X along the arm outward, face up, the
     left a 180° turn of the right about the vertical (not a mirror); thighs (−12/8, 65, 6.5) and shins (−11/7, 30,
     2.2) on the front, case X up, face forward, **the same on both legs, not mirrored**; feet (−7/11, 9, 4) on top,
     case X forward, face up. `smooth` and `honey` place the pods the same way; `thorny` moves the head and chest.
   - HipsAlt pod: on the lower back (the answer; the brief's §5.1 marker table agrees). The wear guide draws none.
     Only `velvet` and `smooth` use HipsAlt; `honey`, `spiny`, `velvetGreen` and `bc` disable it.
   - Phone: front of the belt, (0, 100, 12) turned 180° (`threejsscene.js:296-298`); screen against the body, rear
     camera looking ahead (`new_device_code/phone_app/README.md:3`, `HOW_IT_WORKS.md:79-82`).
   - Box: all 15 pods face up with one rotation, a 5×3 grid on a small table in front of the laptop
     (`trees/bc/wearguide/threejsscene.js:280-327` and `:395-461`).
   - T-pose: facing the laptop (the answer), the avatar's forward (+Z) in the box guide.
   - **Chip axes** (answer; `assumption` until a bench test), by the datasheet's convention (the ICM-20948's Z is
     perpendicular to the package) with the board flat in the case: chip +Z = out of the case face (`pod.glb` +Y),
     chip +X = along the case length (`pod.glb` +X), chip +Y = `pod.glb` −Z (right-handed). The rotation within the
     face and the signs are unverified (the module may be turned or upside down), so the mounting stays one
     parameter per pod. **Bench test (Step 5):** lay a pod face up, then on its long edge, then on its short edge,
     reading the raw accelerometer each time; the axis reading about +8192 (1 g at ±4 g) points up. The velvet tree
     cannot settle it: it maps the DMP's world frame to three.js, whatever the mounting.
5. Trees: **`velvet` on Nov 13 and on Nov 14.**
6. USB: **no pod was plugged or unplugged after the box step**, on either day.
7. Nov 14: **Zero in T-pose was pressed**, in the T-pose, before the page's Set T-pose.
8. Disk and compute budget: not answered; the brief's rule 5 (`CLAUDE.md` rule 13) holds, as an `assumption`.
9. Docs (2026-11-18): **do not write Step 3's Markdown explanation files**, to save tokens; list them to request
   later (below). `CLAUDE.md` rule 14 records the change.

**What the answers change.** Two of the brief's readings of Nov 14 (§2.4, items 1 and 3) are ruled out: with no
replug, the left leg's +75° needs another cause (a pod rebooting by itself, a pod turned in the box, the box step
itself); a Zero in T-pose rules out the automatic Zero at Start, and the log below shows `camera_facing=forward`,
right for the front of the belt. So the +74° travel and the 116 cm root (p_y ≈ 1.32 m) are open for S3-M1. Both legs
are mounted alike in the wear guide, so §4.4's mirrored pair would need a different mounting from the guide's to
explain Nov 13's outward drift; S3-M6 compares it with scale error × turning, as planned.

**The phone-app log.** `real_suit/phone_app_logs/hips_20261114_162125.csv`, from the Mac's Downloads folder. Its
epoch column runs 2026-11-15 00:21:25–00:23:02 UTC, 26 min before the take (00:49:16–00:52:25 UTC, the BVH's
provenance lines), so it pairs with no recording. Its name is one hour behind those times read in Arizona time (MST);
a phone set to Pacific time would explain it (unverified). Header: `page_tree=velvetFamily`, `camera_facing=forward`,
`hip_height_m=1.000`, `lever_cam_m=0,0,0.1` (manual), `rate_hz=30`. Zero epochs at t = 21005.13 s (heading 1.11°, at
the first good frame: the automatic Zero) and 21027.99 s (−47.69°).

**Files added.** `docs/brief/STEP3_BRIEF.md` (the master prompt below its first line of dashes; `diff` against the
source: identical; 1,405 lines); `manifests/real_suit.csv` (18 files, 259,733,202 bytes). **Changed.** `CLAUDE.md`
(three briefs, Step 3 scope and rules, the answers), `WORKLOG.md`.

**Commands.**
- `cp -Rp` of `mesquite-mocap/Mesquite_benchmarks/.` (17 files: seven BVH, five Motive CSV, five TAK) and the log
  into `$MESQ_DATA_DIR/real_suit/` (248 MB); every SHA-256 equals its source's. `python scripts/check_env.py --require
  real_suit`: 18 of 18 ok.
- `git clone --no-checkout` of the local `mesquite-mocap`, `checkout --detach 87d7c3b`, origin set to GitHub, into
  `$MESQ_DATA_DIR/mesquite-mocap` (399 MB). `87d7c3b` is on GitHub's `main`; there `Mesquite_benchmarks` is a
  submodule (at `6db7b8d`), not checked out. Disk before the copy: 77 GiB free (52 GB on 2026-11-16).
- `bash scripts/check_all.sh` in the worktree at `88763e5` with these files uncommitted, 2026-11-18 22:32 UTC, 172 s:
  `ruff` clean; `pytest` 213 passed; validation 51 passed, 0 failed, 0 skipped of 51. After the rebase onto `step2`
  (`83d3d89`), 22:39 UTC, 232 s: `ruff` clean; `pytest` **266 passed**; validation **52 passed, 0 failed, 0 skipped
  of 52**. No code changed, so the regenerated `docs/VALIDATION.md` (its run line only) was not committed.

**Step 3's Markdown parts to request later** (Step 3 brief sections; none written):
- `docs/step_3/` (§12): `README.md` (entry point, reading order, which file answers which question);
  `MENTAL_PICTURE.md`; `PIPELINE_EXPLAINED.md`; `MATH_EXPLAINED.md`; `CODE_EXPLAINED.md`; `THEORY.md`; `REFERENCES.md`
  with `references.bib`; `FIDELITY.md` (the ledger, named in S3-M6's "done when"); `DATASET_CARD.md` (twin_v1 and the
  loader's specification; S3-M7, needed before Danyal writes the loader); `LEARN_THE_MODEL.md` (the twelve-lesson
  guide; S3-M8, needed before Danyal starts the models); `MODEL_CARD_TEMPLATE.md`.
- §11: `README.md` (picture, Step 3 status, quickstart); `docs/SIMULATOR.md`; `docs/DATA.md`; `docs/PIPELINE.md`,
  `docs/MATH.md` and `docs/CODE.md`; `docs/MUST_KNOW.md` (a Step 3 section); `docs/analysis_plan_DRAFT.md` (H6, H7,
  the recording split, the dataset-size rule, TCN-P's contrasts; the split must be frozen before S3-M6 fits).

**Open.** The chip axes (the bench test above); the causes of Nov 14's left leg, travel direction and root height
(S3-M1); the budget, if the default does not suit.

## 2026-11-18 · S3-M1 (the recordings measured) · branch `s3-m1` (stacked on `s3-m0`; nothing pushed, rule 2)

**Plan stated.** `mesqsim/real/`: BVH and Motive readers, marker naming and following, clock alignment, Φ1-Φ12, one
command; then reproduce Step 3 brief §5.3 or explain every difference here. `mesqsim/twin/page_frame.py` holds the
frame conversions (rule 10).

**Files added.** `mesqsim/twin/{__init__,page_frame}.py`; `mesqsim/real/{__init__,bvh,motive,markers,align,stats,
recordings,analysis}.py`; `mesqsim/cli_step3.py` (`real-stats`); `tests/test_page_frame.py`,
`tests/test_real_{readers,markers,stats,recordings}.py`.
**Changed.** `mesqsim/cli.py` (two lines at the end of `main` register Step 3's commands: a first version added them
inside `cli.py` and moved the lines that `docs/explain/CODE_EXPLAINED.md` links to, which the link test caught),
`mesqsim/validation.py` (check 49.1 at the end; Step 3 checks are numbered 39 + n for brief §10 item n),
`tests/conftest.py` (`real_results`), `pyproject.toml` (markers `real`, `node`).

**What was built.**
- Frames (`page_frame.py`): three.js -> world (x, y, z) -> (x, -z, y), +90° about X, so the avatar faces world -Y
  with its left at +X, as Danyal faced in Motive; Motive -> three.js (x, z, -y) as the brief gives; their composition
  Motive -> world is the identity (tested). BVH rotations are the product of the channels in order (three.js
  r157's `BVHLoader.js:177-187`).
- BVH (`bvh.py`): hierarchy, channels, the page's `;` provenance lines; forward kinematics in the file's frame; the
  sites of §5.1 (the head pod 10 cm along the Head bone's +Y, since the page writes End Sites as 0 0 0).
- Markers (`markers.py`): T-pose naming by height order, forward from the pelvis marker (back) to the chest marker
  (front), left/right by side; then *continuity* (a fragment starting within 0.5 s where the lost marker's velocity
  predicts, within 8 cm + 50 cm/s of gap) alternating with *templates* (longest fragments first, the free site whose
  median body coordinates about the head marker match within 25 cm; a left site stays left of its twin, a thigh above
  its shin), three times. First versions linked by body coordinates only and swapped both thighs at 34 s in
  `13-41-17` (both markers lost during a turn) and misread `13-59-24` while its pelvis marker was hidden for 27 s; the
  tracker's facing now sums every cue (pelvis -> head, pelvis -> chest, and the thighs', shins' and upper arms'
  left-right axes). Share of frames each site is seen, over the five takes: head 98-100 %, pelvis 92-100 %, shins
  83-100 %, thighs 44-100 % (the lowest `13-47-28`'s right thigh), chest 71-99 %, upper arms 70-100 %, forearms
  29-100 %; 1-16 fragments left unnamed per take (strays and ghosts); 91-99 % of moving frames head toward the chest
  side, so forward is proved in every take.
- Clocks (`align.py`): coarse lag from the wall clocks, fine lag in ±4 s maximising the leg height-change
  correlation, *and the frame interval* (32.80-33.60 ms).
- Statistics (`stats.py`, `analysis.py`) and `python -m mesqsim real-stats`, which prints the comparison and writes
  `$MESQ_OUTPUT_DIR/real_stats_g<commit>/summary.json`.

**Findings.**
- **The page of 2026-11-13 recorded every 33.0 ms although it wrote `Frame Time: 0.0333333`.** At 1/30 s no single
  lag aligns the long takes (`13-41-17`: leg r 0.30); at 33.0 ms one lag holds in every 40 s window (−0.34 to −0.41 s)
  with r 0.69-0.83. Fitted intervals: 33.00 and 33.01 ms on the long takes, 32.91-33.05 ms on the short ones. The new
  page measured 33.0 ms on Nov 14 (`setInterval(1000/30)`). With the fitted interval the two computers' clocks agree
  within 0.18-0.90 s; the brief's 1.2-3.3 s was the disagreement under the asserted frame time. BVH-only statistics
  keep the file's frame time, as the reference did; their times are 1 % long on Nov 13. For the twin: the old
  recorder asserts 1/30 s while sampling at 33.0 ms.
- "Height change" for the alignment is the height minus its centred 2 s moving mean, the drawn height taken relative
  to the drawn root (whose height is the phone's, halved, and moves every drawn site). With it the legs correlate at
  0.63-0.87 in the four well-tracked takes and negatively when left and right are swapped (the brief's check), and
  0.33-0.67 in `13-59-24`, whose single-leg swings leave one leg still.
- Nov 14: the root's travel minus facing spreads over the whole circle (mode 60-90°, median +64°, circular mean +83°).
- Nov 4: the 17.3 min of the static hold end at the first restart (LeftHand, 1044.3 s = 17.41 min; a second at
  1116.6 s); 12 of the 13 other pods drift less than 1.8° over the window, the Head 14.1° (−0.81°/min); the rest at
  most 1.68°.

**The reproduction of §5.3** (`python -m mesqsim real-stats` at `47af39b` plus these files, 9.5 s): **48 of 62
reference values within tolerance.** Reproduced: Φ1 6/6; Φ3 drawn shins 4/6, Nov 14 thighs; Φ4; Φ5 scale 5/5 and
heights 5/5; Nov 14's root height; Φ6 6/6; Φ7 5/6; Φ8 4/7; Φ9; Φ10 6/6; Φ12. The 14 differences, after comparing the
definitions (no bug found; nothing was tuned to a reference value), with their reasons in
`mesqsim/real/recordings.py:EXPLAINED_DIFFERENCES`:
- Φ3, the left shin of `13-42-39` and `13-47-28`, drawn (−75° and −54° against −63° and −40°) and true (+29° and
  +19° against +12° and +1°), and `13-44-44`'s true left shin (+10.3° against +5°, 0.3° beyond the tolerance): the
  left shin's swing is nearly round there (minor over major spread 0.70-0.87; the right shin's 0.38-0.66), so its
  axis moves with the frames counted as walking in takes with 10-19 s of it. Keeping only frames that move forward
  reproduces `13-42-39` but breaks Nov 14, whose travel is 74° off the facing; the literal definition stays.
- Φ7 Nov 14 (+63.7° against +73.7°): the spread above; the median moves with the 99 s (here) against 96 s of walking.
- Φ8 `13-41-17`, `13-59-24`, Nov 4 (15.5, 47.8, 85.6 % against 11.8, 40.8, 79 %): the brief's median over bones of
  the recorded (local) rotations. Global rotations give 0.4-46 %, far off. Pooled over bones without Hips, 6 of 7
  values match (Nov 4 79.5 %, `13-41-17` 10.5 %), all but `13-59-24` (55 %); the median over bones is kept, as the
  brief defines it. Danyal may know which was meant.
- Φ11, the upper arms (all five pairs; components within 0.10 in 4 of 10): the arms are the hardest markers to follow
  and r depends on which frames each analysis names; `13-41-17`'s left arm comes out +0.83 against −0.42 under
  heights, heights relative to the root, height changes, and either clock. Not reproduced.

**Check.** `bash scripts/check_all.sh` at `0f7673a` (clean worktree), 2026-11-18 23:27 UTC, 257 s: `ruff` clean;
`pytest` **298 passed**; validation **53 passed, 0 failed, 0 skipped of 53**; 49.1: 48 of 62 reference values within
tolerance, 14 explained differences. A first run failed one test: `test_explain_links` (the moved `cli.py` lines).

**Decisions.** Φ8 and Φ9 use each bone's recorded (local) rotation; Φ12 each pod's drawn world rotation. Φ1 uses the
standard deviation of both body coordinates together. Φ7 is positive anticlockwise seen from above (to the left of the
facing; the reference's sign agrees on all six). A jump "snaps back" if the bone returns within 5° of its pre-jump
rotation within 0.5 s (this module's choice). Recording ids `nov04`, `nov13_1341` ... `nov13_1359`, `nov14`; the
proposed split (§5.4) and suit versions are in `recordings.py` for Danyal to freeze.

**Open.** Whether the reference Φ8 was pooled; the arm tracking behind Φ11.

## 2026-11-18 · S3-M2 (the page twin) · branch `s3-m2` (stacked on `s3-m1`; nothing pushed)

**Previous milestone's commits.** `0f7673a`, `f0befde` on `s3-m1` (local).

**Plan stated.** Port the page's pose chain, smoothing, root and recorder to Python with every rule and defect; run
the real `js/custom_icm.js` and `js/bvh_converter.js` unmodified in Node as the golden reference; three scenarios in
`velvet` and `smooth`; commit the golden outputs as fixtures; reproduce the root's fixed point.

**Files added.** `mesqsim/twin/{three_math,page,page_scenarios,golden}.py`; `tools/page_golden/{run.mjs,
make_fixtures.py}`; `tests/test_page_twin.py`; `tests/fixtures/page_golden/` (six scenarios of 358-480 kB and the
avatar's node tree, 12 kB; synthetic). **Changed.** `mesqsim/validation.py` (check 40.1).

**What was built.**
- `three_math.py`: three.js r157's quaternion multiply, conjugate (its `invert`), normalise, slerp (every branch,
  including "equal: keep the old value", which is what freezes a still bone and makes Φ8's exact repeats), Euler
  'XYZ' through `Matrix4.compose`, vector lerp; and `js_number`, JavaScript's `Number::toString` for the BVH writer.
- `page.py`: `PageTwin.handle(msg, t)`, `calibrate()`, `record_start/tick/stop`, with the rules of brief §2.2:
  world delta raw · calibration⁻¹ (the ignored `bc`); the tree map; `tposeOffset`, then the mounting offset, fitted
  once at the first Set T-pose with the page's flip test; a child's local from its parent pod's stored target global
  (rig bones in between ignored; the feet store no global; Spine falls back to HipsAlt while Hips' global is the
  identity); one slerp per message with `slerpDict`; Hips from HipsAlt unless it is silent for over 1 s (whole-ms
  wall clock, as a `Date`); the root with `posswap`, × 100, `− initialPosition`, + 100 cm, lerp 0.1, the
  zero-coordinate drop, and `initialPosition` aliased to the hips at Set T-pose; the recorder's joints, OFFSETs,
  Euler angles, measured frame time and provenance lines. A tree without a bone's entry (`_HipsAlt_DISABLED`) makes
  the page throw; the twin drops the message at the same point.
- The avatar: `load_rig` reads `glbs/scene2.glb` as GLTFLoader builds it, including its renaming (each bone has a
  zero-offset copy hung under it; names are reserved depth first, so the page's `getObjectByName` finds the real
  bones and the copies become `mmHips_1` ...). Rest rotations are all identity; the offsets equal the OFFSETs in
  Danyal's BVH files (LeftArm 12.655, −0.266, −2.601 cm).
- `run.mjs`: three.js r157 and its GLTFLoader from `build-static/`, the avatar prepared as `threejsscene.js:433-470`,
  `slerpDict` evaluated from `index.html`, the tree's `meta.json`, both page scripts unmodified in a `vm` context
  with inert stubs (DOM, jQuery, toasts, moment, KalmanFilter, timers) and a scripted clock; the recorder of the
  page's inline script (`index.html:373-431`) is mirrored there.
- Scenarios (`page_scenarios.py`): pods report Y(−ψ) · S(t) · M with random norths and mountings at 16 Hz, the phone
  at 30 Hz, a Set T-pose and a 33 ms recorder: a T-pose then a left-leg swing; a 90° turn while the phone walks 1 m;
  the pelvis swaying while HipsAlt is silent from 2.0 to 4.0 s.

**Results** (`python tools/page_golden/make_fixtures.py`, Node v24.11.1, three r157, mesquite-mocap `87d7c3b`, 1.3 s;
`pytest tests/test_page_twin.py`). Six scenario-tree pairs, 6,182 messages: largest difference in a drawn quaternion
or the hips position **6.7e-16**, in a BVH number **9.2e-14** (4,949-7,709 numbers per file; last-digit differences
from V8's and macOS's maths libraries), BVH headers identical; the Node test rebuilds the fixtures identically. The
root's fixed point: a still phone at p puts the hips at 100 p + c before Set T-pose and at **(100 p + c) / 2** after
it (to 1e-9). In the HipsAlt scenario the phone's messages leave the hips alone for the first second of silence,
then drive them. Check 40.1 passes.

**Check.** `bash scripts/check_all.sh` at `bc1299e` (clean worktree), 2026-11-19 03:05 UTC, 253 s: `ruff` clean;
`pytest` **322 passed** (the Node test included); validation **54 passed, 0 failed, 0 skipped of 54**.

**Decisions.** `ideal_root_box` copies `initialPosition` as the brief says: travel is drawn 1:1, but the hips' height
becomes relative to the T-pose's (x* = 100 (p − p_T)); it takes a heading correction per pod (B1 later) applied as
Y(δ) · q before the snapshot, so at t and at the T-pose alike. `three_math.py` repeats three.js's operation order
rather than reusing `mesqsim.conventions` (same conventions), because bit-level equality decides when a bone stops.

**For Danyal.** mesquite-mocap is licensed GPL-3.0, and `mesqsim/twin/page.py` is a port of its page code (the
fixtures are its outputs). Nothing is published while Step 3 stays local, but before this repository publishes Step
3 the licence of that module needs a decision.

## 2026-11-19 · S3-M3 (bodies, sites, sensor physics) · branch `s3-m3` (stacked on `s3-m2`; nothing pushed)

**Previous milestone's commits.** `bc1299e`, `d0cf1f9` on `s3-m2` (local).

**Plan stated.** SMPL+H with each sequence's shape and gender; pod and phone sites on the skin from the wear guide;
surface frames, mountings, strap errors; ω and f from rotation and position splines; the four exact checks; the
skin-against-bone angle and the smoothing's effect on |f| on AMASS walking.

**Files added.** `mesqsim/twin/{smplh,sites,bodymotion,physics,body_report}.py`, `configs/twin.yaml`,
`tests/test_twin_body.py`. **Changed.** `mesqsim/cli_step3.py` (`twin-body`), `mesqsim/validation.py` (41.1).

**What was built.**
- `smplh.py`: SMPL+H in numpy for a few vertices (shape, pose correctives, 52-joint forward kinematics, skinning
  with each vertex's non-zero weights), all 156 pose values (Step 1's loader keeps 66). Against `smplx.lbs.lbs` on
  `Walking_3_poses.npz`: vertices within 4.6e-9 m, joints within 8.3e-9 m.
- `sites.py`: each site is a ray in the rest pose from a point on the bone (the wear guide's fractions: upper arm
  0.41, forearm 0.39, thigh 0.69, shin 0.53) toward the front, the back or up; the vertex skinned to the segment that
  lies outermost on it. Chosen once on the neutral template (`SITE_VERTEX`; SMPL+H numbers vertices alike in every
  gender and shape). The head's lift (7 cm) and the feet's fraction (0.65) were set so the normal tilts as the wear
  guide's pod face: head 11-13° up (guide 20°), feet 59-61° (guide 70°); thighs and shins tilt down 9-17° (guide
  15°, 10°) unprompted. Mountings: chip +Z out of the skin everywhere; identity, except a half turn about the normal
  on the thighs and shins (case up, bone down); the phone (b̂, t̂, −n̂). Step 1's `NOMINAL_MOUNT` points the leg
  pods' x down the bone; the twin follows the wear guide.
- `bodymotion.py`: per site, F · M, the position and the segment's rotation; 0.23 s for 27 s of AMASS at 120 Hz.
- `physics.py`: `RotationSpline`'s first derivative is the body-frame rate (scipy 1.17.1; pinned by a test), so it
  is ω directly; positions through a smoothing spline with λ = 1 / (h (2π f_c)⁴), whose measured response is
  1 / (1 + (f / f_c)⁴) to three decimals.

**Results** (`python -m mesqsim twin-body --files 12`: the first 12 walking files of the training split, 116 s, IMU
at 225 Hz, 14 s). Skin against bone, median / P95 / max: head 0.38 / 1.17 / 1.86°, chest 1.18 / 4.22 / 6.30°,
lower back 2.19 / 3.77 / 4.46°, arms and forearms 0.67-0.93 / 1.63-2.03 / 2.70-3.19°, hands 0.79-0.80 / 1.98-2.06 /
4.70-6.13°, thighs 1.63-1.72 / 3.89-3.91 / 4.41-4.93°, shins 0.80-0.96 / 1.53-2.28 / 2.00-3.66°, feet 0.85-1.11 /
1.77-2.03 / 2.49-2.80°, the phone's belt site 3.42 / 5.84 / 9.12°. Specific force |f|, P99: unsmoothed 28-73 m/s²
(motion-capture jitter, twice differentiated), at 16 Hz 14-26, **at 8 Hz 13.7-24.8** (the feet highest), at 4 Hz
12.8-21.9; medians stay at g (9.6-10.9 at 8 Hz). Most of the jitter lies above 16 Hz and most of the gait below 8 Hz,
so 8 Hz stays (`assumption`). At 8 Hz no walking sample exceeds the accelerometer's ±4 g on any axis, and at most
0.004 % the gyroscope's ±2000 °/s (the right hand). Check 41.1: still pod 4.4e-14, spin 3.0e-12 rad/s, free fall
4.7e-8 m/s², circle 1.6e-6 relative, amplitude 0.500 at the cutoff.

**A bug the tests caught.** The first smoothing formula put h on the wrong side (λ too small by h², so "8 Hz" did not
smooth); the response test failed, and a first report claimed 0.2-2 % accelerometer saturation, which was that bug.
The head's surface axis was first the neck-to-head bone, which turns with the neck: 6.3° median, 22° P95 against the
head; it is now the head joint's own up.

**Check.** `bash scripts/check_all.sh` at `49c3587` (clean worktree), 2026-11-19 03:22 UTC, 256 s: `ruff` clean;
`pytest` **336 passed**; validation **55 passed, 0 failed, 0 skipped of 55**.

**Decisions.** The twin's parameters live in `configs/twin.yaml` (`twin:`, and an empty `learning:` for Danyal), so
`configs/default.yaml` and the config hashes that name Step 1's and Step 2's runs are untouched. The IMU rate
(225 Hz) is an assumption until S3-M4 reads the firmware.

## 2026-11-20 · S3-M4 (sensor errors, fusion, wire, phone and app) · branch `s3-m4` (stacked on `s3-m3`; nothing pushed)

**Previous milestone's commits.** `49c3587`, `0adda78` on `s3-m3` (local).

**Plan stated.** Sensor errors (§4.4); VQF six-axis, Mahony and Step 1's analytic mode in the loop, with the twist
rule for labels (§4.5); the binary and the old JSON wire (§4.6); ARKit-like reports, the app ported from its Swift
core, the link and the phone's labels (§4.7); checks 42.1-45.1 (§10 items 3-6).

**Files added.** `mesqsim/twin/{config,labels,sensors,fusion,fusion_report,wire,phone_app,app_golden,phone,checks}.py`;
`tools/app_golden/{main.swift,make_fixtures.py}`; `tests/fixtures/app_golden/app_golden.json` (1,263 kB);
`tests/fixtures/page_golden/phone_zero_guard_{velvet,smooth}.json` (311, 312 kB);
`tests/test_twin_{fusion,wire,phone}.py`. **Changed.** `configs/twin.yaml` (`imu.oversample` replaces `imu.rate_hz`;
`sensors`, `fusion`, `wire`, `phone`, `app`), `mesqsim/twin/{sites,page_frame,page_scenarios,body_report}.py`,
`mesqsim/cli_step3.py` (`twin-fusion`), `mesqsim/validation.py` (40.1's text; 42.1-45.1), `pyproject.toml` (marker
`swift`), `tests/test_twin_body.py`, `tests/test_page_twin.py`.

**What was built.**
- `sensors.py`: ω̃ = (I + S_g) A_g ω + b_g(t) + n_g, f̃ likewise, then 16-bit rounding and clipping at ±2000 °/s and
  ±4 g. The ideal gyroscope sample is the mean rate over its interval, Log(R_{k-1}⁻¹ R_k) / Δt, so a filter that
  integrates exponentially follows the truth exactly when nothing is wrong. Labels: N_g 0.015 °/s/√Hz and S_g ±1.5 %
  `datasheet` (as the Step 1 brief quoted DS-000189; the PDF itself still could not be fetched); everything else
  `assumption`, with b_g,0 (per axis σ 0.2 °/min) and B_g (0) to be fitted in S3-M6.
- `fusion.py`: VQF (`updateBatch`, w-first converted to x-y-z-w), Mahony (predict then correct, exact exponential
  steps), and the analytic mode, which is Step 1's own `PodHeading` and Gauss-Markov η. The filter runs at the chip's
  own rate (`twin.imu.oversample` = 1, `firmware`: `Pod_Watch_Binary.ino:482-490` asks for a Quat6 every DMP cycle,
  about 55 Hz measured in Phase 1); 4 is the sweep.
- `labels.py`: E = q R⁻¹, ψ = -wrap(2 atan2(E_z, E_w)), the tilt as the swing left after the twist; δ and ε.
- `wire.py`: Step 1's frame and the parser's message (`mesq_parser.js:101-117`, its `BONE_NAMES` read back from the
  clone in a test); the old dongle's line (`Dongle__JSON.ino:59`), numbers at `json_decimals`, half away from zero,
  "-0.00" kept.
- `phone_app.py`: the app's `Core` ported line by line (`Geometry`, `HipsFrame`, `PageTree.wire`, `HubMessage`,
  `PosePipeline`, `LeverArmFit`). `tools/app_golden` compiles the unmodified Swift files (`swiftc -O -swift-version 5`,
  Swift 6.3.3) with a harness that passes every double as its bit pattern: 1,260 pipeline frames (the automatic Zero,
  a requested Zero, a Zero refused with the camera tipped down, one waiting out limited tracking, all four tracking
  policies, a link refusing sends, the sign flip, three trees and three facings), 2,110 encoded poses, 410 Shepperd
  conversions and 4 lever-arm fits. **The port matches to the bit: 0 differences, 0 ulps.** MATH.md §3-§8 and the
  Swift code agreed everywhere I compared them, so the fidelity ledger has no "Swift wins" row yet.
- `phone.py`: ARKit's report R̂_C = Y(-ψ_A) R_C Exp(η), its path integrated in its own drifting frame, ARKit's north
  set by the camera's view at its start (not drawn), Poisson episodes (limited: orientation only; relocalising:
  nothing, then a jump), float32 at the boundary, the app, a link with two sends in flight and latest pose wins,
  and the labels. The message is q_m = Y(-ψ_ph) R_C Exp(η) K, with K = R̂_C(T0)⁻¹ Y(ψ0 [- π for thorny]) read from
  the app's Zero, so ψ_ph = ψ0 + ψ_A: **the app's Zero cancels ARKit's own north** (a test turns ARKit's start and
  ψ_ph does not move). `phone_tables` writes `phone.parquet` and `labels_phone.parquet` as §4.7 lists them.
- The page golden test gained `phone_zero_guard` (positions rounded plainly to 1 mm, 27 of 90 holding an exact 0 or
  -0): the real page leaves the hips where they are for each, and the page twin agrees to 4.4e-16. The six earlier
  fixtures regenerated byte for byte.

**Results.**
- Checks: 42.1 VQF and Mahony keep ψ within 3.6e-13° with perfect samples; a 0.5 °/s vertical bias gives
  ψ(60 s) = -29.991° against -b t = -29.991°; 1 % scale over three turns gives -10.800° against -10.8°; Step 1's
  reports through the twist rule return Step 1's ψ within 8.9e-16 rad. 43.1 hand check 12.000000000000°. 44.1 the
  golden frame and message exact; JSON within half a step. 45.1 as above, 10,000 random poses, the longest exactly
  95 bytes, the zeros nudged.
- `python -m mesqsim twin-fusion --files 12` (116 s of walking per site, filters at 55 Hz, 7 s): with perfect
  sensors VQF's tilt error is 0.91-1.08° median, 1.3-5.6° P95 (the hands highest), and ψ moves at most 0.57°
  (≤ 0.16° per turn about the vertical); Mahony 1.35-1.90° median, 2.0-5.0° P95, at most 1.7°. With the drawn
  errors both filters change ψ by a median 0.9-2.7° per turn, at most 6.2°. A decomposition on four files puts
  nearly all of it on the gyroscope's scale error times the turning (3.0° median at the end; misalignment 0.10°,
  noise and rounding 0.08°, b_g,0 0.09°, the accelerometer 0.08°): the brief's k Θ term, and these clips turn about
  once each.
- Bias removal (a still pod, (0.2, -0.3, 0.5) °/s): VQF's rest estimation has 94 % of the vertical component after
  2 s, 99 % after 5 s, and ψ ends at -0.84° after 120 s instead of -60°. Turning gently (±30°, 0.1 Hz) it never
  detects rest and removes nothing of the vertical; Mahony's integral, and VQF's motion estimation, never see it
  (six axes cannot), only the horizontal components. So in the twin, vertical bias is removed only when a pod
  rests.
- Sign reconciliation: q = Y(-ψ) R gives ψ = -b t for a bias b and ψ = -k Θ for a scale error k, so Step 1's
  ψ = a + b_1 t + k_1 Θ has b_1 = -b and k_1 = -k; Step 1 drew both symmetric about 0, so nothing it reported
  changes.

**Bugs the work caught.**
- The phone's mounting in S3-M3 was UIKit's device frame (x right, y up). ARKit's camera frame is +X toward the Home
  button, +Y "up" in landscape, +Z out of the screen: worn upright with the screen inward, (-t̂, b̂, -n̂). The app's
  orientation does not depend on it (the mounting cancels in q_R), its position does, through the lever arm.
- Mahony first compared the accelerometer at t_k with the estimate from t_(k-1): 2.7e-6 rad of heading drift with
  perfect samples. It now predicts with the gyroscope, then corrects.
- The first walking report started each filter mid-stride, aligned to an accelerometer reading with walking in it:
  VQF's start-up then wandered by up to 39° in tilt and 7° in heading. Pods boot still in their box, so the report
  now gives each filter 3 s still first.
- Mahony's integral gain k_P/10 (written as "the usual order", which was not a source) let walking's persistent
  acceleration error build a false bias: up to 40° of heading in 27 s with perfect sensors. It is now k_P/100, a
  100 s bias time constant, VQF's own bias forgetting time; that gives 1.7° at most.
- The first fusion report expressed heading change in °/min, which on short clips that turn about once read as 8-24
  °/min; it is now per turn.
- Two app-golden scenarios turned the camera about its own Y (the body's right) instead of the body's vertical.

**Check.** `bash scripts/check_all.sh` at `c78a064` (clean worktree), 2026-11-21 02:21 UTC, 274 s: `ruff` clean;
`pytest` **385 passed**; validation **59 passed, 0 failed, 0 skipped of 59**.

**Decisions.**
- The DMP fuses at its output rate (`oversample` 1); `body_report.py` keeps 225 Hz only as its evaluation rate.
- `json_loop_ms` was dropped: the old pod's loop checks every tick, so its sends lie on a 32 ms grid.
- Check 43.1's "δ and ε recomputed from ψ match the tables" needs the twin's session tables (S3-M5); check 44.1's
  "Nov 4's exact-repeat share with the fitted decimals" needs the fit (S3-M6). Both checks say so in their text.
- §4.4's mirrored-pair hypothesis for Nov 13's outward leg drift is S3-M6's comparison (the wear guide mounts both
  legs alike, so it needs a different mounting).

**For Danyal.** DS-000189 rev 1.3 (the ICM-20948 datasheet) could not be fetched (TDK's link serves a landing page,
SparkFun's copy refuses). A copy of the PDF would turn the accelerometer's noise, scale, misalignment and bias, now
`assumption`, into `datasheet` values.

**Commands.** `python -m pytest -q` (data present): **385 passed** in 139 s. `python -m mesqsim twin-fusion --files
12`. `python tools/app_golden/make_fixtures.py` (Swift 6.3.3); `python tools/page_golden/make_fixtures.py` (Node
24.11.1). The mesquite-mocap clone stays clean (`git status` empty after both builds).

## 2026-11-21 · S3-M5 (procedure, slips, programs, long sessions, one twin session) · branch `s3-m5` (stacked on `s3-m4`; nothing pushed)

**Previous milestone's commits.** `c78a064`, `ab1cbb4` on `s3-m4` (local).

**Plan stated.** Long sessions joined from one subject's clips (§4.9) and the protocol program (§4.8); the
procedure with its slips (§4.8); one command for one twin session end to end, with the files and labels of §4.11;
checks 40.2 (the BVH in the page's loader), 43.2 (δ and ε from the tables), 46.1 (§4.12), 47.1 (joins), 48.1
(determinism).

**Files added.** `mesqsim/twin/{program,session,session_checks}.py`, `tools/page_golden/load_bvh.mjs`,
`tests/test_twin_session.py`. **Changed.** `configs/twin.yaml` (`program`, `procedure`, `page`),
`mesqsim/twin/phone.py` (per-frame ψ_ph; a turn hook for §4.12), `mesqsim/cli_step3.py` (`twin-session`),
`mesqsim/validation.py` (40.2, 43.2, 46.1, 47.1, 48.1).

**What was built.**
- `program.py`: clips resampled to 60 Hz (slerp per joint), each next piece turned about the vertical through its
  first pelvis point and shifted so its root continues the last one, a 0.5 s smoothstep blend labelled
  `transition`; one subject per session, from its own split. Synthetic pieces on the subject's own poses: T-pose,
  still, hip twists, leg swings forward and to the side, arm raises forward and to the side, a turn on the spot.
  `random_program` draws one subject's clips up to a drawn length with still stands and T-poses between.
- The protocol program (§4.8) on **KIT subject 3** (training split, the one subject whose clips cover most of it):
  his `walk_6m_straight_line` cut at 4 m, his `turn_left` U-turns, his squats; synthetic T-poses, twists, swings,
  raises and the two 360° turns on the spot. 29 pieces, 189.9 s. Every walk leg lies on one line (|x| < 0.2 m).
- `session.py`: the procedure (box, strapping with Step 1's carry path and the position by smoothstep, the body
  holding its first pose until the program starts, the app's Start at -20 s, its automatic Zero, the requested Zero
  at +1 s in the T-pose, Set T-pose at +2.5 s, recording from +4 s), the posture error on the first T-pose, the four
  slips, per pod Step 1's timing per boot epoch, the sensors and the fusion on the chip's own samples, the wire,
  Step 1's radio, the phone and its app, the page twin on every message in arrival order, and the files: Step 1's
  `frames.parquet`, `session.json`, `labels_ticks` (ψ, tilt, ψ_ph, ε, the truth), `labels_joints` (δ and Step 1's
  `delta_offset_rad`, so Step 2's tools read it), `labels_steps` (per 32 Hz step the true segment orientations, the
  motion type, `transition`, the slips, every ψ), `phone.parquet`, `labels_phone.parquet`, `meta.json`,
  `page/twin.bvh`. `python -m mesqsim twin-session --seed 1`: the protocol session (35 s of procedure, 225 s in
  all) in 8.9 s, 105,140 frames (2,209 lost), 6,264 phone messages, 23 MB on disk (`labels_ticks` 11.6 MB).
- Step 2's own tools read these files unchanged: on the protocol session B1's median relative-heading error is
  5.2° and GEO-ID's 3.5° (P90 17.2° and 13.4°), a first look ahead to S3-M9.

**Results.**
- 47.1 joins: protocol 2.47 rad/s at most on a blend against a P99 of 2.88 (28 joins); a one-minute random program
  2.71 against 2.88 (8 joins).
- 46.1 (§4.12): turning every pod and the phone by Y(0.7 rad) shifts every ψ and ψ_ph by -β within 8.9e-16 and
  leaves ε and δ within 1.8e-15 rad; turning the left shin alone by Y(-0.4 rad) moves only its ψ and ε by -γ and the
  δ of l_knee (it is the child: -γ) and l_ankle (the parent: +γ), within 1.3e-15; moving the room changes nothing
  (the phone's lines are identical: ARKit's origin is where it started). B1 and GEO-ID move by 0.0035° and 0.0067°
  under the common turn.
- 43.2: ε and δ recomputed from the stored ψ equal the stored ones exactly.
- 48.1: seed 3 twice, 9 files, identical bytes.
- 40.2: see below.
- All four modes run end to end (VQF, Mahony, Step 1's analytic, the old JSON wire with its 32 ms grid, and an
  oversample of 4).

**Findings in the page, for Danyal (mesquite-mocap `87d7c3b`).**
- `build-static/BVHLoader.js`, which `player.html` loads (`js/threejssceneplayer.js:6, :332`), cannot be
  constructed on the page's own three.js r157: it is a pre-class loader calling `Loader.call(this, manager)`, and
  r157's `Loader` is a class ("Class constructor Loader cannot be invoked without 'new'").
- The page's own BVH files cannot be read by that loader even so: `js/bvh_converter.js` writes three `;` provenance
  lines after `Frame Time:`, BVH has no comment syntax, and the loader reads them as frames and runs out of numbers.
  Danyal's Nov 14 file fails exactly so; the older page's Nov 13 files have no such lines and parse.
- Check 40.2 therefore asks for what fidelity means here: the twin's BVH fails in the page's loader exactly as the
  real Nov 14 file does (constructor, then the same parse error at the same lines), and with the `;` lines removed
  both parse completely with the page's own parser (27 bones, one keyframe per frame, every value finite).

**Bugs the work caught.**
- The first join matched the pelvis's instantaneous facing; the pelvis yaws about ±15° within a stride, so the
  protocol's back-and-forth walk fanned out by 15-17° a leg (6.2 m wide). The facing is now averaged over 1 s
  (`heading_window_s`), and a walk can name its travel direction (the protocol's legs alternate on one line).
- KIT subject 3's `turn_left` clips are U-turns that wander 1.2 m, not turns on the spot; the protocol's two 360°
  turns are synthetic.
- A test assumed seed 3 had no slips; it draws "Zero at Start", which is right.
- A test bounded the box-phase tilt by 1°; the accelerometer's assumed misalignment allows up to 1.7°.

**Decisions.**
- A session's body holds the program's first pose (a T-pose) from Start until the program begins; the pods are
  carried to it during the strapping.
- The phone is on the belt from Start; ARKit starts with the app, 20 s before the T-pose (`assumption`).
- Slip probabilities (`twin.procedure.p_*`) are `assumption` until S3-M6 fits what the recordings allow (weakly).
- 48.1's "identical dataset manifest" comes with the dataset (S3-M7).

**Commands.** `python -m mesqsim twin-session --seed 1` (twice, `shasum` of every file: identical); `node
tools/page_golden/load_bvh.mjs` on the twin BVH and on the Nov 13 and Nov 14 BVHs; `pytest tests/test_twin_session.py`
(13 passed, 46 s); `python -m pytest -q` (data present): **403 passed** in 194 s.
