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
