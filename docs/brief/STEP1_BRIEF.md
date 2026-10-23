# Master prompt: Mesquite ML, Step 1 (data, environment, simulator)

*For Danyal's collaborator. Open Claude Code at the root of the experiments repository and paste everything below the
line as the first message. Claude will first save it in the repository as `docs/brief/STEP1_BRIEF.md`.*

---

You are working in a **private** Git repository for a research project led by **Danyal Khorami** (the project owner) on
**Mesquite**, a low-cost inertial motion-capture suit. A collaborator is running this session with you. You have no
access to Danyal's other repositories or files. Everything you need is in this brief. Where it is silent, make a
choice, write it down and test it. Where only a person can act, for example registering for a dataset, or where a
decision would change the research, stop and ask the collaborator.

**Your task is Step 1 only.** Build a trustworthy environment that later steps can rely on:

- the datasets, obtained legally and verified;
- a pinned Python environment;
- a simulator that turns real human motion into the data the Mesquite suit would send, with exact labels;
- the decoding step shared with real recordings;
- an evaluation harness with the four simplest baselines;
- tests that prove each part is accurate;
- documentation that lets Danyal understand, run, check and manage all of it;
- a record of everything you add and push.

Do **not** build the geometric estimator or any learned model. Those are Step 2.

## 0. Rules for this repository

1. **First commit.** Save this brief, unchanged, as `docs/brief/STEP1_BRIEF.md`. Write `CLAUDE.md` with these rules,
   so that later sessions follow them.
2. **Keep it private.** The repository and its results stay private until Danyal says otherwise; a disclosure question
   is still open.
3. **No data in git.** Never commit datasets, body-model files, downloaded archives, generated datasets, model outputs
   or credentials. The AMASS and SMPL+H licences forbid redistribution.
   - Keep data in `data/` and outputs in `outputs/`, both in `.gitignore`; an environment variable may point elsewhere.
   - Commit instead: download instructions and scripts, a manifest of every data file with its size and SHA-256, and
     small *synthetic* test fixtures.
   - Keep every committed file under 5 MB.
4. **Work in milestones (§7).** For each one:
   - create a branch and implement;
   - run the full check (§9);
   - update the docs and `WORKLOG.md`;
   - commit, push, and open a pull request into `main` stating what changed and the check output.

   Merge only when the collaborator says Danyal approved. Never force-push `main`.
5. **Never invent data, results or numbers.** Every number in the documentation comes either from a command you ran (say
   which) or from a cited source. Label every simulator parameter **datasheet**, **firmware**, **assumption** or
   **measured**.
6. **Tests run without licensed data.** Unit and property tests use synthetic motion. Tests that need AMASS or DIODEM
   are marked and skip cleanly when the data are absent.
7. **One convention per quantity (§4.1), tested.** Most bugs in this project will be sign, order or frame errors.
8. **Talk to the collaborator.** Before each milestone, say in plain words what you will do. After it, say what you
   did and how you checked it.
9. **Stop at the end of Step 1, or when blocked.** Then write the report in §11.

## 1. The whole project in one picture

*Put a version of this at the top of `README.md`. Danyal wants every overview to open with a picture that can be replayed
with closed eyes: the purpose, the steps, the pipeline, and what goes in and comes out.*

**The suit.** A person wears fifteen watches, on the pelvis, chest, head, upper arms, forearms, hands, thighs, shins and
feet. Inside each watch a motion chip fuses its gyroscope and accelerometer into an orientation. Gravity tells every
chip which way is down, so tilt is right. There is no compass, though. Each chip invents its own private "north" when
it switches on, and that north slowly slides: a little every minute, and a little more every time the watch spins
around.

**The data path.** Thirty-two times a second, each watch radios a 16-byte postcard holding:
- its orientation;
- a postcard number;
- the time its picture was taken.

A hub passes the postcards to a laptop, which turns each watch into a bone of an avatar.

**The problem.** The norths disagree. So a joint computed from two watches shows a false twist about the vertical: the
thigh twists against the shin when the knee only bent. Later steps build programs that estimate, for every joint, how
far the child watch's north has slid relative to its parent's. They use body geometry, a small neural network, or
both.

**Step 1, your job, builds the ground those programs stand on:**
1. Take real human motion from a public motion-capture database (AMASS).
2. Compute how each of the fifteen body segments was oriented at every instant.
3. Put virtual watches on the segments. Give each its own sliding north, its own clock and its own radio losses, as the
   real hardware has.
4. Write out the postcards the real suit would have sent. Because we add the errors ourselves, we know the right answer
   for every joint at every instant.
5. Decode the postcards with the same code that will decode the real suit's recordings.
6. Measure how wrong the joints are with no correction and with the simplest corrections.

A second public dataset (DIODEM) adds a reality check: real sensors on mechanical chains, with cameras recording the
truth.

```text
AMASS sequence (SMPL+H poses) ──► 15 segment orientations s_i(t)
   + random facing, mounting m_i, session timeline (box → strapping → T-pose → idle → motion)
   ──► each pod's chip samples at its own rate f_i (50.1–60 Hz)         [labels: true ψ_i(t)]
   ──► reported q_i = Y(−ψ_i)·s_i·m_i·Exp(η_i), with w ≥ 0
   ──► 32 Hz send grid: newest sample, held repeats, count, ms_lo, int16
   ──► radio loss in bursts; host arrival times
   ──► frames table (the same schema a real logger writes)   +   labels table (kept separate)
STEP 0, shared with real recordings: unwrap ms_lo; fresh, held or lost; norm check; box registration; T-pose
   ──► per-pod series ──► baselines B0, B1, O1, O2 ──► error against the labels ──► tables and figures
DIODEM: download → load → drop the magnetometer → six-axis VQF → relative heading against optical truth (sanity check)
```

## 2. The research question (context only; Step 1 does not test it)

Using only the fused orientations Mesquite already transmits, where (at which joints and in which motions) does a small
causal temporal network recover relative heading better than explicit joint geometry? And does a combination of the
two, each weighted by its own calibrated confidence, beat both?

**The hard limit.** Rotate the whole room, with the wearer in it, about the vertical, and every watch sends the same
numbers. So nothing computed from the watches can tell how the body is turned in the room. Only the headings of the
watches *relative to one another* can be estimated. Never claim to measure north or the body's heading.

## 3. The Mesquite facts you need

### 3.1 Pods and segments

| id | name | segment | rendered |
|---|---|---|---|
| 0 | Head | head | yes |
| 1 | Spine | chest (upper trunk) | yes |
| 2 | HipsAlt | pelvis (the root) | yes |
| 3, 4, 5 | LeftArm, LeftForeArm, LeftHand | left upper arm, forearm, hand | yes |
| 6, 7, 8 | RightArm, RightForeArm, RightHand | right upper arm, forearm, hand | yes |
| 9, 10, 11 | LeftUpLeg, LeftLeg, LeftFoot | left thigh, shin, foot | yes |
| 12, 13, 14 | RightUpLeg, RightLeg, RightFoot | right thigh, shin, foot | yes |
| 15, 16 | LeftShoulder, RightShoulder | shoulder blades | no: simulate only the 15 rendered pods |

### 3.2 The wire frame (16 bytes, little-endian)

| offset | size | field | meaning |
|---|---|---|---|
| 0, 1 | 1 + 1 | sync | `0xAA 0x55` |
| 2 | 1 | id | pod id 0–16 |
| 3 | 1 | batt | battery 0–100 % (clamped) |
| 4, 6, 8, 10 | 2 each | qx, qy, qz, qw | int16 = trunc(float32(v) × 32767), saturated to ±32767 (never −32768) |
| 12 | 2 | count | uint16 packet counter, +1 per *send*, wraps 65535 → 0 |
| 14 | 2 | ms_lo | uint16 = (pod milliseconds when the sample was *decoded*) mod 65,536 |

- **Byte order.** Note that the quaternion is sent x, y, z, w (scalar last).
- **Sign.** The chip gives x, y and z, and the pod rebuilds w = √(1 − x² − y² − z²), so on the wire **w ≥ 0** always.
- **Golden vector**, which your codec must reproduce byte for byte: id 2, batt 87, w 0.75, x 0.5, y −0.5, z 0.25, count
  0x1234, ms_lo 0xABCD gives `aa550257ff3f01c0ff1fff5f3412cdab`. (That quaternion is not unit length; it tests the codec
  only.)
- **Reference.** The public firmware repository `github.com/Danyal-0-1/mesquite-mocap` defines this frame in
  `Device code/Pod_Watch_Binary/mesq_packet.h`. It is optional to read.

### 3.3 Timing

- **Chip rate.** Each pod's chip produces a fused sample at nominally 55 Hz: 1.1 kHz / (1 + 19). Because of its clock
  tolerance the real rate lies anywhere in **50.1–60.0 Hz** per part, plus about ±1 % over temperature.
- **Send grid.** The pod sends on a fixed grid of **31,250 µs (32 Hz)** on its own clock. Its phase after boot is
  `id × 31,250 / 17` µs. Pods boot at slightly different moments, so in practice the phases are scattered.
- **Fresh, discarded and held.** At each send the pod takes the *newest* sample. Older unsent samples are discarded:
  about 42 % at 55 Hz, 36 % at 50.1 Hz, 47 % at 60 Hz. If no new sample arrived since the last send, it resends the
  last one, with the same `ms_lo`. That is a **held** frame. Since the chip is faster than the grid, held frames come
  only from stalls of the sample task: unmeasured, so model them as an optional process that is off by default.
- **Decode time.** `ms_lo` is the decode time, about 0–11 ms after the true sample time (an assumption from the
  firmware's polling). It wraps every 65.536 s. Every pod's clock has its own epoch.
- **Two different gaps.** A repeated `ms_lo` means a held sample. A gap in `count` means frames lost on the radio.
- **Arrival time.** Host arrival time is kept but is never used as sample time.
- **Reboot.** After a reboot a pod is silent for a few seconds, then `count` and `ms_lo` restart near 0 and its north is
  re-chosen.

### 3.4 The tree: 14 joints, parent → child

| joint | parent → child | type |
|---|---|---|
| lower back | HipsAlt → Spine | 3-DoF |
| neck | Spine → Head | 3-DoF |
| L/R shoulder | Spine → Left/RightArm | 3-DoF |
| L/R elbow | Left/RightArm → Left/RightForeArm | ≈2-DoF (flexion, forearm rotation) |
| L/R wrist | Left/RightForeArm → Left/RightHand | ≈2-DoF |
| L/R hip | HipsAlt → Left/RightUpLeg | 3-DoF |
| L/R knee | Left/RightUpLeg → Left/RightLeg | hinge |
| L/R ankle | Left/RightLeg → Left/RightFoot | ≈2-DoF |

### 3.5 A session on the real suit

The order of a session matters, because errors grow with time since boot:
1. All pods are switched on together in a box, with their axes aligned. The box record is a few seconds of frames there.
2. The pods are taken out and strapped on.
3. The wearer stands in a T-pose (arms horizontal). It is the calibration snapshot.
4. Optionally, the wearer raises one arm forward.
5. Motion.

## 4. The mathematics

### 4.1 Conventions (one module owns them; tests pin them)

- **Quaternions** are stored as `(x, y, z, w)`, scalar last, as in `scipy.spatial.transform.Rotation`. Use the Hamilton
  product. `q` and `−q` are the same rotation. Before quantising, canonicalise to `w ≥ 0`.
- **Orientation meaning.** An orientation R maps body coordinates to world coordinates (v_world = R·v_body). Composition
  reads right to left.
- **World frame.** It is gravity-aligned with **+Z up** and an arbitrary but fixed heading. Convert every dataset into
  it at load time, and test the conversion.
- **Y(α)** is the rotation by α about +Z, right-handed: Y(90°) maps +x to +y.
- **Heading** of R is the first angle of the intrinsic Z-Y-X (yaw-pitch-roll) decomposition. Test the property
  heading(Y(α)·R) = wrap(heading(R) + α).
- **wrap(x)** returns x in (−180°, 180°]. Use circular statistics for every heading quantity.
- **Units.** Use radians inside the code and degrees in reports. Name each unit in the variable name or the docstring.

### 4.2 The measurement model

```text
s_i(t)   true orientation of segment i (segment → world)
m_i      mounting of pod i on its segment (pod → segment), constant
η_i(t)   small tilt-and-noise error of the chip's fusion

what pod i reports:   q_i(t) = Y(−ψ_i(t)) · s_i(t) · m_i · Exp(η_i(t))

heading of pod i's private frame:
   ψ_i(t) = a_i + b_i·t + k_i·Θ_i(t) + w_i(t)    (+ a jump at each reboot of pod i)
      a_i    offset at boot
      b_i    residual bias drift (deg/min)
      k_i    gyroscope scale-factor error
      Θ_i(t) signed rotation of pod i about the vertical, accumulated since boot
             (integrate the vertical component of its angular velocity)
      w_i(t) angle random walk
```

**Plain reading.** Each report is the segment's true orientation. The strap carries it on the right; a rotation about
the vertical that belongs to that pod alone carries it on the left. Drift, spinning and reboots change the left
rotation.

### 4.3 Error-model values

Source: TDK InvenSense ICM-20948 datasheet DS-000189 rev 1.3, Tables 1 and 5. Datasheet values are typical, so they are
planning bounds, not measurements of these pods.

| term | value for Step 1 | label |
|---|---|---|
| angle random walk w_i | noise density 0.015 °/s/√Hz → σ = 0.367° after 10 min, 0.90° after 1 h for one pod (√2 × for a pair) | datasheet |
| scale factor k_i | uniform within ±1.5 % (25 °C); ±3 % for a stress setting. One full turn gives up to 5.4° (10.8°) | datasheet |
| residual drift b_i | unknown. The chip must remove most of its gyro bias (an uncorrected initial bias of up to 5 °/s would drift hundreds of degrees per minute), and what remains is undocumented. Sweep σ_b ∈ {0.05, 0.2, 1.0} °/min | assumption |
| initial offsets a_i | **two modes**, because which one is true is unmeasured: *aligned*, where the chip sets north from its own heading at start-up, so pods booted together agree up to the box-slot error (σ_box ∈ {0.5, 2, 5}°); and *arbitrary*, uniform in (−180°, 180°] | assumption |
| reboot | default none; stress setting 1 per pod per hour, about 3 s of silence, then a new north set at that moment | assumption |
| tilt error η_i | about 0.5° per axis | assumption |
| chip sample rate f_i | uniform 50.1–60.0 Hz per pod | datasheet |
| decode delay | uniform 0–11 ms | assumption |
| radio loss | two-state (Gilbert–Elliott) bursts; sweep mean loss {0.5, 2, 5} % and mean burst {1, 3, 10} frames | assumption |
| sample-task stalls (held frames) | default off; optional stall of S ms every 3 s, S swept | assumption |
| mounting m_i, T-pose posture error, facing error | random within stated ranges (for example ±20°, 5° per segment, ±15°) | assumption |
| quantisation | int16 / 32767, truncated: step about 0.0035°, negligible | firmware |

Every *assumption* row will be replaced by Danyal's bench measurements later: a 70-minute box evening and a session with
pods taped to one board. Keep all of them in one config file, with their label and a source note.

**Making drift visible.** AMASS clips last seconds to minutes, so give each simulated session an **idle interval** of
0–30 min (assumption) between the T-pose and the motion. Sample the error accumulated over that interval from the same
model: drift b_i·Δt, random walk N(0, N_g²·Δt), and spin k_i·ΔΘ_i, with an assumed amount of turning.
Concatenating sequences of one subject is an optional later variant.

### 4.4 Target and correction

```text
target for joint (parent p, child c):   δ_pc(t) = wrap(ψ_c(t) − ψ_p(t))
correction, outward from the pelvis:    ĉ_root = 0,   ĉ_c = ĉ_p + δ̂_pc,   q̃_c = Y(ĉ_c) · q_c
with exact estimates every corrected pod becomes  Y(−ψ_root) · s_i · m_i : one shared, unknown heading
```

**Hand-checkable example; make it a unit test.** Take a parent at the root with ψ_p = 12° and a child with ψ_c = 5°.
1. δ_pc = −7°.
2. The correction is ĉ_c = −7°, so q̃_c = Y(−7°)·Y(−5°)·s_c·m_c = Y(−12°)·s_c·m_c. The child now shares its parent's
   north.
3. A sign error leaves a **14°** twist instead of 0°. Test for it.

### 4.5 What cannot be known; turn it into tests

- Rotate every pod's report by one common Y(β). Every label δ and every baseline output must be unchanged: exactly for
  labels, within 0.5° for estimators.
- Rotate one pod's frame by a constant Y(γ). Only the targets of joints touching that pod may change, each by exactly
  ±γ.
- A correction acts about the vertical only, so tilt must not change. Test it.

### 4.6 Step 0: deterministic repairs, shared by simulated and real data

1. **Time.** Unwrap `ms_lo` per pod, using arrival order and `count`. Mark held frames (repeated `ms_lo`), lost frames
   (`count` gaps, with wrap) and reboots (`count` and `ms_lo` jumping back).
2. **Integrity.** Decode int16/32767. Reject a frame whose |‖q‖ − 1| exceeds a documented tolerance. Canonicalise the
   sign.
3. **Box registration.** In the box all pods are aligned, so the twist of each pod's box record about the vertical
   estimates its initial heading up to the box-slot error. Only the differences between pods matter.
4. **T-pose snapshot**, as the suit does it: the world deltas are Δ_i(t) = q_i(t)·q_i(T)⁻¹.
5. Facing from an arm raise is Step 2. Leave it out.

### 4.7 Metrics and the Step 1 baselines

- **Primary metric:** the relative-heading error |wrap(δ̂ − δ)| per joint and per 32 Hz step. Report the mean, median
  and P95 per joint and joint type, at 1, 10 and 30 minutes since boot.
- **Secondary metric:** the parent-relative orientation error, the geodesic angle 2·atan2(‖v‖, |w|) of each joint's
  error quaternion. Compute it with the simulator's true mountings, and say so.
- **Baselines:**

| ID | estimate δ̂ | note |
|---|---|---|
| B0 | 0 (no correction, after Step 0) | the size of the problem |
| B1 | box registration, plus the T-pose | the honest "no learning" baseline |
| O1 | the true initial offsets, a_c − a_p | oracle, simulation only: how much error is offset rather than drift |
| O2 | the true δ | oracle ceiling: its error must be exactly 0 |

## 5. Data

### 5.1 AMASS: human motion, for training and testing in simulation

- **Access.** Every person who downloads AMASS registers at amass.is.tue.mpg.de and accepts its non-commercial research
  licence. The SMPL+H body model, from the MANO/SMPL+H site, has the same rule. Redistribution is forbidden. The
  collaborator registers and downloads by hand; you write the instructions, the expected file list and the verification
  script. Never store credentials.
- **Model files.** Prefer the `.npz` versions, to avoid `chumpy`.
- **Which subsets.** Start with one small subset for the end-to-end slice. Then choose a few subsets that cover walking
  straight and in circles, turning, bending, reaching, sitting down and standing still (for example CMU, KIT and
  BMLmovi; check their sizes first). Record the choice and the reason in `docs/DATA.md`.
- **Exclude TotalCapture from training.** Its motions may form a later test set.
- **Format.** Inspect the release you downloaded and handle its keys, for example `poses`, `trans`, `betas`, `gender`
  and the frame-rate key. Frame rates differ between subsets.
- **Orientations.** The first 66 values of `poses` are axis-angle rotations of the root and 21 body joints in the SMPL
  tree. Global rotations compose down the tree: R_0 = Exp(root), R_j = R_parent(j)·Exp(local_j). Segment orientations
  need only the poses. Positions, needed later for figures, need the body model.
- **Up axis.** Find AMASS's up axis from the data: during standing, the pelvis's body up axis must map to world up.
  Convert to +Z-up and test it.
- **Mapping:**

| Mesquite segment | SMPL joint whose global rotation is that segment's orientation |
|---|---|
| HipsAlt (pelvis) | 0 pelvis |
| Spine (chest) | 9 spine3 (document this choice; spine2 is the alternative) |
| Head | 15 head |
| Left/RightUpLeg | 1 / 2 hips |
| Left/RightLeg | 4 / 5 knees |
| Left/RightFoot | 7 / 8 ankles |
| Left/RightArm | 16 / 17 shoulders |
| Left/RightForeArm | 18 / 19 elbows |
| Left/RightHand | 20 / 21 wrists |

  At the SMPL rest pose, a T-pose with arms horizontal, every segment's global rotation equals the root's, R_0. So the
  rest pose is the natural simulated T-pose. Left and right are the subject's own sides.
- **Splits.** Split by subject, within each subset, into train, validation and test (for example 70/15/15).
  - Keep every window and every corruption variant of one sequence in one split.
  - Write the split manifest (CSV) and record its SHA-256 in `WORKLOG.md`.
  - After that, Danyal freezes it. Nothing is trained on it in Step 1, but it must exist before Step 2.

### 5.2 DIODEM: real sensors with optical truth (sanity check only in Step 1)

- **Where.** Harvard Dataverse doi:10.7910/DVN/SGJLZA (version 3.0), or Zenodo record 15511879. The paper is in
  Scientific Data, doi:10.1038/s41597-025-05468-w.
- **Licence.** CC0 1.0 on Dataverse and CC BY 4.0 on Zenodo; cite it either way. It is open, so you may script the
  download.
- **Content.** 46 minutes of mechanical five-segment "arm" and "gait" chains; no people. Ten Xsens MTw Awinda IMUs, two
  per segment (one rigid, one foam-mounted), raw accelerometer, gyroscope and magnetometer at 40 Hz. OptiTrack truth
  from 12 cameras, with segment frames built from four markers each. There is no official train/test split.
- **What to do in Step 1:**
  1. Download and verify the checksums.
  2. Load one trial.
  3. Drop the magnetometer and run VQF in six-axis mode (`pip install vqf`; Laidig & Seel, Information Fusion 2023)
     for two IMUs on adjacent segments.
  4. Plot their relative heading against the optical relative heading, as one sanity figure.

  Check for an official loader from the dataset's authors, and use it if one exists. Make no claims from it yet.

### 5.3 Real suit recordings: arriving later, so define the contract now

Danyal will record the real suit with a read-only logger. Your simulator must write **the same frames table**, so the
same Step 0 code reads both.

- **Frames table:** `frames.parquet`, one row per received frame, with columns `t_host_s` (float64), `pod_id` (uint8),
  `batt` (uint8), `qx`, `qy`, `qz`, `qw` (int16, raw wire values), `count` (uint16) and `ms_lo` (uint16). A sidecar
  `session.json` gives the phases (box, T-pose, motion) as host-time ranges.
- **Labels:** `labels.parquet`, simulation only, never read by Step 0 or any estimator. It holds:
  - per pod and send tick: true sample time, received, fresh, true ψ_i;
  - per joint and 32 Hz step: true δ_pc.
- **Metadata:** `meta.json` with the full config, seed, git commit, package versions, source sequence and split, and
  every pod's drawn parameters.

## 6. Environment

- **Python.** Use Python 3.11 in a virtual environment (`python -m venv .venv`).
- **Pinned packages.** `requirements.txt` with exact `==` versions, and `requirements-lock.txt` from `pip freeze`.
  Include numpy, scipy, pandas, pyarrow, matplotlib, h5py, pyyaml, tqdm, torch (the CPU build is enough), smplx, vqf,
  requests, pytest, hypothesis and ruff.
- **`scripts/check_env.py`.** It prints the Python and package versions, the OS and CPU. It also checks that every
  import works and that the data files exist with the right checksums.
- **Reproducibility.** Every random draw comes from one seeded generator per run. The same config and seed must give
  byte-identical outputs.
- **Run identity.** A run is identified by `(git commit, config hash, seed)`. Store it in the run's `meta.json`, and
  name output folders after it.
- **CI.** A GitHub Actions workflow runs ruff and the synthetic test suite on every push and pull request.

## 7. Milestones (each ends with a pushed branch, a pull request and a WORKLOG entry)

| # | Milestone | Done when |
|---|---|---|
| M1 | Skeleton, environment, `CLAUDE.md`, the brief saved, CI; a conventions module and a wire-frame codec, both with tests | the golden vector, the convention properties and the §4.4 example pass |
| M2 | Data: AMASS and SMPL+H instructions; DIODEM download; manifests with checksums; loaders; the AMASS → Mesquite mapping | `check_env.py` passes; one AMASS sequence gives 15 orientations; the up-axis and left/right checks pass; a stick-figure image and the DIODEM sanity figure are in `docs/figures/` |
| M3 | Thin end-to-end slice: one sequence → constant offsets only → frames → Step 0 → B0 and O2 | B0 error equals the offset differences, up to quantisation; O2 error is 0 |
| M4 | The full simulator: the error model, timing, held frames, loss, reboots, the session timeline, configs and a CLI | every check in §9 passes |
| M5 | Split manifest; generate a dataset (size configurable); B0, B1, O1 and O2 under the default and swept parameters; the first results table | tables and figures in `docs/VALIDATION.md` and `docs/RESULTS_STEP1.md`, reproducible from one command |
| M6 | Documentation complete (§10); a final pass of the full check; the report (§11); tag `step1` | Danyal can follow `README.md` from a fresh clone to the results |

## 8. The simulator

- **Command line.** For example, `python -m mesqsim simulate --config configs/default.yaml --seed 1 --out outputs/...`
  and `python -m mesqsim baselines --run outputs/...`.
- **Pipeline, in order:**
  1. Load a sequence and choose a random facing.
  2. Compute the segment orientations.
  3. Build the session timeline.
  4. Draw each pod's parameters.
  5. Generate each pod's chip sample times and its true ψ_i.
  6. Interpolate the segment orientations to those times with slerp.
  7. Form the reports q_i and canonicalise to w ≥ 0.
  8. Run the send grid: newest sample, held frames, `count` and `ms_lo`.
  9. Quantise.
  10. Drop frames with the loss model.
  11. Add arrival times.
  12. Write the frames table, the labels table and `meta.json`.
- **One config file.** It holds every parameter with its label and source note, with sweeps as lists.
- **Managing runs.** Outputs go only to `outputs/`, with run identity in the folder name. A `clean` command and a
  `list-runs` command help keep it tidy. Describe in `docs/SIMULATOR.md` how to add a new error term or a new dataset
  without breaking the tests.

## 9. Accuracy: checks that must pass

One command, `scripts/check_all.sh`, runs ruff, every test and the validation script. The validation script writes its
numbers into `docs/VALIDATION.md`, with the date and the commit. Required checks:

1. **Codec.** The golden vector both ways. Saturation at ±32767. Battery clamped at 100. `count` and `ms_lo` wrap at
   65,536.
2. **Conventions.**
   - Y(90°) maps x to y.
   - heading(Y(α)·R) = wrap(heading(R) + α).
   - wrap(190°) = −170° and wrap(−180°) = 180°.
   - q and −q give the same rotation.
   - The composition order is pinned.
3. **The §4.4 example.** The correct sign leaves 0°; the wrong sign leaves 14°.
4. **Invariances (§4.5),** including that tilt is unchanged by any correction.
5. **Zero corruption.** With every error off, no loss and exact timing: B0 = 0 up to quantisation (≤ 0.01°), and O2 = 0.
   With only the initial offsets on, in the arbitrary mode with zero box-slot error: B1 = O1 = 0, while B0 is not.
6. **Analytic growth,** by Monte Carlo over many pods. Each must agree within 5 %:
   - random walk only: σ of ψ at 600 s = 0.367°, and σ of a pair's relative heading = 0.519°;
   - drift only: δ(t) = (b_c − b_p)·t exactly;
   - one full turn with k = 1.5 %: 5.4°.
7. **Timing.**
   - Discarded fractions of 0.36, 0.42 and 0.47 at 50.1, 55 and 60 Hz.
   - No held frames with stalls off; with stalls on, the expected held rate.
   - Step 0 unwraps `ms_lo` correctly across at least three wraps.
   - Step 0's fresh, held and lost flags equal the simulator's truth exactly.
   - The simulated loss rate and burst length match the loss model's parameters.
8. **Reproducibility.** The same config and seed give identical hashes; a different seed gives different output.
9. **Splits.** No subject appears in two splits. All variants of a sequence stay in one split. The manifest's hash is
   unchanged.
10. **AMASS mapping.**
    - Rest pose: every segment equals the facing rotation.
    - The up axis is verified.
    - While walking, left and right thighs alternate.
    - Knee flexion stays in an anatomical range.
11. **DIODEM.**
    - The file list and checksums match the manifest.
    - The rate is 40 Hz, and quaternion norms are 1.
    - The six-axis relative heading drifts while the optical one stays bounded (figure).
12. **Baseline order.** In expectation O2 = 0 ≤ O1 ≤ B1. In the *arbitrary* offset mode, B0 is much larger than B1.

## 10. Documentation for Danyal

Write for a reader who knows the project but not your code. Define every symbol, unit and file format. Use the picture
first, then the precise version.

- `README.md`: the picture (§1); what is in the repository; a quickstart from a fresh clone (environment, data, checks,
  simulator, baselines); where everything is; the current status.
- `docs/PIPELINE.md`: the pipeline diagram. For every stage, what goes in and comes out (schemas with units), and the
  command that runs it.
- `docs/MATH.md`: the conventions, the measurement model, every error term with its value and label, the target and
  correction, the worked example, Step 0, the metrics, what cannot be known, and why each test exists.
- `docs/CODE.md`: an index of every file and what it is for; how the modules connect; where to extend.
- `docs/DATA.md`: each dataset (content, licence, how a person obtains it, where it lives, checksums, format), the
  AMASS → Mesquite mapping, and the splits.
- `docs/SIMULATOR.md`: how to run it, configure it, sweep parameters, manage and clean runs, and validate after a change.
- `docs/VALIDATION.md`: each check (what, why, command, expected value, latest result with date and commit), and the
  figures.
- `docs/RESULTS_STEP1.md`: the B0, B1, O1 and O2 tables, labelled with the parameter set that produced them.
- `docs/MUST_KNOW.md`: what Danyal must understand to use this safely:
  - the simulator is a model, and its numbers depend on assumed parameters until the bench measures them;
  - the room's heading is unknowable;
  - labels are exact only in simulation;
  - the quaternion order and the signs;
  - the leakage rules;
  - the licences;
  - how to tell that something is broken;
  - a list of open questions;
  - ten self-check questions with answers, such as: *What does a repeated `ms_lo` mean? Why is B1 not zero? Which
    parameters are measured and which assumed? Why can no model recover north?*
- `docs/analysis_plan_DRAFT.md`: the research question, metrics, splits and thresholds as a **draft**. Danyal approves
  and freezes it.
- `WORKLOG.md`: one entry per work session or milestone, newest last, with:
  - the date and branch;
  - the files added or changed;
  - the commands run and their results (test counts, check output);
  - the data downloaded (names, sizes, checksums);
  - the decisions made, and why;
  - the open questions;
  - the commit hashes pushed. A commit cannot contain its own hash, so fill in the hash at the start of the next entry.

## 11. When you stop: the report

Post the report to the collaborator, and add it as the last `WORKLOG.md` entry:
1. What was built, as one list of files.
2. The output of `scripts/check_all.sh`.
3. The first results table, with its parameter set.
4. Every assumption still in the config.
5. What is blocked, and on whom.
6. What Danyal must decide, for example the split freeze, the AMASS subsets, and the Spine joint (spine3 or spine2).
7. A proposed plan for Step 2: the geometric estimator. Do not start it.
