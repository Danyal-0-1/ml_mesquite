You are continuing a research repository for **Mesquite**, a low-cost inertial motion-capture suit. The project is led
by **Danyal Khorami**, the project owner. Steps 1 and 2 were built here by earlier sessions, from
`docs/brief/STEP1_BRIEF.md` and `docs/brief/STEP2_BRIEF.md`. A collaborator or Danyal is running this session with you.

Beyond this brief and this repository you may use three things:
- the public repository of the suit's firmware, browser page and phone app, `github.com/Danyal-0-1/mesquite-mocap`,
  read-only, pinned at commit `87d7c3b` (§2.2);
- Danyal's recordings, which the person copies into `$MESQ_DATA_DIR/real_suit/` (§2.3);
- the papers, datasets and software in the reading list (Appendix), each verified before you cite it.

Where the brief is silent, make a choice, write it down and test it. Stop and ask in three cases:
- only a person can act: copying the recordings, describing the suit, registering for a dataset, recording;
- a decision would change the research;
- a job would break the disk or compute budget (rule 5).

**Step 3 has two halves.** On 2026-11-16 Danyal merged the data work into Step 3:
- **the twin and its data.** Rebuild the simulator so that what it writes behaves like Danyal's real suit, prove that
  against their recordings, and generate the training data, stored and versioned outside git;
- **the model kit, for the original Step 3.** **Danyal writes the learned models themselves**, to learn how: the data
  loader, the networks, the loss and the training loop. That includes their own model, a network that estimates every
  pod's heading relative to the phone. You write none of that. You prepare everything around it: the data and its
  specification, a scorer that grades any model, the tests a model must pass, the numbers to beat, and a guide (§7).

**Your task, in this order:**
1. Close what Step 3 needs, and ask the person what only they know (S3-M0, §9).
2. Read Danyal's recordings and measure them, reproducing the analysis of 2026-11-14 (S3-M1).
3. Build the **Mesquite twin** (§4): pods on a real body's skin; the motion felt by a gyroscope and an accelerometer;
   the chip's six-axis fusion in the loop; the wire and the radio; the phone and its app; the session as Danyal runs
   it; and the browser page's own pose chain and recorder, checked against the real page code (S3-M2 to S3-M5).
4. **Calibrate the twin** against the recordings, and measure the gap that remains (§5, S3-M6).
5. **Generate the training data**, stored, versioned and verified, with a command that adds more on demand (§6, S3-M7).
   Run the generation; do not stop at the code.
6. **Prepare the model kit** (§7, S3-M8): the data specification, the model contract and the scorer, the tests a model
   must pass, the learning-curve tools and a step-by-step guide. Write no learned model.
7. **Score the baselines and the oracle** on the twin's test subjects and through the page twin (§8, S3-M9). These are
   the numbers Danyal's models must beat.
8. **Write the explanations in `docs/step_3/`** (§12, S3-M10): the mathematics, the programming and the references,
   ready to cite in the course report, together with the picture, the pipeline, the theory and the guide.
9. Write the report (§13), then stop.

Do **not** build any of these:
- a learned model, its data loader, its loss or its training loop. Danyal writes them (§7). Short snippets that show a
  library's interface are fine in the guide; a finished model, loader or loop is not;
- the fusion of geometry with learning (Step 4);
- any change to `mesquite-mocap`: Danyal fixes the firmware and the page; you only read them;
- anything that touches hardware: no serial port, no hub, no pod, no phone.

## 0. Rules for this repository (Step 3 additions)

1. **First commit.**
   - Save this brief, unchanged, as `docs/brief/STEP3_BRIEF.md`. If you were given it as a file at the repository root,
     keep only the text below its first line of dashes, and do not commit the root copy.
   - Update `CLAUDE.md`: keep every rule that still applies, add the rules below, point to all three briefs, and make
     Step 3 the current scope. Never edit the earlier briefs.
2. **The earlier briefs still hold where this one is silent:**
   - Step 1: the suit facts (§3), conventions (§4.1), measurement model (§4.2), data rules (§5), environment (§6);
   - Step 2: the estimator contract (§5.2), the common clock (§5.3), the invariance tests (§5.9), the evaluation (§6).

   Before you start, read both briefs, then `CLAUDE.md`, `README.md`, `WORKLOG.md` and every file in `docs/`.
3. **Visibility and disclosure.** The repository is public.
   - Before the first push of Step 3 code, ask once: *"Should the Step 3 code (the twin, the numbers fitted from your
     recordings, and the model kit) be pushed to the public repository, or kept local?"*
   - Follow the answer and record it in `WORKLOG.md`. Until you have it, commit locally.
4. **No data in git, now including Danyal's recordings.**
   - Never commit: the recordings (BVH, Motive CSV and TAK, the phone app's logs); anything derived from them row by
     row; generated datasets; trained weights; datasets, body models, archives or credentials (Step 1, rule 3).
   - Commit: code, configs, manifests (name, size, SHA-256), small synthetic fixtures and summary statistics.
   - Keep every committed file under 5 MB.
   - Never write a home-folder path into a committed file. Write paths relative to `$MESQ_DATA_DIR` or
     `$MESQ_OUTPUT_DIR`.
5. **Disk and compute budget.**
   - Before writing more than 20 GB under `$MESQ_OUTPUT_DIR`, or starting a job expected to run longer than 8 hours,
     state the estimate and ask.
   - Always leave at least 15 GB free on the disk. On Danyal's Mac about 52 GB were free on 2026-11-16.
6. **`mesquite-mocap` is read-only and pinned.**
   - Clone it outside this repository, for example to `$MESQ_DATA_DIR/mesquite-mocap`, and check out `87d7c3b`.
   - Record the commit in every run's `meta.json`. If you ever use a newer commit, rerun the page golden test (§4.10)
     and say so.
7. **Never invent numbers** (Step 2, rule 6). Two labels widen:
   - `measured` now also covers a number read directly from one of Danyal's recordings; name the file and the command;
   - `fitted` now also covers a twin parameter estimated on the **calibration recordings** (§5.4), never on the check
     recordings.
8. **Truth stays with the scorer.**
   - No estimator reads labels, `meta.json` or the twin's truth, at inference or anywhere else. That includes the
     learned models.
   - Targets reach training through one function, `mesqsim/learn/targets.py`, which you write and test. Danyal's code
     will use it, and nothing else, to read targets.
   - The twin's truth lives in `twin:`. Leave an empty, documented `learning:` section for Danyal's settings.
     Randomisation and sweeps change only the twin.
9. **Deployable inputs only.** Every model input must exist in the live stream:
   - pod frames, after Step 0;
   - phone messages as the page receives them.

   Nothing from the truth, the future, OptiTrack or the phone app's private log enters an input.
10. **One convention per quantity** (Step 1, rule 7).
    - The page works in three.js's +Y-up world, and the BVH has its own Euler convention. Both conversions live in one
      module, `mesqsim/twin/page_frame.py`, with tests.
    - Convert at the boundary. Everything else works in the repository's +Z-up conventions.
11. **Banner.** Every result on the twin carries one line: the twin's version, "AMASS motion", and how many twin
    parameters are `fitted`, `measured`, `datasheet` and `assumption`. Every result on Danyal's recordings is
    descriptive, because there are only seven of them.
12. **Talk to the person running the session.** Before each milestone, say in plain words what you will do. After it,
    say what you did and how you checked it. Refer to Danyal by name, or as they/them.
13. **Milestones and pull requests:** as in Step 2, rule 5.
14. **Step 3 is not finished** until all of these hold:
    - `docs/step_3/` is complete, the guide included;
    - the training dataset is stored and verified;
    - the model kit is ready, and the baselines and the oracle are scored;
    - the full check passes.

    Then write the report (§13) and stop.

## 1. The whole project in one picture, and what Step 3 adds

*Keep a version of this at the top of `README.md`, updated for Step 3.*

**The suit today.** Fifteen pods are strapped to the pelvis, chest, head, upper arms, forearms, hands, thighs, shins and
feet.
- Each pod's chip fuses a gyroscope and an accelerometer. Gravity gives every pod the same "down", but there is no
  compass, so each pod invents its own private north when it switches on. That north slides as the session goes on.
- Thirty-two times a second each pod radios a 16-byte postcard to a hub.
- An iPhone rides on the belt, running the Mesquite Hips app. Its camera watches the room, so it knows where the hips
  are and which way they face, in a frame fixed when it was zeroed. It sends a short message thirty times a second.
- A web page turns the postcards into an avatar's bones and the phone's position into the avatar's travel, and records
  BVH files.

**What went wrong on 2026-11-14.**
- The avatar's left leg swung sideways when Danyal stepped forward. That leg's pods had a north about 75° away from the
  others'.
- The avatar barely left the start, and drifted sideways. The page halves every distance, and the phone's frame was
  turned about 74° from the body.

**What Step 3 builds first: a flight simulator with the real cockpit.** Step 1's simulator is a sketch of the suit: it
writes what a pod would report from a formula. The twin rebuilds the suit piece by piece:
- it puts every pod on the skin of a real person's body, where Danyal straps it;
- it lets the pod's gyroscope and accelerometer feel that body's motion, with the chip's own errors;
- it runs a six-axis fusion like the chip's, so a north slides because of physics, not because a formula says so;
- it packs the result into the same postcards, sent through the same lossy radio;
- it carries a phone on the belt that tracks the hips as the phone does, and an app that zeroes, rounds and sends as
  Danyal's app does;
- it runs the session as Danyal runs it, including the slips: a pod restarted by a USB cable after the box, a Zero not
  taken in the T-pose;
- it draws the avatar with the page's own code, so it writes the same kind of BVH file Danyal records.

Then the twin is checked against Danyal's recordings. The same measurements, run on a real file and on a twin file,
must come out alike.

**Then the data.** Thousands of simulated sessions of real people's motion (AMASS), each with known truth: enough that
more data would no longer help the network, as a learning curve decides. The data is stored outside git, with a
manifest and a dataset card.

**Then the networks, which Danyal writes.**
- **TCN-J** works per joint: how far has each child pod's north slid from its parent's?
- **TCN-P, Danyal's model,** works per pod: how far has each pod's north slid from the phone's?

Each reads the last eight seconds of postcards and phone messages, and answers with a confidence. Turning every pod by
its answer puts the whole suit on the phone's north: the body then faces the way the phone says it walks. The room's
true north stays unknown; the phone's frame becomes the one frame everything shares.

Danyal is learning to build them, so this step builds the workshop, not the networks: the data, its specification, a
scorer, the tests, the numbers to beat, and a guide that walks from opening the first data file to a trained model.

```text
AMASS subject (SMPL+H poses, shape) ──► motion program: that subject's clips joined into a 3–30 min session
   box → Start → strapping → (slips) → phone Zero → T-pose → walking, turning, arm raises, squats, standing still
   ──► the body's skin: pod sites and the belt site, their positions and surface frames       [truth: R_i(t), p_i(t)]
   ──► physics: angular rate ω and specific force f in each pod's own frame
   ──► sensor errors (bias, scale, misalignment, noise) ──► six-axis fusion in the loop ──► chip samples, ≈ 55 Hz
   ──► firmware: 32 Hz send grid, held frames, count, ms_lo, int16 (or the old JSON rounding)
   ──► radio: bursts of loss, arrival times                                             [labels: ψ_i(t), δ_pc(t)]
   phone: belt pose ──► ARKit-like errors and tracking states ──► the app: Zero, lever arm, wire turn, ≤ 95 B, 30 Hz
                                                                                       [labels: ψ_ph(t), ε_i(t)]
   ──► frames.parquet + phone.parquet + session.json (+ labels, meta.json)       the format a real logger writes
   ──► PAGE TWIN, js/custom_icm.js@87d7c3b ported and checked against the real code: tree maps, T-pose deltas,
       smoothing, the root from the phone, the recorder ──► twin BVH    ◄── compared with Danyal's BVH files
STEP 0 + FEATURES ──► training store: HDF5 shards, manifest, dataset card; more on demand
   ──► THE MODEL KIT: data specification, model contract, tests, learning-curve tools, the guide
   ──► Danyal's models, written later: TCN-J (δ̂ per joint), TCN-P (ε̂ per pod, relative to the phone), rivals
   ──► scorer, for any model: errors, coverage, strata ──► through the page twin: does the leg swing forward, does
       the body face where it walks? ──► docs/RESULTS_STEP3.md: the baselines and the oracle, the numbers to beat
```

**What goes in and comes out.**

| | What |
|---|---|
| In | AMASS and SMPL+H (licensed, already downloaded); `mesquite-mocap` at `87d7c3b`; Danyal's recordings; Danyal's description of the suit (S3-M0); one YAML config |
| Out, the twin | per session: the frames and phone tables a real logger would write; labels; a twin BVH drawn by the page twin |
| Out, the data | `$MESQ_OUTPUT_DIR/datasets/twin_v1/`: features, targets and masks in shards; a manifest; a dataset card |
| Out, the model kit | the data specification, the model contract and scorer, the tests, the learning-curve tools and the guide. Danyal's models will output per joint δ̂, or per pod ε̂, in radians in (−π, π], with a concentration κ̂ ≥ 0 and an "uncertain" flag, at every 32 Hz step |
| Never out | geographic north, or the body's heading in the room. The phone's Zero frame is a shared frame, not north |

**The six steps** (state on 2026-11-16):

| Step | Work | State |
|---|---|---|
| 1. Ground | simulator, Step 0, baselines | done; tag `step1` |
| 2. Geometry | the per-joint geometric estimator GEO | S2-M0 merged; S2-M1 committed locally; S2-M2 to S2-M8 to do |
| **3. Twin, data and the model kit** | **this brief**; Danyal then writes and trains the models | starts after Step 2 (§2.1) |
| 4. Fusion | geometry and learning, each trusted where it is strong | after Step 3 |
| 5. Real suit | bench measurements; OptiTrack with a marker triad per pod; worn sessions. It runs alongside, and its numbers feed the twin | recordings of 2026-11-04, -13 and -14 exist |
| 6. Test and report | held-out subjects, DIODEM and the real suit; the course report | — |

## 2. Where things stand (checked on 2026-11-16)

### 2.1 This repository

- `main` is `9b06673`, the merge of S2-M0 (pull request 1, merged on 2026-11-16). It is tagged `step1`.
- **AMASS is downloaded**: HumanEva, BMLmovi, CMU and KIT, SMPL+H G; 8,143 files, 246 subjects, 23.2 hours of motion.
  So is the SMPL+H body model (gender-specific, 16 shape parameters).
  - The split manifest's SHA-256 is `d36448adb76762251668b9940933e8a14df664ad6d3277fe3a930f7d55049fb3`: 173 training,
    37 validation and 36 test subjects. It waits for Danyal's freeze.
  - Only 767 sequences last 20 s or more. The median clip lasts 4.6 s (BMLmovi) to 17.5 s (HumanEva). §4.9 decides how
    to make long sessions.
- `docs/RESULTS_STEP1.md` runs on AMASS. B1, the box registration, has a median error of 3.33°, 5.53° and 9.65° at 1,
  10 and 30 minutes since boot; B0 is about 90°.
- **Step 2 is in progress.** S2-M1 was committed on 2026-11-16 (`a3167cc`, branch `s2-m1`), locally, on Danyal's Mac.
  It is not pushed while the Step 2 disclosure question is open. S2-M2 to S2-M8 are still to do.

**If `step2` does not exist when you start**, ask the person which of two options to take:
- **(a), recommended:** wait until Step 2 is finished.
- **(b):** run S3-M0 to S3-M7 now, from `main`. These are the recordings, the twin and the data, and they do not need
  Step 2.
  - Put the new code in new modules only, so that the two steps merge cleanly.
  - S3-M8 onward then waits for `step2`, and every comparison with GEO waits with it.

### 2.2 The system as it is now: `mesquite-mocap` at `87d7c3b`

Read these files; never change them. Line numbers are at `87d7c3b`.

| Part | Where | What matters for the twin |
|---|---|---|
| pod firmware | `new_device_code/Pod_Watch_Binary/` | the ICM-20948 DMP's Game Rotation Vector (Quat6): six-axis fusion, magnetometer off, so the heading at boot is arbitrary (`Pod_Watch_Binary.ino:467-474`). ODR divider 0, about 55 Hz (`:482-490`). The raw gyroscope and accelerometer streams are enabled but never read or sent (`:476-480`). The 16-byte frame (`mesq_packet.h`) and the 31,250 µs send grid (`mesq_pod_core.h`) are those of the Step 1 brief §3.2–3.3 |
| hub | `new_device_code/Dongle_Binary/` | USB records of at most 96 bytes (`mesq_hub_core.h:173`), so a phone line is at most 95 bytes |
| page parser | `js/mesq_parser.js` | binary frame → `{bone, x, y, z, w, count, millis}`; `count` is per send, `millis` is the decode time |
| pose chain | `js/custom_icm.js` | per message, in arrival order: the raw quaternion; the T-pose snapshot `calibration`, set in `calibrate()` (`:227`); the world delta raw·calibration⁻¹ (`multiplyQuaternions(raw, refInv, bc)`: three.js ignores the third argument; 17 sites, `:584` to `:896`); the tree map `getTransformedQuaternion` (`:439-476`); `applyMountingOffset` (`:37-50`, offsets computed once in `calibrate()`); local = parent global⁻¹ · child global; then `bone.quaternion.slerp(target, slerpDict[bone] or 0.24)`, once per message |
| root | `js/custom_icm.js` | the Hips orientation comes from HipsAlt unless HipsAlt has been silent for over 1 s (`:582`). The phone's position: `posswap` if the tree has one (`:478-498`), × 100 (`:10`), minus `initialPosition`, + 100 cm up (`:946-951`), then a lerp of 0.1 per message (`:956`). `initialPosition` is the live hips position itself, not a copy (`:228`). A position containing an exact 0 is dropped (`:931`) |
| smoothing | `index.html:138-157` (`slerpDict`) | Hips and HipsAlt 0.35; Spine, Head and upper arms 0.4; forearms and hands 0.35; thighs 0.28; shins 0.15; feet 0.11 |
| trees | `trees/<id>/meta.json` | per bone an `axis` map {order, sign}; for Hips, a `posswap` only in `smooth` |
| recorder | `index.html:395-412`; `js/bvh_converter.js` | `setInterval(1000/30)` samples every joint's local rotation (`index.html:399-407`); Euler XYZ in degrees (`bvh_converter.js:7-13`); OFFSETs from the rig (`:121`); End Sites `0 0 0` (`:133`); the measured frame time and `MESQ_CAPTURE_START/END` (`:155-197`) |
| avatar | `glbs/scene2.glb`; three.js r157 in `build-static/` (`three.module.js`, `GLTFLoader.js`, `BVHLoader.js`) | the drawn skeleton. The page code runs unmodified in Node's `vm` module with inert stubs for the DOM, jQuery and toasts |
| phone app | `new_device_code/phone_app/`: `MATH.md`, `HOW_IT_WORKS.md`, `CODE.md` | ARKit with `.gravity` alignment; the Zero frame (§3); the lever arm (§4); the wire turn per page tree (§3.4); the message of at most 95 bytes, with a precision ladder and a zero nudge (§5); 30 Hz, latest pose wins (§6); orientation only while tracking is limited, nothing while initialising or relocalising (§7); no timestamp in the message (§8) |

### 2.3 Danyal's recordings

The person copies them before you start, from the folder that holds both repositories:

```sh
mkdir -p "$MESQ_DATA_DIR/real_suit"
cp -R mesquite-mocap/Mesquite_benchmarks/. "$MESQ_DATA_DIR/real_suit/"
```

Ask for the phone app's log of 2026-11-14 as well, if it exists (`hips_20261114_17*.csv`, in the Files app under On My
iPhone, then Mesquite Hips). Write `manifests/real_suit.csv` (names, sizes, SHA-256); the files stay out of git.

| Folder under `real_suit/` | Files | What it is |
|---|---|---|
| `Test_Nov4_calibration_tests/` | `MMcap_bvh_2026-11-4-15-29-14.bvh`: 19.2 min, 34,647 frames | old suit (JSON path); a static hold: the pods lie still and the root never moves |
| `Test_Nove13_Optitrack_vs_M/Mesquite/` | five BVH files, `MMcap_bvh_2026-11-13-13-41-17`, `-13-42-39`, `-13-44-44`, `-13-47-28` and `-13-59-24`: 267, 44, 45, 59 and 163 s | old suit: JSON pods and dongle; an older page (no provenance lines, a frame time written as exactly 1/30 s); the old phone. Ten pods plus HipsAlt: no hand or foot pods, and those channels are exactly 0 |
| `Test_Nove13_Optitrack_vs_M/Optitrack/` | five Motive CSV exports (and the `.tak` originals), 120 Hz | the truth: one reflective marker on each of 11 pods; unlabeled markers, metres, Z up; every take starts in a T-pose |
| `Test_Nov14_new_code/` | `MMcap_bvh_2026-11-14-17-52-25.bvh`: 189 s, 5,724 frames | the new suit: binary pod and hub firmware, the new page (UTC `MESQ_CAPTURE_START/END`, a measured frame time of 33.0 ms), the iOS app; 14 pods plus HipsAlt. No OptiTrack |

- **Pairing.** Each OptiTrack take overlaps one BVH by 42–264 s. A BVH is named when it is saved, so its name is the end
  of the take. The two computers' clocks agree within 1.2–3.3 s.
- **What the files cannot show:** a pod's own orientation (one marker gives a position only), lost packets (the page's
  smoothing hides them), and which tree was used (no BVH records it). S3-M0 asks Danyal about the trees.

### 2.4 What the recordings showed

Danyal analysed them on 2026-11-14 and 15, in a private folder you cannot read. The findings you need are these; §5.3
lists the numbers to reproduce.

1. **The pods' norths are never lined up.** Each pod's heading starts wherever it faced at boot. The page subtracts each
   pod's T-pose reading, but never registers the norths against each other, and it ignores the box record. A pod whose
   north is off by ψ draws its bone's motion about an axis turned by ψ about the vertical:
   `Δ_drawn = Y(ψ)ᵀ · Δ_true · Y(ψ)`. A forward step then becomes a sideways swing.
   - **Nov 13:** both shins swing outward, the left by −30° to −63° and the right by +17° to +64°, against a truth within
     ±14°. The offset grows by 20–40° over a few minutes, outward on both legs.
   - **Nov 14:** the left thigh and shin swing at +78° and +73° from the first window on, and the right leg at +9° to
     +15°. No leg bone jumps by more than 15° between frames, so whatever turned the left leg happened before recording.
     A pod restarted by a USB cable after the box step (Nov 4: replugging restarts a pod) would do it.
2. **The root walks half the distance.** Because `initialPosition` is the live hips position, each phone message does
   x ← 0.8·x + 0.1·(100·p + c), with c = (0, 100, 0) cm, so the hips settle at **x\* = (100·p + c)/2**. On Nov 13 the
   fitted scale against the truth was 0.43–0.50, and the hips stood at 66–83 cm against a pelvis marker at 102–105 cm.
3. **The phone's frame was turned on Nov 14.** The root travelled at a constant +74° from the facing, and stood at
   116 cm, so the phone reported y ≈ 1.32 m. Both fit a Zero taken automatically at Start, with the phone in the hand,
   rather than **Zero in T-pose**, or a wrong *Rear camera looks* setting. The root also moved at 18 cm/s while walking,
   against 32–43 cm/s on Nov 13; the files cannot say why.
4. **Trees.** `velvet` and `smooth` give every pod the same map. `honey`, `spiny` and `velvetGreen` turn the left arm's
   map by 90° about the vertical; `velvetGreen` mirrors the right arm (determinant −1); `thorny` tilts several maps.
   On Nov 13 the left upper arm's height correlated with the truth at r = −0.42, −0.49, −0.92, −0.17 and +0.57.
5. **The forward kinematics is sound.** The page's skeleton composes correctly; the errors come from the inputs.
6. **The wire.** The old JSON dongle rounds every component to 0.01, about 1.1°. On Nov 4, 79 % of still frames repeat
   exactly. Nov 14's binary path gives 7 %. Nov 13 gives 3–41 %, which two decimals alone would not produce.
7. **The static hold** (Step 2 brief §4.1): 12 of 13 pods within 1.8° for 17.3 min; the Head pod drifted −2.0°/min for
   7 min, then −0.3°/min; replugging USB power restarted a pod, 78° wrong for about 16 s, then a new north.

## 3. Research questions and hypotheses

The proposal's question still frames the project: where does a small causal temporal network recover relative heading
better than joint geometry, and does combining them beat both? Step 3 adds two questions: is the twin close enough to
the real suit to train on, and does anchoring to the phone help?

| ID | Hypothesis | Step 3's part |
|---|---|---|
| H3 | a TCN's advantage over geometry is larger at 3-DoF joints (hips, shoulders, lower back, neck) than at hinge and 2-DoF joints | the scorer reports it per joint type once Danyal's TCN-J exists; the test is completed in Step 4 |
| H5 | the predicted confidence is calibrated on held-out subjects (68 % and 95 % intervals cover within ±5 points) and falls where errors grow | the scorer measures it for any model with a κ̂ |
| **H6** | **the twin reproduces the recordings better than Step 1's simulator**: more of the core statistics (§5.2) fall inside its 90 % predictive intervals, and a classifier separates its BVH windows from real ones less well (§5.7) | descriptive: seven recordings |
| **H7** | **anchoring to the phone pays at the far end of the body**: TCN-P's error relative to the phone frame at the forearms, hands, shins and feet is lower than the chain estimate (§7.3); and through the page twin it brings the travel–facing mismatch of a mis-zeroed phone within ±20° | tested on held-out subjects once Danyal's TCN-P exists; the scorer and the page-level test are ready for it |

**"Better"** is Step 2's definition: the paired difference in median error has a 95 % bootstrap interval over held-out
subjects that excludes zero, and exceeds a practical margin of 1°, *proposed* and frozen by Danyal in the analysis plan.

**The hard limit, restated.** Turn the room with everything in it about the vertical, and every pod and the phone send
the same numbers. ε_i, the heading of pod i relative to the phone, is observable; north is not. Never claim to measure
north or the body's heading in the room. "The phone's frame" means the frame the phone was zeroed in.

## 4. The twin

**Principle: fidelity first, and every simplification written down.** Each part below says what it models and how it
is checked. Keep a fidelity ledger in `docs/step_3/FIDELITY.md`: one row per property of the real suit, with how the
twin models it, the evidence, and the gap that remains.

**Where the code goes:**
- `mesqsim/real/`: reading and measuring the recordings;
- `mesqsim/twin/`: the twin;
- `mesqsim/learn/`: features, the store, the models and training;
- `tools/page_golden/`: the Node harness of §4.10.

Keep Step 1's `mesqsim/sim/` working unchanged: it becomes the twin's `analytic` mode and the baseline for H6. Reuse its
timing, radio, codec and session code wherever it fits.

### 4.1 What is fixed already

- The conventions of `mesqsim/conventions.py`: (x, y, z, w), Hamilton product, body → world, +Z up, Y(α), heading,
  wrap.
- Step 0, and the frames contract (Step 1 brief §5.3). It gains one table, `phone.parquet` (§4.7).
- Notation used below:
  - R_i(t) is pod i's true orientation (pod → world);
  - q_i(t) is what pod i reports, in its own private world;
  - ψ_i(t) is pod i's heading error, its north (§4.5);
  - δ_pc = wrap(ψ_c − ψ_p) is a joint's relative heading;
  - ψ_ph(t) is the phone frame's heading error;
  - **ε_i = wrap(ψ_i − ψ_ph) is pod i's heading relative to the phone**, the target of TCN-P.

### 4.2 Bodies, pod sites and mountings

- **The body.** Use SMPL+H with each sequence's own shape (`betas`) and gender, through the downloaded model files.
- **Sites.** Each pod sits at one surface point (a vertex, or a triangle and barycentric weights) where Danyal straps
  it. The phone has a belt site.
  - The placement comes from Danyal's answer in S3-M0: a photo in T-pose or a description per pod.
  - Until then, use a documented default table, labelled `assumption`.
- **The surface frame F_i(t).** Its axes are:
  - n̂, the outward vertex normal;
  - t̂, the segment's long axis projected into the tangent plane and normalised;
  - b̂ = n̂ × t̂, so that [t̂, b̂, n̂] is right-handed.
- **The pod's orientation.** R_i(t) = F_i(t)·M_i^nom·E_i, where:
  - M_i^nom is how the pod's own axes sit in that frame (which pod axis runs along the limb, which points out of the
    skin), from Danyal's description;
  - E_i is the strap error, uniform within ±`twin.body.strap_error_max_deg` per axis (`assumption`; Step 1 used ±20°).
- **Positions.** p_i(t) is the site's position from linear blend skinning.
- **The ablation.** `twin.body.pod_frame: segment` uses Step 1's s_i·m_i instead. Report the angle between the two:
  skinning lets the skin twist against the bone.
- **Two skeletons.** The truth uses the subject's SMPL joints. The drawing uses the avatar `glbs/scene2.glb`, through
  the page twin. Never mix them, and say which one every number uses.

### 4.3 What the sensors feel

```text
ω_i(t) = (R_i(t)ᵀ·Ṙ_i(t))^∨                                   angular rate in the pod's frame, rad/s
f_i(t) = R_i(t)ᵀ·(p̈_i(t) − g),   g = (0, 0, −9.81) m/s²        specific force in the pod's frame, m/s²
```

- AMASS runs at 60–120 Hz. Interpolate the rotations with a C²-continuous rotation spline, for example
  `scipy.spatial.transform.RotationSpline` (test which frame its rate is expressed in), and the positions with a
  smoothing spline. Evaluate both at `twin.imu.rate_hz` and differentiate the splines analytically.
- Differentiation amplifies motion-capture noise, so the smoothing is a documented choice (`assumption`). Report its
  effect on ‖f‖ during walking.
- **Checks:**
  - standing still gives ω = 0 and f = R_iᵀ·(0, 0, 9.81);
  - a constant spin about a body axis gives a constant ω;
  - free fall gives f = 0;
  - a site on a circle at constant speed gives the centripetal force exactly.

### 4.4 Sensor errors

```text
ω̃ = (I + S_g)·A_g·ω + b_g(t) + n_g          f̃ = (I + S_a)·A_a·f + b_a + n_a
b_g(t) = b_g,0 + (a random walk of density B_g)          n_g, n_a white, with densities N_g, N_a
```

- **The terms.** S is a diagonal matrix of scale errors, A a small misalignment, b a bias.
- **Labels.** Values come from the TDK ICM-20948 datasheet, DS-000189 rev 1.3, where it gives them (`datasheet`), and
  are `assumption` otherwise. The residual gyroscope bias after the chip's own calibration (b_g,0) and its random walk
  (B_g) are `fitted` (§5.5).
- **Mounting matters.** A bias on whichever pod axis stands vertical on the body integrates into heading. A mirrored
  pair of pods (left and right legs) turns the same per-axis bias into opposite drifts. Test that hypothesis for Nov
  13's outward drift of both legs, against scale error × turning (§5.5).

### 4.5 The chip's fusion, in the loop

- **Three modes,** `twin.fusion.mode`:
  - `analytic`: Step 1's model, unchanged;
  - `vqf6d`: VQF in six-axis mode (the `vqf` package, already pinned);
  - `mahony`: a Mahony complementary filter, so that no result hangs on one filter.
- **Start and output.** Each filter starts at its pod's boot, with yaw 0 in its own frame: that is the Game Rotation
  Vector's arbitrary north. It runs at `twin.imu.rate_hz`, and its output is taken at the chip's sample times (50.1–60
  Hz per part, Step 1), canonicalised to w ≥ 0.
- **Reboots.** After a reboot the filter restarts, so the new north is wherever the pod faces at that moment, not a
  random draw.
- **The labels, from the error rotation:**

  ```text
  E_i(t) = q_i(t) · R_i(t)⁻¹                     (world frame; canonical, E_w ≥ 0)
  ψ_i(t) = −wrap(2·atan2(E_z, E_w))              minus the twist of E about +Z: the north
  tilt error = the angle of the swing left over once that twist is removed
  ```

  - Hand check: E = Y(−12°) has E_z = −sin 6° and E_w = cos 6°, so ψ = 12°.
  - In `analytic` mode with η = 0 this equals Step 1's ψ exactly. Test it.
- **Checks against analytic expectations:**
  - with no sensor error, ψ_i stays at its boot value. Report the tilt error during walking, which is now the filter's
    own dynamic error;
  - with the filter's own bias estimation switched off, a constant bias b on the axis that stays vertical makes the
    filter believe in an extra turn b·t about +Z, so ψ_i(t) = −b·t, within 1 %. With bias estimation on, report how
    much of b the filter removes, and how fast;
  - a scale error k on that axis, over N full turns, adds an extra turn of k·N·360°. Derive the sign in
    `MATH_EXPLAINED.md` from q = Y(−ψ)·R, and reconcile it with Step 1's k·Θ term.

### 4.6 Firmware, wire and radio

- Reuse Step 1's send grid, held frames, `count`, `ms_lo`, int16 quantisation, Gilbert–Elliott loss and latency.
- **Wire modes,** `twin.wire.mode`:
  - `binary16`, the default. The Step 1 golden vector must still pass;
  - `json`, with `twin.wire.json_decimals`. The old dongle prints 2 decimals. Nov 13's exact repeats suggest more, so
    fit it (§5.5).

### 4.7 The phone and its app

```text
true pose          R_ph(t) = F_belt(t)·M_ph          p_ph(t) = the belt site's position
ARKit-like report  R̂_ph(t) = Y(−ψ_ph(t))·R_ph(t)·Exp(η_ph(t)),     ψ_ph(t) = a_ph + w_ph(t) (+ jumps at relocalisation)
                   p̂_ph(t) = ∫ from t₀ to t of Y(−ψ_ph(τ))·(1 + σ_s)·ṗ_ph(τ) dτ + w_p(t)   (a drifting frame bends the path)
tracking           episodes of "limited" (orientation only) and "relocalising" (nothing sent), Poisson in time
the app            the Zero (explicit in T-pose, or automatic at Start), the lever arm, the hip height, the wire turn
                   for the Suit tree setting, the ≤ 95-byte message with its precision ladder, 30 Hz, latest pose wins
```

- **The app.** Implement it from `new_device_code/phone_app/MATH.md` §3–§8. Where `MATH.md` and the Swift code
  disagree, the Swift code at `87d7c3b` wins; say so in the fidelity ledger.
- **Frames.** ARKit's frame is +Y up. Convert it at the boundary, in one tested function.
- **Values.** They are `assumption`, unless you find and verify a published evaluation of ARKit's accuracy. Step 5 will
  measure them from the app's log and markers on the phone.
- **Labels.** ψ_ph(t) uses the same twist rule as the pods, with E_ph = R̂_ph·R_ph⁻¹ after the message is converted to
  +Z up; and ε_i(t) = wrap(ψ_i(t) − ψ_ph(t)) for every pod.
- **`phone.parquet`.** One row per message, as the hub would log it:
  - `t_host_s`;
  - `line`: the text of the line;
  - the parsed numbers exactly as rounded on the wire: `qx`, `qy`, `qz`, `qw`, `px`, `py`, `pz`, with the position null
    when orientation only;
  - `len_bytes`.

  Its labels go in `labels_phone.parquet`.

### 4.8 The session as Danyal runs it

`twin.procedure`, extending Step 1's timeline:

1. **Box.** The pods go in the box with their axes aligned, up to the box-slot error. They boot inside it at
   **Start**. The page's **Box calibration** keeps the box record, which the page at `87d7c3b` never uses.
2. **Strapping.** Step 1's carry-and-strap moves and spins.
3. **Slips,** each with its own probability (`assumption` until fitted):
   - a pod, or both pods of one limb, restarted by a USB cable after the box. It reboots with a new north wherever it
     faces on the body: Nov 14's left leg;
   - the phone's Zero taken automatically at Start, not in the T-pose;
   - *Rear camera looks* not matching how the phone is worn (±90°);
   - *Suit tree* not matching the page's tree (0° or 180°).
4. **The phone's Zero,** if taken, then the page's **Set T-pose** (`calibrate()`): the T-pose snapshot, with Step 1's
   posture error.
5. **An optional forward arm raise** (Step 2 §5.8).
6. **Recording:** on, the motion program, off.

**The protocol program.** Offer the script of Danyal's next OptiTrack session as one motion program, built from
matching AMASS clips:
- 5 s still in a T-pose;
- three sharp hip twists;
- a 4 m walk forward and back, three times;
- a 360° turn on the spot, once each way;
- each leg swung forward and back five times, then out to the side five times;
- each arm raised forward three times, then out to the side three times;
- three slow squats;
- 30 s standing still;
- 5 s in a T-pose.

### 4.9 Long sessions from short clips

AMASS clips last seconds, real sessions last minutes, and drift needs minutes. **This brief decides:** a session is
built from **one subject's** clips, never mixing subjects, joined end to end. This settles the open question in
`docs/DATA.md` §1.2 for Step 3. Step 1 and Step 2 results keep their own settings.

- **Joining.**
  - Turn and shift each next clip about the vertical so that its first root pose continues the previous clip's last:
    heading and horizontal position. Keep the clip's own height.
  - Blend over `twin.program.blend_s` (`assumption`, about 0.5 s), slerping every joint's local rotation and lerping the
    root.
  - Insert still periods and T-poses (the SMPL rest pose) where the program asks.
  - Label every blend `transition`, so the scorer can leave it out.
- **Length.** Each session lasts a length drawn from `twin.program.length_min` (`assumption`, for example 3 to 30).
- **Splits.** Clips stay in their subject's split.
- **Check.** No joint's angular speed at a join exceeds the P99 of the clips' own speeds.
- **Could:** compose by action with the BABEL labels for AMASS (Punnakkal et al. 2021), only if a person downloads
  them.

### 4.10 The page twin

- **Port** the page's pose chain, smoothing, root and recorder to Python, in `mesqsim/twin/page.py`. Port every rule in
  §2.2, defects included, and process messages in arrival order, as the page does.
- **Two modes:**
  - `page.mode: as_is` is the page at `87d7c3b`;
  - `page.mode: ideal_root_box` is a counterfactual and must be named as one. It copies `initialPosition` instead of
    aliasing it, so 1 m is drawn as 1 m. It applies Step 0's box registration (B1) at t and at T.
- **The golden test against the real code.** It lives in `tools/page_golden/`.
  - A Node script loads three.js r157 from `build-static/` and the avatar with the real `GLTFLoader`.
  - It runs `js/custom_icm.js` and `js/bvh_converter.js` unmodified in a `vm` context, with inert stubs for the DOM,
    jQuery, toasts and `moment`.
  - It reads the real `trees/<id>/meta.json`, and parses `slerpDict` from `index.html`.
  - It feeds scripted message sequences and writes every bone's quaternion after each message, and the exported BVH.
- **Scenarios.** At least these three, each in `velvet` and in `smooth`:
  - a T-pose, then a leg swing;
  - a turn while the phone moves;
  - HipsAlt silent for 2 s.
- **Agreement.** The port must match every bone quaternion to 1e-5 and every BVH number to 1e-4.
- **Fixtures.** Commit the small golden outputs as fixtures, so the Python test also runs where Node is missing. Mark
  the Node test `@pytest.mark.node`, and record Node's version.
- **The twin BVH.** The page twin's recorder, run on the twin's messages, writes it. The interval jitter is
  `twin.page.frame_interval_ms`, `measured` on Nov 14 as 33.0 ms mean.

### 4.11 A twin session's files and labels

- **Files.**
  - Step 1's files: `frames.parquet`, `session.json`, `labels_*.parquet` and `meta.json`;
  - `phone.parquet` and `labels_phone.parquet`;
  - optionally `page/`: the twin BVH, and the page twin's bone quaternions for checks.
- **New labels:**
  - per pod and send tick: ψ_i, the tilt error and ε_i;
  - per joint and 32 Hz step: δ_pc;
  - per phone message: ψ_ph and the true phone pose;
  - per 32 Hz step: the true segment orientations (scorer only), the motion type, the `transition` flag and the slip
    events.

### 4.12 What cannot be known, as tests

- Turn every pod report **and the phone** by one common Y(β). Every δ and ε label is unchanged exactly, and every
  estimator within 0.5°.
- Turn one pod i alone by Y(γ). Then q_i becomes Y(γ)·q_i, so ψ_i becomes ψ_i − γ. Exactly ε_i and the δ of the joints
  touching pod i change, by the amounts that follow. Write the signs out in the test.
- Moving the room's origin changes only the phone's position offsets; no heading label changes.

## 5. Calibrating the twin against the recordings

### 5.1 Reading the recordings: `mesqsim/real/`

- **BVH.** Read the hierarchy, the channels (BVH Euler order as declared; the page writes Euler XYZ) and the provenance
  lines. Compute forward kinematics in the file's +Y-up frame, then convert at the boundary.
  - The sites: mid-thigh, mid-shin, mid-upper-arm and mid-forearm; the head pod 10 cm up the head; the chest at Spine2;
    the pelvis at Hips.
  - The avatar faces +Z, with its left at +X.
- **Motive CSV.** Format 1.25: metres, Z up, unlabeled markers, 120 Hz. Its header rows hold the capture start time.
  - Convert with (x, y, z) → (x, z, −y), a proper rotation of −90° about X. The two worlds then differ only by a turn
    about the vertical and a shift.
- **Naming the markers.** Use the T-pose at the start of each take:

  | Height in the T-pose | Markers | Pod |
  |---|---|---|
  | 1.71–1.74 m | 1 | head |
  | 1.45–1.52 m | 4: two at 0.33–0.37 m from the midline, two at 0.56–0.60 m | upper arms (inner), forearms (outer) |
  | 1.28–1.29 m | 1, on the midline | chest |
  | 1.04–1.05 m | 1, on the midline | pelvis (the HipsAlt pod, on the lower back) |
  | 0.58–0.69 m | 2 | thighs |
  | 0.34–0.39 m | 2 | shins |

  - **Facing.** Danyal faced −Y, so left is +X. Prove it each take: you walk toward the chest marker and away from the
    pelvis marker.
  - **Gaps.** A fragment that appears within 2 s at nearly the same place in the body's frame inherits the name. A
    second pass gives each left-over fragment to the free site whose usual body-frame position it matches.
  - **The check.** Leg height changes correlate with Mesquite's own legs at r = 0.64–0.95 in the well-tracked takes,
    and are negative if left and right are swapped.
- **Clocks.** The wall clocks give the offset to a few seconds. The fine offset is the one that best correlates the four
  leg sites' height changes, searched over ±4 s. That alignment is on the legs, so absolute delay cannot be measured
  from these files; only relative timing can.

### 5.2 The statistics Φ

Compute them in `mesqsim/real/stats.py` with **one** code path for real files and twin files.

| ID | Statistic | Definition |
|---|---|---|
| Φ1 | walking | frames where both shins' horizontal displacement from the pelvis, in body coordinates, has a standard deviation above 3 cm over 1 s; reported in seconds |
| Φ2 | facing | BVH: the root's image of the avatar's +Z, made horizontal and averaged as a unit vector over 1.5 s. OptiTrack: the horizontal direction from the pelvis marker to the head marker, averaged the same way |
| Φ3 | swing direction | per leg site, during walking: the major axis (PCA) of its horizontal position relative to the pelvis, in (forward, right) coordinates of the facing; the angle atan2(right, forward), wrapped into (−90°, 90°]. Positive means forward-right. It is an axis, so compare values modulo 180° |
| Φ4 | swing over time | Φ3 in consecutive 30 s windows, and its slope in degrees per minute |
| Φ5 | root path against the truth | a 2-D similarity fit (Umeyama 1991) from the pelvis marker's horizontal path (the median of the visible markers where it is hidden) to the root's: scale, turn and RMS residual; and the median heights of both |
| Φ6 | root speed | the median horizontal speed of the root while walking, and the share of walking frames below 5 cm/s |
| Φ7 | travel minus facing | the median of wrap(direction of the root's horizontal velocity − facing) over walking frames at 5 cm/s or more |
| Φ8 | exact repeats | per pod-carrying bone, among still frames (a frame-to-frame rotation below 0.2°), the share below 1e-6°; the median over bones |
| Φ9 | jumps | per bone, frames with a frame-to-frame rotation above 15°, and whether each snaps back within 0.5 s |
| Φ10 | header | the frame time, and whether it is measured or asserted |
| Φ11 | height correlation | Pearson r between a drawn site's height change and its marker's: arms, and legs for the left–right check |
| Φ12 | static hold | per bone, the drift in degrees per minute, the frame-to-frame noise, and restart events |

### 5.3 Reproduce the analysis of 2026-11-14 first (the gate of S3-M1)

Nov 13 takes are listed in file order: `13-41-17`, `13-42-39`, `13-44-44`, `13-47-28`, `13-59-24`.

| Statistic | Recording | Reference | Tolerance |
|---|---|---|---|
| Φ1 walking, s | Nov 13; Nov 14 | 172, 10, 17, 19, 141; 96 | ±15 % |
| Φ3 shins, Mesquite, left / right | Nov 13 | −30/+19, −63/+43, −58/+50, −40/+64, −31/+17 | ±5°, axial |
| Φ3 shins, truth, left / right | Nov 13 | +11/−1, +12/+14, +5/+3, +1/−10, +2/−2 | ±5°, axial |
| Φ3 shin (thigh), left / right | Nov 14 | +73 (+78) / +9 (+15) | ±5°, axial |
| Φ4 shins over 30 s windows | Nov 13 `13-41-17` | left −31° at 0 s to −73° at 240 s; right +3° to +38° | ±7° |
| Φ5 scale | Nov 13 | 0.43, 0.45, 0.50, 0.48, 0.47 | ±0.03 |
| Φ5 heights, pelvis marker / root, cm | Nov 13 | 103/66, 102/77, 104/82, 103/82, 105/83 | ±3 cm |
| root height, cm | Nov 14 | 116–117 (one 15 s sample at 92) | ±3 cm |
| Φ6 root speed, cm/s | Nov 13; Nov 14 | 43.3, 38.8, 41.5, 31.8, 41.7; 18.1 | ±10 % |
| Φ7 travel minus facing | Nov 13; Nov 14 | −24.9°, −15.0°, −10.6°, −7.6°, −6.6°; +73.7° | ±5° |
| Φ8 exact repeats | Nov 13; Nov 14; Nov 4 | 11.8, 11.4, 2.6, 5.1, 40.8 %; 6.7 %; 79 % | ±3 points |
| Φ9 leg jumps | Nov 14 | none | exact |
| Φ10 frame time | Nov 13; Nov 14 | 1/30 s, asserted; 0.033000 s, measured | exact |
| Φ11 left / right upper arm | Nov 13 | −0.42/+0.64, −0.49/+0.97, −0.92/+0.99, −0.17/+0.49, +0.57/+0.35 | ±0.10 |
| Φ12 static hold | Nov 4 | 12 of 13 pods within 1.8° for 17.3 min | exact count |

- Match the definitions in §5.1–5.2.
- If a value falls outside its tolerance, compare the definitions and fix any bug. Otherwise write the difference and
  its reason in `WORKLOG.md`.
- Never tune a method to hit a number.

### 5.4 Calibration recordings and check recordings

- **Propose a split** in `docs/analysis_plan_DRAFT.md`; Danyal freezes it. The suggestion:
  - **calibrate** on Nov 4, Nov 13 `13-41-17` and `13-59-24` (the long takes, which carry the drift), and Nov 14 (the
    only recording of the new suit);
  - **check** on Nov 13 `13-42-39`, `13-44-44` and `13-47-28`, the three best-tracked takes.
- **Suit versions.** The twin has two: `suit_2026_11_13`, with the JSON wire, the older recorder and the old phone; and
  `suit_2026_11_14`, with the binary wire, the page at `87d7c3b` and the iOS app. Fit the shared parameters on both,
  and each version's own parameters on its own recordings.
- **Limits.** Never fit on the check recordings. Nov 14 is the only new-suit recording, so its check is the calibration
  itself: say so.

### 5.5 What is fitted, what is measured, and what stays assumed

| Parameter | Evidence in the recordings | Label |
|---|---|---|
| residual gyroscope bias b_g,0 (spread per axis) and its random walk B_g | Φ12 on Nov 4; Φ4 drift on Nov 13 | `fitted` |
| gyroscope scale error | Φ4 drift against the amount of turning (from the OptiTrack path) | `fitted`, bounded by the datasheet |
| mirrored mounting against scale error × turning, as the cause of the outward drift (§4.4) | Φ4, left against right | compare the two fits; report both |
| fusion gain (VQF's tilt time constant, or Mahony's gains) | Φ11 height dynamics; Φ8 jitter | `fitted` |
| box-slot error | Φ3 at the start of Nov 13 | `fitted` |
| probability that a limb restarted after the box, and the size of its offset | Φ3 on Nov 14 (left leg), Nov 13 | `fitted`, weakly: say so |
| `json_decimals` of the old wire | Φ8 on Nov 4 and Nov 13 | `fitted`, discrete |
| the phone's frame errors: Zero at Start, the camera setting | Φ7 and the root height on Nov 14 | `fitted`, weakly |
| frame time of the new recorder | Φ10 | `measured` |
| radio loss, latency, the phone's tracking episodes, ARKit drift | nothing in these files | `assumption`, swept |

### 5.6 The fit

- **The method.** Use approximate Bayesian computation by rejection (Beaumont et al. 2002), a form of the method of
  simulated moments (McFadden 1989; Gouriéroux et al. 1993).
  1. For each calibration recording, build matching twin sessions: the same suit version, and AMASS walking that
     matches the recording's walking time, speed and turning rate within 20 %. Take turning from the OptiTrack pelvis
     path where there is one, else from the BVH root.
  2. Draw parameter sets θ from documented priors. For each, simulate several sessions per recording, run them through
     the page twin, and compute Φ.
  3. Score each θ by the distance between simulated and real Φ, each statistic scaled by its spread across
     simulations.
  4. Keep the closest few per cent. They form the posterior.
- **The report.** For every fitted parameter: its median and 90 % range, and whether the recordings narrowed the prior
  at all. With seven recordings many will not: keep those as `assumption` ranges.
- **The cost.** Estimate it before running (rule 5).

### 5.7 The gap that remains, and gate G-T

- **Predictive checks.** For each recording and core statistic (Φ3, Φ4, Φ5, Φ6, Φ7, Φ8), state where the real value
  falls in the twin's posterior predictive distribution, and whether it is inside the 90 % interval. Do the same for
  Step 1's simulator (`analytic` mode, through the same page twin).
- **A classifier two-sample test** (Lopez-Paz and Oquab 2017).
  - Train a small classifier to tell real 10 s BVH windows from twin windows.
  - The features are a fixed, documented vector per window, built from the drawn rotations of the pod-carrying bones:
    for example quantiles of each bone's frame-to-frame rotation, the swing statistics and the root speed. Never the
    truth.
  - Validate leaving one recording out. Report the AUC with a bootstrap interval, for the twin and for Step 1's
    simulator. An AUC of 0.5 means the classifier cannot tell them apart.
- ***Should:*** a kernel two-sample test, MMD with a permutation test (Gretton et al. 2012).
- **Gate G-T, the end of S3-M6.** The twin generates the dataset if both hold:
  - it covers at least as many core statistics as Step 1's simulator;
  - its AUC is not higher.

  If it fails, report why, improve it once (one more round of fitting, or one change to the model), and continue with
  whichever generator is closer, saying so. H6 is answered here.

## 6. The training data

### 6.1 One training example

- **Storage.** A session is stored whole, at the 32 Hz step grid. Cutting windows is the job of the loader, which
  Danyal writes; this specification is what the loader can rely on.
- **Per step:**
  - **inputs** (§7.1), computed by Step 0 and `mesqsim/learn/features.py` from `frames.parquet` and `phone.parquet`.
    The training store and live inference use the same code; test that a streaming computation gives the stored values;
  - **masks:** fresh, held or lost per pod; the phone's message age and state;
  - **targets:** δ_pc for 14 joints and ε_i for 15 pods, in radians, with validity masks;
  - **strata:** time since boot, motion type, `transition`, slip events, suit version, wire mode, HipsAlt present or
    not.
- **Windows,** as the guide recommends them: 18 s long, with the loss on the last 10 s, so that the 7.9 s receptive
  field lies inside the window.
- **Names.** Danyal's practice code expects these, so use them in each session's group, `/sessions/<id>/`:

  | Array | Shape and type | What |
  |---|---|---|
  | `t_s` | (T,) float64 | step times, in s |
  | `x_pod` | (T, 15, 8) float32 | per-pod features (§7.1), pods in the Step 1 brief's id order |
  | `x_joint` | (T, 14, 6) float32 | per-joint 6-D relative rotation, joints in the Step 1 brief's §3.4 order |
  | `x_phone` | (T, 8) float32 | the phone's features |
  | `x_pod_phone` | (T, 15, 6) float32 | each pod's rotation relative to the phone, 6-D |
  | `state_pod` | (T, 15) uint8 | 0 lost, 1 held, 2 fresh |
  | `state_phone` | (T,) uint8 | 0 no message, 1 orientation only, 2 full |
  | `y_delta`, `valid_delta` | (T, 14) float32 in radians; (T, 14) bool | the targets per joint |
  | `y_eps`, `valid_eps` | (T, 15) float32 in radians; (T, 15) bool | the targets per pod |
  | `strata/*` | (T,) each | time since boot, motion type, `transition`, slip |

  The group's attributes are `subject`, `split`, `suit_version` and `conditions`. Only `mesqsim/learn/targets.py`
  reads the `y_*` arrays.

### 6.2 What varies

- **Twin parameters.** Draw them from the fitted posterior, widened by `twin_dataset.widen` (`assumption`, for example
  1.5× in log scale for rates). For unmeasured parameters, use Step 1's sweep ranges. This is domain randomisation
  (Tobin et al. 2017): the real suit should look like one more draw.
- **Conditions.** Mix the session-level conditions, each share an `assumption`:
  - slips (§4.8);
  - both suit versions;
  - HipsAlt present or absent;
  - the phone's settings, right and wrong;
  - the protocol program and free programs.
- **Coverage.** Every stratum of (joint type, or pod) × motion type × time since boot (0–5, 5–15, 15–35 min) holds at
  least `twin_dataset.min_windows` training windows (`assumption`, for example 1,000).

### 6.3 How much: generate now, decide with Danyal's model later

Nobody can say how much data is enough without a model to train, and the model is Danyal's. So:

1. **Generate now** a first `twin_v1`: every training subject, with `twin_dataset.variants` corruption draws per
   session (`assumption`, start at 4), and at least enough to meet the coverage rule of §6.2, within the budget
   (rule 5). Run it, verify it, and report the hours, sessions, subjects, windows per stratum, disk size and wall
   time.
2. **Prepare the decision.**
   - Mark nested parts of the training subjects in `index.parquet` (1/16, 1/8, 1/4, 1/2 and all), so that every
     learning-curve run uses the same parts.
   - Write `mesqsim/learn/curve.py`. It fits err(n) = c + a·n^(−b) (Hestness et al. 2017) to the validation median
     error against the training hours n, and applies the rule. The data is **sufficient** when both hold:
     - one more doubling is predicted to improve the error by less than both `twin_dataset.sufficient_gain_deg`
       (`assumption`, 0.1°) and `twin_dataset.sufficient_gain_rel` (2 %);
     - three seeds agree within their spread.
   - Test it on made-up curves with known answers.
3. **More on demand.** `python -m mesqsim twin-dataset --name twin_v1 --extend <factor>` adds sessions
   deterministically: more corruption draws first, then new compositions of the clips. It never changes a stored
   session, and it updates the manifest.
4. **Later, Danyal** trains their model on the nested parts, calls `curve.py`, and extends the data if the rule says
   so. The guide (§7.5) walks through it.

### 6.4 Storage

- **Location.** `$MESQ_OUTPUT_DIR/datasets/twin_v1/`, never in git. A new version (`twin_v2`) is made when the twin or
  its fit changes; nothing is edited in place.
- **Contents:**
  - `shards/<split>_<nnnnn>.h5`: HDF5 (`h5py` is pinned), one group per session, chunked in time, compressed;
  - `index.parquet`: per session, its shard, subject, split, length, conditions and strata counts;
  - `meta.json`: commit, config hash, seeds, the feature specification and its version, and the `mesquite-mocap`
    commit;
  - `MANIFEST.csv`: every file's name, size and SHA-256. Commit a copy as `manifests/twin_v1.csv`;
  - `README.md`: how to load it.
- **No loader.** Danyal writes it (§7). They get the specification in `docs/step_3/DATASET_CARD.md`, and the tests of
  §7.4 that their loader must pass.
- **Verification.** `python -m mesqsim verify-dataset --name twin_v1` re-hashes every shard against the manifest,
  checks that no subject appears in two splits, and checks the window rules.
- **The dataset card.** Write `docs/step_3/DATASET_CARD.md` after Gebru et al. (2021).

## 7. The model kit: everything Danyal needs to write the models

**The split of work.** Danyal wants to learn to program the models and to feed them data, so they write the loader,
the networks, the loss and the training loop. You write everything those rest on:
- the stored features (§7.1);
- the designs to follow (§7.2);
- the geometry they combine with (§7.3);
- the contract, the scorer and the tests (§7.4);
- the guide (§7.5).

### 7.1 Features (the proposal's model card)

All of them are computed from deployable inputs (rule 9), and all are unchanged by a common heading. Check C5 proves
this to machine precision.

**You compute and store them** in `twin_v1`, with the same code that live inference will use. They are data
preparation, and Step 2's geometry shares it. Everything from reading the stored arrays onward is Danyal's.

- **Per pod** (15 × 8):
  - tilt g_i = R̂_iᵀ·e_z (3);
  - the rotation rate between fresh samples, Log(R̂_i(t₋)ᵀ·R̂_i(t))/(t − t₋) in rad/s (3);
  - the age since the last fresh sample, in s (1);
  - the held flag (1).
- **Per joint** (14 × 6): the first two columns of R̂_pᵀ·R̂_c, a continuous 6-D encoding of the relative rotation
  (Zhou et al. 2019).
- **The phone** (8): the same eight features, computed from the phone's messages on the host clock, with the held flag
  replaced by an "orientation only" flag.
- **Per pod, relative to the phone** (15 × 6): the first two columns of R̂_phᵀ·R̂_i.
- **The clock.** Put pods and phone on one host clock with Step 2's clock module. The phone's messages carry no
  timestamp, so its time is the arrival time.

### 7.2 The reference designs Danyal will implement

These are specifications for Danyal's code, from the proposal's model card. Do not implement them. The tests of §7.4
check Danyal's versions against them, and the guide (§7.5) teaches them.

| Model | Inputs | Size (from the layer sizes; a test recounts them) | Priority for Danyal |
|---|---|---|---|
| **TCN-J** | per joint, shared weights: parent pod 8 + child pod 8 + relative rotation 6 + joint type one-hot 3 = 25. 64 channels, kernel 3, six residual blocks with dilations 1–32 and two causal convolutions each, a 1×1 projection in the first block, a 1×1 head of 3 outputs | 142,595 parameters; receptive field 253 samples = 7.9 s at 32 Hz | Must |
| **TCN-P** (Danyal's model) | per pod, shared weights: the pod 8 + the phone 8 + phone-relative rotation 6 + segment one-hot 15 = 37; the same body as TCN-J | 145,667 parameters | Must |
| **TCN-P-chain** | composition, no new weights: ε̂ = RIGID (phone → HipsAlt) plus TCN-J's δ̂ along the tree (§7.3) | — | Must |
| TCN-B | one whole-body TCN on the 204 joint-and-pod inputs, the same depth | 190,954 parameters | Should |
| MLP-B | framewise, 204 → 256 → 256 → 42 | 129,066 parameters | Should |
| GRU-B | whole-body GRU, 128 hidden units | 133,674 parameters | Should |
| TCN-P-B | whole-body TCN-P: the 204 inputs + the phone 8 + 15 × 6 phone-relative = 302 inputs → 15 pods | count it | Could |

- **Output per joint or pod:** (cos, sin), normalised, and log κ̂, clamped to a documented range.
- **The receptive field.** 1 + Σ_blocks 2·(k − 1)·d = 1 + 4·(1 + 2 + 4 + 8 + 16 + 32) = 253.

### 7.3 RIGID, and composing along the tree

You write RIGID and the chain helper. They are geometry, not learning, and the RIGID + B1 baseline needs them; TCN-P-
chain then needs only Danyal's TCN-J.

- **RIGID** estimates the heading of two sensors on one rigid body, here the phone and HipsAlt on the pelvis.
  - Both see the same physical rotation, so their **world-frame** rates differ only by the two norths:
    ω̂_ph = Y(ψ_h − ψ_ph)·ω̂_h = Y(ε_h)·ω̂_h, with ω̂ = Log(q(t)·q(t₋)⁻¹)/(t − t₋).
  - Let a_t and b_t be the horizontal parts of ω̂_ph and ω̂_h at paired instants. Then Step 2's closed form applies:
    ε̂_h = atan2(Σ b_t × a_t, Σ a_t·b_t).
  - The vertical parts must agree; that is a check.
  - Its confidence is κ = I/σ̂², as in Step 2 §5.4.
  - Worked example, make it a test: ω = (1, 0, 0) rad/s, ψ_h = 12° and ψ_ph = 5° give C = cos 7° = 0.992546,
    S = sin 7° = 0.121869, so ε̂_h = 7.000°.
  - Mountings do not enter, because the rates are in world frames.
- **The chain.** ε_c = ε_p + δ_pc, because ψ_c − ψ_ph = (ψ_c − ψ_p) + (ψ_p − ψ_ph). Starting from ε̂_HipsAlt, add
  TCN-J's δ̂ outward along the tree.
- **Its confidence.** For large κ, the chain's variance is approximately Σ_j 1/κ_j. Recalibrate it on validation.
- **Without HipsAlt** (trees that disable it), the phone is the pelvis sensor, and the chain starts at the hips and the
  lower back.

### 7.4 What you build: the contract, the scorer and the tests

- **The contract.** It is built on Step 2's estimator contract. A model is anything with one method: given the stored
  inputs and masks of one session, or a live stream, it returns `theta_rad` (δ per joint, or ε per pod), `kappa` and
  `uncertain` at every 32 Hz step. Write it as a `typing.Protocol` in `mesqsim/learn/contract.py`, with shapes and
  units in the docstring.
- **The scorer.** `python -m mesqsim score-model --model <module>:<Class> --dataset twin_v1 --split val` does three
  things:
  - it loads any model that meets the contract;
  - it runs the model causally;
  - it writes the tables of §8.2: per joint or pod, per stratum, with coverage and selective risk.

  With `--through-page` it runs the page-level test of §8.3.
- **Two dummy models,** which prove the tests work and teach nothing about networks:
  - `Zeros` predicts 0 with κ = 0. It must pass every contract test, and score exactly as B0;
  - `Cheat` reads the labels. It must fail the label-blindness test.
- **The tests Danyal's code must pass,** in `tests/test_model_kit.py`. Each one skips cleanly until the module it tests
  exists (`pytest.importorskip`), so the full check passes before Danyal starts. The tests import these names, which
  are the ones Danyal's one-week practice uses:
  - `mesqsim/learn/loader.py`: `WindowDataset(path, split, window_steps=576, loss_steps=320, seed=0)`;
  - `mesqsim/learn/models.py`: `TCN(in_features, channels=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32),
    out_features=3)`;
  - `mesqsim/learn/loss.py`: `cos_loss(cos_sin, theta, mask)` and `von_mises_nll(cos_sin, log_kappa, theta, mask)`;
  - `mesqsim/learn/train.py`: `train(model, train_loader, val_loader, cfg)`.

  The tests themselves:
  - *The loader:*
    - no window crosses a session or a split;
    - windows are causal;
    - masks line up with the inputs;
    - a fixed seed gives the same batches.
  - *The loss:* the von Mises negative log-likelihood matches reference values that you compute with SciPy. The values
    include large κ, where a naive log I₀ overflows.
  - *The networks:*
    - the parameter count equals the count worked out by hand from the layer sizes: 142,595 for the reference TCN-J,
      145,667 for TCN-P;
    - the receptive field is 253 steps;
    - changing any input after t leaves every output at or before t unchanged;
    - turning every pod and the phone by one common angle leaves every output identical;
    - inference runs with the labels deleted;
    - one small batch can be overfitted;
    - training on CPU with fixed seeds is deterministic.
  - *TCN-P's use of the phone:* when one pod alone is turned by 30°, 90° and 180°, a trained TCN-P shifts its output by
    the expected amount, within 2°.
- **Measure, do not assume.** The scorer reports a model's time per 32 Hz step on a named CPU.

### 7.5 The guide: `docs/step_3/LEARN_THE_MODEL.md`

It is a course that takes Danyal from opening the first data file to a trained, scored TCN-P. Danyal may arrive with
practice versions of the loader, the TCN, the loss and the training loop, written on a toy problem in a one-week plan.
If so, the guide's first job is to move them into `mesqsim/learn/` and adapt the loader to `twin_v1`'s arrays.

Each lesson has:
- the idea, with a picture;
- a worked example with real numbers from `twin_v1`;
- a small exercise;
- the test of §7.4 that checks the exercise;
- hints, and the mistakes people usually make.

It contains **no finished loader, model, loss or training loop.** Short snippets that show a library's interface (h5py,
`torch.nn.Conv1d`, `torch.utils.data.DataLoader`) are fine.

**Study first.** Danyal is learning from the resources below. Open every lesson with a "Study first" line that names
the exact module, lecture or page; all of them are in the Appendix, under "Learning resources".
- Andrew Ng's *Deep Learning Specialization* on Coursera: five courses, and Danyal is on Course 1;
- Andrej Karpathy's *Neural Networks: Zero to Hero*;
- Afshine and Shervine Amidi's Stanford cheatsheets (the CS 229 refreshers, CS 230) and their PyTorch data post;
- the PyTorch tutorials, and *Dive into Deep Learning* (d2l.ai).

| Lesson | Study first |
|---|---|
| 1 | the h5py *Quick Start Guide*; NumPy's *absolute basics for beginners* |
| 2 | PyTorch's *Datasets & DataLoaders* and *Writing Custom Datasets*; the Amidis' *A detailed example of how to generate your data in parallel with PyTorch* |
| 3 | Shervine's *Linear Algebra and Calculus refresher*; 3Blue1Brown's *Essence of linear algebra*; Zhou et al. 2019 |
| 4 | Shervine's CS 230 CNN cheatsheet (the receptive field); DLS Course 4, module 1; Karpathy's *makemore Part 5*, which builds WaveNet's hierarchy but leaves dilated convolutions out; Bai et al. 2018 |
| 5 | DLS Course 4, module 2 (ResNets, 1×1 convolutions); He et al. 2016 |
| 6 | Shervine's *Probabilities and Statistics refresher*; Beyer et al. 2015; Prokudin et al. 2018; Kendall & Gal 2017 |
| 7 | DLS Courses 1 and 2; Karpathy's micrograd and *makemore* Parts 2–3; PyTorch's *Learn the Basics* |
| 8 | DLS Course 3 (mismatched training and test distributions); Guo et al. 2017 |
| 9 | §7.3 of this brief; Zhou et al. 2019 |
| 10 | DLS Course 3; Hestness et al. 2017 |
| 11 | §8.3 of this brief |
| 12 | Karpathy's *A Recipe for Training Neural Networks*; Shervine's CS 230 *Deep Learning Tips and Tricks* |

The lessons:
1. **Open `twin_v1`:** one session's arrays, their shapes, units and masks. Plot one pod's tilt and one joint's δ.
2. **The loader:** cut 18 s windows with the loss on the last 10 s; masks; batches; never across a split.
3. **Why these features:** tilt, rates between fresh samples, the 6-D rotation, and why a common turn cannot change
   them.
4. **Causal convolutions:** a causal convolution, dilation and the receptive field. Work out the 253 steps by hand.
5. **TCN-J:** a residual block, then the whole network. Count its 142,595 parameters by hand before the test does.
6. **The output on a circle:** (cos, sin) and log κ; the von Mises negative log-likelihood; a stable log I₀; training
   the mean first with 1 − cos, then adding the κ head.
7. **The training loop:**
   - AdamW at about 1e-3, batches of 16–32 windows, early stopping on validation, three seeds;
   - the proposal's curriculum: drift after a fixed registration, then residual offsets, then resets and slips;
   - masks for held and lost steps.
8. **Scoring:** `score-model`, reading its tables, and why B1 always stays in them: an offset the box removes is no
   learned improvement.
9. **TCN-P:** the phone as the anchor; TCN-P-chain through RIGID; which wins where (H7).
10. **Is the data enough?** The learning curve with `curve.py`, and `--extend`.
11. **Through the page:** apply the corrections at t and at the T-pose, and watch the Nov 14 leg swing forward again.
12. **Debugging:** shapes, masks, units, label leaks, causality leaks, a loss that will not fall, a κ̂ that lies.

Write the settings the guide recommends into the empty `learning:` section as comments, labelled `assumption`.

## 8. The scorer, the baselines and the numbers to beat

### 8.1 Methods compared

| ID | What | Uses simulator truth |
|---|---|---|
| B0, B1, O1, O2 | Step 1's baselines, on δ, and composed along the tree on ε | O1, O2 |
| GEO-ID, GEO-K | Step 2's estimators, if `step2` exists | GEO-K |
| RIGID + B1 chain | the phone link by RIGID, then the box registration along the tree | no |
| Danyal's models | TCN-J, TCN-P, TCN-P-chain and the rivals, scored by the same command once written (§7) | no |
| twin ablations | the baselines and the oracle on twins built with segment frames instead of surface frames, and with `analytic` instead of `vqf6d`. Danyal can later train on one and test on another | no |

### 8.2 Metrics and strata

- **Primary.** For δ: |wrap(δ̂ − δ)| per joint and 32 Hz step. For ε: |wrap(ε̂ − ε)| per pod and step. Report the mean,
  median and P95.
- **Derived.** For any per-pod model, δ̂_pc = ε̂_c − ε̂_p on the joint metric, so per-pod and per-joint models meet on
  the same scale.
- **Confidence.** Coverage of the 68 % and 95 % intervals, a reliability diagram and selective risk (Step 2 §6.2).
- **Compute.** Time per step for all joints or pods.
- **Strata.** Joint type or pod; time since boot; motion type; slip; suit version; wire mode; HipsAlt present or not.
  Leave out `transition` steps.

### 8.3 Through the page: the symptoms of 2026-11-14

- **The set-up.** Take held-out test sessions with Nov 14's slips: both left-leg pods restarted after the box; the
  phone zeroed at Start; `velvet`; HipsAlt present.
- **Applying the correction.** Turn each pod by its estimate at t **and** at the T-pose, as the proposal requires for
  the world-delta chain: q̃_i(t) = Y(ε̂_i(t))·q_i(t) and q̃_i(T) = Y(ε̂_i(T))·q_i(T). The phone is the anchor and is
  never turned. Then draw with the page twin.
- **What to report:**
  - the drawn shins' Φ3, with the target within ±15° of the truth;
  - Φ7, travel minus facing, with the target within ±20°;
  - each under `as_is` alone, `as_is` + RIGID and B1, `as_is` + O2 (the true ε: what a perfect TCN-P would reach), and
    `ideal_root_box` + O2. Danyal's models slot in later through `score-model --through-page`.
- **Say plainly what no heading correction fixes:** the halved root (a page defect) and mirrored tree maps.

### 8.4 The retro path on Danyal's recordings (Should)

- **Why it works.** The BVH holds the drawn world rotations. Back in the pod world through the tree's map P, a bone's
  rotation is the world delta Δ_i(t) = q_i(t)·q_i(T)⁻¹, up to a constant on the right (`applyMountingOffset`). With
  q_i = Y(−ψ_i)·s_i·m_i:

  ```text
  Δ_i(t) = Y(−ψ_i(t)) · s_i(t) · m'_i,     m'_i = m_i · R_i(T)⁻¹ · Y(ψ_i(T)),  a constant
  ```

  So a "virtual pod" read from a BVH has exactly the measurement model's form, with the same ψ_i(t) and a different
  mounting. Prove it in `MATH_EXPLAINED.md`, and test it on twin sessions.
- **What you build:**
  - the converter from a BVH to virtual-pod inputs;
  - `twin_v1_bvh`: the twin's sessions, passed through the page twin and its recorder, and converted the same way;
  - Danyal's seven recordings, converted too;
  - a before-and-after command that applies any model's corrections and redraws.
- **What you run now:**
  - the oracle (O2) on twin sessions, before and after, to show the ceiling;
  - the "before" measures on the recordings: Φ3 against the OptiTrack truth on Nov 13; Φ3 of both legs and Φ7 on
    Nov 14.
- **What Danyal does later:** trains TCN-J-bvh on `twin_v1_bvh`, and runs it on the recordings. TCN-P cannot run here,
  because no BVH holds the phone's orientation.
- **The tree.** A constant map turned about the vertical (the 90° left arm) is absorbed into a virtual pod's heading; a
  mirror is not. So ask which tree ran, and report under each candidate if no one knows.

### 8.5 Statistics

- The unit is the held-out subject. Use the paired bootstrap over subjects (state the number of resamples), and Holm's
  correction across the plan's primary contrasts, as in Step 2 §6.3.
- On the recordings, report per recording, descriptively.

### 8.6 Gate G3

From the proposal: on validation, is any learned model ahead of GEO-ID at any joint type? It is asked once Danyal's
models exist, and the scorer prints the comparison.
- If not, the comparison is finished as a negative result, and reported as one.
- If `step2` is absent, G3 waits.

## 9. Milestones (each ends with a pull request and a WORKLOG entry)

| # | Milestone | Done when |
|---|---|---|
| S3-M0 | **Start.** Save the brief; check the state (§2.1). Ask the person, once, all of these: (1) the disclosure question (rule 3); (2) whether the recordings are copied (§2.3), and the app's log; (3) the suit facts: each pod's placement (a photo in T-pose answers it best), how the pods sit in the box, where the phone sits and which way its rear camera faces, the app's settings on Nov 14, and whether Zero in T-pose was pressed; (4) which tree ran on Nov 13 and on Nov 14, and whether any pod was plugged or unplugged after the box step; (5) the disk and compute budget; (6) if `step2` is missing, option (a) or (b). Record each answer; for each unanswered one, a default labelled `assumption` | the questions are asked; `manifests/real_suit.csv` written |
| S3-M1 | `mesqsim/real/`: the BVH and Motive readers, marker naming and tracking, clock alignment, Φ1–Φ12 (§5.1–5.2) | §5.3 is reproduced, or every difference is explained in `WORKLOG.md` |
| S3-M2 | The page twin (§4.10) and its Node golden test | the golden test passes for every scenario and both trees; the root's fixed point x\* = (100·p + c)/2 is reproduced |
| S3-M3 | Bodies, sites, surface frames, mountings, the phone's belt site (§4.2); the sensor physics (§4.3) | the physics checks pass; the surface-against-segment angle is reported |
| S3-M4 | Sensor errors, fusion in the loop, wire modes, the phone and its app (§4.4–4.7) | the fusion checks of §4.5 pass; no app message exceeds 95 bytes over 10,000 random poses; the golden frame still passes |
| S3-M5 | The procedure and its slips, the motion programs, long sessions, the protocol program; one command for one twin session end to end (§4.8, §4.9, §4.11) | the join check passes; identical bytes for the same seed; a twin BVH opens in the page's own `BVHLoader` |
| S3-M6 | The calibration and the gap (§5.4–5.7): fitted parameters with ranges; predictive checks; the classifier test; gate G-T | H6 answered; `docs/step_3/FIDELITY.md` filled in |
| S3-M7 | The dataset (§6): `twin_v1` generated and verified; the nested parts; `curve.py`; `--extend`; the dataset card | `verify-dataset` passes; `curve.py`'s tests pass; the dataset's size and coverage are in `docs/RESULTS_STEP3.md` |
| S3-M8 | The model kit (§7): the stored features, the contract, `score-model`, RIGID and the chain helper, `targets.py`, the dummy models, the tests, and the guide `LEARN_THE_MODEL.md` | `Zeros` scores exactly as B0; `Cheat` fails the label-blindness test; every test for Danyal's code skips cleanly; every worked example in the guide is reproduced by a test |
| S3-M9 | The numbers to beat (§8). One command, for example `python scripts/make_results_step3.py`, writes `docs/RESULTS_STEP3.md` and its figures: the baselines and the oracle, the page-level test, and (Should) the retro path's converter and "before" measures | every table reproducible from one command; the banner present |
| S3-M10 | `docs/step_3/` (§12); the documentation updates (§11); the report (§13); tag `step3` | every file exists; every link and `path:line` resolves; every worked example is reproduced by a test; the full check passes |

## 10. Checks that must pass

Add them to `mesqsim/validation.py`, so they appear in `docs/VALIDATION.md` and run in `scripts/check_all.sh`. Every
Step 1 and Step 2 check must keep passing. Tests that need the recordings are marked `@pytest.mark.real`, and skip
cleanly without them.

1. **The page twin** matches the real page code (§4.10).
2. **Physics:** the four checks of §4.3.
3. **Fusion:** no error gives a constant ψ; a bias gives ψ = −b·t; a scale error gives the turn-proportional term;
   `analytic` mode with η = 0 equals Step 1's ψ.
4. **Labels:** the twist rule's hand check; δ and ε recomputed from ψ match the tables.
5. **The wire:** the golden vector; `json` mode with the fitted decimals reproduces Nov 4's exact-repeat share within
   5 points.
6. **The phone:** no message longer than 95 bytes; a zero is nudged; the page twin's zero guard behaves as the page's.
7. **Invariances** (§4.12): the common turn, the phone included; one pod alone.
8. **Joins:** the §4.9 check.
9. **Determinism:** the same seed gives identical session files and an identical dataset manifest.
10. **The analysis is reproduced:** §5.3 (`real`).
11. **The dataset:** the manifest matches; no subject in two splits; windows causal, never crossing a session or a
    split; the stored features equal a streaming recomputation.
12. **The model kit:**
    - C5: the stored inputs are invariant to a common turn, to machine precision;
    - `Zeros` meets the contract and scores exactly as B0, and `Cheat` fails the label-blindness test;
    - the loss reference values agree with SciPy;
    - the tests for Danyal's code (§7.4) run, and skip cleanly while their modules do not exist.
13. **RIGID:** the 7° example; exact recovery on noise-free rigid motion; κ = 0 for a rotation about the vertical
    only.
14. **Reproducibility:** the same config and seed give an identical `docs/RESULTS_STEP3.md`, apart from its timestamp
    line.
15. **The explanations:** `tests/test_explain_links.py` and `tests/test_explain_examples.py` cover `docs/step_3/`.

## 11. Documentation updates

Write for a reader who knows the project but not your code. Define every symbol, unit and file format, and put the
picture first.

- `README.md`: the picture (§1), Step 3's status, a quickstart from a fresh clone to the Step 3 results, and a link to
  `docs/step_3/README.md` as the place to start.
- `docs/SIMULATOR.md`: the twin. It covers its modes, how to run one session, and how to add a sensor error, a slip or
  a motion program.
- `docs/DATA.md`: the recordings' contract (§2.3, §5.1), and the dataset's layout and versions.
- `docs/PIPELINE.md`, `docs/MATH.md` and `docs/CODE.md`: the new stages, formulas and modules.
- `docs/MUST_KNOW.md`, a Step 3 section:
  - what the twin may and may not be used to claim;
  - how thin the calibration is: seven recordings;
  - why ε is observable and north is not;
  - how to tell that the twin, the dataset or a model is broken;
  - ten new self-check questions, with answers.
- `docs/analysis_plan_DRAFT.md`: H6 and H7, the recording split, the dataset-size rule, and TCN-P's contrasts, all
  marked DRAFT for Danyal to freeze.
- `docs/VALIDATION.md` and `docs/RESULTS_STEP3.md`: generated, never edited by hand.
- `WORKLOG.md`: one entry per milestone, as before.

## 12. `docs/step_3/`: the explanations, ready for the report

Danyal asked for this directory by name. It explains Step 3 and carries everything the course report needs to cite.
Write it in the style of a CS50 lecture:
- start from a concrete picture;
- define every term at its first use;
- show every step of every derivation;
- work small numerical examples by hand;
- explain the code line by line where it matters;
- end with self-check questions and their answers.

```text
docs/step_3/
  README.md               the entry point: the picture, the reading order, which file answers which question
  MENTAL_PICTURE.md       the close-your-eyes picture of the twin, the data and the two networks; no equations
  PIPELINE_EXPLAINED.md   one session, from an AMASS subject to a stored example, a scored model and a results table
  MATH_EXPLAINED.md       the mathematics, derived step by step
  CODE_EXPLAINED.md       the programming
  THEORY.md               the ideas, from first principles to the literature
  REFERENCES.md           every source used for the code and the mathematics, verified, with where each is used
  references.bib          the same references in BibTeX, keys matching REFERENCES.md
  FIDELITY.md             the fidelity ledger (§4)
  DATASET_CARD.md         twin_v1 (Gebru et al. 2021), with the full specification a loader can rely on
  LEARN_THE_MODEL.md      the guide: from the first data file to a trained, scored TCN-P (§7.5)
  MODEL_CARD_TEMPLATE.md  for Danyal to fill in for each model they train (Mitchell et al. 2019)
```

**The reading order:**

```text
MENTAL_PICTURE ─► PIPELINE_EXPLAINED ─► MATH_EXPLAINED ─► CODE_EXPLAINED ─► THEORY ─► REFERENCES
 "what happens?"   "what happens to      "why does it      "how is it         "where do   "what do I cite?"
                    the data?"            work?"            built?"            the ideas
                                                                               come from?"
```

`LEARN_THE_MODEL.md` comes last, when Danyal starts writing the models.

1. **`README.md`** holds:
   - the pipeline above;
   - what each file answers, and roughly how long it takes to read;
   - which file to open for a given question (for example, "why does a bias drift the heading?" → MATH, fusion);
   - a link to each file. Each file links back here and on to the next one.
2. **`MENTAL_PICTURE.md`**: words, one diagram and one data in/out table. It covers:
   - the real suit and its slips;
   - the twin as a flight simulator with the real cockpit;
   - the body's skin and the pods on it;
   - what a gyroscope feels and why a north slides;
   - the phone as the one shared compass for heading, but not north;
   - the page twin, and why its BVH must look like Danyal's;
   - the recordings as the judge;
   - the data as a library whose size a learning curve chooses;
   - the two networks: per joint, and per pod relative to the phone;
   - what goes in, what comes out, and what never can.
3. **`PIPELINE_EXPLAINED.md`.** Follow one twin session, from an AMASS subject to `docs/RESULTS_STEP3.md`, and one real
   recording through `mesqsim/real/`.
   - **The stages:** the motion program; body and sites; physics; sensors; fusion; firmware and radio; the phone and
     its app; the procedure; the files; the page twin; the statistics; calibration; the store; features; the scorer;
     the page-level test; the retro path. Say where Danyal's model will plug in.
   - **For every stage:**
     - the command;
     - the input files, with their schemas and units;
     - the `module:function` that runs;
     - what it computes, in words and in one formula;
     - the output files;
     - a worked example with the real numbers of one session you ran (name the command, seed and commit);
     - what can go wrong, and which check catches it.
4. **`MATH_EXPLAINED.md`.** Every implemented formula names the function that implements it and the test that checks
   it. Its worked example is reproduced by a test. The topics:
   - **rotations:** quaternions, Exp and Log, slerp, rotation splines, and the swing–twist split behind the ψ label;
   - **the body:** linear blend skinning, surface frames, and why skin turns against bone;
   - **what sensors feel:** ω = (RᵀṘ)^∨ and f = Rᵀ(p̈ − g), derived; numerical differentiation and noise;
   - **sensor errors:** scale, misalignment, bias, angle random walk, bias instability; how datasheet numbers and the
     Allan variance map onto the model's parameters;
   - **six-axis fusion:** the complementary-filter idea, then VQF and Mahony in outline; why heading cannot be observed
     without a magnetometer; a bias's vertical component becoming ψ = −b·t; the mirrored pair; scale error × turning;
   - **the phone:** its frames, the Zero and the error model, pointing to the app's `MATH.md` for the lever arm;
   - **the page:**
     - world deltas;
     - a tree map as the conjugation P·R·Pᵀ when it is a rotation, and what a mirror does instead;
     - local = parent⁻¹ · child;
     - the per-message slerp as an exponential filter with τ = −Δt / ln(1 − α), with a worked example: α = 0.15 at
       32 Hz gives τ = 0.192 s;
     - the root's fixed point (100·p + c)/2, derived;
     - Euler XYZ in the BVH;
     - Δ_drawn = Y(ψ)ᵀ·Δ_true·Y(ψ), and why a step becomes a sideways swing;
   - **the retro path:** the virtual-pod proof (§8.4);
   - **the statistics:** PCA swing direction as an axis, circular means, the Umeyama similarity fit, Pearson's r;
   - **calibration:** simulated moments and ABC rejection, distances, posteriors, predictive checks; why an AUC of 0.5
     means "cannot tell apart"; MMD;
   - **the data:** learning curves and their power-law fit; coverage; leakage; why the split is by subject;
   - **the networks,** which Danyal implements (for each formula, name the test of §7.4 that will check it):
     - the 6-D rotation encoding and why continuity matters;
     - causal dilated convolutions and the receptive-field arithmetic (253);
     - the parameter counts, worked: 142,595 and 145,667;
     - the von Mises likelihood, its loss, and stable log I₀;
     - RIGID's closed form (the 7° example);
     - the chain, and its approximate κ;
     - calibration and coverage, and selective risk;
   - **statistics:** the paired subject bootstrap, Holm's correction and practical margins;
   - self-check questions with answers.
5. **`CODE_EXPLAINED.md`, the programming.**
   - **A map** of every new module (`mesqsim/real`, `mesqsim/twin`, `mesqsim/learn`, `tools/page_golden`): what each
     owns and who calls it, as a diagram.
   - **Four commands traced end to end**, function by function, with `path:line` links: one twin session; the
     statistics of one recording; generating and verifying the dataset; scoring a model with `score-model`, using
     `Zeros`.
   - **The key code, line by line**, CS50 style: the twist label; the sensor model; the fusion loop; the app's message;
     the page port and its golden test; the ABC loop; the store; the scorer; RIGID; the tests for Danyal's code. For
     each block:
     - what it does, and why;
     - what would break if it were wrong;
     - which test guards it.
   - **The tests:** what each test file proves.
   - **How to extend:** add a sensor error, a slip, a motion program, a statistic, a model or a recording.
   - **How to debug:** frame mix-ups (+Y and +Z up; pod, segment and surface), signs, units, label leaks, causality
     leaks, and a twin that drifts from the page.
   - Self-check questions with answers.
6. **`THEORY.md`** cites the full reference (authors, year, title, venue, DOI) for every claim:
   - simulation-to-reality transfer and domain randomisation;
   - synthetic inertial data from motion capture (DIP, TransPose, IMUSim, IMUTube);
   - system identification and simulation-based inference;
   - observability of relative heading, and of heading relative to an anchor;
   - inertial sensor error models;
   - six-axis fusion;
   - temporal convolutional networks, and how they compare with recurrent networks;
   - directional statistics and probabilistic outputs on the circle;
   - calibration of predicted uncertainty;
   - data scaling;
   - evaluation design;
   - a bridge to Step 4: fusing two von Mises estimates.
7. **`REFERENCES.md` and `references.bib`, for the report.**
   - **Every source used anywhere in Step 3's code or mathematics:** papers, datasets (AMASS **and each subset used:
     CMU, KIT, BMLmovi, HumanEva**, which AMASS asks users to cite), body models (SMPL, SMPL+H), the datasheet, the
     software (NumPy, SciPy, PyTorch, h5py, smplx, vqf, three.js), the `mesquite-mocap` repository at its commit, and
     Danyal's recordings as internal data with their dates.
   - **For each:** a BibTeX key; the full citation; the DOI, arXiv id or URL; its licence (data and software); what it
     was used for; and where (`path:line`, or a section of these files).
   - **Verification.** Open each DOI or arXiv id, confirm the title, authors and year, and record "verified
     <date>". A reference you cannot verify is marked so, and nothing rests on it.
   - **A table: report section → keys.** The sections are the twin, calibration, data, models, evaluation and
     statistics. Danyal can then cite each part of the method in one step.
   - **`references.bib`** holds the same entries, keys identical; a test checks that the two files agree.
8. **`FIDELITY.md`, `DATASET_CARD.md`, `LEARN_THE_MODEL.md` and `MODEL_CARD_TEMPLATE.md`**, as described in §4,
   §6.4 and §7.

**Rules for all these files:**
- Every number comes from a command you ran (name it, with the commit) or from a cited source.
- Every code reference is `path:line` or a line range, points at the current code, and is checked by
  `tests/test_explain_links.py`.
- Nothing contradicts `docs/MATH.md` or the code. Where they disagree, fix the one that is wrong and say so in
  `WORKLOG.md`.

## 13. When you stop: the report

Post the report to the person running the session, and add it as the last `WORKLOG.md` entry:

1. What was built, as one list of files.
2. The output of `scripts/check_all.sh`.
3. **The twin:** the reproduction of §5.3; the fitted parameters with their ranges; the gap (predictive checks and the
   classifier test, the twin against Step 1); G-T; H6.
4. **The data:** hours, sessions, subjects, strata and disk; where it is stored, and its manifest hash; how to extend it.
5. **The numbers to beat:** the baselines and the oracle, with their banner; the page-level test; the retro path's
   "before" measures. Also say how Danyal starts: the first lesson of `LEARN_THE_MODEL.md`, and the command that scores
   their model.
6. Every assumption still in the config, and every `fitted` value with its command.
7. What is blocked, and on whom.
8. **What Danyal must decide:**
   - freezing the recording split, the dataset-size rule and the analysis plan;
   - the disclosure question;
   - whether `twin_v1` is good enough to keep, or Step 5's bench should come first;
   - the next OptiTrack session (a marker triad per pod, three markers on the phone, named markers, the sync event);
   - whether the pod firmware should send the raw gyroscope and accelerometer streams the chip already produces
     (enabled but never read, `Pod_Watch_Binary.ino:476-480`), which would give the networks accelerations.
9. Where the explanations are: `docs/step_3/`.
10. A proposed plan for Step 4, fusion. Do not start it.

## Appendix: reading list

Verify each entry before citing it (§12, item 7). The Step 2 brief's list still applies (Seel 2014; Olsson 2020;
Laidig 2019 and 2022; Lehmann 2020; Kok 2022; Kok 2026; Cereatti 2024; Laidig & Seel 2023; Bachhuber 2025; Mardia &
Jupp 2000).

**Bodies and motion data**
- N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, M. J. Black (2019). AMASS: Archive of motion capture as surface
  shapes. *ICCV 2019*, 5442–5451. doi:10.1109/ICCV.2019.00554
- M. Loper, N. Mahmood, J. Romero, G. Pons-Moll, M. J. Black (2015). SMPL: A skinned multi-person linear model. *ACM
  Transactions on Graphics* 34(6), 248. doi:10.1145/2816795.2818013
- J. Romero, D. Tzionas, M. J. Black (2017). Embodied hands: Modeling and capturing hands and bodies together. *ACM
  Transactions on Graphics* 36(6), 245. doi:10.1145/3130800.3130883 *(SMPL+H)*
- The subsets used: CMU Graphics Lab Motion Capture Database (mocap.cs.cmu.edu); C. Mandery et al. (2015), the KIT
  whole-body human motion database, *ICAR 2015*; S. Ghorbani et al. (2021), MoVi, *PLOS ONE*; L. Sigal, A. O. Balan,
  M. J. Black (2010), HumanEva, *International Journal of Computer Vision* 87, 4–27. *(Find and verify the DOIs.)*
- A. R. Punnakkal et al. (2021). BABEL: Bodies, action and behavior with English labels. *CVPR 2021*. *(Could, §4.9)*

**Synthetic inertial data from motion capture**
- Y. Huang, M. Kaufmann, E. Aksan, M. J. Black, O. Hilliges, G. Pons-Moll (2018). Deep inertial poser: Learning to
  reconstruct human pose from sparse inertial measurements in real time. *ACM Transactions on Graphics* 37(6), 185.
  doi:10.1145/3272127.3275108
- X. Yi, Y. Zhou, F. Xu (2021). TransPose: Real-time 3D human translation and pose estimation with six inertial
  sensors. *ACM Transactions on Graphics* 40(4), 86. doi:10.1145/3450626.3459786
- A. D. Young, M. J. Ling, D. K. Arvind (2011). IMUSim: A simulation environment for inertial sensing algorithm design
  and evaluation. *IPSN 2011*, 199–210.
- H. Kwon et al. (2020). IMUTube: Automatic extraction of virtual on-body accelerometry from video for human activity
  recognition. *Proceedings of the ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies* 4(3), 87.
  doi:10.1145/3411841

**Inertial sensing and fusion**
- M. Kok, J. D. Hol, T. B. Schön (2017). Using inertial sensors for position and orientation estimation. *Foundations
  and Trends in Signal Processing* 11(1–2), 1–153. doi:10.1561/2000000094
- N. El-Sheimy, H. Hou, X. Niu (2008). Analysis and modeling of inertial sensors using Allan variance. *IEEE
  Transactions on Instrumentation and Measurement* 57(1), 140–149. doi:10.1109/TIM.2007.908635
- R. Mahony, T. Hamel, J.-M. Pflimlin (2008). Nonlinear complementary filters on the special orthogonal group. *IEEE
  Transactions on Automatic Control* 53(5), 1203–1218. doi:10.1109/TAC.2008.923738
- S. O. H. Madgwick, A. J. L. Harrison, R. Vaidyanathan (2011). Estimation of IMU and MARG orientation using a gradient
  descent algorithm. *IEEE ICORR 2011*. doi:10.1109/ICORR.2011.5975346
- TDK InvenSense. ICM-20948 datasheet, DS-000189, revision 1.3.

**Rotations, skeletons and files**
- J. Solà (2017). Quaternion kinematics for the error-state Kalman filter. arXiv:1711.02508
- J. Solà, J. Deray, D. Atchuthan (2018). A micro Lie theory for state estimation in robotics. arXiv:1812.01537
- K. Shoemake (1985). Animating rotation with quaternion curves. *SIGGRAPH '85*, *Computer Graphics* 19(3), 245–254.
  doi:10.1145/325165.325242
- M. Meredith, S. Maddock (2001). Motion capture file formats explained. University of Sheffield, Department of
  Computer Science, CS-01-11. *(BVH)*
- S. Umeyama (1991). Least-squares estimation of transformation parameters between two point patterns. *IEEE
  Transactions on Pattern Analysis and Machine Intelligence* 13(4), 376–380. doi:10.1109/34.88573

**Calibration of simulators and the gap to reality**
- M. A. Beaumont, W. Zhang, D. J. Balding (2002). Approximate Bayesian computation in population genetics. *Genetics*
  162(4), 2025–2035. doi:10.1093/genetics/162.4.2025
- D. McFadden (1989). A method of simulated moments for estimation of discrete response models without numerical
  integration. *Econometrica* 57(5), 995–1026. doi:10.2307/1913621
- C. Gouriéroux, A. Monfort, E. Renault (1993). Indirect inference. *Journal of Applied Econometrics* 8(S1), S85–S118.
  doi:10.1002/jae.3950080507
- J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, P. Abbeel (2017). Domain randomization for transferring deep
  neural networks from simulation to the real world. *IROS 2017*, 23–30. doi:10.1109/IROS.2017.8202133
- D. Lopez-Paz, M. Oquab (2017). Revisiting classifier two-sample tests. *ICLR 2017*. arXiv:1610.06545
- A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, A. Smola (2012). A kernel two-sample test. *Journal of
  Machine Learning Research* 13, 723–773.

**Learning**
- S. Bai, J. Z. Kolter, V. Koltun (2018). An empirical evaluation of generic convolutional and recurrent networks for
  sequence modeling. arXiv:1803.01271
- A. van den Oord et al. (2016). WaveNet: A generative model for raw audio. arXiv:1609.03499
- Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li (2019). On the continuity of rotation representations in neural networks.
  *CVPR 2019*, 5745–5753. arXiv:1812.07035
- S. Prokudin, P. Gehler, S. Nowozin (2018). Deep directional statistics: Pose estimation with uncertainty
  quantification. *ECCV 2018*. arXiv:1805.03430
- C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger (2017). On calibration of modern neural networks. *ICML 2017*, PMLR 70,
  1321–1330.
- I. Loshchilov, F. Hutter (2019). Decoupled weight decay regularization. *ICLR 2019*. arXiv:1711.05101
- J. Hestness et al. (2017). Deep learning scaling is predictable, empirically. arXiv:1712.00409
- K. He, X. Zhang, S. Ren, J. Sun (2016). Deep residual learning for image recognition. *CVPR 2016*, 770–778.
  doi:10.1109/CVPR.2016.90
- L. Beyer, A. Hermans, B. Leibe (2015). Biternion nets: Continuous head pose regression from discrete training labels.
  *GCPR 2015*.
- A. Kendall, Y. Gal (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS
  2017*. arXiv:1703.04977

**Documentation and statistics**
- T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, K. Crawford (2021). Datasheets for
  datasets. *Communications of the ACM* 64(12), 86–92. doi:10.1145/3458723
- M. Mitchell et al. (2019). Model cards for model reporting. *FAT\* 2019*, 220–229. doi:10.1145/3287560.3287596
- B. Efron, R. J. Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- S. Holm (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics* 6(2),
  65–70.

**Software** (cite the versions pinned in `requirements-lock.txt`)
- C. R. Harris et al. (2020). Array programming with NumPy. *Nature* 585, 357–362. doi:10.1038/s41586-020-2649-2
- P. Virtanen et al. (2020). SciPy 1.0: Fundamental algorithms for scientific computing in Python. *Nature Methods*
  17, 261–272. doi:10.1038/s41592-019-0686-2
- A. Paszke et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
- three.js r157 (MIT licence), as vendored in `mesquite-mocap/build-static/`.

**Learning resources.** These feed the guide's "Study first" lines (§7.5), not the citations. Every link was checked
on 2026-11-17.
- The Deep Learning Specialization: <https://www.coursera.org/specializations/deep-learning>. Its courses are under
  `https://www.coursera.org/learn/`: `neural-networks-deep-learning`, `deep-neural-network`,
  `machine-learning-projects`, `convolutional-neural-networks` and `nlp-sequence-models`.
- Karpathy:
  - *Neural Networks: Zero to Hero*: <https://karpathy.ai/zero-to-hero.html>;
  - *A Recipe for Training Neural Networks*: <https://karpathy.github.io/2019/04/25/recipe/>.
- The Amidis:
  - the CS 230 cheatsheets: <https://stanford.edu/~shervine/teaching/cs-230/>;
  - the CS 229 refreshers: <https://stanford.edu/~shervine/teaching/cs-229/refresher-algebra-calculus> and
    <https://stanford.edu/~shervine/teaching/cs-229/refresher-probabilities-statistics>;
  - the data post: <https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel>.
- PyTorch:
  - *Learn the Basics*: <https://docs.pytorch.org/tutorials/beginner/basics/intro.html>;
  - *Writing Custom Datasets*: <https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html>;
  - `torch.special`: <https://docs.pytorch.org/docs/stable/special.html>.
- Others:
  - h5py: <https://docs.h5py.org/en/stable/quick.html>;
  - d2l.ai: <https://d2l.ai/>;
  - 3Blue1Brown: <https://www.3blue1brown.com/topics/linear-algebra>;
  - quaternions: <https://eater.net/quaternions>.
