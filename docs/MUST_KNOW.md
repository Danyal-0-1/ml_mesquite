# Must know: using this repository safely

*Status: Steps 1 and 2 complete (tag `step2`, S2-M8). §10 is the Step 2 section.*

Read this before trusting any number the repository produces. Each section is short; the self-check questions at the
end test whether it stuck.

## 1. The simulator is a model, and most of its parameters are guesses

The simulator turns motion into the frames the suit would send, with errors we put in ourselves. Its numbers are only
as good as its parameters.

- `python -m mesqsim params` lists every parameter with its value, label and source. On 2026-11-13 it reported
  **44 parameters: 3 datasheet, 3 firmware, 38 assumption, 0 measured**.
- `datasheet` values (chip rate, scale factor, angle random walk) are typical values from the ICM-20948 datasheet,
  as quoted by the Step 1 brief. They are planning bounds, not measurements of these pods.
- Every `assumption` row (drift, box-slot error, reboots, losses, timing, mountings…) waits for Danyal's bench
  measurements. The sweeps in [RESULTS_STEP1.md](RESULTS_STEP1.md) show how much each guess matters.
- The first real-suit measurement (a static hold on 2026-11-04, Step 2 brief §4.1) is not in the config yet: 12 of 13
  pods stayed within 1.8° for 17.3 minutes, one drifted −2.0 °/min for 7 minutes, and a USB plug or unplug rebooted a
  pod each time.
- So: a simulated result says what *would* happen *if* the suit behaved as configured. It is not a measurement.

## 2. The room's heading is unknowable

Turn the whole room, with the wearer in it, about the vertical, and every pod sends exactly the same numbers. So no
method can ever recover north or the body's facing in the room. Only headings **relative to one another** (δ per
joint) can be estimated. Check 4.2 tests it: turning every north by 73° moves the labels by 1.5e-13° and B1 by
0.0039° ([VALIDATION.md](VALIDATION.md)). Never claim to measure north or the body's heading.

## 3. Labels are exact only in simulation

- `labels*.parquet` is the simulator's truth: per pod and send tick (true sample time, received, fresh, ψ_i) and per
  joint and 32 Hz step (true δ). `meta.json` holds every pod's drawn parameters.
- A real recording has no labels. Anything that reads labels or `meta.json` cannot run on the real suit.
- Only the scorer and the oracles (O1, O2) may read them. Step 0 and every deployable estimator never do.

## 4. The quaternion order and the signs

- In code, quaternions are `(x, y, z, w)`, scalar last, as in `scipy.spatial.transform.Rotation`. On the wire the
  fields are also `qx, qy, qz, qw`. **w is last.** Many tools (and most papers) put w first.
- On the wire **w ≥ 0** always (the pod rebuilds w from x, y, z). q and −q are the same rotation.
- Products are Hamilton; `quat_mul(a, b)` applies b first, then a. Orientations map body → world. The world is +Z up.
- Y(α) turns +x towards +y. Heading is the first angle of intrinsic Z-Y-X. wrap maps into (−180°, 180°].
- All of this lives in [mesqsim/conventions.py](../mesqsim/conventions.py), pinned by tests. Never re-implement it.
- **The sign check:** ψ_parent = 12°, ψ_child = 5° gives δ = −7°. A sign error leaves a 14° twist (check 3.1).

## 5. The leakage rules

| rule | why |
|---|---|
| split by **subject**, within each subset; a subject is in exactly one split | a model would otherwise recognise the person, not the joint |
| every window and corruption variant of a sequence stays in its sequence's split (check 9.3) | variants of one clip are near-copies |
| TotalCapture is never trained or fitted on | it may be a later test set |
| the split manifest is frozen once written; changing it needs `--force` and Danyal's approval | results must not be chosen by re-splitting |
| fit only on train (Ω) or validation (κ scale, q, hold threshold); never on test | a number tuned on test is not a test result |
| Step 0 and estimators never read labels or `meta.json` | they would not work on the real suit, and would look too good |
| host arrival time is never a sample time | it carries radio latency and jitter; Step 0 uses `ms_lo` |

## 6. The licences

| data | licence | what it means here |
|---|---|---|
| AMASS | non-commercial research licence; redistribution forbidden | each person registers at amass.is.tue.mpg.de and downloads by hand; never commit the data, anything derived from it, or a password |
| SMPL+H body model | the MANO / SMPL+H licence, same rules | register at mano.is.tue.mpg.de |
| DIODEM v3.0 | CC0 1.0 (Dataverse) or CC BY 4.0 (Zenodo) | scripted download; cite Bachhuber et al. 2025 |

Only names, sizes and SHA-256 checksums (`manifests/`) are committed. Data live in `$MESQ_DATA_DIR`, outputs in
`$MESQ_OUTPUT_DIR`, both outside git.

## 7. How to tell that something is broken

- `bash scripts/check_all.sh` does not end with `== ALL CHECKS PASSED`, or
  [VALIDATION.md](VALIDATION.md)'s summary shows a failure.
- **O2 is not exactly 0.** O2 uses the true δ, so any error means the scorer joined estimates and labels wrongly.
- **B1 is better than O1** in expectation. B1 is the true offset difference plus slot noise, so it cannot beat O1
  (check 12.1).
- **Errors near 14°** where 0 was expected: a sign error in δ or in the correction (check 3.1).
- **Errors that double instead of vanishing**: parent and child swapped.
- **About 57-fold errors**: degrees passed to a radian function.
- **Means that jump by 360°**: angles averaged linearly instead of with circular statistics.
- **Step 0 flags disagree with the labels** (fresh, held, lost): a timing or unwrapping bug (check 7.4).
- **Results that are too good under a sweep**: something read the simulator's truth.

## 8. Open questions (on 2026-11-13)

- **AMASS:** HumanEva, BMLmovi, CMU and KIT downloaded and verified on 2026-11-13 (8,143 motion files, 246
  subjects); the split manifest is written (SHA-256 `d36448ad…`, [DATA.md](DATA.md) §3) and waits for Danyal to
  freeze it. Whether to add TotalCapture as a holdout is Danyal's decision.
- **The window length:** with `dataset.window_s` = 20 s only 767 of 8,143 sequences are long enough, and BMLmovi
  gives just 2 windows ([DATA.md](DATA.md) §1.2). Shorter windows or concatenation would change which motions are
  simulated.
- **The analysis plan:** [analysis_plan_DRAFT.md](analysis_plan_DRAFT.md) waits for Danyal to freeze it.
- **The Spine joint:** SMPL spine3 (default) or spine2.
- **The offset mode:** whether the chip's north is `aligned` or `arbitrary` at boot is unmeasured; both are swept.
- **The bench:** every `assumption` row waits for the 70-minute box evening and the taped-board session.
- **Real recordings:** the suit still writes the JSON path; no recording exists in the frames-table format yet.

## 9. Ten self-check questions

1. **What does a repeated `ms_lo` mean, and what does a gap in `count` mean?**
   A repeated `ms_lo` is a **held** frame: the pod resent its last sample because no new one was decoded (a stall of
   the sample task). A gap in `count` means frames **lost** on the radio.
2. **Why is B1 not zero?**
   Box registration only removes the offsets at the box, up to each pod's slot error. Everything accrued since then
   (drift, angle random walk, scale-factor error times turning) remains, and after a reboot its estimate is stale. On
   the provisional synthetic run, B1's mean error at 10 min since boot is 6.70° ([RESULTS_STEP1.md](RESULTS_STEP1.md),
   Table 1).
3. **Which parameters are measured and which assumed?**
   None is measured yet. Of 44: 3 datasheet, 3 firmware, 38 assumption (`python -m mesqsim params`).
4. **Why can no model recover north?**
   Turning the whole room about the vertical changes no report, so every north gives the same data. Only relative
   headings are observable.
5. **In which order is a quaternion stored, and why is w never negative on the wire?**
   `(x, y, z, w)`, scalar last, in code and on the wire. The pod rebuilds w = √(1 − x² − y² − z²) ≥ 0; q and −q are
   the same rotation, so nothing is lost.
6. **ψ_parent = 12° and ψ_child = 5°. What is δ, and what does a sign error leave?**
   δ = wrap(5° − 12°) = −7°. Correcting with the wrong sign leaves a 14° twist.
7. **Why must O2's error be exactly 0, and what does it mean if it is not?**
   O2 uses the true δ, so its error is 0 by construction. A non-zero O2 means the scorer matched the wrong labels to
   the estimates: a pipeline bug, not a modelling result.
8. **Why must no subject appear in two splits?**
   A model could learn the person's own way of moving and look better than it would on a stranger. The unit of
   analysis is the held-out subject.
9. **Can host arrival time be used as the sample time?**
   No. It includes radio, hub and USB latency and jitter. Step 0 uses the pod's `ms_lo`; host time only resolves
   wraps and reboots.
10. **What happens to B1 when a pod reboots?**
    The pod is silent for a few seconds, `count` and `ms_lo` restart near 0 and it picks a new north. Step 0 starts a
    new epoch, but B1 still uses the old box registration, so its error jumps. In the provisional sweep with one
    reboot per pod per hour, B1's mean error is 29.89° against 6.99° without reboots
    ([RESULTS_STEP1.md](RESULTS_STEP1.md), Table 4).

---

## 10. Step 2: the geometric estimator

### 10.1 GEO-K is a ceiling, never a result for the suit

GEO-K receives the simulator's **true mountings** (how each pod sits on its segment) through its one `truth` argument,
so it knows the anatomical axes in every pod's frame. The real suit can never know them. Use GEO-K only to ask
*how much could geometry do with perfect axes?*; the gap between GEO-K and GEO-ID is the price of finding the axes
ourselves. Every table stars it (`USES_TRUTH`: O1, O2, GEO-K). Never quote a GEO-K number as what the suit achieves.

### 10.2 The blind spot and the sign

- **The blind spot.** The knee estimate compares the shadows of the knee axis on the floor, seen from the thigh and
  from the shin. When the knee axis stands vertical (lying on one's side, some kneeling poses), the shadow is a dot:
  the window carries no information and κ = 0, by construction (check 22.1).
- **The sign.** An axis u and its opposite −u fit the motion equally well. A child axis flipped against the parent's
  turns δ by 180°. Votes decide it: the vertical rule (the true axis has the same vertical component in both pods)
  and the T-pose rule (the wrong sign would mean the knee turned half round since the T-pose); with the T-pose tie the
  relative sign is right by construction. Without a vote the window is *ambiguous* and gets κ = 0 (check 24.1).
- **GEO-ID needs a knee first.** The facing comes from the knee axes, the T-pose frames from the facing, and the
  range of motion of every other joint from the T-pose frames. Where the knee axes are refused (the motion does not
  identify them), GEO-ID *is* B1 at every joint, by construction. Table 3b of
  [RESULTS_STEP2.md](RESULTS_STEP2.md) says how often that happened on test.

### 10.3 Why κ is near zero at ball joints

A hip, shoulder, lower back or neck has no pin, only walls: its range of motion Ω (fitted on the training subjects).
While the motion stays inside the walls, every δ is equally consistent with "the joint is inside its range": the
cost is flat and κ = 0 (check 26.1). Information appears only near a wall, so the 3-DoF results depend on how often
everyday motion reaches its limits — the excitation strata of the results. Expect κ ≈ 0 most of the time there;
it is the honest answer, not a bug.

### 10.4 Everything is provisional

- **Descriptive only.** Danyal has not frozen the split or the analysis plan yet, so every Step 2 comparison is
  descriptive, whatever its p-value. The banner at the top of RESULTS_STEP2 says so.
- **The suit is still assumed.** Most error-model parameters are assumptions until the bench measures them; the
  estimator's fitted beliefs (κ scales, model errors, q, κ_hold) were fitted on simulated validation sessions under
  those assumptions. A different real drift or turning changes them.
- **SMPL is not a person.** The AMASS knees are SMPL knees (not exact hinges); the calibration fitted there transfers
  poorly to DIODEM's mechanical hinges (Table 10 shows both settings).
- **Some choices were made after the first results**: the tied knee-axis threshold and the κ_hold fallback were
  chosen on validation data, but the problems were first noticed in a first test run and in test-split sessions used
  for the explanation. WORKLOG (S2-M6) records each choice, its validation evidence and what prompted it; a frozen
  plan should fix them before any confirmatory test.

### 10.5 How to tell that the estimator is broken

- **GEO-ID differs from B1 on a session with no information** (no knee motion): it must equal B1 exactly
  (`test_geo_id_equals_b1_on_noise_free_motion_without_information`, check 27.1).
- **A common yaw of the room changes GEO's output** (check 28.1), or **deleting later frames changes earlier answers**
  (causality, check 21.1), or **deleting the labels changes anything** (label leak, check 21.3).
- **A knee off by almost exactly 180°**: a sign flip (look at `sign_tally` in the window diagnostics).
- **Coverage far from 68 % and 95 %** on validation (check 30.1): the calibration no longer matches the solvers;
  refit it with `fit-geometry calibration` after any solver change.
- **Pair timing beyond 15 ms** (check 31.1): the clock map is wrong.
- **GEO-K worse than B1 at hinges with drift on and fusion noise off** (check 29.1): the solver or its axes are wrong.
- **GEO-ID suspiciously close to O2**: it is reading truth somewhere.

### 10.6 Ten Step 2 self-check questions

1. **What does GEO-K know that GEO-ID does not, and what is GEO-K for?** The true mountings, hence the true joint
   axes in the pods' frames. It is the ceiling of the geometry, not a deployable result.
2. **Why is the knee's estimate blind when its axis is vertical?** The method compares the axis's horizontal
   shadows; a vertical axis has no shadow, so I = 0 and κ = 0.
3. **What does a flipped knee axis do to δ, and what prevents it?** It turns δ by 180°. The vertical and T-pose
   votes, and the T-pose tie of the two axes.
4. **Why can GEO-ID say nothing about a hip until a knee has been identified?** The hip's range of motion needs the
   T-pose frames, which need the facing, which comes from the knee axes.
5. **Why is κ ≈ 0 at a shoulder in most everyday motion?** The range-of-motion cost is flat inside the limits: every
   δ fits equally until the motion reaches a wall.
6. **What does κ = 0 mean in the filter?** No information: the window changes nothing, and the joint keeps its
   previous belief (at the start, box registration).
7. **What happens to a joint's belief when one of its pods reboots?** κ drops to 0 at the first frame of the new
   boot epoch; the geometry must find the new δ from scratch, as the reboot table shows.
8. **Why was κ recalibrated, and on which data?** The raw window κ treats correlated samples as independent and is
   far too confident; the scale and model error per joint type were fitted on the **validation** subjects only.
9. **Why are the Step 2 p-values not evidence for or against H2 yet?** The plan and the split are not frozen:
   everything is descriptive until Danyal freezes them.
10. **GEO-ID equals B1 at every joint of a session. Is that a bug?** Usually not: the knee axes were refused (the
    motion did not identify them), so nothing else could run. It is a bug only if the knees were clearly identifiable
    (GEO-K's windows informative and the contrast low) — then look at the identification thresholds.
