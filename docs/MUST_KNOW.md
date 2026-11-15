# Must know: using this repository safely

*Status: Step 1 complete (S2-M0). A Step 2 section is added when the geometric estimator is done (S2-M8).*

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
