# Analysis plan (DRAFT: for Danyal to approve and freeze)

> **DRAFT. Not frozen.** Nothing here is a result. Every item marked **proposal** is a choice made by the session so
> that the plan is complete; Danyal accepts, changes or rejects it. Once frozen, this file is renamed
> `analysis_plan.md`, its SHA-256 goes in `WORKLOG.md`, and any later change is a dated amendment at the end.
> Sources: Step 1 brief §2, §5.1, §4.7; Step 2 brief §3, §6. A number without a source here is a proposal.

## 1. Picture first

We want to know **where joint geometry can recover relative heading δ from Mesquite's fused orientations, and where it
cannot**, before any learned model exists. Every method is scored on simulated sessions whose truth we know, built from
AMASS motion of people who never appear in any fitting. One person is one observation. A method "wins" only if it is
better by more than a practical margin, with an interval that excludes zero.

## 2. Research question and hypotheses

**Research question** (Step 2 brief §3). Using only the fused orientations Mesquite already transmits, where does a
small causal temporal network recover relative heading better than explicit joint geometry, at which joints and in
which motions? And does combining the two, each weighted by its calibrated confidence, beat both?

| ID | Hypothesis (Step 2 brief §3) | tested in | data |
|---|---|---|---|
| H1 | after box registration, the real suit's relative-heading error is dominated by drift and spin-proportional error, not by the initial offset | bench, later | the real suit |
| **H2** | **geometry recovers relative heading at hinge and 2-DoF joints (knees, elbows, wrists, ankles) during walking and arm motion, but not at 3-DoF joints (hips, shoulders, lower back, neck) in everyday motion** | **Step 2** | AMASS test subjects (simulated) |
| H3 | a TCN's advantage over geometry is larger at 3-DoF joints than at hinge and 2-DoF joints | Steps 3–4 | AMASS test subjects |
| H4 | fusing geometry and learning is never worse than the better one at any joint type, and better on average | Step 4 | AMASS test subjects |
| H5 | the predicted confidence is calibrated on held-out simulated subjects (68 % and 95 % intervals cover within ±5 points) and falls where errors grow | Steps 2–6 (Step 2: geometry only) | AMASS test subjects |

**The hard limit.** Only headings relative to one another are estimable (turning the room changes no report). No
analysis claims north or the body's heading.

## 3. Data and splits

- **Motion.** AMASS, SMPL+H gender-specific release. Subsets downloaded on 2026-11-13: HumanEva, BMLmovi, CMU, KIT
  (the Step 1 brief §5.1 examples plus one small walking subset): 8,143 sequences, 246 subjects, 23.2 h. Sizes and
  SHA-256 in `manifests/amass_archives.csv`; details in [DATA.md](DATA.md) §1.2.
- **Windows (needs a decision).** Sessions use motion windows of `dataset.window_s` = 20 s (`assumption`). Only 767
  sequences are that long, so BMLmovi (everyday single actions, median 4.6 s) is almost absent, which bears directly
  on H2's "everyday motion". **Proposal:** windows of 8 s, the Step 2 solver window, or one subject's sequences
  concatenated; Danyal chooses before Step 2 results are read.
- **Split** (Step 1 brief §5.1; [mesqsim/splits.py](../mesqsim/splits.py)): by subject, within each subset,
  70 / 15 / 15 % train / validation / test by number of subjects, largest remainder, ordered by a stable hash. Every
  window and every corruption variant of a sequence stays in its sequence's split.
- **TotalCapture** is never trained or fitted on (Step 1 brief §5.1). It is not downloaded yet; if it is, it is a
  `holdout` set, reported separately.
- **The split manifest** is `manifests/split_manifest.csv`; its SHA-256 is recorded in
  `manifests/split_manifest.sha256` and `WORKLOG.md`. **Danyal freezes it** with this plan. After freezing, a change
  needs `split --write --force` and a dated amendment here.
- **DIODEM** (mechanical chains, no people) is out of distribution. The unit there is the experiment. No claim about
  people rests on it (Step 2 brief §6.4).
- **Synthetic motion** (`mesqsim/synth.py`) is for tests and for provisional runs only. Results on it are labelled
  PROVISIONAL (SYNTHETIC MOTION) and are descriptive.

## 4. What may be fitted, and on which split

| quantity | fitted on | label | source |
|---|---|---|---|
| range-of-motion box Ω per 3-DoF joint (e.g. 0.5th–99.5th percentiles of true joint angles) | train | `fitted` | Step 2 brief §5.6 |
| κ scale factor per joint type | validation | `fitted` | Step 2 brief §5.4 |
| filter process noise `geometry.drift_q_rad2_per_s` | validation | `assumption`, then `fitted` | Step 2 brief §5.7 |
| hold threshold `geometry.kappa_hold` | validation | `fitted` | Step 2 brief §5.7 |
| anything else | not fitted; `assumption` with a source note | | Step 2 brief rule 6 |

Nothing is ever fitted, tuned or chosen by looking at the **test** split. Test results are computed once per frozen
configuration.

## 5. Estimand and metrics

- **Target.** δ_pc(t) = wrap(ψ_c(t) − ψ_p(t)) per joint (14 joints, Step 1 brief §3.4) at every 32 Hz step of the
  motion phase.
- **Primary metric** (Step 1 brief §4.7, Step 2 brief §6.2): |wrap(δ̂ − δ)| in degrees, per joint and step.
- **Secondary metric:** parent-relative orientation error, the geodesic angle of each joint's error rotation, computed
  with the simulator's true mountings (and labelled as such).
- **Confidence (H5):** coverage of the 68 % and 95 % von Mises intervals (computed numerically from κ); a reliability
  diagram; selective risk (error against the fraction of steps whose κ is above a threshold).
- **Compute:** time per 32 Hz step for all 14 joints, on a named CPU.

## 6. The unit of analysis and the summary per unit

- **Unit:** the held-out **subject** (AMASS test split). Steps and sessions are never independent observations.
  A subject is the pair (subset, subject id): CMU and KIT reuse the ids 10, 11, 12 and 63 for different people
  (`manifests/split_manifest.csv`), so grouping by the id alone would merge them.
- **Per-subject summary** (**proposal**): for each subject, method and joint group, the **median** of the primary
  error over all steps of all that subject's test sessions. Means and P95 are reported alongside, summarised the same
  way.
- **Joint groups:** hinge (2 knees), 2-DoF (2 elbows, 2 wrists, 2 ankles), 3-DoF (lower back, neck, 2 shoulders,
  2 hips); and each joint alone (secondary).

## 7. Methods compared in Step 2

B0, B1, O1, O2 (Step 1); GEO-K (uses simulator truth: axes and mountings; never a deployable result); GEO-ID (the
deployable estimator); ablations of GEO-ID: no filter; hinge only; windows of 4, 8 and 16 s (Step 2 brief §6.1).

## 8. Primary contrasts for H2

**"Better"** (Step 2 brief §3): the paired difference in per-subject median error has a 95 % bootstrap interval over
held-out subjects that excludes zero, **and** the difference exceeds the practical margin **Δ = 1°** (proposed in the
brief; frozen here).

| contrast | comparison | H2 is supported if | stratum |
|---|---|---|---|
| C1 | GEO-ID vs B1, hinge joints | GEO-ID better than B1 | walking and arm motion |
| C2 | GEO-ID vs B1, 2-DoF joints | GEO-ID better than B1 | walking and arm motion |
| C3 | GEO-ID vs B1, 3-DoF joints | **not** better: the upper end of the interval for the improvement B1 − GEO-ID is below Δ (**proposal**: an equivalence-style reading of "does not recover") | everyday motion |

- **Paired design.** Every method runs on the same sessions (same seeds). The difference is taken per subject, then
  bootstrapped.
- **Bootstrap** (**proposal**): resample subjects with replacement, 10,000 resamples, percentile intervals, one fixed
  seed recorded in the results file.
- **Multiplicity:** Holm's correction across C1–C3 (Step 2 brief §6.3), at family-wise α = 0.05 (**proposal**).
- **Reported at** 10 minutes since boot as the primary time point (**proposal**); 1 and 30 minutes as secondary.
- **Default condition:** the `default` config. Sweeps and presets are secondary analyses.

## 9. Motion strata (needs a decision)

H2 speaks of "walking and arm motion" and "everyday motion". AMASS has no uniform action labels across subsets.
**Proposals**, for Danyal to choose:

1. **Excitation strata (Step 2 brief §6.2), computed by the scorer from the truth:** terciles of the true hinge
   information I per window (hinge and 2-DoF joints), and of the time near the range-of-motion limits (3-DoF joints).
   Objective and available for every sequence. "Walking and arm motion" ≈ the upper two terciles; "everyday motion" =
   all windows.
2. **Name-based categories:** a hand-checked mapping from sequence names (e.g. CMU and KIT file descriptions) to
   walk / arm / other, written once to a committed CSV before any test result is seen.

Until chosen, C1–C3 are reported on all motion windows and on the excitation terciles, descriptively.

## 10. Secondary analyses (descriptive; no multiplicity claim)

- Every method by joint, joint type, time since boot (1, 10, 30 min) and condition: clean, drift, low motion, missing
  bursts, resets (Step 2 brief §6.2).
- The drift sweep σ_b ∈ {0.05, 0.2, 1.0} °/min (Step 1 brief §4.3) plus the stress value 2.0 °/min (Step 2 brief §4.1,
  `assumption`).
- Re-acquisition after a reboot (`stress` preset): error against time since the reboot.
- **H5 (geometry):** 68 % and 95 % coverage per joint type on test, after recalibration on validation; the target
  ±5 points is a result, not a check.
- GEO-K against GEO-ID: the cost of identifying the axes (GEO-K is a ceiling, not a deployable result).
- DIODEM: per joint and experiment, descriptive.

## 11. Rules that protect the analysis

- Estimators never read labels or `meta.json`; GEO-K's truth comes through one named argument (Step 2 brief rule 7).
- The estimator's priors (`geometry:`) never read the simulator's `error_model:` (rule 8).
- Every contrast in §8 is reported whatever its outcome. Negative results count.
- While the motion is synthetic, or the split and this plan are not frozen, results are descriptive and say so.
- A run is identified by (git commit, config hash, seed), recorded in every results file.

## 12. Decisions for Danyal (tick to freeze)

- [ ] The AMASS subsets (HumanEva, BMLmovi, CMU, KIT), and whether to add TotalCapture as a holdout.
- [ ] The split manifest and its SHA-256 (`d36448adb76762251668b9940933e8a14df664ad6d3277fe3a930f7d55049fb3`).
- [ ] The motion window length (§3: 20 s now; 8 s or concatenation proposed).
- [ ] The margin Δ = 1°.
- [ ] The per-subject summary (median) and the primary time point (10 min).
- [ ] The reading of C3 (equivalence-style "not better").
- [ ] The motion strata (§9: option 1, option 2, or both).
- [ ] Bootstrap: 10,000 resamples, percentile intervals; Holm at α = 0.05.
- [ ] The Spine joint: SMPL spine3 (default) or spine2.
