# The pipeline, stage by stage, with the numbers of one real session

*Explanation pipeline, file 2 of 5 · about 45 minutes · [index](README.md) · previous:
[MENTAL_PICTURE.md](MENTAL_PICTURE.md) · next: [MATH_EXPLAINED.md](MATH_EXPLAINED.md)*

[MENTAL_PICTURE.md](MENTAL_PICTURE.md) told the story with watches, postcards and hinge pins. This file follows **one
real session** through all ten stages, from an AMASS file to `docs/RESULTS_STEP2.md`, and then the DIODEM path. At
every stage you get the command, the files that go in (with their columns and units), the function that runs, what
it computes in words and in one formula, the files that come out, the actual numbers of this session, and what can
go wrong (with the check that would catch it).

Two words used throughout. A **pod** is one of the 15 sensors (a "watch"). A **joint** connects a parent pod to a
child pod (14 joints); its **relative heading** δ is how far the child's private north has slid from the parent's,
the one number every method tries to estimate. The left knee is joint 10: parent pod 9 (left thigh), child pod 10
(left shin). We follow it most closely.

---

## The session we follow

```bash
git checkout d6c0487                         # the commit these numbers come from
export MESQ_DATA_DIR=/path/to/data           # the folder that holds amass/ and diodem/
python -m mesqsim simulate --seed 1 \
    --motion "$MESQ_DATA_DIR/amass/CMU/13/13_02_poses.npz" --set session.idle_s=562
# prints the run folder: <output dir>/run_gd6c0487_c5ce0b3f7_s1
python -m mesqsim baselines --run <that folder> --methods B0 B1 O1 O2 GEO-K GEO-ID
```

- **Motion:** `CMU/13/13_02`, a 29 s clip of the AMASS **test** split (its subject is never used for fitting). We
  chose it because its knees move a lot (left-knee flexion varies with a standard deviation of 45°) and because it
  shows the deployable estimator's successes and failures side by side.
- **Seed 1** draws every corruption: mountings, norths, drift, losses.
- **`session.idle_s=562`** puts the motion 10 minutes after the pods were switched on: 5 s in the box, 30 s of
  strapping, a 3 s T-pose, then 562 s of unrecorded waiting, so that drift has time to grow. 10 minutes is the time
  at which H2's contrasts are read.
- The run folder's name says where it came from: commit `d6c0487`, config hash `5ce0b3f7` (the default config with
  this one override), seed 1. Rerunning the two commands at that commit gives the same bytes (check 8.1).

The numbers below were read from that folder with a few lines of pandas (the section "Reproducing the numbers
above", at the end, shows how). Times are **true time** in seconds since the first pod could be switched on, unless they say "host".

---

## Stage 1: The motion source

**Command:** part of `simulate`. **Runs:** `mesqsim.sim.motion:MotionSource.from_spec` →
`mesqsim.data.amass:load_sequence` → `mesqsim.data.amass:segment_orientations`.

**In:** an AMASS `.npz` file, the SMPL+H body model's joint rotations fitted to optical motion capture:

| key | shape here | meaning |
|---|---|---|
| `poses` | (3477, 156) | 52 joints × 3 axis-angle numbers (radians), per frame |
| `trans` | (3477, 3) | root position (metres; not used) |
| `betas` | (16,) | body shape (not used) |
| `mocap_framerate` | 120.0 | frames per second |

**What it computes.** Forward kinematics down the SMPL tree: the orientation of each of the 15 body segments that
carry a pod is the product of the joint rotations from the root to that segment. AMASS's world is already +Z up
(check 10.3 verifies it on the data), so no conversion is needed:

```text
S_i(t) = Exp(θ_root(t)) · Exp(θ_k1(t)) · … · Exp(θ_i(t))        (segment i → world, a unit quaternion)
```

(the simulator later turns the whole clip about the vertical to the session's facing, stage 2).

**Out (in memory):** `seg_q`, shape (3477, 15, 4), quaternions (x, y, z, w), and `fps = 120`. Duration
(3477 − 1)/120 = **28.97 s**.

**This session.** How much each joint moves in this clip (the scorer's *excitation*, computed from the truth,
`mesqsim.results_step2:excitation`): the standard deviation of the main joint angle is 45.0° (left knee), 36.2°
(right knee), 25.5° (elbows), 24.2° and 22.1° (wrists), 15.0° and 14.9° (ankles). For the ball joints it is the
share of frames within 5° of a range-of-motion limit: 9 % at the left hip, 3.5 % at the right hip, ≈ 0 at the lower
back, neck and shoulders. Keep that last line in mind: a ball joint that never nears a limit gives the geometry
nothing to work with (stage 7).

**What can go wrong.** A y-up world (SMPL's own) mistaken for z-up: every heading would be garbage (check 10.3,
`test_amass_world_is_z_up`). Left and right swapped (check 10.4, `test_amass_walking_thighs_alternate`). The chest
pod on the wrong spine joint: `motion.spine_joint` is an *assumption* (spine3, joint 9) that Danyal decides.

---

## Stage 2: The session plan

**Command:** part of `simulate`. **Runs:** `mesqsim.sim.session:plan_session(cfg, seed, motion)`.

**In:** `configs/default.yaml` (every parameter with a value, a label — datasheet, firmware, assumption, measured,
fitted — and a source) and the motion's duration.

**What it computes.** The timeline and every pod's private parameters, each drawn from its own named random stream
(so changing one subsystem never reshuffles another's draws):

```text
box [0, 5) → strapping [5, 35) → T-pose [35, 38) → idle [38, 600), not recorded → motion [600, 628.97]
```

**This session** (from `meta.json`, key `plan`):

| quantity | pod 9 (left thigh) | pod 10 (left shin) |
|---|---|---|
| boots at | 0.203 s | 0.265 s |
| clock rate error | −29.4 ppm | −33.6 ppm |
| chip sample rate | 51.40 Hz | 50.57 Hz |
| private north at boot, a_i (offset mode `arbitrary`) | +63.04° | −125.55° |
| slot error in the box, ε_i | +0.55° | −0.95° |
| gyro drift, b_i | +0.292 °/min | −0.374 °/min |
| scale-factor error, k_i | −1.296 % | −1.393 % |
| whole turns while strapped | −1 | +1 |
| reboots | none | none |

The host clock reads true time plus 886.2227 s. The box lies at heading −0.38°; the wearer faces 2.33° in the
T-pose and turns round twice during the idle (`idle_turns` = 2, the same for every pod: one body). None of this is
visible to an estimator: it lives in `meta.json`, which only the scorer may open.

**Out:** the plan (written to `meta.json` in stage 4) and `session.json`, the logger's sidecar: the phases in
**host** time (box 886.22–891.22 s, strapping 891.22–921.22 s, T-pose 921.22–924.22 s, idle 924.22–1486.22 s
marked `"recorded": false`, motion 1486.22–1515.19 s).

**What can go wrong.** A different seed must give a different session and the same seed identical bytes (check
8.1, `test_same_seed_same_bytes_different_seed_differs`). A test subject must never feed a fit (checks 9.1–9.3).

---

## Stage 3: Truth, timing and the radio

**Command:** part of `simulate`. **Runs:** `mesqsim.sim.simulate:simulate_session`, which calls
`mesqsim.sim.truth:pod_orientation` (where each pod truly points), `mesqsim.sim.heading:build_pod_heading` (how its
north slides), `mesqsim.sim.timing:simulate_timing` (when it samples and sends) and `mesqsim.sim.radio`
(`arrival_times`, `gilbert_elliott_lost`: when, and whether, each frame arrives).

**What it computes.** First the true pod orientation P_i(t): flat in the box, carried and strapped, held in the
T-pose, then the AMASS segment orientation times the pod's mounting m_i. Then the heading error of the pod's
private frame,

```text
ψ_i(t) = a_i + b_i·t + k_i·Θ_i(t) + w_i(t)
```

(offset at boot, drift, scale-factor error times the pod's accumulated turning about the vertical Θ_i, random
walk). Then what the pod reports, the **measurement model**:

```text
q_i(t) = Y(−ψ_i(t)) · P_i(t) · Exp(η_i(t))        (Y = rotation about the vertical; η = small tilt error of the fusion)
```

The target of the whole project is δ = wrap(ψ_child − ψ_parent).

**This session, pod by pod.**

| true time | ψ_9 (thigh) | ψ_10 (shin) | δ_knee = wrap(ψ_10 − ψ_9) | of which offset, wrap(a_10 − a_9) |
|---|---|---|---|---|
| in the box (mean) | +63.07° | −125.56° | 171.37° | 171.40° |
| 38 s, end of T-pose | | −130.13° | 161.03° | 171.40° |
| 600 s, motion starts | +62.51° | −143.95° | 153.54° | 171.40° |
| 628.97 s, motion ends | | −142.98° | 153.16° | 171.40° |

Read the last two columns. The offset part never changes (171.40°): box registration can remove it. The rest has
slid by −17.86° before the motion starts:

- −10.34° during strapping and T-pose: each pod made one whole turn while carried, in opposite directions, and
  each pod's gyro over- or under-counts turns by its scale-factor error: pod 9, −1.296 % × (−360°) = +4.67°; pod 10,
  −1.393 % × (+360°) = −5.01°; so δ moved by −9.68°, plus −0.39° of drift and a little twist and random walk;
- −7.49° during the idle: relative drift (−0.374 − 0.292) °/min × 9.37 min = −6.24°; the wearer's two turns, which
  every pod shares, count only through the *difference* of the scale factors, (−1.393 % + 1.296 %) × 720° = −0.70°;
  and −0.55° of twist and random walk.

That −17.86° is what B1 cannot see and what the geometry must find.

**Timing and radio.** Each chip samples at its own rate (≈ 50–60 Hz); the pod sends its latest sample every
31.25 ms of its own clock; frames are lost in bursts (`radio.loss_mean` 2 %, bursts of 3 frames). In this session:
31,505 send ticks the host could know about, **30,908 frames received**, 605 lost by the radio inside recorded
phases, no stalls. For pod 10: 2,102 ticks (119 in the box, 960 strapping, 96 T-pose, 927 motion), 2,059 received.

**What can go wrong.** The sign of ψ in the measurement model: a wrong sign leaves exactly 2|δ| of error; the
12°/5° worked example pins it (check 3.1, `test_worked_example_section_4_4`). Random walk, drift and scale factor
each have an exact check (6.1–6.3); the chip-rate resampling (7.1) and the loss model (7.5) too.

---

## Stage 4: Frames and labels

**Command:** the end of `simulate`. **Runs:** `mesqsim.io:write_session` (parquet with zstd compression, JSON with
sorted keys: the same seed gives the same bytes).

**Out:** `sessions/s000/` in the run folder.

| file | rows × columns | who may read it | columns (units) |
|---|---|---|---|
| `frames.parquet` (830 kB) | 30,908 × 9 | everyone: what a real logger writes | `t_host_s` (s, host clock, float64), `pod_id` (0–14), `batt` (%), `qx qy qz qw` (int16, value × 1/32767), `count` (sends since boot mod 65,536), `ms_lo` (pod-clock ms mod 65,536) |
| `session.json` (882 B) | — | everyone | the phases in host time, `recorded` true/false |
| `labels_ticks.parquet` (2.3 MB) | 31,505 × 20 | **scorer only** | per pod and tick: true send/arrival/sample/decode times (s), received, fresh, `psi_rad`, the true pod orientation |
| `labels_joints.parquet` (284 kB) | 29,232 × 6 | **scorer only** | per 32 Hz step (2,088 steps) and joint: `t_s` (true s), `phase`, `joint`, `delta_rad`, `delta_offset_rad` |
| `meta.json` (55 kB) | — | **scorer only** | the plan, every parameter with label and source, package versions, counts |

**This session.** Pod 10's first frame arrived at host time 887.5087 s with `batt` 78, quaternion
(215, 164, 28957, 15331), `count` 0 and `ms_lo` 1006. Divided by 32767 that is (0.00656, 0.00500, 0.88372,
0.46788): a pod lying flat (x and y components ≈ 0), turned 124.20° about the vertical. That is the box's heading
(−0.38°) plus the pod's slot error (−0.95°) minus its private north (a_10 = −125.55°): −0.38° − 0.95° + 125.55° ≈
124.23°, up to the fusion's tilt error.

**What can go wrong.** A frames table that does not match the contract (`test_frames_table_matches_the_contract`).
An estimator that opens a `labels_*` file or `meta.json`: GEO-ID gives identical bytes with every label deleted
(check 21.3, `test_label_blind_identical_without_labels`).

---

## Stage 5: Step 0, the post office

**Command:** the first thing `baselines` does. **Runs:** `mesqsim.step0:decode(frames, session)`.

**In:** `frames.parquet` and `session.json` — never a label.

**What it computes.** Per pod, in arrival order: `count` is unwrapped into `tick` (sends since boot); `ms_lo` into
`ms_unwrapped` (and `t_pod_s` = ms / 1000, the decode time on the pod's clock); a frame whose `ms_lo` repeats the
previous one is **held**, a new one **fresh**; ticks missing between two received frames of one recorded stretch are
**lost**; a count that jumps against the elapsed time starts a new boot **epoch** (a reboot). Quaternions are
divided by 32767, checked (|‖q‖ − 1| ≤ 0.01, else `valid = False`), normalised and given w ≥ 0. The one formula,
unwrapping a 16-bit counter with the host time as a guide:

```text
tick_k = tick_{k−1} + Δ,   Δ ≡ count_k − count_{k−1} (mod 65536),   Δ chosen nearest to (t_host,k − t_host,k−1)/31.25 ms
```

Then **box registration**, the baseline B1 (`mesqsim.step0:box_headings`): the circular mean heading h_i of each
pod's box frames, and δ̂_B1 = wrap(h_parent − h_child).

**Out:** `series.parquet` (2.05 MB): 31,505 rows (one per pod and knowable tick) × 18 columns: `pod_id`, `epoch`,
`tick`, `count`, `status` (fresh / held / lost), `received`, `valid`, `t_host_s`, `ms_unwrapped`, `t_pod_s`, `batt`,
`norm_err`, `qx qy qz qw` (floats) and two bookkeeping columns.

**This session.** 30,908 fresh, 0 held (no stalls in this session), 597 lost (the radio's 605 minus 8 ticks the host
cannot know about at the edges of recordings); one epoch per pod. Pod 10's `ms_lo` wrapped 9 times and was
unwrapped each time. Box registration: h_9 = −63.40° (118 box frames), h_10 = 124.31° (119 frames), so

```text
δ̂_B1(knee) = wrap(−63.40° − 124.31°) = wrap(−187.71°) = 172.29°
```

The truth in the box is 171.37°: B1 starts 0.92° off (mostly the slot errors, +0.55° − (−0.95°) = 1.50°, partly
cancelled by heading noise). At the motion it is 172.29° − 153.54° = **18.75° off**, because the norths kept sliding.

**What can go wrong.** A count wrap read as a reboot, or a reboot missed (`test_long_gap_with_count_wraps_is_not_a_reboot`,
`test_reboot_starts_a_new_epoch_and_its_early_ticks_are_lost`); fresh/held/lost flags that differ from the truth
(check 7.4, `test_step0_rebuilds_every_tick_and_flag`); the unrecorded idle counted as loss
(`test_unrecorded_idle_gap_is_not_counted_as_loss`).

---

## Stage 6: The common clock and the pairs

**Command:** `baselines --methods GEO-ID` (also GEO-K). **Runs:** inside `mesqsim.estimators.geo:GeoEstimator.run`:
`mesqsim.estimators.clock:pod_samples`, `epoch_starts`, `map_epoch` (with `fit_host_map`), `pair_samples`,
`body_rates`.

**In:** `series.parquet` (only `fresh` and `valid` rows) and `session.json`.

**Why a clock is needed.** The geometry compares thigh and shin *at the same instant*, but each pod stamps its
samples with its own drifting clock, and the host only knows when frames *arrived* (a few ms later, with jitter).
So each pod's clock is mapped onto the host's, causally, from the frames received so far:

```text
t_host ≈ α + β · t_pod          fitted to the lower envelope of t_host − t_pod (the fastest arrivals)
```

(the minimum in each 2 s block, a line through those minima, a slope only when there are ≥ 5 blocks, clipped to
±200 ppm). Then a **pair** at every fresh parent sample: the child's orientation slerped between its two fresh
samples around that instant, if they are at most 0.1 s apart. For axis identification, **body rates**
ω = Log(q(t₋)⁻¹·q(t))/Δt between consecutive fresh samples, in which the pod's private north cancels.

**This session.** Windows use frames from the motion on (stage 7), so ten seconds into the motion the maps rest on
10 s of data: β − 1 = +115 ppm for pod 9 and −173 ppm for pod 10, where the true clock errors call for about +29
and +34 ppm. Ten seconds cannot pin a slope, but they need not: over 10 s, 200 ppm moves a time by 2 ms. What matters
is the pairs' timing. Over the whole session GEO-ID formed **83,287 pairs** (all joints and windows); the true
time between the child's interpolated instant and the parent's sample was a median 2.90 ms, P95 7.89 ms, max
12.40 ms — inside the 15 ms bound of check 31.1 (`test_pair_timing_within_the_decode_delay_bound`). The knee's
8 s windows held 31 pairs one second into the motion, 95 after three, 244 after ten (at most 8 s × 32 Hz = 256).

**What can go wrong.** A slope fitted through two minima after a gap: one such fit once gave +1.5 % and 41 ms
errors, hence the 5-block rule (`test_short_data_gives_an_offset_only_map`). Rates that see the private north
(`test_rates_ignore_a_constant_heading_offset`). Frames used before they arrived (causality, check 21.1,
`test_causal_deleting_later_frames_changes_nothing_before`).

---

## Stage 7: The window solvers, per joint type

**Command:** as stage 6. **Runs:** `GeoEstimator._window` every `geometry.step_s` = 1 s over the last
`geometry.window_s` = 8 s of pairs; `GeoEstimator._solve` picks the solver of the joint's type.

**What it computes.** One estimate (δ̂_w, κ) per joint and window. κ is the von Mises concentration: 0 means "no
information", and 1/√κ is roughly the standard deviation in radians.

| joint type | joints | GEO-K (simulator truth: the true axes and mountings) | GEO-ID (deployable) |
|---|---|---|---|
| hinge | knees | `hinge_closed_form` with the true axes | axes identified from motion (`_identify`), then `hinge_closed_form` and the sign votes (`_hinge_id`) |
| 2-DoF | elbows, wrists, ankles | `twodof_solve` (the 2-DoF constraint) | range of motion in T-pose frames (`_rom` → `rom_solve`) |
| 3-DoF | hips, shoulders, lower back, neck | range of motion with the true mountings | range of motion in T-pose frames |

The hinge in one formula: with a_k and b_k the horizontal shadows of the knee axis seen from the thigh pod and from
the shin pod (a_k = horizontal part of q_p,k · u_p, b_k likewise), the least-squares relative heading is

```text
δ̂ = atan2(S, C),   C = Σ a_k·b_k,   S = Σ (b_k × a_k)_z,   κ = √(C² + S²) / σ̂²
```

(derived in [MATH_EXPLAINED §6](MATH_EXPLAINED.md#6-the-hinge-the-knees)). GEO-ID must first find the axis: in the
T-pose the knee is straight and its axis horizontal, so only one angle is unknown (`identify_hinge_axes_tpose`); a fit
is accepted only if it stands out from the other angles (contrast ≤ 0.8) and a wrong angle fits measurably badly
(spread ≥ 0.1). From the accepted knee axes it gets the body's **facing**, hence each pod's frame relative to its
segment from the T-pose (m̂_i) — and only then can the range-of-motion solvers of all the other joints run.

**This session (GEO-ID).**

- **Right knee:** axes accepted from the third window on; 26 of 28 windows informative; at the end the identified
  axes lay 9.05° (thigh) and 2.82° (shin) from the truth.
- **Facing:** found from the right knee in the window three seconds into the motion. The ankles, solved after the
  knees in that same window, use it at once; every other joint from the next window. Before that, every 2-DoF and
  3-DoF window says `no_frames`.
- **Left hip** (3-DoF, range of motion): at +5 s, δ̂_w = 101.92° against a true 109.90° (−8.0°), κ = 147 before
  calibration, 27.2 after (± 11.0°); at +10 s, 107.52° against 109.87° (−2.4°).
- **Left elbow** (2-DoF, range of motion): at +5 s, −44.85° against −50.61° (+5.8°) with κ = 4.35 after calibration
  (± 27°): almost no information, as the theory predicts inside the limits.
- **Left knee — a failure worth seeing.** Its axes were refused for the first 12 windows; then tied fits were
  accepted (the last one with contrast 0.73). They are poor: the thigh axis lies 24.5° from the truth (the shin's
  9.7°), and the window estimate is
  biased: at +28 s, δ̂_w = 169.15° against a true 153.23° (+15.9°), with a raw κ of 12,540 that calibration cuts to
  33.8 (± 9.9°). GEO-K, with the true axis, gets this knee to a median 6.9° over the motion.

**Calibration** (`mesqsim.estimators.geo:calibrated_kappa`). Neighbouring samples in a window are not independent,
so the raw κ overstates the confidence. Each window's κ is replaced by s·κ/(1 + κ·σ_m²), with the scale s and the
model error σ_m fitted per joint type on the **validation** split (hinge: s = 0.01178, σ_m = 0.94°; 3-DoF: s = 3.73,
σ_m = 20.7°; 2-DoF: s = 0.557, σ_m = 3° assumed). The hinge cap s/σ_m² = 43.9 means no knee window ever claims better
than about ± 8.6° (E26 in [MATH_EXPLAINED §10](MATH_EXPLAINED.md#10-calibration)).

**What can go wrong.** The hinge sign (a flipped axis gives δ + 180°: `test_flipped_axis_is_corrected_by_the_tpose_rule`,
check 24.1); the vertical-axis blind spot (check 22.1, `test_vertical_axis_is_the_blind_spot`); two equal 2-DoF
minima (check 25.1, `test_two_equal_minima_are_flagged_ambiguous`); a range of motion that claims information inside
the limits (check 26.1, `test_rom_has_almost_no_information_inside_the_limits`); an answer that depends on the room's
heading (check 28.1). Rigid thigh–shin motion accepted as a knee axis
(`test_tpose_tied_identification_recovers_the_axis_and_refuses_rigid_motion`).

---

## Stage 8: The filter

**Command:** as stage 6. **Runs:** `mesqsim.estimators.vmfilter:JointFilter` (`init`, `reset`, `update`, `query`),
driven by `GeoEstimator.run` and `replay`.

**What it computes.** A belief VM(μ, κ) per joint that is updated by events:

```text
init (end of the box):   μ = δ̂_B1,  κ0 = 1/(2·σ_box²)                     σ_box = 2° (assumption)
predict (between events): A(κ′) = A(κ) · exp(−q·Δt/2)                       q = 5.623e−5 rad²/s (fitted)
update (each window):     κ′e^{iμ′} = κe^{iμ} + (κ_w/8)·e^{iδ̂_w}            the product of two von Mises densities
reset (a reboot):         κ = 0 for the joints of the rebooted pod
```

The factor 1/8 = step/window: consecutive 8 s windows overlap by 7 s, so each counts for one eighth. A step is
reported `uncertain` when κ < κ_hold = 158.5, fitted on validation: no threshold left its retained steps with a P95
error ≤ 10° (the best was 10.16°), so κ_hold is the threshold with the smallest retained P95 that still keeps at least
10 % of the steps. The estimate is still reported.

**This session, the left knee.**

| when | μ | κ | ± 95 % | truth | error | uncertain |
|---|---|---|---|---|---|---|
| init, 5 s (box end) | 172.29° | 410.4 | ± 5.5° | 171.34° | 0.95° | no |
| motion start, 600 s | 172.29° | 28.4 | ± 21.3° | 153.54° | 18.75° | yes |
| +10 s | 172.29° | 27.9 | ± 21.5° | 153.17° | 19.12° | yes |
| +20 s | 170.74° | 62.7 | ± 14.2° | 153.54° | 17.20° | yes |
| +28.9 s (end) | 170.80° | 95.0 | ± 11.6° | 153.16° | 17.64° | yes |

The idle shrank κ from 410.4 to 28.4 without moving μ: drift has no known direction, so the belief only widens
(the large-κ rule gives 1/(1/410.4 + q·595 s) = 27.9; the exact rule 28.4). The 95 % interval (± 21.3°) now just
contains the error (18.75°). Once the knee's windows arrive, μ moves toward them — toward 169°, the biased window
value, not the truth (153°) — but slowly, because each window counts for 33.8/8 = 4.2 against the belief's 28.

Over the motion, the left knee's median error is B1 19.01°, **GEO-ID 18.14°**, GEO-K 6.91°. 92 % of this session's
motion steps are `uncertain` (hinge 100 %, 2-DoF 82 %, 3-DoF 100 %).

**What can go wrong.** A wrong prediction or product rule (check 27.1,
`test_large_kappa_prediction_matches_the_variance_sum`, `test_update_is_the_product_of_the_two_densities`); a reboot
that does not reset (`test_a_reboot_resets_the_joints_of_that_pod`); GEO-ID drifting away from B1 without any
information (`test_geo_id_equals_b1_on_noise_free_motion_without_information`); over- or under-confidence (check
30.1, coverage on validation).

---

## Stage 9: The scorer, the strata and the statistics

**Command:** the end of `baselines`. **Runs:** `mesqsim.evaluate:evaluate` → `Scorer.errors`; then
`mesqsim.pipeline:summarise`.

**In:** every method's δ̂ at the 32 Hz steps, and — now, and only here — the labels.

**What it computes.** The **primary metric**, per step and joint, |wrap(δ̂ − δ)|, and the **secondary metric**, the
geodesic angle between the joint rotation implied by the frames with the *true* mountings and the true joint
rotation (it adds fusion tilt error and quantisation, so even O2 is not 0).

**Out:** `results/errors.parquet` (77,868 rows = 6 methods × 927 motion steps × 14 joints; columns `session`,
`baseline`, `step`, `t_s`, `phase`, `joint`, `err_heading_deg`, `err_orient_deg`, `kappa`, `uncertain`),
`results/summary.csv` and `summary.md`.

**This session**, all 14 joints over the motion (degrees):

| method | mean | median | P95 | what it is |
|---|---|---|---|---|
| B0 | 115.54 | 114.21 | 178.91 | no correction |
| B1 | 7.02 | 4.23 | 18.86 | box registration |
| O1 | 7.10 | 5.76 | 17.97 | true offsets at boot (oracle) |
| O2 | 0.00 | 0.00 | 0.00 | the truth (oracle) |
| GEO-K | 5.93 | 4.71 | 11.71 | geometry with the true axes and mountings (uses simulator truth) |
| **GEO-ID** | **7.58** | **7.00** | **17.65** | deployable geometry |

By joint type (mean): hinge B1 10.67°, GEO-ID 9.89°, GEO-K 5.36°; 2-DoF 8.16°, 8.73°, 7.56°; 3-DoF 4.66°, 5.65°,
4.50°. Two lessons in one session. O1 is no better than B1: the error is drift and turning, not offset, and O1 only
knows the offsets. And GEO-ID helps at most joints where B1 is bad and hurts where B1 happened to be good. Median
errors, B1 → GEO-ID: where B1 is above 5°, the left knee 19.0° → 18.1°, the lower back 5.5° → 3.8°, the right
shoulder 12.2° → 10.4°, the right wrist 11.4° → 10.0° and the right ankle 12.6° → 8.3° improve (the left wrist,
11.4° → 15.1°, and the left ankle, 10.4° → 11.1°, do not); where B1 is below 4°, the neck, the left shoulder, both
elbows and both hips get worse (the right hip 3.9° → 8.1°). The windows have errors of their own, a few degrees (the
left hip's −8° to −2°), and they replace a box estimate that was, by luck, better. Over many sessions the filter
weighs the two correctly on average (the coverage of Table 6 in the results); within one session it cannot know which
joints are the lucky ones.

**Strata.** For the Step 2 results each session belongs to a motion start (1, 10 or 30 min since boot) and, per
joint, an excitation tercile cut within its joint type (this clip's knees, at 45° and 36° of flexion spread, fall in
the top tercile). **The unit of analysis is the subject**: steps and sessions of one person are not independent, so
each contrast first takes one median per subject, then compares GEO-ID with B1 by a paired bootstrap over subjects
(10,000 resamples, fixed seed) and corrects the three contrasts with Holm's method
([MATH_EXPLAINED §11](MATH_EXPLAINED.md#11-metrics-and-statistics)).

**What can go wrong.** Scoring outside the motion phase (results report the motion only); a method reading truth
without being marked (`USES_TRUTH`, every table stars O1, O2 and GEO-K); steps treated as independent (the bootstrap
resamples subjects).

---

## Stage 10: The results

**Command** (in a clean checkout, so that the commit in the file is exact):

```bash
python scripts/make_results_step2.py --source amass --seed 1        # 519 test sessions, sweeps, DIODEM
```

**Runs:** `mesqsim.geo_sessions:dataset_jobs` (one job per test sequence window × motion start, the same seeds a
dataset run would use: `mesqsim.dataset:session_seed`), `mesqsim.geo_sessions:run_jobs` (worker processes),
`mesqsim.results_step2:session_worker` (simulate, decode, run every method, keep compact error arrays) and the
tables, statistics and figures in `scripts/make_results_step2.py`.

**In:** the split manifest `manifests/split_manifest.csv` (test split only; hash-locked by the tool, Danyal's
freeze still pending), the AMASS files, the config.
Nothing is written per session: sessions live in memory.

**Out:** `docs/RESULTS_STEP2.md` and `docs/figures/results_step2_{growth,reliability,selective,sweeps,reboot}.png`.

**Where our session fits.** `CMU/13/13_02` is one of the 117 test sequences long enough for a 20 s window. The
results use its first 20 s three times (motion at 1, 10 and 30 min), each with its own seed, so they are cousins of
the session above, not copies.

**The run** at commit `d6c0487` (519 test sessions, 18.5 min of compute on an Apple M2 Pro), all 14 joints, mean
error (degrees):

| method | 1 min | 10 min | 30 min |
|---|---|---|---|
| B1 | 3.90 | 7.08 | 11.91 |
| O1 * | 3.17 | 6.67 | 11.65 |
| GEO-K * | 3.75 | 6.30 | 9.08 |
| GEO-ID | 3.95 | 6.69 | 9.96 |

(`*` = uses simulator truth.) Read it by columns. At 1 min the box registration is fresh and nothing beats it: GEO-ID
is 0.05° worse than B1. The box ages and the geometry does not, so by 30 min GEO-ID is 1.95° better than B1, and
GEO-K, with perfect axes, 2.83° better. By joint type at 30 min: hinge 12.15° → 9.25° (GEO-K 6.37°), 2-DoF
11.76° → 9.75°, 3-DoF 11.97° → 10.41°. GEO-ID found the facing in 358 of the 519 sessions (Table 3b); in the other
161 it is B1 at every joint.

**H2** (geometry helps at hinge and 2-DoF joints, not at 3-DoF joints), at 10 min, GEO-ID − B1 per subject:

| contrast | joint type | GEO-ID − B1 (mean) | 95 % interval | p (Holm) | reading under the draft plan |
|---|---|---|---|---|---|
| C1 | hinge (upper two excitation terciles) | −0.98° | [−2.02, −0.07] | 0.09 | not shown better: the interval excludes 0, but the gain is just under the 1° margin |
| C2 | 2-DoF (upper two terciles) | −0.78° | [−1.79, 0.04] | 0.13 | not shown better |
| C3 | 3-DoF (all motion) | −0.08° | [−0.66, 0.44] | 0.81 | supports H2: an improvement above the margin is excluded |

So the data lean the way H2 says — geometry helps most at the knees, a little at the 2-DoF joints, not at the ball
joints — but by the plan's own definition of "better" (interval excluding 0 *and* a gain above 1°) no contrast is
established.

Every comparison is **descriptive**: the analysis plan and the split are not frozen yet, so no confirmatory claim is
made (the banner at the top of the file says so). Table 3b of the results says how often GEO-ID could identify the
knee axes and the facing at all; without them GEO-ID *is* B1, by construction.

**What can go wrong.** Results from a dirty tree (the commit in the file would lie: run in a clean worktree); a
run that is not reproducible (check 33.1: same config and seed, identical file apart from the Run line); a fitted
value fitted on test (every `fitted` source names its command and the validation split).

---

## The DIODEM path: real sensors, optical truth

DIODEM (Weygers et al.) records chains of rigid segments — a robot-like arm and a gait chain — with IMUs and optical
motion capture. It has no people, no box and no T-pose, so it tests one thing: can GEO-ID track the relative heading
of a real hinge from real six-axis fusion?

**Command:** part of `make_results_step2.py` (`--no-diodem` skips it). **Runs:** `mesqsim.diodem_geo:all_hinges` →
`run_hinge` for every hinge of every experiment.

**What it computes.** Each IMU's six-axis VQF orientation at 40 Hz (`vqf_six_axis`, magnetometer dropped, so the
heading drifts as in the suit). The truth from the optical segment orientation: q_vqf = Y(−ψ)·q_opt, so
ψ = −heading(q_vqf·q_opt⁻¹) (`psi_from_vqf_and_optical`) and δ = wrap(ψ_child − ψ_parent), exactly as in the
simulator. The two IMUs become pods 9 and 10 of a Step 0-shaped series with one `motion` phase (`diodem_series`), and
GEO-ID runs unchanged: with no box it starts at κ = 0, and with no T-pose it uses the free axis fit and the vertical
sign rule. The reference is the **start oracle**: δ held at its true value at the first sample, whose error is the
six-axis drift the geometry has to track.

**Numbers** (Table 10 of the results, commit `d6c0487`, beliefs fitted on AMASS validation): on the arm chain's
first hinge in experiment 1 (seg2 → seg3, 6.1 min) the start registration ends 80.7° off — the six-axis drift — while
GEO-ID's median error is 15.18° (P95 29.63°), with its first geometric information after 3 s. Over all 27 hinge runs
GEO-ID's median error is below the start registration's in 21; in the other six the start registration itself
stayed good (median error under 4°). With the beliefs assumed before the AMASS fit, exp01's median is 7.79°: the AMASS knee calibration
(no window better than ± 8.6°) is too pessimistic for a near-perfect mechanical hinge, so the filter follows fast
drift more slowly.

**What can go wrong.** The sign of ψ for real data (check 32.1 builds a synthetic rotation with a known ψ); data
files that differ from the manifest (check 11.1). And interpretation: these are mechanical chains, out of
distribution for people — the experiment is the unit, and no claim about the suit rests on them.

---

## Reproducing the numbers above

```python
import pandas as pd, numpy as np
from mesqsim import io
run = "<the run folder>"
d = io.read_session(f"{run}/sessions/s000")          # frames, session, labels, meta
plan = d["meta"]["plan"]
print(plan["phases"], np.rad2deg(plan["pods"][10]["offset_rad"]))          # stage 2
lj = d["labels_joints"]; knee = lj[lj.joint == 10]
print(np.rad2deg(knee.delta_rad.iloc[[0, -1]]))                              # stage 3: the target
s = pd.read_parquet(f"{run}/sessions/s000/series.parquet")
print(s[s.pod_id == 10].status.value_counts())                               # stage 5
e = pd.read_parquet(f"{run}/results/errors.parquet")
print(e[e.joint == 10].groupby("baseline").err_heading_deg.median())         # stage 9
```

For stages 6–8, `GeoEstimator(cfg).run(series, session, host_times, diagnostics=True)` returns the per-window table
(`diag.windows`: time, joint, number of pairs, δ̂_w, raw κ, status) and the identified axes (`diag.state`).

---

## Self-check questions

1. **Which files may an estimator read, and which only the scorer?** Estimators: `frames.parquet` (through Step 0's
   series) and `session.json`. Scorer only: `labels_ticks.parquet`, `labels_joints.parquet`, `meta.json`. Check 21.3
   proves GEO-ID gives the same bytes with the labels deleted.
2. **The knee's δ was 171.37° in the box and 153.54° at the motion. Which part can B1 remove?** Only the offset
   (171.40°, fixed at boot). The −17.86° that slid in during strapping and the idle is invisible to the box.
3. **Why was O1 no better than B1 in this session?** O1 knows the true offsets at boot and nothing about drift; B1
   removes the same offsets up to the slot errors. Here the error is almost all drift and turning, so the two carry
   almost the same information (mean 7.10° against 7.02°).
4. **Why does a 0.1 s maximum gap matter for pairs?** The child is interpolated between two samples; across a long
   gap (lost frames) the interpolation would invent motion, so no pair is formed.
5. **β − 1 was −173 ppm against a true +34 ppm. Is that a bug?** No: 10 s of data cannot fix a slope, and 200 ppm
   over 10 s is 2 ms. The pairs' timing error (median 2.9 ms, max 12.4 ms) is what matters, and it stays below 15 ms.
6. **Why did the left elbow's window carry almost no information?** It is solved by the range of motion, which is
   flat while the joint stays inside its limits: κ = 4.4, about ± 27°.
7. **Why are the 2-DoF and 3-DoF windows `no_frames` at first?** GEO-ID needs the facing, which comes from the knee
   axes; until a knee is identified, no T-pose frames exist.
8. **What went wrong at the left knee, and what did the filter do about it?** A poorly identified axis (24.5° off)
   biased every window by about +16° (169° against a true 153°). Calibration capped each window at ± 8.6° and the
   filter counts each for 1/8, so after 16 windows μ had moved only from 172.3° to 170.8°: the error fell from 18.75°
   to 17.64°. The same caution that limits the damage of a bad axis also limits the benefit of a good one.
9. **What does `uncertain` mean for a step with κ = 95.0?** κ < κ_hold = 158.5. On validation even the most confident
   steps kept a P95 error just above 10° (10.16° at κ_hold), and steps below κ_hold did worse. The estimate is still
   reported; a user may hold or flag it.
10. **Why is the subject, not the step, the unit of the statistics?** Steps of one session are strongly correlated,
    and sessions of one person share a body and a style; counting them as independent would shrink the intervals
    without new evidence.

---

*Next: [MATH_EXPLAINED.md](MATH_EXPLAINED.md) derives every formula used above, step by step.*
