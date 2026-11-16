# The mathematics

*Status: M4. Complete for the simulator, Step 0 and the Step 1 baselines.*

## 1. Picture first

Every pod reports an orientation that is right about tilt and wrong about heading. The heading error is one
rotation about the vertical, Y(−ψ_i), placed on the **left** of the true orientation, and each pod has its own.
For a joint, only the difference of the two pods' ψ matters: δ = ψ_child − ψ_parent. Rotate the child's report
by Y(δ) and it shares its parent's north. Do this outward from the pelvis and every pod shares the pelvis pod's
north, which itself stays unknown.

## 2. Conventions (owned by [mesqsim/conventions.py](../mesqsim/conventions.py))

| quantity | convention | pinned by |
|---|---|---|
| quaternion storage | `(x, y, z, w)`, scalar last, float64, as in `scipy.spatial.transform.Rotation` | `test_quaternion_order_is_xyzw_like_scipy` |
| product | Hamilton: i·j = k. `quat_mul(a, b)` applies `b` first, then `a` | `test_hamilton_product`, `test_composition_order_is_right_to_left` |
| sign | `q` and `−q` are the same rotation; canonical form has `w ≥ 0` (the wire's form) | `test_q_and_minus_q_are_the_same_rotation` |
| orientation | maps body to world: `v_world = R · v_body` | `test_orientation_maps_body_to_world` |
| world frame | gravity-aligned, **+Z up**, heading arbitrary but fixed | dataset loaders (M2) |
| Y(α) | rotation by α about world +Z, right-handed; Y(90°) maps +x to +y | `test_y90_maps_x_to_y` |
| heading | first angle of the intrinsic Z-Y-X decomposition, `atan2(R[1,0], R[0,0])`; so heading(Y(α)·R) = wrap(heading(R) + α) | `test_heading_shifts_by_alpha` |
| wrap | into (−180°, 180°]: wrap(190°) = −170°, wrap(−180°) = 180° | `test_wrap_values` |
| Exp / Log | rotation vector (axis × angle, radians) ↔ quaternion | `test_exp_log_roundtrip_and_geodesic_angle` |
| geodesic angle | 2·atan2(‖v‖, \|w\|), in [0°, 180°] | same |
| units | radians in code, degrees in reports; the unit is in the name (`_rad`, `_deg`, `_s`, `_ms`) | review |
| statistics of angles | circular mean `atan2(mean sin, mean cos)`, circular std `sqrt(−2 ln R)` | `test_circular_statistics` |

Heading is ill-conditioned when the body x axis is vertical (pitch ±90°). The relative-heading target δ does not
use the heading decomposition at all; it is defined from the ψ_i of the measurement model.

## 3. Target and correction (owned by [mesqsim/correction.py](../mesqsim/correction.py))

```text
target for joint (parent p, child c):   δ_pc(t) = wrap(ψ_c(t) − ψ_p(t))
correction, outward from the pelvis:    ĉ_root = 0,   ĉ_c = ĉ_p + δ̂_pc,   q̃_c = Y(ĉ_c) · q_c
```

Why it works: with q_c = Y(−ψ_c)·s_c·m_c and exact estimates, ĉ_c = ψ_c − ψ_root, so
q̃_c = Y(ψ_c − ψ_root)·Y(−ψ_c)·s_c·m_c = Y(−ψ_root)·s_c·m_c. Every pod ends up with the same unknown heading.

### Worked example (a unit test and validation check 3.1)

Parent at the root with ψ_p = 12°, child with ψ_c = 5°.

1. δ_pc = wrap(5° − 12°) = −7°.
2. ĉ_c = 0 + (−7°), so q̃_c = Y(−7°)·Y(−5°)·s_c·m_c = Y(−12°)·s_c·m_c: the child shares the parent's north.
3. With the wrong sign, Y(+7°)·Y(−5°) = Y(+2°), which differs from Y(−12°) by **14°**.

## 4. What cannot be known, as tests

| statement | tests |
|---|---|
| Rotating every report by one Y(β) changes no target δ (exactly) and moves estimators only by quantisation | `test_common_yaw_leaves_targets_unchanged_exactly`; check 4.2 re-simulates a whole session with β = 73° added to every north: labels move < 1e-9°, B1 ≤ 0.5° |
| Rotating one pod's frame by Y(γ) changes only the targets of its joints, by exactly ±γ | `test_single_pod_offset_moves_only_its_joints_by_gamma`; check 4.3 re-simulates with γ = 23° on one pod |
| A correction acts about the vertical only, so tilt never changes | `test_correction_never_changes_tilt`, check 4.1 |

## 5. The measurement model and the session

```text
s_i(t)   true orientation of segment i (segment -> world); segment frame = SMPL canonical body frame
m_i      mounting of pod i on its segment (pod -> segment), constant
P_i(t)   = s_i(t)·m_i, the true pod orientation (pod -> world)
η_i(t)   small tilt-and-noise error of the chip's fusion, a rotation vector in pod axes

what pod i reports:   q_i(t) = Y(−ψ_i(t)) · P_i(t) · Exp(η_i(t)),   canonicalised to w ≥ 0, then quantised

heading of pod i's private frame, in boot epoch e (owned by mesqsim/sim/heading.py):
   ψ_i(t) = a_e + b_i·(t − boot_e) + k_i·(Θ_i(t) − Θ_i(boot_e)) + w_i(t)
```

### 5.1 Every term, its value and its label

All values are in [configs/default.yaml](../configs/default.yaml). `python -m mesqsim params` prints them with
their sources.

| term | model | default | swept | label |
|---|---|---|---|---|
| offset a_e | `arbitrary`: uniform in (−180°, 180°]; `aligned`: the pod's own heading when its chip starts (in the box: box heading + ε_i), so its report starts at heading 0 | arbitrary | aligned, arbitrary | assumption |
| box slot ε_i | N(0, σ_box²) about the vertical | σ_box = 2° | 0.5°, 2°, 5° | assumption |
| drift b_i | N(0, σ_b²), constant per pod, counted from each boot | σ_b = 0.2 °/min | 0.05, 0.2, 1.0 °/min | assumption |
| scale factor k_i | uniform in ±k_max | 1.5 % | 1.5 %, 3 % | datasheet |
| angle random walk w_i | Brownian, variance N_g²·t | N_g = 0.015 °/s/√Hz (0.367° after 10 min) | — | datasheet |
| fusion error η_i | Gauss–Markov per pod axis, stationary σ, correlation time τ | σ = 0.5°, τ = 1 s | — | assumption |
| reboots | Poisson per pod; silent for 3 s; then count, clock and Θ restart and a new north a_e is chosen | 0 per hour | 0, 1 per hour | assumption |
| idle turning | whole turns ~ round(N(0, (0.5·√minutes)²)) of the wearer | 0.5 turns/√min | — | assumption |
| quantisation | trunc(float32(v)·32767), step about 0.0035° | — | — | firmware |

### 5.2 Θ: the turning that the scale factor multiplies

Θ_i(t) is the signed rotation of pod i about the world vertical since its first boot. It is the integral of the
vertical component of the pod's world angular velocity.

- The simulator integrates it along the true trajectory P_i on a 240 Hz grid, summing z(Log(P_{k+1}·P_k⁻¹)).
- Tilting about a horizontal axis adds nothing, and turning back cancels.
- One full turn is 2π exactly (`test_theta_counts_one_full_turn_despite_tilt_wobble`). With k = 1.5 % it gives
  0.015 × 360° = 5.4° (check 6.3, through the whole simulator on a walk in a circle).
- Θ grows during strapping (each pod's own whole turns while carried), during motion, and over the idle interval.
- The idle interval is never recorded, so its turning is one step: the net twist about the vertical between the
  pose before and after it, plus whole turns drawn from the idle-turning model. The random walk over the idle is
  one increment N(0, N_g²·Δt). Inside the idle, Θ and w are interpolated linearly; nothing there is recorded.

### 5.3 The session in true time (seconds from the first power-on)

1. **Box.** Pods boot within 0.5 s of each other, lying flat with aligned axes: P_i = Y(box heading + ε_i).
2. **Strapping** (30 s). Each pod moves to its strap during its own 2–6 s window:
   P_i = Y(2π·n_i·g)·slerp(P_box, P_worn, g), with g a smoothstep of time and n_i ∈ {−1, 0, 1} whole turns.
3. **T-pose** (3 s). s_i = Y(φ_T)·Rx(90°)·Exp(e_i): the SMPL rest pose, upright, facing φ_T, with posture error e_i
   (RMS angle 5°).
4. **Idle** (0 s by default; swept 0, 10, 30 min). Not recorded. The wearer holds the T-pose for the first half
   and the motion's first pose for the second.
5. **Motion.** s_i(t) = Y(φ_m)·S_i(t − t_motion), the source clip turned so that it starts facing φ_T ± 15°.

φ_T and the box heading are uniform: the room's heading is unknowable, so the simulator draws it at random.

### 5.4 Timing, held frames and the radio

- **Chip.** Samples at f_i ~ U(50.1, 60.0) Hz, starting 0.2–0.5 s after boot. Each sample is decoded 0–11 ms
  later; `ms_lo` = floor(pod ms at decode) mod 65,536. The pod clock runs at 1 + ρ_i, with ρ_i within ±50 ppm.
- **Send grid.** Sends follow the pod clock at ms = 1000 + id·31.25/17 + j·31.25 and carry the newest decoded
  sample; `count` = j mod 65,536.
- **Discarded samples.** The pod sends one of about f_i/32 samples, so a share 1 − 32/f_i is discarded: 0.361,
  0.418 and 0.467 at 50.1, 55 and 60 Hz (check 7.1).
- **Held frames.** With f_i > 32 Hz and decode jitter ≤ 11 ms, a send always finds something new, so held frames
  need **stalls** (off by default).
  - A stall of S ms comes every 3 ± 0.5 s. Nothing is decoded during it. At its end the newest waiting sample is
    decoded, and the decode task needs 1 ms before the next, so distinct samples never share a `ms_lo`.
  - Expected held frames per stall: (S + E[u])/T − 1 for S ≥ T = 31.25 ms, where E[u] = 1/(2f) + f·D²/12 is the
    mean time since the last decode (D = 11 ms). Check 7.2 compares the simulated rate with this.
- **Radio.** Losses follow a two-state Gilbert–Elliott chain per pod, with P(bad→good) = 1/B and
  P(good→bad) = L/(1−L)·1/B, giving mean loss L and mean burst B (check 7.5; defaults L = 2 %, B = 3).
- **Arrival.** Arrival = send + 2 ms + Exp(mean 3 ms), first-in first-out per pod; host time = arrival + an
  arbitrary epoch.

## 6. Step 0 (owned by [mesqsim/step0.py](../mesqsim/step0.py); never reads labels)

Per pod, in arrival order:

1. **count → tick.** The host-time gap between two frames predicts the number of sends, E = Δt_host / 31.25 ms.
   The observed count difference is unwrapped by the number of 65,536-wraps closest to E, giving Δtick.
   - If |Δtick − E| > 4 + 200 ppm · E, the pod **rebooted**: a new `epoch` starts with tick = count. Epochs count
     the reboots the recording can see: a pod that reboots twice while nothing is recorded shows one jump. The labels
     number epochs the same way and keep the true boot index in `boot_epoch`. This was found when a results sweep
     with reboots crashed; tested by `test_two_reboots_in_the_unrecorded_idle_look_like_one`, and by 30 stress
     sessions with 834 reboots and 0 mismatches.
   - Host time only resolves wraps and reboots. It is never a sample time.
2. **ms_lo → ms_unwrapped.** Within an epoch, ms_lo is unwrapped by the number of wraps closest to Δtick·31.25 ms.
   The first frame of an epoch is placed by its tick (1000 ms + tick·31.25 ms; any error under 32 s is harmless).
3. **held / fresh.** Equal unwrapped ms_lo to the previous received frame means held; otherwise fresh.
4. **lost** (shared rule, [mesqsim/recording.py](../mesqsim/recording.py)):
   - Recorded phases that touch merge into recorded intervals.
   - A missing tick is lost when the received frames before and after it lie in the same recorded interval, and the
     later one is in the tick's own boot epoch. A new epoch's ticks before its first received frame are therefore
     lost.
   - Ticks at a recording's edges, or sent just before a reboot, cannot be known by any logger. They appear neither
     in Step 0's output nor in the labels.
   - Why a rule and not a guess: deciding by estimated arrival time puts ticks within a few milliseconds of a
     recording boundary on the wrong side (the first M3 test run marked 5 ticks at a 600 s idle gap as lost, through
     pod-clock drift over the gap).
5. **Integrity.** q = int16 / 32767. Frames with |‖q‖ − 1| > 0.01 are kept but marked invalid. Quantisation alone
   moves the norm by less than 1.3·10⁻⁴. Valid quaternions are normalised and canonicalised.
6. **Box registration.** h_i is the circular mean of heading(q_i) over pod i's valid box-phase frames of its first
   boot epoch. In the box, q_i = Y(−a_i)·Y(box heading + ε_i), so h_i = box heading + ε_i − a_i, and
   wrap(h_p − h_c) = wrap(a_c − a_p) + (ε_p − ε_c): the offset difference up to the slot errors. Only differences
   between pods mean anything. In the aligned mode every h_i is 0, because each chip took its own box heading as
   north.
7. **T-pose snapshot**, as the suit does it: q_i(T) is the mean of pod i's T-pose frames, and
   Δ_i(t) = q_i(t)·q_i(T)⁻¹. With no errors Δ_i(t) is the pod's true world rotation since the T-pose (tested to
   0.02°). It removes the mountings for an avatar; the secondary metric uses the true mountings instead, so B1's
   numbers come from the box alone.

## 7. Metrics and baselines (owned by [mesqsim/evaluate.py](../mesqsim/evaluate.py))

- **Grid.** True times t_k on multiples of 1/32 s, from 0.25 s after every pod has sent to the end of motion, in
  recorded phases only. Results are reported over the motion phase.
- **Primary metric:** |wrap(δ̂_pc(t_k) − δ_pc(t_k))| per joint and step, in degrees.
- **Secondary metric (uses the simulator's true mountings).** At each step, take every pod's latest valid received
  frame by host time. Ĵ = (q_p m_p⁻¹)⁻¹·Y(δ̂_pc)·(q_c m_c⁻¹) is the joint rotation the corrected frames imply (only
  δ̂_pc matters, because the outward correction cancels everywhere else). J = (P_p m_p⁻¹)⁻¹·(P_c m_c⁻¹) is the truth
  at the **same sample instants**. The error is the geodesic angle of J⁻¹Ĵ. Comparing at the sample instants
  removes timing error, so the metric measures heading, fusion and quantisation error.
- **Why B0's secondary error equals the offset difference.** With q_i = Y(−a_i)·s_i·m_i,
  J⁻¹Ĵ = s_c⁻¹·Y(a_p − a_c)·s_c, whose angle is |wrap(a_c − a_p)|. Check 5.2 confirms it through the decoded
  frames to within 0.0081°, which is quantisation.

| baseline | δ̂ | what its error measures |
|---|---|---|
| B0 | 0 | the whole problem: offsets, drift, turning, noise |
| B1 | wrap(h_p − h_c) from box registration, constant | what a pipeline without learning leaves: slot errors, everything accrued since the box, and stale estimates after a reboot |
| O1 | wrap(a_c − a_p), the true offsets of the current boot epochs (oracle, simulation only) | the error that is drift rather than offset |
| O2 | the true δ (oracle) | nothing: its primary error must be exactly 0 (pipeline check) |

Expected order (check 12.1): O2 = 0 ≤ O1 ≤ B1, and B0 ≫ B1 when the offsets are arbitrary. B1 cannot beat O1: its
estimate is the true offset difference plus slot noise. In the aligned mode B0 ≈ B1, because the offsets already
agree up to the slot errors.

## 8. Why each check exists

| check | what would be wrong if it failed |
|---|---|
| 1.x codec | frames decoded with wrong values, signs or wraps |
| 2.x conventions | a sign, order or frame error somewhere downstream |
| 3.1 worked example | the correction applied with the wrong sign (twice the error) |
| 4.1–4.3 invariances | the pipeline depends on the unknowable room heading, or one pod's north leaks into other joints |
| 5.1–5.3 zero corruption | error appears where none was put in: a bookkeeping, timing or metric bug |
| 6.1–6.3 analytic growth | the random walk, drift or scale-factor term has the wrong scale, units or sign |
| 7.1–7.5 timing | discards, held frames, unwrapping, flags or losses do not follow the firmware and radio models |
| 8.1 reproducibility | results cannot be traced to (commit, config, seed) |
| 10.x AMASS mapping | the wrong joint, axis, side or up direction feeds every simulated session |
| 11.x DIODEM | the reality-check data are not what the loader assumes |
| 12.1 baseline order | an oracle or a baseline is computed against the wrong labels |
