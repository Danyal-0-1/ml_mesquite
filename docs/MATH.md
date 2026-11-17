# The mathematics

*Status: S2-M3. Complete for the simulator, Step 0 and the Step 1 baselines; Step 2 sections §9–§15 (clock, von Mises, events, hinge, 2-DoF, range of motion, facing).*

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

## 9. Step 2: one clock for parent and child (owned by [mesqsim/estimators/clock.py](../mesqsim/estimators/clock.py))

The geometry compares a parent and a child pod **at the same instant**, but each pod stamps samples with its own
clock: `t_pod_s` is Step 0's unwrapped `ms_lo`, the **decode** time on the pod clock (0–11 ms after the sample).
The host stamps **arrival**.

- **Pod time → host time,** per pod and boot epoch: t_host ≈ α + β·t_pod. Arrival = decode + a positive delay
  (waiting for the next send, radio, hub), so the map follows the **lower envelope** of r = t_host − t_pod: the
  minimum of r in each block of `geometry.clock_block_s` (2 s) of pod time, then a least-squares line through the
  block minima. The slope is fitted only through at least 5 block minima (10 s) and is clipped to ±200 ppm (Step
  0's clock tolerance); otherwise β = 1. *Why:* three seconds after a 10-minute idle gap two block minima gave a
  slope of +1.5 % and paired samples 41 ms apart (check 31.1 failed); with β = 1 a ±50 ppm clock costs at most
  0.5 ms over 10 s. The fit uses only frames of the pod's current epoch that arrived by the analysis time and
  within the last `geometry.clock_window_s` (30 s): causal. A latency common to both pods cancels in the pair.
- **Fresh samples only:** `status == fresh` and `valid`. A held frame repeats a sample and carries nothing new.
- **Pairs:** at each fresh parent sample τ_p, the child is slerped between its bracketing fresh samples τ_c,j ≤ τ_p
  ≤ τ_c,j+1 if they are at most `geometry.max_pair_gap_s` (0.1 s) apart; otherwise there is no pair.
- **Rates** between consecutive fresh samples of one pod, in its own (body) frame, at the midpoint:
  ω = Log(q(t₋)⁻¹·q(t)) / (t − t₋). With q = Y(−ψ)·P and constant ψ, q(t₋)⁻¹·q(t) = P(t₋)⁻¹·Y(ψ)·Y(−ψ)·P(t) =
  P(t₋)⁻¹·P(t): the private north cancels (`test_rates_ignore_a_constant_heading_offset`).
- **Timing check (31.1, scorer side):** for every pair, the child's true time at the slerp point minus the parent's
  true sample time. Bound: 15 ms = the 0–11 ms decode-delay spread (the two pods' delays are independent) + 4 ms
  for the clock fit. Under the default config the latest run gave P50 2.9 ms, P95 7.8 ms, max 12.7 ms
  ([VALIDATION.md](VALIDATION.md)). Sample-task stalls (stress preset) decode a sample up to one chip period late:
  max 19.1 ms there, reported but not tested.

## 10. Step 2: the von Mises toolkit (owned by [mesqsim/conventions.py](../mesqsim/conventions.py))

A belief about an angle is a von Mises density VM(μ, κ) ∝ exp(κ·cos(x − μ)) (Mardia & Jupp 2000). κ = 0 is
uniform ("no information"); for large κ it is nearly normal with variance 1/κ.

| quantity | formula | function | test |
|---|---|---|---|
| mean resultant length | A(κ) = I₁(κ)/I₀(κ) (scaled Bessel functions; asymptotic series 1 − 1/(2κ) − 1/(8κ²) − … above κ = 1000) | `vm_mean_resultant`, `vm_one_minus_resultant` | `test_mean_resultant_is_bessel_ratio_and_inverse_roundtrips` |
| inverse | κ = A⁻¹(r): Best–Fisher start, Newton steps | `vm_kappa_from_resultant` | same |
| product of two beliefs | κ·e^{iμ} = κ₁e^{iμ₁} + κ₂e^{iμ₂} | `vm_product` | `test_product_of_two_von_mises_is_their_resultant` |
| prediction over Δt | A(κ′) = A(κ)·exp(−q·Δt/2); for large κ, κ′ = 1/(1/κ + q·Δt) | `vm_predict_kappa` | `test_predict_rule_and_large_kappa_limit` |
| central interval | half-width h with P(|x − μ| ≤ h) = level; π·level at κ = 0 | `vm_halfwidth_rad` | `test_interval_halfwidth_matches_scipy_and_probability` |
| log density | κ(cos x − 1) − log(2π·i0e(κ)) | `vm_logpdf` | `test_logpdf_is_normalised` |

Cardan angles (`matrix_to_cardan_rad`, intrinsic sequences such as `"XZY"`, pinned against scipy) are added here
for the range-of-motion solver.

## 11. Step 2: the estimator's events (owned by [mesqsim/estimators/geo.py](../mesqsim/estimators/geo.py))

- **Init** at the end of the box phase: μ = B1's estimate wrap(h_p − h_c), κ₀ = 1/σ_pair² with
  σ_pair = √2·`geometry.box_sigma_deg` (the estimator's belief, not the simulator's slot error). A pod without box
  frames, or already in a later boot epoch, gives its joints κ₀ = 0. Before the box ends, the answer is a running
  box registration (frames so far) with κ = 0.
- **Reset:** when Step 0 shows a pod's first frame of a new boot epoch, its joints drop to κ = 0.
- **Window** every `geometry.step_s` from the **motion** phase on (from the T-pose only if a recording has no
  motion phase), never inside an unrecorded phase: each joint is solved over the pairs of the last
  `geometry.window_s` of its pods' current boot epochs. The result enters the filter with weight
  `step_s/window_s`, because consecutive windows share 7/8 of their samples (each sample then counts about once).
  *Why not the T-pose:* it is a calibration snapshot, not joint mechanics. The wearer holds one posture, so a
  static window repeats one configuration ~256 times and claims a huge κ, and the posture's error has nothing to do
  with the joint's axes (the simulator gives every segment 5°, which makes the knee non-hinge): GEO-K's knee was
  3.3° off with κ ≈ 10⁶ through the T-pose before this rule. The T-pose *snapshot* is still used (§12, §15).
- **Model error.** Every window's variance is statistical plus model error: 1/κ_w = 1/κ_stat + σ_m², with σ_m per
  joint type (`geometry.hinge_model_sigma_deg`, `twodof_model_sigma_deg`, `rom_model_sigma_deg`). One window cannot
  pin δ more tightly than the joint model holds; the residual-based κ_stat cannot see a bias that rotates the
  whole window (it is absorbed into δ̂).
- **Output** at a query time: the latest event's belief predicted forward; `uncertain` = κ < `geometry.kappa_hold`.

## 12. Step 2: hinge joints, the knees (owned by [mesqsim/estimators/hinge.py](../mesqsim/estimators/hinge.py))

**Closed form.** The hinge axis is one physical line h; in the parent pod's frame it is u_p, in the child's u_c.
With q_i = Y(−ψ_i)·s_i·m_i: q_p·u_p = Y(−ψ_p)·h and q_c·u_c = Y(−ψ_c)·h, so q_p·u_p = Y(δ)·q_c·u_c with
δ = ψ_c − ψ_p. Let a_t, b_t be the horizontal (x, y) parts of q_p·u_p and q_c·u_c over a window:

```text
C = Σ a_t·b_t      S = Σ (b_t × a_t) = Σ (b_x a_y − b_y a_x)      δ̂ = atan2(S, C)      I = √(C² + S²)
Σ |a_t − Y(δ)·b_t|² = Σ (|a_t|² + |b_t|²) − 2·I·cos(δ − δ̂)
```

With Gaussian noise of variance σ² per component, log-likelihood(δ) = const + (I/σ²)·cos(δ − δ̂): exactly von Mises,
**κ = I/σ̂²**, σ̂² = RSS/(2N − 1), floored by `geometry.hinge_sigma_floor_deg`. A vertical axis has no horizontal
part: I = 0, κ = 0 (the blind spot). One instant with a non-vertical axis already determines δ.

*Worked example (`test_worked_example_12_and_5_degrees_gives_minus_7`, check 22.1):* h = x horizontal,
ψ_p = 12°, ψ_c = 5°: a = (cos 12°, −sin 12°), b = (cos 5°, −sin 5°); C = cos 7° = 0.992546,
S = −sin 7° = −0.121869; δ̂ = −7.000°.

| piece | GEO-K (simulation only) | GEO-ID (deployable) |
|---|---|---|
| axes | u = m⁻¹·e_x with the true mountings; e_x is the thigh's left-right axis at the SMPL rest pose ([anatomy.py](../mesqsim/estimators/anatomy.py)) | identified from body rates: ‖ω_p × u_p‖ = ‖ω_c × u_c‖ at every instant (Seel, Raisch & Schauer 2014), least squares in spherical coordinates from 8 fixed starts (later: warm start + 2), over the rate pairs of the last 120 s with ‖ω_p‖ + ‖ω_c‖ ≥ 20 °/s; accepted when the RMS residual is ≤ 0.5 × the RMS rate |
| relative sign | known | votes, tallied over windows (a fit that moves > 30° resets the tally): **vertical rule** — the true axis has equal vertical components in both pods, so Σ (q_p u_p)_z·(q_c u_c)_z > 0; it votes only when the mean product is ≥ sin²(5°); **T-pose rule** — K = q_p⁻¹·Y(d)·q_c·J_T⁻¹, with J_T the pod-to-pod rotation in the T-pose under box registration's δ, is the knee's rotation since the T-pose: about \|flexion\| for the right d, exactly 180° for d + 180° when the axis is horizontal (a half turn about the vertical composed with a turn about a horizontal axis); it votes when the median angles differ by ≥ 60°. No votes: the window is ambiguous, κ = 0 |

The T-pose rule realises the brief's *Should* ("resolve the sign with the knee's one-way flexion"): it is what
makes GEO-ID work while walking, where the knee axis stays near horizontal and the vertical rule cannot vote.
On a synthetic walk every window voted for the right sign; identified axes were within 0.02°–0.98° of the truth
with fusion noise on (scratch run, 4 sessions) and within 0.001° on noise-free motion (check 24.1).

## 13. Step 2: 2-DoF joints, the elbows, wrists and ankles (owned by [mesqsim/estimators/twodof.py](../mesqsim/estimators/twodof.py))

Axis j1 fixed in the parent and j2 in the child at a constant angle γ (Laidig et al. 2019):
(q_p·j1)·(Y(δ)·q_c·j2) = cos γ. With a = q_p·j1, b = q_c·j2:

```text
P_t·cos δ + Q_t·sin δ + R_t = 0,    P = a_h·b_h,  Q = b_h × a_h,  R = a_z·b_z − cos γ      (one equation per instant)
cost(δ) = Σ (P cos δ + Q sin δ + R)² = A0 + A1 cos δ + B1 sin δ + A2 cos 2δ + B2 sin 2δ
A0 = Σ (P² + Q²)/2 + R²,  A1 = 2ΣPR,  B1 = 2ΣQR,  A2 = Σ (P² − Q²)/2,  B2 = ΣPQ
```

The cost is evaluated on a 0.5° grid and each local minimum refined by Newton steps. **κ** (Laplace) = cost″(δ̂) /
(2σ̂²), σ̂² = cost/(N − 1) (floor `geometry.twodof_sigma_floor_deg`). **Two minima:** a degree-two polynomial can
have two; when the vertical terms vanish (R ≡ 0, e.g. γ = 90° with one axis horizontal) cost(δ) = cost(δ + 180°).
Rule: if the second minimum's cost exceeds the best by less than 2σ̂²·3 (likelihood ratio below e³ ≈ 20), the
window is **ambiguous** and κ = 0 (`geometry.twodof_ambiguity_llr`). Axes (SMPL segment frames, γ = 90°):
elbow j1 = y (flexion, vertical at the rest pose), j2 = x (forearm long axis); wrist j1 = z, j2 = y; ankle
j1 = x, j2 = z. GEO-K uses them with the true mountings. GEO-ID uses the range of motion here (the brief's rule
until j1, j2 are identified from motion, a *Should* not built).

## 14. Step 2: range of motion, the 3-DoF joints (owned by [mesqsim/estimators/rom.py](../mesqsim/estimators/rom.py))

```text
J(δ, t) = (q_p·m_p⁻¹)⁻¹·Y(δ)·(q_c·m_c⁻¹)          = s_p⁻¹·s_c when δ = δ_pc
cost(δ) = Σ_t dist(angles(J(δ, t)), Ω)²           dist = Euclidean distance outside the box (radians), 0 inside
```

- **Angles:** one intrinsic Cardan sequence per joint in SMPL axes ([anatomy.py](../mesqsim/estimators/anatomy.py)):
  lower back, neck, hips `XZY` (the ISB Z-X-Y order of Wu et al. 2002: flexion, lateral bending/abduction, axial
  rotation, rewritten in SMPL axes); shoulders `ZYX` (elevation in the frontal plane, horizontal flexion, axial
  rotation about the humerus, which is horizontal at SMPL's rest pose); 2-DoF joints their universal-joint order.
- **Fast form:** Rz(δ) = P + cos δ·(I − P) + sin δ·[e_z]×, so J(δ, t) = M0_t + cos δ·M1_t + sin δ·M2_t; the 1° grid
  over δ costs a few multiply-adds per sample.
- **Ω:** per angle the 0.5th and 99.5th percentiles of the **training** subjects' true angles,
  `python -m mesqsim fit-geometry rom --source amass --write` (0.01° histograms over every training sequence,
  subsampled to about 30 Hz; `fitted` in [configs/default.yaml](../configs/default.yaml), with the command, data,
  split hash, frame count and commit in each source note). Before the fit the boxes are unbounded and the
  range-of-motion windows carry no information.
- **κ:** the cost is exactly flat for every δ that keeps the motion inside the box, so the curvature at a minimum
  (Laplace) is zero there and blind to how wide the flat region is. The window's von Mises is therefore fitted to
  w(δ) ∝ exp(−(cost(δ) − min cost)/(2σ_rom²)) on the grid by matching its circular mean: μ = arg Σ w·e^{iδ},
  κ = A⁻¹(|Σ w·e^{iδ}| / Σ w). Motion that stays inside the box only rules out δ errors large enough to push the joint
  out of its range: κ ≈ 2 for a ±10° hip sweep inside a wide box (a 95 % interval of ±104°), exactly 0 without a box
  (check 26.1). When the motion reaches the limits, δ̂ comes within 1° on noise-free data (0.37°, check 26.1).

## 15. Step 2: GEO-ID's facing and T-pose frames (owned by [mesqsim/estimators/geo.py](../mesqsim/estimators/geo.py))

The range-of-motion cost needs each pod's segment frame, m_i. The T-pose snapshot gives it up to the wearer's
facing in the room: in the T-pose every segment frame is F_T = Y(φ)·Rx(90°) (the SMPL rest pose, upright, facing φ),
so with the corrections ĉ_i(T) at the T-pose (box registration), Y(ĉ_i(T))·q_i(T) = Y(−ψ_root)·F_T·m_i and
m̂_i = F_T⁻¹·Y(ĉ_i(T))·q_i(T) once φ is known. The brief's way to φ is an optional forward arm raise (§5.8, a
*Should*). This repository gets it from the knees instead, which needs no extra step in the protocol:

1. **Which way the knee axis points.** K = q_p⁻¹·Y(δ̂)·q_c·J_T⁻¹ is the shin's rotation since the T-pose in the
   thigh pod's frame; for a hinge its rotation vector is θ·(the axis pointing to the body's left), θ > 0 in flexion
   (SMPL flexes the knee about +x, the left axis). Windows vote with the sign of the median Log(K)·u_p over samples
   flexed by at least 20° (`geometry.facing_flex_min_deg`).
2. **Facing.** In the T-pose the left-pointing knee axis is the body's left axis: φ = heading of
   Y(ĉ_thigh(T))·q_thigh(T)·u_left; with both knees, their circular mean.
3. **T-pose frames** m̂_i for every pod whose chain from the pelvis had a box prior at the T-pose. A reboot later
   does not invalidate them: m̂_i describes the strap, not the north.

On synthetic walks (scratch run, fusion noise and the 5° T-pose posture error on) m̂ was within a median 5.4–5.7°
of the true mountings, about the posture error. Before the knees are identified, GEO-ID's 2-DoF and 3-DoF joints
hold their box estimate and their κ decays (the brief's rule when facing is not ready).

