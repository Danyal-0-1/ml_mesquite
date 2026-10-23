# The mathematics

*Status: M3. Conventions, target and correction, the measurement model as far as the M3 slice uses it (offsets),
the timing model, Step 0 and the metrics. M4 adds drift, scale factor, random walk, tilt error, loss, stalls,
reboots, box registration, B1 and O1.*

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

| statement | test |
|---|---|
| Rotating every report by one Y(β) changes no target δ (exactly) | `test_common_yaw_leaves_targets_unchanged_exactly` |
| Rotating one pod's frame by Y(γ) changes only the targets of its joints, by exactly ±γ | `test_single_pod_offset_moves_only_its_joints_by_gamma` |
| A correction acts about the vertical only, so tilt never changes | `test_correction_never_changes_tilt`, check 4.1 |

## 5. The measurement model and the session

```text
s_i(t)   true orientation of segment i (segment -> world); segment frame = SMPL canonical body frame
m_i      mounting of pod i on its segment (pod -> segment), constant
P_i(t)   = s_i(t)·m_i, the true pod orientation (pod -> world)
η_i(t)   small tilt-and-noise error of the chip's fusion (M4)

what pod i reports:   q_i(t) = Y(−ψ_i(t)) · P_i(t) · Exp(η_i(t)),   canonicalised to w ≥ 0, then quantised
heading of pod i's private frame:   ψ_i(t) = a_i + b_i·t + k_i·Θ_i(t) + w_i(t)   (+ a jump at each reboot)
```

In M3, ψ_i(t) = a_i: a constant offset per pod.

| offset mode | a_i | meaning |
|---|---|---|
| `arbitrary` (default) | uniform in (−180°, 180°] | each chip picks an unrelated north |
| `aligned` | heading of the pod in its box slot, wrap(box heading + ε_i) | the chip's north is its own start-up heading, so pods booted together agree up to the slot errors ε_i ~ N(0, σ_box²) |
| `none` | 0 | for checks only |

**The session in true time** (seconds from the first power-on):

1. **Box.** Pods lie flat with aligned axes: P_i = Y(box heading + ε_i).
2. **Strapping.** Each pod moves to its strap during its own window:
   P_i = Y(2π·n_i·g)·slerp(P_box, P_worn, g), with g a smoothstep of time and n_i whole turns.
3. **T-pose.** s_i = Y(φ_T)·Rx(90°)·Exp(e_i): the SMPL rest pose, upright, facing φ_T, with posture error e_i
   (RMS angle 5°).
4. **Idle.** Not recorded. The wearer holds the T-pose for the first half and the motion's first pose for the second.
5. **Motion.** s_i(t) = Y(φ_m)·S_i(t − t_motion), with φ_m chosen so the motion starts facing φ_T ± 15°.

φ_T and the box heading are uniform: the room's heading is unknowable and the simulator draws it at random.

**Timing (§3.3 of the brief):**

- The chip samples at f_i ~ U(50.1, 60.0) Hz, starting 0.2–0.5 s after boot.
- Each sample is decoded 0–11 ms later. `ms_lo` = floor(pod ms at decode) mod 65,536. The pod clock runs at
  1 + ρ_i, with ρ_i within ±50 ppm.
- Sends follow the pod clock at ms = 1000 + id·31.25/17 + j·31.25 and carry the newest decoded sample. With
  f_i > 32 Hz and decode jitter ≤ 11 ms, no send ever finds nothing new, so held frames need stalls (M4).
- `count` = j mod 65,536.
- Arrival = send + 2 ms + Exp(mean 3 ms), first-in first-out per pod; host time = arrival + an arbitrary epoch.

## 6. Step 0 (owned by [mesqsim/step0.py](../mesqsim/step0.py); never reads labels)

Per pod, in arrival order:

1. **count → tick.** The host-time gap between two frames predicts the number of sends, E = Δt_host / 31.25 ms.
   The observed count difference is unwrapped by the number of 65,536-wraps closest to E, giving Δtick.
   - If |Δtick − E| > 4 + 200 ppm · E, the pod **rebooted**: a new `epoch` starts with tick = count.
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

| baseline | δ̂ | from |
|---|---|---|
| B0 | 0 | M3 |
| O2 | the true δ (oracle) | M3 |
| B1 | box registration, plus the T-pose | M4 |
| O1 | the true offsets a_c − a_p (oracle) | M4 |
