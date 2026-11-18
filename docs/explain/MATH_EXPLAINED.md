# The mathematics, derived step by step

*Explanation pipeline, file 3 of 5 · about 90 minutes · [index](README.md) · previous:
[PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md) · next: [CODE_EXPLAINED.md](CODE_EXPLAINED.md)*

This file derives every formula Step 2 uses, and the Step 1 formulas it stands on. Each formula names the
**function** that implements it and the **test** that checks it. Each worked example has a label (E1, E2, …) and is
recomputed by `tests/test_explain_examples.py`, so the numbers here cannot silently drift from the code. Angles are
in degrees in the text and radians in the code. [MATH.md](../MATH.md) is the compact reference version; this file is
the slow, explained one.

---

## 1. Rotations

### 1.1 Matrices and quaternions

An orientation R maps a body's own coordinates to world coordinates: **v_world = R·v_body**. The world has +Z
**up**. A unit quaternion q = (x, y, z, w) — the scalar **last**, as in scipy — encodes the rotation by angle θ about
the unit axis u as

```text
q = (u_x·sin(θ/2), u_y·sin(θ/2), u_z·sin(θ/2), cos(θ/2))
```

Two quaternions compose with the **Hamilton product** `quat_mul(a, b)`, which means *first b, then a* — right to left,
like matrices. q and −q are the same rotation: the angle θ about u and the angle 2π − θ about −u turn every vector the
same way.

Code: [`quat_mul`](../../mesqsim/conventions.py), `quat_rotate`, `quat_to_matrix`. Tests:
`test_hamilton_product`, `test_composition_order_is_right_to_left`, `test_q_and_minus_q_are_the_same_rotation`
(tests/test_conventions.py).

**E1 — Y(90°).** The rotation by α about the vertical is Y(α) = (0, 0, sin(α/2), cos(α/2)). For α = 90°:
Y(90°) = (0, 0, 0.707107, 0.707107), and it maps +x to **(0, 1, 0)**: it turns +x towards +y. That fixes the sign
of every heading in the project (validation check 2.1). Code: `yaw_quat`.

**E2 — order matters.** Let Rz90 = Y(90°) and Rx90 = 90° about x. Applied to +y, Rz90·Rx90 gives **+z** (first x
lifts +y to +z, then the turn about z leaves +z alone), while Rx90·Rz90 gives **−x** (first z turns +y to −x, then x
leaves −x alone). Swapping the order is the most common silent bug (check 2.5).

**E3 — q and −q.** For any q, `quat_to_matrix(q)` and `quat_to_matrix(−q)` are equal, because every entry of the
matrix is a product of two quaternion components, and two sign flips cancel (check 2.4).

### 1.2 Exp, Log, slerp and the geodesic angle

A rotation vector v = θ·u (radians) is turned into a quaternion by **Exp(v) = (u·sin(θ/2), cos(θ/2))**, and back by
**Log**, which returns the angle in [0, π]. Interpolation between two orientations along the shortest arc (slerp) is
q₀·Exp(s·Log(q₀⁻¹·q₁)) for s from 0 to 1. The **geodesic angle** of a rotation — how far it turns — is
2·atan2(‖(x, y, z)‖, |w|); the |w| makes q and −q agree.

Code: `exp_map`, `log_map`, `quat_slerp`, `geodesic_angle_rad`. Test: `test_exp_log_roundtrip_and_geodesic_angle`.

**E6.** Exp((0, 0, 0.3)) is a turn of 0.3 rad about z; its geodesic angle is **0.3 rad** (17.19°).

### 1.3 Heading, wrap, tilt

- **Heading** is the first angle of the intrinsic Z-Y-X decomposition: atan2(R₁₀, R₀₀), the direction of the body's
  x axis projected on the floor. Turning about the vertical adds to it: **heading(Y(α)·R) = wrap(heading(R) + α)**.
  Code: `heading_rad`. Test: `test_heading_shifts_by_alpha` (check 2.2 over 100,000 orientations: 1.8·10⁻¹⁵ rad).
- **wrap** maps an angle into (−180°, 180°]. Code: `wrap_deg`, `wrap_rad`.
- **Tilt** is the world's up direction seen in the body's axes, Rᵀ·e_z (`gravity_in_body`). A rotation about the
  vertical on the left, Y(α)·R, leaves it unchanged, because Rᵀ·Y(α)ᵀ·e_z = Rᵀ·e_z. That is why heading corrections
  never change tilt (check 4.1).

**E4.** wrap(190°) = **−170°**, wrap(−180°) = **180°**, wrap(540°) = **180°**.

**E5.** An orientation R tilted a little and facing about 30° has heading **28.221°**; Y(50°)·R has heading
**78.221°**, exactly 50° more.

## 2. The measurement model, and what cannot be known

### 2.1 Every term

The Step 1 brief §4.2 models what pod i reports as

```text
q_i(t) = Y(−ψ_i(t)) · s_i(t) · m_i · Exp(η_i(t))
```

Read it from right to left, as the composition reads:

| term | meaning | value in the default config (label) |
|---|---|---|
| η_i(t) | the fusion's small tilt-and-noise error, in the pod's axes | 0.5° per axis, correlation time 1 s (assumption) |
| m_i | the **mounting**: how the pod sits on its segment (pod → segment), constant | within ±20° of a nominal placement (assumption) |
| s_i(t) | the segment's **true** orientation (segment → world) | from AMASS motion |
| Y(−ψ_i(t)) | the pod's **private north**: one turn about the vertical, on the left | ψ_i = a_i + b_i·t + k_i·Θ_i + w_i (§2.3 of THEORY) |

The private-north term sits on the *left* — in world coordinates — because it is the reference direction the chip
chose, not a property of the body. Code: [`simulate_session`](../../mesqsim/sim/simulate.py)
(`q = Y(−ψ)·P·Exp(η)` with P = s·m).

### 2.2 Turning the room: why only relative heading is observable

Suppose every pod's north is turned by the same β: ψ_i → ψ_i + β for all i. Every report becomes
Y(−β)·q_i. Now take any two pods p and c and the quantity a joint can see, the rotation between them after a
relative correction δ:

```text
q_p⁻¹ · Y(δ) · q_c    →    (Y(−β)·q_p)⁻¹ · Y(δ) · Y(−β)·q_c  =  q_p⁻¹ · Y(β) · Y(δ) · Y(−β) · q_c  =  q_p⁻¹ · Y(δ) · q_c
```

because rotations about the same axis commute: Y(β)·Y(δ)·Y(−β) = Y(δ). Nothing a joint can see depends on β, and δ
itself, ψ_c − ψ_p, is unchanged by adding β to both. A turned room is invisible; **only the differences of the norths
— the relative headings — can be estimated**.

**E7.** With ψ_p = 12°, ψ_c = 5° the target is δ = −7°; with both turned by β = 73° (85° and 78°) it is still
**−7°**. Check 4.2 does this to a whole simulated session: the labels move by 1.5·10⁻¹³°.

## 3. The target and the outward correction

### 3.1 The target

For a joint from parent pod p to child pod c, the target is

```text
δ_pc(t) = wrap(ψ_c(t) − ψ_p(t))
```

Code: [`relative_heading_rad`](../../mesqsim/correction.py). Tests: `test_worked_example_section_4_4` and
`test_same_example_through_the_tree_functions` (tests/test_correction.py).

### 3.2 The correction, outward from the pelvis

Corrections ĉ_i are accumulated outward from the pelvis pod (the root): ĉ_root = 0, ĉ_c = ĉ_p + δ̂_pc, and each
report is corrected on the left: **q̃_c = Y(ĉ_c)·q_c**. With exact estimates, ĉ_c = ψ_c − ψ_root, so

```text
q̃_c = Y(ψ_c − ψ_root)·Y(−ψ_c)·s_c·m_c = Y(−ψ_root)·s_c·m_c
```

Every corrected pod now shares the **root's** north, which itself stays unknown (§2.2). Code:
`chain_corrections_rad`, `apply_corrections`.

**E8 — the brief's example.** Parent at the root with ψ_p = 12°, child with ψ_c = 5°.

1. δ = wrap(5° − 12°) = **−7°**.
2. ĉ_c = 0 + (−7°), so q̃_c = Y(−7°)·Y(−5°)·s_c·m_c = Y(−12°)·s_c·m_c: the child shares the parent's north.
3. With the wrong sign, Y(+7°)·Y(−5°) = Y(+2°), which differs from Y(−12°) by **14°**.

A sign error doubles the error instead of removing it (check 3.1: right sign 4.8·10⁻¹⁵°, wrong sign 14.000000°).

### 3.3 Why corrections are also applied at the T-pose

The suit's avatar uses **world deltas** relative to the T-pose snapshot q_i(T): Δ_i(t) = q_i(t)·q_i(T)⁻¹. If the
corrections change between the T-pose and now, the consistent world delta of the corrected reports is

```text
q̃_i(t)·q̃_i(T)⁻¹ = Y(ĉ_i(t))·q_i(t)·q_i(T)⁻¹·Y(−ĉ_i(T))
```

so the T-pose snapshot must be corrected **with the corrections that held at the T-pose**, q̃_i(T) = Y(ĉ_i(T))·q_i(T)
(Step 2 brief §5.6). GEO-ID's T-pose frames (§8.3) are built exactly this way, from the box registration in force at
the end of the T-pose.

## 4. Step 0's arithmetic

Step 0 ([step0.py](../../mesqsim/step0.py)) turns raw frames into a series; it never reads labels.

- **count → tick.** count is a 16-bit send counter that wraps at 65,536. Between two frames of a pod the host-time
  gap predicts E = Δt_host/31.25 ms sends; the observed difference (count₂ − count₁) mod 65,536 is unwrapped by the
  number of wraps closest to E. If the result disagrees with E by more than 4 + 200 ppm·E sends, the pod
  **rebooted** and a new epoch starts.
- **ms_lo → pod time.** ms_lo (the pod-clock millisecond of the sample's *decode*, mod 65,536) is unwrapped with the
  tick difference times 31.25 ms; it becomes `t_pod_s`.
- **fresh / held / lost.** The same unwrapped ms_lo as the previous received frame means **held** (a stall resent
  the old sample); a missing tick between two received frames of one recorded interval is **lost**.

**E9.** Counts 65,533, 65,534, 65,535, 0, 1 arriving 31.25 ms apart are ticks 65,533 … **65,537**: one wrap, no loss
(`test_count_wrap_is_continuous`, tests/test_step0.py).

**E10.** Frames at ticks 10, 11, 14, 15 with ms_lo 1300, 1331, 1425, 1425: ticks 12 and 13 are **lost**, and tick 15
is **held** (same ms_lo as tick 14) (`test_lost_ticks_and_held_frames`).

- **Box registration.** In the box each pod lies flat in a slot: its report is Y(−a_i)·Y(box heading + ε_i), so its
  heading is h_i = box heading + ε_i − a_i. Differences cancel the unknown box heading:
  **h_p − h_c = (a_c − a_p) + (ε_p − ε_c)**, the true offset difference plus the slot errors. B1 is
  δ̂ = wrap(h_p − h_c). Code: `box_headings`; check 5.3.
- **T-pose snapshot.** q_i(T) is the mean of pod i's frames in the T-pose phase (`tpose_snapshot`).

## 5. One clock for parent and child

### 5.1 The lower envelope

A pod stamps each sample with its decode time on its own clock, t_pod; the host stamps arrival, t_host. Arrival is
**later** than decode by a positive, variable delay d (waiting for the next send, the radio, the hub):
t_host = α + β·t_pod + d with d ≥ d_min. So the residual r = t_host − t_pod (with β ≈ 1) has a **floor**, and its
smallest values trace the line α + (β − 1)·t_pod + d_min. The code takes the minimum of r in each 2 s block of pod
time and fits a straight line through those minima; the slope only through at least 5 blocks (10 s) and within
±200 ppm, otherwise β = 1. A latency floor common to both pods of a joint shifts both mapped times equally and cancels
in the pair. Code: [`fit_host_map`](../../mesqsim/estimators/clock.py). Tests:
`test_lower_envelope_maps_decode_time_to_host_time_within_two_ms`, `test_a_latency_common_to_two_pods_cancels_in_the_pair`.

**E12.** Four samples at pod times 10.0, 10.5, 11.0, 11.5 s arrive 4, 12, 3 and 20 ms after host time 500 + t_pod:
they fall in one 2 s block, so β = 1 and α = 500 + the smallest delay = **500.003 s**.

### 5.2 Rates ignore the private north

The rotation rate between two consecutive fresh samples, in the pod's own axes, is
**ω = Log(q(t₋)⁻¹·q(t)) / (t − t₋)**. With a constant private north, q = Y(−ψ)·P, so

```text
q(t₋)⁻¹·q(t) = P(t₋)⁻¹·Y(ψ)·Y(−ψ)·P(t) = P(t₋)⁻¹·P(t)
```

and the rate is exactly the true body rate: **rates do not see the north**. Code: `body_rates`. Test:
`test_rates_ignore_a_constant_heading_offset`.

**E13.** For a rate (0.5, −0.2, 1.1) rad/s and any constant ψ, `body_rates` returns (0.5, −0.2, 1.1) exactly
(`test_rates_of_a_constant_rotation_are_exact`).

### 5.3 Pairs

At each fresh parent sample, the child is slerped between its two bracketing fresh samples if they are at most
0.1 s apart (`pair_samples`). The scorer checks the result against the truth: the child's true time at the slerp
point minus the parent's true sample time was P50 2.9 ms, P95 7.8 ms, max 12.7 ms under the default config (check 31.1).

## 6. The hinge: the knees

### 6.1 One line seen from two pods

The hinge axis is one physical line h. In the parent pod's frame it is u_p, in the child pod's u_c, and with the
measurement model

```text
q_p·u_p = Y(−ψ_p)·h      q_c·u_c = Y(−ψ_c)·h      ⟹      q_p·u_p = Y(−ψ_p)·Y(ψ_c)·q_c·u_c = Y(δ)·q_c·u_c
```

Y(δ) leaves the vertical component alone, so only the **horizontal parts** a_t = (q_p u_p)_xy and b_t = (q_c u_c)_xy
carry δ: a_t = R(δ)·b_t, with R(δ) the 2-D rotation.

### 6.2 The closed form, derived

Minimise the squared residual over a window of N pairs:

```text
Σ ‖a_t − R(δ)b_t‖² = Σ ‖a_t‖² + Σ ‖b_t‖² − 2 Σ a_t·R(δ)b_t
```

Write R(δ)b = (b_x cos δ − b_y sin δ, b_x sin δ + b_y cos δ). Then

```text
a·R(δ)b = cos δ (a_x b_x + a_y b_y) + sin δ (b_x a_y − b_y a_x)
Σ a_t·R(δ)b_t = C cos δ + S sin δ,      C = Σ a_t·b_t,      S = Σ (b_t × a_t)
```

and C cos δ + S sin δ = I·cos(δ − δ̂) with **I = √(C² + S²)** and **δ̂ = atan2(S, C)**. So

```text
Σ ‖a_t − R(δ)b_t‖² = Σ (‖a_t‖² + ‖b_t‖²) − 2·I·cos(δ − δ̂)
```

It is smallest at δ = δ̂. Code: [`hinge_closed_form`](../../mesqsim/estimators/hinge.py). Tests:
`test_exact_recovery_over_random_deltas_axes_and_poses` (300 random windows, error < 10⁻⁹ rad; check 22.1(a):
8.9·10⁻¹⁶ rad).

**E14 — the worked example.** A horizontal axis h = x with ψ_p = 12°, ψ_c = 5°: a = (cos 12°, −sin 12°),
b = (cos 5°, −sin 5°). C = cos 12° cos 5° + sin 12° sin 5° = cos 7° = **0.992546**;
S = cos 5°·(−sin 12°) − (−sin 5°)·cos 12° = −sin 7° = **−0.121869**; δ̂ = atan2(−0.121869, 0.992546) = **−7.000°**.

### 6.3 The likelihood is exactly von Mises

Add independent Gaussian noise of variance σ² to every component. The log-likelihood of δ is minus the squared
residual over 2σ²; with §6.2 that is const + **(I/σ²)·cos(δ − δ̂)**: exactly a von Mises density (§9) with mean δ̂ and
concentration **κ = I/σ²**. σ² is estimated from the residual: the minimum sum Σ(‖a‖² + ‖b‖²) − 2I over its 2N − 1
degrees of freedom (2N components, one fitted parameter), floored at (0.5°)² so noise-free data do not give κ = ∞.

**E15.** a = {(1, 0), (0, 1)} and b = a turned by −10°: δ̂ = **10.000°**, I = **2**, residual 0. Perturb a to
{(1, 0.02), (−0.01, 1)}: δ̂ = 10.86°, I = 2.0002, RSS = 5.0·10⁻⁵, σ̂² = RSS/3, **κ ≈ 1.2·10⁵** (a spread
1/√κ ≈ 0.17°). Check 23.1 confirms over 2,000 noisy windows that the spread of δ̂ matches 1/√κ (ratio 0.989).

### 6.3.1 The blind spot

A vertical axis has no horizontal part: a_t = b_t = 0, so C = S = I = 0 and κ = 0 — **no information**, whatever the
norths. **E16:** for u = (0, 0, 1) and any rotations about the vertical, I < 10⁻¹² and κ = 0 (check 22.1(c);
`test_vertical_axis_is_the_blind_spot`).

### 6.4 Finding the axis from motion

**Why the rate condition holds.** For a hinge, the child's world angular velocity is the parent's plus a turn about
the axis: ω_c^w = ω_p^w + θ̇·j^w. The components perpendicular to j^w are therefore equal as vectors, so their lengths
are equal, and lengths do not depend on the frame they are written in:

```text
‖ω_p × u_p‖ = ‖ω_c × u_c‖      at every instant, in each pod's own axes      [Seel 2014]
```

GEO-ID fits u_p, u_c by least squares on these residuals (`identify_hinge_axes`; 0.001° on noise-free motion, check
24.1). **Two failures and their fixes:**

1. *Rigid motion.* If thigh and shin turn together (θ̇ = 0), ω_c is ω_p written in another frame, and every pair
   (u_p, u_c = R_rel·u_p) satisfies the condition: nothing is identified. **Contrast test:** the best cost must be at
   most a fraction of the median cost over alternative axes: 0.25 for the free four-angle fit, 0.8 for the T-pose-tied
   fit below (`geometry.axis_contrast_max`, `geometry.axis_tpose_contrast_max`). But in a rigid motion *every* cost is
   near zero, and their ratio is just noise (0.79 in the test below), so the tied fit must also show that a wrong axis
   fits badly: the RMS residual at the median azimuth must be at least 0.1 × the RMS rate (`geometry.axis_min_spread`;
   rigid motion: below 0.01, `test_tpose_tied_identification_recovers_the_axis_and_refuses_rigid_motion`).
2. *Too many unknowns.* **T-pose tie:** in the T-pose the knee is straight and its axis horizontal, so u_p ⊥ g_p (world
   up in the thigh pod frame) and u_c = q_c(T)⁻¹·Y(−δ_T)·q_p(T)·u_p (one line in both segments, with box
   registration's δ_T). Only the azimuth of u_p in the horizontal plane is left: a 1° grid and a refinement
   (`identify_hinge_axes_tpose`). A local four-angle refinement then corrects T-pose posture error, kept within 15°.
   On 60 AMASS validation sessions this took the axes from a median 19.5° to 5.8° off the truth (WORKLOG, S2-M4).

### 6.5 The sign

The condition cannot tell u from −u. Flipping **one** of u_p, u_c turns b into −b, which moves δ̂ by **180°**. Two
rules vote:

- **Vertical rule.** Y(δ) leaves vertical components alone, so for the true axis (q_p u_p)_z = (q_c u_c)_z and
  Σ (q_p u_p)_z (q_c u_c)_z > 0. It votes when the mean product is at least sin²(5°) (`vote_vertical`).
- **T-pose rule.** K = q_p⁻¹·Y(d)·q_c·J_T⁻¹ is the knee's rotation since the T-pose under the hypothesis d. For the
  right d it is a turn of |flexion| about the axis. For d + 180° it is a **half turn about the vertical composed with
  a turn about a horizontal axis**, whose angle is exactly 180° whatever the flexion (`vote_tpose`).

**E17.** Compose Y(180°) with a turn θ about the horizontal x axis: for θ = 0°, 30°, 90° the product's w component is
0 (to 10⁻¹⁶), so its angle is **180°** each time — while the correct hypothesis would show 0°, 30°, 90°.

With the T-pose tie the relative sign is fixed by construction; the votes remain as a cross-check.

## 7. The 2-DoF constraint: elbows, wrists, ankles

### 7.1 One equation per instant

The first axis j₁ is fixed in the parent, the second j₂ in the child, at a constant angle γ [Laidig 2019]. In world
terms (q_p·j₁)·(Y(δ)·q_c·j₂) = cos γ. Write a = q_p·j₁ and b = q_c·j₂ with horizontal parts a_h, b_h and vertical parts
a_z, b_z. Y(δ) rotates only the horizontal part, so (§6.2's algebra)

```text
a·Y(δ)b = a_h·R(δ)b_h + a_z b_z = P cos δ + Q sin δ + a_z b_z,     P = a_h·b_h,  Q = b_h × a_h
P_t cos δ + Q_t sin δ + R_t = 0,     R_t = a_z b_z − cos γ
```

**E18 — check by hand.** With exact data and the true δ, Y(δ)·q_c·j₂ is the child's axis in the *parent's* north, so
the dot product is the true cos γ and every equation holds (`test_every_equation_holds_at_the_true_delta`; check 25.1:
error 5.0·10⁻¹³ rad over 120 random windows).

### 7.2 The cost is a trigonometric polynomial

Square and sum, using cos² = (1 + cos 2δ)/2, sin² = (1 − cos 2δ)/2, sin·cos = sin 2δ/2:

```text
cost(δ) = A0 + A1 cos δ + B1 sin δ + A2 cos 2δ + B2 sin 2δ
A0 = Σ (P² + Q²)/2 + R²,   A1 = 2ΣPR,   B1 = 2ΣQR,   A2 = Σ (P² − Q²)/2,   B2 = ΣPQ
```

Five sums describe the whole cost (`twodof_coefficients`; `test_trig_polynomial_equals_the_sum_of_squares`). It is
evaluated on a 0.5° grid and every local minimum is refined by Newton steps.

### 7.3 Two minima

A polynomial in cos 2δ and sin 2δ has period 180°. When the vertical terms vanish (R_t = 0 for all t, e.g. γ = 90° with
one axis horizontal), A1 = B1 = 0 and **cost(δ) = cost(δ + 180°)**: two equally good answers. The rule: if the second
minimum's cost exceeds the best by less than 2σ̂²·3 (a likelihood ratio below e³ ≈ 20), the window is **ambiguous**
and κ = 0. **E19:** the built twin of check 25.1 has a cost gap of 0 and is flagged.

### 7.4 κ by the Laplace approximation

Near the minimum, cost(δ) ≈ cost(δ̂) + ½·cost″(δ̂)·(δ − δ̂)². With residual variance σ̂² = cost/(N − 1), the
log-likelihood is −cost/(2σ̂²), whose curvature is cost″/(2σ̂²): **κ = cost″(δ̂)/(2σ̂²)**.

## 8. The range of motion: ball joints

### 8.1 The cost

```text
J(δ, t) = (q_p·m_p⁻¹)⁻¹ · Y(δ) · (q_c·m_c⁻¹)          = s_p⁻¹·s_c, the joint rotation, when δ = δ_pc
cost(δ) = Σ_t dist(angles(J(δ, t)), Ω)²                 dist = Euclidean distance outside the box, 0 inside
```

angles(·) is the joint's Cardan sequence (`matrix_to_cardan_rad`; the hips, spine and neck use SMPL `XZY`, the ISB
Z-X-Y order) and Ω one interval per angle, fitted on the training subjects' true angles (the 0.5th and 99.5th
percentiles; `fit_limits_deg`).

### 8.2 The fast form, derived

A turn about z splits into its parts along and across the axis: with P = e_z·e_zᵀ (projection on z) and
K = [e_z]× (the cross product with z),

```text
Rz(δ) = P + cos δ·(I − P) + sin δ·K
```

(on the z component it is the identity; on the horizontal plane it is the 2-D rotation). With matrices
A = R(q_p·m_p⁻¹) and B = R(q_c·m_c⁻¹):

```text
J(δ) = Aᵀ·Rz(δ)·B = M0 + cos δ·M1 + sin δ·M2,     M0 = AᵀPB,  M1 = Aᵀ(I − P)B,  M2 = AᵀKB
```

so a whole grid of δ costs only a few multiply-adds per sample (`rom_terms`, `rom_angles`;
`test_rom_fast_form_equals_direct_joint_rotation`).

### 8.3 Why κ is near zero at the hip

Every δ that keeps the motion inside the box has cost **exactly 0**. The cost is flat over that whole interval, so
its curvature (Laplace) is 0 there and says nothing about how wide the flat region is. The code therefore fits the
window's von Mises by **matching the circular mean** of w(δ) ∝ exp(−(cost − min cost)/(2σ_rom²)) on a 1° grid:
μ = arg Σ w·e^{iδ}, κ = A⁻¹(|Σ w·e^{iδ}| / Σ w).

**E21.** If the motion rules out only δ errors beyond ±80° (w uniform on ±80°), the mean resultant length is
sin(80°)/(80° in radians) = **0.7053**, so **κ = 2.05**: a 95 % interval wider than ±100°. That is almost no
information — and the expected answer at a hip in everyday motion. Check 26.1 finds κ = 1.97 for such a case, exactly 0
without any box, and an error under 0.4° when the motion does reach the limits.

## 9. The von Mises distribution and the filter

### 9.1 Density, A(κ), intervals

```text
VM(x; μ, κ) = exp(κ·cos(x − μ)) / (2π·I₀(κ))         A(κ) = E[cos(x − μ)] = I₁(κ)/I₀(κ)
```

For large κ, A(κ) ≈ 1 − 1/(2κ) and the distribution is close to a normal one with variance 1/κ [Mardia 2000]. Code:
`vm_mean_resultant` (scaled Bessel functions, an asymptotic series above κ = 1000), `vm_kappa_from_resultant`
(the inverse by Newton steps), `vm_halfwidth_rad` (central intervals). Tests in tests/test_conventions_step2.py.

**E22.** A(1) = **0.4464**; A(1000) = **0.99950**, against 1 − 1/2000 = 0.99950.

**E23.** For κ = 30 the central 95 % interval has half-width **20.70°** and the 68 % interval **10.46°**.

### 9.2 The product of two beliefs

```text
κ₁cos(x − μ₁) + κ₂cos(x − μ₂) = Re[(κ₁e^{iμ₁} + κ₂e^{iμ₂})·e^{−ix}] = κ·cos(x − μ),     κ·e^{iμ} = κ₁e^{iμ₁} + κ₂e^{iμ₂}
```

so the product of two von Mises densities is von Mises (`vm_product`; `test_product_of_two_von_mises_is_their_resultant`).

**E24.** κ₁ = 4 at 10° and κ₂ = 2 at 40° give **μ = 19.90°, κ = 5.819**: pulled towards the more confident belief,
and more confident than either.

### 9.3 Prediction

Heading drift is modelled as a random walk of variance q per second. A random walk over Δt is a wrapped normal whose
mean resultant length is e^{−q·Δt/2}; the belief keeps its mean and loses resultant length:

```text
A(κ′) = A(κ)·exp(−q·Δt/2)          large κ:   1 − 1/(2κ′) ≈ (1 − 1/(2κ))(1 − qΔt/2)   ⟹   κ′ ≈ 1/(1/κ + q·Δt)
```

— the variances add (`vm_predict_kappa`; `test_predict_rule_and_large_kappa_limit`; check 27.1).

**E25.** κ = 1000, q = 10⁻⁵ rad²/s, Δt = 60 s: the exact rule gives **κ′ = 625.3**; the large-κ formula gives 625.0.

### 9.4 The filter, event by event

Per joint ([vmfilter.py](../../mesqsim/estimators/vmfilter.py)): **init** at the end of the box phase with B1's estimate
and κ₀ = 1/(2σ_box²) (= 410 for σ_box = 2°); **predict** between events; **update** with each window's (δ̂_w, κ_w)
by the product; **reset** to κ = 0 when a pod of the joint starts a new boot epoch. Each window enters with weight
step/window = 1/8, because consecutive 8 s windows share 7/8 of their samples. At a query time the answer is the
latest event's belief predicted forward to it: causal by construction (check 21.1).

## 10. Calibration

A window's statistical κ assumes independent residuals; real ones are correlated, and the joint model is only
approximately right. The filter therefore receives

```text
κ_w = s · κ_stat / (1 + κ_stat·σ_m²)          variance = (1/κ_stat + σ_m²) / s
```

with a scale s and a model error σ_m per joint type, fitted by maximum likelihood of the window errors on the
**validation** split (`fit_calibration`; `python -m mesqsim fit-geometry calibration --source amass`). The drift q
is the value on a log grid that maximises the filter's log-likelihood at every motion step, and κ_hold the smallest
threshold whose retained steps have a P95 error of at most 10°.

**E26.** GEO-ID's fitted knee calibration is s = 0.011776, σ_m = 0.9385°. For a well-observed window (κ_stat → ∞)
κ_w tends to s/σ_m² = **43.89**, a spread of **8.65°**; at κ_stat = 10⁴ it is **31.97** and at 10⁵ **42.31**. Both
fitted numbers mean something: s ≪ 1 says a raw window κ overstates its information by orders of magnitude, and the
cap says no knee window is worth more than about ±8.6° (the SMPL knee is not an exact hinge, and identified axes are
a few degrees off).

## 11. Metrics and statistics

- **Primary metric:** |wrap(δ̂ − δ)| per joint and 32 Hz step. **Secondary:** the geodesic angle of each joint's error
  rotation with the true mountings (`Scorer.errors`).
- **Coverage:** the share of steps with |error| ≤ h(κ, level), for 68 % and 95 % (`coverage`).
- **Selective risk:** the error of the steps kept when only the most confident share f is kept (`selective_risk`).
- **Paired bootstrap over subjects.** For each subject the median error of GEO-ID and of B1 over its test steps; the
  differences d_s are resampled with replacement 10,000 times; the 95 % percentile interval of their mean and a
  two-sided p = 2·min(P(mean* ≤ 0), P(mean* ≥ 0)) (`bootstrap_contrast`).
- **Holm:** sort the m p-values; multiply the i-th smallest by m − i + 1; make them non-decreasing (`holm`).

**E28.** Differences (−3, −1, −2, 0.5, −1.5) over five subjects: mean **−1.4**, 95 % interval **[−2.4, −0.4]**,
p = **0.0092** (10,000 resamples, the code's fixed seed). The interval excludes 0 and the improvement exceeds 1°:
"better" under the draft plan's rule.

**E29.** p = 0.01, 0.04, 0.03 for C1, C2, C3: Holm gives **0.03** (C1: 3 × 0.01), **0.06** (C3: 2 × 0.03), **0.06**
(C2: 1 × 0.04, raised to stay non-decreasing).

## 12. Self-check questions

1. **Why does Y(δ) act only on the horizontal part of a vector?** It is a rotation about the vertical: it leaves the
   z component unchanged and turns (x, y) by δ.
2. **Derive δ̂ = atan2(S, C) in one line.** The residual is a constant minus 2(C cos δ + S sin δ) = 2I·cos(δ − δ̂),
   largest at δ = atan2(S, C).
3. **Why is the hinge κ zero for a vertical axis?** Its horizontal parts vanish, so C = S = I = 0.
4. **Why can a rigid thigh-and-shin motion not identify the knee axis?** The rate condition then holds for every
   axis pair related by the fixed relative rotation; the contrast test detects this.
5. **What does flipping only u_c do?** b → −b, so δ̂ moves by 180°.
6. **Why does the 2-DoF cost have period 180° when R_t = 0?** It then contains only cos 2δ and sin 2δ terms.
7. **Why is the range-of-motion κ almost 0 inside the box?** The cost is exactly flat there; moment matching of a
   near-uniform weight gives a resultant length near 0.
8. **Give the filter's update and prediction in one line each.** κe^{iμ} ← κe^{iμ} + κ_w e^{iδ̂_w};
   A(κ) ← A(κ)·e^{−qΔt/2}.
9. **What does the calibration cap s/σ_m² mean?** The largest concentration any single window can claim, whatever its
   residual says.
10. **Why are corrections applied at the T-pose snapshot too?** So that world deltas q̃(t)·q̃(T)⁻¹ compare reports in
    the same north; otherwise the change of correction since the T-pose would appear as motion.

---

*Next: [CODE_EXPLAINED.md](CODE_EXPLAINED.md) shows how all of this is built.*
