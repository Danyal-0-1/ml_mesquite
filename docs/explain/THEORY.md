# The theory: where the ideas come from

*Explanation pipeline, file 5 of 5 · about 60 minutes · [index](README.md) · previous:
[CODE_EXPLAINED.md](CODE_EXPLAINED.md)*

The other files explain *what* the repository does and *how*. This one explains *why it can work at all*, from first
principles to the research it rests on. Each section ends with what it means for Mesquite. Citations in brackets,
such as [Seel 2014], point to the full references in §10. Numbers measured in this repository name the command and
commit that produced them.

---

## 1. Rotations and their representations

### 1.1 The group of rotations

A rotation of three-dimensional space keeps lengths, angles and handedness. Written as a 3×3 matrix R it satisfies
RᵀR = I and det R = +1. All such matrices form a group, **SO(3)**: composing two rotations gives a rotation, and every
rotation can be undone. SO(3) is *not* flat like ordinary space. It is a curved three-dimensional manifold, which is
why "averaging two rotations" or "adding a small rotation" needs care [Solà 2018].

An **orientation** is a rotation that maps a body's own axes to the world's axes; in this repository
v_world = R·v_body, and the world has +Z pointing up. Composition reads right to left: R_world←seg·R_seg←pod maps pod
axes to world axes. Most bugs in the project are a swapped order or a missing inverse ([MATH.md](../MATH.md) §2).

### 1.2 Quaternions and the double cover

A unit quaternion q = (x, y, z, w) with x² + y² + z² + w² = 1 encodes the rotation by angle θ about the unit axis u as
(u·sin(θ/2), cos(θ/2)). The **Hamilton product** of quaternions composes rotations. Two facts matter:

- **The double cover.** q and −q encode the *same* rotation (θ about u equals 2π − θ about −u). The unit quaternions
  wrap twice around SO(3). Every algorithm must treat q and −q as equal; the wire format removes the ambiguity by
  forcing w ≥ 0 (the pod rebuilds w = √(1 − x² − y² − z²)).
- **Storage order.** This project, like `scipy.spatial.transform.Rotation`, stores (x, y, z, w) — scalar last. Many
  papers and tools (VQF, DIODEM's files) put w first; the conversion happens exactly once, at load time.

### 1.3 The exponential and logarithm maps

Near any rotation, SO(3) looks like flat three-dimensional space: its **tangent space**. A **rotation vector**
v = θ·u (axis times angle) lives there, and the **exponential map** Exp(v) turns it into a rotation; the **logarithm**
Log(R) turns a rotation back into a rotation vector with angle in [0, π]. These maps give principled tools [Solà 2018]:

- **Angular rate** between two orientations a short time Δt apart: ω = Log(q(t₋)⁻¹·q(t))/Δt, in the body's own axes.
- **Interpolation (slerp):** q₀·Exp(u·Log(q₀⁻¹·q₁)) moves along the shortest great circle.
- **The geodesic angle** 2·atan2(‖v‖, |w|) of the error rotation between two orientations is the rotation distance.

### 1.4 Euler and Cardan angles

Any rotation can be written as three successive rotations about named axes. With three *different* axes (a Cardan
or Tait–Bryan sequence, such as Z-Y-X) the decomposition has a **singularity** (gimbal lock) when the middle angle is
±90°: the first and third axes line up and their angles stop being separately defined. Biomechanics nonetheless
reports joint motion as Cardan angles, because each angle has an anatomical meaning (flexion, abduction, axial
rotation); the International Society of Biomechanics fixes which sequence to use for which joint [Grood 1983;
Wu 2002; Wu 2005], including for wearable sensors [Cereatti 2024]. **Heading** in this project is the first angle of
the intrinsic Z-Y-X decomposition: the direction of the body's x axis projected on the floor.

### 1.5 Why representations will matter for learning

Every representation of 3-D rotations in four or fewer numbers is discontinuous somewhere (quaternions at the double
cover, Euler angles at gimbal lock); neural networks regress continuous representations, such as two columns of the
rotation matrix, more accurately [Zhou 2019]. Step 3 will feed orientations to networks, so this choice comes back
there.

**For Mesquite.** Every rotation convention lives in one module ([conventions.py](../../mesqsim/conventions.py)),
pinned by tests against scipy; Step 2 adds Cardan angles and the von Mises tools there, under the same rule.

## 2. Inertial sensing and six-axis fusion

### 2.1 What the sensors measure

A **gyroscope** measures angular rate ω in its own axes. Integrating q̇ = ½·q⊗(ω, 0) propagates orientation. An
**accelerometer** measures specific force: gravity's reaction plus linear acceleration. Averaged over time, or when
the body moves gently, it points "up" in the sensor's axes, which fixes the two **tilt** angles [Kok 2017].

### 2.2 Why heading is unobservable without a magnetometer

Rotate the sensor about the gravity vector. The accelerometer sees exactly the same vector; only the gyroscope notices,
and only the *change*. So an absolute heading has no measurement at all: a six-axis fusion (gyroscope and
accelerometer, no magnetometer) must **choose** a reference heading at start-up and then **integrates** heading
changes, accumulating every gyroscope error along the way [Kok 2017; Laidig 2023]. Mesquite's pods run such a fusion:
each pod's heading reference — its "private north" — is arbitrary at boot and drifts.

### 2.3 Gyroscope error terms

The Step 1 brief models the heading error of pod i as ψ_i(t) = a_i + b_i·t + k_i·Θ_i(t) + w_i(t) (+ a jump at each
reboot). Each term has a physical cause:

| term | cause | value used (source) |
|---|---|---|
| offset a_i | the reference chosen at boot | arbitrary or box-aligned (assumption; Step 1 brief §4.3) |
| drift b_i | residual gyroscope bias after on-chip correction | σ_b = 0.2 °/min, swept 0.05–2.0 (assumption; the 2.0 value from the real suit's static hold, Step 2 brief §4.1) |
| scale factor k_i | the rate is mis-scaled, so every turn adds k·360° | ±1.5 % (datasheet as quoted by the Step 1 brief: ICM-20948, DS-000189 rev 1.3, Tables 1 and 5) |
| angle random walk w_i | white rate noise integrates to a random walk | 0.015 °/s/√Hz, so σ = 0.367° after 10 min (same datasheet) |

The simulator reproduces these scalings (validation checks 6.1–6.3 in [VALIDATION.md](../VALIDATION.md)).

### 2.4 The closed on-chip fusion

Mesquite's chip runs a fusion whose internals are closed. The repository therefore models only its *output*:
correct tilt up to a small error η (0.5° per axis, an assumption), and the heading error above. What it cannot tell
us — how fast the bias is learned, how motion or temperature affect it — is exactly what the bench measurements are
for. The open VQF algorithm [Laidig 2023] plays the chip's role on DIODEM, where its six-axis relative heading
drifted −80.7° in 6.1 minutes on one hinge (validation check 11.3).

## 3. Gauge freedom and identifiability

### 3.1 The room-rotation symmetry

Turn every pod's private north by the same angle β. Every report changes by the same rotation about the vertical, on
the left: q_i → Y(β)·q_i. Every relative quantity between two pods, q_p⁻¹·Y(δ)·q_c, is unchanged when δ is measured
between the *same* norths. Equivalently, turn the room with the wearer inside: the pods cannot tell. This is a
**gauge freedom**: one global parameter (the common heading) that no measurement constrains. Only quantities invariant
under it are identifiable — here, the **relative headings δ** between pods [Kok 2022].

**For Mesquite.** The targets are the 14 relative headings, never an absolute one. Validation checks 4.2 and 28.1
turn every north by 73° and confirm the labels move by 10⁻¹³° and the GEO estimates by at most 0.03°.

### 3.2 Relative heading is observable — sometimes

For two segments joined by a joint, the joint's mechanics relate their orientations. Whether the relative heading
can then be recovered depends on the joint type and the **excitation**: Kok et al. analyse the observability of
relative motion in kinematic chains and show that it depends on the joint's constraints and on the motion [Kok 2022].
The recent review by Kok et al. notes that one- and two-dimensional joint methods lose the relative heading while the
joint axis is vertical [Kok 2026, §4.2] — the **blind spot** this repository tests (validation check 22.1(c)).

### 3.3 Turning the identifiability argument into tests

Three symmetries become tests: a common yaw changes nothing; one pod's yaw changes only its two joints, by exactly ±γ;
a heading correction never changes tilt ([MATH.md](../MATH.md) §4; checks 4.1–4.3, 28.1).

## 4. Kinematic constraints

### 4.1 Hinge joints and axis identification

A **hinge** (one degree of freedom) turns about one physical axis j fixed in both segments. The angular velocity of
the child minus that of the parent is along j. Hence the components of the two angular velocities *perpendicular* to
j are one vector, and in each sensor's own axes **‖ω_p × u_p‖ = ‖ω_c × u_c‖** at every instant, where u_p, u_c are j
in the two sensor frames. Seel et al. used this to identify u_p and u_c by least squares from gait data, without any
heading information [Seel 2014]. Olsson et al. made axis estimation robust and added a success indicator, but they
need accelerations, which Mesquite does not transmit [Olsson 2020].

Two weaknesses matter for Mesquite. First, a **rigid motion** (thigh and shin turning together) satisfies the
condition for *every* pair of axes with matching angles, so the fit is not identified; on AMASS validation the free
fit's axes were a median 19.5° from the truth (scratch diagnostic recorded in WORKLOG, S2-M4). Second, the condition
cannot tell u from −u. This repository adds two constraints from the **T-pose** (the knee axis is horizontal there
and one line in both segments) and a contrast test, which cut the axis error to a median 5.8°
([MATH.md](../MATH.md) §12).

**Relative heading from the axis.** Once u_p, u_c are known, the axis's *horizontal* parts in the two pods' worlds
differ only by the rotation Y(δ); the least-squares δ has a closed form and an exact von Mises likelihood
([MATH_EXPLAINED.md](MATH_EXPLAINED.md) §6).

### 4.2 Two-degree-of-freedom joints

Elbow (flexion and forearm rotation), wrist and ankle behave approximately like **universal joints**: a first axis j₁
fixed in the parent, a second j₂ fixed in the child, at a constant angle γ. Laidig et al. showed that the constraint
(R_p·j₁)·(R_c·j₂) = cos γ gives magnetometer-free relative heading in real time [Laidig 2019] and later estimated j₁,
j₂ from motion as well [Laidig 2022]. Each instant gives one equation in δ, so the solution can be ambiguous (two
minima). In this repository the constraint is exact on synthetic universal joints (check 25.1), but SMPL's elbows,
wrists and ankles turn 9.1–20.4° (RMS) about the third axis on AMASS training motion
(`python -m mesqsim fit-geometry rom --source amass`, commit `894ba79`), which limits it.

### 4.3 Range of motion

Ball joints (hip, shoulder, spine, neck) have three degrees of freedom: no axis constrains δ. Lehmann et al. used the
**range of motion**: with δ wrong, the implied joint rotation leaves its anatomical range; with 8 s windows re-solved
every second, they tracked arbitrary joints without magnetometers [Lehmann 2020]. The constraint is informative only
when the motion approaches the limits — which everyday motion seldom does. McGrath and Stirling combine such
constraints with sensor models in a factor graph [McGrath 2020], a framework that could unify the three joint types.

### 4.4 Joint conventions

The ISB recommendations fix each joint's coordinate systems and Cardan sequence [Grood 1983; Wu 2002; Wu 2005;
Cereatti 2024]. This repository writes the hip, spine and neck sequence (ISB Z-X-Y) in SMPL's axes as `XZY`, and uses a
Cardan sequence regular at SMPL's arms-out rest pose for the shoulder instead of ISB's Y-X-Y
([anatomy.py](../../mesqsim/estimators/anatomy.py)).

## 5. Directional statistics

### 5.1 The von Mises distribution

An angle is not a number on a line: 359° is close to 1°. The natural "normal distribution" on the circle is the
**von Mises** distribution VM(μ, κ) with density exp(κ·cos(x − μ)) / (2π·I₀(κ)), where I₀ is a modified Bessel function
[Mardia 2000, §3.5]. κ ("kappa") is the **concentration**: κ = 0 is uniform, and for large κ the distribution is close
to a normal one with variance 1/κ.

### 5.2 Mean, variance and A(κ)

The **mean resultant length** E[e^{i(x−μ)}] = A(κ) = I₁(κ)/I₀(κ) runs from 0 (uniform) to 1 (a point). The circular
mean of samples is the direction of their average unit vector, and 1 − R (R its length) is the circular variance
[Mardia 2000]. Linear averages of angles are wrong near ±180°; every heading statistic in this repository is circular.

### 5.3 Intervals

A central interval [μ − h, μ + h] holding a given probability has a half-width h(κ) that is computed numerically from
the von Mises distribution function; it tends to π·level at κ = 0 and to z/√κ for large κ ([conventions.py](../../mesqsim/conventions.py), `vm_halfwidth_rad`).

### 5.4 Products

Multiply two von Mises densities: the exponent is κ₁cos(x − μ₁) + κ₂cos(x − μ₂), which equals κ·cos(x − μ) with
κ·e^{iμ} = κ₁e^{iμ₁} + κ₂e^{iμ₂}. So **the product of two von Mises densities is von Mises** (up to normalisation) —
the basis of the filter's update and of fusing two estimators in Step 4.

### 5.5 The wrapped normal

Wrapping a normal distribution N(μ, σ²) around the circle gives the wrapped normal, whose mean resultant length is
exactly e^{−σ²/2}. A random walk in heading is a wrapped normal with σ² growing linearly in time, so its resultant
length shrinks by e^{−q·Δt/2}. Matching it to a von Mises belief gives the filter's prediction rule
A(κ′) = A(κ)·e^{−q·Δt/2} [Mardia 2000].

## 6. Estimation

### 6.1 Least squares and maximum likelihood

With independent Gaussian errors, maximising the likelihood is minimising the sum of squared residuals. For the
hinge, the residual sum Σ‖a_t − Y(δ)b_t‖² equals a constant minus 2·I·cos(δ − δ̂), so the likelihood in δ is exactly
von Mises with κ = I/σ² ([MATH_EXPLAINED.md](MATH_EXPLAINED.md) §6). No approximation is needed.

### 6.2 The Laplace approximation

When the likelihood is not von Mises, expand the negative log-likelihood to second order at its minimum: the
curvature gives a Gaussian (or von Mises) approximation with concentration equal to the curvature. This is how the
2-DoF solver obtains κ = cost″(δ̂)/(2σ̂²). It fails when the minimum is flat — as the range-of-motion cost is inside
its box — so there the repository matches the first circular moment of exp(−cost/2σ²) instead.

### 6.3 Recursive Bayesian filtering on the circle

A belief about δ is carried forward in two steps: **predict** (the heading diffuses: the belief widens) and
**update** (multiply by the likelihood of new evidence and renormalise). With von Mises beliefs, both steps stay in
the family: prediction by the wrapped-normal rule, update by the product rule. The result is a one-dimensional
Bayesian filter on the circle per joint ([vmfilter.py](../../mesqsim/estimators/vmfilter.py)).

### 6.4 Correlated residuals overstate confidence

The least-squares κ assumes independent residuals. Real residuals are correlated in time (the fusion error has a 1 s
correlation time; the joint model is wrong in a slowly varying way). N correlated residuals carry less information than
N independent ones, so κ = I/σ̂² is too large. The cure is **recalibration on held-out data**: here one scale factor
and one model-error term per joint type, fitted by maximum likelihood on validation — the circular analogue of
temperature scaling for neural network confidences [Guo 2017]. On AMASS validation the recalibrated model lets one
knee window be worth at most κ ≈ 44 (a spread of about 8.6°), while the raw statistical κ of a well-observed window is
orders of magnitude larger (tens of thousands: 12,540 for one window in PIPELINE_EXPLAINED's session); after
recalibration GEO-ID's 68 %/95 % intervals covered 69.2 %/92.6 % of the validation steps at the knee
(`python -m mesqsim fit-geometry calibration --source amass --seed 1`, commit `f85ec19`).

## 7. Evaluation theory

- **Paired designs.** Every method runs on the same sessions; differences are taken within a session or subject,
  which removes the variation between sessions from the comparison.
- **The unit of analysis.** Thousands of 32 Hz steps of one person are not thousands of independent observations;
  the independent unit is the held-out **subject**. Treating steps as independent would produce absurdly narrow
  intervals (pseudo-replication).
- **The bootstrap.** Resampling subjects with replacement approximates the sampling distribution of a statistic
  without assuming its form [Efron 1993]; percentile intervals are its simplest output.
- **Multiple comparisons.** Testing three contrasts at 5 % each inflates the chance of one false positive; Holm's
  step-down procedure controls the family-wise error and is uniformly more powerful than Bonferroni [Holm 1979].
- **Leakage.** Information from the test data reaching the method or its tuning makes results optimistic [Kaufman
  2012]. Hence: subjects split once; Ω fitted on train, calibration on validation, nothing on test; estimators never
  open labels.
- **Oracles and ceilings.** O2 (knows δ) must score exactly 0 — a test of the scorer, not of a method. GEO-K (knows
  the axes) bounds what geometry can do; the gap to GEO-ID is the cost of identification.
- **Selective prediction.** A confidence is useful if keeping only confident outputs reduces the error; the
  error–coverage ("selective risk") curve shows it [Geifman 2017].
- **Negative results count.** A geometry that does *not* beat box registration at ball joints is a finding (it is
  H2's second half), and is reported with the same care as a success.

## 8. A bridge to Steps 3 and 4

- **Temporal convolutional networks (TCNs)** — causal, dilated one-dimensional convolutions over a time window —
  match or beat recurrent networks on many sequence tasks while being simple and parallel [Bai 2018]. Step 3 will train
  causal TCNs on the same decoded orientations.
- **Von Mises outputs.** A network can output (μ, κ) of a von Mises distribution and be trained with its negative
  log-likelihood, which yields a calibrated uncertainty for angles [Prokudin 2018].
- **Fusion by products.** If geometry and learning each output a von Mises belief, the product rule of §5.4 fuses
  them, each weighted by its own calibrated κ — Step 4's hypothesis H4. Calibration (§6.4) is what makes such weights
  meaningful.
- **Inputs.** Orientations should enter a network in a continuous representation [Zhou 2019].

## 9. How the pieces fit Mesquite

| question | theory | where in the repository |
|---|---|---|
| what can be estimated? | gauge freedom → relative headings only | [MATH.md](../MATH.md) §4; checks 4.2, 28.1 |
| why does the knee help? | hinge constraint; closed-form ML | `estimators/hinge.py`; checks 22.1, 23.1 |
| how is the knee axis found? | Seel's rate condition + T-pose ties | `identify_hinge_axes_tpose`; check 24.1 |
| what about elbows and hips? | 2-DoF constraint; range of motion | `twodof.py`, `rom.py`; checks 25.1, 26.1 |
| how sure is each answer? | von Mises likelihood, Laplace, recalibration | `calibrated_kappa`, `geo_fit.py`; check 30.1 |
| how is it carried in time? | Bayesian filter on the circle | `vmfilter.py`; check 27.1 |
| how is it judged? | paired, subject-level bootstrap, Holm | `results_step2.py`; RESULTS_STEP2 |

## 10. References

Reading list of the Step 2 brief:

- [Seel 2014] T. Seel, J. Raisch, T. Schauer (2014). IMU-based joint angle measurement for gait analysis. *Sensors*
  14(4), 6891–6909. doi:10.3390/s140406891
- [Olsson 2020] F. Olsson, M. Kok, T. Seel, K. Halvorsen (2020). Robust plug-and-play joint axis estimation using
  inertial sensors. *Sensors* 20(12), 3534. doi:10.3390/s20123534
- [Laidig 2019] D. Laidig, D. Lehmann, M.-A. Bégin, T. Seel (2019). Magnetometer-free realtime inertial motion tracking
  by exploitation of kinematic constraints in 2-DoF joints. *EMBC 2019*, 1233–1238. doi:10.1109/embc.2019.8857535
- [Laidig 2022] D. Laidig, I. Weygers, T. Seel (2022). Self-calibrating magnetometer-free inertial motion tracking of
  2-DoF joints. *Sensors* 22(24), 9850. doi:10.3390/s22249850
- [Lehmann 2020] D. Lehmann, D. Laidig, R. Deimel, T. Seel (2020). Magnetometer-free inertial motion tracking of
  arbitrary joints with range of motion constraints. *IFAC-PapersOnLine* 53(2), 16016–16022.
  doi:10.1016/j.ifacol.2020.12.401
- [Kok 2022] M. Kok, K. Eckhoff, I. Weygers, T. Seel (2022). Observability of the relative motion from inertial data in
  kinematic chains. *Control Engineering Practice* 125, 105206. doi:10.1016/j.conengprac.2022.105206
- [McGrath 2020] T. McGrath, L. Stirling (2020). Body-worn IMU human skeletal pose estimation using a factor
  graph-based optimization framework. *Sensors* 20(23), 6887. doi:10.3390/s20236887
- [Kok 2026] M. Kok, I. Weygers, H. Osman, D. Weber, R. Li, T. Seel, A. Seth (2026). Inertial human motion capture:
  from biomechanics to recent sensor fusion methods and back. *Journal of Biomechanics* 206, 113542.
  doi:10.1016/j.jbiomech.2026.113542
- [Cereatti 2024] A. Cereatti et al. (2024). ISB recommendations on the definition, estimation, and reporting of joint
  kinematics in human motion analysis applications using wearable inertial measurement technology. *Journal of
  Biomechanics* 173, 112225. doi:10.1016/j.jbiomech.2024.112225
- [Laidig 2023] D. Laidig, T. Seel (2023). VQF: highly accurate IMU orientation estimation with bias estimation and
  magnetic disturbance rejection. *Information Fusion* 91, 187–204. doi:10.1016/j.inffus.2022.10.014
- [Bachhuber 2025] S. Bachhuber, D. Lehmann, I. Weygers, T. Seel (2025). DIODEM: a diverse inertial and optical
  dataset of kinematic chain motion. *Scientific Data* 12, 1285. doi:10.1038/s41597-025-05468-w
- [Mardia 2000] K. V. Mardia, P. E. Jupp (2000). *Directional Statistics*. Wiley.

Further references (details verified on 2026-11-17):

- [Grood 1983] E. S. Grood, W. J. Suntay (1983). A joint coordinate system for the clinical description of
  three-dimensional motions: application to the knee. *Journal of Biomechanical Engineering* 105(2), 136–144.
  doi:10.1115/1.3138397
- [Wu 2002] G. Wu et al. (2002). ISB recommendation on definitions of joint coordinate system of various joints for
  the reporting of human joint motion — part I: ankle, hip, and spine. *Journal of Biomechanics* 35(4), 543–548.
  doi:10.1016/S0021-9290(01)00222-6
- [Wu 2005] G. Wu, F. C. T. van der Helm, H. E. J. Veeger, M. Makhsous, P. Van Roy, C. Anglin, J. Nagels,
  A. R. Karduna, K. McQuade, X. Wang, F. W. Werner, B. Buchholz (2005). ISB recommendation on definitions of joint
  coordinate systems of various joints for the reporting of human joint motion — part II: shoulder, elbow, wrist and
  hand. *Journal of Biomechanics* 38(5), 981–992. doi:10.1016/j.jbiomech.2004.05.042
- [Kok 2017] M. Kok, J. D. Hol, T. B. Schön (2017). Using inertial sensors for position and orientation estimation.
  *Foundations and Trends in Signal Processing* 11(1–2), 1–153. doi:10.1561/2000000094 (arXiv:1704.06053)
- [Solà 2018] J. Solà, J. Deray, D. Atchuthan (2018). A micro Lie theory for state estimation in robotics.
  arXiv:1812.01537
- [Zhou 2019] Y. Zhou, C. Barnes, J. Lu, J. Yang, H. Li (2019). On the continuity of rotation representations in
  neural networks. *CVPR 2019*. arXiv:1812.07035
- [Efron 1993] B. Efron, R. J. Tibshirani (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- [Holm 1979] S. Holm (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of
  Statistics* 6(2), 65–70. JSTOR 4615733
- [Kaufman 2012] S. Kaufman, S. Rosset, C. Perlich, O. Stitelman (2012). Leakage in data mining: formulation,
  detection, and avoidance. *ACM Transactions on Knowledge Discovery from Data* 6(4), 15.
  doi:10.1145/2382577.2382579
- [Guo 2017] C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger (2017). On calibration of modern neural networks. *ICML 2017*,
  PMLR 70, 1321–1330. arXiv:1706.04599
- [Geifman 2017] Y. Geifman, R. El-Yaniv (2017). Selective classification for deep neural networks. *NeurIPS 2017*.
  arXiv:1705.08500
- [Bai 2018] S. Bai, J. Z. Kolter, V. Koltun (2018). An empirical evaluation of generic convolutional and recurrent
  networks for sequence modeling. arXiv:1803.01271
- [Prokudin 2018] S. Prokudin, P. Gehler, S. Nowozin (2018). Deep directional statistics: pose estimation with
  uncertainty quantification. *ECCV 2018*. arXiv:1805.03430
- [Mahmood 2019] N. Mahmood, N. Ghorbani, N. F. Troje, G. Pons-Moll, M. J. Black (2019). AMASS: archive of motion
  capture as surface shapes. *ICCV 2019*. arXiv:1904.03278

## 11. Self-check questions

1. **Why are q and −q the same rotation, and how does the wire format cope?** A quaternion encodes θ about u as
   (u·sin(θ/2), cos(θ/2)); −q is 2π − θ about −u, the same rotation. The pod forces w ≥ 0.
2. **Why can a six-axis fusion not know its heading?** Rotating about gravity leaves the accelerometer unchanged; only
   heading *changes* are measured, so the reference is chosen at start-up and errors integrate.
3. **Which gyroscope error grows with turning, and which with time?** The scale factor grows with the total turning
   (k·Θ); the bias drift grows linearly in time; the random walk as √t.
4. **What is a gauge freedom here?** The common heading of all pods: turning it changes no measurement, so only
   relative headings are identifiable.
5. **State Seel's hinge condition and its blind spot for identification.** ‖ω_p × u_p‖ = ‖ω_c × u_c‖ at every instant;
   it identifies nothing when the two segments rotate rigidly together.
6. **Why is the hinge likelihood exactly von Mises?** The squared residual is a constant minus 2I·cos(δ − δ̂); with
   Gaussian noise the log-likelihood is (I/σ²)·cos(δ − δ̂).
7. **Why is the product of two von Mises densities von Mises?** κ₁cos(x−μ₁) + κ₂cos(x−μ₂) is a single cosine with
   κe^{iμ} = κ₁e^{iμ₁} + κ₂e^{iμ₂}.
8. **Where does the prediction rule A(κ′) = A(κ)e^{−qΔt/2} come from?** A heading random walk of variance q·Δt is a
   wrapped normal whose resultant length multiplies by e^{−qΔt/2}; the von Mises belief is matched to it.
9. **Why must κ be recalibrated, and on which data?** Correlated residuals overstate information; the scale is fitted
   on validation subjects, never test.
10. **Why is the subject, not the step, the unit of analysis?** Steps and sessions of one person are strongly
    dependent; only subjects are independent draws from the population the claim is about.

---

*Back to the [index](README.md).*
