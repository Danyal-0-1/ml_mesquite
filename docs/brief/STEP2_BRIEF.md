You are continuing a research repository for **Mesquite**, a low-cost inertial motion-capture suit. The project is led
by **Danyal Khorami**, the project owner. Step 1 was built here by an earlier session, from the brief in
`docs/brief/STEP1_BRIEF.md`. A collaborator or Danyal is running this session with you.

You have no access to Danyal's other repositories or files. Everything you need is in this brief and in this
repository.
- Where the brief is silent, make a choice, write it down and test it.
- Stop and ask in two cases:
  - only a person can act, for example registering for a dataset or recording the real suit;
  - a decision would change the research.

**Your task is Step 2, then its explanations, in this order:**

1. Close the parts of Step 1 that Step 2 depends on (milestone S2-M0, §7).
2. Build the **geometric estimator**. For each of the 14 joints it estimates the relative heading δ, with its own
   confidence, from the decoded orientations alone and causally.
3. Evaluate it against Step 1's baselines, with the tests that prove it is right.
4. **After Step 2 is done, build the explanation pipeline (§10):**
   - an explanation of the code;
   - an explanation of the mathematics;
   - an explanation of what happens in the pipeline;
   - a theory file, `THEORY.md`;
   - a mental-picture file, `MENTAL_PICTURE.md`.
5. Write the report (§11), then stop.

Do **not** build any of the following. They belong to Steps 3 and 4:
- the learning features;
- any learned model;
- the fusion of geometry with learning.

## 0. Rules for this repository

1. **First commit.**
   - Save this brief, unchanged, as `docs/brief/STEP2_BRIEF.md`. If you were given it as a file at the repository root,
     keep only the text below its first line of dashes, and do not commit the root copy.
   - Update `CLAUDE.md`: keep every Step 1 rule that still applies, add the rules below, and point to both briefs.
   - Never edit `docs/brief/STEP1_BRIEF.md`.
2. **The Step 1 brief still holds where this one is silent.** That covers:
   - its facts about the suit (§3);
   - its conventions (§4.1);
   - its measurement model (§4.2);
   - its data rules (§5);
   - its environment (§6).

   Before you start, read it, then `CLAUDE.md`, `README.md`, `WORKLOG.md` and every file in `docs/`.
3. **Visibility and disclosure.** The repository is public: Danyal removed the private notice on 2026-11-12. The
   project's relative-heading estimators are a possible invention-disclosure question.
   - Before the first push of Step 2 estimator code, ask once: *"Should the Step 2 estimator code be pushed to the
     public repository, or kept local?"*
   - Follow the answer and record it in `WORKLOG.md`.
   - Until you have an answer, commit locally.
4. **No data in git** (Step 1, rule 3).
   - Never commit datasets, body models, archives, generated datasets, model outputs or credentials.
   - Commit manifests and small synthetic fixtures only.
   - Keep every committed file under 5 MB.
5. **Milestones and pull requests.** S2-M0 creates `main` (§7). From then on, for each milestone:
   1. create a branch from `main` and implement;
   2. run the full check (§8);
   3. update the docs and `WORKLOG.md`;
   4. commit and push (rule 3);
   5. open a pull request into `main` that states what changed and gives the check output.

   Merge only when the person running the session says that Danyal approved. Never force-push `main`. If `gh` is
   missing, give the person the pull-request link to open by hand.
6. **Never invent data, results or numbers.** Every number in the documentation comes either from a command you ran
   (say which) or from a cited source.
   - Every new parameter goes in `configs/default.yaml` with a source note and a label: `datasheet`, `firmware`,
     `assumption` or `measured`, or the new `fitted`.
   - `fitted` means estimated on the training or validation split by a named command, never on test data.
7. **Estimators never read labels.** Only the scorer and the oracles open `labels_*.parquet` or `meta.json`.
   - An estimator that uses simulator truth gets it through one explicit argument, and its name says so. This applies
     to the known-axis variant GEO-K (§5.4).
   - Every table marks such an estimator as using simulator truth.
8. **The estimator has its own beliefs.** Its priors live in a new config section, `geometry:`, labelled like every
   other parameter. It never reads the simulator's `error_model:` section, which is the truth. The sweeps change only
   the simulator.
9. **Causal.** An estimate at time t uses only frames that arrived at or before t. This is tested (§8).
10. **One convention per quantity** (Step 1, rule 7).
    - Import rotations, `Y(α)`, heading, `wrap` and circular statistics from `mesqsim/conventions.py`.
    - Add any new helper there, with a test.
    - Use radians in the code and degrees in reports, with the unit in every variable name.
11. **Results stay provisional until AMASS arrives.** Until AMASS is downloaded and the split is frozen, label every
    simulated result PROVISIONAL (SYNTHETIC MOTION), as Step 1 did.
12. **Talk to the person running the session.** Before each milestone, say in plain words what you will do. After it,
    say what you did and how you checked it.
13. **Step 2 is not finished until the explanation pipeline (§10) exists and the full check passes.** Then write the
    report (§11) and stop.

## 1. The whole project in one picture, and what Step 2 adds

*Keep a version of this at the top of `README.md`, updated for Step 2. Danyal wants every overview to open with a
picture that can be replayed with closed eyes: the purpose, the steps, the pipeline, and what goes in and comes out.*

**The suit.** A person wears fifteen watches. Each fuses a gyroscope and an accelerometer into an orientation. Gravity
gives every watch the same "down", so tilt is right. There is no compass, though:
- each watch invents its own private north when it switches on;
- that north slides a little every minute;
- it slides a little more every time the watch spins around.

Thirty-two times a second each watch radios a 16-byte postcard. Because the norths disagree, a joint computed from two
watches shows a false twist about the vertical.

**What Step 1 built.** A simulator that turns real human motion into the postcards the suit would send, with errors we
added ourselves, so the right answer is known. It also built:
- Step 0, the decoder shared with real recordings;
- four baselines: do nothing (B0), register the pods in the box (B1), and two oracles (O1, O2) that know the answer.

**What Step 2 adds: a geometry solver for every joint.**
- **The knee is a door hinge.** The hinge pin is one physical line, and both the thigh watch and the shin watch see it,
  each in its own private frame. Tilt is shared, so both agree on how steeply the pin points; only their norths differ.
- **The pin's shadow.** Lay the two views of the pin flat on the floor. The angle between the two shadows is exactly how
  far the shin watch's north has slid from the thigh watch's: the relative heading δ.
- **The blind spot.** When the pin stands vertical, its shadow is a dot and says nothing.
- **Elbows, wrists and ankles** have two pins at a fixed angle. That gives one equation per instant instead of a direct
  answer.
- **Hips, shoulders, the lower back and the neck** are ball joints with no pin at all. The only clue is that they cannot
  bend past their limits, so they say something only when the motion reaches a limit.
- **Confidence and memory.** Every joint's answer comes with a confidence. A small filter carries the answers forward:
  - it starts from the box registration;
  - it sharpens when the joint gives information;
  - it slowly loses confidence while the joint gives none;
  - it starts again when a watch reboots.

```text
frames.parquet + session.json ──► STEP 0 (Step 1) ──► series.parquet
   ──► common clock: fresh samples of parent and child at the same instants; rotation rates between fresh samples
   ──► per joint, every 1 s, over the last 8 s:
         hinge   (knees)                               closed form: δ̂ and κ        axes known (GEO-K) or identified (GEO-ID)
         2-DoF   (elbows, wrists, ankles)              one-dimensional constraint cost
         3-DoF   (hips, shoulders, lower back, neck)   range-of-motion cost
   ──► von Mises filter per joint, started from box registration (B1), reset at reboots
   ──► δ̂(t), κ(t), "uncertain" at every 32 Hz step
   ──► scorer (the only reader of labels): errors, coverage, strata ──► docs/RESULTS_STEP2.md
DIODEM: six-axis VQF per IMU ──► the same per-joint estimator ──► against the optical relative heading
```

**What goes in and comes out.**
- **In:** Step 0's decoded series and `session.json`, nothing else. GEO-K also gets the simulator's true mountings.
- **Out:** for every joint and 32 Hz step:
  - the relative heading δ̂ in radians, in (−π, π];
  - a concentration κ ≥ 0, where 0 means "no information";
  - an "uncertain" flag.

  The scorer adds error tables, coverage and figures.
- **Never out:** the body's heading in the room, or north. Only relative headings can be estimated.

## 2. Where Step 1 left things (checked on 2026-11-12)

**Built: milestones M1 to M5** of the Step 1 brief:
- the conventions module and the wire codec;
- the AMASS loader and mapping, tested on synthetic AMASS-format files;
- DIODEM v3.0, downloaded and verified;
- the full simulator;
- Step 0;
- the baselines B0, B1, O1 and O2;
- splits, dataset generation, sweeps and `scripts/make_results.py`.

`docs/VALIDATION.md` (2026-10-23, commit `e644460`) records 30 checks passed, 0 failed and 4 skipped of 34. The four
skipped checks need AMASS. `docs/RESULTS_STEP1.md` is provisional, because it runs on synthetic motion.

**Open:**
1. **AMASS and SMPL+H are not downloaded.** A person must register; the collaborator was going to. As a result:
   - checks 10.2–10.5 skip;
   - there is no split manifest;
   - every result runs on synthetic motion.
2. **The work log is incomplete.** `WORKLOG.md` stops after M4; the M5 commits (`5f494f4`, `a53c510`) have no entry.
3. **M6 has not started.** None of these exist yet:
   - `docs/MUST_KNOW.md` and `docs/analysis_plan_DRAFT.md`;
   - a finished `README.md` quickstart;
   - the Step 1 report;
   - the tag `step1`.
4. **GitHub holds only `m5-results`**, which is the default branch. There is no `main` and no pull request.
5. **Most parameters are still guesses.** Of the 44 labelled parameters in `configs/default.yaml`, 38 are `assumption`.

## 3. The research question, the hypothesis Step 2 tests, and the limits

**Research question** (context). Using only the fused orientations Mesquite already transmits, where does a small causal
temporal network recover relative heading better than explicit joint geometry, at which joints and in which motions?
And does combining the two, each weighted by its calibrated confidence, beat both?

Step 2 builds the geometry and tests **H2**:

| ID | Hypothesis | Step |
|---|---|---|
| H1 | after box registration, the real suit's relative-heading error is dominated by drift and spin-proportional error, not by the initial offset | bench, later |
| **H2** | **geometry recovers relative heading at hinge and 2-DoF joints (knees, elbows, wrists, ankles) during walking and arm motion, but not at 3-DoF joints (hips, shoulders, lower back, neck) in everyday motion** | **2** |
| H3 | a TCN's advantage over geometry is larger at 3-DoF joints than at hinge and 2-DoF joints | 3–4 |
| H4 | fusing geometry and learning is never worse than the better one at any joint type, and better on average | 4 |
| H5 | the predicted confidence is calibrated on held-out simulated subjects (68 % and 95 % intervals cover within ±5 points) and falls where errors grow | 2–6 (Step 2 measures it for geometry) |

**"Better"** means both of these:
- the paired difference in median relative-heading error has a 95 % bootstrap interval over held-out subjects that
  excludes zero;
- the difference exceeds a practical margin of 1°, which is *proposed* and frozen by Danyal in the analysis plan.

**The hard limit.** Rotate the whole room, with the wearer in it, about the vertical, and every watch sends the same
numbers. Only headings relative to one another can be estimated. Never claim to measure north or the body's heading.

## 4. New facts since Step 1

1. **The first measurement of the real suit** (a static hold on 2026-11-04). The older suit, using the JSON path, lay
   still for 19.25 minutes.
   - 12 of 13 untouched pods stayed within 1.8° of their calibrated orientation for 17.3 minutes.
   - That path rounds each quaternion component to 0.01, about 1.1°, so drift below about 0.1°/min is invisible.
   - The Head pod drifted −2.0°/min for 7 minutes, then −0.3°/min.
   - Plugging a pod into USB power, and unplugging it, restarted it each time: 78° wrong for about 16 s, then a new
     north.

   For Step 2 this means:
   - add a drift stress value σ_b = 2.0 °/min to the sweeps. Label it `assumption` and give the measurement as its
     source note: one pod's early drift does not measure the spread;
   - report GEO under the `stress` preset's reboots.
2. **No real recording exists in the frames-table format yet.**
   - The suit still runs the JSON path, which writes text, not 16-byte frames. Do not build a JSON reader unless asked.
   - The binary frame is now defined in `new_device_code/Pod_Watch_Binary/mesq_packet.h` of
     `github.com/Danyal-0-1/mesquite-mocap`. It moved there from `Device code/` on 2026-11-04, and the frame itself is
     unchanged.
3. **DIODEM's joints**, verified in Step 1 (`docs/DATA.md` §2.3):
   - arm: seg1–seg2 spherical, then seg2–seg3, seg3–seg4 and seg4–seg5 hinges;
   - gait: seg5–seg1 hinge, seg1–seg2 and seg2–seg3 saddle (2-DoF), seg3–seg4 hinge.

   Step 1's sanity pair drifted −80.3° in 6.13 minutes in six-axis mode, because the gyroscope offsets reach about
   0.6°/s. So DIODEM tests how fast an estimator tracks.

## 5. The mathematics of Step 2

### 5.1 What is fixed already (import it; never redefine it)

The following come from Step 1 (`mesqsim/conventions.py`, `mesqsim/correction.py`, `mesqsim/evaluate.py`) and are
tested there:
- the quaternion order `(x, y, z, w)`;
- the Hamilton product;
- orientations map body to world;
- the world is +Z up;
- `Y(α)` turns +x towards +y;
- heading is intrinsic Z-Y-X;
- `wrap` maps into (−180°, 180°].

The model and the correction:

```text
what pod i reports:      q_i(t) = Y(−ψ_i(t)) · s_i(t) · m_i · Exp(η_i(t))
target for joint p → c:  δ_pc(t) = wrap(ψ_c(t) − ψ_p(t))
correction:              ĉ_root = 0,  ĉ_c = ĉ_p + δ̂_pc,  q̃_c = Y(ĉ_c)·q_c
B1 (box registration):   δ̂_pc = wrap(h_p − h_c), with h the box heading of each pod from Step 0
```

Check by hand: ψ_p = 12° and ψ_c = 5° give δ = −7°. A sign error leaves a 14° twist (Step 1, check 3.1).

### 5.2 The estimator contract

Every Step 2 estimator has one form. For example, in `mesqsim/estimators/`:

```python
class Estimate(NamedTuple):
    delta_rad: np.ndarray   # (K, 14) in (-pi, pi]
    kappa: np.ndarray       # (K, 14) >= 0; 0 means "no information"
    uncertain: np.ndarray   # (K, 14) bool: kappa below the hold threshold

def estimate(series, session, t_query_host_s, cfg, *, truth=None) -> Estimate: ...
```

- **The query times.** `t_query_host_s` holds the host times of the scorer's 32 Hz steps. They are times, not labels.
- **Causality and determinism.** The estimate at a query time uses only frames with `t_host_s` at or before it, and the
  same inputs always give the same output.
- **`truth`.** Only GEO-K uses it: the true mountings from `meta.json`, passed by the scorer.
- **Registration with the scorer.** Register the estimators beside B0, B1, O1 and O2 in `mesqsim/evaluate.py`.
- **New columns.** `errors.parquet` gains `kappa` and `uncertain`. They are NaN for methods without a confidence.

### 5.3 One clock for parent and child

The geometry compares a parent and a child at the same instant, but every pod has its own clock.

- **Map pod time to host time.** For each pod and boot epoch, map Step 0's `t_pod_s` (decode time on the pod's clock)
  to host time with a causal affine map, t_host ≈ α + β·t_pod.
  - Arrival times lie *above* the decode times by a positive latency, so fit the lower envelope (for example, a low
    quantile of the residuals), not the mean.
  - A latency common to both pods cancels in a pair.
- **Use only fresh samples** (`status == fresh` and `valid`). A held repeat carries no new information.
- **Pair the samples.** At each fresh parent sample, take the child's orientation by slerp between its two bracketing
  fresh samples, if both lie within a maximum gap (`geometry.max_pair_gap_s`, label `assumption`). Otherwise there is
  no pair.
- **Rotation rate** between consecutive fresh samples of one pod, in the body frame:

  ```text
  ω = Log(q(t₋)⁻¹ · q(t)) / (t − t₋)
  ```

  A constant ψ_i leaves it unchanged, because the Y(−ψ_i) cancels; test this.
- **A check in simulation** (scorer side). Compare the paired time difference with the true one from `labels_ticks`
  (`t_sample_s`). Report the P50, P95 and maximum of the differential timing error in ms. It should be of the order of
  the decode delay (0–11 ms). State the bound you test against, and why.

### 5.4 Hinge joints (the knees)

The hinge axis is one physical line h in both segments. Let u_p be the axis in the parent pod's frame and u_c the axis
in the child pod's frame. Then

```text
q_p·u_p = Y(−ψ_p)·h  and  q_c·u_c = Y(−ψ_c)·h,   so   q_p·u_p = Y(δ_pc)·q_c·u_c
```

Let a_t and b_t be the **horizontal parts** (x and y) of q_p(t)·u_p and q_c(t)·u_c over the window's pairs:

```text
model        a_t = Y(δ)·b_t + noise              (Y acting in the horizontal plane)
C = Σ_t a_t·b_t          S = Σ_t (b_t × a_t)      (2-D cross product: b_x·a_y − b_y·a_x)
δ̂ = atan2(S, C)          I = √(C² + S²)
Σ_t ‖a_t − Y(δ)·b_t‖² = Σ_t (‖a_t‖² + ‖b_t‖²) − 2·I·cos(δ − δ̂)
```

- **The likelihood.** With independent Gaussian noise of variance σ² per component, the log-likelihood of δ is
  const + (I/σ²)·cos(δ − δ̂). That is exactly a von Mises density. Its mean is δ̂ and its concentration is
  **κ = I/σ̂²**, where σ̂² is the residual sum of squares divided by (2N − 1).
- **Recalibration.** The residuals are correlated in time, so fit one scale factor for κ per joint type on validation
  (label `fitted`).
- **The blind spot.** When the axis is vertical, a_t = b_t = 0, so I = 0 and κ = 0.
- **No motion needed with known axes.** One instant with a non-vertical axis already determines δ.
- **Worked example; make it a test.** Take a horizontal axis h = x, with ψ_p = 12° and ψ_c = 5°.
  - a = (cos 12°, −sin 12°) and b = (cos 5°, −sin 5°).
  - C = cos 7° = 0.992546 and S = −sin 7° = −0.121869.
  - So δ̂ = −7.000°.
  - Turn h vertical: I = 0.
- **GEO-K: known axes** (simulation only; uses simulator truth).
  - The knee axis comes from the SMPL thigh frame: the body's left–right axis at the rest pose. Document the choice.
  - In pod frames it is u = m⁻¹·u_seg, with m from `meta.json` (`mount_q`, pod → segment).
  - On AMASS, measure how hinge-like the knee really is. Report the RMS of the true knee rotation about the two other
    axes on the training split.
- **GEO-ID: axes identified from motion.**
  - For a hinge the rate components perpendicular to the axis are the same physical vector, so ‖ω_p × u_p‖ = ‖ω_c × u_c‖
    at every instant (Seel, Raisch and Schauer 2014).
  - Fit u_p and u_c in spherical coordinates by least squares (for example `scipy.optimize.least_squares`, several
    starts), using pairs with enough rotation.
  - Report the residual and, in simulation (scorer side), the angle to the true axis.
- **The sign.** The condition cannot tell u from −u, and a wrong sign gives a **180° error**.
  - For the true axis, the vertical components agree: (q_p·u_p)_z = (q_c·u_c)_z, and both pods measure them.
  - So fix the signs so that Σ_t (q_p·u_p)_z·(q_c·u_c)_z > 0.
  - If that sum is too small to trust (the axis stays almost horizontal), report the window as ambiguous. Resolving it
    with the knee's one-way flexion is a *Should*.
  - Write a test with a deliberately flipped axis.

### 5.5 2-DoF joints (elbows, wrists, ankles)

The constraint is that of Laidig et al. (2019, 2022):
- the first axis j₁ is fixed in the parent and the second axis j₂ in the child;
- the two axes keep a constant angle γ, often 90°, so (q_p·j₁)·(Y(δ)·q_c·j₂) = cos γ.

Write a = q_p·j₁ and b = q_c·j₂, with horizontal parts a_h, b_h and vertical parts a_z, b_z:

```text
(a_h·b_h)·cos δ + (b_h × a_h)·sin δ + a_z·b_z = cos γ                       one equation per instant
cost(δ) = Σ_t ((a_h·b_h)·cos δ + (b_h × a_h)·sin δ + a_z·b_z − cos γ)²     search δ over (−180°, 180°], then refine
```

- **Confidence.** κ comes from the Laplace approximation at the minimum: the curvature of the cost divided by twice the
  residual variance.
- **Two minima.** The cost can have two minima. If the second is as good as the first within noise, mark the window
  ambiguous and set κ = 0. Document the rule.
- **The axes.** For GEO-K, define j₁ and j₂ for each 2-DoF joint in the SMPL rest frame, for example the elbow's flexion
  axis in the upper arm and the forearm's long axis in the forearm. On the training split, measure how much each joint
  rotates about its third axis: the constraint is only as good as that number is small.
- **GEO-ID.** Identifying j₁ and j₂ from motion, as in Laidig et al. 2022, is a *Should*. Until then, GEO-ID uses the
  range-of-motion cost (§5.6) at these joints.
- **Check by hand.** With exact data and the true δ, every equation holds, because Y(δ)·q_c·j₂ = Y(−ψ_p)·j₂ in world
  terms and the dot product of the two axes is cos γ.

### 5.6 3-DoF joints (hips, shoulders, lower back, neck): range of motion

This follows Lehmann et al. 2020:

```text
J_pc(δ, t) = (q_p·m_p⁻¹)⁻¹ · Y(δ) · (q_c·m_c⁻¹)          equal to s_p⁻¹·s_c when δ = δ_pc
cost(δ)    = Σ_t dist(angles(J_pc(δ, t)), Ω_pc)²          dist = distance outside the allowed box, 0 inside
```

- **The angles.** angles(·) is the joint's anatomical decomposition: one documented Cardan sequence per joint,
  following the ISB recommendations (Cereatti et al. 2024).
- **The allowed box Ω_pc.** One interval per angle, fitted on the **training subjects' true joint angles only**, for
  example their 0.5th and 99.5th percentiles. The label is `fitted`; name the command.
- **The search.** A one-dimensional search over δ. The cost is flat until the motion reaches a limit, so κ (from the
  Laplace approximation) is often close to 0. That is the expected result, not a bug.
- **Segment frames.**
  - GEO-K uses the true mountings.
  - GEO-ID uses the T-pose snapshot of Step 0, with the current corrections also applied at the T-pose:
    q̃_i(T) = Y(ĉ_i(T))·q_i(T). That is the rule for Mesquite's world-delta chain, Δ_i(t) = q_i(t)·q_i(T)⁻¹.
  - If facing (§5.8) is not ready, GEO-ID holds the box estimate at these joints, and the filter's κ decays.
- **An option** (*Could*). Lehmann et al. model δ inside a window as an offset plus a rate. Report it as an ablation if
  you build it.

### 5.7 Windows, the filter, reboots and the hold rule

- **Windows.** Re-solve every joint every 1 s over the last 8 s, causally. These are `geometry.window_s` and
  `geometry.step_s`, labelled `assumption`; the source is Lehmann et al. 2020, which used 8 s and 1 s. The ablations
  use 4, 8 and 16 s.
- **A von Mises filter per joint.**
  - **Start.** It starts at the end of the box phase from B1's estimate, with κ₀ = 1/σ_pair² and
    σ_pair = √2·`geometry.box_sigma_deg`, in radians. That is the estimator's belief (`assumption`), not the
    simulator's box-slot truth. With no information, GEO stays at B1.
  - **Predict.** Between updates, propagate as A(κ) ← A(κ)·exp(−q·Δt/2), with A(κ) = I₁(κ)/I₀(κ):
    - compute A with `scipy.special.i0e` and `i1e`, and invert it numerically;
    - for large κ this tends to κ ← (1/κ + q·Δt)⁻¹; test it;
    - q is `geometry.drift_q_rad2_per_s`: `assumption` at first, then `fitted` on validation.
  - **Update.** With a window's estimate (δ̂_w, κ_w):

    ```text
    z = κ·e^{iμ} + κ_w·e^{iδ̂_w}       μ ← arg z       κ ← |z|
    ```

    The product of two von Mises densities is exactly von Mises.
  - **Output.** At each query time, the filter state is predicted forward from its last update.
- **Reboots.** When Step 0 starts a new epoch for either pod of a joint, reset that joint's filter to κ = 0. The new
  north is unknown, and box registration no longer applies. In Step 1, B1's error rose to about 30° after a reboot, so
  measure how fast GEO re-acquires.
- **The hold rule.** "uncertain" = κ below `geometry.kappa_hold`, which is `fitted` on validation. While a joint is
  uncertain its estimate still holds; the flag only says so.

### 5.8 Facing from one forward arm raise (*Should*)

- **The simulator.** Add an optional `arm_raise` phase after the T-pose: one arm raised forward about the body's
  left–right axis. Make it configurable and labelled, with its own label rows.
- **The estimator.** The upper-arm pod's rotation axis during the raise is horizontal and points left–right. Facing is
  its heading plus a documented ±90°, in the box-registered frame.
- **The test.** With no noise, recover the simulated facing to within 0.1°. Report the error distribution under the
  defaults.
- **The use.** The T-pose frames that GEO-ID's range-of-motion cost needs.

### 5.9 What cannot be known, as tests (Step 2 additions)

- Rotate every pod's reports by one common Y(β): every GEO output changes by less than 0.5°.
- Rotate one pod's frame by a constant Y(γ): only the joints that touch that pod change, each by ±γ within 0.5°.
- A correction never changes tilt (Step 1, check 4.1).
- Delete the labels from a session folder: GEO-ID's output is byte-identical.

### 5.10 Where most bugs will be

| Bug | Symptom | Guard |
|---|---|---|
| child minus parent swapped, or `Y(δ)` applied to the wrong side | errors double instead of vanishing | the 12°/5° example |
| hinge-axis sign | errors near 180° | the flipped-axis test |
| m against m⁻¹ (pod → segment, not segment → pod) | GEO-K fails on clean data | the zero-noise exact-recovery test |
| reading `error_model:` or labels inside an estimator | results too good under sweeps | rule 8; the label-deletion test |
| host arrival time used as sample time | errors grow with radio jitter | the timing check |
| degrees in a radian function | 57-fold errors | units in names |
| averaging angles linearly near ±180° | means jump by 360° | circular statistics only |
| a reboot not reset | about 30° of stale error after a restart | the reboot test |

## 6. Evaluation

### 6.1 Methods compared

| ID | Method | Uses simulator truth |
|---|---|---|
| B0, B1, O1, O2 | Step 1's baselines, unchanged | O1, O2 |
| GEO-K | the per-joint estimator with known axes and mountings | yes: axes and mountings |
| GEO-ID | the deployable estimator: axes identified, T-pose frames, filter | no |
| ablations | GEO-ID without the filter (windows only); hinge only (other joints at B1); windows of 4, 8 and 16 s | no |

### 6.2 Metrics and strata

- **Primary:** relative-heading error |wrap(δ̂ − δ)| per joint and 32 Hz step: mean, median and P95.
- **Secondary:** Step 1's parent-relative orientation error, with the true mountings.
- **Confidence:**
  - coverage of the 68 % and 95 % von Mises intervals (computed numerically from κ);
  - a reliability diagram;
  - selective risk: the error against the fraction of steps whose κ is above a threshold.
- **Compute:** time per 32 Hz step for all 14 joints, on a named CPU.
- **Strata.** Report every result in each of these groupings:
  - by joint type, and per joint;
  - at 1, 10 and 30 minutes since boot;
  - by excitation tercile: the true hinge information I, and the time near the range-of-motion limits, computed by the
    scorer from the truth;
  - by condition (the Step 1 presets and sweeps): clean, drift, low motion, missing bursts, resets.

### 6.3 Statistics

- **The unit.**
  - With AMASS, the unit is the held-out subject. Frames and steps are never independent observations.
  - On synthetic motion, the unit is the session, and the results are descriptive only.
- **Comparisons.** Methods are compared pairwise on the same sessions. Intervals are 95 % bootstrap intervals at subject
  level; state the number of resamples.
- **Multiple contrasts.** Apply Holm's correction across the analysis plan's primary contrasts.
- **The plan.** If the plan is not frozen when you run the comparison, report descriptively and say so.

### 6.4 DIODEM

- **The data.** Use the rigid IMUs, with six-axis VQF per IMU, as in Step 1's sanity analysis.
- **The truth.** The relative heading from the optical segment orientations, built on Step 1's
  `six_axis_relative_heading` in `mesqsim/data/diodem.py`:
  - per IMU, ψ_i comes from the heading of q_i^VQF·(q_i^opt)⁻¹, once both are in the +Z-up world. Document its sign
    and test it on a synthetic rotation;
  - then δ = wrap(ψ_c − ψ_p).
- **The estimators per joint type:**
  - hinges: GEO-ID's hinge estimator, with identified axes;
  - saddle joints: the 2-DoF constraint, if identified; otherwise report them as not covered;
  - the spherical joint: range of motion, or not covered.
- **The pipeline.**
  - *Must:* run on the 40 Hz series directly.
  - *Should:* run through Step 1's 32 Hz send grid and Step 0.
- **Reporting.** Report per joint and per experiment. The experiment is the unit, and the results are out of
  distribution: no claim about people rests on them.

## 7. Milestones (each ends with a pull request and a WORKLOG entry)

| # | Milestone | Done when |
|---|---|---|
| S2-M0 | **Close Step 1.** (1) Write the missing M5 entry in `WORKLOG.md` from the M5 commits and their check output. (2) M6: `docs/MUST_KNOW.md` as the Step 1 brief §10 describes it, `docs/analysis_plan_DRAFT.md` (§3's hypotheses, metrics, units, splits, contrasts, thresholds and margins, marked DRAFT for Danyal to freeze), the README quickstart, and the Step 1 report (Step 1 brief §11). (3) With the person's approval, create `main` on GitHub at the accepted Step 1 commit and tag it `step1`. (4) Ask whether AMASS and SMPL+H are downloaded. If yes: run checks 10.2–10.5, `python -m mesqsim split --write` (record the SHA-256), and regenerate `docs/RESULTS_STEP1.md` on AMASS. If no: say what is blocked, and continue on synthetic motion | the full check passes; the WORKLOG has M5 and M6; Danyal has the analysis-plan draft, and the split manifest if AMASS exists, to freeze |
| S2-M1 | The estimator contract and its integration with the scorer (§5.2); the common clock and pairing; rotation rates (§5.3); the `geometry:` config section | the causality, label-deletion and rate-invariance tests pass; the timing check is reported |
| S2-M2 | Hinge estimator: GEO-K, then axis identification and GEO-ID (§5.4) | the closed-form, likelihood, blind-spot and flipped-axis tests pass; axes recovered within 1° on noise-free synthetic hinge motion |
| S2-M3 | The 2-DoF constraint with known axes (identification *Should*); the range-of-motion cost, with Ω fitted on train (§5.5, §5.6) | exact recovery on noise-free data; κ ≈ 0 on motion that never reaches a limit; the ambiguity rule tested |
| S2-M4 | Windows, the filter, reboot resets, the hold rule and κ recalibration (§5.7); facing (*Should*, §5.8) | the filter tests pass; GEO-ID never does worse than B1 on noise-free motion with no information |
| S2-M5 | DIODEM (§6.4) | the DIODEM checks pass; a figure of GEO-ID against the optical truth for one hinge |
| S2-M6 | The Step 2 results: one command, for example `python scripts/make_results_step2.py`, writes `docs/RESULTS_STEP2.md` and its figures: every method, stratum and sweep, coverage and selective risk, compute, and H2 under the frozen plan, or descriptively | every table is reproducible from one command; the provisional banner is present while the motion is synthetic |
| S2-M7 | **The explanation pipeline (§10)**, once S2-M0 to S2-M6 are merged | the six files exist; every link and `path:line` reference resolves; every worked example is reproduced by a test; the full check passes |
| S2-M8 | Documentation complete (§9); the report (§11); tag `step2` | Danyal can follow `README.md` from a fresh clone to the Step 2 results and the explanations |

## 8. Accuracy: checks that must pass

Add them to `mesqsim/validation.py`, so they appear in `docs/VALIDATION.md` and run in `scripts/check_all.sh`. The
Step 1 checks must keep passing.

1. **Contract.**
   - *Causality:* deleting every frame after t leaves every estimate at or before t unchanged.
   - *Determinism:* identical bytes for the same inputs.
   - *Label blindness:* GEO-ID's output is identical with the labels deleted.
2. **Hinge, exact.** With zero noise, and over random δ, axes and poses, δ̂ = δ to 1e-9 rad. The 12°/5° example gives
   −7.000°. A vertical axis gives I < 1e-12 and κ = 0.
3. **Hinge likelihood.** Over 2,000 noisy Monte Carlo windows, the circular spread of δ̂ matches 1/√κ within 10 %.
   (Report it; this verifies the formula, not the real noise.)
4. **Axis identification.**
   - On noise-free synthetic hinge motion, both axes are recovered within 1°.
   - A deliberately flipped axis is corrected, or the window is flagged as ambiguous.
   - The result is reported on AMASS knees once AMASS is available.
5. **2-DoF, exact.** With zero noise and known axes, δ̂ = δ to 1e-6 rad. A built case with two equal minima is flagged
   as ambiguous.
6. **Range of motion.**
   - When the motion crosses a limit, δ̂ comes within 1° of δ on noise-free data.
   - When it never reaches a limit, κ is below a stated small value.
7. **Filter.**
   - Without updates, μ stays at its start and κ follows A(κ)·exp(−qΔt/2) exactly.
   - For large κ this matches (1/κ + qΔt)⁻¹ to 1 %.
   - The product update equals the closed form.
   - A reboot resets κ to 0.
8. **Invariances** (§5.9): the common yaw, the single-pod rotation and tilt.
9. **Ordering with drift on, fusion noise off and known axes:** O2 = 0 ≤ GEO-K ≤ B1 at every hinge.
10. **Coverage** on validation, after recalibration. Report the 68 % and 95 % coverage per joint type. The test is that
    the numbers exist and lie between 0 and 1; H5's ±5-point target is a result, not a check.
11. **Timing.** The differential timing error of the pairs is within your stated bound (§5.3).
12. **DIODEM.** The files match the manifest, the truth's sign test passes, and the one-hinge figure is regenerated.
13. **Reproducibility.** The same config and seed give an identical `RESULTS_STEP2.md`, apart from its timestamp line.

## 9. Documentation for Danyal (update the Step 1 documents)

Write for a reader who knows the project but not your code. Define every symbol, unit and file format, and put the
picture first.

- `README.md`: the picture (§1), updated; Step 2's status; a quickstart from a fresh clone to the Step 2 results; a link
  to `docs/explain/README.md` as the place to start learning.
- `docs/PIPELINE.md`: the estimator stage, with its schemas and commands.
- `docs/MATH.md`: §5's mathematics, with the choices you made: joint axes, Cardan sequences, Ω, the clock fit, the
  ambiguity rules.
- `docs/CODE.md`: the new modules, and how they connect.
- `docs/SIMULATOR.md`: the `arm_raise` phase (if built) and the new sweep value.
- `docs/VALIDATION.md` and `docs/RESULTS_STEP2.md`: generated, never edited by hand.
- `docs/MUST_KNOW.md`: a Step 2 section. It covers:
  - what GEO-K may and may not be used for;
  - the blind spot and the sign ambiguity;
  - why κ is near 0 at ball joints;
  - the provisional status;
  - how to tell that the estimator is broken;
  - ten new self-check questions with answers.
- `WORKLOG.md`: one entry per milestone, as in Step 1. Each entry records:
  - the date and branch;
  - the files changed;
  - the commands and their results;
  - the decisions and why;
  - the open questions;
  - the previous entry's commit hashes.

## 10. After Step 2 is done: the explanation pipeline

When S2-M0 to S2-M6 are merged and the full check passes, stop building and explain. Danyal will learn the project from
these files, so they matter as much as the code.

Write them in `docs/explain/`. The reader knows the project's goal but not your code. Write in the style of a CS50
lecture:
- start from a concrete picture;
- define every term at its first use;
- show every step of every derivation;
- work small numerical examples by hand;
- explain the code line by line where it matters;
- end with self-check questions and their answers.

**The explanation pipeline** is the order in which the files are read. Each answers one question:

```text
MENTAL_PICTURE.md ─► PIPELINE_EXPLAINED.md ─► MATH_EXPLAINED.md ─► CODE_EXPLAINED.md ─► THEORY.md
 "what happens?"     "what happens to the     "why does it work?"   "how is it built?"     "where do the ideas
                      data, stage by stage?"                                                come from?"
```

1. **`docs/explain/README.md`, the entry point.** It holds:
   - the pipeline above;
   - what each file answers, and roughly how long it takes to read;
   - which file to open for a given question (for example, "why is κ zero at the hip?" → MATH, §range of motion);
   - a link to each file. Each file links back here and on to the next one.
2. **`docs/explain/MENTAL_PICTURE.md`, the close-your-eyes picture** of Steps 1 and 2. Words, one diagram and one data
   in/out table; no equations. It covers:
   - **the suit:** the watches and their private norths, the postcards, the hub, the avatar and its false twist;
   - **the room:** turn the room and nothing changes, so north can never be known;
   - **the simulator:** a flight simulator for the suit, real human motion with errors we chose;
   - **Step 0:** a post office that sorts postcards into fresh, held and lost;
   - **the baselines:** do nothing, the box, and the two oracles;
   - **the geometry:** the knee's hinge pin and its shadow on the floor; the vertical-pin blind spot; two pins at an
     elbow; the walls of a ball joint's range;
   - **confidence and memory:** confidence as the sharpness of the shadow; the filter as a memory that fades and starts
     again at a reboot;
   - **what goes in, what comes out, and what never can.**
3. **`docs/explain/PIPELINE_EXPLAINED.md`, what happens in the pipeline.** Follow one session from an AMASS file (or a
   synthetic clip) to `docs/RESULTS_STEP2.md`, and the DIODEM path as well.
   - **The stages:**
     1. the motion source;
     2. the session plan;
     3. truth, timing and the radio;
     4. frames and labels;
     5. Step 0;
     6. the common clock and the pairs;
     7. the window solvers per joint type;
     8. the filter;
     9. the scorer, the strata and the statistics;
     10. the results.
   - **For every stage:**
     - the command;
     - the input files, with their schemas and units;
     - the `module:function` that runs;
     - what it computes, in words and in one formula;
     - the output files;
     - a worked example with the real numbers of one session you ran (name the command, seed and commit);
     - what can go wrong, and which check catches it.
4. **`docs/explain/MATH_EXPLAINED.md`, the mathematics**, derived step by step. Every implemented formula names the
   function that implements it and the test that checks it, and its worked example is reproduced by a test
   (`tests/test_explain_examples.py`). The topics:
   - **rotations:** matrices, quaternions (x, y, z, w), the Hamilton product, why q and −q are the same rotation,
     composition order, Exp and Log, slerp, the geodesic angle; Y(α), heading, wrap, gravity and tilt;
   - **the measurement model**, term by term, with Step 1's values and labels; why only relative heading is observable,
     with the proof by turning the room;
   - **the target and the outward correction:** the 12°/5° example, and the world-delta equation that explains why
     corrections are also applied at the T-pose;
   - **Step 0's arithmetic:** unwrapping, held and lost frames, knowable ticks, box registration, the T-pose snapshot;
   - **the common clock:** the lower-envelope fit, rates between fresh samples, and the proof that they ignore a
     constant heading offset;
   - **the hinge:** the closed form derived from the least-squares cost; its exact von Mises likelihood and κ = I/σ̂²;
     the blind spot; axis identification and why its condition holds; the sign rule;
   - **the 2-DoF constraint:** its derivation, why it can have two minima, the Laplace approximation;
   - **the range-of-motion cost:** the Cardan sequences used, and why the cost is flat inside the limits;
   - **the von Mises distribution:** its density, mean, A(κ) and intervals; the product of two densities; the
     prediction rule and its large-κ limit;
   - **metrics and statistics:** |wrap|, the geodesic angle, coverage, selective risk, the paired subject-level
     bootstrap, Holm's correction;
   - self-check questions with answers.
5. **`docs/explain/CODE_EXPLAINED.md`, the code.**
   - **A map of every module:** what it owns and who calls it, as an updated diagram.
   - **One command traced end to end:** the command that scores GEO on a run, function by function, in call order, with
     `path:line` links.
   - **The Step 2 modules line by line**, CS50 style: the estimator contract, the clock, the hinge, the axis
     identification, the 2-DoF solver, the range-of-motion solver, the filter and the scorer's changes. For each block:
     - what it does, and why;
     - what would break if it were wrong;
     - which test guards it.
   - **The Step 1 modules function by function**, where Step 2 depends on them: `conventions`, `correction`, `step0`,
     `evaluate`, `io`, `config`.
   - **The tests:** what each test file proves.
   - **How to extend the code:** add a joint model, an estimator, a check or a dataset.
   - **How to debug:** sign flips, frame order, units, causality leaks, label leaks.
   - Self-check questions with answers.
6. **`docs/explain/THEORY.md`, the theory** behind the methods, from first principles to the literature. Cite the full
   reference (authors, year, title, venue, DOI) for every claim:
   - **rotations and their representations:** SO(3), quaternions and the double cover, the exponential and logarithm
     maps, and why continuous representations matter for learning later;
   - **inertial sensing and six-axis fusion:**
     - integrating the gyroscope;
     - gravity for tilt;
     - why heading is unobservable without a magnetometer;
     - bias, scale-factor error and angle random walk, with the datasheet numbers of the Step 1 brief;
     - what the closed on-chip fusion does and does not tell us;
   - **gauge and identifiability:**
     - the room-rotation symmetry;
     - relative against absolute heading;
     - observability results for kinematic chains;
     - the vertical-axis blind spot;
   - **kinematic constraints:** hinge joints and axis identification, 2-DoF joints, range of motion, the ISB joint
     conventions;
   - **directional statistics:** the von Mises and wrapped normal distributions, the circular mean and variance, Bessel
     functions and A(κ), intervals, products;
   - **estimation:**
     - least squares and maximum likelihood;
     - the Laplace approximation;
     - recursive Bayesian filtering on the circle;
     - why treating correlated residuals as independent overstates confidence, and how recalibration fixes it;
   - **evaluation theory:** paired designs, the unit of analysis, the bootstrap, multiple comparisons, leakage, oracles
     and ceilings, and why negative results count;
   - **a bridge to Steps 3 and 4:** temporal convolutional networks, von Mises losses, and fusing two estimators by the
     product of their densities;
   - the reading list in the appendix of this brief, and self-check questions with answers.

**Rules for all six files:**
- Every number comes from a command you ran (name it, with the commit) or from a cited source.
- Every code reference is `path:line` or a line range, and points at the current code. A test,
  `tests/test_explain_links.py`, checks that every link and line reference resolves.
- Steps 1 and 2 are explained together: Step 2 in full depth, and Step 1 as far as Step 2 depends on it.
- Nothing in them contradicts `docs/MATH.md` or the code. Where they disagree, fix the one that is wrong and say so in
  `WORKLOG.md`.

## 11. When you stop: the report

Post the report to the person running the session, and add it as the last `WORKLOG.md` entry:

1. What was built, as one list of files.
2. The output of `scripts/check_all.sh`.
3. The Step 2 results tables, with their parameter set and the provisional banner if the motion is synthetic.
4. H2: what the data say, by joint type, and how firmly (frozen plan or descriptive).
5. Every assumption still in the config, including the `geometry:` section, and every `fitted` value with its command.
6. What is blocked, and on whom: AMASS, the plan freeze, the bench measurements.
7. What Danyal must decide:
   - freezing the analysis plan and the split;
   - pushing the estimator code (rule 3);
   - switching to the DIODEM-only project if AMASS is still missing;
   - the Spine joint (spine3 or spine2).
8. Where the explanations are: `docs/explain/` and its six files.
9. A proposed plan for Step 3, the learned estimators: the TCN-J, TCN-B, MLP-B and GRU-B baselines with von Mises
   outputs. Do not start it.

## Appendix: reading list (cite these in THEORY.md and MATH_EXPLAINED.md)

- T. Seel, J. Raisch, T. Schauer (2014). IMU-based joint angle measurement for gait analysis. *Sensors* 14(4),
  6891–6909. doi:10.3390/s140406891 *(hinge-axis identification from angular rates)*
- F. Olsson, M. Kok, T. Seel, K. Halvorsen (2020). Robust plug-and-play joint axis estimation using inertial sensors.
  *Sensors* 20(12), 3534. doi:10.3390/s20123534 *(axis estimation with a success indicator; it needs accelerations,
  which Mesquite does not send)*
- D. Laidig, D. Lehmann, M.-A. Bégin, T. Seel (2019). Magnetometer-free realtime inertial motion tracking by
  exploitation of kinematic constraints in 2-DoF joints. *EMBC 2019*, 1233–1238. doi:10.1109/embc.2019.8857535
- D. Laidig, I. Weygers, T. Seel (2022). Self-calibrating magnetometer-free inertial motion tracking of 2-DoF joints.
  *Sensors* 22(24), 9850. doi:10.3390/s22249850
- D. Lehmann, D. Laidig, R. Deimel, T. Seel (2020). Magnetometer-free inertial motion tracking of arbitrary joints with
  range of motion constraints. *IFAC-PapersOnLine* 53(2), 16016–16022. doi:10.1016/j.ifacol.2020.12.401 *(8 s windows
  re-solved every 1 s; δ as an offset plus a drift)*
- M. Kok, K. Eckhoff, I. Weygers, T. Seel (2022). Observability of the relative motion from inertial data in kinematic
  chains. *Control Engineering Practice* 125, 105206. doi:10.1016/j.conengprac.2022.105206
- T. McGrath, L. Stirling (2020). Body-worn IMU human skeletal pose estimation using a factor graph-based optimization
  framework. *Sensors* 20(23), 6887. doi:10.3390/s20236887
- M. Kok, I. Weygers, H. Osman, D. Weber, R. Li, T. Seel, A. Seth (2026). Inertial human motion capture: from
  biomechanics to recent sensor fusion methods and back. *Journal of Biomechanics* 206, 113542.
  doi:10.1016/j.jbiomech.2026.113542 *(§4.2: 1-D and 2-D joint methods lose relative heading while the axis is
  vertical)*
- A. Cereatti et al. (2024). ISB recommendations on the definition, estimation, and reporting of joint kinematics in
  human motion analysis applications using wearable inertial measurement technology. *Journal of Biomechanics* 173,
  112225. doi:10.1016/j.jbiomech.2024.112225
- D. Laidig, T. Seel (2023). VQF: highly accurate IMU orientation estimation with bias estimation and magnetic
  disturbance rejection. *Information Fusion* 91, 187–204. doi:10.1016/j.inffus.2022.10.014
- S. Bachhuber, D. Lehmann, I. Weygers, T. Seel (2025). DIODEM: a diverse inertial and optical dataset of kinematic chain
  motion. *Scientific Data* 12, 1285. doi:10.1038/s41597-025-05468-w
- K. V. Mardia, P. E. Jupp (2000). *Directional Statistics*. Wiley. *(the von Mises distribution, A(κ), circular
  statistics)*
