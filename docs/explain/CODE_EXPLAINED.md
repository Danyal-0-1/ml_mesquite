# The code: how it is built

*Explanation pipeline, file 4 of 5 · about 60 minutes · [index](README.md) · previous:
[MATH_EXPLAINED.md](MATH_EXPLAINED.md) · next: [THEORY.md](THEORY.md)*

This file walks through the code the way a CS50 lecture walks through a program: first the map, then one command
followed call by call, then the Step 2 modules block by block — for each block what it does and why, what would break
if it were wrong, and which test guards it. Then the Step 1 modules that Step 2 stands on, what every test file
proves, how to extend the code and how to debug it.

Every code reference is a link to the current line (`tests/test_explain_links.py` fails if a line moves away from
the symbol it names). [CODE.md](../CODE.md) is the compact reference table of modules and functions; this file is
the guided tour.

---

## 1. A map of every module

```text
                              configs/default.yaml ── config.py (every parameter: value, label, source)
                                        │
 AMASS .npz ── data/amass.py ──┐        ▼
 synthetic ─── synth.py ───────┴─► sim/motion.py ─► sim/session.py ─► sim/simulate.py ─┬─ sim/heading.py  ψ_i(t)
                                    (S_i(t))        (the plan)        (one session)     ├─ sim/truth.py    P_i(t)
                                                                                        ├─ sim/timing.py   samples, sends
                                                                                        └─ sim/radio.py    loss, arrival
                                                                                               │
                         pipeline.py / dataset.py / geo_sessions.py ◄──────────────────────────┘
                                        │            (io.py writes and reads session folders)
                                        ▼
                    frames.parquet + session.json ─► step0.py ─► series ──┬─► evaluate.py: B0, B1 (box), O1, O2
                                                  (codec.py, recording.py)│
                                                                          └─► estimators/geo.py: GEO-K, GEO-ID
                                                                                 ├─ clock.py     one time line, pairs
                                                                                 ├─ hinge.py     knees
                                                                                 ├─ twodof.py    2-DoF constraint
                                                                                 ├─ rom.py       range of motion
                                                                                 ├─ anatomy.py   axes, Cardan orders
                                                                                 └─ vmfilter.py  memory and confidence
                                        │
   labels_*.parquet, meta.json ─────────┴─► evaluate.py: Scorer ─► results/ ─► results.py (Step 1),
     (scorer only)                                                            results_step2.py + scripts/make_results_step2.py
                                                                              geo_fit.py (fits on train / validation)
```

| module | owns | called by |
|---|---|---|
| `conventions.py` | quaternions (x, y, z, w), Y(α), heading, wrap, Exp/Log, slerp, Cardan angles, von Mises helpers | everything |
| `body.py` | the 15 pods, the 14-joint tree (parent, child, type) | everything |
| `config.py` | loading, overriding, hashing and editing `configs/default.yaml` | `cli.py`, every runner |
| `codec.py`, `recording.py` | the 16-byte wire frame; what a recording is and which missing frames are *lost* | simulator, Step 0 |
| `sim/*` | the simulated session: motion, plan, ψ, truth, timing, radio | `pipeline.py`, `dataset.py`, `geo_sessions.py` |
| `io.py` | session folders on disk (parquet, JSON) | `pipeline.py`, `dataset.py` |
| `step0.py` | decoding frames into the per-pod series; box registration; T-pose snapshot | `pipeline.py`, `evaluate.py`, estimators |
| `correction.py` | targets δ and the outward heading correction | `evaluate.py`, `geo.py` (facing) |
| `estimators/*` | the Step 2 estimators (contract, clock, solvers, filter, GEO) | `evaluate.py`, `results_step2.py`, `geo_fit.py`, `diodem_geo.py` |
| `evaluate.py` | the baselines, the method registry, the scorer | `pipeline.py`, `results_step2.py` |
| `pipeline.py`, `dataset.py`, `runs.py`, `paths.py` | run folders: simulate, decode, score | `cli.py`, `scripts/` |
| `geo_sessions.py` | sessions in memory, in worker processes (fits and Step 2 results) | `geo_fit.py`, `results_step2.py` |
| `geo_fit.py` | fitting Ω (train) and the calibration (validation) | `cli.py fit-geometry` |
| `results.py`, `results_step2.py` | the tables and statistics of the two results documents | `scripts/make_results*.py` |
| `geo_score.py`, `diodem_geo.py` | scorer-side Step 2 checks (pair timing); GEO-ID on DIODEM | `validation.py`, `make_results_step2.py` |
| `validation.py` | every numbered check, registered once, reported and tested | `scripts/validate.py`, `tests/test_validation.py` |
| `data/amass.py`, `data/diodem.py`, `splits.py`, `manifest.py` | the datasets, their manifests and the subject split | `cli.py`, `dataset.py`, checks |

The one rule the map encodes: **estimators sit left of the labels.** Everything an estimator imports or reads comes
from the frames and `session.json`; `labels_*.parquet` and `meta.json` enter only at the scorer.

---

## 2. One command traced end to end

```bash
python -m mesqsim baselines --run <run folder> --methods GEO-ID
```

The call order, with each step's line:

1. `python -m mesqsim` runs `mesqsim/__main__.py:5`, which calls [`main`](../../mesqsim/cli.py#L268): it builds the
   parser and dispatches `baselines` to [`cmd_baselines`](../../mesqsim/cli.py#L46). That checks the method names
   against `METHODS` and calls [`score_run`](../../mesqsim/pipeline.py#L90).
2. `score_run` sees a GEO method and loads the estimator's beliefs with
   [`load_config`](../../mesqsim/pipeline.py#L103) — the repository's `configs/default.yaml`, never the run's own
   simulator config. For every session folder it calls [`read_session`](../../mesqsim/io.py#L94), then
   [`step0.decode`](../../mesqsim/pipeline.py#L107) (Step 0, [`decode`](../../mesqsim/step0.py#L144)) and writes
   `series.parquet`, then [`ev.evaluate`](../../mesqsim/pipeline.py#L109).
3. [`evaluate`](../../mesqsim/evaluate.py#L187) builds the [`Scorer`](../../mesqsim/evaluate.py#L163) (the only object
   that holds the truth) and, per method, [`method_estimate`](../../mesqsim/evaluate.py#L196). For GEO-ID that asks
   [`geo_estimator`](../../mesqsim/evaluate.py#L124) for a `GeoEstimator(cfg)` — no truth argument — and queries it at
   the host times of the 32 Hz grid.
4. [`GeoEstimator`](../../mesqsim/estimators/geo.py#L188)'s [`run`](../../mesqsim/estimators/geo.py#L208) does the
   work:
   - [`pod_samples`](../../mesqsim/estimators/geo.py#L216) keeps each pod's fresh, valid samples;
     [`epoch_starts`](../../mesqsim/estimators/geo.py#L217) finds when the host learnt of each boot epoch;
   - a [`JointFilter`](../../mesqsim/estimators/geo.py#L222) for the 14 joints;
   - the event loop over [`self._events`](../../mesqsim/estimators/geo.py#L228): one *init* at the end of the box
     ([`_box_init`](../../mesqsim/estimators/geo.py#L344)), the *T-pose snapshot* at the end of the T-pose
     ([`self._tpose_snapshot`](../../mesqsim/estimators/geo.py#L237): q_i(T) and the box belief, kept for the knee
     tie and the T-pose frames), a *reset* per reboot, a *window* every second;
   - each window: [`self._window`](../../mesqsim/estimators/geo.py#L239) returns (δ̂_w, raw κ) for every joint and
     [`calibrated_kappa`](../../mesqsim/estimators/geo.py#L240) turns the raw κ into the filter's weight.
5. Inside [`_window`](../../mesqsim/estimators/geo.py#L356): [`self._window_context`](../../mesqsim/estimators/geo.py#L359)
   maps every pod's clock and selects the last 8 s (and takes the T-pose snapshot only as a fallback,
   [`self._tpose_snapshot`](../../mesqsim/estimators/geo.py#L361)); then per joint
   [`pair_samples`](../../mesqsim/estimators/geo.py#L370) and [`self._solve`](../../mesqsim/estimators/geo.py#L372).
6. [`_solve`](../../mesqsim/estimators/geo.py#L427) dispatches by joint type and mode. GEO-ID's knee goes to
   [`_hinge_id`](../../mesqsim/estimators/geo.py#L583) → [`_identify`](../../mesqsim/estimators/geo.py#L504) → the
   closed form → the sign votes → [`_vote_knee_direction`](../../mesqsim/estimators/geo.py#L607) →
   [`_update_facing`](../../mesqsim/estimators/geo.py#L622); every other GEO-ID joint goes to
   [`_rom`](../../mesqsim/estimators/geo.py#L464) in the T-pose frames that `_update_facing` produced.
7. After the loop, the recorded events are replayed ([`replay`](../../mesqsim/estimators/geo.py#L243)) to answer at the
   query times, and the hold flag is set from [`kappa_hold`](../../mesqsim/estimators/geo.py#L250). The result is an
   `Estimate(delta_rad, kappa, uncertain)`.
8. Back in the scorer, [`errors`](../../mesqsim/evaluate.py#L179) computes both metrics; `score_run` writes
   `results/errors.parquet`, `summary.csv` and `summary.md`, and `cmd_baselines` prints the "all joints" rows.

[PIPELINE_EXPLAINED.md](PIPELINE_EXPLAINED.md) shows this same command on a real session, with its numbers.

---

## 3. The Step 2 modules, block by block

### 3.1 The contract (contract.py)

```python
class Estimate(NamedTuple):
    delta_rad: np.ndarray   # (K, 14) relative heading in (-pi, pi]
    kappa: np.ndarray       # (K, 14) von Mises concentration >= 0; 0 means "no information"
    uncertain: np.ndarray   # (K, 14) bool: kappa below the hold threshold (the estimate still holds)
```

- **What and why.** [`Estimate`](../../mesqsim/estimators/contract.py#L31) is the one shape every estimator
  returns, so the scorer, the results and Step 3's learned models can treat all methods alike. κ = 0 is a
  legitimate answer ("I know nothing"); `uncertain` never hides the estimate. [`GeoTruth`](../../mesqsim/estimators/contract.py#L40)
  is the *only* door through which simulator truth (the true mountings) can enter, and only GEO-K opens it;
  [`from_meta`](../../mesqsim/estimators/contract.py#L57) is called by the scorer, never by an estimator.
- **What would break.** An estimator that read `meta.json` itself would look excellent and be useless on the real
  suit; one that read future frames would be impossible online.
- **Tests.** `test_causal_deleting_later_frames_changes_nothing_before`, `test_deterministic_identical_bytes`,
  `test_label_blind_identical_without_labels`, `test_the_estimator_sees_only_its_own_beliefs`,
  `test_geo_k_needs_truth_and_geo_id_refuses_it` (checks 21.1–21.3).

### 3.2 The clock (clock.py)

- **[`pod_samples`](../../mesqsim/estimators/clock.py#L48)** keeps rows with `status == "fresh"` and `valid`. A held
  frame repeats an old sample; using it would put an old orientation at a new time.
- **[`fit_host_map`](../../mesqsim/estimators/clock.py#L80)** finds α, β of t_host ≈ α + β·t_pod. Line by line: the
  residual r = t_host − t_pod is the pod-to-host latency plus the clock offset; the minimum of r in each 2 s block is
  the frame that waited least (the lower envelope); with fewer than
  [`MIN_SLOPE_BLOCKS`](../../mesqsim/estimators/clock.py#L97) minima only an offset is fitted; otherwise a
  least-squares line through the minima, its slope clipped by [`np.clip`](../../mesqsim/estimators/clock.py#L101) to
  ±200 ppm (the tolerance Step 0 assumes). *What would break:* a slope through two noisy minima after an idle gap once
  gave +1.5 % and 41 ms timing errors (WORKLOG, S2-M1). *Tests:*
  `test_lower_envelope_maps_decode_time_to_host_time_within_two_ms`, `test_short_data_gives_an_offset_only_map`,
  `test_a_latency_common_to_two_pods_cancels_in_the_pair`.
- **[`map_epoch`](../../mesqsim/estimators/clock.py#L106)** is the causal wrapper: only frames that *arrived* by the
  analysis time, only one boot epoch (a reboot restarts the pod clock), the map fitted on the last 30 s.
- **[`pair_samples`](../../mesqsim/estimators/clock.py#L140)**: for every parent sample, the child's two bracketing
  samples; if they are at most 0.1 s apart, the child is interpolated with
  [`quat_slerp`](../../mesqsim/estimators/clock.py#L155). *What would break:* pairing across a burst of lost frames
  would invent a straight-line motion. *Test:* `test_pairs_need_two_bracketing_samples_within_the_gap`.
- **[`body_rates`](../../mesqsim/estimators/clock.py#L159)**: ω = Log(q₋⁻¹·q)/Δt, in the pod's own frame. The
  private north Y(−ψ) sits on the *left* of every report and cancels in q₋⁻¹·q — the reason axis identification never
  needs to know δ. *Test:* `test_rates_ignore_a_constant_heading_offset` (and E13 in MATH_EXPLAINED).

### 3.3 The solvers (hinge.py, twodof.py, rom.py, anatomy.py)

**The anatomy tables.** [`HINGE_AXES`](../../mesqsim/estimators/anatomy.py#L43) (the knee flexes about the segment x
axis in both thigh and shin), [`TWODOF_AXES`](../../mesqsim/estimators/anatomy.py#L49) (elbow, wrist, ankle: one axis
fixed in each segment, at 90°) and [`CARDAN_SEQ`](../../mesqsim/estimators/anatomy.py#L59) (the angle order per joint
for the range of motion) are written once, in SMPL segment frames, and imported by the solvers, the fits and the
scorer's excitation measure. A wrong entry would make GEO-K wrong while everything still runs: the tests
`test_true_joint_angles_follow_the_cardan_sequence` and `test_exact_recovery_over_random_deltas_axes_and_poses`
pin them.

**The hinge closed form** ([`hinge_closed_form`](../../mesqsim/estimators/hinge.py#L52)), line by line:

```python
c = float(np.sum(a * b))                                     # C = Σ a·b  (horizontal shadows, parent and child)
s = float(np.sum(b[:, 0] * a[:, 1] - b[:, 1] * a[:, 0]))      # S = Σ (b × a)_z
info = float(np.hypot(c, s))                                 # I = √(C² + S²): how much the window says
rss = max(float(np.sum(a * a) + np.sum(b * b)) - 2.0 * info, 0.0)   # the least-squares residual at the optimum
sigma2 = max(rss / (2 * n - 1), sigma_floor_rad**2)          # its spread, never below the fusion floor
return HingeFit(float(np.arctan2(s, c)), float(min(info / sigma2, KAPPA_MAX)), info, rss, n)
```

δ̂ = atan2(S, C) maximises Σ a·Y(δ)b; κ = I/σ̂² is exact for this likelihood (MATH_EXPLAINED §6.3). When the knee
axis stands vertical every shadow is a dot, I = 0 and κ = 0: the blind spot, by construction rather than by a
special case. The [`sigma2`](../../mesqsim/estimators/hinge.py#L65) floor stops noise-free data from claiming
infinite confidence. *Tests:* `test_worked_example_12_and_5_degrees_gives_minus_7`,
`test_vertical_axis_is_the_blind_spot`, `test_likelihood_spread_matches_one_over_sqrt_kappa` (checks 22.1, 23.1).

**Axis identification.** [`identify_hinge_axes`](../../mesqsim/estimators/hinge.py#L95) is the free fit (four
angles, least squares on ‖ω_p × u_p‖ − ‖ω_c × u_c‖ from fixed starts; its `contrast` compares the best cost with 16
probe axes). [`identify_hinge_axes_tpose`](../../mesqsim/estimators/hinge.py#L198) is GEO-ID's main route: with the
T-pose, u_p is horizontal there and u_c = m_rel·u_p, so a 1° grid over one azimuth plus a golden-section refinement
replaces a four-dimensional search. It returns two measures of identifiability: `contrast` (best cost / median
cost) and `spread` (the residual at the median azimuth relative to the rates). In GEO-ID's
[`_identify`](../../mesqsim/estimators/geo.py#L504), the tie comes from
[`self._tpose_tie`](../../mesqsim/estimators/geo.py#L528) and the fit is refused unless it stands out
([`axis_tpose_contrast_max`](../../mesqsim/estimators/geo.py#L531), 0.8) and a wrong axis fits badly
(`axis_min_spread`, 0.1). *What would break:* without the spread test a rigid thigh–shin motion — every azimuth fits,
all costs near zero — could pass with any contrast (0.79 in the test). *Tests:*
`test_axes_identified_within_one_degree_on_noise_free_hinge_motion`,
`test_tpose_tied_identification_recovers_the_axis_and_refuses_rigid_motion` (check 24.1).

**The sign.** An axis and its opposite fit equally well, and a flipped child axis turns δ by 180°.
[`vote_vertical`](../../mesqsim/estimators/hinge.py#L145) votes when the axis tilts: the true axis has the same
vertical component seen from both pods. [`vote_tpose`](../../mesqsim/estimators/hinge.py#L157) votes by asking which
of δ and δ + 180° keeps the knee's rotation since the T-pose plausible (the wrong one is a half turn about the
vertical). In [`_hinge_id`](../../mesqsim/estimators/geo.py#L583) the votes are tallied over windows; a tied fit
fixes the relative sign by construction ([`tally = max(tally, 1)`](../../mesqsim/estimators/geo.py#L595)). *Tests:*
`test_flipped_axis_is_corrected_by_the_vertical_rule`, `test_flipped_axis_is_corrected_by_the_tpose_rule`.

**Facing and T-pose frames.** [`_vote_knee_direction`](../../mesqsim/estimators/geo.py#L607) decides which way the knee
axis points (the knee only flexes one way); [`_update_facing`](../../mesqsim/estimators/geo.py#L622) turns the
knee axes into the body's facing and every pod's segment frame from the T-pose,
[`m_hat`](../../mesqsim/estimators/geo.py#L650) = F_T⁻¹·Y(c_i(T))·q_i(T). This is why GEO-ID's other joints wait for
a knee: no knee axes, no facing, no T-pose frames, no range of motion.

**The 2-DoF constraint** (GEO-K only; [`twodof_terms`](../../mesqsim/estimators/twodof.py#L51),
[`twodof_coefficients`](../../mesqsim/estimators/twodof.py#L59), [`twodof_solve`](../../mesqsim/estimators/twodof.py#L90)).
Each pair gives P, Q, R; the cost Σ(P cos δ + Q sin δ + R)² is a degree-2 trigonometric polynomial with five
coefficients, so it is evaluated on a 1° grid for free, every local minimum is refined by Newton steps, κ comes from
the curvature (Laplace), and a second minimum within a likelihood ratio of e³ makes the window *ambiguous* (κ = 0)
instead of guessing. *Tests:* `test_trig_polynomial_equals_the_sum_of_squares`, `test_two_equal_minima_are_flagged_ambiguous`
(check 25.1).

**The range of motion** ([`rom_terms`](../../mesqsim/estimators/rom.py#L62), [`rom_angles`](../../mesqsim/estimators/rom.py#L70),
[`rom_cost`](../../mesqsim/estimators/rom.py#L85), [`rom_solve`](../../mesqsim/estimators/rom.py#L91)). The joint
rotation for a hypothesis δ is J(δ) = M0 + cos δ·M1 + sin δ·M2, so the five matrix entries a Cardan sequence needs
are computed for the whole 1° grid at once; the cost is the squared distance outside the fitted box Ω; the
"likelihood" exp(−cost/2σ²) is summarised by its circular mean and a moment-matched κ. Inside the limits the cost
is flat: `frac_violating = 0`, κ = 0. *Tests:* `test_rom_fast_form_equals_direct_joint_rotation`,
`test_rom_recovers_delta_when_the_motion_reaches_the_limits`, `test_rom_has_almost_no_information_inside_the_limits`
(check 26.1).

### 3.4 The events and the filter (geo.py, vmfilter.py)

**Events.** [`_events`](../../mesqsim/estimators/geo.py#L273) builds a time-ordered list: one init at the end of
the box, the T-pose snapshot at the end of the T-pose, one reset whenever a pod's new boot epoch first reaches the
host, one window every `step_s` from the start of the motion
([`_t_geom_start`](../../mesqsim/estimators/geo.py#L257): the T-pose is a snapshot, not joint mechanics), none inside
the unrecorded idle. Sorting by (time, kind) puts a reset before a window at the same instant. The snapshot is its
own event because of a bug found in S2-M6: taken at the first window, it came after the resets of pods that had
rebooted in the idle, and those pods lost their T-pose frames for good
(`test_a_reboot_in_the_idle_keeps_the_tpose_frames`).

**The filter** ([`JointFilter`](../../mesqsim/estimators/vmfilter.py#L26)), method by method:

- [`_advance`](../../mesqsim/estimators/vmfilter.py#L40) refuses events out of order and predicts:
  [`vm_predict_kappa`](../../mesqsim/estimators/vmfilter.py#L44) shrinks κ by A(κ′) = A(κ)·e^{−qΔt/2}; μ stays.
- [`init`](../../mesqsim/estimators/vmfilter.py#L52) sets the box belief; [`reset`](../../mesqsim/estimators/vmfilter.py#L59)
  sets κ = 0 for the joints of a rebooted pod (the mean is kept but weighs nothing).
- [`update`](../../mesqsim/estimators/vmfilter.py#L65) multiplies densities with
  [`vm_product`](../../mesqsim/estimators/vmfilter.py#L68): κ′e^{iμ′} = κe^{iμ} + κ_w e^{iδ_w}. A window with κ_w = 0
  changes nothing — which is how "no information" stays harmless.
- [`query`](../../mesqsim/estimators/vmfilter.py#L72) answers at any time from the latest recorded state, predicted
  forward: causal by construction.

**Calibration and replay.** [`calibrated_kappa`](../../mesqsim/estimators/geo.py#L75) applies s·k/(1 + k·σ_m²) per
joint type; [`replay`](../../mesqsim/estimators/geo.py#L99) re-runs a recorded event list with other parameters, which
lets [`geo_fit.py`](../../mesqsim/geo_fit.py) fit q, the scales and κ_hold on thousands of validation sessions
without re-solving a single window, and lets the results compute the no-filter and hinge-only ablations exactly.
*What would break:* a predict rule with the wrong sign of q would make the belief sharpen with time; forgetting the
reset would keep a stale north after a reboot. *Tests:* `test_without_updates_the_mean_stays_and_kappa_follows_the_exact_rule`,
`test_large_kappa_prediction_matches_the_variance_sum`, `test_update_is_the_product_of_the_two_densities`,
`test_reset_drops_kappa_to_zero_and_events_must_be_ordered`, `test_replay_reproduces_the_estimator_exactly`,
`test_a_reboot_resets_the_joints_of_that_pod`, `test_geo_id_equals_b1_on_noise_free_motion_without_information`,
`test_kappa_hold_meets_the_target_or_falls_back_to_the_smallest_tail` (check 27.1).

### 3.5 The scorer's changes (evaluate.py, results_step2.py, geo_fit.py, geo_sessions.py, geo_score.py)

- [`evaluate.py`](../../mesqsim/evaluate.py) registers the six GEO methods beside the four baselines;
  [`USES_TRUTH`](../../mesqsim/evaluate.py#L51) lists the methods that read simulator truth (O1, O2, GEO-K), and every
  table stars them. `errors.parquet` gained `kappa` and `uncertain` (missing for the baselines).
  *Test:* `test_scorer_columns_kappa_and_uncertain`.
- [`geo_sessions.py`](../../mesqsim/geo_sessions.py) re-creates exactly the sessions a dataset run would write (same
  windows, motion starts and seeds) but in worker processes, returning only compact arrays: the 519 test sessions
  need no run folders (a few gigabytes) on disk.
- [`session_worker`](../../mesqsim/results_step2.py#L46) runs every method on one session; the no-filter and
  hinge-only ablations are replays of GEO-ID's events (`test_replayed_ablations_equal_the_registered_estimators`).
  [`bootstrap_contrast`](../../mesqsim/results_step2.py#L120) and [`holm`](../../mesqsim/results_step2.py#L134) are the
  statistics (`test_bootstrap_and_holm`; E28, E29).
- [`geo_fit.py`](../../mesqsim/geo_fit.py): `fit_rom` on **train** (Ω, the 0.5/99.5 % percentiles of the true joint
  angles) and `fit_calibration` on **validation** (scales, model errors, q, κ_hold). Both write `fitted` values into
  the config with their command, commit and split in the source text.
- [`pair_timing_errors_ms`](../../mesqsim/geo_score.py#L22) is scorer-side: it joins the estimator's pairs to the true
  sample times (check 31.1).

---

## 4. The Step 1 modules Step 2 stands on

**`conventions.py`** — the one definition of every rotation convention.
[`quat_mul`](../../mesqsim/conventions.py#L85) is the Hamilton product with (x, y, z, w) storage;
[`yaw_quat`](../../mesqsim/conventions.py#L168) is Y(α); [`heading_rad`](../../mesqsim/conventions.py#L175) the
heading of a rotation (the angle of its x axis's horizontal projection); [`wrap_rad`](../../mesqsim/conventions.py#L196)
maps into (−π, π]; [`exp_map`](../../mesqsim/conventions.py#L206), [`log_map`](../../mesqsim/conventions.py#L215),
[`geodesic_angle_rad`](../../mesqsim/conventions.py#L226), [`quat_slerp`](../../mesqsim/conventions.py#L232);
Step 2 added [`matrix_to_cardan_rad`](../../mesqsim/conventions.py#L311) and the von Mises helpers
([`vm_product`](../../mesqsim/conventions.py#L406), [`vm_predict_kappa`](../../mesqsim/conventions.py#L417),
[`vm_halfwidth_rad`](../../mesqsim/conventions.py#L447)). If anything here is wrong, every number is wrong, which is
why `tests/test_conventions.py` and `tests/test_conventions_step2.py` pin each function against scipy or a hand
example.

**`correction.py`** — [`relative_heading_rad`](../../mesqsim/correction.py#L26) defines δ = wrap(ψ_c − ψ_p);
[`chain_corrections_rad`](../../mesqsim/correction.py#L36) accumulates the outward correction from the pelvis
(c_child = c_parent + δ̂); GEO-ID uses it at the T-pose to put every pod into the pelvis pod's north before it
computes the facing. *Tests:* `test_worked_example_section_4_4`, `test_correction_never_changes_tilt`.

**`step0.py`** — [`decode`](../../mesqsim/step0.py#L144) (Stage 5 of PIPELINE_EXPLAINED),
[`box_headings`](../../mesqsim/step0.py#L173) (B1, and GEO's starting belief), [`tpose_snapshot`](../../mesqsim/step0.py#L186).
Step 2 depends on its `status`, `valid`, `epoch` and `t_pod_s` columns being exactly right. *Tests:*
`tests/test_step0.py`, `test_step0_rebuilds_every_tick_and_flag`.

**`evaluate.py`** — [`estimate`](../../mesqsim/evaluate.py#L75) (the four baselines),
[`method_estimate`](../../mesqsim/evaluate.py#L112) (any method), [`Scorer`](../../mesqsim/evaluate.py#L163). The
primary metric is one line, [`errors`](../../mesqsim/evaluate.py#L179): |wrap(δ̂ − δ)|.

**`io.py`** — [`write_session`](../../mesqsim/io.py#L82) and [`read_session`](../../mesqsim/io.py#L94): zstd parquet
and sorted-key JSON, so the same seed gives the same bytes (check 8.1).

**`config.py`** — [`Config`](../../mesqsim/config.py#L37): `cfg["geometry.window_s"]` reads a value;
[`with_overrides`](../../mesqsim/config.py#L93) makes a modified copy (sweeps, `--set`);
[`hash`](../../mesqsim/config.py#L87) names run folders; [`section`](../../mesqsim/config.py#L103) gives an estimator
only the `geometry:` section (rule 8: the estimator never sees the simulator's `error_model:`);
[`set_leaf_in_file`](../../mesqsim/config.py#L175) writes fitted values into the YAML in place, comments kept.
*Tests:* `tests/test_config_write.py`, `test_the_estimator_sees_only_its_own_beliefs`.

---

## 5. The tests: what each file proves

| file | proves |
|---|---|
| `test_conventions.py`, `test_conventions_step2.py` | the rotation and von Mises conventions, against scipy and hand examples |
| `test_codec.py` | the wire frame, byte for byte (the golden vector), wraps and saturation |
| `test_correction.py` | δ, the outward correction, the 12°/5° example, tilt invariance, room invariance |
| `test_step0.py`, `test_sim_slice.py` | Step 0 rebuilds every tick and flag; the frames contract; B0 and O2 on clean data |
| `test_sim_full.py` | the error model: turning, scale factor, fusion noise, reboots, the shared idle turns |
| `test_amass.py`, `test_diodem.py`, `test_splits.py` | the datasets load as documented; the split never leaks a subject |
| `test_clock.py` | the clock map, pairs and rates |
| `test_geo_contract.py` | causality, determinism, label-blindness, the truth door, output ranges, pair timing |
| `test_hinge.py` | closed form, likelihood, blind spot, axis identification, the sign rules, rigid motion refused |
| `test_twodof_rom.py` | the 2-DoF polynomial and ambiguity; the range-of-motion fast form, recovery and flatness; Ω's fit |
| `test_filter.py` | predict, product, reset, replay, GEO-ID = B1 without information, the κ_hold rule |
| `test_results_step2.py` | ablation replays equal the estimators; bootstrap, Holm, coverage, selective risk |
| `test_config_write.py` | fitted values are written in place, everything else kept |
| `test_validation.py` | every registered check of `docs/VALIDATION.md` passes (quick mode) |
| `test_repo_hygiene.py` | no file over 5 MB, no data committed |
| `test_explain_examples.py` | every worked example of MATH_EXPLAINED.md, recomputed |
| `test_explain_links.py` | every link and line reference in these explanation files resolves |

---

## 6. How to extend the code

**Add a joint model** (say, a hinge model for the elbow):
1. write the solver as a pure function of pairs (quaternions in, (δ, κ) out) in `mesqsim/estimators/`, with a
   docstring that names the brief section and the formula;
2. add its axes to `anatomy.py` in SMPL segment frames;
3. dispatch to it in [`_solve`](../../mesqsim/estimators/geo.py#L427) and give it a calibration class in
   `_solver_type` (its κ must be fitted on validation like the others);
4. test exact recovery on noise-free synthetic motion, the "no information" case (κ = 0) and an ambiguity if the
   model has one; add a numbered check to `validation.py` if it changes a reported number.

**Add an estimator** (Step 3 will add learned ones): implement
`estimate(series, session, t_query_host_s, cfg, *, truth=None) -> Estimate`; read only the series, `session.json`
and your own config section; register it in `evaluate.py` (`METHODS`, `method_estimate`; add it to `USES_TRUTH` if it
reads truth). Copy the three contract tests of `test_geo_contract.py` for it: causal, deterministic, label-blind.

**Add a check:** decorate a function in `validation.py` with
[`register`](../../mesqsim/validation.py#L57)`(id, title, what, why, expected, command)`, return a `CheckResult(text,
passed)`, support `quick` mode, then run `python scripts/validate.py` (it rewrites `docs/VALIDATION.md`) and
`pytest tests/test_validation.py`.

**Add a dataset:** a loader in `mesqsim/data/` that returns segment orientations in the +Z-up world and SMPL segment
frames; a manifest of every file (path, size, SHA-256) via `manifest.py`; subjects split by `splits.py` before any
fit; checks like 10.1–10.5 (up axis, left/right, joint ranges) before trusting a single number.

---

## 7. How to debug

**Sign flips.** Symptom: an error near 2|δ| (the 12°/5° example leaves 14°) or near 180°. Check, in this order: the
measurement model's Y(−ψ) (check 3.1), the target's order wrap(ψ_c − ψ_p) (`test_worked_example_section_4_4`), the
hinge sign tally (`diag.windows` has `sign_tally` per window). A δ off by exactly 180° at one knee is a flipped axis.

**Frame order.** Symptom: errors that grow with how much a joint bends. A quaternion q maps body to world, and
compositions read right to left: q_p⁻¹·q_c is the child in the parent's frame. `test_composition_order_is_right_to_left`
pins it; when in doubt, rotate a known vector with `quat_rotate` and look.

**Units.** Code is in radians, text and tables in degrees, times in seconds, `ms_lo` in milliseconds, quaternions
in int16 counts until Step 0 divides by 32767. κ behaves like 1/σ² with σ in radians, so 1/√κ is a standard
deviation in radians. A κ that looks absurd (10⁴) may be a raw window κ before
[`calibrated_kappa`](../../mesqsim/estimators/geo.py#L75).

**Causality leaks.** Symptom: an estimator that is good from its first second, or better on a recording cut short.
Run `test_causal_deleting_later_frames_changes_nothing_before`: it deletes frames after a time and demands identical
answers before it. Every selection in the clock uses `t_host_s <= t_now`, never sample times the host could not have
known.

**Label leaks.** Symptom: GEO-ID close to O2. Run `test_label_blind_identical_without_labels` (it deletes every label
file and `meta.json` and compares bytes), and grep the estimator package for `labels_`, `meta`, `psi_rad`,
`delta_rad` outside `contract.py`. Only GEO-K may receive truth, and only through `GeoTruth`.

**The fastest view of a session:** `GeoEstimator(cfg).run(series, session, host_times, diagnostics=True)` returns
`diag.windows` (one row per window and joint: pairs, δ̂_w, raw κ, status — `no_pairs`, `no_axes`, `no_frames`,
`blind`, `ambiguous`, `flat`, `unbounded`, `ok`) and `diag.state` (identified axes, sign tallies, facing).

---

## 8. Self-check questions

1. **Where does an estimator get its parameters, and why not from the run's config?** From `configs/default.yaml`'s
   `geometry:` section ([`section`](../../mesqsim/config.py#L103)); the run's config holds the simulator's truth
   (`error_model:`), which an estimator must never see.
2. **Which object is allowed to read the labels?** The [`Scorer`](../../mesqsim/evaluate.py#L163) (and the
   scorer-side modules `geo_score.py`, `geo_sessions.py`, `results_step2.py`); never an estimator.
3. **Why does `pod_samples` drop held frames?** A held frame repeats an old sample; pairing it at its new time
   would claim that the pod did not move.
4. **What does κ = 0 mean in `JointFilter.update`, and why is that convenient?** No information: κ_w e^{iδ_w} = 0
   adds nothing, so solvers can always return a value and say "ignore me" with κ = 0.
5. **Why are the GEO-ID ablations replays instead of separate runs?** The events (δ̂_w, raw κ) do not depend on the
   filter's parameters, so replaying them is exact and saves re-solving every window
   (`test_replayed_ablations_equal_the_registered_estimators`).
6. **Why must GEO-ID solve a knee before any other joint?** The facing comes from the knee axes, the T-pose frames
   from the facing, and the range-of-motion solvers need the T-pose frames.
7. **What would `MIN_SLOPE_BLOCKS = 2` break?** Slopes through two noisy minima, as after an idle gap: once +1.5 %
   and timing errors of 41 ms, breaking check 31.1.
8. **How do you add a learned estimator without touching the scorer?** Implement the contract, register it in
   `evaluate.py`, copy the three contract tests; the scorer and the results treat it like any other method.
9. **A knee is off by exactly 180°. Where do you look?** The sign: `sign_tally` in `diag.windows`, the vertical and
   T-pose votes, and whether the axes were tied (a tied fit cannot flip the relative sign).
10. **Which test fails if someone moves `hinge_closed_form` twenty lines down?** `tests/test_explain_links.py`: its
    link here names `hinge_closed_form`, and that name must stay on the linked line.

---

*Next: [THEORY.md](THEORY.md) — where the ideas come from, with full references.*
