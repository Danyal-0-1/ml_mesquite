# Dataset card: `twin_v1`

This card follows *Datasheets for datasets* (Gebru et al. 2021). Section 5 is the full specification that a loader
can rely on: read it before writing `mesqsim/learn/loader.py`. Every number here was computed from the dataset itself
or comes from the work log. Examples D1 to D6 are recomputed by
[tests/test_step3_guide_examples.py](../../tests/test_step3_guide_examples.py) whenever the dataset is present.

Back to the [index](README.md). The course that uses this card is [LEARN_THE_MODEL.md](LEARN_THE_MODEL.md).

## 1. At a glance

| | |
|---|---|
| what | simulated sessions of the Mesquite suit and phone, made by the Mesquite twin, with the true headings as labels |
| where | `$MESQ_OUTPUT_DIR/datasets/twin_v1/`, outside git (on Danyal's Mac: `../mesquite_work/outputs/datasets/twin_v1`) |
| size | 19.42 GB in the 88 files that `MANIFEST.csv` lists: 84 HDF5 shards, `index.parquet`, `meta.json`, `README.md` and `coverage_train.csv` |
| sessions | 665: train 519, validation 74, test 72 (D1) |
| motion | 200.4 hours of AMASS motion, 246 subjects, split by subject (D1) |
| one step | one tick of the host clock at 32 Hz (0.03125 s) |
| made by | `python -m mesqsim twin-dataset --name twin_v1 …` at commit `8adc115` (WORKLOG S3-M7, 5,040 s on 5 workers) |
| checked by | `python -m mesqsim verify-dataset --name twin_v1`, which re-hashes every file against `manifests/twin_v1.csv` (27 s) |
| manifest | SHA-256 of `MANIFEST.csv`: `f65b7ee61cea591c5b413e6801db4a1ce1d3885cab98896ffa9f1a3363dd877a` |
| versions | `feature_version` 1; twin config hash `5de4440cd537…`; `mesquite-mocap` at `87d7c3b`; calibration `twin_calibration_g4a03aeb-dirty` (200 draws, 20 kept) |

## 2. Motivation

- **Why it exists.** A learned model needs the true heading of every pod at every step, and the real suit cannot
  give it: OptiTrack recorded only seven takes, without a marker triad per pod. The twin simulates the whole chain:
  AMASS motion, the skin, the sensors, the chip's fusion, the wire, the phone and the page. It was calibrated against
  Danyal's recordings, and gate G-T passed (26 of 42 statistics covered, against Step 1's 14). So its sessions come
  with exact labels.
- **What it is for.** Training and validating the Step 3 models: TCN-J (δ per joint) and TCN-P (ε per pod, relative
  to the phone). `python -m mesqsim score-model` scores them on the held-out subjects.
- **Who made it.** It was generated on Danyal's Mac in Step 3 (S3-M7, 2026-11-21) by the twin in `mesqsim/twin/`.

## 3. Composition

**One session** is one simulated recording of one AMASS subject wearing the suit with the phone on the belt. It
follows the procedure in `mesqsim/twin/session.py`:
1. the pods boot lying flat in their box;
2. they are carried to their straps, while the phone app starts and zeroes itself at its first frame;
3. in the T-pose the phone is zeroed again (unless that slip is drawn), and the page's Set T-pose is pressed;
4. a motion program follows: the subject's own AMASS clips joined into one long session.

Sessions last from 3.7 to 33.1 minutes, median 18.0, which is 7,141 to 63,611 steps, median 34,537 (D3).

**D1. The splits** (from `index.parquet`; a subject is its AMASS subset and id, because CMU and KIT reuse ids):

| split | sessions | subjects | hours | steps | shards | reference windows (§5.7) |
|---|---|---|---|---|---|---|
| train | 519 | 173 | 152.3 | 17,548,179 | 65 | 54,153 |
| val | 74 | 37 | 26.4 | 3,037,218 | 10 | 9,400 |
| test | 72 | 36 | 21.7 | 2,504,962 | 9 | 7,734 |

Each training subject appears in 3 draws and each validation or test subject in 2. A draw is the same clips with new
corruptions: new norths, biases, slips and losses.

**D2. The conditions** (sessions per split, train / val / test):

| condition | train | val | test | what it means |
|---|---|---|---|---|
| suit `suit_2026_11_14` | 263 | 40 | 45 | the new suit: 14 pods plus HipsAlt, the binary wire (`binary16`) |
| suit `suit_2026_11_13` | 256 | 34 | 27 | the old suit: 10 pods plus HipsAlt (no hands, no feet), the JSON wire |
| HipsAlt absent | 83 | 11 | 14 | the tree without the second pelvis pod: pod 2 is absent all session |
| a limb restarts | 235 | 29 | 34 | one pod, or a limb's two upper pods, restarts over USB after the box; its filter restarts where the pod faces, with a new north |
| phone zeroed at Start | 315 | 45 | 43 | the phone's only Zero is the automatic one at Start, in the hand; there is none in the T-pose |
| camera mismatch | 134 | 29 | 23 | the app's camera setting is a quarter turn off (±90°) |
| tree mismatch | 23 | 2 | 5 | the app's tree setting is the wrong one |
| protocol program | 3 | 0 | 0 | KIT subject 3's protocol program; every other session is a random program |

**What is not in it.** No real recordings: those are in `$MESQ_DATA_DIR/real_suit/` and are measured, not trained
on. No raw gyroscope or accelerometer streams, because the pods do not send them. No positions, and no BVH files.
Nothing about the body's facing in the room, which no deployable input can reveal.

## 4. How it was made, and the labels

- **The chain.** The steps are: AMASS motion; the SMPL+H skin; the pod and phone sites; the sensor physics and
  errors; the six-axis fusion (VQF) in the loop; the wire; the phone app; Step 0's decoding; the features. Every twin
  parameter is in `configs/twin.yaml` with its label (2 fitted, 10 measured, 2 datasheet, 5 firmware and 59
  assumption, as in the banner of `docs/RESULTS_STEP3.md`). Each session's parameters are a draw from the
  calibration's 20 kept draws, widened by 1.5.
- **The features** are computed by `mesqsim/learn/features.py`, the same function that live inference calls. The
  value at step k uses only what had arrived by its host time t_k. It describes the body at t_k − 0.1 s
  (`feature_lag_s`), when each pod's samples usually bracket that moment.
- **The labels** are the twin's truth:
  - δ_pc = wrap(ψ_c − ψ_p) is how far a child pod's north has slid from its parent's;
  - ε_i = wrap(ψ_i − ψ_ph) is pod i's north relative to the phone's;
  - both are in radians, wrapped to (−π, π].
- **D5. The chain identity.** ε_c = ε_p + δ_pc, because ψ_c − ψ_ph = (ψ_c − ψ_p) + (ψ_p − ψ_ph). At step 26,013 of
  `val_KIT_12_v0`:
  - ε of LeftUpLeg (pod 9) is −74.486°;
  - δ of the left knee is −113.013°;
  - their sum, −187.499°, wraps to 172.501°, which is ε of LeftLeg (pod 10).

## 5. The specification a loader can rely on

### 5.1 Files

```text
twin_v1/
  shards/<split>_<nnnnn>.h5   HDF5, 8 sessions per shard, one group per session: /sessions/<id>/
  index.parquet               one row per session: where it is and what it contains
  meta.json                   how it was made
  MANIFEST.csv                every file's name, size and SHA-256 (a copy is committed as manifests/twin_v1.csv)
  coverage_train.csv          training windows per stratum (pod or joint type × motion × time since boot)
  README.md                   how to open it
```

`index.parquet` has these columns:

| column | what |
|---|---|
| `id` | the session id, e.g. `val_KIT_12_v0`: split, subset, subject, draw |
| `split` | `train`, `val` or `test`. Never mix them |
| `shard` | the shard that holds the session, e.g. `shards/val_00072.h5` |
| `steps`, `length_s` | its length in steps and in seconds |
| `subset`, `subject` | the AMASS subject. The pair is the identity |
| `variant`, `composition`, `program_seed`, `seed` | which draw, which composition of the clips, and the seeds |
| `program` | e.g. `random:KIT/12:23.84min` |
| `suit_version`, `wire`, `hipsalt` | the conditions of D2 |
| `slips` | JSON: `limb_restart` (its pods and time) or null; `zero_at_start`; `camera_mismatch_deg`; `tree_mismatch` |
| `part` | training subjects only: the smallest nested part the subject belongs to (16, 8, 4, 2 or 1; §5.8) |

`meta.json` holds `name`, `commit`, `seed`, `feature_version`, `feature_lag_s`, `hours`, `sessions`, `subjects`,
`options`, `calibration`, `mesquite_mocap` and `twin_config_hash`.

### 5.2 One session's arrays

Every array is on the session's step grid of T steps. The arrays are gzip-compressed, chunked 2,048 steps at a time,
so reading a 576-step window touches one or two chunks.

| array | shape, type | units and meaning |
|---|---|---|
| `t_s` | (T,) float64 | host time of each step, s, 0.03125 s apart; about 0 is when the first pod booted |
| `x_pod` | (T, 15, 8) float32 | per pod, in the order of §5.3: tilt g = R̂ᵀe_z (3, a unit vector: world up in the pod's frame); the rotation rate between its last two fresh samples, Log(R̂(t₋)ᵀR̂(t))/(t − t₋), in its own frame (3, rad/s); the age of its last fresh sample (1, s); held (1: 1 if its last frame repeated a sample) |
| `x_joint` | (T, 14, 6) float32 | per joint: the first two columns of R̂_pᵀR̂_c, column by column [r11, r21, r31, r12, r22, r32] (the 6-D rotation of Zhou et al. 2019) |
| `x_phone` | (T, 8) float32 | the phone's eight, from its messages, with "orientation only" in place of held |
| `x_pod_phone` | (T, 15, 6) float32 | per pod: the first two columns of R̂_phᵀR̂_i, the pod's orientation seen from the phone |
| `state_pod` | (T, 15) uint8 | 0 lost (nothing arrived since the previous step), 1 held (a repeat arrived), 2 fresh |
| `state_phone` | (T,) uint8 | 0 no message yet, 1 orientation only, 2 full |
| `y_delta`, `valid_delta` | (T, 14) float32, bool | δ per joint, rad. Valid where both pods are on and outside a transition |
| `y_eps`, `valid_eps` | (T, 15) float32, bool | ε per pod, rad. Valid where the pod is on, the phone has been zeroed, and outside a transition |
| `strata/time_since_boot_s` | (T,) float32 | seconds since the first pod booted |
| `strata/motion` | (T,) uint8 | the program piece's kind (§5.4) |
| `strata/transition` | (T,) bool | inside a blend between two clips. The targets are invalid there |
| `strata/slip` | (T,) uint8 | bits: 1 a restart has happened, 2 zeroed at Start, 4 camera mismatch, 8 tree mismatch |

The group's attributes are `subject` (e.g. `KIT/12`), `split`, `suit_version` and `conditions`, a JSON string. They
are never labels.

**Values a loader must expect:**
- **A pod absent all session** (by suit version, or HipsAlt off) has zero features and state 0 at every step. In
  `val_KIT_12_v0`, an old-suit session, pods 5, 8, 11 and 14 are zero throughout.
- **A pod lost for a while** keeps its last values, and its age grows.
- **The phone before its first message** has zeros and state 0.
- **Invalid targets are stored as 0, except inside a transition,** where they keep the blend's value: 2,258,768 of
  the 8,369,798 invalid ε cells of the validation split (δ: 2,046,384 of 8,606,024). Always multiply by the valid
  mask; never trust a target where its mask is False, zero or not.
- **ε can be valid one step before the phone's first message arrives.** The Zero has happened, but its message is
  still in flight. In `val_KIT_12_v0` that is step 458 (16.03 s) against 459.

**D4. How often** (all 74 validation sessions):

| | fraction of steps |
|---|---|
| pod steps fresh / held / lost | 81.29 % / 0 % / 18.71 % (lost includes absent pods) |
| phone steps full / orientation only / no message | 79.26 % / 1.02 % / 19.72 % |
| valid δ (joint steps) | 79.76 % |
| valid ε (pod steps) | 81.63 % |

### 5.3 The orders

| pod | 0 Head | 1 Spine | 2 HipsAlt | 3 LeftArm | 4 LeftForeArm | 5 LeftHand | 6 RightArm | 7 RightForeArm |
|---|---|---|---|---|---|---|---|---|
| | **8** RightHand | **9** LeftUpLeg | **10** LeftLeg | **11** LeftFoot | **12** RightUpLeg | **13** RightLeg | **14** RightFoot | |

| joint | parent → child | type | joint | parent → child | type |
|---|---|---|---|---|---|
| 0 lower_back | 2 → 1 | 3-DoF | 7 r_wrist | 7 → 8 | 2-DoF |
| 1 neck | 1 → 0 | 3-DoF | 8 l_hip | 2 → 9 | 3-DoF |
| 2 l_shoulder | 1 → 3 | 3-DoF | 9 r_hip | 2 → 12 | 3-DoF |
| 3 r_shoulder | 1 → 6 | 3-DoF | 10 l_knee | 9 → 10 | hinge |
| 4 l_elbow | 3 → 4 | 2-DoF | 11 r_knee | 12 → 13 | hinge |
| 5 r_elbow | 6 → 7 | 2-DoF | 12 l_ankle | 10 → 11 | 2-DoF |
| 6 l_wrist | 4 → 5 | 2-DoF | 13 r_ankle | 13 → 14 | 2-DoF |

The code holds them as `mesqsim.body.PODS` and `mesqsim.body.JOINTS` (with `.parent`, `.child`, `.jtype`).

### 5.4 Motion codes and slips

`strata/motion` codes: 0 preparation, 1 tpose, 2 still, 3 transition, 4 walk, 5 turn, 6 squat, 7 kick, 8 arm, 9 jump,
10 bend, 11 stairs, 12 kneel, 13 other, 14 twist, 15 leg_swing, 16 leg_side, 17 arm_forward, 18 arm_side and
19 turn_spot. They are in `mesqsim.learn.store.MOTION_CODES`, and the slip bits are in `SLIP_BITS`.

### 5.5 Reading it

The library interface only:

```python
import h5py, pandas as pd
idx = pd.read_parquet(root / "index.parquet")            # root: a pathlib.Path to twin_v1/
row = idx[idx["split"] == "val"].iloc[0]
with h5py.File(root / row["shard"], "r") as h5:
    g = h5["sessions"][row["id"]]
    x = g["x_pod"][1000:1576]                             # a slice reads only the chunks it needs: (576, 15, 8)
```

**The labels go through one door.** `mesqsim.learn.targets.read_targets(group, start, end)` returns `y_delta`,
`valid_delta`, `y_eps` and `valid_eps`, either for one window or, without `start` and `end`, for the whole session.
No other module reads the `y_*` arrays (CLAUDE.md rule 8). The `Cheat` model is caught because it imports this
module. The scorer reads the targets the same way.

**Memory.** A training step holds about 1.4 KB of arrays, so the training split is about 24 GB once decompressed. It
does not fit in this Mac's memory, so read windows from the shards. Open each file once per DataLoader worker, in the
worker rather than in `__init__`.

### 5.6 What every input guarantees

- **Deployable.** Every input exists in the live stream: pod frames after Step 0, and the phone's messages as the
  page receives them. Nothing comes from the truth, the future or OptiTrack.
- **Causal.** The value at step k uses only data that had arrived by t_k. A model that is causal on these arrays is
  causal live.
- **Blind to a common turn.** Turning every pod and the phone by one heading changes no input. Check C5 proves it to
  machine precision. So no model can learn north itself: δ comes from the body, and ε needs the phone
  (`x_pod_phone`).

### 5.7 Windows (the reference rule)

- A window is 576 steps (18 s). Its loss region is its last 320 steps (10 s). The 256 steps before the loss region
  (8 s) cover the TCN's 253-step memory, so every scored step has a full history inside the window.
- `mesqsim.learn.store.windows(steps)` gives the reference windows. They end at steps 576, 896, 1,216 and so on,
  every 320 steps, so the loss regions tile the session with no overlap. Every step after the first 256 lies in
  exactly one loss region, until the last 320-step tile that fits.
- A window never leaves its session, and never mixes splits.
- `val_KIT_12_v0` has 49,684 steps and so 154 windows, the first three [0, 576), [320, 896) and [640, 1216) (G2 in
  the course). The split totals are in D1.

### 5.8 Splits and parts

- **Splits.** The split is by subject: no subject has sessions in two splits (`verify-dataset` checks it).
  - Train on `train`.
  - Choose settings and stop early on `val`.
  - Open `test` once, at the end.
- **Parts.** The training subjects come in nested parts, for the learning curve (`mesqsim/learn/curve.py`). A
  session's `part` is p when its subject is in the 1/p part and in every larger one. The 1/p part is
  `idx[(idx.split == "train") & (idx.part >= p)]`.

**D6. The parts:**

| part | sessions | subjects | hours | windows |
|---|---|---|---|---|
| 1/16 | 30 | 10 | 6.8 | 2,394 |
| 1/4 | 129 | 43 | 34.2 | 12,141 |
| all | 519 | 173 | 152.3 | 54,153 |

## 6. Uses

- **Intended uses.**
  - Training TCN-J and TCN-P and their rivals.
  - Choosing settings on `val`.
  - Scoring with `score-model` and through the page (`--through-page`).
  - The learning curve (`--extend` adds training draws; `val` and `test` never change).
- **Not intended.**
  - Any claim about the real suit. The twin is a model of it, and the next OptiTrack session is the only real test.
  - Tuning on `test`.
  - Real-time claims, which need the scorer's µs per step on a named CPU.

## 7. Known limits

- `meta.json` says 172 training subjects. There are 173: an earlier count merged CMU and KIT subjects that share an
  id (WORKLOG S3-M7). The index is right.
- No step of the validation split is "held" (D4). The chips sample at 50.1-60 Hz, faster than the 32 Hz send
  grid, so a frame almost always carries a new sample.
- The coverage rule holds in 121 of 864 strata. "Other" and "still" are covered everywhere; walking, turning, the
  T-pose and the rare and protocol motions are not (WORKLOG S3-M10, item 4).
- The twin covers 26 of 42 statistics of the recordings, and a classifier cannot tell its windows from real ones. But
  it does not drift the way the real shins drift in motion (WORKLOG S3-M10, item 8).
- B1 equals B0 on the twin. The pods boot in the box, so the box registration reads zeros. An offset the box would
  remove is no learned improvement.

## 8. Maintenance

- It is never edited in place: a new version is a new name.
- Extending it adds training draws: `python -m mesqsim twin-dataset --name twin_v1 --calibration
  <twin_calibration_g4a03aeb-dirty> --extend 1.33` adds a fourth draw, about 4.7 GB.
- After any change, `python -m mesqsim verify-dataset --name twin_v1` re-hashes it.
- The numbers on this card were computed from `index.parquet` and the shards by `pandas` and `h5py` reads, and are
  checked by `tests/test_step3_guide_examples.py` (D1-D6).
