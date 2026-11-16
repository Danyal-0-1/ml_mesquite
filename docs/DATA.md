# Data

*Status: S2-M0 (2026-11-13). AMASS downloaded (HumanEva, BMLmovi, CMU, KIT; SMPL+H G), verified and split; the
split waits for Danyal to freeze it. SMPL+H body model (extended, 16 shape parameters) downloaded and verified on
2026-11-14. DIODEM downloaded and verified. `python scripts/check_env.py --require amass_archives amass_files smplh
diodem` passes.*

## Picture first

Two public datasets feed Step 1. **AMASS** is human motion: every joint's rotation over time, for hundreds of people.
From it we compute how each of the 15 body segments was oriented, then simulate what the suit would send. **DIODEM**
is a reality check: real IMUs on two mechanical five-segment chains, with cameras giving the true orientation of every
segment. Neither dataset, nor anything derived from it, ever enters git. Only instructions, scripts and manifests
(file name, size, SHA-256) are committed.

| dataset | what | licence | how it is obtained | where it lives | manifest |
|---|---|---|---|---|---|
| AMASS | human motion, SMPL+H poses | non-commercial research licence, no redistribution | each person registers and downloads by hand | `$MESQ_DATA_DIR/amass/` | `manifests/amass_archives.csv`, `manifests/amass_files.csv` (written by `scripts/verify_amass.py`) |
| SMPL+H body model | body model needed for joint *positions* (figures only) | same rules as AMASS | each person registers and downloads by hand | `$MESQ_DATA_DIR/body_models/smplh/` | `manifests/smplh.csv` |
| DIODEM v3.0 | IMUs and optical truth on mechanical chains | CC0 1.0 (Dataverse) | `python scripts/download_diodem.py` | `$MESQ_DATA_DIR/diodem/` | `manifests/diodem.csv` |

`$MESQ_DATA_DIR` defaults to `data/` in the repository (git-ignored). On the collaborator's machine it is
`D:\mesquite_work\data`, outside OneDrive.

## 1. AMASS

**Reference.** Mahmood, Ghorbani, Troje, Pons-Moll, Black. *AMASS: Archive of Motion Capture as Surface Shapes.*
ICCV 2019. Site: <https://amass.is.tue.mpg.de>.

### 1.1 How to obtain it (a person must do this)

1. Register at <https://amass.is.tue.mpg.de> with your own account and accept the licence (non-commercial
   research use; redistribution forbidden). Everyone who uses the data registers individually; never share an
   account, a password or the files. Never store credentials in this repository.
2. Open the **Download** page. **Before downloading, note the size of each subset you choose**, and add it to the
   WORKLOG entry. The verification script also records the sizes of the downloaded archives.
3. For each chosen subset, download the **SMPL+H** body data (the page labels the body-model variants, e.g. "SMPL+H G";
   take SMPL+H rather than SMPL-X, because the brief's mapping uses SMPL+H poses. The loader also reads SMPL-X files if
   that is all there is.)
4. Save the archives, unchanged, in `$MESQ_DATA_DIR/amass/archives/`.
5. Register at <https://mano.is.tue.mpg.de> (the MANO / SMPL+H site), accept its licence, and download the
   **extended SMPL+H model** (the `.npz` version, which avoids the `chumpy` dependency). Put its `male/`,
   `female/` and `neutral/` folders, each with `model.npz`, in `$MESQ_DATA_DIR/body_models/smplh/`.
6. Run `python scripts/verify_amass.py`. It
   - records every archive's size and SHA-256 in `manifests/amass_archives.csv`;
   - extracts each archive safely (refusing paths outside the folder) to `$MESQ_DATA_DIR/amass/<Subset>/`;
   - records every motion file in `manifests/amass_files.csv`;
   - loads every file once and reports frame rates and failures;
   - records the body model in `manifests/smplh.csv`.
7. Run `python scripts/check_env.py --require amass_archives amass_files smplh` and `bash scripts/check_all.sh`.
   The AMASS checks (10.2–10.5 in [VALIDATION.md](VALIDATION.md)) stop being "skipped".
8. Commit the three manifests (names, sizes and checksums only) and note the download in `WORKLOG.md`.

### 1.2 Which subsets (proposal; Danyal decides)

| purpose | proposal | why |
|---|---|---|
| end-to-end slice (M3) | one **small** subset that contains walking, e.g. HumanEva or SFU (check sizes on the page) | fast to download and process; enough to prove the pipeline |
| simulation datasets (M5) | **CMU**, **KIT** and **BMLmovi** (the brief's examples) | together they cover walking straight and in circles, turning, bending, reaching, sitting down and standing still; check sizes first |
| never for training | **TotalCapture** | may form a later test set (brief §5.1). `verify_amass.py` flags it; the split tool will exclude it from training |

**Downloaded on 2026-11-13** (the collaborator's own AMASS account; the proposal above, with HumanEva as the small
walking subset; TotalCapture, 83,967,231 bytes, and SFU, 62,318,066 bytes, not downloaded). Sizes are those the
download server reported before downloading and match the files; SHA-256 in `manifests/amass_archives.csv`.

| subset (SMPL+H G) | archive bytes | motion files | subjects | frame rate | median length |
|---|---|---|---|---|---|
| HumanEva | 33,071,034 | 28 | 3 | 120 Hz | 17.5 s |
| BMLmovi | 748,793,904 | 1,801 | 86 | 120 Hz | 4.6 s |
| CMU | 1,993,171,135 | 2,082 | 102 | 120 Hz (60 Hz for 495 files) | 9.6 s |
| KIT | 2,196,810,275 | 4,232 | 55 | 100 Hz | 6.1 s |
| **total** | 4,971,846,348 | 8,143 (11.71 GB extracted) | 246 | | 23.2 h of motion |

Sources: `python scripts/verify_amass.py` (files, frame rates, 0 load failures), `python -m mesqsim split`
(subjects), and a scratch count of `trans` frames / `mocap_framerate` per file (lengths).

**SMPL+H body model, downloaded on 2026-11-14** (the collaborator's own MANO account): the "Extended SMPL+H model",
`smplh.tar.xz`, 113,231,292 bytes (the size the server reported), SHA-256
`46d5b8687be48c91181fa88271feff3a5e83aa62a481fad8a0bcb9254b2a74f1`, kept in `$MESQ_DATA_DIR/body_models/archives/`.
It holds `female/`, `male/` and `neutral/model.npz` (87,259,106 bytes each; SHA-256 in `manifests/smplh.csv`), a
licence and an `info.txt`: gender-specific models with 16 shape parameters, made for AMASS.

**Only 767 of the 8,143 sequences are at least 20 s long**, the default `dataset.window_s` (an `assumption`). With it,
BMLmovi contributes 2 windows and KIT 397, against CMU's 756: the simulated sessions are mostly CMU and KIT, and the
everyday actions of BMLmovi (short single actions) are almost absent. Shorter windows, or concatenating one
subject's sequences (brief §4.3, an optional variant), would change this. **Danyal decides**; the Step 1 results on
AMASS use the default until then.

### 1.3 Format

Inspected keys of the two layouts the loader handles (`mesqsim/data/amass.py`, tested on synthetic files written in
both layouts):

| key | SMPL+H release (`*_poses.npz`) | SMPL-X release (`*_stageii.npz`) | used |
|---|---|---|---|
| `poses` | (N, 156): 52 joints × 3 axis-angle | (N, 165) | first 66 values: root + 21 body joints |
| frame rate | `mocap_framerate` | `mocap_frame_rate` | yes (rates differ between subsets) |
| `trans` | (N, 3) metres | (N, 3) | figures only |
| `betas` | shape coefficients | shape coefficients | figures only (body model) |
| `gender` | string (unicode or bytes) | string | figures only |

Subject folders also hold `shape.npz` (betas and gender only); the loader skips it. When the files arrive, the
verification script prints the keys it saw, and this table must be checked against them (an open item until then).

### 1.4 From poses to the 15 segments

Global rotations compose down the SMPL tree: R_0 = Exp(root), R_j = R_parent(j)·Exp(local_j). R_j maps the canonical
body frame of the bone that starts at joint j (x: subject's left, y: up, z: forward, at the rest pose) to the world.
The segment orientation s_i of pod i is one joint's global rotation:

| pod | Mesquite segment | SMPL joint |
|---|---|---|
| 2 | HipsAlt (pelvis) | 0 pelvis |
| 1 | Spine (chest) | **9 spine3** (default; 6 spine2 is the alternative, `spine_joint=6`) |
| 0 | Head | 15 head |
| 9 / 12 | Left / RightUpLeg | 1 / 2 hips |
| 10 / 13 | Left / RightLeg | 4 / 5 knees |
| 11 / 14 | Left / RightFoot | 7 / 8 ankles |
| 3 / 6 | Left / RightArm | 16 / 17 shoulders |
| 4 / 7 | Left / RightForeArm | 18 / 19 elbows |
| 5 / 8 | Left / RightHand | 20 / 21 wrists |

- **Rest pose.** With every local rotation zero (the SMPL T-pose, arms horizontal), every segment equals R_0. That is
  the simulated T-pose (check 10.1).
- **Left and right** are the subject's own sides: SMPL +x points to the subject's left, and `up × forward = left` is
  tested.
- **Up axis.** AMASS worlds are +Z up, so the conversion to our world is the identity. This is not assumed: check 10.3
  verifies it on the downloaded data (the pelvis's body up axis must average to world +Z).
- **Spine choice.** spine3 is the uppermost trunk joint below the neck and collars, which is where a chest pod sits.
  spine2 is lower. Danyal decides; the switch is one argument.

## 2. DIODEM

**Reference.** Bachhuber, Lehmann, Weygers, Seel. *DIODEM – A Diverse Inertial and Optical Dataset of kinEmatic chain
Motion.* Scientific Data (2025), doi:10.1038/s41597-025-05468-w. Data: Harvard Dataverse doi:10.7910/DVN/SGJLZA,
version 3.0, **CC0 1.0** (also on Zenodo record 15511879 under CC BY 4.0). Cite both in anything that uses it.

### 2.1 What was downloaded

`python scripts/download_diodem.py` (2026-10-22) fetched the file list from the Dataverse API (version 3.0: 485 files,
2,738,355,798 bytes in all) and downloaded the part Step 1 needs:

- `dataset/`: the processed per-motion CSV files, 264 files, 525,812,448 bytes;
- `make_dataset/readme.md`, `make_dataset/preprocess/metadata.yaml`, `make_dataset/preprocess/setup.json`: the
  authors' notes.

Total: **267 files, every one checked against the MD5 that Dataverse publishes**. Size and SHA-256 are in
`manifests/diodem.csv`. Not downloaded: `videos/` (1.71 GB), `make_dataset/raw_data/` (raw logs), `cad_files/`,
`images/` (`--all` fetches them).

Two machine-specific obstacles were handled in `mesqsim/data/net.py`:

- Dataverse answers HTTP 403 to the default `python-requests` User-Agent, so the scripts send their own.
- Avast's Web Shield re-signs HTTPS on the collaborator's machine, so Python uses the Windows trust store
  (`truststore`). Integrity rests on the MD5 check either way.

### 2.2 Format (inspected)

- One folder per experiment and motion: `dataset/{arm,gait}/expNN/motionMM_<name>/`. Arm chain: exp01–05.
  Gait chain: exp06–11.
- Three CSV files per motion: `*_imu_rigid.csv`, `*_imu_nonrigid.csv` and `*_omc.csv`. Each has a first line
  `# sampling frequency: <Hz>`, a units line (`seconds/meters/radians/a.u.`), then a header row.
- IMU files: `seg{1..5}_{acc,gyr,mag}_{x,y,z}` at **40 Hz** in every file (check 11.2). Units: acc m/s² (median norm
  9.73–9.81 m/s² in exp01 and exp06), gyr rad/s, mag a.u.
- Optical files: four markers per segment (`seg{k}_marker{m}_{x,y,z}`, metres) and the segment orientation
  `seg{k}_quat_{w,x,y,z}`, **scalar first**. The rate is **30 Hz in exp01–03 and 120 Hz in exp04–11** (header scan
  over all 88 optical files). The loader converts to our `(x, y, z, w)` order with `conventions.wxyz_to_xyzw`.
- Optical quaternions are unit to 4.4e-16 over 1,199,524 samples (check 11.2).

### 2.3 Frames: verified from the data

Every number in this section and in §2.5 is printed by `python scripts/diodem_inspect.py` (exp01 arm, exp06 gait).

- **The optical world is +Z up.** The paper defines an east/north/up frame. Verified: the gravity direction from
  six-axis VQF and from the optical segment orientation agree with a mean cosine of 1.000.
- **The optical quaternion maps segment to earth** (paper: "orientation of the segment's body coordinate system
  relative to the earth reference coordinate system"). This matches our convention.
- **The rigid IMUs' axes are the segment axes** in the processed data. The same tilt agreement holds with no
  mounting rotation at all, and per-axis gyroscope rates correlate 0.92–0.995 with the optical body rates.
- **Chains** (paper Fig. 4 and Table 4, verified on exp01 and exp06): adjacent segments' nearest markers are 109–134
  mm apart, while non-adjacent pairs are ≥ 372 mm apart. The hinge pairs keep ≥ 96.8 % of their relative angular
  velocity variance on one axis.
  - arm: seg1–seg2 spherical, seg2–seg3 hinge, seg3–seg4 hinge, seg4–seg5 hinge;
  - gait: seg5–seg1 hinge, seg1–seg2 saddle, seg2–seg3 saddle, seg3–seg4 hinge.

### 2.4 Official loader

The authors publish a loader, PyPI `imt-diodem` (module `diodem`, version 1.1.11, MIT). It was read before use. It
downloads files on demand from a GitHub mirror (default) or Dataverse, without checksums, into
`DIODEM_CACHE_FOLDER`. We use it with `backend="dataverse"` and point `DIODEM_CACHE_FOLDER` at our verified copy, so
it never downloads anything itself (`mesqsim.data.diodem.load_trial`). It concatenates all motions of an experiment
and resamples every signal to one rate. At 40 Hz the IMU samples are unchanged and the optical data are interpolated.

### 2.5 The Step 1 sanity analysis (no claims)

`mesqsim.data.diodem.six_axis_relative_heading` runs VQF in six-axis mode (magnetometer dropped) on the rigid IMUs
of two adjacent segments. It compares their relative heading with the optical one. Figure:
[figures/diodem_relative_heading.png](figures/diodem_relative_heading.png).

Observation, recorded only (arm exp01, all 13 motions, 6.13 min; numbers from check 11.3 and
`scripts/diodem_inspect.py`):

- The optical relative heading of the hinge seg2→seg3 spans 17.3°.
- The six-axis estimate departs from it by up to 91.4° and ends at −80.3°. Most of this accrues in the first
  3 minutes, before the first "pause" motions.
- A least-squares fit of each gyroscope axis against the optical body rate shows constant offsets of up to about
  0.6 °/s (e.g. seg2: −0.54, −0.20, −0.64 °/s). That matches the drift rate: 0.5 °/s is 30° per minute. It is the
  residual-bias effect the brief's drift term b_i models.

## 3. Splits (owned by [mesqsim/splits.py](../mesqsim/splits.py))

- **By subject, within each subset.** Every sequence of one subject goes to one split: 70 % train, 15 % val, 15 % test
  by number of subjects, apportioned by largest remainder. A subset with 20 subjects gets 14/3/3; one with 3 gets
  2/1/0; nothing is lost to rounding.
- **Stable order.** Subjects are ordered by SHA-256 of `mesquite-step1-splits-v1:<subset>:<subject>`, not by a random
  draw. Shuffling the file list, or adding a new subset, does not move any existing subject (tested).
- **TotalCapture** is never trained on: all its subjects go to `holdout`, a possible later test set.
- **Windows and variants.** The dataset generator looks up each sequence's split in the manifest. Every window and
  every corruption variant of a sequence lands in that split (check 9.3).
- **The manifest.** `python -m mesqsim split --write` writes `manifests/split_manifest.csv` (columns `path` relative
  to `$MESQ_DATA_DIR/amass`, `subset`, `subject`, `split`; names only, no data) and records its SHA-256 in
  `manifests/split_manifest.sha256`.
- **Freezing.** Once the hash file exists, the manifest is frozen: `split --write` refuses to replace it with a
  different one (`--force` overrides, and needs Danyal's approval), and check 9.2 fails if the file changes.
- **Status:** written on 2026-11-13 by `python -m mesqsim split --write`; SHA-256
  `d36448adb76762251668b9940933e8a14df664ad6d3277fe3a930f7d55049fb3` (also checked with `sha256sum`). **Danyal
  approves and freezes it** before Step 2 results are read.

  | split | subjects | sequences | 20 s windows (≤ 3 per sequence) |
  |---|---|---|---|
  | train | 173 (BMLmovi 60, CMU 72, HumanEva 2, KIT 39) | 5,974 | 884 |
  | val | 37 (13, 15, 1, 8) | 1,104 | 110 |
  | test | 36 (13, 15, 0, 8) | 1,065 | 173 |
- **Provisional results without AMASS** use synthetic stand-in sequences (`dataset.synthetic_sources`: walking,
  turning in a circle, standing; four pseudo-subjects each), split by the same rules.

## 4. Real suit recordings

The frames-table contract (`frames.parquet`, `session.json`) that Danyal's logger and the simulator share is defined in
[PIPELINE.md](PIPELINE.md) (M3).
