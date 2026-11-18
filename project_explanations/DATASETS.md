# The two datasets: AMASS and DIODEM

This project uses two public datasets. They do different jobs:

| | **AMASS** | **DIODEM** |
|---|---|---|
| what it is | recorded **human** motion: how every body joint rotated, frame by frame | **real IMUs** on two **mechanical** five-segment chains, with cameras giving the truth |
| what we use it for | the "real bodies" inside the simulator: every simulated session moves like one AMASS recording | a reality check: does the estimator work on real sensor data, not only on simulated data? |
| sensors in the data | none (it is a fitted body model; IMU errors are simulated by us) | 10 real 9-axis IMUs (accelerometer, gyroscope, magnetometer) |
| truth | the body model's joint rotations (exact by construction in our simulation) | optical motion capture (markers and cameras) |
| people | yes: 246 subjects in the four subsets we downloaded | no people: mechanical chains |
| licence | non-commercial research; **no redistribution** | CC0 1.0 (public domain) on Harvard Dataverse |
| where it lives | outside the repository: `$MESQ_DATA_DIR/amass/` (on Danyal's Mac: `../mesquite_work/data/amass/`) | `$MESQ_DATA_DIR/diodem/` |

Nothing from either dataset is ever committed to git: only scripts, instructions and **manifests** (file names,
sizes and SHA-256 checksums) are. The details of downloading, verifying and splitting are in
[docs/DATA.md](../docs/DATA.md); this page explains what the data *are*.

---

## 1. AMASS: human motion as body-model rotations

### 1.1 What it is

**AMASS** (*Archive of Motion Capture as Surface Shapes*; Mahmood, Ghorbani, Troje, Pons-Moll and Black, ICCV 2019,
<https://amass.is.tue.mpg.de>) is not one recording session. It **collects many existing optical motion-capture
datasets** from different labs and puts them all into one common format. The paper's abstract describes
15 marker-based motion-capture datasets, unified with a method called **MoSh++**, giving more than 40 hours of motion,
more than 300 subjects and more than 11,000 motions ([paper page](https://is.tue.mpg.de/publications/amass-iccv-2019),
[arXiv 1904.03278](https://arxiv.org/abs/1904.03278v1)).

### 1.2 What happened in the original recordings (the hardware)

Every AMASS subset was originally recorded with **optical, marker-based motion capture**: the person wears small
reflective markers on a tight suit, several infrared cameras around the room see the markers, and software
triangulates each marker's 3-D position many times per second. Different labs used different systems:

| subset | original dataset | capture setup (as its authors describe it) | what people did |
|---|---|---|---|
| **CMU** | CMU Graphics Lab Motion Capture Database | a Vicon optical system; its description names 12 Vicon infrared MX-40 cameras | a wide range of activities, from locomotion and sports to everyday scenarios and interactions |
| **KIT** | KIT Whole-Body Human Motion Database (Mandery et al., ICAR 2015; IEEE T-RO 2016) | Vicon optical capture with passive markers at 100 Hz (eight stationary and two portable cameras, 56 markers, in Mandery's description) | walking (straight, in circles, backwards), manipulation, balancing, many everyday tasks |
| **BMLmovi** | MoVi (Ghorbani et al., PLOS ONE 2021) | optical motion capture recorded together with **video** from 4 viewpoints and **IMUs**; 90 people (60 women, 30 men) | a predefined set of everyday actions and sports movements, plus one self-chosen movement |
| **HumanEva** | HumanEva-I/II (Sigal, Balan and Black, IJCV 2010) | marker-based (Vicon) motion capture synchronised with video cameras (7 in HumanEva-I) | walking, jogging, gesturing and other common actions (HumanEva-I: 4 subjects, 6 actions) |

Sources: [AMASS](https://is.tue.mpg.de/publications/amass-iccv-2019);
[CMU database description](https://www.roboticscenter.ai/datasets/cmu-mocap/);
[Mandery et al. 2015](https://h2t.iar.kit.edu/pdf/Mandery2015a.pdf);
[MoVi, arXiv 2003.01888](https://arxiv.org/pdf/2003.01888);
[HumanEva](https://ps.is.mpg.de/publications/sigal-ijcv-10b).

### 1.3 What MoSh++ turned the markers into

Raw marker positions are awkward to use: every lab placed its markers differently. MoSh++ fits a **body model** to the
markers instead. The model used here is **SMPL+H**: a skeleton of 52 joints (22 body joints plus the hands) with a
realistic body surface. For every frame of every recording, AMASS stores:

| key in each `*_poses.npz` file | shape | meaning |
|---|---|---|
| `poses` | (frames, 156) | 52 joints × 3 numbers: each joint's rotation relative to its parent, as an axis-angle vector (radians) |
| `trans` | (frames, 3) | where the body's root (the pelvis) is, in metres |
| `betas` | (16,) | the person's body shape |
| `mocap_framerate` | number | frames per second (120 Hz for CMU, BMLmovi and HumanEva; 100 Hz for KIT; some CMU files 60 Hz) |
| `gender` | text | which gendered body model fits the person |

So AMASS gives, frame by frame, **how each body segment was oriented in the room**. That is exactly what our
simulator needs: the suit's 15 pods sit on 15 of these segments (thigh, shin, forearm, …), and from the segment
orientations we compute where each pod pointed, then simulate what the pod's chip would have reported, with its
errors.

### 1.4 What we downloaded

On 2026-11-13, with our own AMASS account, four subsets in the **SMPL+H G** format (numbers from
`python scripts/verify_amass.py` and `python -m mesqsim split`, recorded in [docs/DATA.md](../docs/DATA.md)):

| subset | archive size | motion files | subjects | frame rate | median length |
|---|---|---|---|---|---|
| HumanEva | 33 MB | 28 | 3 | 120 Hz | 17.5 s |
| BMLmovi | 749 MB | 1,801 | 86 | 120 Hz | 4.6 s |
| CMU | 1.99 GB | 2,082 | 102 | 120 Hz (60 Hz for 495 files) | 9.6 s |
| KIT | 2.20 GB | 4,232 | 55 | 100 Hz | 6.1 s |
| **total** | 4.97 GB (11.71 GB unpacked) | **8,143** | **246** | | **23.2 hours** |

We also downloaded the **SMPL+H body model** (from <https://mano.is.tue.mpg.de>), which turns rotations into joint
positions; we need it only to draw figures.

### 1.5 What is important about AMASS for this project

- **It is motion, not sensor data.** AMASS has no IMU readings. The drift, noise, lost radio frames and reboots of the
  Mesquite suit are *simulated* on top of the AMASS motion (`mesqsim/sim/`). AMASS gives realistic bodies; the
  errors are our model, with every parameter labelled (most are still assumptions until the bench measures them).
- **The truth is exact.** Because the simulator knows each pod's true orientation, every estimate can be scored
  exactly — something no real recording can offer.
- **The subjects are split, and the split is sacred.** Subjects (not files) are divided 70/15/15 into train,
  validation and test: 173, 37 and 36 subjects. Anything *fitted* (the range-of-motion limits, the confidence
  calibration) uses train or validation only; results are reported on the test subjects. A person never appears in
  two splits, because a method could otherwise learn that person's own way of moving.
- **Most recordings are short.** Only 767 of the 8,143 sequences last at least 20 s (our window length); BMLmovi's
  short single actions are almost absent from the simulated sessions, which are mostly CMU and KIT. In the test
  split, 117 sequences are long enough, giving 519 simulated test sessions.
- **The body model's knee is not a perfect hinge.** SMPL knees have three rotational degrees of freedom, and the
  fitted motion uses them a little. This limits how well a "the knee is a hinge" method can work, even with perfect
  knowledge of the axes (GEO-K's errors in [docs/RESULTS_STEP2.md](../docs/RESULTS_STEP2.md)).
- **The licence forbids sharing.** Each person who uses AMASS must register and download it themselves. Nothing
  derived from it (simulated sessions included) may be redistributed or committed to the public repository.

---

## 2. DIODEM: real IMUs on mechanical chains

### 2.1 What it is

**DIODEM** (*A Diverse Inertial and Optical Dataset of kinEmatic chain Motion*; Bachhuber, Lehmann, Weygers and Seel,
*Scientific Data* 2025, doi:10.1038/s41597-025-05468-w; data: Harvard Dataverse doi:10.7910/DVN/SGJLZA, CC0 1.0)
records **real inertial sensors** moving on **mechanical five-segment chains**, together with **optical motion capture**
as ground truth. Its authors built it to let researchers study inertial motion tracking problems one at a time —
for example tracking without a magnetometer, with few sensors, with misaligned sensors, or with sensors that wobble
on soft attachments ([paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12287372/)).

### 2.2 What happened in the recordings (the hardware)

- **Two chains.** Each is five rigid segments joined by mechanical joints:
  - the **arm** chain: three hinges (1-degree-of-freedom joints) and one spherical joint;
  - the **gait** chain: two hinges and two saddle (2-degree-of-freedom) joints.

  Which segment pairs form the hinges was verified on the data (docs/DATA.md §2.3): arm seg2–seg3, seg3–seg4,
  seg4–seg5; gait seg5–seg1 and seg3–seg4.
- **Ten IMUs**: two per segment, **Xsens MTw Awinda** wireless sensors, each with a 3-axis accelerometer, gyroscope
  and magnetometer, sampled at **40 Hz**. On each segment one IMU is fixed **rigidly** on a 3-D-printed mount and the
  other is attached **non-rigidly**, on foam, to create realistic motion artefacts.
- **Optical truth**: an **OptiTrack Prime 22** system with **12 cameras**; four reflective markers on each segment
  (20 in all). The paper gives 30 Hz; the processed files we downloaded state 30 Hz for experiments 1–3 and 120 Hz
  for experiments 4–11 in their headers. From the four markers the authors build each segment's orientation; the IMU
  and optical clocks were aligned by cross-correlating the angular velocities each system measured.
- **Eleven experiments**, 46 minutes in all: experiments 1–5 on the arm chain, 6–11 on the gait chain, each made of
  several motions (fast, slow, pauses, …).

### 2.3 What we downloaded

`python scripts/download_diodem.py` (2026-10-22): the processed per-motion CSV files (264 files, 526 MB) and the
authors' notes — 267 files, every one checked against the checksum Dataverse publishes. Each motion has three files:
the rigid IMUs, the non-rigid IMUs, and the optical data (marker positions and segment orientation quaternions,
stored scalar-first, which our loader converts to our (x, y, z, w) order).

### 2.4 How we use it, and what is important

- **We use only the six-axis part, on purpose.** The Mesquite chip fuses accelerometer and gyroscope only, with no
  magnetometer, so its heading drifts. We run the same kind of fusion (VQF in six-axis mode) on DIODEM's rigid IMUs,
  so their headings drift like the suit's: on the arm chain's first hinge in experiment 1, a perfect registration at
  the start is off by about 80° after 6 minutes. The optical truth says what the relative heading really was.
- **It tests the estimator on real sensors.** GEO-ID runs unchanged on two DIODEM IMUs as if they were the thigh
  and shin pods (a hinge). DIODEM has no box and no T-pose, so the estimator starts with no information and must find
  the hinge axis and the relative heading from the motion alone. The results per experiment are Table 10 of
  [docs/RESULTS_STEP2.md](../docs/RESULTS_STEP2.md).
- **It is out of distribution.** These are machines, not people: the joints are near-perfect hinges, the motions are
  not human, and the experiment (not the sample) is the unit. DIODEM can show that the method works on real sensor
  data; it cannot show how well it works on a person.
- **Only the hinges are covered.** The saddle joints would need a 2-degree-of-freedom axis identification (not built),
  and the spherical joint a range of motion learnt for these chains (no data to learn it from).

---

## 3. The two together, in one picture

```text
   AMASS (people, many labs, optical mocap → SMPL+H rotations)          DIODEM (machines, real IMUs + optical truth)
                      │                                                                     │
                      ▼                                                                     ▼
   our simulator adds the suit's errors (drift, noise, loss, reboots)     six-axis fusion (VQF), no magnetometer
                      │                                                                     │
                      ▼                                                                     ▼
   thousands of simulated sessions, exact truth, split by subject         a few real recordings, optical truth
                      │                                                                     │
                      └──────────────► the estimators (B1, GEO-K, GEO-ID, …) ◄──────────────┘
                                       scored against the truth
```

AMASS answers "how well could this work on many people, under our model of the suit's errors?"; DIODEM answers
"does it survive real sensor data at all?". Neither replaces recordings of the real Mesquite suit on people, which is
the next step once the bench measurements exist.
