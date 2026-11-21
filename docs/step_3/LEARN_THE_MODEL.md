# Learn the model: from `twin_v1` to a trained, scored TCN-P

This course takes you from opening the first data file to a trained TCN-P with its score on the validation split.
TCN-P is the network that estimates each pod's heading relative to the phone. You write four files:

| file | what it holds | its tests (in `tests/test_model_kit.py`) |
|---|---|---|
| `mesqsim/learn/loader.py` | `WindowDataset`: 18 s windows, the loss on the last 10 s, masks, one split only | `test_loader_windows`, `test_loader_windows_stay_in_one_session_of_the_split_and_are_causal`, `test_loader_masks_line_up_with_the_inputs` |
| `mesqsim/learn/models.py` | `TCN`, the causal network; later the wrapper that `score-model` loads | `test_parameter_counts_by_hand`, `test_receptive_field_and_causality`, `test_danyals_network_is_blind_to_a_common_turn`, `test_a_trained_tcn_p_follows_one_pod_turned_alone` |
| `mesqsim/learn/loss.py` | `cos_loss` and `von_mises_nll` | `test_danyals_cos_loss`, `test_danyals_von_mises_nll_matches_scipy` |
| `mesqsim/learn/train.py` | `train(model, train_loader, val_loader, cfg)` | `test_one_small_batch_can_be_overfitted_and_training_is_deterministic` |

**No finished code is written for you,** neither here nor anywhere else in the repository (Step 3 brief §7.5): you
learn by writing it. What you get instead:
- the specification ([DATASET_CARD.md](DATASET_CARD.md));
- the tests, which turn from "skipped" to pass or fail as each file appears;
- each lesson's idea, a worked example with real numbers, an exercise, hints and the usual mistakes;
- short snippets of a library's interface (`h5py`, `torch.nn.Conv1d`, `torch.utils.data.DataLoader`).

The worked examples (G1 to G8) are recomputed by
[tests/test_step3_guide_examples.py](../../tests/test_step3_guide_examples.py), so a number on this page that stops
matching the code fails the check.

Back to the [index](README.md). Read the [dataset card](DATASET_CARD.md) first (about 20 minutes).

## 0. Before you start

**Setup** (Danyal's Mac; the paths are relative to the repository folder):

```bash
source .venv/bin/activate
export MESQ_DATA_DIR=../mesquite_work/data MESQ_OUTPUT_DIR=../mesquite_work/outputs
export MESQ_TEST_DATASET="$MESQ_OUTPUT_DIR/datasets/twin_v1"   # the loader tests read it
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"   # 2.14.1 True here
python -m pytest tests/test_model_kit.py -q                   # 4 passed, 10 skipped: yours skip until they exist
```

**The rules you must not break:**
1. **Labels go through one door.** Your loader gets targets only from `mesqsim.learn.targets.read_targets`. Your model
   never imports `mesqsim.learn.targets` at all. `Cheat` shows what the scorer does to a model that does (CLAUDE.md
   rule 8).
2. **Causal.** The output at step k may use inputs up to step k only. The tests check it, and the scorer measures it.
3. **Train on `train`; decide on `val`.** Open `test` once, at the very end.
4. **Weights never enter git.** Save them under `$MESQ_OUTPUT_DIR/models/`; `*.pt` is already ignored.

**If you already have practice code** (the brief's one-week plan on a toy problem), start there:
1. Move each piece into the file above that matches it.
2. Rename to the signatures in the table.
3. Adapt the loader to `twin_v1`'s arrays (lesson 2).
4. Run the tests: what they reject is what still needs adapting.

### The two-day track

| when | lesson | about | done when |
|---|---|---|---|
| day 1, morning | 1. Open `twin_v1` | 1 h | your printout matches G1 |
| | 2. The loader | 3 h | the three loader tests pass |
| day 1, afternoon | 4. Causal convolutions | 1 h | you can write the 253 by hand |
| | 5. The network | 2 h | the parameter-count and causality tests pass |
| | 6. The output on a circle | 1.5 h | both loss tests pass |
| day 2, morning | 7. The training loop | 2 h, then a run of about 1 h | the overfit test passes; a model trains on the 1/4 part |
| day 2, afternoon | 8. Scoring | 2 h | `score-model` scores your TCN-P on all 74 validation sessions |
| | 9. TCN-P and the phone | 1 h | the turned-pod test passes |

After the two days come lessons 3, 10, 11 and 12, then TCN-J, the chain, three seeds and the curriculum (the last
section). The hours are a guess for someone who programs well and is new to PyTorch. Debugging is the biggest
unknown.

---

## Lesson 1. Open `twin_v1`

**Study first:** the h5py *Quick Start Guide* (<https://docs.h5py.org/en/stable/quick.html>); NumPy's *absolute basics
for beginners*.

**The idea.** A session is one long recording on a 32 Hz grid. Every array has T rows, one per step. The inputs
(`x_*`, `state_*`) are what the live stream would deliver. The targets (`y_*`) are the truth, with masks that say
where they mean something.

```text
step:      0 ............... 1065 .. 1225 ....................................................... 49684
motion:    preparation       T-pose  the program: still, walk, turn and more T-poses, joined by 16-step
           (box, straps)             transitions (356 runs of one motion code in all)
phone:     no message until step 459 (16.06 s), then mostly full messages
valid ε:   from step 458 on (the phone's Zero), except in transitions and for absent pods
```

**G1. One step of `val_KIT_12_v0`** (shard `shards/val_00072.h5`). The session is 49,684 steps, 25.9 min. It is the
old suit (`suit_2026_11_13`), on the JSON wire. `t_s` runs from 1.71875 s to 1554.3125 s in steps of 0.03125 s.
Motion steps: preparation 1,065, T-pose 1,440, still 18,420, walk 18,962, turn 6,965, transition 2,832.

At step 26,013 (t = 814.625 s), the subject is walking, and LeftLeg (pod 10) has just sent a fresh sample:

| `x_pod[26013, 10]` | tilt g | rate (rad/s) | age (s) | held |
|---|---|---|---|---|
| values | 0.6890, 0.2179, −0.6912 | 0.0496, −0.3476, −0.8534 | 0.0216 | 0 |

- The tilt is a unit vector (norm 1.0000), and the rate's norm is 0.923 rad/s.
- The targets there are δ of the left knee −113.013° and ε of LeftLeg 172.501°.
- Over the whole session the knee's δ slides from 0.53° (first valid step) to −68.87° (last), and LeftLeg's ε from
  −36.81° to 86.87°. That slide is the drift the model must follow.
- Pods 5, 8, 11 and 14 (the hands and feet) are lost at every step, because the old suit has none.

**Exercise.** Open the session with `h5py` (dataset card §5.5).
1. Print every array's shape and dtype.
2. Print `x_pod[26013, 10]` and compare it with G1.
3. Get the targets with `read_targets(group)`. Plot LeftLeg's tilt z (`x_pod[:, 10, 2]`) and the left knee's δ
   (`np.rad2deg(y_delta[:, 10])`, only where `valid_delta[:, 10]`) against `t_s`.
4. Print the lost fraction per pod: `(state_pod == 0).mean(axis=0)`.

**Check.** Your numbers match G1 (`tests/test_step3_guide_examples.py::test_G1_one_step_of_val_KIT_12_v0` recomputes
them).

**Hints and usual mistakes.**
- The angles are in radians; convert only to print.
- An invalid target is stored as 0. Plot it unmasked and you see flat zeros that are not data.
- `x_pod` for an absent pod is all zeros. That is not "a pod lying flat"; it is "no pod".

---

## Lesson 2. The loader

**Study first:** PyTorch's *Datasets & DataLoaders* and *Writing Custom Datasets*
(<https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html>); the Amidis' *A detailed example of how
to generate your data in parallel with PyTorch* (<https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel>).

**The idea.** The network sees 18 s at a time, a window of 576 steps. The loss counts only the window's last 10 s
(320 steps). In the first 8 s (256 steps), the network's memory of 253 steps fills up, so every scored step has its
full history.

```text
steps:    0           256                       576                       896                      1216
window 0  [ warm-up   |  loss region (scored)   )
window 1              [ warm-up   (320 to 576)  |  loss region            )
window 2                                        [ warm-up                 |  loss region            )
          the loss regions [256,576) [576,896) [896,1216) ... tile the session: each step is scored once
```

**One item, for TCN-P,** is one window with all 15 pods:
- `x`: (15, 37, 576), float32;
- `theta`: (15, 576), radians;
- `mask`: (15, 576), bool.

A pod's 37 inputs are its `x_pod` (8), the phone's `x_phone` (8), its `x_pod_phone` (6) and a one-hot of its index
(15). Lesson 9 says why. The pods share the network's weights. Your collate function flattens a batch of B items to
N = 15·B sequences: `x` (N, 37, 576), `theta` (N, 576), `mask` (N, 576). That is the convention `train` and the
tests use.

**G2. The windows of `val_KIT_12_v0`.** `mesqsim.learn.store.windows(49684)` gives 154 windows:
- The first three are [0, 576), [320, 896) and [640, 1216).
- The last ends at 49,536. The last 148 steps, and the first 256, are in no loss region.
- The validation split has 9,400 windows: 294 batches of 32.
- The mask of a window is (inside its last 320 steps) AND `valid_eps`.

**Exercise.** Write `WindowDataset(path, split, window_steps=576, loss_steps=320, seed=0)`, with:
- `windows`: a list of `(session_id, start, end)`, Python ints, end exclusive, one per item. The tests read it.
- `__len__` and `__getitem__(i)`, which returns the dict above.
- the sessions of `split` only, from `index.parquet`.
- the windows of `mesqsim.learn.store.windows(steps)` for each session (deterministic).
- `seed`, kept for any randomness you add later (draw it from `np.random.default_rng(seed)`).

Then a collate function and a `DataLoader`:

```python
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=4,
                    persistent_workers=True, generator=torch.Generator().manual_seed(0))
```

**Test** (`MESQ_TEST_DATASET` set, as in §0): `python -m pytest tests/test_model_kit.py -k loader -v`. Three tests:
- a fixed seed gives the same items;
- each window lies inside one session of the split and is 576 steps long;
- the mask has the window's time length, as `x` does, and is False on its first 256 steps.

**Hints.**
- **Read the window, not the session.** `g["x_pod"][start:end]` reads only the chunks it needs.
  `read_targets(g, start, end)` gives that window's targets the same way.
- **Open files in the worker, not in `__init__`.** An open `h5py.File` cannot be sent to a worker process (macOS
  starts workers by spawning). Keep a dict from shard path to file, filled the first time each worker opens one.
- **Build the 37 inputs with NumPy, on the window's arrays** (T = 576):
  - `x_pod` is (T, 15, 8) and `x_pod_phone` is (T, 15, 6);
  - `x_phone` (T, 8) becomes (T, 15, 8) with `np.broadcast_to`;
  - the one-hot is `np.eye(15)`, broadcast to (T, 15, 15);
  - `np.concatenate(..., axis=-1)` gives (T, 15, 37), and a transpose gives (15, 37, T).
  Put this in one function, because lesson 8's wrapper must build exactly the same inputs.
- **For the first runs, use one part.** The 1/4 part (129 sessions, dataset card D6) is enough for a first model;
  select it with `idx.part >= 4`.

**Usual mistakes.**
- A window that runs past its session's end, or one that mixes splits.
- Reading `y_eps` straight from `h5py` instead of through `read_targets`.
- A mask that is True in the warm-up, or that ignores `valid_eps`. You would train toward stored zeros.
- `theta` in degrees.

---

## Lesson 3. Why these features (later)

**Study first:** Shervine's *Linear Algebra and Calculus refresher*; 3Blue1Brown's *Essence of linear algebra*; Zhou et
al. 2019.

- **Tilt.** Gravity is the one direction every pod agrees on, so the tilt g = R̂ᵀe_z needs no north.
- **The rate.** It is measured between fresh samples in the pod's own frame. A constant north cancels from
  R̂(t₋)ᵀR̂(t).
- **The 6-D rotation.** The first two columns of a rotation matrix are continuous, where Euler angles and
  quaternions jump (Zhou et al. 2019).

So turning every pod and the phone by one heading changes no input. Check C5 proves it, and your network inherits it
(`test_danyals_network_is_blind_to_a_common_turn`). Skip this lesson on the two-day track; come back for the
report.

---

## Lesson 4. Causal convolutions

**Study first:** Shervine's CS 230 CNN cheatsheet (the receptive field, <https://stanford.edu/~shervine/teaching/cs-230/>);
DLS Course 4, module 1; Karpathy's *makemore Part 5* (WaveNet's hierarchy, without dilation); Bai et al. 2018.

**The idea.** A causal convolution at step t sees only steps ≤ t. With kernel k and dilation d, its taps are t,
t − d, …, t − (k − 1)·d.
- **Left padding.** Padding the input on the left by (k − 1)·d keeps the output as long as the input and never looks
  ahead. For k = 3 and d = 32 that pad is 64 steps.
- **Dilation.** Doubling d in each layer grows the memory exponentially with depth.

```text
layer 2 (d=2):  y2[t] = w0·y1[t-4] + w1·y1[t-2] + w2·y1[t]
layer 1 (d=1):  y1[t] = v0·x[t-2]  + v1·x[t-1]  + v2·x[t]
=> y2[t] depends on x[t-6] ... x[t]: 1 + (k-1)·(1 + 2) = 7 steps
```

**G3. Worked numbers.**
- **An impulse response.** Take a causal convolution with weights w = [1, 2, 3] (PyTorch order), d = 2 and no bias,
  on the input x = [1, 0, 0, 0, 0, 0, 0]. The output is [3, 0, 2, 0, 1, 0, 0]: the taps come out reversed, d apart,
  and nothing appears before the impulse.
- **The receptive field.** The network has six blocks of two convolutions, with k = 3 and d = 1, 2, 4, 8, 16, 32:
  1 + Σ_blocks 2·(k − 1)·d = 1 + 4·63 = 253 steps, which is 7.9 s at 32 Hz.

**Exercise.**
1. Write one causal convolution as a small `torch.nn.Module`: pad on the left, then convolve.
2. Feed it the impulse above and check the output.
3. Write the 253 out by hand.

```python
import torch.nn.functional as F
y = conv(F.pad(x, (left, 0)))      # conv = torch.nn.Conv1d(c_in, c_out, kernel_size=3, dilation=d)
```

**Test:** `test_receptive_field_and_causality`, after lesson 5. A change at step 400 or later must not change
outputs before step 400. A change at step 399 − 253 must not reach step 399, and one at step 399 − 252 must.

**Usual mistakes.**
- `padding="same"`, or symmetric padding, which looks into the future: the causality test fails.
- Padding by (k − 1), forgetting the factor d.

---

## Lesson 5. The network

**Study first:** DLS Course 4, module 2 (ResNets, 1×1 convolutions); He et al. 2016.

**The idea.** A residual block is two causal convolutions with a shortcut around them. The shortcut lets the block
learn a correction and keeps deep training stable. Where the channel count changes (the first block, 37 → 64),
the shortcut is a 1×1 convolution; elsewhere it is the identity.

```text
x ──┬─► causal conv (k=3, d) ─► ReLU ─► causal conv (k=3, d) ─► ReLU ──► (+) ─► ReLU ─► out
    └────────── 1×1 conv (first block only) or identity ────────────────┘

(N, 37, T) ─► block d=1 ─► d=2 ─► d=4 ─► d=8 ─► d=16 ─► d=32 ─► 1×1 head ─► (N, 3, T): cos, sin, log κ
```

**G4. The parameters, counted by hand.** A `Conv1d(c_in, c_out, k)` has c_in·c_out·k weights plus c_out biases.

| part | TCN-J (25 inputs) | TCN-P (37 inputs) |
|---|---|---|
| block 1, first convolution | 25·64·3 + 64 = 4,864 | 37·64·3 + 64 = 7,168 |
| block 1, second convolution | 64·64·3 + 64 = 12,352 | 12,352 |
| block 1, 1×1 shortcut | 25·64 + 64 = 1,664 | 37·64 + 64 = 2,432 |
| blocks 2-6, two convolutions each | 5 · 2 · 12,352 = 123,520 | 123,520 |
| head, 1×1 to 3 outputs | 64·3 + 3 = 195 | 195 |
| **total** | **142,595** | **145,667** |

**Exercise.** Write `TCN(in_features, channels=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32), out_features=3)`.
Its `forward(x)` takes (N, C_in, T) and returns (N, 3, T): the raw cos, sin and log κ. Count its parameters with
`sum(p.numel() for p in net.parameters())` before the test does.

**Tests:** `test_parameter_counts_by_hand`, `test_receptive_field_and_causality`, and
`test_danyals_network_is_blind_to_a_common_turn` (needs AMASS; it passes by itself, because the inputs are blind).

**Hints.**
- **No normalisation layers.** BatchNorm, LayerNorm and weight norm add parameters, so the count fails. Bai et al.
  use weight norm; this design does not.
- **Dropout is free** (no parameters), but leave it out at first.
- **`torch.nn.ModuleList`** holds the six blocks; build the dilations from the argument.
- **Clamp log κ** (the third channel) to a documented range, e.g. [−5, 12] (κ from 0.007 to 1.6·10⁵), with
  `torch.clamp`, in `forward` or in the wrapper of lesson 8. Not in the loss: its test goes to κ = 10⁷.

---

## Lesson 6. The output on a circle

**Study first:** Shervine's *Probabilities and Statistics refresher*; Beyer et al. 2015; Prokudin et al. 2018; Kendall
& Gal 2017; `torch.special` (<https://docs.pytorch.org/docs/stable/special.html>).

**The idea.** An angle is a point on a circle, not a number on a line. 179° and −179° are 2° apart, but their
squared difference says 358°. So the network outputs a point (c, s); after normalising it to unit length,
θ̂ = atan2(s, c).

```text
                 s
                 │    • (c, s) = (cos θ̂, sin θ̂)
                 │   /
                 │  / θ̂ − θ: the error, always in (−180°, 180°]
                 │ /    • (cos θ, sin θ): the target
        ─────────┼──────────── c
```

- **The mean first:** 1 − cos(θ̂ − θ) = 1 − (c·cos θ + s·sin θ), with (c, s) normalised. It is 0 when right and 2
  when opposite, and smooth everywhere.
- **Then the confidence.** The network also says how sure it is, κ ≥ 0, through the von Mises density
  p(θ) = exp(κ·cos(θ − θ̂)) / (2π·I₀(κ)). Its negative log-likelihood is
  NLL = −κ·cos(θ − θ̂) + log 2π + log I₀(κ).
  - For large κ, the spread is σ ≈ 1/√κ radians.
  - Confident and wrong is punished hard; unsure is charged a little everywhere.
- **The stable formula.** I₀(κ) overflows long before κ is large, so compute
  log I₀(κ) = log i0e(κ) + κ, where `torch.special.i0e` is the scaled Bessel function.

**G5. Worked numbers** (radians inside; the degrees are only for reading):

| error | κ (σ ≈ 1/√κ) | NLL |
|---|---|---|
| 0° | 50 (8.10°) | −1.0345 |
| 10° | 50 (8.10°) | −0.2749 |
| 10° | 5 (25.62°) | 0.2185 |
| 30° | 50 (8.10°) | 5.6642 |
| 0.5° | 10⁴ (0.57°) | −3.3054 |

- 1 − cos 10° = 0.01519.
- I₀(10⁴) is about 10^4340.5, far past float64's 1.8·10³⁰⁸. The naive formula returns inf; the stable one gives
  −3.3054.

**Exercise.** In `loss.py`, write two functions. Each returns the mean over the True entries of `mask`:
- `cos_loss(cos_sin, theta, mask)`: `cos_sin` (N, T, 2), `theta` (N, T) in radians, `mask` (N, T) bool;
- `von_mises_nll(cos_sin, log_kappa, theta, mask)`: the same, plus `log_kappa` (N, T, 1).

The network's (N, 3, T) becomes these with `out[:, :2].transpose(1, 2)` and `out[:, 2:3].transpose(1, 2)`.

**Tests:** `test_danyals_cos_loss` and `test_danyals_von_mises_nll_matches_scipy` (seven cases, κ up to 10⁷, in
float64).

**Hints and usual mistakes.**
- **Normalise (c, s) inside both losses,** dividing by its norm plus a small ε. The network's raw outputs are not
  unit vectors.
- **The masked mean** is (loss · mask).sum() / mask.sum(), with the denominator at least 1. A plain `.mean()` counts
  the masked-out steps.
- **Keep the input's dtype:** the test passes float64.
- **Do not clamp log κ in the loss** (lesson 5).
- **If you train κ from the start, it can collapse:** while the mean is still wrong, the cheapest answer is κ → 0.
  Train with `cos_loss` first (lesson 7).

---

## Lesson 7. The training loop

**Study first:** DLS Courses 1 and 2; Karpathy's micrograd and *makemore* Parts 2-3 (<https://karpathy.ai/zero-to-hero.html>);
PyTorch's *Learn the Basics* (<https://docs.pytorch.org/tutorials/beginner/basics/intro.html>); Loshchilov & Hutter
2019 (AdamW).

**The idea.**

```text
for each epoch:
    for each batch:  forward ─► loss (masked) ─► zero the gradients ─► backward ─► clip ─► optimiser step
    every so often:  the validation loss on a fixed set of windows (eval mode, no gradients)
                     keep the best weights; stop when it has not improved for a few checks
```

**The recommended settings** are in `configs/twin.yaml`, section `learning:`, as comments labelled `assumption`:

| setting | value |
|---|---|
| optimiser | AdamW at 1e-3, weight decay 0.01 |
| batch | 32 windows (× 15 pods = 480 sequences) |
| loss | `cos_loss` for the first 2 epochs, then `von_mises_nll` |
| gradient clipping | 1.0 |
| validation | every 200 batches, on 1,000 fixed validation windows, with patience 3 |
| seeds | 3 seeds (0, 1, 2) once the model works |
| device | `mps` on this Mac; the test runs on the CPU |

**G6. How long an epoch takes.** One epoch is one pass over a part's windows (dataset card D6):

| part | windows | batches of 32 | estimated time on the GPU (MPS) | on the CPU |
|---|---|---|---|---|
| 1/16 (30 sessions) | 2,394 | 75 | about 1 min | about 4 min |
| 1/4 (129 sessions) | 12,141 | 380 | about 6 min | about 19 min |
| all (519 sessions) | 54,153 | 1,693 | about 25 min | about 1.4 h |

These are estimates. One `Conv1d(64, 64, 3)`, forward and backward, on a batch of 480 sequences of 592 steps was
measured on this Mac (Apple M2 Pro, torch 2.14.1): 59.6 ms on the GPU and 198.6 ms on the CPU. The network has 12 such
layers. Adding about 25 % for everything else gives about 0.9 s per batch on the GPU and 3 s on the CPU. Reading the
data adds to it, so measure your own:

```python
import time, torch
conv = torch.nn.Conv1d(64, 64, 3, dilation=8).to("mps")
x = torch.randn(480, 64, 592, device="mps", requires_grad=True)
t0 = time.perf_counter(); conv(x).sum().backward(); torch.mps.synchronize(); print(time.perf_counter() - t0)
```

**Exercise.** Write `train(model, train_loader, val_loader, cfg)`:
- `cfg` has `epochs`, `lr` and `seed`, and may have your own keys, read with defaults (`cfg.get("cos_epochs",
  cfg["epochs"])` and so on).
- Each batch is a dict `x` (N, C, T), `theta` (N, T), `mask` (N, T).
- Return the history.
- Save the best `state_dict` under `$MESQ_OUTPUT_DIR/models/`.

Then train TCN-P:
1. Run on the 1/16 part, just to see the loss fall.
2. Run on the 1/4 part: the first real model.

**Test:** `test_one_small_batch_can_be_overfitted_and_training_is_deterministic`. With `{"epochs": 300, "lr": 3e-3,
"seed": 0}` on one tiny batch, the cos loss must fall by 90 %. Two runs must end with identical weights on the CPU.

**Hints.**
- **Determinism.**
  - Call `torch.manual_seed(cfg["seed"])` first thing in `train`.
  - Use the CPU unless `cfg` names a device.
  - Never shuffle without a seeded generator.
- **Train and eval modes.** `model.train()` while training, `model.eval()` and `torch.no_grad()` for validation.
- **Mean first, by default.** Let `cos_epochs` default to all epochs, so the test trains the mean only. Set it to 2
  for real runs.
- **Do not wait for whole epochs.** Watch the validation loss every 200 batches.
- **Run long jobs detached, and checkpoint.** Closing a terminal kills the jobs it started, so start long runs with
  `nohup … &` or `subprocess.Popen(..., start_new_session=True)`, and save the best weights as you go. Use 4 or 5 data
  workers: this Mac has 17 GB, and other apps use it too.

**Usual mistakes.**
- Forgetting `optimizer.zero_grad()`.
- Validating in train mode.
- A loss over masked-out steps.
- A learning rate 10× too high: the loss jumps about, then becomes NaN.
- Saving weights inside the repository.

---

## Lesson 8. Scoring

**Study first:** DLS Course 3 (mismatched training and test distributions); Guo et al. 2017.

**The idea.** The scorer does not know your network. It knows the contract (`mesqsim/learn/contract.py`):

```text
SessionInputs (one whole session, no labels) ─► your wrapper ─► Prediction:
    theta_rad (T, 15) wrapped to (−π, π]    kappa (T, 15) ≥ 0    uncertain (T, 15) bool
```

`python -m mesqsim score-model --model <module>:<Class>` constructs `Class()` without arguments, so the wrapper
loads its own weights, from a path in an environment variable or a default under `$MESQ_OUTPUT_DIR/models/`. The
scorer then does five things:
- checks every prediction (`check_prediction`);
- measures causality at three steps per session;
- scores every validation session;
- writes `summary.json`, `table.csv` (per pod and per stratum), `derived_joint_table.csv` and `reliability.csv`;
- writes them to `$MESQ_OUTPUT_DIR/scores/<module>-<Class>__twin_v1__val/`.

**G8. The numbers to compare with.**
- **On the validation split** (`score-model`, WORKLOG S3-M8):
  - a model that knows nothing about ε (`ZerosEps`) scores a median 91.51° (mean 92.76°, P95 174.40°);
  - on δ, B0 scores 22.65° and the oracle O2 0.00°.
- **On the test split** (`docs/RESULTS_STEP3.md`, to look at only at the end): ε per pod is B0 83.25°, RIGID+B1
  43.25°, RIGID+GEO 29.69°.
- **RIGID's numbers on the validation split** come from one run:
  `python scripts/make_results_step3.py --split val --no-ablations --no-page --no-figures --out <file>` (about 45 min).
  It writes `results_step3/twin_v1/baselines_val.csv`, which `score-model` then reads for gate G3.

**Exercise.**
1. Write the wrapper, a class with `target = "eps"` and `predict(inputs)`, in `models.py`.
2. Build the inputs with the same function as the loader.
3. Run the network on the whole session at once. It is causal, so its outputs are causal.
4. Turn (c, s, log κ) into the three arrays.
5. Score:

```bash
python -m mesqsim score-model --model mesqsim.learn.models:TCNP --dataset twin_v1 --split val --max-sessions 5
python -m mesqsim score-model --model mesqsim.learn.models:TCNP --dataset twin_v1 --split val   # all 74
```

**Hints.**
- **Wrapping.** `mesqsim.conventions.wrap_rad(np.arctan2(s, c))` meets the contract. `np.arctan2` alone can
  return −π exactly, which the contract rejects (rule 7: one convention, imported).
- **The confidence.** κ = exp(log κ). Set `uncertain` where κ is below a threshold you document, e.g. κ < 4 (σ about
  29°). The scorer's selective-risk table shows whether the flag helps.
- **Shapes.** The output is (T, 15) float64: transpose (15, 3, T) and convert.
- **Read the reliability table.** If your 68 % intervals hold far fewer than 68 % of the errors, κ̂ is
  overconfident. GEO-ID's was (WORKLOG S3-M10).
- **B1 stays in every table.** On the twin it equals B0, because the pods boot in the box, so an offset the box
  would remove is no learned improvement.

---

## Lesson 9. TCN-P and the phone

**Study first:** §7.3 of [the Step 3 brief](../brief/STEP3_BRIEF.md); Zhou et al. 2019.

**The idea.** Turn the whole room and every input stays the same (lesson 3), so no network can learn a pod's north
from the pod alone. Relative to the phone it can:
- `x_pod_phone` holds R̂_phᵀR̂_i = R_phᵀ·Y(−ε_i)·R_i, where R are the true orientations (noise aside);
- so ε_i sits between the two true orientations, and it enters the inputs nowhere else.

The one-hot tells the shared network which pod it is looking at.

```text
                phone (the anchor: ψ_ph)
             ┌───────┼────────┐
          HipsAlt  Spine ...  LeftLeg ...     each pod i: ε_i = wrap(ψ_i − ψ_ph), from its own 37 inputs
```

**G7. Worked numbers.**
- The inputs: 8 + 8 + 6 + 15 = 37.
- `x_pod_phone[26013, 10]` of `val_KIT_12_v0` is [0.1695, −0.6018, 0.7805, −0.9724, 0.0265, 0.2316]. That is two
  columns of a rotation matrix: each has unit length, and they are orthogonal (the test checks both to 10⁻⁵).
- The chain identity from the dataset card (D5) links the two models' targets at that step: ε_LeftLeg 172.501° =
  wrap(ε_LeftUpLeg −74.486° + δ_l_knee −113.013°).

**Exercise.** Train TCN-P on the 1/4 part (lesson 7), score it (lesson 8), then run the turned-pod test:

```bash
MESQ_TEST_TCNP=mesqsim.learn.models:TCNP python -m pytest tests/test_model_kit.py -k follows -v   # needs AMASS
```

**The test.** Turning pod 10's reports alone by γ = 30°, 90° and 180° must move that pod's ε̂ by −γ, within 2°
(median over the session's last half). A model that ignores `x_pod_phone` fails it.

**Then (not in two days):**
- **TCN-P-chain** needs TCN-J (lesson 5's network with 25 inputs, `target = "delta"`):
  ε̂ = RIGID's phone → HipsAlt link plus TCN-J's δ̂ added outward along the tree (`mesqsim/learn/rigid.py`).
- **H7** asks which wins where. Anchoring to the phone should pay most at the far end of the body (forearms, hands,
  shins, feet).

---

## After the two days

- **Lesson 10, is the data enough?** Study first: DLS Course 3; Hestness et al. 2017.
  - Train on the parts 1/16, 1/8, 1/4, 1/2 and 1, and plot the validation error against the hours.
  - `mesqsim/learn/curve.py` decides whether more data would still help (its rule: 0.1° and 2 %). If it would,
    `twin-dataset --extend` adds training draws.
- **Lesson 11, through the page.** Study first: §8.3 of the brief.
  - `score-model --through-page` draws the Nov 14 sessions again with your ε̂ applied, and measures the shins.
  - The page test shows that a perfect per-pod model fixes the drawing only when the phone was zeroed in the T-pose
    (`docs/RESULTS_STEP3.md`).
- **Lesson 12, debugging.** Study first: Karpathy's *A Recipe for Training Neural Networks*
  (<https://karpathy.github.io/2019/04/25/recipe/>); Shervine's CS 230 *Deep Learning Tips and Tricks*. Check, in
  order:
  1. shapes, printed at every stage;
  2. masks (how many True per batch?);
  3. units (radians everywhere);
  4. label leaks (`grep -n targets` in your model);
  5. causality leaks (the causality test, the scorer's `causality_error_rad`);
  6. a loss that will not fall (overfit one batch first; then lower the learning rate);
  7. a κ̂ that lies (`reliability.csv`).
- **Seeds and the curriculum.**
  - Three seeds give a spread, not a lucky run.
  - The proposal's curriculum: windows with drift but no slips first, then slips, restarts and transitions. Use
    `strata/slip` and `strata/transition` to choose them.
- **A model card.** For each trained model, record its data part, settings, seeds and scores, after Mitchell et al.
  2019. `MODEL_CARD_TEMPLATE.md` can be asked for.

## Self-check

1. **Why can TCN-P not learn a pod's north without the phone?** A common turn changes no input (check C5), so the
   inputs carry no absolute north; only the pod's rotation relative to the phone carries ε.
2. **Why are only the last 320 steps of a window scored?** The network remembers 253 steps. The 256 steps before the
   loss region give every scored step its full history.
3. **Why train with 1 − cos before the von Mises NLL?** While the mean is wrong, the NLL is cheapest with κ → 0, and
   the confidence head learns to give up.
4. **Why is log I₀(κ) computed as log i0e(κ) + κ?** I₀(κ) overflows float64 for κ above about 700. I₀(10⁴) is about
   10^4340.
5. **Your model's median ε on validation is 85°. Is that progress?** Barely: a model that knows nothing scores
   91.51°. Compare with RIGID+B1's validation numbers (lesson 8) before celebrating.
6. **Why must the loader and the wrapper share one input function?** If training and inference build the 37 inputs
   differently, the scored model is not the trained one. That bug passes every unit test.
