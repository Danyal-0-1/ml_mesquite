# Lesson 2, built by you: the loader for TCN-P

Prepared on 2026-11-22 against `main` at `14cc1fe`, with `twin_v1` present on this Mac. It expands Lesson 2 of
[LEARN_THE_MODEL.md](../step_3/LEARN_THE_MODEL.md) in the style of
[MOTION_SIMULATION_TUTORIAL.md](MOTION_SIMULATION_TUTORIAL.md): first the idea and the mathematics, then the code,
function by function.

**What you get here, and what you do not.** The loader is yours to write. The course and the Step 3 brief (§7.5)
promise no finished loader, and CLAUDE.md rule 14 keeps it that way. For every function you get:

1. what it is for, and its contract (signature, shapes, dtypes, units);
2. pseudocode;
3. the mathematics behind it;
4. a line-by-line build plan: each pseudocode line, the Python it needs (shown on a toy, not on your arrays), and why;
5. the conventions and style choices, with their reasons;
6. a check against real numbers from `twin_v1`;
7. drills to master it once it passes.

When your file passes its tests, ask for a line-by-line review of `mesqsim/learn/loader.py`.

Every number below was computed from `twin_v1` or measured on this Mac (Apple M2 Pro; torch 2.14.1, h5py 3.16.0,
NumPy 2.4.6). Toy examples are marked as toys. Section 12 lists how each number was obtained.

**Where your code goes.** The kit's tests import `mesqsim.learn.loader`
([tests/test_model_kit.py:135](../../tests/test_model_kit.py#L135)). The `loader.py` at the repository root holds
your Lesson 1 script, and no test imports it. Keep that script (rename it if you like, for example
`lesson1_open_twin.py`) and create `mesqsim/learn/loader.py` for this lesson. The same goes later for `models.py`,
`loss.py` and `train.py`: the tests look for them in `mesqsim/learn/`.

## How to work through this lesson

Use the same loop for every function. The habit matters more than the function:

```text
contract ─► pseudocode ─► predict a number on paper ─► write the test (it fails: red)
        ─► write the code (it passes: green) ─► clean it up (refactor) ─► explain it aloud, line by line
```

| step | what | time |
|---|---|---|
| 1 | Read §1 to §3 with a pencil. Predict each worked number before you read it | 40 min |
| 2 | Type the skeleton (§4.0). The kit's loader tests now fail instead of skipping | 10 min |
| 3 | `pod_inputs` and `loss_mask`, test first, on synthetic arrays (§4.1, §4.2, §5) | 40 min |
| 4 | `WindowDataset.__init__`. Check 9,400 and 12,141 in a REPL (§4.3) | 25 min |
| 5 | The file cache and `__getitem__`. Check window 80's golden values (§4.4, §4.5) | 35 min |
| 6 | `collate` and the `DataLoader`. Check determinism and speed (§4.6, §4.7) | 30 min |
| 7 | The three kit tests and ruff, then the drills (§8) | 30 min, plus drills |

That is about 3.5 hours; the course budgets 3. The order matters more than the minutes. The pure functions come
first because they need no data and no worker processes, so their bugs are cheap to find.

---

## 1. The picture

```text
index.parquet (665 sessions) ──keep one split (and maybe one part)──► the sessions of the split
     │
     └─ for each session: store.windows(steps) ──► self.windows = [(session_id, start, end), ...]
                                                    built in __init__; no HDF5 file opened (val: 9,400 items)

the DataLoader, every epoch:
  RandomSampler: π = randperm(N, generator) ─► BatchSampler: 32 indices per batch
        │
        ▼     (each worker is a spawned process holding a pickled copy of the dataset)
  worker k:  [ds[i] for i in batch] ─► collate(items) ─► tensors in shared memory ─► main process ─► train()

ds[i]:  (sid, s, e) = windows[i]
        g     = the session's HDF5 group (its shard opened once per worker, then kept open)
        x     = pod_inputs(x_pod[s:e], x_phone[s:e], x_pod_phone[s:e])   (15, 37, 576) float32
        y     = read_targets(g, s, e)                                     the one door for labels
        theta = y["y_eps"] transposed                                     (15, 576) float32, rad
        mask  = loss_mask(y["valid_eps"], 320)                            (15, 576) bool
```

**Why a map-style dataset.** PyTorch has two kinds of dataset (the `torch.utils.data` documentation, "Dataset
Types"). A *map-style* dataset has `__len__` and `__getitem__(i)`: random access by index, so a sampler decides the
order and a shuffle is exact. An *iterable-style* dataset yields items in its own order; it is meant for streams that
cannot be indexed. Our windows form a fixed, countable list, so map-style is the right kind. In Fluent Python's terms
(chapter 1's `FrenchDeck`, chapter 12), a map-style dataset is simply an object that follows the sequence protocol.

**Why not load everything into memory.** One training step of `twin_v1` holds about 1.4 KB of arrays, so the
training split is about 24 GB once decompressed ([DATASET_CARD.md §5.5](../step_3/DATASET_CARD.md)). This Mac has
17 GB. So each item is read from disk when it is needed.

---

## 2. The contract

**One item** (`ds[i]`), for TCN-P:

| key | shape | dtype | units and meaning |
|---|---|---|---|
| `x` | (15, 37, 576) | float32 | pod p's 37 inputs at each window step: channels 0-7 its `x_pod`, 8-15 the phone's `x_phone`, 16-21 its `x_pod_phone`, 22-36 the one-hot of p |
| `theta` | (15, 576) | float32 | ε of each pod in radians, from `read_targets`' `y_eps` |
| `mask` | (15, 576) | bool | True where the loss counts: the window's last 320 steps, where `valid_eps` |

**One batch** (`collate` of B items) holds torch tensors: `x` (N, 37, 576), `theta` (N, 576), `mask` (N, 576), with
N = 15·B. Sequence n is pod p of item b, where n = 15·b + p.

**The invariants.** These are the properties your code must keep. Each one becomes a test.

- **I1.** Every window lies inside one session of the split: 0 ≤ start < end ≤ steps, and end − start = 576.
- **I2.** The windows are a deterministic function of `index.parquet`, the split and the part. No randomness.
- **I3.** The mask is False on local steps 0 to 255. On steps 256 to 575 it equals `valid_eps`.
- **I4.** `x`, `theta` and `mask` have the same time length.
- **I5.** `theta` is in radians; `x` and `theta` are float32; `mask` is bool.
- **I6.** Labels enter only through `read_targets` (CLAUDE.md rule 8).
- **I7.** Item i is a pure function of i: the same bytes in any process, in any order.
- **I8.** In a batch, sequence n is pod n mod 15 of item ⌊n/15⌋.

The kit's three loader tests check I1, I3, I4 and part of I7
([tests/test_model_kit.py:145-173](../../tests/test_model_kit.py#L145-L173)). Your own tests (§5) check the rest.

---

## 3. The mathematics

### 3.1 Windows that tile a session

| symbol | meaning | value |
|---|---|---|
| $T$ | the session's length in steps; its steps are $0, \dots, T-1$ | 49,684 for `val_KIT_12_v0` |
| $W$ | the window's length | 576 steps (18 s) |
| $L$ | the loss region's length: the window's last $L$ steps | 320 steps (10 s) |
| $H = W - L$ | the warm-up | 256 steps (8 s) |
| $R$ | the network's receptive field (Lesson 4) | 253 steps |

The reference rule is `mesqsim.learn.store.windows` ([mesqsim/learn/store.py:395](../../mesqsim/learn/store.py#L395)):

```python
ends = np.arange(WINDOW_STEPS, steps + 1, LOSS_STEPS)
return np.stack([ends - WINDOW_STEPS, ends], axis=1)
```

So window $j$ ends at $e_j = W + jL$ and starts $W$ steps earlier:

$$
\text{window } j = [\,jL,\; jL + W\,),\qquad j = 0, 1, \dots, n(T) - 1.
$$

**How many windows.** Window $j$ exists when it ends inside the session: $W + jL \le T$, that is
$j \le (T - W)/L$. The integers $j = 0, \dots, \lfloor (T - W)/L \rfloor$ qualify, so

$$
n(T) = \begin{cases}
\left\lfloor \dfrac{T - W}{L} \right\rfloor + 1 & \text{if } T \ge W, \\[2mm]
0 & \text{if } T < W.
\end{cases}
$$

For `val_KIT_12_v0`: $(49{,}684 - 576)/320 = 49{,}108/320 = 153.46$, so $n = 153 + 1 = 154$.

**The loss regions tile.** Window $j$'s loss region is its last $L$ steps:

$$
[\,e_j - L,\; e_j\,) = [\,H + jL,\; H + (j+1)L\,).
$$

Region $j$ ends at $H + (j+1)L$, which is exactly where region $j+1$ starts. Consecutive regions touch and never
overlap, so their union is the single interval $[H, H + nL)$. Every step in it is scored exactly once per epoch.

**The inverse map.** Global step $g \ge H$ is scored by window $j = \lfloor (g - H)/L \rfloor$ (when $j < n$), at local
step $k = g - jL$. For G1's step 26,013: $j = \lfloor 25{,}757/320 \rfloor = 80$, window $[25{,}600, 26{,}176)$,
local step $k = 413$. Section 4.5 uses this window as its golden example.

**What is never scored.** The first $H = 256$ steps, and the last

$$
T - e_{n-1} = (T - W) - (n-1)L = (T - W) - L\left\lfloor \frac{T - W}{L} \right\rfloor = (T - W) \bmod L
$$

steps. For `val_KIT_12_v0`: $49{,}108 \bmod 320 = 148$. So 154 × 320 = 49,280 of the session's 49,684 steps are
scored: 99.2 %.

**The price of a full history.** Windows $j$ and $j+1$ share $W - L = 256$ steps: the warm-up of window $j+1$ is the
last 256 steps of window $j$. Away from the ends, a step lies in one or two windows, $W/L = 1.8$ on average. So the
network computes 1.8 steps for every scored step: 55.6 % of its work is scored. That is the price of giving every
scored step its full history (§3.2).

**Worked numbers (G2), recomputed.**

| sessions | count | windows | batches of 32 | last batch |
|---|---|---|---|---|
| val | 74 | 9,400 | 294 | 24 |
| train, 1/16 part | 30 | 2,394 | 75 | 26 |
| train, 1/4 part | 129 | 12,141 | 380 | 13 |
| train, all | 519 | 54,153 | 1,693 | 9 |
| test (open it once, at the end) | 72 | 7,734 | 242 | 22 |

Short sessions: $T$ = 575 gives 0 windows; 576 gives 1; 895 gives 1; 896 gives 2; 897 gives 2.

*Predict before reading on:* how many windows does a 1,000-step session have, and which of its steps are scored?
(The answer is in §9.)

### 3.2 Why 256 steps of warm-up are enough

A causal network with receptive field $R$ computes its output at step $t$ from the inputs $x_{t-R+1}, \dots, x_t$,
which is $x_{t-252}, \dots, x_t$ for $R = 253$. Inside a window, the network pads on the left with zeros (Lesson 4).
An output at a local step $t < R - 1 = 252$ therefore reads some padding zeros: it is not the output the deployed
network would give on the live stream.

The first scored local step is $H = 256 \ge R - 1 = 252$, with 4 steps to spare. So for every scored local step
$k \in [H, W)$, at global step $g = s + k$,

$$
y^{\text{window}}_k = f(x_{g-252}, \dots, x_g) = y^{\text{session}}_g .
$$

The network's output on the window equals its output on the whole session at every scored step. This is exact, not
approximate. It is why training on windows and running Lesson 8's wrapper on whole sessions agree. It holds because
the network is causal, has a finite receptive field and has no layer that mixes steps across the whole window.
BatchNorm in training mode normalises over the batch and all time steps, so it would break this equality; Lesson 5
leaves normalisation layers out.

If you ever change $W$ or $L$, the condition to keep is $W - L \ge R - 1$.

### 3.3 The mask

For pod $p$ and local step $k$ of a window that starts at step $s$, with $v$ the stored `valid_eps`:

$$
m_{p,k} = \mathbf{1}[\,k \ge W - L\,] \;\wedge\; v_{s+k,\,p}.
$$

A cell is excluded for either of two independent reasons:
- **an incomplete history** (the warm-up, §3.2);
- **a meaningless target**: the pod is absent or off, the phone has not been zeroed yet, or the step is inside a
  transition. The first two are stored as 0 ([DATASET_CARD.md §5.2](../step_3/DATASET_CARD.md)). Inside a
  transition the stored value is the blend's, a plausible angle that no real motion produced: in the validation
  split, 2,258,768 of the 8,369,798 invalid ε cells are nonzero, all of them in transitions.

Lesson 6's losses take the masked mean

$$
\mathcal{L} = \frac{\sum_{n,t} m_{n,t}\,\ell_{n,t}}{\max\left(1, \sum_{n,t} m_{n,t}\right)} .
$$

Without the second factor, every invalid cell would pull the network toward a target that means nothing (a 0 or a
blend). In an old-suit session, 4
of the 15 pods (5, 8, 11 and 14: the hands and feet) are absent at every step.

**Worked numbers** (`val_KIT_12_v0`, through `read_targets`):

| window | True cells of 15 × 576 = 8,640 | why |
|---|---|---|
| 0, steps [0, 576) | 1,298 (15.0 %) | ε becomes valid at step 458, the phone's Zero, so each of the 11 present pods has 118 valid steps (458 to 575) |
| 80, steps [25,600, 26,176) | 3,168 (36.7 %) | 288 per present pod: the loss region's 320 steps minus two 16-step transitions (loss-region steps 34-49 and 236-251) |
| all 9,400 validation windows | 37,059,405 of 81,216,000 (45.63 %) | 82.14 % of the loss-region cells; on average 3,942.5 True cells per item |

**A counting trick.** To count the valid steps in a loss region without a loop, use a prefix sum:
$c_t = \sum_{u < t} v_u$, so the count over $[a, b)$ is $c_b - c_a$. That is O(1) per window after O(T) work per
session. The 37,059,405 above was computed this way.

### 3.4 The 37 inputs: concatenation, broadcasting and the one-hot

For pod $p$ at step $t$, the input vector stacks four blocks:

$$
z_{t,p} = \big[\; x^{\text{pod}}_{t,p} \;;\; x^{\text{phone}}_{t} \;;\; x^{\text{pod-phone}}_{t,p} \;;\; e_p \;\big]
\in \mathbb{R}^{8+8+6+15} = \mathbb{R}^{37}.
$$

| channels | block | contents ([DATASET_CARD.md §5.2](../step_3/DATASET_CARD.md)) |
|---|---|---|
| 0-7 | `x_pod[t, p]` | tilt (3, a unit vector), rate between fresh samples (3, rad/s), age (1, s), held (1) |
| 8-15 | `x_phone[t]` | the phone's eight, with "orientation only" in place of held; the same for every pod |
| 16-21 | `x_pod_phone[t, p]` | the first two columns of $\hat R_{\rm ph}^\top \hat R_p$, the pod seen from the phone |
| 22-36 | $e_p$ | the one-hot of the pod's index: 1 at channel 22 + p |

The output layout is channels-first, because `Conv1d` expects (N, C, T): $X_{p,c,t} = z_{t,p,c}$. This is a
permutation of the axes, $(t, p, c) \to (p, c, t)$.

**Broadcasting** (NumPy's rule). Compare the shapes from the last axis backwards. Two sizes are compatible when they
are equal or one of them is 1; a missing axis counts as 1. A size-1 axis is stretched with stride 0, so nothing is
copied.
- `x_phone` (T, 8) gets a pod axis: (T, 1, 8). Broadcast to (T, 15, 8), each pod sees the same 8 numbers.
- The identity $I_{15}$ is (15, 15). Broadcast to (T, 15, 15), each step sees the same one-hot rows.
- `np.concatenate` along the last axis needs every other axis to agree: (T, 15) everywhere. It makes the one copy.

**Type promotion.** `np.concatenate` of float32 and float64 arrays returns float64 (NumPy's promotion rules, NEP 50).
`np.eye` returns float64 unless you ask for float32. A float64 `x` reaches the network and fails with
`RuntimeError: Input type (double) and bias type (float) should be the same`.

**The one-hot is an embedding.** For any matrix $A \in \mathbb{R}^{64 \times 15}$, $A e_p = A_{:,p}$, its column $p$.
So appending $e_p$ to the inputs of a linear layer adds a learned vector per pod, which is exactly what
`torch.nn.Embedding(15, 64)` does. In block 1's first causal convolution ($k = 3$, $d = 1$, weights
$W \in \mathbb{R}^{64 \times 37 \times 3}$), the one-hot channels contribute

$$
\beta_p = \sum_{\tau=0}^{2} W_{:,\,22+p,\,\tau} \quad \text{at every step } t \ge 2,
$$

a per-pod bias, constant in time. It costs 15·64·3 = 2,880 of that convolution's 7,168 parameters, and 15·64 = 960
of the 1×1 shortcut's 2,432. The fifteen pods share every other weight, which gives 15 times more training sequences
for the same parameters, and the one-hot lets the network specialise cheaply. Lesson 9 says why `x_pod_phone` is the
input that carries ε.

**One function for training and scoring.** Lesson 8's wrapper must build exactly these inputs for a whole session.
If training and scoring build them differently, the scored model is not the trained one, and that bug passes every
unit test ([LEARN_THE_MODEL.md](../step_3/LEARN_THE_MODEL.md), self-check 6). Google's *Rules of Machine Learning*
(rule 32) says the same: re-use code between the training pipeline and the serving pipeline whenever possible.

### 3.5 Batching: stacking and flattening

Stacking B items gives $X_{b,p,c,t}$, of shape (B, 15, 37, T). Merging the first two axes in row-major (C) order
gives the flat index

$$
n = 15\,b + p, \qquad (b, p) = \operatorname{divmod}(n, 15).
$$

The pods are the fast index, because in C order the last of the merged axes varies fastest. To regroup the
network's outputs later, `out.unflatten(0, (B, 15))` undoes the merge.

| array | per item | per batch of 32 |
|---|---|---|
| `x` | 15·37·576·4 = 1,278,720 bytes | |
| `theta` | 15·576·4 = 34,560 bytes | |
| `mask` | 15·576·1 = 8,640 bytes | |
| total | 1,321,920 bytes | 42,301,440 bytes (42.3 MB), N = 480 sequences |

### 3.6 Shuffling and the effective batch size

Each epoch the sampler draws a uniformly random permutation $\pi$ of $\{0, \dots, N-1\}$, and batch $k$ is
$\{\pi(32k), \dots, \pi(32k+31)\}$. That is sampling without replacement: every window once per epoch.

**Why shuffle.** Neighbouring windows depend strongly on each other: they share 256 of their 576 steps, the subject,
the suit, the norths and the slips. Take the mean of $B$ gradient terms that each have variance $\sigma^2$ and an
average pairwise correlation $\bar\rho$:

$$
\operatorname{Var}(\bar g) = \frac{1}{B^2}\Big(B\sigma^2 + B(B-1)\bar\rho\,\sigma^2\Big)
= \frac{\sigma^2}{B}\big(1 + (B-1)\bar\rho\big),
\qquad B_{\rm eff} = \frac{B}{1 + (B-1)\bar\rho}.
$$

An illustration, not a measurement: if 32 consecutive windows of one session had $\bar\rho = 0.5$, the batch would
be worth $B_{\rm eff} = 32/16.5 \approx 1.9$ independent windows. Windows drawn from the whole split have
$\bar\rho$ near 0, so $B_{\rm eff}$ is near 32. This is Kish's design effect from survey sampling.

### 3.7 Reading from HDF5, and the throughput budget

**What a slice reads.** `g["x_pod"][s:e]` asks HDF5 for rows s to e − 1, and HDF5 reads and decompresses only the
chunks those rows touch. The chunks hold 2,048 steps ([mesqsim/learn/store.py:174](../../mesqsim/learn/store.py#L174));
one chunk of `x_pod` is 2,048·15·8·4 = 983,040 bytes. A window touches two chunks when it straddles a chunk
boundary, that is, when $s \bmod 2048 > 2048 - 576 = 1472$. The reference starts are multiples of 320, and
$\gcd(320, 2048) = 64$, so $s \bmod 2048$ runs evenly through the 32 multiples of 64 below 2,048. Eight of them
exceed 1,472, so exactly 25 % of windows read two chunks per array (39 of `val_KIT_12_v0`'s 154). An item reads
seven arrays this way: three inputs, plus the four that `read_targets` returns.

**What it costs** (60 random validation windows, the OS file cache warm):

| operation | median | 90th percentile |
|---|---|---|
| open the shard, read one window's 3 inputs and its targets, close it | 7.04 ms | 12.95 ms |
| the same, with the shard kept open | 6.76 ms | 12.65 ms |
| open and close a shard, nothing read | 0.05 ms | |
| read a whole session's same arrays (10 sessions) | 157.7 ms | |
| the NumPy work of one item: concatenate, transpose, copy (synthetic arrays) | 0.29 ms | |
| stack and flatten 32 items | 1.57 ms | |

**The budget.** G6 estimates about 0.9 s per batch on this Mac's GPU, so the loader must deliver at least
32/0.9 ≈ 36 windows per second.
- At about 7.3 ms per window, one worker delivers about 137 windows per second. Four workers, if they scale, deliver
  about 550: roughly 15 times what is needed.
- Reading whole sessions instead costs 158 ms per item: about 6.3 windows per second per worker, 25 with four. That
  is below 36, so the GPU would sit idle almost 30 % of the time.

So reading the window rather than the session decides whether the GPU is fed. Keeping files open buys almost no
speed here (0.05 ms); it is there for correctness (§4.4). All of this is an estimate until you measure your own
loader (drill 4.7.2).

---

## 4. The build, function by function

Write them in this order: the pure functions (4.1, 4.2) and their tests first, then the dataset (4.3 to 4.5), then
`collate` and the `DataLoader` (4.6, 4.7). Bernhardt calls this design *functional core, imperative shell*: the
computation lives in pure functions that are easy to test, and a thin layer does the I/O.

### 4.0 The module's skeleton

Type this first, run the kit's tests, and watch them fail (red) instead of skipping. It contains only the contract:
imports, types and stubs. Every body is yours.

```python
"""TCN-P's training windows from ``twin_v1`` (Lesson 2 of docs/step_3/LEARN_THE_MODEL.md).

One item is one window of one session with all 15 pods: ``x`` (15, 37, W) float32, ``theta`` (15, W) float32 in
radians, ``mask`` (15, W) bool. :func:`collate` turns B items into N = 15·B sequences; sequence n is pod n % 15 of
item n // 15. Labels come only from :func:`mesqsim.learn.targets.read_targets` (CLAUDE.md rule 8).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, TypedDict

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..body import N_PODS
from .store import LOSS_STEPS, PARTS, WINDOW_STEPS, windows
from .targets import read_targets

__all__ = ["SPLITS", "Window", "Item", "pod_inputs", "loss_mask", "WindowDataset", "collate"]

SPLITS = ("train", "val", "test")


class Window(NamedTuple):
    """Item i's place: the session, and its steps ``start`` to ``end`` (end exclusive), as Python ints."""

    session_id: str
    start: int
    end: int


class Item(TypedDict):
    x: np.ndarray  # (15, 37, W) float32
    theta: np.ndarray  # (15, W) float32, rad
    mask: np.ndarray  # (15, W) bool


def pod_inputs(x_pod: np.ndarray, x_phone: np.ndarray, x_pod_phone: np.ndarray) -> np.ndarray:
    """The docstring of §4.1."""
    raise NotImplementedError


def loss_mask(valid: np.ndarray, loss_steps: int) -> np.ndarray:
    """The docstring of §4.2."""
    raise NotImplementedError


class WindowDataset(Dataset):
    """The docstring of §4.3."""

    def __init__(self, path: str | Path, split: str, window_steps: int = WINDOW_STEPS, loss_steps: int = LOSS_STEPS,
                 seed: int = 0, *, part: int | None = None) -> None:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, i: int) -> Item:
        raise NotImplementedError


def collate(batch: list[Item]) -> dict[str, torch.Tensor]:
    """The docstring of §4.6."""
    raise NotImplementedError
```

**Line by line.**
- **The module docstring** states the contract in three sentences. It is the first thing a reader (or you, in a
  month) needs.
- **`from __future__ import annotations`** makes annotations lazy strings, as in every module of this repository.
- **The imports come in three groups,** standard library, third party, then this package, each sorted (ruff's `I`
  rule). Inside `mesqsim` the repository uses relative imports, as `targets.py` does (`from ..body import N_PODS`).
- **`from .targets import read_targets`** is the only label import allowed in a loader (rule 8). Your *model* will
  never import `targets` at all.
- **`__all__`** lists the public names: what others may rely on, and what `from … import *` exports.
- **`SPLITS`** is a module constant (UPPER_CASE), so the valid splits are written once.
- **`Window(NamedTuple)`** is a typed record that is still a tuple (Fluent Python, chapter 5). The kit's test unpacks
  it (`for sid, start, end in a.windows`), and you can read `w.session_id` instead of `w[0]`.
- **`Item(TypedDict)`** documents the dict's keys and types for a type checker. At run time it is a plain `dict`, so
  the `DataLoader` and the tests see a dict (chapter 15).
- **Subclassing `Dataset`** is not required (the `DataLoader` only calls `len` and `[]`), but it states the intent and
  gives you `ds_a + ds_b` (a `ConcatDataset`).
- **The defaults `WINDOW_STEPS` and `LOSS_STEPS`** equal the course's 576 and 320, without magic numbers.
- **The bare `*` before `part`** makes it keyword-only: you must write `part=4`, so it cannot be passed by position by
  mistake. `part` is an addition to the course's signature that you will want for the first runs (the 1/4 part).
- **`raise NotImplementedError`** in each stub turns the kit's tests from "skipped" to "failed". That is the red step
  of test-driven development.

**Check it.**
`MESQ_TEST_DATASET="$MESQ_OUTPUT_DIR/datasets/twin_v1" python -m pytest tests/test_model_kit.py -k loader -v` now
reports 3 failures with `NotImplementedError`. While the bodies are stubs, `python -m ruff check` also flags the
imports they do not use yet (rule F401). That is expected in the red step, and it clears as you fill the bodies in.

### 4.1 `pod_inputs`: the 37 inputs, one function for training and scoring

**What we need.** The network wants a (37, T) sequence for each pod. The loader needs it for a 576-step window;
Lesson 8's wrapper needs it for a whole session. One function serves both (§3.4).

**Contract.**

```python
def pod_inputs(x_pod: np.ndarray, x_phone: np.ndarray, x_pod_phone: np.ndarray) -> np.ndarray:
    """TCN-P's inputs, one sequence per pod: (15, 37, T) float32, C-contiguous, for any T.

    ``x_pod`` (T, 15, 8), ``x_phone`` (T, 8) and ``x_pod_phone`` (T, 15, 6) as stored in ``twin_v1``: a window's
    slice or a whole session. Pod p's channels: 0-7 its ``x_pod``, 8-15 ``x_phone``, 16-21 its ``x_pod_phone``,
    22-36 the one-hot of p.
    """
```

**Pseudocode.**

```text
pod_inputs(x_pod, x_phone, x_pod_phone):
1  T, P ← the first two sizes of x_pod
2  require P = N_PODS, x_pod (T, P, 8), x_phone (T, 8), x_pod_phone (T, P, 6); else raise ValueError naming all three
3  phone  ← x_phone with a pod axis inserted, broadcast to (T, P, 8)                # a view: no copy
4  onehot ← the P × P identity in float32, broadcast to (T, P, P)                   # a view: no copy
5  z      ← concatenate [x_pod, phone, x_pod_phone, onehot] along the last axis     # (T, P, 37): the one copy
6  return z with its axes ordered (pod, channel, time), C-contiguous, float32       # (P, 37, T)
```

**The mathematics.** Lines 3 to 5 compute $z_{t,p}$ of §3.4, and line 6 is the permutation
$X_{p,c,t} = z_{t,p,c}$. Line 2 is cheap insurance: `x_pod` and `x_pod_phone` differ only in their last size, so if
you swap them by accident, the shape check catches it, instead of a model that trains on nonsense.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | unpack `.shape`: `t, p, _ = a.shape` | read sizes from the data, so the function works for any T |
| 2 | compare shapes; `raise ValueError(f"...")` with the actual shapes in the message | fail at the boundary, and say what came and what was expected |
| 3 | `a[:, None, :]` inserts an axis; `np.broadcast_to(view, shape)` stretches it | stride 0: fifteen "copies" for free |
| 4 | `np.eye(n, dtype=np.float32)`, then `np.broadcast_to` | the dtype keeps the result float32 (§3.4) |
| 5 | `np.concatenate([...], axis=-1)` | `axis=-1` means "the last axis", whatever the rank |
| 6 | `.transpose(1, 2, 0)`, then `np.ascontiguousarray` | `transpose` only permutes strides; the copy gives a dense array for torch |

The toy session below shows lines 3 to 6. It uses a toy with T = 4 steps, 3 pods, 2 phone values and its own
features of width 2, not your arrays:

```python
>>> import numpy as np
>>> a = np.arange(8, dtype=np.float32).reshape(4, 2)     # toy "phone": 4 steps, 2 values
>>> a[:, None, :].shape                                   # insert a pod axis of length 1
(4, 1, 2)
>>> b = np.broadcast_to(a[:, None, :], (4, 3, 2))         # 3 toy pods see the same 2 values
>>> b.strides, b.flags.writeable                          # stride 0 on the pod axis; a read-only view
((8, 0, 4), False)
>>> b[2].tolist()                                         # step 2, every pod
[[4.0, 5.0], [4.0, 5.0], [4.0, 5.0]]
>>> e = np.eye(3, dtype=np.float32)                       # row p is the one-hot e_p
>>> e[1].tolist(), np.broadcast_to(e, (4, 3, 3)).shape
([0.0, 1.0, 0.0], (4, 3, 3))
>>> z = np.concatenate([np.zeros((4, 3, 2), np.float32), b, np.broadcast_to(e, (4, 3, 3))], axis=-1)
>>> z.shape, z.dtype, z[2, 1].tolist()                    # step 2, pod 1: [own 2 | phone 2 | one-hot 3]
((4, 3, 7), dtype('float32'), [0.0, 0.0, 4.0, 5.0, 0.0, 1.0, 0.0])
>>> zt = z.transpose(1, 2, 0)                             # (t, p, c) -> (p, c, t): permuted strides, no copy
>>> zt.shape, zt.flags.c_contiguous, np.ascontiguousarray(zt).flags.c_contiguous
((3, 7, 4), False, True)
>>> bool((zt[1, :, 2] == z[2, 1]).all())                  # X[p, c, t] == z[t, p, c]
True
>>> x32 = np.zeros((4, 15, 8), np.float32)                # the dtype trap of §3.4
>>> np.concatenate([x32, np.broadcast_to(np.eye(15), (4, 15, 15))], axis=-1).dtype
dtype('float64')
```

**Style.**
- **Keep it pure:** no file, no torch, no global state. Lesson 8's wrapper and your tests can call it without data.
- **Check the pod count against `N_PODS`** ([mesqsim/body.py:67](../../mesqsim/body.py#L67)) instead of writing 15.
  You may also name the width, for example `N_INPUTS = 8 + 8 + 6 + N_PODS`, for the tests and Lesson 5.
- **Shape comments** such as `# (T, 15, 8)` after an array line are the one kind of "what" comment that pays for
  itself in array code.

**Check it.** The test of §5.1 passes. On real data, window 80's golden values do too (§4.5).

**Master it.**
1. Tomorrow, rewrite it from a blank file in 10 minutes without notes, and run the test.
2. Prove $A e_p = A_{:,p}$ on paper, then check it in NumPy for a random 64 × 15 matrix.
3. Write TCN-J's 25-input builder for one joint (parent pod 8, child pod 8, `x_joint` 6, joint-type one-hot 3), and
   compare it with `_tcn_j_input` in [tests/test_model_kit.py:209](../../tests/test_model_kit.py#L209). TCN-J will
   need it after the two days.
4. Explain why the phone's 8 numbers are repeated for every pod instead of fed once. (The network sees one pod's
   sequence at a time, so each sequence must carry the anchor.)

### 4.2 `loss_mask`: where the loss counts

**What we need.** The cells of §3.3, for a window of any length.

**Contract.**

```python
def loss_mask(valid: np.ndarray, loss_steps: int) -> np.ndarray:
    """Where the loss counts: (n, T) bool, True on the last ``loss_steps`` of the T steps, where ``valid`` (T, n)."""
```

**Pseudocode.**

```text
loss_mask(valid, L):
1  require valid to be a 2-D bool array; else raise ValueError
2  T ← the number of rows of valid
3  require 0 < L ≤ T; else raise ValueError
4  in_loss ← for k = 0 … T − 1: (k ≥ T − L)                  # (T,) bool
5  return valid transposed AND in_loss, made C-contiguous     # (n, T): in_loss broadcasts over the rows
```

**The mathematics.** $m_{p,k} = \mathbf{1}[k \ge T - L] \wedge v_{k,p}$. The broadcasting in line 5 aligns the last
axes. A (15, T) array with a (T,) array works. A (T, 15) array with a (T,) array compares T with 15 and fails.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | `valid.ndim`, `valid.dtype != bool` and `raise ValueError` | `y_eps` and `valid_eps` are both (T, 15): only the dtype tells them apart. A uint8 array would pass `&` silently and give a uint8 mask, and `.T` leaves a 1-D array as it is |
| 2 | `valid.shape[0]` | the window's length, from the data |
| 3 | a chained comparison, `not 0 < L <= T`, and `raise ValueError` | L = 0 gives an all-False mask (a zero loss); L > T scores the warm-up. Both are silent otherwise |
| 4 | `np.arange(T) >= T - L` | a vectorised comparison; no loop |
| 5 | `.T`, `&` and `np.ascontiguousarray` | `&` on bool arrays is element-wise AND and keeps bool; it states the intent better than `*`. A ufunc lays its output out like its inputs, so `valid.T & in_loss` is F-ordered; the copy makes it C-ordered, like `x` and `theta` |

```python
>>> np.arange(6) >= 4                                         # toy: T = 6, L = 2
array([False, False, False, False,  True,  True])
>>> v = np.array([[True, False], [True, True], [False, True]])  # toy valid: 3 steps, 2 pods
>>> v.T & (np.arange(3) >= 1)                                 # (2, 3) & (3,): the time mask broadcasts over pods
array([[False,  True, False],
       [False,  True,  True]])
>>> v.T & np.arange(3) >= 1                                   # the precedence trap: read as (v.T & arange) >= 1
array([[False,  True, False],
       [False,  True, False]])
>>> np.zeros((576, 15), bool) & np.zeros(576, bool)           # the wrong axis order
ValueError: operands could not be broadcast together with shapes (576,15) (576,)
```

**The precedence trap.** `&` binds tighter than `>=`. Without parentheses, the loader's sizes give
`(valid.T & k) >= 256`, where `valid.T & k` is 0 or 1: an all-False mask, a zero loss and no error. The kit's mask
test still passes, because it checks only the warm-up. Name the comparison (`in_loss`), and test both halves of the
mask.

**Style.** Take T from the data, as in 4.1. Name the argument `valid`, not `valid_eps`: the same function serves
TCN-J's `valid_delta` later.

**Check it.** Through `__getitem__` (§4.5): window 0 of `val_KIT_12_v0` has 1,298 True cells, window 80 has 3,168.

**Master it.**
1. Before running anything, derive 1,298 = 11 × 118 and 3,168 = 11 × 288 from §3.3's reasons.
2. Write a Hypothesis property test: for random T ≥ L and random `valid`, the first T − L columns are all False and
   the rest equal `valid.T`. (The repository already uses Hypothesis, for example in `tests/test_conventions.py`.)
3. Recompute 37,059,405 for the validation split with the prefix-sum trick of §3.3.

### 4.3 `WindowDataset.__init__`: the index of items

**What we need.** The list of items, decided once from the metadata: which sessions (the split, maybe a part), and
which windows of each. No HDF5 file is opened here.

**Contract.**

```python
class WindowDataset(Dataset):
    """TCN-P's training windows of one split of a ``twin_v1``-style dataset (Lesson 2).

    Item i is the window ``windows[i] = (session_id, start, end)`` of the reference rule
    (:func:`mesqsim.learn.store.windows`) with all 15 pods: ``x`` (15, 37, W) float32, ``theta`` (15, W) float32 rad,
    ``mask`` (15, W) bool, True on the last ``loss_steps`` steps where ``valid_eps``. ``part`` (train only) keeps the
    nested part 1/part. The shards open lazily, once per process. ``seed`` is kept for randomness added later.
    """

    def __init__(self, path: str | Path, split: str, window_steps: int = WINDOW_STEPS, loss_steps: int = LOSS_STEPS,
                 seed: int = 0, *, part: int | None = None) -> None:
```

**Pseudocode.**

```text
__init__(path, split, W, L, seed, part):
1   root ← path as an absolute Path (resolved once, so a later change of directory cannot break it)
2   require split ∈ SPLITS
3   require (W, L) = (WINDOW_STEPS, LOSS_STEPS)          # store.windows implements the reference rule only
4   require part is None, or (split = train and part ∈ PARTS)
5   idx ← the columns id, split, part, shard and steps of root / "index.parquet"
6   rows ← the rows of idx whose split is split (and whose part ≥ part, if a part is given)
7   shard_of ← {session id → root / shard} for those rows
8   windows ← [Window(id, start, end) for each row, for each (start, end) of store.windows(row's steps)], Python ints
9   require windows is not empty
10  keep L (for the mask), the seed, and an empty cache {shard path → open file}
```

**The mathematics.** The item list is the disjoint union over the split's sessions $S$:

$$
\mathcal{W} = \bigcup_{s \in S} \{\, (s, a, b) : (a, b) \in \texttt{windows}(T_s) \,\}, \qquad
N = |\mathcal{W}| = \sum_{s \in S} n(T_s).
$$

- **I1 holds by construction.** Line 6 keeps only the split's sessions. In `store.windows`, every end satisfies
  $e \le T$ (the stop of `np.arange` is $T + 1$, exclusive), and every start $e - W \ge 0$ because $e \ge W$.
- **I2 holds** because every line is a pure function of the file and the arguments: no random number is drawn, and
  pandas' boolean filtering keeps the file's row order.
- **The parts are nested sets,** $P_{16} \subset P_8 \subset P_4 \subset P_2 \subset P_1$. The index stores, for each
  training subject, the smallest part that contains it ($p(s)$, from `store.nested_parts`), so $s \in P_q \iff p(s)
  \ge q$. The filter is therefore `part >= q`. With `==`, you would get the ring $P_q \setminus P_{2q}$, which for
  q = 4 has 66 sessions and 6,174 windows instead of 129 and 12,141, and no error. For q = 16 both rules give 30
  sessions, so only a test at a larger part tells them apart.
- **The cost.** Looking up item i is O(1) list indexing. The 54,153 training windows take about 8 MB (measured with
  `tracemalloc`). PyTorch's `ConcatDataset` stores only cumulative sizes and finds an item with `bisect` in
  O(log S): better for millions of windows, not needed here.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | `pathlib.Path(path).resolve()`; `/` joins paths | accepts a `str` or a `Path`; no string surgery. The worker processes and every later lookup see the same absolute path, whatever the current directory |
| 2-4 | `if … not in …: raise ValueError(f"… {x!r}")` | `!r` prints the repr, so `'Val'` versus `'val'` is visible |
| 5 | `pd.read_parquet(path, columns=[...])` | Parquet stores columns separately, so naming the five you use reads only those, and the code then states what it depends on |
| 6 | boolean masks, with each comparison in parentheses | `&` binds more tightly than `==` and `>=` (toy below) |
| 7 | a dict comprehension | session to file in O(1) |
| 8 | a nested list comprehension over `rows.itertuples(index=False)`; `.tolist()` | `.tolist()` turns NumPy int64 into Python int |
| 9 | `if not windows: raise ValueError(...)` | an empty dataset is a bug upstream: say so now, not an hour into training |
| 10 | plain attributes; an empty dict for the cache | an empty cache pickles; an open file does not (§4.4) |

```python
>>> from pathlib import Path
>>> Path("twin_v1") / "shards" / "val_00072.h5"
PosixPath('twin_v1/shards/val_00072.h5')
>>> import pandas as pd
>>> df = pd.DataFrame({"split": ["train", "val"], "part": [4, 0]})        # toy index
>>> df[df.split == "train" & df.part >= 4]                                # the precedence trap
TypeError: Cannot perform 'rand_' with a dtyped [int64] array and scalar of type [bool]
>>> len(df[(df.split == "train") & (df.part >= 4)])
1
>>> next(df.itertuples(index=False))                                      # rows as named tuples: r.split, r.part
Pandas(split='train', part=4)
>>> [(name, k) for name, n in [("a", 2), ("b", 1)] for k in range(n)]     # the for clauses nest left to right
[('a', 0), ('a', 1), ('b', 0)]
>>> w = np.array([[0, 576], [320, 896]])                                  # the kind of array store.windows returns
>>> type(w[0, 0]).__name__, type(w.tolist()[0][0]).__name__
('int64', 'int')
```

**Why Python ints.** A NumPy int64 behaves like an int until something serializes it: `json.dumps` raises
`TypeError: Object of type int64 is not JSON serializable`. Type hints convert nothing: a `Window` built from int64
values keeps int64 values (Fluent Python, chapter 8: hints are not checked at run time). The kit's tests ask for
Python ints.

**Style.**
- **Do the cheap metadata work eagerly and the expensive I/O lazily.** A wrong split or part fails at construction.
- **Keep the session order of `index.parquet`.** It is the generation order, grouped by shard. A validation pass
  that does not shuffle then reads consecutive windows from the same file and the same chunks.
- **Store the seed, but create no random generator yet** (§4.5 explains where randomness belongs). You are not going
  to need it now.
- **`part` works on training sessions only:** validation and test sessions all have part 0. Refuse it elsewhere
  rather than return an empty dataset.

**Check it.** With `ds = WindowDataset(os.environ["MESQ_TEST_DATASET"], "val")`, you should see:
- `len(ds)` is 9,400;
- if you keep the index's order, `ds.windows[0]` is
  `Window(session_id='val_BMLmovi_Subject_11_F_MoSh_v0', start=0, end=576)` and `ds.windows[1]` starts at 320;
- `len(WindowDataset(root, "train", part=4))` is 12,141, and with `part=16` it is 2,394.

**Master it.**
1. Compute n(T) by hand for T = 575, 576, 895, 896 and 897, then check with `store.windows`.
2. Generalize: write `window_bounds(steps, window_steps, loss_steps)`. Property-test it with Hypothesis: it equals
   `store.windows(steps)` for (576, 320), and its loss regions tile for any W > L > 0. Then decide, and write in the
   docstring, whether `WindowDataset` should accept other sizes, and under which condition (§3.2).
3. Read `ConcatDataset` in `torch/utils/data/dataset.py`. Rebuild your index with cumulative sums and
   `bisect.bisect_right`, and compare its memory and speed with the list.

### 4.4 The file cache: `_group`, `__getstate__` and `close`

**What we need.** `__getitem__` needs a session's HDF5 group. On macOS, every `DataLoader` worker is *spawned*: a new
Python process that receives the dataset by `pickle`. An open h5py file cannot be pickled
(`TypeError: h5py objects cannot be pickled`). So each process must open the files itself, and the dataset must
pickle without them.

**Contract.**

```python
def _group(self, session_id: str) -> h5py.Group:
    """The session's group, ``/sessions/<id>/``. Its shard is opened on first use in this process and kept open."""

def __getstate__(self) -> dict:
    """What pickle sends to a worker: everything except the open files."""

def close(self) -> None:
    """Close every file this process opened."""
```

**Pseudocode.**

```text
_group(session_id):
1  shard ← shard_of[session_id]
2  f ← the cached file of shard, or nothing
3  if nothing: f ← open shard read-only, and cache it
4  return the group sessions/session_id of f

__getstate__():
5  state ← a shallow copy of the instance's attribute dict
6  replace the cache in state with a new empty dict
7  return state

close():
8  close every cached file, then empty the cache
```

**The mathematics.** This is memoization: a function from shard path to open file, computed once per process, so a
process holds at most one handle per shard (10 for the validation split, 65 for all of train). Under spawn, pickle
calls `__getstate__` in the parent and sends that dict; the worker rebuilds the object from it, so every worker starts
with an empty cache.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | a dict lookup | built in `__init__` |
| 2-3 | `dict.get(key)` returns `None` when absent; then `h5py.File(path, "r")` | read-only: many processes may read one file at once |
| 4 | `f["sessions"][session_id]` (or `f[f"sessions/{session_id}"]`) | a `Group`; nothing is read until you slice a dataset |
| 5-7 | `self.__dict__.copy()` | a shallow copy: the original object keeps its own files |
| 8 | a loop over the dict's values, then `.clear()` | for tests, notebooks and long scripts |

The pattern on a toy that keeps open text files, just as the dataset keeps shards:

```python
>>> import pickle
>>> class Lines:
...     def __init__(self, paths):
...         self.paths = list(paths)
...         self._open = {}                          # path -> open file, filled lazily, per process
...     def _file(self, path):
...         f = self._open.get(path)                 # None the first time in this process
...         if f is None:
...             f = self._open[path] = open(path)    # open once and remember it
...         return f
>>> r = Lines(["pyproject.toml"]); _ = r._file("pyproject.toml")
>>> pickle.dumps(r)
TypeError: cannot pickle '_io.TextIOWrapper' object
>>> def getstate(self):                              # in your class this is the method __getstate__
...     state = self.__dict__.copy()
...     state["_open"] = {}
...     return state
>>> Lines.__getstate__ = getstate
>>> len(pickle.loads(pickle.dumps(r))._open), len(r._open)   # the copy has no open file; the original keeps its own
(0, 1)
```

**Why `__getstate__` matters even with lazy opening.** The kit's tests call `a[0]` in the main process, and that
opens a shard there. If the same object is later handed to a `DataLoader` with workers, it is pickled with an open
file inside, and the workers fail to start. `__getstate__` makes the object safe to send at any moment.

**Why not open in `__init__`, and why not reopen for every item?**
- Opening in `__init__` fails under spawn (pickling). Under fork, which Linux uses by default, the children would
  inherit the parent's open HDF5 handles, which HDF5 does not support safely.
- Reopening for every item works, and here it costs only 0.05 ms. But it rereads the file's metadata and throws
  away HDF5's chunk cache each time. One handle per shard per process is the standard pattern and the course's hint.

**Style.**
- `dict.get` with a `None` check ("look before you leap") is clear here; `try`/`except KeyError` ("easier to ask
  forgiveness") is just as Pythonic (Fluent Python, chapter 3).
- Do not use `functools.lru_cache` on a method to cache files. It holds a strong reference to every instance, and its
  cache lives outside the object, so `close()` cannot clear it.
- Do not close files in `__del__`: when it runs is not guaranteed. Worker processes close their files when they
  exit. If you like, add `__enter__` and `__exit__` so that `with WindowDataset(...) as ds:` closes them (chapter 18).

**Check it.** After `ds[0]`, `pickle.loads(pickle.dumps(ds))` works, and the copy's cache is empty.

**Master it.**
1. Turn that check into a test (§5.2).
2. Make the cache safe under fork too: store `os.getpid()` next to it, and start a new empty cache when the pid has
   changed. Explain why Linux needs this and macOS does not.
3. Read `_worker_loop` in `torch/utils/data/_utils/worker.py` (torch 2.14.1). Find where the worker's seed is set
   (lines 275-282) and where your dataset arrives.

### 4.5 `__len__`, `__getitem__` and `__repr__`

**What we need.** The sequence protocol, `len(ds)` and `ds[i]`, which is all the `DataLoader` calls. Add a
`__repr__` that tells you which dataset you hold when you meet it in a debugger.

**Contract.**

```python
def __len__(self) -> int:
    """The number of windows."""

def __getitem__(self, i: int) -> Item:
    """Window i: ``x`` (15, 37, W) float32, ``theta`` (15, W) float32 rad, ``mask`` (15, W) bool."""

def __repr__(self) -> str:
    """For example ``WindowDataset(split='val', part=None, sessions=74, windows=9400)``."""
```

**Pseudocode.**

```text
__len__():
1  return the number of windows

__getitem__(i):
2  session_id, start, end ← windows[i]                             # an out-of-range i raises IndexError: keep it
3  g ← _group(session_id)
4  x ← pod_inputs(the rows start … end − 1 of g's x_pod, x_phone and x_pod_phone)    # (15, 37, W)
5  y ← read_targets(g, start, end)                                 # the one door for labels (rule 8)
6  theta ← y's y_eps transposed to (pods, time), C-contiguous      # (15, W) float32, radians
7  mask ← loss_mask(y's valid_eps, L)                              # (15, W) bool
8  return {"x": x, "theta": theta, "mask": mask}

__repr__():
9  return "WindowDataset(split=…, part=…, sessions=…, windows=…)"
```

**The mathematics.** The windows are half-open intervals: $[a, b)$ has $b - a$ elements, and $[a, b) \cup [b, c) =
[a, c)$ without overlap. The tiling of §3.1 and Python's slicing use the same convention; Dijkstra argues for it in
EWD 831. Line 4 reads rows `start` to `end − 1`, touching one or two chunks per array (§3.7). Lines 6 and 7 give
$\theta_{p,k} = y_{\text{start}+k,\,p}$ and the mask of §3.3. I7 holds because every line is a pure function of the
file's contents and i.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | `len()` of the list | `len(ds)` calls `__len__` (Fluent Python, chapter 1) |
| 2 | tuple unpacking: `name, a, b = ("s", 0, 5)` | a `Window` is a tuple. Let the `IndexError` propagate: `for item in ds` stops on it, through the iteration fallback of chapter 17 |
| 3 | your `_group` | |
| 4 | h5py slicing, `dataset[a:b]`, which returns a NumPy array | reads only the chunks it needs. Never read `[:]` and slice afterwards |
| 5 | `read_targets(group, start, end)` ([mesqsim/learn/targets.py:36](../../mesqsim/learn/targets.py#L36)) | the labels' only door |
| 6 | `.T`, then `np.ascontiguousarray` | (W, 15) to (15, W), dense |
| 7 | your `loss_mask` | |
| 8 | a dict literal with the three keys | the keys are the contract; `Item` documents them |
| 9 | an f-string: `f"{type(self).__name__}(split={self.split!r}, …)"` | `type(self).__name__` stays right in a subclass (chapter 11) |

**A trap in line 4.** An h5py slice past a dataset's end does not fail: it returns fewer rows. `x_pod[49500:50076]`
of `val_KIT_12_v0` gives 184 rows, not 576. A wrong window therefore fails later, in `collate`. That is why I1 is
tested on its own.

**Where randomness will go.** When you add randomness, for example a random window phase or noise for augmentation,
draw it inside `__getitem__` from a generator seeded by the seed and the index:
`rng = np.random.default_rng([self.seed, i])`. The same (seed, i) gives the same draws in any worker and in any
order, so I7 still holds. Different indices give independent streams, because NumPy's `SeedSequence` hashes the whole
list.

```python
>>> rng = np.random.default_rng
>>> rng([0, 5]).random(), rng([0, 5]).random(), rng([0, 6]).random()     # (seed 0, item 5) twice, then item 6
(0.38213610129812814, 0.38213610129812814, 0.41209234750155577)
```

A `Generator` created once in `__init__` would instead be pickled into every worker in the same state. A pickled
`Generator` draws the same numbers as the original, so four workers would repeat each other's "random" draws.

**Check it: window 80 of `val_KIT_12_v0`.** Find its position with
`i = ds.windows.index(("val_KIT_12_v0", 25600, 26176))`. A plain tuple compares equal to a `Window`. Then, at local
step k = 26,013 − 25,600 = 413:

| what | where in `ds[i]` | value |
|---|---|---|
| LeftLeg's `x_pod` (G1) | `x[10, 0:8, 413]` | 0.6890, 0.2179, −0.6912, 0.0496, −0.3476, −0.8534, 0.0216, 0 |
| the phone's eight, the same for every pod | `x[10, 8:16, 413]` | 0.0155, 0.1429, 0.9896, 0.2638, 0.1285, 1.2648, 0.0058, 0 |
| LeftLeg seen from the phone (G7) | `x[10, 16:22, 413]` | 0.1695, −0.6018, 0.7805, −0.9724, 0.0265, 0.2316 |
| the one-hot | `x[10, 22:37, 413]` | 1 at channel 32 (22 + 10), 0 elsewhere |
| the target (G1) | `theta[10, 413]` | 3.0107 rad, which is 172.501° |
| the mask | `mask[10, 413]`, `mask.sum()`, `mask[[5, 8, 11, 14]].any()` | True, 3,168, False |
| window 0 of the same session | `mask.sum()` | 1,298 |

Shapes and dtypes of `ds[0]`: `x` (15, 37, 576) float32, `theta` (15, 576) float32, `mask` (15, 576) bool.

**Master it.**
1. Turn the table into a test (§5.2).
2. Time `ds[i]` for 200 random i with `time.perf_counter()`. Compare with the 7 ms of §3.7. Then profile with
   `python -m cProfile -s cumtime your_script.py` and name the three largest costs.
3. Iterate the whole validation split once (with 4 workers) and check that the masks hold 37,059,405 True cells.

### 4.6 `collate`: from items to a batch

**What we need.** The `DataLoader` hands `collate_fn` a list of B items. `train` wants one dict of tensors in the
(N, C, T) convention.

**Contract.**

```python
def collate(batch: list[Item]) -> dict[str, torch.Tensor]:
    """B items to one batch of N = 15·B sequences, sequence n = 15·b + p: ``x`` (N, 37, W) float32,
    ``theta`` (N, W) float32 rad, ``mask`` (N, W) bool."""
```

**Pseudocode.**

```text
collate(batch):
1  for each key k of ("x", "theta", "mask"):
2      a ← the B arrays batch[b][k], stacked along a new first axis       # (B, 15, …): the one copy
3      t ← a as a torch tensor, sharing its memory                         # float32 stays float32, bool stays bool
4      out[k] ← t with its first two axes merged                           # (15·B, …): a view
5  return out
```

**The mathematics.** §3.5: n = 15·b + p. Stacking copies once. `torch.from_numpy` shares the NumPy array's memory,
and `flatten(0, 1)` of a contiguous tensor is a view, so neither copies.

**Line by line.**

| line | the Python it needs | why |
|---|---|---|
| 1 | a loop or a dict comprehension over a tuple of keys | the same three steps for each key |
| 2 | `np.stack(list_of_arrays)` (or `torch.stack` of tensors) | a new axis 0 of size B |
| 3 | `torch.from_numpy(array)` | no copy; the dtype is kept |
| 4 | `tensor.flatten(0, 1)` | merges axes 0 and 1 in C order |
| 5 | return the dict | the keys `train` and the tests expect |

```python
>>> import torch
>>> items = [torch.full((3, 2), float(b)) for b in range(4)]     # toy: 4 items, 3 pods, 2 values
>>> s = torch.stack(items); f = s.flatten(0, 1)
>>> tuple(s.shape), tuple(f.shape), f[7], divmod(7, 3)           # row 7 is item 2, pod 1
((4, 3, 2), (12, 2), tensor([2., 2.]), (2, 1))
>>> a = np.zeros(3, np.float32); t = torch.from_numpy(a); a[0] = 7; t   # t shares a's memory
tensor([7., 0., 0.])
```

**Where it runs, and what that requires.**
- With workers, `collate_fn` runs *inside the worker*. The batch is assembled in parallel with training and crosses
  to the main process once, as tensors in shared memory. So do the stacking here, not in `train`.
- Under spawn, `collate_fn` is pickled *by reference*, by its module and name. It must be a module-level `def`. A
  lambda or a nested function fails:
  `PicklingError: Can't pickle <function <lambda> ...>: attribute lookup <lambda> on __main__ failed`.
- `torch.from_numpy` warns on a read-only array ("The given NumPy array is not writable, and PyTorch does not support
  non-writable tensors"), such as a view returned by `np.broadcast_to`. Your items are fresh arrays, so this does not
  happen unless an item returns such a view.

**Style.** A comprehension over the three keys is short and idiomatic. Three explicit lines are just as good when
they read more clearly to you. Choose one and be consistent.

**Check it.** A batch from the `DataLoader` holds `x` (480, 37, 576) float32, `theta` (480, 576) float32 and `mask`
(480, 576) bool. The index-map test of §5.2 passes.

**Master it.**
1. Test the index map: `out["x"][15 * b + p]` equals `items[b]["x"][p]` for every b and p.
2. Undo it with `unflatten(0, (B, 15))`, and check the round trip.
3. Look at one batch just before the network, as Karpathy's recipe advises. For n = 15·b + 10 (item b's LeftLeg),
   plot the tilt z (`x[n, 2]`), θ in degrees where the mask is True, and the mask. Compare with your Lesson 1 plot.

### 4.7 The `DataLoader`, argument by argument

```python
from torch.utils.data import DataLoader
loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=4,
                    persistent_workers=True, generator=torch.Generator().manual_seed(0))
```

| argument | what it does | how (torch 2.14.1) |
|---|---|---|
| `ds` | the map-style dataset | the `DataLoader` only calls `len(ds)` and `ds[i]` |
| `batch_size=32` | 32 windows per batch: 480 sequences after `collate` | a `BatchSampler` groups the sampler's indices by 32; `drop_last=False` keeps the short last batch |
| `shuffle=True` | a new order every epoch | `RandomSampler.__iter__` yields `torch.randperm(N, generator=g).tolist()` |
| `collate_fn=collate` | turns 32 items into one batch | runs in the worker (§4.6) |
| `num_workers=4` | 4 processes read and assemble batches while the main process trains | on macOS each worker is spawned: it imports your module and receives a pickled copy of `ds` and `collate` |
| `persistent_workers=True` | the workers live across epochs | no new spawn per epoch, and each worker's open files stay open |
| `generator=...` | a reproducible order and reproducible worker seeds | the permutation and the workers' base seed are drawn from it. `manual_seed` returns the generator itself, so the call chains |

**Defaults worth knowing.**
- `prefetch_factor=2` per worker: up to 8 batches in flight, about 338 MB here.
- `pin_memory=False`. Pinned memory is a CUDA feature; on MPS, torch turns it off and warns.
- `in_order=True`: batches come back in the sampler's order, whichever worker finishes first. This is why the number
  of workers cannot change your batches.
- `timeout=0`: wait for a worker as long as it takes.

**Seeds in the workers.** Each worker sets `random.seed` and `torch.manual_seed` to the base seed plus its id, and
seeds NumPy's *legacy* global generator. A `Generator` object that you stored on the dataset is not reseeded (§4.5).

**Determinism, proved.** The order of batches in epoch e is a function of the generator's seed and e only (the
sampler's permutations). Each item is a pure function of its index (I7), `collate` is pure, and `in_order=True`
returns batches in the sampler's order. So the batches are a function of (the dataset, the seed, e), whatever the
number of workers and however the processes are scheduled.

**A script that uses the loader.** Every script that starts workers needs the main guard. Under spawn, each worker
imports the script's module again, and without the guard it would start workers of its own. You then see
`RuntimeError: An attempt has been made to start a new process before the current process has finished its
bootstrapping phase`, followed by `DataLoader worker (pid …) exited unexpectedly`.

```python
import time

import torch
from torch.utils.data import DataLoader

from mesqsim.learn.loader import WindowDataset, collate
from mesqsim.paths import output_dir


def main() -> None:
    ds = WindowDataset(output_dir() / "datasets" / "twin_v1", "train", part=4)
    loader = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate, num_workers=4,
                        persistent_workers=True, generator=torch.Generator().manual_seed(0))
    for k, batch in enumerate(loader):
        if k == 0:  # the first batch includes starting the workers: time from the second one
            print({name: (tuple(v.shape), v.dtype) for name, v in batch.items()})
            t0 = time.perf_counter()
        elif k == 20:
            print(f"{(time.perf_counter() - t0) / 20:.3f} s per batch")
            break


if __name__ == "__main__":
    main()
```

**The validation loader.** Lesson 7 validates every 200 batches on 1,000 fixed validation windows. Take evenly
spaced indices, `np.linspace(0, len(val_ds) - 1, 1000).round().astype(int)`, wrap them with
`torch.utils.data.Subset(val_ds, indices.tolist())`, and use `shuffle=False`. Even spacing covers every session in
proportion and never changes between runs.

**Check it.** The script prints `{'x': ((480, 37, 576), torch.float32), 'theta': ((480, 576), torch.float32),
'mask': ((480, 576), torch.bool)}` and a time per batch. Compare that time with the GPU's estimated 0.9 s.

**Master it.**
1. Determinism: two loaders with the same seed give identical first three batches, with `num_workers=0` and with 4.
   Then remove the generator and watch the test fail.
2. Throughput: measure batches per second for `num_workers` = 0, 1, 2, 4 and 5. Where does it stop improving? Is it
   above about 1.1 batches per second, the GPU's estimated rate?
3. Build the validation loader above and count its batches: ⌈1000/32⌉ = 32, the last one with 8 windows.

---

## 5. The tests you write yourself

Put them in a file of your own, for example `tests/test_my_loader.py`; the kit's `tests/test_model_kit.py` stays as
it is. Tests that need `twin_v1` skip cleanly without it, as the kit's `_loader_pair` does
([tests/test_model_kit.py:135](../../tests/test_model_kit.py#L135)).

### 5.1 One test written out, as a model for the others

This test was checked against a throwaway implementation that meets the contract; that implementation was then
deleted.

```python
"""My tests for mesqsim/learn/loader.py. Synthetic arrays only, so they run without twin_v1 (CLAUDE.md rule 6)."""

import numpy as np
import pytest

loader = pytest.importorskip("mesqsim.learn.loader")


def test_pod_inputs_places_each_block_in_its_channels():
    rng = np.random.default_rng(0)
    t = 7  # any length: a window or a whole session
    x_pod = rng.standard_normal((t, 15, 8), dtype=np.float32)
    x_phone = rng.standard_normal((t, 8), dtype=np.float32)
    x_pod_phone = rng.standard_normal((t, 15, 6), dtype=np.float32)

    x = loader.pod_inputs(x_pod, x_phone, x_pod_phone)

    assert x.shape == (15, 37, t) and x.dtype == np.float32 and x.flags.c_contiguous
    for p in (0, 10, 14):
        np.testing.assert_array_equal(x[p, 0:8], x_pod[:, p].T)  # its own pod's eight
        np.testing.assert_array_equal(x[p, 8:16], x_phone.T)  # the phone's eight, the same for every pod
        np.testing.assert_array_equal(x[p, 16:22], x_pod_phone[:, p].T)  # its orientation seen from the phone
        np.testing.assert_array_equal(x[p, 22:37], np.broadcast_to(np.eye(15)[p][:, None], (15, t)))  # one-hot
```

**Why it is built this way.**
- **Arrange, act, assert,** separated by blank lines, so a reader sees the setup, the call and the claims.
- **Random but seeded inputs.** All-zero inputs would hide a swapped block, because zeros equal zeros.
- **T = 7, not 576,** so a hard-coded window length fails.
- **Pods 0, 10 and 14:** the first and the last catch off-by-one errors, and 10 is the course's LeftLeg.
- **The expected values come straight from the inputs,** not from running the same code a second time.
- **The test's name states the property,** so a failure report reads as a sentence.

### 5.2 The tests to write next

| test | what it checks | needs |
|---|---|---|
| `test_loss_mask_is_false_in_the_warm_up_and_equals_valid_after` | I3, for any T and L | nothing |
| `test_pod_inputs_rejects_swapped_arguments` | `ValueError` when `x_pod` and `x_pod_phone` are swapped | nothing |
| `test_collate_puts_pod_p_of_item_b_at_row_15b_plus_p` | I8, and the three dtypes | nothing |
| `test_window_80_of_val_KIT_12_v0_matches_the_course` | the golden table of §4.5 | `MESQ_TEST_DATASET` |
| `test_the_splits_and_parts_have_their_window_counts` | val 9,400; train `part=4` 12,141; `part=16` 2,394 | `MESQ_TEST_DATASET` |
| `test_a_used_dataset_still_pickles` | `ds[0]`, then a pickle round trip; the copy's cache is empty | `MESQ_TEST_DATASET` |
| `test_one_seed_gives_the_same_batches_with_workers` | the first 3 batches of two loaders with `num_workers=2` | `MESQ_TEST_DATASET` |
| `test_the_validation_masks_hold_37_059_405_cells` | the whole split, marked `@pytest.mark.slow` | `MESQ_TEST_DATASET` |

---

## 6. Conventions and style, and why

1. **Write the contract first.** A docstring's first line says what the function returns; then come shapes, dtypes
   and units. The reader needs the contract, not the story (PEP 257; this repository's modules, for example
   [mesqsim/learn/targets.py](../../mesqsim/learn/targets.py)).
2. **Put the units in the name or in the contract.** `theta` is radians by contract. Any new variable carries its
   unit: `_rad`, `_s` (CLAUDE.md, Practicalities). A unit error is this project's most likely silent bug.
3. **No magic numbers.** Import `N_PODS`, `WINDOW_STEPS`, `LOSS_STEPS` and `PARTS`, and read the other sizes from the
   arrays. A changed suit then changes one place.
4. **Modern type hints.** Write built-in generics (`list[Window]`), unions with the `|` operator, and
   `from __future__ import annotations` (PEP 484, 585 and 604). Hints make contracts readable and let mypy or pyright
   find bugs, but they check nothing at run time; validate at the boundaries.
5. **Records are `NamedTuple`s and items are `TypedDict`s:** names instead of positions, while still a tuple and a
   dict at run time (Fluent Python, chapters 5 and 15).
6. **Functional core, imperative shell.** `pod_inputs`, `loss_mask` and `collate` are pure; only `_group` touches
   files. Pure functions are tested without data, in milliseconds (rule 6).
7. **Fail fast, loudly, with the values.** Raise `ValueError` in `__init__` and show the bad value with `!r`. A wrong
   split should stop you at construction, not after an hour of training.
8. **Use `raise`, never `assert`, to check inputs in library code.** `python -O` removes `assert` statements. Keep
   `assert` for tests and for internal states that "cannot happen".
9. **Deterministic by construction.** No random draw at all for now; later, one generator per item from (seed, i).
   Never touch global random state in a library.
10. **No print, no plots and no heavy imports in a library module.** Each spawned worker imports your module, so
    importing matplotlib there (as your Lesson 1 script does) slows every worker's start. Keep scripts and the
    package apart.
11. **Order the imports and use relative ones inside the package:** standard library, third party, local, each
    sorted (ruff's `I` rule), with `from ..body import N_PODS` as the rest of `mesqsim` does.
12. **Mark what is public.** List the public names in `__all__`; start private names with `_` (`_group`, `_files`).
    PEP 8 calls these the public and internal interfaces.
13. **Comments say why.** The exception is a shape comment such as `# (T, 15, 8)`, which is the most useful comment
    in array code.
14. **Lines of at most 120 characters, ruff clean.** That is this repository's configuration (`pyproject.toml`). Run
    `python -m ruff check mesqsim/learn/loader.py` before every commit, and `bash scripts/check_all.sh` (about 13
    minutes) before a push.
15. **The repository's rules:** labels through `read_targets` only (rule 8); a causal window, whose inputs end where
    its loss region ends (rule 10); no data or weights in git (rule 3); tests that skip without data (rule 6).

---

## 7. Usual mistakes: what you see, and the fix

| mistake | what you see | fix |
|---|---|---|
| a window past its session's end | nothing at first: h5py returns fewer rows (184 instead of 576); then `RuntimeError: stack expects each tensor to be equal size, but got [15, 37, 576] at entry 0 and [15, 37, 184] at entry 1` | use `store.windows`; test I1 |
| windows from two splits | no error; validation scores that look too good | filter by split once; test every window's session |
| `y_eps` read straight from h5py | no error, but rule 8 is broken and the label audit fails | `read_targets` only |
| a mask True in the warm-up | the kit's mask test fails; otherwise the model trains on outputs that saw zero padding | `loss_mask`; test I3 |
| a mask that ignores `valid_eps` | training toward stored zeros: absent pods, the phone before its Zero, transitions | AND it with `valid_eps` |
| `theta` in degrees | no error; the cos loss treats 172.5 as radians | never convert in the loader |
| `np.eye` without a dtype | `x` is float64; then `RuntimeError: Input type (double) and bias type (float) should be the same` | `np.eye(N_PODS, dtype=np.float32)`; test the dtype |
| broadcasting along the wrong axis | `ValueError: operands could not be broadcast together with shapes (576,15) (576,)` | transpose first, or index the time mask with `[:, None]` |
| `&` without parentheses | `TypeError: Cannot perform 'rand_' with a dtyped [int64] array and scalar of type [bool]` | put each comparison in parentheses |
| a file opened in `__init__`, or cached before the workers start | `TypeError: h5py objects cannot be pickled` when the workers start | a lazy cache and `__getstate__` |
| a lambda or nested function as `collate_fn` | `PicklingError: Can't pickle <function <lambda> ...>` | a module-level function |
| no main guard in a script with workers | `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | put the work in `main()` under `if __name__ == "__main__":` |
| NumPy int64 in `windows` | fine until something serializes it: `TypeError: Object of type int64 is not JSON serializable` | `.tolist()` or `int()` |
| a `Generator` made in `__init__`, used in `__getitem__` | every worker repeats the same "random" draws | a generator per item from (seed, i) |
| `shuffle=True` without a generator | different batches on every run | pass a seeded `torch.Generator` |
| a part filter on the validation split | an empty dataset (validation sessions have part 0) | refuse `part` unless the split is `train` |
| whole sessions read in `__getitem__` | about 22 times slower per item (158 ms against 7 ms) | slice the window |

---

## 8. Master it: the ladder after the lesson

Each section above ends with its own drills. Once the loader works, climb these in order:

1. **Reproduce.** Tomorrow, from a blank file and without notes, write `pod_inputs`, `loss_mask` and `collate` in 20
   minutes, then the dataset in 30, and run your tests. Do it again in a week. Spaced repetition is what makes it
   stick.
2. **Explain.** Write half a page, without notes: why the windows tile, why 256 ≥ 252, why the mask has two factors,
   and why training and scoring share one input function. A vague sentence marks the part you do not understand yet
   (the Feynman technique).
3. **Vary.** Generalize the windows to any (W, L) with W − L ≥ R − 1 and property-test them against `store.windows`.
   Then sketch TCN-J's variant: θ from `y_delta`, the mask from `valid_delta`, 25 inputs per joint, 14 joints, so
   N = 14·B.
4. **Measure.** Items per second against the number of workers; a profile of `__getitem__`. Decide with numbers
   whether anything needs optimizing; §3.7 says it probably does not.
5. **Read the source.** In `torch/utils/data/`: `dataloader.py` (`_MultiProcessingDataLoaderIter`),
   `_utils/worker.py` (`_worker_loop`), `sampler.py` (`RandomSampler`, `BatchSampler`) and `dataset.py`
   (`ConcatDataset`, `Subset`). Find where the base seed is drawn and where `in_order` puts the batches back in order.
6. **Extend with care.** Add a random window phase per epoch for training only, drawn from
   `np.random.default_rng([seed, epoch])`. Explain why validation keeps the reference windows: comparability with
   `score-model`, with G2 and with the coverage tables.
7. **Review.** Use §6 and §7 as a checklist against your file, then ask for a line-by-line review.

---

## 9. Self-check

1. How many windows does a 1,000-step session have, and which steps are scored?
2. Why is the warm-up 256 steps and not 0?
3. Your `x` is float64. Where does that come from, and what fails?
4. Why can `__init__` not open the shards?
5. You keep `self.rng = np.random.default_rng(0)` from `__init__` and draw from it in `__getitem__`, with 4 workers.
   What happens?
6. Row 100 of a flattened batch belongs to which item and which pod?
7. How many windows and sequences are in the last validation batch?
8. A teammate wants to read whole sessions in `__getitem__` and cut the window in memory, "because it is simpler".
   What do you answer, with numbers?
9. Why is the mask False on pods 5, 8, 11 and 14 everywhere in `val_KIT_12_v0`?
10. Why does `in_order=True` matter for determinism?

---

**Answers. Try the questions before reading on.**

1. $n = \lfloor (1000 - 576)/320 \rfloor + 1 = 2$: the windows [0, 576) and [320, 896). Steps 256 to 895 are scored
   (640 steps); the first 256 and the last 104 = (1000 − 576) mod 320 are not.
2. The receptive field is 253 steps. An output at a local step below 252 reads padding zeros, so it differs from the
   live network's output. From 256 on, the window's outputs equal the whole session's (§3.2).
3. From `np.eye` without `dtype=np.float32`: float32 with float64 promotes to float64. The first `Conv1d` raises
   `RuntimeError: Input type (double) and bias type (float) should be the same`.
4. On macOS each worker receives the dataset by pickle, and an open h5py file cannot be pickled. Under fork, sharing
   HDF5 handles across processes is unsafe.
5. Each worker receives an identical copy of the generator, so the workers repeat each other's draws. Use a
   generator per item, `default_rng([seed, i])`.
6. `divmod(100, 15) = (6, 10)`: item 6, pod 10 (LeftLeg).
7. 9,400 − 293·32 = 24 windows, so 360 sequences.
8. A whole session costs about 158 ms against 7 ms for a window. Four workers would deliver about 25 windows per
   second against the 36 the GPU needs, so it would idle almost 30 % of the time.
9. It is an old-suit session (`suit_2026_11_13`), which has no hands and no feet. Those pods are absent, so
   `valid_eps` is False for them.
10. Batches come back in the sampler's order, not in the order the workers finish, so the timing of the processes
    cannot change the batches.

---

## 10. What is next

**Before you leave Lesson 2:**
- the three kit loader tests pass with `MESQ_TEST_DATASET` set, and your synthetic tests pass without it;
- `python -m ruff check mesqsim/learn/loader.py` is clean;
- you have plotted one batch (drill 4.6.3);
- your loader and your tests are committed on a branch (never data or weights, rule 3).

**Then Lesson 4, causal convolutions** (1 hour; the two-day track skips Lesson 3). It connects to this lesson at
every step:
- the warm-up H = 256 exists because of the 253 steps you will derive by hand there;
- your batch's `x` (480, 37, 576) is what Lesson 5's network eats;
- `theta` and `mask` are what Lesson 6's losses eat;
- your `DataLoader` is what Lesson 7's `train` iterates.

When you are ready, ask for (1) a line-by-line review of `mesqsim/learn/loader.py`, and (2) Lesson 4 in this same
format.

---

## 11. References

**The course and the repository.**
- [LEARN_THE_MODEL.md](../step_3/LEARN_THE_MODEL.md), Lesson 2; [DATASET_CARD.md](../step_3/DATASET_CARD.md), §5;
  the Step 3 brief, [§7.4 and §7.5](../brief/STEP3_BRIEF.md).
- [mesqsim/learn/store.py:395](../../mesqsim/learn/store.py#L395) (`windows`),
  [mesqsim/learn/targets.py:36](../../mesqsim/learn/targets.py#L36) (`read_targets`),
  [mesqsim/learn/contract.py:59](../../mesqsim/learn/contract.py#L59) (`SessionInputs.from_group`, which Lesson 8's
  wrapper receives), [tests/test_model_kit.py:135-173](../../tests/test_model_kit.py#L135-L173) (the loader tests).

**PyTorch.**
- *Datasets & DataLoaders*: <https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html>
- *Writing Custom Datasets, DataLoaders and Transforms*:
  <https://docs.pytorch.org/tutorials/beginner/data_loading_tutorial.html>
- `torch.utils.data` (dataset types, multi-process loading, platform-specific behaviours, randomness in workers):
  <https://docs.pytorch.org/docs/stable/data.html>
- *Reproducibility*: <https://docs.pytorch.org/docs/stable/notes/randomness.html>
- A. and S. Amidi, *A detailed example of how to generate your data in parallel with PyTorch*:
  <https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel>

**h5py and NumPy.**
- h5py, *Datasets* (slicing, chunked storage): <https://docs.h5py.org/en/stable/high/dataset.html>; *File objects*
  (the chunk cache): <https://docs.h5py.org/en/stable/high/file.html>
- NumPy, *Broadcasting*: <https://numpy.org/doc/stable/user/basics.broadcasting.html>; `numpy.broadcast_to`:
  <https://numpy.org/doc/stable/reference/generated/numpy.broadcast_to.html>
- NumPy, *Random Generator*: <https://numpy.org/doc/stable/reference/random/generator.html>; *Parallel random number
  generation*: <https://numpy.org/doc/stable/reference/random/parallel.html>; NEP 50, type promotion:
  <https://numpy.org/neps/nep-0050-scalar-promotion.html>

**Python.**
- L. Ramalho, *Fluent Python*, 2nd ed., O'Reilly, 2022: ch. 1 (the data model), 2 (sequences, comprehensions,
  slicing), 3 (dictionaries), 5 (data class builders), 8 and 15 (type hints), 11 (a Pythonic object), 12 (sequence
  protocols), 17 (iterators), 18 (`with`), 19 (concurrency models).
- `pickle`, *Handling Stateful Objects*: <https://docs.python.org/3/library/pickle.html#handling-stateful-objects>
- `multiprocessing`, *Contexts and start methods* and *Programming guidelines*:
  <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>,
  <https://docs.python.org/3/library/multiprocessing.html#programming-guidelines>
- PEP 8, PEP 257, PEP 484, PEP 604: <https://peps.python.org/>; the *Google Python Style Guide*:
  <https://google.github.io/styleguide/pyguide.html>; Hypothesis: <https://hypothesis.readthedocs.io/>

**Engineering.**
- A. Karpathy, *A Recipe for Training Neural Networks*, 2019: <https://karpathy.github.io/2019/04/25/recipe/>
- M. Zinkevich, *Rules of Machine Learning*, rule 32:
  <https://developers.google.com/machine-learning/guides/rules-of-ml>
- E. W. Dijkstra, *Why numbering should start at zero* (EWD 831), 1982:
  <https://www.cs.utexas.edu/~EWD/transcriptions/EWD08xx/EWD831.html>
- G. Bernhardt, *Boundaries* (functional core, imperative shell), 2012:
  <https://www.destroyallsoftware.com/talks/boundaries>
- L. Kish, *Survey Sampling*, Wiley, 1965 (the design effect).

---

## 12. Verification notes

Run from the repository root with `.venv/bin/python`, `PYTHONPATH=.`, and `MESQ_OUTPUT_DIR=../mesquite_work/outputs`,
on 2026-11-22.

- **Windows (G2).** `windows(49684)`: 154 windows, the first three [0, 576), [320, 896) and [640, 1216), the last
  [48960, 49536), dtype int64, 148 steps left after it. `windows(T)` for T = 575, 576, 895, 896, 897: 0, 1, 1, 2, 2.
  `windows(1000)`: [0, 576) and [320, 896).
- **Split totals.** Sums of `len(windows(steps))` over `index.parquet`'s rows: val 74 sessions and 9,400 windows;
  test 72 and 7,734; train 519 and 54,153; `part >= 16` 30 and 2,394; `part >= 4` 129 and 12,141 (36 shards). No
  validation row has a part of 4 or more. The batch counts and last-batch sizes follow by division.
- **The index.** Columns: composition, hipsalt, id, length_s, program, program_seed, seed, shard, slips, split,
  steps, subject, subset, suit_version, variant, wire, part. Part values: 0, 1, 2, 4, 8, 16. The first validation
  session in file order is `val_BMLmovi_Subject_11_F_MoSh_v0`.
- **The arrays** of `val_KIT_12_v0` (`shards/val_00072.h5`; old suit, HipsAlt present): `x_pod` (49684, 15, 8)
  float32, `x_phone` (49684, 8) float32, `x_pod_phone` (49684, 15, 6) float32, `y_eps` float32, `valid_eps` bool,
  all gzip-compressed with the shuffle filter, in chunks of 2,048 steps. Pods 5, 8, 11 and 14 are all zero in window
  80.
- **Golden values.** Read with h5py slicing and `read_targets` for window [25600, 26176) at local step 413: the
  values of the table in §4.5; `y_eps` 172.50131°, valid.
- **Masks.** Window 0: 1,298 True cells, 118 per present pod, ε valid from step 458. Window 80: 3,168, 288 per
  present pod; `strata/transition` is True on 32 steps of its loss region, in two runs (34-49 and 236-251). Over the
  validation split, with `read_targets` and prefix sums: 37,059,405 True cells of 81,216,000. Added on 2026-11-23:
  `valid.T & in_loss` comes out F-ordered (strides (1, 15)); the unparenthesised `valid.T & np.arange(t) >= t -
  loss_steps` gives 0 True cells on windows 0 and 80; the nonzero invalid targets were counted with h5py over the 74
  validation sessions.
- **The index of items** (2026-11-23, with a throwaway `__init__` in a scratch copy, deleted afterwards): the counts
  of §3.1; `val_KIT_12_v0`'s windows at positions 7,082 to 7,235, window 80 at 7,162; 10, 36, 10, 65 and 9 shards for
  val, train 1/4, train 1/16, all of train and test; train, val and test share no subject (173, 37 and 36 subjects);
  `==` instead of `>=` on the part gives the counts of §4.3; the training dataset holds 8.0 MB and pickles to 1.12 MB;
  a relative shard path stops resolving after `os.chdir`.
- **Finite inputs.** `x_pod`, `x_phone` and `x_pod_phone` of all 74 validation sessions (3,037,218 steps) hold no
  NaN or infinity.
- **Timing.** 60 validation windows drawn with `np.random.default_rng(0)`, after one warm-up pass, timed with
  `time.perf_counter`; the whole-session figure over the first 10 of them; the array work over 200 repetitions on
  synthetic (576, 15, ·) float32 arrays; stacking over 50 repetitions of 32 items.
- **Chunks.** In `windows(10**6)`, 25 % of windows straddle a 2,048-step chunk boundary; 39 of `val_KIT_12_v0`'s 154
  do.
- **Library behaviour** (each error message above was produced by running it): h5py's clipped slice (184 rows); the
  pickling errors of an h5py file, a lambda and an open text file; float64 promotion from `np.eye`; the `Conv1d`
  dtype error; the broadcasting and precedence errors; `torch.stack`'s size error; `torch.from_numpy`'s read-only
  warning; the identical draws of a pickled `Generator`; the `default_rng([0, 5])` and `([0, 6])` values; the
  missing-main-guard error, from a 4-line script with `num_workers=1`. In torch 2.14.1's sources:
  `_utils/worker.py` lines 275-282 (worker seeds), `sampler.py` (`torch.randperm` with the generator),
  `dataloader.py` (`prefetch_factor` 2, `in_order=True`, pinned memory turned off on MPS, the base seed drawn from
  the loader's generator).
- **Memory.** 54,153 `Window` tuples take about 8 MB (`tracemalloc`). Item and batch sizes are arithmetic from the
  shapes and dtypes.
- **The example test** of §5.1 passed against a throwaway implementation that meets the contract, which was then
  deleted. No loader code exists in the repository other than what you write.
