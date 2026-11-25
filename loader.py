

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
import pandas
import torch
from torch.utils.data import Dataset

from ..body import N_PODS
from .store import LOSS_STEPS, PARTS, WINDOW_STEPS, windows
from .targets import read_targets


N_POD_FEATURES = 8 # X_POD
N_PHONE_FEATURES = 8 # X_PHONE
N_POD_PHONE_FEATURES = 6 # X_POD_PHONE
N_INPUTS = N_POD_FEATURES + N_PHONE_FEATURES + N_POD_PHONE_FEATURES + N_PODS

__all__ = ["SPLITS", "Window", "Item", "pod_inputs","loss_mask","WindowDataset","collate"]

SPLITS = ("train", "val", "test")

class Window(NamedTuple):
    """ Item i's place: the session, and its steps ``start`` to ``end`` (end exclusive), as Python ints."""

    session_id: str
    start: int
    end: int

class Item(TypedDict):
    x:np.ndarray # (15,37,W) float32
    theta: np.ndarray # (15,W) float32, rad
    mask: np.ndarray # (15,W) bool

def pod_inputs(x_pod: np.ndarray, x_phone: np.ndarray, x_pod_phone: np.ndarray)-> np.ndarray:
    """The TCN-P's inputs, one sequence per pod: (15, 37, T) float 32, c-contigous for any T.
     ``x_pod`` (T,15,8), ``x_phone``(T, 8) and ``x_pod_phone`` (T,15,6) as stored in ``twin_v1``:
      slice or a whole session. Pod p's channels: 0-7 its ``x_pod``, 8-15 ``x_phone``, 16-21 its``x_pod_phone_fetures
      ``, 22-36 the one-hot of p."""
    t = x_pod.shape[0]
    # check all three shapes at once
    expected = {"x_pod": (t, N_PODS, N_POD_FEATURES), "x_phone":(t, N_PHONE_FEATURES),
                "x_pod_phone": (t, N_PODS, N_POD_PHONE_FEATURES)}
    actual = {"x_pod": x_pod.shape, "x_phone": x_phone.shape, "x_pod_phone": x_pod_phone.shape}
    if actual != expected:
        raise ValueError(f"pod_inputs: expected shapes {expected}, got {actual}")
    phone = np.broadcast_to(x_phone[:, None,:], (t, N_PODS, N_PHONE_FEATURES)) # (T,15,8)
    onehot = np.broadcast_to(np.eye(N_PODS, dtype=np.float32), (t,N_PODS,N_PODS)) # (T,15,15)
    # glue the four bricks along the last channel axis.
    z = np.concatenate([x_pod,phone,x_pod_phone,onehot],
                       axis=-1)
    # Permutaion of the (t,p,c) -> (p,c,t)
    return np.ascontiguousarray(z.transpose(1,2,0), dtype=np.float32) # (15,37,t)


def loss_mask(valid: np.ndarray, loss_steps: int)-> np.ndarray:
   if valid.ndim != 2 or valid.dtype != bool:
       raise ValueError(f"loss_mask: valid must be a (T, n) bool array, got {valid.dtype} of shape {valid.shape}")
   t = valid.shape[0]
   if not 0 < loss_steps <= t:
       raise ValueError(f"loss_mask: loss_steps must be in 1...{t} (the number of steps), got{loss_steps}")
   in_loss = np.arange(t) >= t - loss_steps
   return np.ascontiguousarray(valid.T & in_loss)

    
class WindowDataset(Dataset):
    """TCN-P"""

    def __init__(self, path:str | Path, split :str, window_steps:int = WINDOW_STEPS, loss_steps: int = LOSS_STEPS, seed: int = 0
                 ,*, part: int |None = None) -> None:

        # Making the path absolute
        root= Path(path).resolve()
        if split not in SPLITS:
            raise ValueError(f"WindowDataset: split must be one of {SPLITS}, got {split!r}")

        # the size only tuple comparison checks both values.
        if (window_steps, loss_steps) != (WINDOW_STEPS, LOSS_STEPS):
            raise ValueError(f"WindowDataset: store.windows implements window_steps={WINDOW_STEPS} and "f"loss_steps={LOSS_STEPS} only, got {window_steps} and {loss_steps}")
        if part is not None and (split != "train" or part not in PARTS):
            raise ValueError(f"WindowDataset: part must be None, or one of {PARTS} with split='train'; got "
                             f"part={part!r} with split={split!r}")
        
        # Read the index so the parquet sotres each column seperatly.
        idx = pd.read_parquet(root / "index.parquet", columns=["id","split","part","shard","steps"])
        # the filter
        keep = idx["split"] == split
        if part is not None:
            keep&=idx["part"] >= part
        rows = idx[keep]
        self.root=root
        self.split = split
        self.part = part
        self.loss_steps = loss_steps
        self.seed = seed

        # the shelf map
        # a dict comperhension {key: value for item in iterable}
        self.shard_of: dict[str, Path] = {r.id: root / r.shard for r in rows.iteruples(index=False)}
        self.windows: list[Window] = [Window(r.id, start, end) for r in rows.itertuples(index=False)
                                      for start, end in windows(r.steps).tolist()]
        # An empty dataset is an error
        if not self.windows:
            raise ValueError(f"WindowDataset: no window in split {split!r} (part {part!r}) of {root}")
        self._files: dict[Path, h5py.File] = {}
        

    
    def __len__(self) ->int:
        raise NotImplemented

    def __getitem__(self, i: int) -> Item:
        raise NotImplemented


def collate(batch: list[Item]) -> dict [str.torch.Tensor]:
    """The docstring of §4.6"""
    raise NotImplemented
