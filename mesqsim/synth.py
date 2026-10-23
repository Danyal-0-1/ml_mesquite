"""Synthetic human motion in AMASS format, so that every test runs without licensed data (brief rule 6).

The motions are smooth, anatomically plausible and deliberately simple: a T-pose, standing still, walking
straight, and walking in a circle. They are *not* a substitute for AMASS in any reported result; they exist
to exercise the loader, the mapping and the simulator with known answers.

Pose conventions follow AMASS/SMPL: axis-angle per joint, canonical body frame x = subject's left, y = up,
z = forward; the AMASS world is +Z up, so an upright root is ``Rz(yaw)·Rx(90°)``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .conventions import exp_map, log_map, quat_mul, yaw_quat
from .data.amass import AmassSequence

__all__ = ["KINDS", "make_sequence", "write_amass_npz"]

KINDS = ("tpose", "stand", "walk", "circle")


def _rx(a):
    return exp_map(np.stack([a, np.zeros_like(a), np.zeros_like(a)], -1))


def _ry(a):
    return exp_map(np.stack([np.zeros_like(a), a, np.zeros_like(a)], -1))


def _rz(a):
    return exp_map(np.stack([np.zeros_like(a), np.zeros_like(a), a], -1))


def make_sequence(kind: str = "walk", duration_s: float = 10.0, fps: float = 60.0, seed: int = 0) -> AmassSequence:
    """A synthetic AMASS-format sequence: poses (N, 66), trans (N, 3) in metres, at ``fps``.

    ``walk``: 1 Hz gait (hips ±25°, knees 5-60°, arm swing) heading in a straight line.
    ``circle``: the same gait while turning at 30 °/s. ``stand``: quiet standing with small sway.
    ``tpose``: the SMPL rest pose (every segment equals the root).
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fps))
    t = np.arange(n) / fps
    zero = np.zeros(n)
    local = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n, 22, 1))  # quaternions, (x, y, z, w)
    yaw0 = rng.uniform(-np.pi, np.pi)

    if kind == "tpose":
        yaw = np.full(n, yaw0)
    elif kind == "stand":
        yaw = yaw0 + np.deg2rad(2.0) * np.sin(2 * np.pi * 0.2 * t)
        sway = np.deg2rad(1.5) * np.sin(2 * np.pi * 0.3 * t + rng.uniform(0, 2 * np.pi))
        local[:, 3] = _rx(sway)  # spine1 sways
        local[:, 16] = _rz(np.full(n, np.deg2rad(-75.0)))  # arms down at the sides
        local[:, 17] = _rz(np.full(n, np.deg2rad(75.0)))
    else:
        turn = np.deg2rad(30.0) if kind == "circle" else 0.0
        yaw = yaw0 + turn * t
        w = 2 * np.pi * 1.0  # one stride per second
        phase = w * t
        hip = np.deg2rad(25.0) * np.sin(phase)
        knee_l = np.deg2rad(5.0) + np.deg2rad(55.0) * np.clip(np.sin(phase - 1.2), 0, None)
        knee_r = np.deg2rad(5.0) + np.deg2rad(55.0) * np.clip(np.sin(phase - 1.2 + np.pi), 0, None)
        local[:, 1] = _rx(-hip)  # left hip: negative x rotation swings the thigh forward
        local[:, 2] = _rx(hip)  # right hip in antiphase
        local[:, 4] = _rx(knee_l)  # knee flexion: positive x rotation
        local[:, 5] = _rx(knee_r)
        local[:, 7] = _rx(np.deg2rad(10.0) * np.sin(phase + 0.5))
        local[:, 8] = _rx(np.deg2rad(10.0) * np.sin(phase + 0.5 + np.pi))
        local[:, 3] = _ry(np.deg2rad(4.0) * np.sin(phase))  # trunk counter-rotation
        local[:, 9] = _ry(np.deg2rad(-6.0) * np.sin(phase))
        swing = np.deg2rad(20.0) * np.sin(phase)
        # arms hang down (about z), then swing in antiphase with the legs (about x)
        local[:, 16] = quat_mul(_rx(swing), _rz(np.full(n, np.deg2rad(-75.0))))
        local[:, 17] = quat_mul(_rx(-swing), _rz(np.full(n, np.deg2rad(75.0))))
        local[:, 18] = _ry(np.full(n, np.deg2rad(20.0)))  # elbows slightly bent
        local[:, 19] = _ry(np.full(n, np.deg2rad(-20.0)))
        local[:, 15] = _rx(np.deg2rad(3.0) * np.sin(2 * phase))

    upright = exp_map([np.pi / 2, 0.0, 0.0])  # SMPL body +y (up) -> AMASS world +Z
    local[:, 0] = quat_mul(yaw_quat(yaw), np.broadcast_to(upright, (n, 4)))
    poses = log_map(local).reshape(n, 66)

    forward = np.stack([np.sin(yaw), -np.cos(yaw), zero], -1)  # body +z (forward) in the world
    speed = 0.0 if kind in ("tpose", "stand") else 1.2  # m/s
    trans = np.cumsum(forward * speed / fps, axis=0) + np.array([0.0, 0.0, 0.95])
    return AmassSequence(poses, trans, np.zeros(16), "neutral", float(fps), f"synthetic:{kind}:seed{seed}")


def write_amass_npz(seq: AmassSequence, path: str | Path, smplx_style: bool = False) -> Path:
    """Write ``seq`` with AMASS keys: SMPL+H style (156 pose values, ``mocap_framerate``) by default,
    or SMPL-X style (165 values, ``mocap_frame_rate``)."""
    path = Path(path)
    width = 165 if smplx_style else 156
    poses = np.zeros((seq.n_frames, width))
    poses[:, :66] = seq.poses
    rate_key = "mocap_frame_rate" if smplx_style else "mocap_framerate"
    np.savez(path, poses=poses, trans=seq.trans, betas=seq.betas, gender=np.array(seq.gender), **{rate_key: seq.fps})
    return path
