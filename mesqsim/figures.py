"""Figures for docs/figures: the AMASS → Mesquite stick figure and the DIODEM sanity figure.

The stick figure needs joint positions. Without the SMPL+H body model it uses generic bone lengths
(:data:`GENERIC_BONES`, an assumption for drawing only) placed with the 15 segment orientations, so it shows
exactly what the mapping produces: which segment is where, left versus right, and which way is up.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import plotstyle
from .conventions import quat_rotate

__all__ = ["GENERIC_BONES", "stick_positions", "stick_figure", "diodem_figure", "random_walk_figure"]

# Bone vectors in the canonical SMPL body frame (x: subject's left, y: up, z: forward), metres.
# Drawing aid only (label: assumption); never used in any computation.
GENERIC_BONES = {
    "hip_l": (2, (0.09, -0.08, 0.0)),  # (segment pod id, offset from the segment's proximal joint)
    "hip_r": (2, (-0.09, -0.08, 0.0)),
    "chest": (2, (0.0, 0.22, 0.0)),
    "neck": (1, (0.0, 0.26, 0.0)),
    "head_top": (0, (0.0, 0.22, 0.0)),
    "shoulder_l": (1, (0.17, 0.22, 0.0)),
    "shoulder_r": (1, (-0.17, 0.22, 0.0)),
    "elbow_l": (3, (0.28, 0.0, 0.0)),
    "wrist_l": (4, (0.25, 0.0, 0.0)),
    "hand_l": (5, (0.09, 0.0, 0.0)),
    "elbow_r": (6, (-0.28, 0.0, 0.0)),
    "wrist_r": (7, (-0.25, 0.0, 0.0)),
    "hand_r": (8, (-0.09, 0.0, 0.0)),
    "knee_l": (9, (0.0, -0.42, 0.0)),
    "ankle_l": (10, (0.0, -0.42, 0.0)),
    "toe_l": (11, (0.0, -0.06, 0.14)),
    "knee_r": (12, (0.0, -0.42, 0.0)),
    "ankle_r": (13, (0.0, -0.42, 0.0)),
    "toe_r": (14, (0.0, -0.06, 0.14)),
}
# (point, parent point, side) in forward-kinematics order
_CHAIN = (
    ("hip_l", "pelvis", "L"), ("hip_r", "pelvis", "R"), ("chest", "pelvis", "C"), ("neck", "chest", "C"),
    ("head_top", "neck", "C"), ("shoulder_l", "chest", "L"), ("shoulder_r", "chest", "R"),
    ("elbow_l", "shoulder_l", "L"), ("wrist_l", "elbow_l", "L"), ("hand_l", "wrist_l", "L"),
    ("elbow_r", "shoulder_r", "R"), ("wrist_r", "elbow_r", "R"), ("hand_r", "wrist_r", "R"),
    ("knee_l", "hip_l", "L"), ("ankle_l", "knee_l", "L"), ("toe_l", "ankle_l", "L"),
    ("knee_r", "hip_r", "R"), ("ankle_r", "knee_r", "R"), ("toe_r", "ankle_r", "R"),
)  # fmt: skip


def stick_positions(seg_q: np.ndarray, root_pos: np.ndarray) -> dict[str, np.ndarray]:
    """Joint positions (N, 3) from segment orientations (N, 15, 4) and pelvis positions (N, 3)."""
    pos = {"pelvis": np.asarray(root_pos, dtype=np.float64)}
    for name, parent, _ in _CHAIN:
        seg, off = GENERIC_BONES[name]
        pos[name] = pos[parent] + quat_rotate(seg_q[:, seg], np.broadcast_to(off, (seg_q.shape[0], 3)))
    return pos


def stick_figure(seg_q, root_pos, t_s, frames, title: str, out: Path) -> Path:
    """Draw the body at a few frames, side by side, in the +Z-up world; left side blue, right side orange."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    pos = stick_positions(seg_q, root_pos)
    colour = {"L": plotstyle.LEFT, "R": plotstyle.RIGHT, "C": plotstyle.CENTRE}
    fig = plt.figure(figsize=(3.0 * len(frames), 3.6))
    for k, f in enumerate(frames):
        ax = fig.add_subplot(1, len(frames), k + 1, projection="3d")
        centre = pos["pelvis"][f]
        for name, parent, side in _CHAIN:
            a, b = pos[parent][f] - centre, pos[name][f] - centre
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=colour[side], lw=2.2, solid_capstyle="round")
        ax.quiver(0, 0, -1.0, 0, 0, 0.3, color=plotstyle.MUTED, lw=1, arrow_length_ratio=0.25)
        ax.text(0.03, 0, -0.68, "+Z up", color=plotstyle.MUTED, fontsize=7)
        ax.set(xlim=(-0.8, 0.8), ylim=(-0.8, 0.8), zlim=(-1.0, 0.8), xticks=[], yticks=[], zticks=[])
        ax.set_box_aspect((1, 1, 1.125))
        ax.set_title(f"t = {t_s[f]:.2f} s", fontsize=9, color=plotstyle.INK_2)
        ax.view_init(elev=12, azim=-60)
    fig.suptitle(title, fontsize=10, color=plotstyle.INK)
    fig.legend(
        handles=[plt.Line2D([], [], color=plotstyle.LEFT, lw=2.2), plt.Line2D([], [], color=plotstyle.RIGHT, lw=2.2)],
        labels=["subject's left", "subject's right"], loc="lower center", ncol=2,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)
    return out


def diodem_figure(res: dict, label: str, out: Path) -> Path:
    """Two panels: optical vs six-axis VQF relative heading; then their difference (the drift)."""
    import matplotlib.pyplot as plt

    plotstyle.apply()
    t_min = res["t_s"] / 60.0
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    a1.plot(t_min, res["rel_opt_deg"], color=plotstyle.SERIES[0], lw=1.2, label="optical (truth)")
    a1.plot(t_min, res["rel_vqf_deg"], color=plotstyle.SERIES[1], lw=1.2, label="six-axis VQF, aligned at t = 0")
    a1.set_ylabel("relative heading (deg)")
    a1.set_title(f"DIODEM {label}: child heading minus parent heading", loc="left")
    a1.legend(loc="upper left")
    a2.plot(t_min, res["delta_deg"], color=plotstyle.SERIES[1], lw=1.5)
    a2.axhline(0, color=plotstyle.AXIS, lw=1)
    a2.set_ylabel("VQF minus optical (deg)")
    a2.set_xlabel("time since start of trial (min)")
    a2.annotate(f"{res['delta_deg'][-1]:+.1f}°", (t_min[-1], res["delta_deg"][-1]), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=8, color=plotstyle.INK_2)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)
    return out


def random_walk_figure(points: list, n_pods: int, out: Path) -> Path:
    """Measured spread of psi (one pod) and of a parent-child pair against N_g sqrt(t).

    ``points``: (t_s, sigma_single_deg, sigma_pair_deg) from the simulator with only the random walk switched on.
    """
    import matplotlib.pyplot as plt

    plotstyle.apply()
    t = np.array([p[0] for p in points])
    s1 = np.array([p[1] for p in points])
    s2 = np.array([p[2] for p in points])
    tt = np.linspace(0, t.max() * 1.05, 200)
    n_g = 0.015
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(tt / 60, n_g * np.sqrt(tt), color=plotstyle.SERIES[0], lw=1.5, label="one pod: N_g·√t (datasheet)")
    ax.plot(tt / 60, n_g * np.sqrt(2 * tt), color=plotstyle.SERIES[1], lw=1.5, label="pair: √2·N_g·√t")
    se = 1 / np.sqrt(2 * n_pods)
    ax.errorbar(t / 60, s1, yerr=s1 * se, fmt="o", color=plotstyle.SERIES[0], ms=6, capsize=3,
                label=f"simulated, {n_pods} pods")  # fmt: skip
    ax.errorbar(t / 60, s2, yerr=s2 * se, fmt="s", color=plotstyle.SERIES[1], ms=6, capsize=3,
                label="simulated, pairs")  # fmt: skip
    ax.set_xlabel("time since boot (min)")
    ax.set_ylabel("spread of ψ (deg, standard deviation)")
    ax.set_title("Angle random walk in the simulator against the datasheet scaling", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, metadata={"Software": None})
    plt.close(fig)
    return out
