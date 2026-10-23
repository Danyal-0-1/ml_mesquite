"""Regenerate the figures in docs/figures.

    python scripts/make_figures.py

* stick_figure_synthetic.png: the mapping drawn on a synthetic walk (always available).
* stick_figure_amass.png: the same on the first AMASS walking sequence found (needs AMASS).
* sim_random_walk.png: the simulator's angle random walk against N_g sqrt(t), one pod and pairs.
* diodem_relative_heading.png: six-axis VQF versus optical relative heading, arm trial exp01,
  hinge joint seg2 -> seg3 (needs DIODEM).
Figures whose data are absent are skipped with a message.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from mesqsim import figures, synth  # noqa: E402
from mesqsim.data import amass, diodem  # noqa: E402

OUT = ROOT / "docs" / "figures"


def main() -> int:
    seq = synth.make_sequence("walk", duration_s=2.0, fps=60.0, seed=3)
    seg = amass.segment_orientations(seq.poses)
    frames = [0, 15, 30, 45]
    figures.stick_figure(seg, seq.trans, seq.t_s, frames, "Synthetic walk mapped to the 15 Mesquite segments "
                         "(generic bone lengths)", OUT / "stick_figure_synthetic.png")
    print("wrote stick_figure_synthetic.png")

    path = amass.find_walking_sequence()
    if path is None:
        print("skipped stick_figure_amass.png: no AMASS walking sequence under", amass.amass_dir())
    else:
        seq = amass.load_sequence(path)
        seg = amass.segment_orientations(seq.poses)
        step = max(1, int(round(seq.fps / 4)))
        frames = [k * step for k in range(4) if k * step < seq.n_frames]
        title = f"AMASS {Path(path).relative_to(amass.amass_dir()).as_posix()} (generic bone lengths)"
        figures.stick_figure(seg, seq.trans, seq.t_s, frames, title, OUT / "stick_figure_amass.png")
        print("wrote stick_figure_amass.png from", path)

    from mesqsim.body import JOINTS
    from mesqsim.validation import _heading_pods

    points, n_sessions = [], 40
    for idle in (0.0, 60.0, 240.0, 600.0, 1200.0, 1762.0):
        pods = _heading_pods(n_sessions, {"error_model.arw_deg_per_sqrt_s": 0.015, "session.idle_s": idle},
                             "synthetic:stand:3")  # fmt: skip
        t_eval = pods[0][0].phases["motion"][0] + 1.0
        psi = np.array([float(hd.psi(t_eval)) for _, _, hd in pods]).reshape(n_sessions, 15)
        pair = np.concatenate([psi[:, j.child] - psi[:, j.parent] for j in JOINTS])
        boot = np.mean([p.boot_s for _, p, _ in pods])
        points.append((t_eval - boot, np.rad2deg(psi.std()), np.rad2deg(pair.std())))
    figures.random_walk_figure(points, n_sessions * 15, OUT / "sim_random_walk.png")
    print("wrote sim_random_walk.png:", ", ".join(f"{t / 60:.1f} min {a:.3f}/{b:.3f} deg" for t, a, b in points))

    if not diodem.is_available():
        print("skipped diodem_relative_heading.png: run scripts/download_diodem.py")
    else:
        data = diodem.load_trial(1, hz=40.0)
        res = diodem.six_axis_relative_heading(data, "seg2", "seg3", hz=40.0)
        figures.diodem_figure(res, "arm exp01, hinge seg2 → seg3", OUT / "diodem_relative_heading.png")
        print(f"wrote diodem_relative_heading.png: final VQF-optical difference {res['delta_deg'][-1]:+.1f} deg, "
              f"max |difference| {np.max(np.abs(res['delta_deg'])):.1f} deg over {res['t_s'][-1] / 60:.2f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
