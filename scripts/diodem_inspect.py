"""Reproduce the DIODEM facts quoted in docs/DATA.md §2.3 and §2.5.

    python scripts/diodem_inspect.py            # exp01 (arm) and exp06 (gait)

For each experiment it prints:
* frame agreement: mean cosine between the gravity direction of six-axis VQF (rigid IMU) and of the optical
  segment orientation, with no mounting rotation; per-axis correlation of gyroscope and optical body rates;
* chain adjacency: for every segment pair, the median distance between the nearest markers, and the share of
  relative angular-velocity variance on its first principal axis (about 1 for a hinge);
* gyroscope offsets: least-squares scale and bias of each gyroscope axis against the optical body rate.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
from vqf import VQF  # noqa: E402

from mesqsim.conventions import (  # noqa: E402
    gravity_in_body,
    log_map,
    quat_conj,
    quat_mul,
    quat_rotate,
    wxyz_to_xyzw,
)
from mesqsim.data import diodem  # noqa: E402

HZ = 40.0


def body_rate(q: np.ndarray) -> np.ndarray:
    """Body-frame angular velocity (rad/s) of an orientation series, central differences, length N-2."""
    return log_map(quat_mul(quat_conj(q[:-2]), q[2:])) * HZ / 2


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if not diodem.is_available():
        print("DIODEM not downloaded; run scripts/download_diodem.py")
        return 1
    for exp in (1, 6):
        d = diodem.load_trial(exp, HZ)
        segs = sorted(d)
        big_q = {s: wxyz_to_xyzw(d[s]["quat"]) for s in segs}
        n = len(big_q[segs[0]])
        print(f"\nexp{exp:02d}: {n} samples at {HZ:g} Hz = {n / HZ / 60:.2f} min")
        print("  frame agreement (rigid IMU vs optical, no mounting rotation):")
        for s in segs:
            imu = d[s]["imu_rigid"]
            q = wxyz_to_xyzw(VQF(1 / HZ).updateBatch(np.ascontiguousarray(imu["gyr"]),
                                                      np.ascontiguousarray(imu["acc"]))["quat6D"])
            cos = np.mean(np.sum(gravity_in_body(q) * gravity_in_body(big_q[s]), axis=1))
            w, g = body_rate(big_q[s]), imu["gyr"][1:-1]
            corr = [np.corrcoef(g[:, k], w[:, k])[0, 1] for k in range(3)]
            fits = []
            for k in range(3):
                (scale, bias), *_ = np.linalg.lstsq(np.c_[w[:, k], np.ones(len(w))], g[:, k], rcond=None)
                fits.append(f"{scale:.4f}/{np.rad2deg(bias):+.2f}")
            print(f"    {s}: gravity cos {cos:.3f}; gyro-optical corr x,y,z {np.round(corr, 3).tolist()}; "
                  f"gyro scale/bias(deg/s) {', '.join(fits)}")
        print("  pairs: median nearest-marker gap (mm), relative angular velocity PC1 share")
        for a, b in itertools.combinations(segs, 2):
            ma = np.stack([d[a][f"marker{k}"] for k in range(1, 5)], 1)
            mb = np.stack([d[b][f"marker{k}"] for k in range(1, 5)], 1)
            gap = np.median(np.min(np.linalg.norm(ma[:, :, None] - mb[:, None], axis=-1).reshape(n, -1), axis=1))
            rel = quat_mul(quat_conj(big_q[a]), big_q[b])
            w_par = quat_rotate(rel[1:-1], body_rate(rel))
            ev = np.linalg.eigvalsh(np.cov(w_par.T))[::-1]
            print(f"    {a}-{b}: gap {gap * 1000:5.0f} mm, PC1 {ev[0] / ev.sum():.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
