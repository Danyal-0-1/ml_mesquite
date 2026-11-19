"""S3-M3's numbers: how far the skin turns against the bone, and what smoothing does to the specific force, on AMASS
walking sequences of the training split (Step 3 brief §4.2-4.3; ``python -m mesqsim twin-body``)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.amass import amass_dir
from ..paths import REPO_ROOT
from .bodymotion import body_tracks, skin_against_bone_deg
from .physics import imu_truth
from .smplh import load_model, load_smplh_sequence

__all__ = ["ACCEL_RANGE_M_S2", "GYRO_RANGE_RAD_S", "walking_files", "body_report"]

ACCEL_RANGE_M_S2 = 4 * 9.80665  # ±4 g, the range the DMP runs the accelerometer at (Danyal, S3-M0: 8192 LSB/g)
GYRO_RANGE_RAD_S = np.deg2rad(2000.0)  # ±2000 °/s (`assumption` until S3-M4 reads the firmware's setting)


def walking_files(n: int, split: str = "train") -> list[str]:
    """The first ``n`` motion files of ``split`` whose name says walk, in manifest order."""
    m = pd.read_csv(REPO_ROOT / "manifests" / "split_manifest.csv")
    m = m[(m["split"] == split) & m["path"].str.contains("walk", case=False)]
    return [p for p in m["path"].tolist() if (amass_dir() / p).is_file()][:n]


def body_report(n_files: int = 12, cutoffs=(None, 16.0, 8.0, 4.0), rate_hz: float = 225.0) -> dict:
    files = walking_files(n_files)
    models: dict = {}
    angles: dict[str, list] = {}
    force: dict = {str(c): {} for c in cutoffs}
    sat: dict[str, list] = {}
    seconds = 0.0
    for rel in files:
        seq = load_smplh_sequence(amass_dir() / rel)
        model = models.setdefault(seq.gender, load_model(seq.gender))
        tr = body_tracks(seq, model)
        seconds += float(tr.t_s[-1])
        for k, name in enumerate(tr.names):
            angles.setdefault(name, []).append(skin_against_bone_deg(tr, k))
            for c in cutoffs:
                imu = imu_truth(tr.t_s, tr.q[:, k], tr.p[:, k], rate_hz, cutoff_hz=c)
                force[str(c)].setdefault(name, []).append(np.linalg.norm(imu.f_m_s2, axis=1))
                if c == 8.0:
                    a = np.any(np.abs(imu.f_m_s2) > ACCEL_RANGE_M_S2, axis=1)
                    g = np.any(np.abs(imu.omega_rad_s) > GYRO_RANGE_RAD_S, axis=1)
                    sat.setdefault(name, []).append(np.stack([a, g], axis=1))
    out = {"files": files, "seconds": seconds, "rate_hz": rate_hz, "sites": {}}
    for name, parts in angles.items():
        a = np.concatenate(parts)
        s = np.concatenate(sat[name])
        out["sites"][name] = {
            "skin_against_bone_deg": {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                                      "max": float(a.max())},
            "force_m_s2": {c: {"median": float(np.median(np.concatenate(force[c][name]))),
                               "p99": float(np.percentile(np.concatenate(force[c][name]), 99))} for c in force},
            "saturated_share_at_8hz": {"accel_4g": float(s[:, 0].mean()), "gyro_2000dps": float(s[:, 1].mean())},
        }
    return out
