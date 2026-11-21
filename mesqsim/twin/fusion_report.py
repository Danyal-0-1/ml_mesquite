"""S3-M4's numbers: the fusion's own error during walking, and how much gyroscope bias each filter removes
(Step 3 brief §4.5; ``python -m mesqsim twin-fusion``).

* **Walking, perfect sensors.** Each pod site of the first walking files of the training split (SMPL+H, skin
  frames, :mod:`.bodymotion`), sampled at the chip's nominal 55 Hz: the ideal gyroscope (the mean rate over each
  interval) and the specific force through the 8 Hz smoothing spline. Each filter first sees ``STILL_S`` of the pod
  held still at the walk's first pose, as a pod boots still in its box: a filter started mid-stride aligns to an
  accelerometer reading with walking in it, and VQF's start-up then wanders by up to 40° in tilt and 7° in heading
  (WORKLOG S3-M4). The still start is left out of the statistics. With nothing wrong but the accelerations of
  walking, the tilt error is the filter's own dynamic error, and any heading change comes from it (second order).
* **Walking, the drawn sensor errors** (``twin.sensors``, a fixed seed): the same, through :func:`.sensors.measure`.
  These clips turn a lot (walks there and back), so the heading change is reported per turn of the site about the
  vertical, |Δψ(end)| / (|Θ| / 360°), the unit of the brief's k Θ term, rather than per minute.
* **Bias removal.** A pod lying still, and one turning gently about the vertical (±30°, 0.1 Hz), for 120 s with a
  bias of (0.2, -0.3, 0.5) °/s, each filter with its own bias estimation on: the share of the vertical component
  the filter has estimated after 2, 5, 10, 30, 60 and 119 s, and ψ at the end against -b t without any estimate.
"""

from __future__ import annotations

import numpy as np

from ..conventions import quat_conj, quat_rotate, yaw_quat
from ..data.amass import amass_dir
from ..sim.heading import theta_increments_rad
from .body_report import walking_files
from .bodymotion import body_tracks
from .config import load_twin_config
from .fusion import fuse
from .labels import report_labels
from .physics import imu_truth
from .sensors import draw_sensor_errors, ideal_gyro_rad_s, measure
from .smplh import load_model, load_smplh_sequence

__all__ = ["RATE_HZ", "walking_fusion", "bias_removal", "fusion_report"]

RATE_HZ = 55.0
STILL_S = 3.0
MODES = ("vqf6d", "mahony")


def _summary(x) -> dict:
    x = np.concatenate([np.atleast_1d(v) for v in x]) if isinstance(x, list) else np.asarray(x)
    return {"median": float(np.median(x)), "p95": float(np.percentile(x, 95)), "max": float(x.max())}


def walking_fusion(n_files: int = 12, cutoff_hz: float = 8.0, seed: int = 1) -> dict:
    cfg = load_twin_config()
    rng = np.random.default_rng(seed)
    dt = 1.0 / RATE_HZ
    files = walking_files(n_files)
    models: dict = {}
    acc: dict = {}
    turning: dict = {}
    seconds = 0.0
    for rel in files:
        seq = load_smplh_sequence(amass_dir() / rel)
        tr = body_tracks(seq, models.setdefault(seq.gender, load_model(seq.gender)))
        t = np.arange(tr.t_s[0], tr.t_s[-1], dt)
        seconds += float(t[-1] - t[1])
        for k, name in enumerate(tr.names):
            if name == "Phone":
                continue
            imu = imu_truth(tr.t_s, tr.q[:, k], tr.p[:, k], RATE_HZ, cutoff_hz=cutoff_hz, t_eval=t)
            n0 = int(STILL_S * RATE_HZ)
            q = np.vstack([np.tile(imu.q[1], (n0, 1)), imu.q[1:]])
            gyr = np.vstack([np.zeros((n0, 3)), ideal_gyro_rad_s(imu.q, dt)])
            f = np.vstack([np.tile(quat_rotate(quat_conj(imu.q[1]), [0.0, 0.0, 9.81]), (n0, 1)), imu.f_m_s2[1:]])
            turns = abs(float(np.sum(theta_increments_rad(q[n0:])))) / (2 * np.pi)
            turning.setdefault(name, []).append(turns)
            err = draw_sensor_errors(cfg, rng)
            g_err, f_err, _ = measure(err, gyr, f, dt, rng)
            for mode in MODES:
                for case, (g, a) in (("perfect", (gyr, f)), ("errors", (g_err, f_err))):
                    psi, tilt = report_labels(fuse(cfg, g, a, dt, mode=mode).q, q)
                    psi, tilt = psi[n0:], tilt[n0:]
                    d = np.rad2deg(np.unwrap(psi) - psi[0])
                    slot = acc.setdefault(mode, {}).setdefault(case, {}).setdefault(name, {"tilt": [], "dpsi": [],
                                                                                           "per_turn": []})
                    slot["tilt"].append(np.rad2deg(tilt))
                    slot["dpsi"].append(np.abs(d))
                    slot["per_turn"].append(np.array([abs(d[-1]) / max(turns, 1e-9)]))
    out = {"files": files, "seconds": seconds, "rate_hz": RATE_HZ, "cutoff_hz": cutoff_hz, "seed": seed,
           "still_start_s": STILL_S, "turns": {n: _summary(v) for n, v in turning.items()}, "modes": {}}
    for mode, cases in acc.items():
        out["modes"][mode] = {case: {name: {"tilt_deg": _summary(s["tilt"]), "abs_dpsi_deg": _summary(s["dpsi"]),
                                            "dpsi_per_turn_deg": _summary(s["per_turn"])}
                                     for name, s in sites.items()} for case, sites in cases.items()}
    return out


def bias_removal(seconds: float = 120.0) -> dict:
    cfg = load_twin_config()
    dt = 1.0 / RATE_HZ
    t = np.arange(0.0, seconds, dt)
    b = np.deg2rad(np.array([0.2, -0.3, 0.5]))
    marks = [2.0, 5.0, 10.0, 30.0, 60.0, seconds - 1.0]
    out = {"bias_deg_s": np.rad2deg(b).tolist(), "marks_s": marks}
    for case, th in (("still", np.zeros(t.size)), ("turning", np.deg2rad(30.0) * np.sin(2 * np.pi * 0.1 * t))):
        r = yaw_quat(th)
        gyr = ideal_gyro_rad_s(np.vstack([r[:1], r]), dt) + b
        acc = quat_rotate(quat_conj(r), np.array([0.0, 0.0, 9.81]))
        for mode in MODES:
            res = fuse(cfg, gyr, acc, dt, mode=mode)
            psi, _ = report_labels(res.q, r)
            share = res.bias_rad_s[[int(m * RATE_HZ) for m in marks]] / b
            out.setdefault(case, {})[mode] = {
                "vertical_share": share[:, 2].tolist(), "horizontal_share_x": share[:, 0].tolist(),
                "psi_end_deg": float(np.rad2deg(np.unwrap(psi)[-1] - psi[0])),
                "psi_end_without_estimate_deg": float(-np.rad2deg(b[2]) * (t[-1] - t[0]))}
    return out


def fusion_report(n_files: int = 12) -> dict:
    return {"walking": walking_fusion(n_files), "bias_removal": bias_removal()}
