"""The model kit's stored inputs (Step 3 brief §7.1): causal, streaming-equal, and blind to a common heading (C5)."""

import numpy as np
import pytest

from mesqsim.conventions import exp_map, quat_mul, yaw_quat
from mesqsim.learn.features import _track_at, six_d


def test_six_d_is_the_first_two_columns():
    r = np.arange(9.0).reshape(3, 3)
    assert np.array_equal(six_d(r), [0.0, 3.0, 6.0, 1.0, 4.0, 7.0])


def test_track_interpolates_what_has_arrived_and_holds_otherwise():
    """Samples every 31.25 ms arriving 20 ms late: at t_k - 0.1 s the two bracketing samples have arrived, so the
    orientation is their slerp; the rate is between the last two; the age is t_k minus the last arrival."""
    tau = np.arange(40) * 0.03125
    arr = tau + 0.02
    w = np.array([0.0, 0.0, 2.0])  # rad/s about z
    qs = exp_map(np.outer(tau, w))
    t_k = np.array([0.5, 0.51, 1.0])
    q, rate, age, n_arr, last, has = _track_at(arr, tau, qs, np.zeros(40, np.int64), t_k, 0.1)
    assert np.allclose(q, exp_map(np.outer(t_k - 0.1, w)), atol=1e-12)
    assert np.allclose(rate, w, atol=1e-9)
    assert np.allclose(age, t_k - arr[last])
    # with no lag, the sample after t has not arrived: hold the latest that has (tau 0.46875 s, arrived at 0.48875 s)
    q, *_ = _track_at(arr, tau, qs, np.zeros(40, np.int64), np.array([0.5]), 0.0)
    assert np.array_equal(q[0], qs[15])


@pytest.fixture(scope="module")
def session():
    from mesqsim.twin import session_checks as sc

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    return sc


def _features(s):
    from mesqsim.config import load_config
    from mesqsim.learn.features import features
    from mesqsim.paths import REPO_ROOT
    from mesqsim.step0 import decode

    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    t = s.labels_steps["t_s"].to_numpy() + s.meta["plan"]["host_epoch_s"]
    series = decode(s.frames, s.session)
    return features(series, s.phone, t, cfg["geometry.clock_window_s"], cfg["geometry.clock_block_s"]), t, cfg


@pytest.mark.amass
def test_features_equal_a_streaming_recomputation(session):
    from mesqsim.learn.features import features
    from mesqsim.step0 import decode

    s = session.short_session()
    f, t, cfg = _features(s)
    assert f.x_pod.shape == (t.size, 15, 8) and f.x_joint.shape == (t.size, 14, 6) and f.x_phone.shape == (t.size, 8)
    assert np.isfinite(f.x_pod).all() and f.x_pod.dtype == np.float32
    for k in (400, 1500, t.size - 1):
        cut = s.frames[s.frames["t_host_s"] <= t[k]]
        g = features(decode(cut, s.session), s.phone[s.phone["t_host_s"] <= t[k]], t[: k + 1],
                     cfg["geometry.clock_window_s"], cfg["geometry.clock_block_s"])  # fmt: skip
        pairs = ((f.x_pod, g.x_pod), (f.x_joint, g.x_joint), (f.x_phone, g.x_phone), (f.x_pod_phone, g.x_pod_phone))
        for a, b in pairs:
            assert np.array_equal(a[k], b[k])
        assert np.array_equal(f.state_pod[k], g.state_pod[k]) and f.state_phone[k] == g.state_phone[k]


@pytest.mark.amass
def test_c5_inputs_are_blind_to_a_common_turn(session):
    """Check C5 (Step 3 brief §7.1, §10 item 12): every pod and the phone turned by one heading, the inputs unchanged
    to machine precision (the wire's int16 rounds a turned report differently, so the turn is applied to the decoded
    series here, exactly as a turned world would deliver it)."""
    from mesqsim.config import load_config
    from mesqsim.learn.features import features
    from mesqsim.paths import REPO_ROOT
    from mesqsim.step0 import decode

    s = session.short_session()
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    t = s.labels_steps["t_s"].to_numpy() + s.meta["plan"]["host_epoch_s"]
    series = decode(s.frames, s.session)
    turned = series.copy()
    q = series[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    ok = np.isfinite(q[:, 3])
    q[ok] = quat_mul(yaw_quat(np.full(ok.sum(), 0.9)), q[ok])
    turned[["qx", "qy", "qz", "qw"]] = q
    phone = s.phone.copy()
    half = 0.9 / 2  # the same turn about the reported frame's +Y (ARKit's up), as a rotation within that frame
    qp = phone[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    phone[["qx", "qy", "qz", "qw"]] = np.stack([np.cos(half) * qp[:, 0] + np.sin(half) * qp[:, 2],
                                                np.cos(half) * qp[:, 1] + np.sin(half) * qp[:, 3],
                                                np.cos(half) * qp[:, 2] - np.sin(half) * qp[:, 0],
                                                np.cos(half) * qp[:, 3] - np.sin(half) * qp[:, 1]], axis=1)
    cw, bs = cfg["geometry.clock_window_s"], cfg["geometry.clock_block_s"]
    a, b = features(series, s.phone, t, cw, bs), features(turned, phone, t, cw, bs)
    for x, y in ((a.x_pod, b.x_pod), (a.x_joint, b.x_joint), (a.x_phone, b.x_phone), (a.x_pod_phone, b.x_pod_phone)):
        assert np.abs(x.astype(float) - y.astype(float)).max() < 1e-6  # float32 storage: machine precision
