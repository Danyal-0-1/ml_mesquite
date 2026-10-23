"""Step 0 on hand-built frame tables, independent of the simulator (it will also read real recordings)."""

import numpy as np
import pandas as pd

from mesqsim.codec import quantize_component
from mesqsim.recording import knowable_ticks, merge_recorded
from mesqsim.step0 import SEND_PERIOD_S, decode

T = SEND_PERIOD_S


def frames(t_host, count, ms_lo, q=None, pod=3):
    n = len(t_host)
    qi = quantize_component(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)) if q is None else np.asarray(q))
    return pd.DataFrame({
        "t_host_s": np.asarray(t_host, float), "pod_id": np.full(n, pod, np.uint8), "batt": np.full(n, 80, np.uint8),
        "qx": qi[:, 0], "qy": qi[:, 1], "qz": qi[:, 2], "qw": qi[:, 3],
        "count": np.asarray(count, np.int64).astype(np.uint16), "ms_lo": np.asarray(ms_lo, np.int64).astype(np.uint16),
    })  # fmt: skip


def test_count_wrap_is_continuous():
    c = [65533, 65534, 65535, 0, 1]
    ms = [(1000 + k * 31) % 65536 for k in range(5)]
    s = decode(frames(100 + np.arange(5) * T, c, ms))
    assert s["tick"].tolist() == [65533, 65534, 65535, 65536, 65537]
    assert (s["epoch"] == 0).all() and (s["status"] == "fresh").all()


def test_lost_ticks_and_held_frames():
    t = 100 + np.array([0, 1, 4, 5]) * T
    s = decode(frames(t, [10, 11, 14, 15], [1300, 1331, 1425, 1425]))
    assert s["tick"].tolist() == [10, 11, 12, 13, 14, 15]
    assert s["status"].tolist() == ["fresh", "fresh", "lost", "lost", "fresh", "held"]
    np.testing.assert_allclose(s.loc[s["status"] == "lost", "t_host_est_s"], 100 + np.array([2, 3]) * T)


def test_reboot_starts_a_new_epoch_and_its_early_ticks_are_lost():
    t = np.r_[100 + np.arange(3) * T, 100 + 2 * T + 4.0 + np.arange(2) * T]
    s = decode(frames(t, [500, 501, 502, 3, 4], [17000, 17031, 17062, 1100, 1131]))
    assert s["epoch"].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]
    assert s.loc[s["epoch"] == 1, "tick"].tolist() == [0, 1, 2, 3, 4]
    assert s.loc[s["epoch"] == 1, "status"].tolist() == ["lost", "lost", "lost", "fresh", "fresh"]


def test_long_gap_with_count_wraps_is_not_a_reboot():
    gap_ticks = 70000  # more than one count wrap, 36.5 minutes
    t = np.r_[100.0, 100 + T, 100 + T + gap_ticks * T]
    c = np.array([0, 1, 1 + gap_ticks]) % 65536
    ms = np.array([1000, 1031, 1031 + gap_ticks * 31.25]).astype(np.int64) % 65536
    session = {"phases": [
        {"name": "a", "t_host_start_s": 99.0, "t_host_end_s": 101.0, "recorded": True},
        {"name": "b", "t_host_start_s": 2100.0, "t_host_end_s": 2400.0, "recorded": True},
    ]}
    s = decode(frames(t, c, ms), session)
    assert s["tick"].tolist() == [0, 1, 1 + gap_ticks]  # the unrecorded gap inserts no lost rows
    assert (s["epoch"] == 0).all()
    assert abs(s["ms_unwrapped"].iloc[2] - s["ms_unwrapped"].iloc[1] - gap_ticks * 31.25) < 1.0


def test_norm_check_and_canonical_sign():
    q = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.9], [-0.6, 0.0, 0.8, 0.0]])
    s = decode(frames(100 + np.arange(3) * T, [0, 1, 2], [1000, 1031, 1062], q))
    assert s["valid"].tolist() == [True, False, True]
    assert np.isnan(s.loc[1, "qw"])
    np.testing.assert_allclose(s.loc[2, ["qx", "qy", "qz", "qw"]].to_numpy(float), [-0.6, 0, 0.8, 0], atol=1e-4)


def test_merge_recorded_and_knowable_ticks():
    phases = [("box", 0, 5, True), ("strap", 5, 30, True), ("idle", 30, 90, False), ("motion", 90, 99, True)]
    iv = merge_recorded(phases)
    assert iv == [(0, 30), (90, 99)]
    delivered = np.array([0, 1, 0, 1, 0, 0, 1, 0], bool)
    interval = np.array([0, 0, 0, 0, -1, 1, 1, 1])
    epoch = np.zeros(8, int)
    np.testing.assert_array_equal(knowable_ticks(delivered, epoch, interval), [0, 1, 1, 1, 0, 0, 1, 0])
    # A reboot between ticks 4 and 5, all one recorded interval: tick 4 was the old epoch's last send, lost just
    # before the reboot, so the host cannot know it; tick 5 is the new epoch's first send, before its first
    # received frame, so it is known to be lost.
    reboot_epoch = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    np.testing.assert_array_equal(knowable_ticks(delivered, reboot_epoch, np.zeros(8, int)), [0, 1, 1, 1, 0, 1, 1, 0])
