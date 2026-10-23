"""Check 1 (brief §9): the wire codec reproduces the firmware frame byte for byte."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mesqsim.codec import (
    FRAME_SIZE,
    Q_SCALE,
    decode_frames,
    dequantize,
    encode_frames,
    parse_stream,
    quantize_component,
)
from mesqsim.conventions import quat_canonical, quat_normalize

GOLDEN_HEX = "aa550257ff3f01c0ff1fff5f3412cdab"
GOLDEN_Q = (0.5, -0.5, 0.25, 0.75)  # x, y, z, w (not unit length; codec test only)


def test_golden_vector_encode():
    buf = encode_frames(pod_id=2, batt=87, q_xyzw=GOLDEN_Q, count=0x1234, ms_lo=0xABCD)
    assert buf.hex() == GOLDEN_HEX


def test_golden_vector_decode():
    f = decode_frames(bytes.fromhex(GOLDEN_HEX))
    assert f.shape == (1,)
    f = f[0]
    assert (f["pod_id"], f["batt"], f["count"], f["ms_lo"]) == (2, 87, 0x1234, 0xABCD)
    assert (f["qx"], f["qy"], f["qz"], f["qw"]) == (16383, -16383, 8191, 24575)
    q = dequantize([f["qx"], f["qy"], f["qz"], f["qw"]])
    np.testing.assert_allclose(q, GOLDEN_Q, atol=1.0 / Q_SCALE)


def test_saturation_never_minus_32768():
    vals = quantize_component([1.0, 1.5, -1.0, -1.5, 2.0, -2.0, 1e9, -1e9])
    assert vals.tolist() == [32767, 32767, -32767, -32767, 32767, -32767, 32767, -32767]


def test_truncation_toward_zero_in_float32():
    # float32(0.1) * 32767 = 3276.7002 -> 3276 ; -0.1 -> -3276 (toward zero, not floor)
    assert quantize_component([0.1, -0.1]).tolist() == [3276, -3276]
    # 0.5 * 32767 = 16383.5 exactly in float32 -> 16383
    assert quantize_component([0.5, -0.5]).tolist() == [16383, -16383]


def test_battery_clamped():
    q = (0.0, 0.0, 0.0, 1.0)
    for batt, expected in [(150, 100), (100, 100), (87, 87), (0, 0), (-5, 0), (99.9, 99)]:
        assert decode_frames(encode_frames(0, batt, q, 0, 0))[0]["batt"] == expected


@pytest.mark.parametrize(
    "value, expected", [(65535, 65535), (65536, 0), (65537, 1), (3 * 65536 + 7, 7), (0, 0)]
)
def test_count_and_ms_lo_wrap(value, expected):
    f = decode_frames(encode_frames(1, 50, (0, 0, 0, 1), count=value, ms_lo=value))[0]
    assert f["count"] == expected
    assert f["ms_lo"] == expected


def test_rejects_negative_w_and_bad_ids():
    with pytest.raises(ValueError):
        encode_frames(0, 50, (0.0, 0.0, 0.6, -0.8), 0, 0)
    with pytest.raises(ValueError):
        encode_frames(17, 50, (0, 0, 0, 1), 0, 0)
    with pytest.raises(ValueError):
        encode_frames(0, 50, (np.nan, 0, 0, 1), 0, 0)


def test_vectorised_matches_single_frames():
    rng = np.random.default_rng(0)
    q = quat_canonical(quat_normalize(rng.normal(size=(20, 4))))
    ids = rng.integers(0, 15, 20)
    counts = rng.integers(0, 200000, 20)
    ms = rng.integers(0, 200000, 20)
    batch = encode_frames(ids, 77, q, counts, ms)
    single = b"".join(encode_frames(ids[i], 77, q[i], counts[i], ms[i]) for i in range(20))
    assert batch == single
    assert len(batch) == 20 * FRAME_SIZE


def test_parse_stream_resyncs_after_noise():
    good = encode_frames([0, 1, 2], 60, [(0, 0, 0, 1)] * 3, [10, 11, 12], [100, 120, 140])
    noisy = b"\x00\x13" + good[:16] + b"\xaa\x01\x55" + good[16:] + b"\xaa"
    frames, skipped = parse_stream(noisy)
    assert frames["pod_id"].tolist() == [0, 1, 2]
    assert frames["count"].tolist() == [10, 11, 12]
    assert skipped == len(noisy) - 3 * FRAME_SIZE


def test_decode_rejects_bad_sync_and_length():
    with pytest.raises(ValueError):
        decode_frames(bytes.fromhex(GOLDEN_HEX)[:15])
    with pytest.raises(ValueError):
        decode_frames(b"\xab" + bytes.fromhex(GOLDEN_HEX)[1:])


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(-1.0, 1.0), min_size=4, max_size=4).filter(lambda v: np.linalg.norm(v) > 0.1))
def test_roundtrip_error_below_one_step(v):
    q = quat_canonical(quat_normalize(np.array(v)))
    f = decode_frames(encode_frames(5, 50, q, 0, 0))[0]
    back = dequantize([f["qx"], f["qy"], f["qz"], f["qw"]])
    # truncation toward zero: error below one quantisation step, never increasing the magnitude
    assert np.all(np.abs(back - q) < 1.0 / Q_SCALE + 1e-7)
    assert np.all(np.abs(back) <= np.abs(q) + 1e-7)
