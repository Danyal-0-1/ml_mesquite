"""The 16-byte Mesquite wire frame (brief §3.2), encoded and decoded exactly as the pod firmware does.

Layout, little-endian::

    offset  size  field   meaning
    0, 1    1+1   sync    0xAA 0x55
    2       1     id      pod id 0-16
    3       1     batt    battery percent, clamped to 0-100
    4..11   2x4   qx qy qz qw   int16 = trunc(float32(v) * 32767), saturated to ±32767
    12      2     count   uint16 send counter, +1 per send, wraps 65535 -> 0
    14      2     ms_lo   uint16 pod milliseconds at sample decode, mod 65536

The quaternion is sent x, y, z, w (scalar last) and the pod rebuilds ``w = sqrt(1 - x² - y² - z²)``,
so a frame on the wire always has ``qw >= 0``. :func:`encode_frames` refuses ``qw < 0`` so that a
sign bug upstream fails loudly; canonicalise with :func:`mesqsim.conventions.quat_canonical` first.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FRAME_SIZE",
    "SYNC",
    "Q_SCALE",
    "WRAP",
    "MAX_POD_ID",
    "FRAME_DTYPE",
    "quantize_component",
    "dequantize",
    "encode_frames",
    "decode_frames",
    "parse_stream",
]

FRAME_SIZE = 16
SYNC = b"\xaa\x55"
Q_SCALE = 32767
WRAP = 65536  # count and ms_lo are uint16
MAX_POD_ID = 16

FRAME_DTYPE = np.dtype(
    [
        ("sync0", "u1"),
        ("sync1", "u1"),
        ("pod_id", "u1"),
        ("batt", "u1"),
        ("qx", "<i2"),
        ("qy", "<i2"),
        ("qz", "<i2"),
        ("qw", "<i2"),
        ("count", "<u2"),
        ("ms_lo", "<u2"),
    ]
)
assert FRAME_DTYPE.itemsize == FRAME_SIZE


def quantize_component(v) -> np.ndarray:
    """Firmware quantisation of quaternion components: ``trunc(float32(v) * 32767)`` saturated to ±32767.

    The product is formed in float32, as on the pod. The result is never -32768.
    """
    v32 = np.asarray(v, dtype=np.float32)
    if not np.all(np.isfinite(v32)):
        raise ValueError("cannot quantise non-finite quaternion components")
    prod = v32 * np.float32(Q_SCALE)  # float32 multiply, like the firmware
    return np.clip(np.trunc(prod), -Q_SCALE, Q_SCALE).astype(np.int16)


def dequantize(v_int16) -> np.ndarray:
    """Decode int16 wire components to float64: ``v / 32767``."""
    return np.asarray(v_int16, dtype=np.float64) / Q_SCALE


def encode_frames(pod_id, batt, q_xyzw, count, ms_lo) -> bytes:
    """Encode N frames to ``N * 16`` bytes.

    Parameters
    ----------
    pod_id : int or (N,) ints in 0-16.
    batt : battery percent; floored and clamped to 0-100.
    q_xyzw : (4,) or (N, 4) float quaternions in ``(x, y, z, w)`` order with ``w >= 0``.
    count : send counter (any non-negative integer; reduced mod 65536).
    ms_lo : pod milliseconds at decode (any non-negative integer; reduced mod 65536).
    """
    q = np.atleast_2d(np.asarray(q_xyzw, dtype=np.float64))
    n = q.shape[0]
    pod_id = np.broadcast_to(np.asarray(pod_id, dtype=np.int64), (n,))
    if np.any((pod_id < 0) | (pod_id > MAX_POD_ID)):
        raise ValueError("pod id must be in 0-16")
    qi = quantize_component(q)
    if np.any(qi[:, 3] < 0):
        raise ValueError("qw < 0 on the wire is impossible; canonicalise to w >= 0 before encoding")
    out = np.zeros(n, dtype=FRAME_DTYPE)
    out["sync0"] = SYNC[0]
    out["sync1"] = SYNC[1]
    out["pod_id"] = pod_id
    out["batt"] = np.clip(np.floor(np.broadcast_to(np.asarray(batt, dtype=np.float64), (n,))), 0, 100)
    out["qx"], out["qy"], out["qz"], out["qw"] = qi[:, 0], qi[:, 1], qi[:, 2], qi[:, 3]
    out["count"] = np.mod(np.broadcast_to(np.asarray(count, dtype=np.int64), (n,)), WRAP)
    out["ms_lo"] = np.mod(np.broadcast_to(np.asarray(ms_lo, dtype=np.int64), (n,)), WRAP)
    return out.tobytes()


def decode_frames(buf: bytes) -> np.ndarray:
    """Decode a buffer of whole, aligned frames into a structured array with :data:`FRAME_DTYPE`.

    Raises ``ValueError`` if the length is not a multiple of 16 or a sync word is wrong. Use
    :func:`parse_stream` for raw byte streams that may contain noise.
    """
    if len(buf) % FRAME_SIZE:
        raise ValueError(f"buffer length {len(buf)} is not a multiple of {FRAME_SIZE}")
    frames = np.frombuffer(buf, dtype=FRAME_DTYPE).copy()
    bad = (frames["sync0"] != SYNC[0]) | (frames["sync1"] != SYNC[1])
    if np.any(bad):
        raise ValueError(f"{int(bad.sum())} frame(s) with a wrong sync word")
    return frames


def parse_stream(buf: bytes) -> tuple[np.ndarray, int]:
    """Extract frames from a raw byte stream, resynchronising on ``0xAA 0x55``.

    A candidate frame is accepted when it starts with the sync word and its pod id is 0-16.
    Returns ``(frames, n_skipped_bytes)``.
    """
    data = bytes(buf)
    found: list[bytes] = []
    skipped = 0
    i = 0
    while i + FRAME_SIZE <= len(data):
        if data[i : i + 2] == SYNC and data[i + 2] <= MAX_POD_ID:
            found.append(data[i : i + FRAME_SIZE])
            i += FRAME_SIZE
        else:
            i += 1
            skipped += 1
    skipped += len(data) - i
    frames = np.frombuffer(b"".join(found), dtype=FRAME_DTYPE).copy()
    return frames, skipped
