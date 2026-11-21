"""The pods' wire, as far as the page sees it (Step 3 brief §4.6): ``twin.wire.mode``.

* ``binary16``, the new suit: Step 1's 16-byte frame (:mod:`mesqsim.codec`, ``Pod_Watch_Binary.ino``), which the
  page's parser turns into ``{bone, id, batt, x, y, z, w, count, millis}`` with each component ``int16 / 32767``
  and ``batt`` / 100 (``js/mesq_parser.js`` ``unpackPodPacket``, lines 101-117 at ``87d7c3b``); ``millis`` is the
  pod's decode millisecond, mod 65536.
* ``json``, the old suit (``Device code/``): the pod sends float32 components over ESP-NOW and the dongle prints
  one line per frame (``Dongle__JSON.ino:59``)::

      {"id":"<mac>", "count":"<c>", "millis":"<m>", "bone":"<bone>", "x":<x>, "y":<y>, "z":<z>, "w":<w>, "batt":<b>}

  The numbers are Arduino ``String`` concatenations of floats: ``twin.wire.json_decimals`` decimals, rounded half
  away from zero, a negative value keeping its sign even when it rounds to zero ("-0.00"). That is how the Arduino
  core's ``dtostrf`` prints (it adds half a unit, then truncates); the core is not pinned in the repository, so the
  rule is an ``assumption`` that matters only at exact ties. ``count`` and ``millis`` are quoted, ``millis`` is the
  pod's clock at the send (``Pod_Watch_JSON.ino:509-512``), ``batt`` the pod's percent / 100. The MAC is synthetic.
  The old firmware also put a ``String`` object inside a ``memcpy``'d struct, so its ``bone`` field could arrive as
  junk that the page's parser repairs (``mesq_parser.js:62-76``); the twin writes the name cleanly.

The old pod's send rule (``Pod_Watch_JSON.ino:497``) is ``millis() > last + 1000 / fps``, fps = 32: every 32 ms of
its clock (``twin.wire.json_send_min_ms``), so its send grid is Step 1's with a 32 ms period.
"""

from __future__ import annotations

import json
import math

import numpy as np

from ..codec import Q_SCALE, decode_frames, encode_frames

__all__ = ["BONE_NAMES", "binary16_bytes", "binary16_messages", "json_number", "json_line", "json_message"]

BONE_NAMES = ("Head", "Spine", "HipsAlt", "LeftArm", "LeftForeArm", "LeftHand", "RightArm", "RightForeArm",
              "RightHand", "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot", "LeftShoulder",
              "RightShoulder")
"""Pod id -> bone, ``js/mesq_parser.js`` ``BONE_NAMES`` (= ``Pod_Watch_Binary.ino``'s ``boneName[]``)."""


def binary16_bytes(pod_id, batt, q_xyzw, count, ms_lo) -> bytes:
    """Frames on the wire, exactly as the pod encodes them (:func:`mesqsim.codec.encode_frames`)."""
    return encode_frames(pod_id, batt, q_xyzw, count, ms_lo)


def binary16_messages(buf: bytes) -> list[dict]:
    """What the page's parser hands ``handleWSMessage`` for whole, aligned frames ``buf``."""
    f = decode_frames(buf)
    out = []
    for r in f:
        pod = int(r["pod_id"])
        out.append({"bone": BONE_NAMES[pod], "id": pod, "batt": int(r["batt"]) / 100,
                    "x": int(r["qx"]) / Q_SCALE, "y": int(r["qy"]) / Q_SCALE, "z": int(r["qz"]) / Q_SCALE,
                    "w": int(r["qw"]) / Q_SCALE, "count": int(r["count"]), "millis": int(r["ms_lo"])})  # fmt: skip
    return out


def json_number(v: float, decimals: int) -> str:
    """A float as the old dongle prints it: ``decimals`` decimals, half away from zero, the sign always kept."""
    v = float(v)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    scale = 10**decimals
    n = math.floor(abs(v) * scale + 0.5)
    whole, frac = divmod(n, scale)
    s = ("-" if v < 0 else "") + str(whole)
    return s + ("." + str(frac).zfill(decimals) if decimals > 0 else "")


def json_line(pod_id: int, q_xyzw, count: int, millis: int, batt_pct: int, decimals: int) -> str:
    """One old-dongle line (no newline). ``q_xyzw`` is converted to float32 first, as the pod's struct holds it."""
    x, y, z, w = (float(c) for c in np.asarray(q_xyzw, dtype=np.float32))
    mac = f"02:00:00:00:00:{int(pod_id):02x}"
    num = lambda v: json_number(v, decimals)  # noqa: E731
    return (f'{{"id":"{mac}", "count":"{int(count)}", "millis":"{int(millis)}", "bone":"{BONE_NAMES[pod_id]}", '
            f'"x":{num(x)}, "y":{num(y)}, "z":{num(z)}, "w":{num(w)}, "batt":{json_number(batt_pct / 100, 2)}}}')


def json_message(line: str) -> dict:
    """The object the page's parser hands on for a well-formed line (``JSON.parse``)."""
    return json.loads(line)
