"""The pods' wire as the page sees it (Step 3 brief §4.6; check 44.1): the binary frame and the old JSON line."""

import json
import re

import numpy as np
import pytest

from mesqsim.codec import Q_SCALE
from mesqsim.paths import data_dir
from mesqsim.twin.wire import BONE_NAMES, binary16_bytes, binary16_messages, json_line, json_message, json_number

GOLDEN_HEX = "aa550257ff3f01c0ff1fff5f3412cdab"  # Step 1's golden frame (tests/test_codec.py)


def test_golden_frame_through_the_twins_wire():
    buf = binary16_bytes(2, 87, (0.5, -0.5, 0.25, 0.75), 0x1234, 0xABCD)
    assert buf.hex() == GOLDEN_HEX
    (m,) = binary16_messages(buf)
    assert m == {"bone": "HipsAlt", "id": 2, "batt": 0.87, "x": 16383 / Q_SCALE, "y": -16383 / Q_SCALE,
                 "z": 8191 / Q_SCALE, "w": 24575 / Q_SCALE, "count": 0x1234, "millis": 0xABCD}


def test_bone_names_are_the_parsers():
    js = data_dir() / "mesquite-mocap" / "js" / "mesq_parser.js"
    if not js.is_file():
        pytest.skip("needs the mesquite-mocap clone")
    block = re.search(r"var BONE_NAMES = \[(.*?)\];", js.read_text(encoding="utf-8"), re.S).group(1)
    assert tuple(re.findall(r'"(\w+)"', block)) == BONE_NAMES


@pytest.mark.parametrize("v, d, s", [(0.5, 2, "0.50"), (1.0, 2, "1.00"), (-0.004, 2, "-0.00"), (0.125, 2, "0.13"),
                                     (-0.996, 2, "-1.00"), (0.12344999, 4, "0.1234"), (-0.5, 0, "-1"),
                                     (0.0, 3, "0.000"), (0.99999, 2, "1.00")])
def test_json_number_prints_like_the_old_dongle(v, d, s):
    assert json_number(v, d) == s


def test_json_line_is_the_old_dongles_and_parses():
    q = np.array([0.12345, -0.5, 0.0004, 0.8631])
    line = json_line(9, q, 77, 123456, 87, 2)
    assert line == ('{"id":"02:00:00:00:00:09", "count":"77", "millis":"123456", "bone":"LeftUpLeg", '
                    '"x":0.12, "y":-0.50, "z":0.00, "w":0.86, "batt":0.87}')
    m = json_message(line)
    assert m["bone"] == "LeftUpLeg" and m["count"] == "77" and m["x"] == 0.12 and m["batt"] == 0.87
    rng = np.random.default_rng(0)
    for d in (2, 3, 4, 6):
        for qi in rng.normal(size=(200, 4)):
            qi = qi / np.linalg.norm(qi)
            m = json.loads(json_line(3, qi, 0, 0, 50, d))
            got = np.array([m[k] for k in "xyzw"])
            assert np.abs(got - qi.astype(np.float32)).max() <= 0.5 * 10.0**-d + 1e-12
