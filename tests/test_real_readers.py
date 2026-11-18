"""The BVH and Motive readers of mesqsim.real, on small synthetic files written in the formats of the recordings."""

import datetime as dt

import numpy as np
import pytest

from mesqsim.real.bvh import bvh_facing, forward_kinematics, pod_bones, read_bvh
from mesqsim.real.motive import read_motive_csv
from mesqsim.real.recordings import bvh_end_local

BVH = """HIERARCHY
ROOT Hips
{
  OFFSET 0 100 0
  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    End Site
    {
      OFFSET 0 0 0
    }
  }
  JOINT LeftUpLeg
  {
    OFFSET 10 -5 0
    CHANNELS 3 Xrotation Yrotation Zrotation
    End Site
    {
      OFFSET 0 -40 0
    }
  }
}
MOTION
Frames: 3
Frame Time: 0.033
; MESQ_CAPTURE_START 2026-11-15T00:49:16.277Z
; MESQ_CAPTURE_END   2026-11-15T00:49:16.376Z
; MESQ_FRAME_TIME_SOURCE measured
0 100 0 0 0 0 0 0 0 0 0 0
0 100 0 0 90 0 0 0 0 0 0 0
1 100 2 0 0 0 0 0 0 0 0 0
"""


@pytest.fixture
def bvh_file(tmp_path):
    p = tmp_path / "MMcap_bvh_2026-11-14-17-52-25.bvh"
    p.write_text(BVH, encoding="utf-8")
    return p


def test_bvh_hierarchy_motion_and_provenance(bvh_file):
    b = read_bvh(bvh_file)
    assert b.names == ["Hips", "Spine", "LeftUpLeg"]
    assert [j.parent for j in b.joints] == [-1, 0, 0]
    assert [j.first_channel for j in b.joints] == [0, 6, 9]
    assert np.allclose(b.joints[2].offset, [10, -5, 0]) and np.allclose(b.joints[2].end_site, [0, -40, 0])
    assert b.n_frames == 3 and b.motion.shape == (3, 12) and b.frame_time_s == 0.033
    assert b.frame_time_source == "measured"
    assert b.capture_start_utc == dt.datetime(2026, 11, 15, 0, 49, 16, 277000, tzinfo=dt.UTC)
    assert bvh_end_local(str(bvh_file)) == dt.datetime(2026, 11, 14, 17, 52, 25)
    assert pod_bones(b) == ["Hips"]  # the only pod bone whose rotation channels are not all zero


def test_older_page_asserts_its_frame_time(tmp_path):
    text = BVH.replace("Frame Time: 0.033", "Frame Time: 0.03333333333333333")
    text = "\n".join(line for line in text.splitlines() if not line.startswith(";")) + "\n"
    p = tmp_path / "old.bvh"
    p.write_text(text, encoding="utf-8")
    b = read_bvh(p)
    assert b.frame_time_source == "asserted" and b.capture_start_utc is None
    p.write_text(BVH.replace("measured", "ASSUMED_1_30_UNTRUSTWORTHY"), encoding="utf-8")  # the page's own words
    assert read_bvh(p).frame_time_source == "asserted"


def test_bvh_rejects_a_short_motion(tmp_path):
    p = tmp_path / "bad.bvh"
    p.write_text(BVH.replace("Frames: 3", "Frames: 4"), encoding="utf-8")
    with pytest.raises(ValueError):
        read_bvh(p)


def test_forward_kinematics_and_facing(bvh_file):
    b = read_bvh(bvh_file)
    pos, _ = forward_kinematics(b)
    assert np.allclose(pos[0, 2], [10, 95, 0])  # the root's position channels, then the offset
    assert np.allclose(pos[1, 2], [0, 95, -10])  # root turned 90° about +Y: (10, -5, 0) -> (0, -5, -10)
    assert np.allclose(pos[2, 0], [1, 100, 2])
    f = bvh_facing(b)  # the avatar's +Z is world -Y; turned +90° about up (anticlockwise) it is world +X
    assert np.allclose(f[0], [0, -1]) and np.allclose(f[1], [1, 0], atol=1e-12)


MOTIVE = (
    "Format Version,1.25,Take Name,Test,Take Notes,,Capture Frame Rate,120.000000,Export Frame Rate,120.000000,"
    "Capture Start Time,2026-11-13 01.41.56.863 PM,Capture Start Frame,1,Total Frames in Take,3,"
    "Total Exported Frames,3,Rotation Type,Quaternion,Length Units,Meters,Coordinate Space,Global\n"
    "\n"
    ",Type,Marker,Marker,Marker,Marker,Marker,Marker\n"
    ",Name,Unlabeled 1,Unlabeled 1,Unlabeled 1,Unlabeled 2,Unlabeled 2,Unlabeled 2\n"
    ',ID,"1:A","1:A","1:A","2:A","2:A","2:A"\n'
    ",Parent,,,,,,\n"
    ",,Position,Position,Position,Position,Position,Position\n"
    "Frame,Time (Seconds),X,Y,Z,X,Y,Z\n"
    "0,0.000000,0.1,0.2,1.7,,,\n"
    "1,0.008333,0.1,0.2,1.7,0.5,0.6,0.3\n"
    "2,0.016667,,,,0.5,0.6,0.3\n"
)


def test_motive_csv(tmp_path):
    p = tmp_path / "take.csv"
    p.write_text(MOTIVE, encoding="utf-8")
    m = read_motive_csv(p)
    assert m.names == ["Unlabeled 1", "Unlabeled 2"] and m.frame_rate_hz == 120.0
    assert m.capture_start_local == dt.datetime(2026, 11, 13, 13, 41, 56, 863000)
    assert m.xyz.shape == (3, 2, 3)
    assert np.allclose(m.xyz[1, 0], [0.1, 0.2, 1.7]) and np.isnan(m.xyz[0, 1]).all() and np.isnan(m.xyz[2, 0]).all()
    assert np.allclose(m.t_s, [0.0, 0.008333, 0.016667])


def test_motive_rejects_other_units(tmp_path):
    p = tmp_path / "mm.csv"
    p.write_text(MOTIVE.replace("Length Units,Meters", "Length Units,Millimeters"), encoding="utf-8")
    with pytest.raises(ValueError):
        read_motive_csv(p)
