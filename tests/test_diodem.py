"""Check 11 (brief §9): DIODEM files, rates, quaternion norms and the six-axis sanity analysis.

Unit tests always run; tests marked ``diodem`` use the download and skip without it.
"""

import hashlib

import numpy as np
import pytest

from mesqsim.data import diodem
from mesqsim.data.net import download_file
from mesqsim.manifest import ManifestRow, read_manifest, record_checksums, verify_manifest, write_manifest
from mesqsim.paths import data_dir, manifest_dir


class FakeResponse:
    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, n):
        for i in range(0, len(self.payload), n):
            yield self.payload[i : i + n]


class FakeSession:
    def __init__(self, payload: bytes):
        self.payload = payload

    def get(self, url, stream=True, timeout=None):
        return FakeResponse(self.payload)


def test_download_verifies_md5_and_size(tmp_path):
    payload = b"x" * 3_000_000
    md5 = hashlib.md5(payload).hexdigest()
    size, sha = download_file(FakeSession(payload), "u", tmp_path / "a" / "f.csv", len(payload), md5)
    assert size == len(payload) and sha == hashlib.sha256(payload).hexdigest()
    with pytest.raises(OSError, match="md5"):
        download_file(FakeSession(payload), "u", tmp_path / "g.csv", len(payload), "0" * 32)
    with pytest.raises(OSError, match="size"):
        download_file(FakeSession(payload), "u", tmp_path / "h.csv", 5, md5)
    assert not (tmp_path / "g.csv").exists() and not (tmp_path / "g.csv.part").exists()


def test_manifest_roundtrip_and_verification(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.bin").write_bytes(b"abc")
    (tmp_path / "d" / "b.bin").write_bytes(b"defg")
    man = tmp_path / "m.csv"
    write_manifest(man, [ManifestRow("d/a.bin", None, ""), ManifestRow("d/b.bin", 4, "0" * 64),
                         ManifestRow("d/c.bin", None, "")])
    status = {p: s for p, s, _ in verify_manifest(man, tmp_path)}
    assert status == {"d/a.bin": "unrecorded", "d/b.bin": "MISMATCH", "d/c.bin": "missing"}
    assert record_checksums(man, tmp_path) == 1
    row = {r.path: r for r in read_manifest(man)}["d/a.bin"]
    assert row.size_bytes == 3 and row.sha256 == hashlib.sha256(b"abc").hexdigest()


def test_read_rate_header(tmp_path):
    f = tmp_path / "x_imu_rigid.csv"
    f.write_text("# sampling frequency: 40\n# units are seconds/meters/radians/a.u.\na,b\n1,2\n", encoding="utf-8")
    assert diodem.read_rate_hz(f) == 40.0


def test_chains_are_trees_of_five_segments():
    for chain in diodem.CHAINS.values():
        segs = {s for p, c, _ in chain for s in (p, c)}
        assert segs == {f"seg{i}" for i in range(1, 6)} and len(chain) == 4


# ------------------------------------------------------------------ real DIODEM (skips without data)


@pytest.mark.diodem
def test_diodem_files_match_manifest(diodem_root):
    results = verify_manifest(manifest_dir() / diodem.MANIFEST, data_dir())
    bad = [r for r in results if r[1] != "ok"]
    assert len(results) == 267 and not bad, bad[:5]


@pytest.mark.diodem
def test_diodem_rates_and_quaternion_norms(diodem_root):
    scan = diodem.scan_rates_and_norms()
    assert scan["imu_rates_hz"] == [40.0]
    assert scan["omc_rates_hz"] == [30.0, 120.0]  # 30 Hz in arm exp01-03, 120 Hz in exp04-11
    assert scan["max_norm_dev"] < 1e-6 and scan["n_quaternions"] > 100000


@pytest.mark.diodem
@pytest.mark.slow
def test_six_axis_relative_heading_drifts(diodem_root):
    res = diodem.six_axis_relative_heading(diodem.load_trial(1), "seg2", "seg3")
    first = res["delta_deg"][res["t_s"] < 5]
    assert abs(res["delta_deg"][-1]) > 2 * np.percentile(np.abs(first), 95)
    assert np.ptp(res["rel_opt_deg"]) < 180.0
