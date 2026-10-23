"""DIODEM: real IMUs on mechanical five-segment chains with optical truth (brief §5.2).

Bachhuber, Lehmann, Weygers, Seel. "DIODEM – A Diverse Inertial and Optical Dataset of kinEmatic chain
Motion." Scientific Data (2025), doi:10.1038/s41597-025-05468-w. Data: Harvard Dataverse
doi:10.7910/DVN/SGJLZA, version 3.0, licence CC0 1.0.

Files are downloaded from Dataverse into ``<data dir>/diodem/`` with the dataset's own folder layout, each
verified against the MD5 Dataverse publishes, and recorded (size, SHA-256) in ``manifests/diodem.csv``.
That layout is what the authors' loader (PyPI ``imt-diodem``, module ``diodem``) expects in its cache folder,
so :func:`load_trial` calls the official loader with ``backend="dataverse"`` and never downloads unverified
copies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from ..manifest import ManifestRow, read_manifest, write_manifest
from ..paths import data_dir, manifest_dir
from .net import download_file, session

__all__ = [
    "DOI",
    "VERSION",
    "DataverseFile",
    "diodem_dir",
    "fetch_file_list",
    "select_files",
    "download",
]

API = "https://dataverse.harvard.edu/api"
DOI = "doi:10.7910/DVN/SGJLZA"
VERSION = "3.0"
SUBDIR = "diodem"
MANIFEST = "diodem.csv"

# What Step 1 needs: the processed per-motion CSV files plus the authors' notes. Excluded by default:
# videos/ (1.7 GB), make_dataset/raw_data/ (raw logs, 0.46 GB), cad_files/, images/.
DEFAULT_PREFIXES = ("dataset/",)
DEFAULT_EXTRA = (
    "make_dataset/readme.md",
    "make_dataset/preprocess/metadata.yaml",
    "make_dataset/preprocess/setup.json",
)


@dataclass(frozen=True)
class DataverseFile:
    path: str  # folder layout of the dataset, e.g. "dataset/arm/exp01/motion01_canonical/exp01_motion01_omc.csv"
    file_id: int
    size_bytes: int
    md5: str


def diodem_dir() -> Path:
    """Folder that mirrors the Dataverse layout (also the official loader's cache folder)."""
    return data_dir() / SUBDIR


def fetch_file_list(version: str = VERSION) -> list[DataverseFile]:
    """List every file of the given dataset version from the Dataverse API."""
    url = f"{API}/datasets/:persistentId/versions/{version}/files?persistentId={DOI}"
    r = session().get(url, timeout=60)
    r.raise_for_status()
    out = []
    for entry in r.json()["data"]:
        df = entry["dataFile"]
        if df.get("originalFileFormat"):
            raise RuntimeError(f"{df['filename']} was ingested by Dataverse; download the original format")
        if df["checksum"]["type"] != "MD5":
            raise RuntimeError(f"unexpected checksum type {df['checksum']['type']}")
        label = entry.get("directoryLabel", "")
        path = f"{label}/{df['filename']}" if label else df["filename"]
        out.append(DataverseFile(path, int(df["id"]), int(df["filesize"]), df["checksum"]["value"]))
    return sorted(out, key=lambda f: f.path)


def select_files(files: list[DataverseFile], everything: bool = False) -> list[DataverseFile]:
    if everything:
        return list(files)
    return [f for f in files if f.path.startswith(DEFAULT_PREFIXES) or f.path in DEFAULT_EXTRA]


def download(files: list[DataverseFile], log=print) -> list[ManifestRow]:
    """Download (or re-verify) the given files into :func:`diodem_dir` and update ``manifests/diodem.csv``.

    A file already present with a recorded SHA-256 and the right size is kept; anything else is
    (re)downloaded and verified against the Dataverse MD5.
    """
    root = diodem_dir()
    man_path = manifest_dir() / MANIFEST
    known = {r.path: r for r in read_manifest(man_path)} if man_path.is_file() else {}
    s = session()
    rows = []
    total = len(files)
    for i, f in enumerate(files, 1):
        rel = f"{SUBDIR}/{f.path}"
        dest = root / f.path
        prev = known.get(rel)
        if prev and prev.sha256 and dest.is_file() and dest.stat().st_size == f.size_bytes:
            rows.append(ManifestRow(rel, f.size_bytes, prev.sha256, f"md5={f.md5}"))
            continue
        size, sha = download_file(s, f"{API}/access/datafile/{f.file_id}", dest, f.size_bytes, f.md5)
        rows.append(ManifestRow(rel, size, sha, f"md5={f.md5}"))
        log(f"[{i}/{total}] {f.path} ({size / 1e6:.1f} MB) md5 ok")
    merged = {r.path: r for r in known.values()}
    merged.update({r.path: r for r in rows})
    write_manifest(man_path, list(merged.values()))
    return rows


def set_loader_cache() -> Path:
    """Point the official loader at our verified copy (it reads ``DIODEM_CACHE_FOLDER``)."""
    os.environ["DIODEM_CACHE_FOLDER"] = str(diodem_dir())
    return diodem_dir()


def is_available() -> bool:
    return (diodem_dir() / "dataset").is_dir()


# Adjacent segments (parent, child, joint) of the two chains. From the paper (Fig. 4, Table 4) and
# verified on exp01 and exp06: adjacent segments' nearest markers are 110-134 mm apart, others >= 372 mm;
# hinge pairs have >= 96 % of their relative angular velocity about one axis (see WORKLOG, M2).
CHAINS = {
    "arm": (("seg1", "seg2", "spherical"), ("seg2", "seg3", "hinge"), ("seg3", "seg4", "hinge"),
            ("seg4", "seg5", "hinge")),
    "gait": (("seg5", "seg1", "hinge"), ("seg1", "seg2", "saddle"), ("seg2", "seg3", "saddle"),
             ("seg3", "seg4", "hinge")),
}  # fmt: skip
ARM_EXPS = (1, 2, 3, 4, 5)
GAIT_EXPS = (6, 7, 8, 9, 10, 11)


def read_rate_hz(csv_path: Path) -> float:
    """Sampling rate from a DIODEM CSV header line ``# sampling frequency: 40``."""
    with open(csv_path, encoding="utf-8") as f:
        first = f.readline()
    if "sampling frequency" not in first:
        raise ValueError(f"{csv_path}: unexpected header {first!r}")
    return float(first.split(":")[1])


def scan_rates_and_norms() -> dict:
    """Rates of every IMU and OMC CSV, and the largest deviation of an optical quaternion norm from 1."""
    import numpy as np
    import pandas as pd

    imu_rates, omc_rates, worst_norm, n_quat = set(), set(), 0.0, 0
    for f in sorted((diodem_dir() / "dataset").rglob("*.csv")):
        rate = read_rate_hz(f)
        if f.name.endswith("_omc.csv"):
            omc_rates.add(rate)
            df = pd.read_csv(f, skiprows=2)
            for seg in range(1, 6):
                q = df[[f"seg{seg}_quat_{c}" for c in "wxyz"]].to_numpy()
                q = q[np.all(np.isfinite(q), axis=1)]
                n_quat += len(q)
                if len(q):
                    worst_norm = max(worst_norm, float(np.max(np.abs(np.linalg.norm(q, axis=1) - 1.0))))
        else:
            imu_rates.add(rate)
    return {"imu_rates_hz": sorted(imu_rates), "omc_rates_hz": sorted(omc_rates), "max_norm_dev": worst_norm,
            "n_quaternions": n_quat}


def load_trial(exp_id: int, hz: float = 40.0) -> dict:
    """All motions of one experiment, concatenated, through the authors' loader, resampled to ``hz``.

    At 40 Hz the IMU signals keep their native samples; the 30 Hz optical data are interpolated.
    Quaternions in the returned dict are the loader's scalar-first ``(w, x, y, z)``.
    """
    set_loader_cache()
    import diodem as official

    return official.load_data(exp_id, motion_start=1, motion_stop=-1, resample_to_hz=hz, backend="dataverse")


def six_axis_relative_heading(data: dict, parent: str, child: str, hz: float = 40.0) -> dict:
    """Relative heading of two adjacent segments from six-axis VQF (rigid IMUs) and from optical truth.

    The rigid IMUs' axes coincide with the optical segment axes in the processed data (verified: tilt from
    VQF and from the optical frames agree to a mean cosine of 1.000), so each VQF orientation is
    ``q_i = Y(-psi_i) · Q_i`` up to fusion error, with ``Q_i`` the optical orientation (segment -> earth,
    z up). Returns time (s), the optical relative heading ``heading(Q_c) - heading(Q_p)``, the VQF one
    (shifted to agree with the optical value at t = 0), and the six-axis relative-heading error
    ``delta(t) = wrap(psi_c - psi_p) - delta(0)``, all in degrees.
    """
    import numpy as np
    from vqf import VQF

    from ..conventions import heading_rad, quat_conj, quat_mul, wrap_rad, wxyz_to_xyzw

    psi, h_opt, h_vqf = {}, {}, {}
    for seg in (parent, child):
        imu = data[seg]["imu_rigid"]
        out = VQF(1.0 / hz).updateBatch(np.ascontiguousarray(imu["gyr"]), np.ascontiguousarray(imu["acc"]))
        q = wxyz_to_xyzw(out["quat6D"])  # six-axis: the magnetometer is dropped
        big_q = wxyz_to_xyzw(data[seg]["quat"])
        psi[seg] = -heading_rad(quat_mul(q, quat_conj(big_q)))  # error rotation ≈ Y(-psi)
        h_opt[seg], h_vqf[seg] = heading_rad(big_q), heading_rad(q)
    n = len(psi[parent])
    t = np.arange(n) / hz
    rel_opt = np.rad2deg(np.unwrap(wrap_rad(h_opt[child] - h_opt[parent])))
    rel_vqf = np.rad2deg(np.unwrap(wrap_rad(h_vqf[child] - h_vqf[parent])))
    delta = np.rad2deg(np.unwrap(wrap_rad(psi[child] - psi[parent])))
    k0 = max(1, int(hz))  # align on the first second
    rel_vqf = rel_vqf - np.mean(rel_vqf[:k0] - rel_opt[:k0])
    delta = delta - np.mean(delta[:k0])
    return {"t_s": t, "rel_opt_deg": rel_opt, "rel_vqf_deg": rel_vqf, "delta_deg": delta}
