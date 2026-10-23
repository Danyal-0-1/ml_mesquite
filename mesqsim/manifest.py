"""Manifests of data files: one CSV per dataset with path, size and SHA-256 (committed; the data are not).

Columns: ``path`` (POSIX, relative to the data directory), ``size_bytes``, ``sha256``, ``note``.
An empty ``sha256`` means "expected file, checksum not recorded yet" (e.g. AMASS archives before the
first download); :func:`record_checksums` fills it in from the downloaded copy.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

__all__ = ["ManifestRow", "sha256_file", "read_manifest", "write_manifest", "verify_manifest", "record_checksums"]

FIELDS = ("path", "size_bytes", "sha256", "note")


@dataclass
class ManifestRow:
    path: str
    size_bytes: int | None
    sha256: str
    note: str = ""


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: Path) -> list[ManifestRow]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = []
        for r in csv.DictReader(f):
            size = r.get("size_bytes", "").strip()
            sha = r.get("sha256", "").strip()
            rows.append(ManifestRow(r["path"], int(size) if size else None, sha, r.get("note", "")))
    return rows


def write_manifest(path: Path, rows: list[ManifestRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(FIELDS)
        for r in sorted(rows, key=lambda r: r.path):
            w.writerow([r.path, "" if r.size_bytes is None else r.size_bytes, r.sha256, r.note])


def verify_manifest(manifest: Path, root: Path, hash_files: bool = True) -> list[tuple[str, str, str]]:
    """Check every manifest row against the files under ``root``.

    Returns ``(path, status, detail)`` triples with status ``ok``, ``missing``, ``unrecorded``
    (present but no checksum in the manifest yet) or ``MISMATCH``.
    """
    out = []
    for row in read_manifest(manifest):
        f = root / row.path
        if not f.is_file():
            out.append((row.path, "missing", ""))
            continue
        size = f.stat().st_size
        if row.size_bytes is not None and size != row.size_bytes:
            out.append((row.path, "MISMATCH", f"size {size} != {row.size_bytes}"))
            continue
        if not row.sha256:
            out.append((row.path, "unrecorded", f"size {size}"))
            continue
        if hash_files:
            digest = sha256_file(f)
            if digest != row.sha256:
                out.append((row.path, "MISMATCH", f"sha256 {digest[:12]}… != {row.sha256[:12]}…"))
                continue
        out.append((row.path, "ok", "" if hash_files else "size only"))
    return out


def record_checksums(manifest: Path, root: Path) -> int:
    """Fill in size and SHA-256 for rows whose file is present but unrecorded. Returns the count filled."""
    rows = read_manifest(manifest)
    filled = 0
    for row in rows:
        f = root / row.path
        if f.is_file() and not row.sha256:
            row.size_bytes = f.stat().st_size
            row.sha256 = sha256_file(f)
            filled += 1
    write_manifest(manifest, rows)
    return filled
