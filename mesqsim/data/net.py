"""HTTP session for scripted downloads of open data.

Two machine-level quirks are handled here once:

* ``truststore`` makes Python use the operating system's certificate store. On the collaborator's machine,
  Avast Web/Mail Shield re-signs HTTPS traffic with a root that only Windows trusts, so Python's bundled CA
  list fails. Elsewhere it changes nothing. Integrity never rests on TLS alone: every download is checked
  against the checksum the data host publishes.
* Harvard Dataverse answers ``403`` to the default ``python-requests`` User-Agent, so we send our own.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

__all__ = ["session", "download_file"]

USER_AGENT = "mesqsim-download/0.1 (Mesquite ML research; scripted download of open data)"


def session() -> requests.Session:
    try:
        import truststore

        truststore.inject_into_ssl()
    except ImportError:  # pragma: no cover - truststore is pinned in requirements.txt
        pass
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def download_file(
    s: requests.Session, url: str, dest: Path, expected_size: int | None, expected_md5: str | None
) -> tuple[int, str]:
    """Stream ``url`` to ``dest`` atomically; verify size and MD5 before renaming into place.

    Returns ``(size_bytes, sha256_hex)`` of the verified file. Raises ``OSError`` on any mismatch.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    md5, sha = hashlib.md5(), hashlib.sha256()
    size = 0
    with s.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for block in r.iter_content(1 << 20):
                f.write(block)
                md5.update(block)
                sha.update(block)
                size += len(block)
    if expected_size is not None and size != expected_size:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{url}: size {size} != expected {expected_size}")
    if expected_md5 is not None and md5.hexdigest() != expected_md5:
        tmp.unlink(missing_ok=True)
        raise OSError(f"{url}: md5 {md5.hexdigest()} != expected {expected_md5}")
    tmp.replace(dest)
    return size, sha.hexdigest()
