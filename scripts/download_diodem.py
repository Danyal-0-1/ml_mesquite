"""Download DIODEM (Harvard Dataverse doi:10.7910/DVN/SGJLZA, version 3.0, CC0 1.0) and verify it.

    python scripts/download_diodem.py          # processed CSVs (dataset/, about 526 MB) and the authors' notes
    python scripts/download_diodem.py --all    # everything, including videos (about 2.7 GB)

Files go to $MESQ_DATA_DIR/diodem/ (default data/diodem/) in the dataset's own folder layout. Each file is
verified against the MD5 that Dataverse publishes; size and SHA-256 are written to manifests/diodem.csv.
Re-running skips files already present and recorded. Cite the dataset and the paper
(Scientific Data, doi:10.1038/s41597-025-05468-w) in anything that uses it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesqsim.data import diodem  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="download every file, including videos")
    args = ap.parse_args()
    files = diodem.select_files(diodem.fetch_file_list(), everything=args.all)
    total = sum(f.size_bytes for f in files)
    print(f"DIODEM v{diodem.VERSION}: {len(files)} files, {total / 1e6:.1f} MB -> {diodem.diodem_dir()}")
    rows = diodem.download(files)
    print(f"verified {len(rows)} files; manifest: manifests/{diodem.MANIFEST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
