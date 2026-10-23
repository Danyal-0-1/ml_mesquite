"""Verify and index the AMASS and SMPL+H files a person downloaded by hand (see docs/DATA.md).

    python scripts/verify_amass.py            # record archives, extract new ones, index motion files
    python scripts/verify_amass.py --no-extract

Expected layout under $MESQ_DATA_DIR (default data/):

    amass/archives/<Subset>.tar.bz2        archives as downloaded ("SMPL+H G" body data)
    amass/<Subset>/<subject>/<seq>_poses.npz   extracted by this script
    body_models/smplh/{male,female,neutral}/model.npz   from the MANO site ("Extended SMPL+H model")

What it writes (all committed; they hold names, sizes and checksums only, never data):

    manifests/amass_archives.csv   every archive: size, SHA-256
    manifests/amass_files.csv      every extracted motion file: size, SHA-256
    manifests/smplh.csv            every body-model file: size, SHA-256

It also loads every motion file once and reports frame rates and failures. TotalCapture is reported
separately: it must never be used for training (brief §5.1).
"""

from __future__ import annotations

import argparse
import sys
import tarfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mesqsim.data import amass  # noqa: E402
from mesqsim.manifest import ManifestRow, sha256_file, write_manifest  # noqa: E402
from mesqsim.paths import data_dir, manifest_dir  # noqa: E402

ARCHIVE_SUFFIXES = (".tar.bz2", ".tar.xz", ".tar.gz", ".tgz", ".tar")


def rel(p: Path) -> str:
    return p.relative_to(data_dir()).as_posix()


def record(paths: list[Path], manifest: str, note=lambda p: "") -> None:
    rows = []
    for i, p in enumerate(paths, 1):
        rows.append(ManifestRow(rel(p), p.stat().st_size, sha256_file(p), note(p)))
        if i % 500 == 0:
            print(f"  hashed {i}/{len(paths)}")
    write_manifest(manifest_dir() / manifest, rows)
    print(f"wrote manifests/{manifest}: {len(rows)} files, {sum(r.size_bytes for r in rows) / 1e9:.2f} GB")


def extract(archive: Path, dest: Path) -> None:
    marker = dest / f".extracted_{archive.name}"
    if marker.exists():
        return
    print(f"extracting {archive.name} ...")
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")  # refuses absolute paths, links outside dest, devices
    marker.write_text("ok\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-extract", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    adir, archives_dir = amass.amass_dir(), amass.amass_dir() / "archives"
    archives = []
    if archives_dir.is_dir():
        archives = sorted(p for p in archives_dir.glob("*") if p.name.endswith(ARCHIVE_SUFFIXES))
    print(f"AMASS folder: {adir}\n{len(archives)} archive(s) in archives/")
    if archives:
        record(archives, "amass_archives.csv")
        if not args.no_extract:
            for a in archives:
                extract(a, adir)

    files = amass.motion_files()
    print(f"{len(files)} motion file(s)")
    if files:
        record(files, "amass_files.csv")
        rates, subsets, failures = Counter(), Counter(), []
        for p in files:
            try:
                seq = amass.load_sequence(p)
                rates[seq.fps] += 1
                subsets[amass.subset_subject(p)[0]] += 1
            except (ValueError, OSError) as exc:
                failures.append(f"{rel(p)}: {exc}")
        print("files per subset:", dict(sorted(subsets.items())))
        print("frame rates (Hz: files):", dict(sorted(rates.items())))
        if any("totalcapture" in s.lower() for s in subsets):
            print("NOTE: TotalCapture present; it is excluded from training splits (brief §5.1).")
        for f in failures:
            print("FAILED:", f)
        if failures:
            return 1

    models = sorted(amass.body_model_dir().rglob("*.npz")) if amass.body_model_dir().is_dir() else []
    print(f"{len(models)} SMPL+H model file(s) in {amass.body_model_dir()}")
    if models:
        record(models, "smplh.csv")
    if not files:
        print("Nothing to index yet: follow docs/DATA.md to download AMASS.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
