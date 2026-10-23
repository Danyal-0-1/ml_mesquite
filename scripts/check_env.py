"""Print the environment and check that it is complete.

Prints the Python and package versions, the OS and the CPU; imports every required package; and
checks every data file listed in ``manifests/*.csv`` (size and SHA-256) under the data directory.

    python scripts/check_env.py            # full check, hashes every data file
    python scripts/check_env.py --fast     # sizes only (AMASS archives are large)
    python scripts/check_env.py --require diodem   # a missing DIODEM file is a failure

Exit status 0 when every import works and no present data file mismatches its manifest.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mesqsim.manifest import verify_manifest  # noqa: E402
from mesqsim.paths import data_dir, manifest_dir, output_dir  # noqa: E402

# distribution name -> import name
PACKAGES = {
    "numpy": "numpy",
    "scipy": "scipy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "matplotlib": "matplotlib",
    "h5py": "h5py",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "torch": "torch",
    "smplx": "smplx",
    "vqf": "vqf",
    "requests": "requests",
    "pytest": "pytest",
    "hypothesis": "hypothesis",
    "ruff": "ruff",
    "imt-diodem": "diodem",
    "truststore": "truststore",
}


def pinned_versions() -> dict[str, str]:
    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    pins = {}
    for line in req.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if "==" in line:
            name, ver = line.split("==")
            pins[name.strip().lower()] = ver.strip()
    return pins


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fast", action="store_true", help="check sizes only, skip SHA-256")
    ap.add_argument("--require", nargs="*", default=[], help="manifest names whose files must be present")
    args = ap.parse_args()

    failures = 0
    print(f"Python   {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        print("  FAIL: the brief pins Python 3.11")
        failures += 1
    print(f"OS       {platform.platform()}")
    print(f"CPU      {platform.processor() or platform.machine()}, {os.cpu_count()} logical cores")
    print(f"data     {data_dir()}  (MESQ_DATA_DIR {'set' if 'MESQ_DATA_DIR' in os.environ else 'not set'})")
    print(f"outputs  {output_dir()}  (MESQ_OUTPUT_DIR {'set' if 'MESQ_OUTPUT_DIR' in os.environ else 'not set'})")

    print("\nPackages (installed / pinned):")
    pins = pinned_versions()
    for dist, mod in PACKAGES.items():
        try:
            ver = md.version(dist)
        except md.PackageNotFoundError:
            ver = "not installed"
        pin = pins.get(dist.lower(), "?")
        try:
            importlib.import_module(mod)
            status = "ok" if ver.split("+")[0] == pin else "VERSION DIFFERS"
        except Exception as exc:  # noqa: BLE001 - report every import problem
            status = f"IMPORT FAILED: {exc.__class__.__name__}: {exc}"
            failures += 1
        print(f"  {dist:<11} {ver:<14} {pin:<10} {status}")

    print("\nData files (manifests/*.csv):")
    manifests = sorted(manifest_dir().glob("*.csv"))
    if not manifests:
        print("  no manifests yet")
    for man in manifests:
        results = verify_manifest(man, data_dir(), hash_files=not args.fast)
        counts: dict[str, int] = {}
        for path, status, detail in results:
            counts[status] = counts.get(status, 0) + 1
            # print every mismatch, but only the first few missing or unrecorded files
            if status == "MISMATCH" or (status != "ok" and counts[status] <= 3):
                print(f"  {man.stem}: {status:<10} {path} {detail}")
        summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
        print(f"  {man.stem}: {len(results)} files: {summary}")
        failures += counts.get("MISMATCH", 0)
        if man.stem in args.require:
            failures += counts.get("missing", 0)

    print("\nRESULT:", "PASS" if failures == 0 else f"FAIL ({failures} problem(s))")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
