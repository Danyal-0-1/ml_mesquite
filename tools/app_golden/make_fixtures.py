"""Write the phone app's golden fixture: run the app's real Swift core on scripted inputs (Step 3 brief §4.7).

    python tools/app_golden/make_fixtures.py [--mocap DIR] [--out tests/fixtures/app_golden/app_golden.json]

Compiles ``tools/app_golden/main.swift`` with ``Core/*.swift`` of the phone app at mesquite-mocap ``87d7c3b``
(``$MESQ_DATA_DIR/mesquite-mocap``, read only; the build goes to a temporary folder) using ``swiftc -O
-swift-version 5``, as the app's own ``tools/run_checks.sh`` does; feeds it
:func:`mesqsim.twin.app_golden.golden_inputs` and writes the inputs, the outputs and their provenance. Needs a
Swift 6 toolchain (``swiftc``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mesqsim.paths import data_dir  # noqa: E402
from mesqsim.twin.app_golden import FIXTURE, compare, golden_inputs  # noqa: E402

MOCAP_COMMIT = "87d7c3b"
CORE = Path("new_device_code/phone_app/MesquiteHips/Sources/MesquiteHips/Core")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mocap", default=str(data_dir() / "mesquite-mocap"))
    ap.add_argument("--out", default=str(FIXTURE))
    args = ap.parse_args(argv)
    mocap, out = Path(args.mocap), Path(args.out)
    swiftc = shutil.which("swiftc")
    if swiftc is None:
        raise RuntimeError("swiftc is not installed")
    commit = subprocess.run(["git", "-C", str(mocap), "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, check=True).stdout.strip()  # fmt: skip
    if commit != MOCAP_COMMIT:
        print(f"warning: mesquite-mocap is at {commit}, not {MOCAP_COMMIT}")
    sources = sorted((mocap / CORE).glob("*.swift"))
    version = subprocess.run([swiftc, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[0]
    inputs = golden_inputs()
    with tempfile.TemporaryDirectory() as tmp:
        exe, ip = Path(tmp) / "app_golden", Path(tmp) / "in.json"
        subprocess.run([swiftc, "-O", "-swift-version", "5", *map(str, sources),
                        str(ROOT / "tools" / "app_golden" / "main.swift"), "-o", str(exe)], check=True)  # fmt: skip
        ip.write_text(json.dumps(inputs), encoding="utf-8")
        outputs = json.loads(subprocess.run([str(exe), str(ip)], capture_output=True, check=True).stdout)
    prov = {"mesquite_mocap": commit, "swiftc": version, "flags": "-O -swift-version 5",
            "sources": {s.name: hashlib.sha256(s.read_bytes()).hexdigest()[:16] for s in sources}}
    fixture = {"provenance": prov, "inputs": inputs, "outputs": outputs}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture, separators=(",", ":")), encoding="utf-8")
    print(f"{out.name}: {out.stat().st_size / 1e3:.0f} kB, {version}")
    print(json.dumps(compare(fixture)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
