"""Write the page golden fixtures: run the real page code in Node on each scripted scenario (Step 3 brief §4.10).

    python tools/page_golden/make_fixtures.py [--mocap DIR] [--out tests/fixtures/page_golden]

For each scenario of :mod:`mesqsim.twin.page_scenarios` and each tree (``velvet``, ``smooth``) it writes
``<scenario>_<tree>.json`` with the input events, the page's outputs (``tools/page_golden/run.mjs``) and their
provenance; and ``rig_scene2.json``, the avatar's node tree, so that ``tests/test_page_twin.py`` runs without Node
or the mesquite-mocap clone. Needs Node and the clone pinned at 87d7c3b (``$MESQ_DATA_DIR/mesquite-mocap``).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mesqsim.paths import data_dir  # noqa: E402
from mesqsim.twin.page import PAGE_EPOCH_MS, load_rig  # noqa: E402
from mesqsim.twin.page_scenarios import SCENARIOS, make_scenario  # noqa: E402

TREES = ("velvet", "smooth")
MOCAP_COMMIT = "87d7c3b"


def mocap_commit(mocap: Path) -> str:
    return subprocess.run(["git", "-C", str(mocap), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def run_page(mocap: Path, inp: dict) -> dict:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("Node is not installed")
    with tempfile.TemporaryDirectory() as tmp:
        ip, op = Path(tmp) / "in.json", Path(tmp) / "out.json"
        ip.write_text(json.dumps(inp), encoding="utf-8")
        # NODE_NO_WARNINGS: build-static/ has no package.json saying "module"; Node warns and parses it as ESM
        subprocess.run([node, str(ROOT / "tools" / "page_golden" / "run.mjs"), str(ip), str(op), str(mocap)],
                       check=True, env=os.environ | {"NODE_NO_WARNINGS": "1"})
        return json.loads(op.read_text(encoding="utf-8"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mocap", default=str(data_dir() / "mesquite-mocap"))
    ap.add_argument("--out", default=str(ROOT / "tests" / "fixtures" / "page_golden"))
    args = ap.parse_args(argv)
    mocap, out = Path(args.mocap), Path(args.out)
    commit = mocap_commit(mocap)
    if commit != MOCAP_COMMIT:
        print(f"warning: mesquite-mocap is at {commit}, not {MOCAP_COMMIT}")
    out.mkdir(parents=True, exist_ok=True)
    rig = load_rig(mocap / "glbs" / "scene2.glb")
    (out / "rig_scene2.json").write_text(json.dumps(rig.to_json()), encoding="utf-8")
    node_version = subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip()
    for name in SCENARIOS:
        for tree in TREES:
            inp = make_scenario(name, tree) | {"epoch_ms": PAGE_EPOCH_MS}
            golden = run_page(mocap, inp)
            prov = golden.pop("provenance") | {"mesquite_mocap": commit}
            path = out / f"{name}_{tree}.json"
            path.write_text(json.dumps({"input": inp, "golden": golden, "provenance": prov}), encoding="utf-8")
            print(f"{path.name}: {len(inp['events'])} events, {len(golden['threw'])} throws, node {node_version}, "
                  f"three r{prov['three_revision']}, {path.stat().st_size / 1e3:.0f} kB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
