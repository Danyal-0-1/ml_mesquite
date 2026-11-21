"""The repository's map stays true: README.md, CHANGELOG.md, REPOSITORY_GUIDE.md and docs/step_3/ point only at things
that exist.

* Every Markdown link ``[text](target)`` resolves to a file or folder; an anchor ``#heading`` matches a heading of its
  target (GitHub's slug rules, shared with tests/test_explain_links.py).
* Every path in backticks that starts at a top-level folder (``mesqsim/twin/page.py``, ``docs/explain/``) exists, and
  every bare code-file name in backticks (``twodof.py``) is a tracked file somewhere. A test id (``path::name``) or a
  line (``path:12``) is checked by its path. A path with a placeholder (``{``, ``<``, ``*``, ``…``) is a pattern, and
  ``PLANNED`` lists the files the briefs plan that do not exist yet (Danyal's four model files).
* The guide names every folder that holds tracked files, down to two levels (``mesqsim/twin/``, ``tools/app_golden/``).
"""

import re
import subprocess
from pathlib import Path

import pytest

from tests.test_explain_links import LINK, headings, outside_fences, slug

ROOT = Path(__file__).resolve().parents[1]
DOCS = ["README.md", "CHANGELOG.md", "REPOSITORY_GUIDE.md", "docs/step_3/README.md", "docs/step_3/DATASET_CARD.md",
        "docs/step_3/LEARN_THE_MODEL.md"]
GUIDE = ROOT / "REPOSITORY_GUIDE.md"
TOP = ("mesqsim/", "tests/", "scripts/", "tools/", "docs/", "configs/", "manifests/", "project_explanations/",
       ".github/")
CODE_SUFFIXES = (".py", ".sh", ".mjs", ".swift", ".yaml")
# Danyal writes these four; until then the documents name them as planned
PLANNED = {f"{p}{m}.py" for p in ("mesqsim/learn/", "") for m in ("loader", "models", "loss", "train")}
TICKED = re.compile(r"`([^`\s]+)`")


def tracked() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [p for p in out.decode().split("\0") if p]


@pytest.mark.parametrize("name", DOCS)
def test_every_link_resolves(name):
    text = outside_fences((ROOT / name).read_text(encoding="utf-8"))
    bad = []
    for _, target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        path, _, anchor = target.partition("#")
        dest = ROOT / name if not path else (ROOT / name).parent / path
        if not dest.exists():
            bad.append(target)
        elif anchor and dest.is_file() and dest.suffix == ".md" and slug(anchor) not in headings(dest):
            bad.append(target)
    assert not bad, f"{name}: {bad}"


@pytest.mark.parametrize("name", DOCS)
def test_every_named_path_exists(name):
    basenames = {Path(p).name for p in tracked()}
    bad = []
    for token in TICKED.findall(outside_fences((ROOT / name).read_text(encoding="utf-8"))):
        token = re.sub(r":\d+(-\d+)?$", "", token.rstrip(".,;:").split("::")[0])
        if any(ch in token for ch in "{<*…$") or token in PLANNED:
            continue
        if token.startswith(TOP):
            if not (ROOT / token).exists():
                bad.append(token)
        elif "/" not in token and token.endswith(CODE_SUFFIXES) and token not in basenames:
            bad.append(token)
    assert not bad, f"{name}: {bad}"


def test_the_guide_names_every_folder():
    text = GUIDE.read_text(encoding="utf-8")
    parts = [Path(p).parts for p in tracked()]
    folders = {"/".join(q[:depth]) + "/" for q in parts for depth in (1, 2) if len(q) > depth}
    missing = sorted(f for f in folders if f not in text)
    assert not missing, f"REPOSITORY_GUIDE.md does not mention {missing}"
