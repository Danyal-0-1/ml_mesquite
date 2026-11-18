"""Every link and code reference in docs/explain/ resolves to the current repository (Step 2 brief §10).

* Markdown links ``[text](target)``: the file exists; an anchor ``#heading`` matches a heading of the target file
  (GitHub's slug rules); a line anchor ``#L12`` or ``#L12-L30`` lies inside the file, and when the link text names a
  symbol in backticks, that symbol appears on the first referenced line, so a reference cannot silently point at code
  that moved.
* Plain-text references ``path/to/file.py:123`` (or ``:123-140``) resolve the same way.
* Every test named in backticks (```test_...```) exists in tests/.
* Every file links back to the index, and each file of the pipeline links on to the next.
"""

import re
import unicodedata
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPLAIN = ROOT / "docs" / "explain"
ORDER = ["MENTAL_PICTURE.md", "PIPELINE_EXPLAINED.md", "MATH_EXPLAINED.md", "CODE_EXPLAINED.md", "THEORY.md"]
FILES = ["README.md", *ORDER]
LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(([^)\s]+)\)")
PATH_LINE = re.compile(r"(?<![\w/.])((?:mesqsim|scripts|tests)/[\w/]+\.py):(\d+)(?:-(\d+))?")
TICKED_TEST = re.compile(r"`(test_[a-z0-9_]+)`")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens."""
    h = heading.strip().lower()
    h = "".join(ch for ch in h if ch in " -_" or unicodedata.category(ch)[0] in "LN")
    return h.replace(" ", "-")


def headings(path: Path) -> set[str]:
    out, fence = set(), False
    for line in path.read_text(encoding="utf-8").split("\n"):
        if line.startswith("```"):
            fence = not fence
        elif not fence and line.startswith("#"):
            out.add(slug(line.lstrip("#")))
    return out


def outside_fences(text: str) -> str:
    """The text with fenced code blocks blanked out (their contents are not links)."""
    out, fence = [], False
    for line in text.split("\n"):
        if line.startswith("```"):
            fence = not fence
            out.append("")
        else:
            out.append("" if fence else line)
    return "\n".join(out)


def check_lines(path: Path, first: int, last: int | None, symbol: str | None, where: str) -> None:
    lines = path.read_text(encoding="utf-8").split("\n")
    last = last or first
    assert 1 <= first <= last <= len(lines), f"{where}: lines {first}-{last} outside {path.name} ({len(lines)} lines)"
    if symbol:
        assert symbol in lines[first - 1], f"{where}: `{symbol}` is not on {path.name} line {first}"


@pytest.mark.parametrize("name", FILES)
def test_every_link_resolves(name):
    path = EXPLAIN / name
    assert path.is_file(), f"{name} is missing"
    text = outside_fences(path.read_text(encoding="utf-8"))
    for label, target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        file_part, _, anchor = target.partition("#")
        dest = (path.parent / file_part).resolve() if file_part else path
        assert dest.exists(), f"{name}: link [{label}]({target}) points to a missing file"
        if not anchor:
            continue
        m = re.fullmatch(r"L(\d+)(?:-L(\d+))?", anchor)
        if m:
            symbol = re.search(r"`([^`]+)`", label)
            check_lines(dest, int(m.group(1)), int(m.group(2)) if m.group(2) else None,
                        symbol.group(1).split("(")[0] if symbol else None, f"{name} [{label}]({target})")
        else:
            assert dest.suffix == ".md", f"{name}: anchor on a non-Markdown file: {target}"
            assert anchor in headings(dest), f"{name}: no heading #{anchor} in {dest.name}"


@pytest.mark.parametrize("name", FILES)
def test_every_path_line_reference_resolves(name):
    text = outside_fences((EXPLAIN / name).read_text(encoding="utf-8"))
    for rel, first, last in PATH_LINE.findall(text):
        dest = ROOT / rel
        assert dest.is_file(), f"{name}: {rel} does not exist"
        check_lines(dest, int(first), int(last) if last else None, None, f"{name}: {rel}:{first}")


@pytest.mark.parametrize("name", FILES)
def test_every_named_test_exists(name):
    defined = set()
    for f in (ROOT / "tests").glob("test_*.py"):
        defined |= set(re.findall(r"^def (test_[a-z0-9_]+)", f.read_text(encoding="utf-8"), flags=re.M))
    for t in TICKED_TEST.findall((EXPLAIN / name).read_text(encoding="utf-8")):
        assert t in defined, f"{name}: `{t}` is not a test in tests/"


def test_the_files_link_back_and_on():
    index = (EXPLAIN / "README.md").read_text(encoding="utf-8")
    for name in ORDER:
        assert f"]({name})" in index, f"the index does not link to {name}"
    for i, name in enumerate(ORDER):
        text = (EXPLAIN / name).read_text(encoding="utf-8")
        assert "](README.md)" in text, f"{name} does not link back to the index"
        if i + 1 < len(ORDER):
            assert f"]({ORDER[i + 1]})" in text, f"{name} does not link on to {ORDER[i + 1]}"
