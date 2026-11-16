"""The simulator configuration: one YAML file, every parameter labelled (brief §4.3, §8).

Each parameter is a leaf mapping::

    drift_sigma_deg_per_min:
      value: 0.2
      label: assumption          # datasheet | firmware | assumption | measured
      source: "brief §4.3: residual drift unknown; swept"
      sweep: [0.05, 0.2, 1.0]    # optional: the values a sweep visits

Any other mapping is a group. Keys are addressed with dots (``error_model.drift_sigma_deg_per_min``).
:meth:`Config.hash` covers the resolved *values* only, so editing a source note does not change a run's identity.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

__all__ = ["LABELS", "Config", "load_config", "parse_override"]

LABELS = ("datasheet", "firmware", "assumption", "measured", "fitted")
"""``fitted``: estimated on the training or validation split by a named command, never on test data (Step 2
brief rule 6). The source note names the command, the data and the split."""


def _is_leaf(node: Any) -> bool:
    return isinstance(node, dict) and "value" in node


class Config:
    def __init__(self, tree: dict, source_path: str | None = None):
        self.tree = tree
        self.source_path = source_path
        self.validate()

    # ------------------------------------------------------------------ access
    def leaves(self) -> Iterator[tuple[str, dict]]:
        def walk(node: dict, prefix: str):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else k
                if _is_leaf(v):
                    yield key, v
                elif isinstance(v, dict):
                    yield from walk(v, key)
                else:
                    raise ValueError(f"{key}: expected a parameter mapping with 'value' or a group, got {v!r}")

        yield from walk(self.tree, "")

    def _node(self, key: str) -> dict:
        node: Any = self.tree
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"no parameter '{key}'")
            node = node[part]
        if not _is_leaf(node):
            raise KeyError(f"'{key}' is a group, not a parameter")
        return node

    def get(self, key: str) -> Any:
        return self._node(key)["value"]

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def param(self, key: str) -> dict:
        return dict(self._node(key))

    def values(self) -> dict:
        """Nested dict of resolved values (what the simulator reads and what the hash covers)."""
        out: dict = {}
        for key, leaf in self.leaves():
            node = out
            *parents, last = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[last] = leaf["value"]
        return out

    def hash(self) -> str:
        """SHA-256 (hex) of the canonical JSON of :meth:`values`."""
        blob = json.dumps(self.values(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    # ------------------------------------------------------------------ changes
    def with_overrides(self, overrides: dict[str, Any]) -> Config:
        """A copy with ``{dotted key: new value}`` applied; unknown keys raise ``KeyError``."""
        tree = copy.deepcopy(self.tree)
        new = Config.__new__(Config)
        new.tree, new.source_path = tree, self.source_path
        for key, val in overrides.items():
            new._node(key)["value"] = val
        new.validate()
        return new

    def section(self, name: str) -> Config:
        """A config holding only the group ``name`` (keys keep their full dotted form, e.g. ``geometry.window_s``).

        The Step 2 estimators receive ``cfg.section("geometry")``: reading the simulator's ``error_model`` (the
        truth) from inside an estimator then raises ``KeyError`` instead of silently using it (brief rule 8).
        """
        if name not in self.tree or _is_leaf(self.tree[name]) or not isinstance(self.tree[name], dict):
            raise KeyError(f"no config group '{name}'")
        new = Config.__new__(Config)
        new.tree, new.source_path = {name: copy.deepcopy(self.tree[name])}, self.source_path
        return new

    def without(self, name: str) -> Config:
        """A copy without the group ``name`` (the simulator's identity does not depend on ``geometry``)."""
        new = Config.__new__(Config)
        new.tree = {k: copy.deepcopy(v) for k, v in self.tree.items() if k != name}
        new.source_path = self.source_path
        return new

    def with_preset(self, name: str) -> Config:
        """Apply a named preset from ``presets.yaml`` next to the config (e.g. ``zero``, ``offsets_only``)."""
        presets = load_presets(self.source_path)
        if name not in presets:
            raise KeyError(f"unknown preset '{name}'; known: {sorted(presets)}")
        return self.with_overrides(presets[name])

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        for key, leaf in self.leaves():
            if leaf.get("label") not in LABELS:
                raise ValueError(f"{key}: label must be one of {LABELS}, got {leaf.get('label')!r}")
            if not str(leaf.get("source", "")).strip():
                raise ValueError(f"{key}: every parameter needs a source note")
            extra = set(leaf) - {"value", "label", "source", "sweep", "unit"}
            if extra:
                raise ValueError(f"{key}: unknown fields {sorted(extra)}")

    def table(self) -> list[tuple[str, Any, str, str]]:
        """(key, value, label, source) for every parameter, for documentation and reports."""
        return [(k, leaf["value"], leaf["label"], leaf["source"]) for k, leaf in self.leaves()]


def load_config(path: str | Path) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        tree = yaml.safe_load(f)
    return Config(tree, str(path))


def load_presets(config_path: str | None) -> dict[str, dict[str, Any]]:
    if config_path is None:
        return {}
    p = Path(config_path).with_name("presets.yaml")
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _block_end(lines: list[str], start: int, end: int, indent: int) -> int:
    """First line at or before ``indent`` (ignoring blanks and comments) in [start, end), else ``end``."""
    for i in range(start, end):
        s = lines[i].strip()
        if s and not s.startswith("#") and _indent(lines[i]) <= indent:
            return i
    return end


def set_leaf_in_file(path: str | Path, key: str, value: Any, label: str, source: str) -> None:
    """Set one parameter's value, label and source in a YAML config file **in place**, keeping every comment and
    every other line as it is (PyYAML would drop the comments). Missing groups and leaves are appended to their
    parent block. Used by the fitting commands, which write ``label: fitted`` with a source naming the command.
    """
    if label not in LABELS:
        raise ValueError(f"label must be one of {LABELS}")
    path = Path(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    parts = key.split(".")
    start, end = 0, len(lines)
    for depth, part in enumerate(parts):
        ind = " " * (2 * depth)
        found = None
        for i in range(start, end):
            ln = lines[i]
            if _indent(ln) == 2 * depth and ln.strip().split(":")[0] == part and ln.strip().startswith(part + ":"):
                found = i
                break
        if found is None:  # append the missing group or leaf at the end of its parent block
            new = [f"{ind}{part}:"]
            insert = end
            while insert > start and not lines[insert - 1].strip():
                insert -= 1
            lines[insert:insert] = new
            found = insert
            end += 1
        start = found + 1
        end = _block_end(lines, start, end, 2 * depth)
    ind = " " * (2 * len(parts))
    flow = yaml.safe_dump(value, default_flow_style=True, width=1 << 20).strip()
    if flow.endswith("\n...") or flow.endswith("..."):
        flow = flow.replace("\n...", "").rstrip(".").strip()
    fields = {"value": flow, "label": label, "source": json.dumps(source, ensure_ascii=False)}
    for name, text in fields.items():
        for i in range(start, end):
            if _indent(lines[i]) == len(ind) and lines[i].strip().startswith(name + ":"):
                lines[i] = f"{ind}{name}: {text}"
                break
        else:
            lines.insert(end, f"{ind}{name}: {text}")
            end += 1
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    check = load_config(path)
    if check.get(key) != yaml.safe_load(flow):
        raise RuntimeError(f"writing {key} into {path} did not round-trip")


def parse_override(text: str) -> tuple[str, Any]:
    """``"a.b=1.5"`` -> ``("a.b", 1.5)``; the value is parsed as YAML (numbers, lists, booleans, strings)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, val = text.split("=", 1)
    return key.strip(), yaml.safe_load(val)
