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

LABELS = ("datasheet", "firmware", "assumption", "measured")


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


def parse_override(text: str) -> tuple[str, Any]:
    """``"a.b=1.5"`` -> ``("a.b", 1.5)``; the value is parsed as YAML (numbers, lists, booleans, strings)."""
    if "=" not in text:
        raise ValueError(f"override must look like key=value, got {text!r}")
    key, val = text.split("=", 1)
    return key.strip(), yaml.safe_load(val)
