"""Session folders on disk: the frames table (same schema as the real logger), labels, session.json, meta.json.

Writing is deterministic: the same data give byte-identical files (no timestamps; sorted JSON keys; fixed
Parquet options), so a run's outputs can be compared by hash (brief §6, check 8).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

__all__ = [
    "FRAMES_SCHEMA",
    "write_parquet",
    "read_parquet",
    "write_json",
    "read_json",
    "write_session",
    "read_session",
    "validate_frames",
    "markdown_table",
]

FRAMES_SCHEMA = pa.schema(
    [
        ("t_host_s", pa.float64()),
        ("pod_id", pa.uint8()),
        ("batt", pa.uint8()),
        ("qx", pa.int16()),
        ("qy", pa.int16()),
        ("qz", pa.int16()),
        ("qw", pa.int16()),
        ("count", pa.uint16()),
        ("ms_lo", pa.uint16()),
    ]
)
"""The frames table (brief §5.3): one row per received frame, raw wire values, host arrival time in seconds."""


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(_jsonable(obj), indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_parquet(path: Path, df: pd.DataFrame, schema: pa.Schema | None = None) -> None:
    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    table = table.replace_schema_metadata(None)  # drop pandas metadata: identical bytes across pandas versions
    pq.write_table(table, path, compression="zstd", compression_level=3, write_statistics=False)


def read_parquet(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def validate_frames(df: pd.DataFrame) -> None:
    """Raise if ``df`` does not have exactly the frames-table columns and types."""
    got = pa.Schema.from_pandas(df, preserve_index=False).remove_metadata()
    if not got.equals(FRAMES_SCHEMA):
        raise ValueError(f"frames table schema differs from the contract:\n{got}\nexpected:\n{FRAMES_SCHEMA}")


def write_session(folder: Path, data, meta: dict) -> Path:
    """Write a :class:`mesqsim.sim.simulate.SessionData` into ``folder``."""
    folder.mkdir(parents=True, exist_ok=True)
    validate_frames(data.frames)
    write_parquet(folder / "frames.parquet", data.frames, FRAMES_SCHEMA)
    write_parquet(folder / "labels_ticks.parquet", data.labels_ticks)
    write_parquet(folder / "labels_joints.parquet", data.labels_joints)
    write_json(folder / "session.json", data.session)
    write_json(folder / "meta.json", meta)
    return folder


def read_session(folder: Path, labels: bool = True) -> dict:
    """Read a session folder. ``labels=False`` returns only what a real recording has (frames, session.json)."""
    folder = Path(folder)
    out = {"frames": read_parquet(folder / "frames.parquet"), "session": read_json(folder / "session.json")}
    if labels:
        out["labels_ticks"] = read_parquet(folder / "labels_ticks.parquet")
        out["labels_joints"] = read_parquet(folder / "labels_joints.parquet")
        out["meta"] = read_json(folder / "meta.json")
    return out


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """A GitHub-flavoured Markdown table of ``df`` (no extra dependency)."""
    def cell(v):
        if isinstance(v, float | np.floating):
            return "" if np.isnan(v) else format(float(v), floatfmt)
        return str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines) + "\n"
