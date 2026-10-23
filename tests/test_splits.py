"""Check 9 (brief §9): splits by subject, TotalCapture held out, variants in one split, a stable manifest."""

import numpy as np
import pandas as pd
import pytest

from mesqsim import cli
from mesqsim.config import load_config
from mesqsim.dataset import generate, pre_motion_s, session_seed, synthetic_sources
from mesqsim.paths import REPO_ROOT
from mesqsim.splits import _counts, assign_splits, check_manifest, manifest_sha256, write_manifest
from mesqsim.synth import make_sequence, write_amass_npz


def tree(n_by_subset: dict) -> pd.DataFrame:
    rows = []
    for subset, n in n_by_subset.items():
        for s in range(n):
            for q in range(2):
                rows.append({"path": f"{subset}/S{s:02d}/seq{q}_poses.npz", "subset": subset, "subject": f"S{s:02d}"})
    return pd.DataFrame(rows)


def test_subjects_never_straddle_splits_and_shares_follow_70_15_15():
    df = assign_splits(tree({"CMU": 20, "KIT": 3, "SFU": 1, "TotalCapture": 5}))
    assert check_manifest(df) == []
    per = df.groupby(["subset", "split"])["subject"].nunique().unstack(fill_value=0)
    assert per.loc["CMU"].to_dict() == {"holdout": 0, "test": 3, "train": 14, "val": 3}
    assert per.loc["KIT", "train"] == 2 and per.loc["KIT"].sum() == 3
    assert per.loc["SFU", "train"] == 1
    assert (df.loc[df["subset"] == "TotalCapture", "split"] == "holdout").all()


def test_largest_remainder_never_loses_a_subject():
    for n in range(0, 60):
        c = _counts(n)
        assert sum(c.values()) == n and c["train"] >= c["val"] >= 0


def test_assignment_ignores_row_order_and_other_subsets():
    base = tree({"CMU": 20, "KIT": 9})
    a = assign_splits(base)
    b = assign_splits(base.sample(frac=1.0, random_state=3))
    pd.testing.assert_frame_equal(a, b)
    c = assign_splits(pd.concat([base, tree({"BMLmovi": 7})], ignore_index=True))
    pd.testing.assert_frame_equal(a, c[c["subset"] != "BMLmovi"].reset_index(drop=True))


def test_manifest_hash_is_stable(tmp_path):
    df = assign_splits(tree({"CMU": 10}))
    h1 = write_manifest(df, tmp_path / "a.csv")
    h2 = write_manifest(df.sample(frac=1.0, random_state=1), tmp_path / "b.csv")
    assert h1 == h2 == manifest_sha256(tmp_path / "a.csv")


def test_check_manifest_reports_problems():
    df = assign_splits(tree({"CMU": 4, "TotalCapture": 2}))
    bad = df.copy()
    bad.loc[0, "split"] = "test" if bad.loc[0, "split"] != "test" else "train"
    bad.loc[bad["subset"] == "TotalCapture", "split"] = "train"
    problems = check_manifest(bad)
    assert any("appears in 2 splits" in p for p in problems)
    assert any("TotalCapture" in p for p in problems)


def test_split_command_on_an_amass_format_tree(tmp_path, monkeypatch):
    data, man = tmp_path / "data", tmp_path / "manifests"
    seq = make_sequence("walk", 1.0, 30.0)
    for rel in ("CMU/01/01_01_poses.npz", "CMU/02/02_01_poses.npz", "CMU/03/03_01_poses.npz",
                "TotalCapture/s1/walking1_poses.npz"):  # fmt: skip
        (data / "amass" / rel).parent.mkdir(parents=True, exist_ok=True)
        write_amass_npz(seq, data / "amass" / rel)
    monkeypatch.setenv("MESQ_DATA_DIR", str(data))
    monkeypatch.setenv("MESQ_MANIFEST_DIR", str(man))
    assert cli.main(["split", "--write"]) == 0
    df = pd.read_csv(man / "split_manifest.csv")
    assert len(df) == 4 and check_manifest(df) == []
    frozen = (man / "split_manifest.sha256").read_text().split()[0]
    assert frozen == manifest_sha256(man / "split_manifest.csv")
    assert cli.main(["split", "--write"]) == 0  # unchanged data: the frozen hash still matches
    (data / "amass" / "CMU" / "04").mkdir()
    write_amass_npz(seq, data / "amass" / "CMU" / "04" / "04_01_poses.npz")
    assert cli.main(["split", "--write"]) == 1  # changed data: refuses to replace a frozen manifest


def test_dataset_keeps_every_window_and_variant_in_its_sequences_split(tmp_path):
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml").with_overrides({
        "dataset.window_s": 4.0, "dataset.windows_per_sequence": 2, "dataset.motion_start_min": [1],
        "dataset.variants": 2,
    })  # fmt: skip
    sources = synthetic_sources(subjects_per_kind=1, duration_s=10.0)
    folder = generate(cfg, 5, sources, out_dir=tmp_path / "run", log=lambda s: None)
    ds = pd.read_csv(folder / "dataset.csv")
    assert len(ds) == 3 * 2 * 1 * 2
    assert (ds.groupby("source")["split"].nunique() == 1).all()
    assert ds["seed"].nunique() == len(ds)
    meta = pd.read_json(folder / "sessions" / ds["session"].iloc[0] / "meta.json", typ="series")
    assert meta["split"] == ds["split"].iloc[0]
    plan = meta["plan"]
    assert plan["phases"]["idle"][1] - plan["phases"]["idle"][0] == pytest.approx(60.0 - pre_motion_s(cfg))


def test_session_seeds_are_stable_and_distinct():
    a = session_seed(1, "x", 0, 10, 0)
    assert a == session_seed(1, "x", 0, 10, 0)
    others = {session_seed(1, "x", 0, 10, 1), session_seed(1, "x", 1, 10, 0), session_seed(1, "y", 0, 10, 0),
              session_seed(2, "x", 0, 10, 0), session_seed(1, "x", 0, 30, 0)}  # fmt: skip
    assert a not in others and len(others) == 5
    assert np.iinfo(np.uint32).max >= a >= 0
