"""Fitting commands write parameters into the YAML config in place, keeping comments (Step 2 brief rule 6)."""

import shutil

import pytest

from mesqsim.config import load_config, set_leaf_in_file
from mesqsim.paths import REPO_ROOT


@pytest.fixture
def cfg_copy(tmp_path):
    dst = tmp_path / "default.yaml"
    shutil.copy(REPO_ROOT / "configs" / "default.yaml", dst)
    shutil.copy(REPO_ROOT / "configs" / "presets.yaml", tmp_path / "presets.yaml")
    return dst


def test_existing_leaf_is_replaced_and_everything_else_kept(cfg_copy):
    before = cfg_copy.read_text(encoding="utf-8").split("\n")
    set_leaf_in_file(cfg_copy, "geometry.kappa_hold", 51.5, "fitted", "fitted by `test` on SYNTH val")
    after = cfg_copy.read_text(encoding="utf-8").split("\n")
    cfg = load_config(cfg_copy)
    assert cfg["geometry.kappa_hold"] == 51.5 and cfg.param("geometry.kappa_hold")["label"] == "fitted"
    changed = [i for i, (a, b) in enumerate(zip(before, after, strict=True)) if a != b]
    # only the value, label and source lines of that leaf change; comments and every other line stay as they were
    head = after.index("  kappa_hold:")
    assert 1 <= len(changed) <= 3 and all(head < i <= head + 3 for i in changed)
    assert all(after[i].strip().split(":")[0] in ("value", "label", "source") for i in changed)
    assert cfg["error_model.drift_sigma_deg_per_min"] == 0.2


def test_new_group_and_leaves_are_appended(cfg_copy):
    box = [[-10.5, 20.25], [-5.0, 5.0], [-30.0, 30.0]]
    set_leaf_in_file(cfg_copy, "geometry.rom_limits_deg.l_hip", box, "fitted", "test")
    set_leaf_in_file(cfg_copy, "geometry.rom_limits_deg.r_hip", box, "fitted", "test")
    cfg = load_config(cfg_copy)
    assert cfg["geometry.rom_limits_deg.l_hip"] == box and cfg["geometry.rom_limits_deg.r_hip"] == box
    assert cfg["geometry.window_s"] == 8.0
    set_leaf_in_file(cfg_copy, "geometry.rom_limits_deg.l_hip", [[0.0, 1.0]] * 3, "fitted", "again")
    assert load_config(cfg_copy)["geometry.rom_limits_deg.l_hip"] == [[0.0, 1.0]] * 3


def test_label_must_be_known(cfg_copy):
    with pytest.raises(ValueError):
        set_leaf_in_file(cfg_copy, "geometry.kappa_hold", 1.0, "guessed", "x")
