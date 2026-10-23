"""The M3 slice: one sequence -> frames -> Step 0 -> B0 and O2 (brief §7 M3; checks 5, 7, 8 in part)."""

import hashlib

import numpy as np
import pytest

from mesqsim import evaluate as ev
from mesqsim import io, step0
from mesqsim.body import JOINTS
from mesqsim.config import load_config
from mesqsim.conventions import wrap_rad
from mesqsim.paths import REPO_ROOT
from mesqsim.sim.motion import MotionSource
from mesqsim.sim.simulate import simulate_session
from mesqsim.sim.timing import pod_ms

CONFIG = REPO_ROOT / "configs" / "default.yaml"


def run(preset=None, seed=1, motion="synthetic:walk:20", **overrides):
    cfg = load_config(CONFIG)
    if preset:
        cfg = cfg.with_preset(preset)
    if overrides:
        cfg = cfg.with_overrides({k.replace("__", "."): v for k, v in overrides.items()})
    d = simulate_session(cfg, seed, MotionSource.from_spec(motion))
    data = {"frames": d.frames, "session": d.session, "labels_ticks": d.labels_ticks,
            "labels_joints": d.labels_joints, "meta": d.meta}  # fmt: skip
    series = step0.decode(d.frames, d.session)
    return d, data, series


@pytest.fixture(scope="module")
def offsets_only():
    return run("offsets_only")


def test_frames_table_matches_the_contract(offsets_only):
    d, _, _ = offsets_only
    io.validate_frames(d.frames)
    assert np.all(np.diff(d.frames["t_host_s"].to_numpy()) >= 0)
    assert np.all(d.frames["qw"].to_numpy() >= 0)  # the wire never carries w < 0


def test_step0_rebuilds_every_tick_and_flag(offsets_only):
    d, _, series = offsets_only
    m = series.merge(d.labels_ticks, on=["pod_id", "epoch", "tick"], how="outer", indicator=True,
                     suffixes=("", "_true"))  # fmt: skip
    assert (m["_merge"] == "both").all()
    np.testing.assert_array_equal(m["status"] == "fresh", m["new_to_host"])
    np.testing.assert_array_equal(m["received"], m["received_true"])
    np.testing.assert_array_equal(m["ms_lo"].astype(int), m["ms_lo_true"].astype(int))


def test_b0_equals_offset_differences_up_to_quantisation(offsets_only):
    """M3 'done when': B0's orientation error, measured through the decoded frames, is |a_c - a_p|."""
    d, data, series = offsets_only
    e = ev.evaluate(data, series)
    offs = np.array([p["offset_rad"] for p in d.meta["plan"]["pods"]])
    expected = np.rad2deg(np.abs(wrap_rad([offs[j.child] - offs[j.parent] for j in JOINTS])))
    b0 = e[e["baseline"] == "B0"]
    assert np.isfinite(b0["err_orient_deg"]).all()
    np.testing.assert_allclose(b0["err_orient_deg"], expected[b0["joint"]], atol=0.02)
    np.testing.assert_allclose(b0["err_heading_deg"], expected[b0["joint"]], atol=1e-9)
    assert expected.max() > 10.0  # arbitrary offsets: the problem is visible


def test_o2_error_is_zero(offsets_only):
    _, data, series = offsets_only
    o2 = ev.evaluate(data, series, baselines=("O2",))
    assert o2["err_heading_deg"].max() == 0.0
    assert o2["err_orient_deg"].max() < 0.02  # quantisation only


def test_zero_corruption():
    """Check 5 (first half): with every error off and exact timing, B0 = 0 up to quantisation and O2 = 0."""
    _, data, series = run("zero")
    e = ev.evaluate(data, series)
    b0, o2 = e[e["baseline"] == "B0"], e[e["baseline"] == "O2"]
    assert b0["err_heading_deg"].max() == 0.0 and o2["err_heading_deg"].max() == 0.0
    assert b0["err_orient_deg"].quantile(0.99) < 0.01
    assert b0["err_orient_deg"].max() < 0.02


@pytest.mark.slow
def test_ms_lo_unwrapped_across_three_wraps():
    """Check 7: 230 s of recording spans more than three ms_lo wraps (65.536 s each); unwrapping is exact."""
    d, _, series = run("offsets_only", motion="synthetic:stand:200")
    plan = d.meta["plan"]
    m = series[series["received"]].merge(d.labels_ticks, on=["pod_id", "epoch", "tick"])
    truth = np.empty(len(m))
    for i, p in enumerate(plan["pods"]):
        sel = (m["pod_id"] == i).to_numpy()

        class P:  # the drawn pod parameters
            boot_s, clock_ppm = p["boot_s"], p["clock_ppm"]

        truth[sel] = np.floor(pod_ms(P, m.loc[sel, "t_decode_s"].to_numpy()))
    assert m["ms_unwrapped"].max() > 3 * 65536
    np.testing.assert_array_equal(m["ms_unwrapped"].to_numpy(), truth)


def test_unrecorded_idle_gap_is_not_counted_as_loss():
    d, data, series = run("offsets_only", session__idle_s=600.0)
    assert (series["status"] == "lost").sum() == 0
    m = series.merge(d.labels_ticks, on=["pod_id", "epoch", "tick"], how="outer", indicator=True)
    assert (m["_merge"] == "both").all()
    assert series["epoch"].max() == 0  # ten minutes without frames is not mistaken for a reboot
    e = ev.evaluate(data, series, baselines=("O2",))
    assert e["err_heading_deg"].max() == 0.0


def _hash(folder):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.iterdir())}


def test_same_seed_same_bytes_different_seed_differs(tmp_path):
    """Check 8: identical config and seed give byte-identical files; another seed does not."""
    hashes = []
    for k, seed in enumerate((7, 7, 8)):
        d, _, _ = run(seed=seed, motion="synthetic:walk:5")
        meta = {"seed": seed, **d.meta}
        hashes.append(_hash(io.write_session(tmp_path / f"r{k}", d, meta)))
    assert hashes[0] == hashes[1]
    assert hashes[0]["frames.parquet"] != hashes[2]["frames.parquet"]
