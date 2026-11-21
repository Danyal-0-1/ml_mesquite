"""The dataset store (Step 3 brief §6; §10 items 9 and 11): windows, nested parts, plans, and a tiny dataset made
twice, extended and verified."""

import numpy as np
import pytest

from mesqsim.learn import store


def test_windows_tile_the_loss_regions_inside_the_session():
    w = store.windows(2000)
    assert (w[:, 1] - w[:, 0] == store.WINDOW_STEPS).all() and w[0, 0] == 0 and w[-1, 1] <= 2000
    assert np.array_equal(np.diff(w[:, 1]), np.full(len(w) - 1, store.LOSS_STEPS))  # loss regions tile, no overlap
    assert 2000 - w[-1, 1] < store.LOSS_STEPS  # and reach the session's end
    assert len(store.windows(store.WINDOW_STEPS - 1)) == 0


def test_read_targets_reads_a_window_or_the_whole_session():
    """A loader reads one window's targets through targets.py (rule 8) without decompressing the whole session."""
    import h5py

    from mesqsim.learn.targets import read_targets

    rng = np.random.default_rng(0)
    arrays = {"y_delta": rng.normal(size=(700, 14)).astype(np.float32), "valid_delta": rng.random((700, 14)) > 0.2,
              "y_eps": rng.normal(size=(700, 15)).astype(np.float32), "valid_eps": rng.random((700, 15)) > 0.2}
    with h5py.File("targets.h5", "w", driver="core", backing_store=False) as h5:
        g = h5.create_group("sessions/s")
        for k, v in arrays.items():
            g.create_dataset(k, data=v, chunks=(256,) + v.shape[1:], compression="gzip")
        whole, part = read_targets(g), read_targets(g, 100, 676)
    for k, v in arrays.items():
        assert np.array_equal(whole[k], v) and np.array_equal(part[k], v[100:676]) and part[k].dtype == v.dtype


def test_nested_parts_are_nested_and_sized():
    subjects = [("S", str(k)) for k in range(173)]
    parts = store.nested_parts(subjects, 1)
    assert set(parts.values()) <= set(store.PARTS)
    for p in store.PARTS:  # a subject whose smallest part is 1/v is in every part 1/p with p <= v
        assert sum(v >= p for v in parts.values()) == len(subjects) // p
    assert store.nested_parts(subjects, 1) == parts


@pytest.fixture(scope="module")
def amass():
    from mesqsim.learn import checks

    if not checks.available():
        pytest.skip("needs AMASS and the SMPL+H model")
    return checks


@pytest.mark.amass
def test_plan_draws_then_compositions(amass):
    p = store.plan_dataset("train", 10, 1, max_subjects=2, per_composition=8)
    for _, g in p.groupby("subject"):
        assert g["program_seed"].iloc[:8].nunique() == 1 and g["program_seed"].iloc[8] != g["program_seed"].iloc[0]
        assert g["seed"].nunique() == 10 and list(g["composition"]) == [0] * 8 + [1] * 2
    tail = store.plan_dataset("train", 6, 1, first_variant=4, max_subjects=2, per_composition=8)
    assert tail.reset_index(drop=True).equals(p[p["variant"] >= 4].reset_index(drop=True))


@pytest.mark.amass
def test_the_same_seed_gives_the_same_manifest_and_extend_only_appends(amass):
    r = amass.manifest_numbers()
    assert r["identical"] and r["files"] == 4 and r["sessions"] == 4  # 3 shards and the index
    assert r["train_variants"] == [0, 1, 2, 3] and r["compositions"] == 1  # more draws of the one composition
    assert r["stored_unchanged"] and r["kept_shards"] == 3 and r["extended_ok"] and r["extended_sessions"] == 6


@pytest.mark.amass
def test_the_tiny_dataset_verifies_and_recomputes_as_a_stream(amass):
    r = amass.dataset_numbers("no_such_dataset")["tiny"]
    assert r["ok"], r
    assert all(r["recomputed"].values()) and r["windows"] > 0
    cov = store.coverage(amass.tiny_dataset("a"))
    assert set(cov.columns) == {"unit", "motion", "since_boot", "windows"}
    assert cov["unit"].nunique() == 15 + 3 and cov["windows"].sum() > 0
