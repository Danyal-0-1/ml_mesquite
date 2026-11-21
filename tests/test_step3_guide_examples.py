"""Every worked example of docs/step_3/, recomputed: LEARN_THE_MODEL.md (G1-G8) and DATASET_CARD.md (D1-D6).

Each test is named after its label. The ones that read ``twin_v1`` skip when it is absent
(``$MESQ_OUTPUT_DIR/datasets/twin_v1``). If a document and the data disagree, fix whichever is wrong and record it in
WORKLOG.md.
"""

import hashlib
import json
import math

import numpy as np
import pandas as pd
import pytest

from mesqsim.conventions import wrap_deg
from mesqsim.learn.store import MOTION_CODES, windows
from mesqsim.paths import output_dir

K = 26013  # the worked-example step of val_KIT_12_v0


@pytest.fixture(scope="module")
def ds():
    root = output_dir() / "datasets" / "twin_v1"
    if not (root / "index.parquet").exists():
        pytest.skip("twin_v1 is not here ($MESQ_OUTPUT_DIR/datasets/twin_v1)")
    return root


@pytest.fixture(scope="module")
def idx(ds):
    return pd.read_parquet(ds / "index.parquet")


@pytest.fixture(scope="module")
def kit12(ds, idx):
    import h5py

    from mesqsim.learn.targets import read_targets

    row = idx.set_index("id").loc["val_KIT_12_v0"]
    with h5py.File(ds / row["shard"], "r") as h5:
        g = h5["sessions/val_KIT_12_v0"]
        a = {k: np.asarray(g[k]) for k in ("t_s", "x_pod", "x_pod_phone", "state_pod", "state_phone")}
        a["motion"] = np.asarray(g["strata/motion"])
        a.update(read_targets(g))
    return row, a


def _deg(rad):
    return float(np.rad2deg(rad))


# ------------------------------------------------------------------------------------------------ LEARN_THE_MODEL.md
@pytest.mark.amass
def test_G1_one_step_of_val_KIT_12_v0(kit12):
    row, a = kit12
    assert row["shard"] == "shards/val_00072.h5" and row["steps"] == 49684 and round(row["length_s"] / 60, 1) == 25.9
    assert row["suit_version"] == "suit_2026_11_13" and row["wire"] == "json"
    t = a["t_s"]
    assert t[0] == 1.71875 and t[-1] == 1554.3125 and np.allclose(np.diff(t), 0.03125)
    names = {v: n for n, v in MOTION_CODES.items()}
    u, c = np.unique(a["motion"], return_counts=True)
    assert {names[int(m)]: int(n) for m, n in zip(u, c, strict=True)} == {
        "preparation": 1065, "tpose": 1440, "still": 18420, "walk": 18962, "turn": 6965, "transition": 2832}
    runs = np.r_[0, np.flatnonzero(np.diff(a["motion"].astype(int))) + 1]
    assert len(runs) == 356 and runs[1] == 1065 and runs[2] == 1225 and a["motion"][1065] == MOTION_CODES["tpose"]
    assert t[K] == 814.625 and a["motion"][K] == MOTION_CODES["walk"] and a["state_pod"][K, 10] == 2
    x = a["x_pod"][K, 10].astype(float)
    np.testing.assert_allclose(x, [0.6890, 0.2179, -0.6912, 0.0496, -0.3476, -0.8534, 0.0216, 0.0], atol=5e-5)
    assert abs(np.linalg.norm(x[:3]) - 1.0) < 1e-4 and round(float(np.linalg.norm(x[3:6])), 3) == 0.923
    assert round(_deg(a["y_delta"][K, 10]), 3) == -113.013 and round(_deg(a["y_eps"][K, 10]), 3) == 172.501
    d, e = a["y_delta"][:, 10][a["valid_delta"][:, 10]], a["y_eps"][:, 10][a["valid_eps"][:, 10]]
    assert (round(_deg(d[0]), 2), round(_deg(d[-1]), 2)) == (0.53, -68.87)
    assert (round(_deg(e[0]), 2), round(_deg(e[-1]), 2)) == (-36.81, 86.87)
    absent = [5, 8, 11, 14]
    assert (a["state_pod"][:, absent] == 0).all() and (a["x_pod"][:, absent] == 0).all()
    assert int(np.argmax(a["valid_eps"].any(axis=1))) == 458
    first_msg = int(np.argmax(a["state_phone"] > 0))
    assert first_msg == 459 and t[first_msg] == 16.0625


def test_G2_the_windows_of_one_session():
    w = windows(49684)
    assert len(w) == 154 and w[:3].tolist() == [[0, 576], [320, 896], [640, 1216]]
    assert w[-1, 1] == 49536 and 49684 - w[-1, 1] == 148
    assert math.ceil(9400 / 32) == 294  # the validation split's windows (D1), in batches of 32


def test_G3_impulse_response_and_receptive_field():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    x = torch.zeros(1, 1, 7)
    x[0, 0, 0] = 1.0
    w = torch.tensor([[[1.0, 2.0, 3.0]]])
    y = F.conv1d(F.pad(x, ((3 - 1) * 2, 0)), w, dilation=2)
    assert y.flatten().tolist() == [3.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0]
    assert 1 + (3 - 1) * (1 + 2) == 7  # two layers, d = 1 and 2
    rf = 1 + sum(2 * (3 - 1) * d for d in (1, 2, 4, 8, 16, 32))
    assert rf == 253 and round(rf / 32, 1) == 7.9 and (3 - 1) * 32 == 64


def test_G4_parameter_counts_by_hand():
    torch = pytest.importorskip("torch")

    def n(c_in, c_out, k):
        return sum(p.numel() for p in torch.nn.Conv1d(c_in, c_out, k).parameters())

    assert (n(25, 64, 3), n(64, 64, 3), n(25, 64, 1), n(37, 64, 3), n(37, 64, 1), n(64, 3, 1)) == (
        4864, 12352, 1664, 7168, 2432, 195)
    assert 5 * 2 * 12352 == 123520
    assert 4864 + 12352 + 1664 + 123520 + 195 == 142595 and 7168 + 12352 + 2432 + 123520 + 195 == 145667


def test_G5_von_mises_numbers():
    from scipy.special import i0, i0e
    from scipy.stats import vonmises

    table = [(0.0, 50.0, -1.0345), (10.0, 50.0, -0.2749), (10.0, 5.0, 0.2185), (30.0, 50.0, 5.6642),
             (0.5, 1e4, -3.3054)]  # fmt: skip
    for err_deg, kappa, nll in table:
        assert round(float(-vonmises.logpdf(math.radians(err_deg), kappa)), 4) == nll
    assert [round(math.degrees(1 / math.sqrt(k)), 2) for k in (50.0, 5.0, 1e4)] == [8.10, 25.62, 0.57]
    assert round(1 - math.cos(math.radians(10.0)), 5) == 0.01519
    assert round((1e4 + math.log(i0e(1e4))) / math.log(10), 1) == 4340.5 and np.isinf(i0(720.0))
    assert np.isfinite(i0(700.0))
    assert round(math.exp(-5.0), 3) == 0.007 and round(math.exp(12.0), -4) == 160000.0  # the log κ clamp [-5, 12]
    assert round(math.degrees(1 / math.sqrt(4.0))) == 29  # κ < 4: σ about 29°


def test_G6_batches_and_the_time_estimate():
    per_batch_gpu, per_batch_cpu = 12 * 0.0596 * 1.25, 12 * 0.1986 * 1.25  # measured layer times, 12 layers, +25 %
    assert round(per_batch_gpu, 1) == 0.9 and round(per_batch_cpu) == 3
    rows = []
    for windows_n in (2394, 12141, 54153):  # D6
        b = math.ceil(windows_n / 32)
        rows.append((b, round(b * per_batch_gpu / 60), round(b * per_batch_cpu / 60)))
    assert rows == [(75, 1, 4), (380, 6, 19), (1693, 25, 84)]  # 84 min on the CPU: about 1.4 h


@pytest.mark.amass
def test_G7_the_inputs_and_two_columns(kit12):
    _, a = kit12
    assert 8 + 8 + 6 + 15 == 37
    v = a["x_pod_phone"][K, 10].astype(float)
    np.testing.assert_allclose(v, [0.1695, -0.6018, 0.7805, -0.9724, 0.0265, 0.2316], atol=5e-5)
    assert abs(np.linalg.norm(v[:3]) - 1) < 1e-5 and abs(np.linalg.norm(v[3:]) - 1) < 1e-5 and abs(v[:3] @ v[3:]) < 1e-5


@pytest.mark.amass
def test_G8_the_numbers_to_compare_with():
    scores = output_dir() / "scores"
    test_csv = output_dir() / "results_step3" / "twin_v1" / "baselines_test.csv"
    if not (scores / "B0__twin_v1__val" / "summary.json").exists() or not test_csv.exists():
        pytest.skip("the stored scores of S3-M8 and S3-M9 are not here")

    def summary(name):
        d = json.loads((scores / name / "summary.json").read_text())
        return round(d["median_deg"], 2), round(d["mean_deg"], 2), round(d["p95_deg"], 2)

    assert summary("mesqsim.learn.dummies-ZerosEps__twin_v1__val") == (91.51, 92.76, 174.40)
    assert summary("B0__twin_v1__val")[0] == 22.65 and summary("O2__twin_v1__val")[0] == 0.0
    t = pd.read_csv(test_csv)
    t = t[(t.stratum == "all") & (t.unit == "all")].set_index(["target", "method"])["median_deg"].round(2)
    assert (t["eps", "B0"], t["eps", "RIGID+B1"], t["eps", "RIGID+GEO"]) == (83.25, 43.25, 29.69)


# ------------------------------------------------------------------------------------------------ DATASET_CARD.md
@pytest.mark.amass
def test_D1_the_splits(ds, idx):
    subj = idx["subset"] + "/" + idx["subject"].astype(str)
    got = {}
    for split, g in idx.groupby("split"):
        got[split] = (len(g), subj[g.index].nunique(), round(g["length_s"].sum() / 3600, 1), int(g["steps"].sum()),
                      g["shard"].nunique(), int(sum(len(windows(int(n))) for n in g["steps"])))  # fmt: skip
    assert got == {"train": (519, 173, 152.3, 17548179, 65, 54153), "val": (74, 37, 26.4, 3037218, 10, 9400),
                   "test": (72, 36, 21.7, 2504962, 9, 7734)}  # fmt: skip
    assert subj.nunique() == 246 and round(idx["length_s"].sum() / 3600, 1) == 200.4
    meta = json.loads((ds / "meta.json").read_text())
    assert meta["subjects"]["train"] == 172 and meta["feature_lag_s"] == 0.1  # 172: the known limit (§7)
    manifest = (ds / "MANIFEST.csv").read_bytes()
    assert hashlib.sha256(manifest).hexdigest().startswith("f65b7ee61cea591c")
    assert len(pd.read_csv(ds / "MANIFEST.csv")) == 88


@pytest.mark.amass
def test_D2_the_conditions(idx):
    slips = idx["slips"].map(json.loads)
    cols = {"suit_2026_11_14": idx["suit_version"] == "suit_2026_11_14",
            "suit_2026_11_13": idx["suit_version"] == "suit_2026_11_13",
            "no_hipsalt": ~idx["hipsalt"].astype(bool),
            "limb_restart": slips.map(lambda s: s.get("limb_restart") is not None),
            "zero_at_start": slips.map(lambda s: bool(s.get("zero_at_start"))),
            "camera_mismatch": slips.map(lambda s: float(s.get("camera_mismatch_deg") or 0.0) != 0.0),
            "tree_mismatch": slips.map(lambda s: bool(s.get("tree_mismatch"))),
            "protocol": idx["program"].str.startswith("protocol")}  # fmt: skip
    got = {k: tuple(int(v[idx["split"] == s].sum()) for s in ("train", "val", "test")) for k, v in cols.items()}
    assert got == {"suit_2026_11_14": (263, 40, 45), "suit_2026_11_13": (256, 34, 27), "no_hipsalt": (83, 11, 14),
                   "limb_restart": (235, 29, 34), "zero_at_start": (315, 45, 43), "camera_mismatch": (134, 29, 23),
                   "tree_mismatch": (23, 2, 5), "protocol": (3, 0, 0)}  # fmt: skip
    assert ((idx["suit_version"] == "suit_2026_11_13") == (idx["wire"] == "json")).all()


@pytest.mark.amass
def test_D3_session_lengths(idx):
    minutes = idx["length_s"] / 60
    assert (round(minutes.min(), 1), round(minutes.median(), 1), round(minutes.max(), 1)) == (3.7, 18.0, 33.1)
    assert (int(idx["steps"].min()), int(idx["steps"].median()), int(idx["steps"].max())) == (7141, 34537, 63611)


@pytest.mark.amass
def test_D4_how_often_in_the_validation_split(ds, idx):
    import h5py

    from mesqsim.learn.targets import read_targets

    pod, phone = np.zeros(3), np.zeros(3)
    vd = ve = steps = 0
    for r in idx[idx["split"] == "val"].itertuples():
        with h5py.File(ds / r.shard, "r") as h5:
            g = h5[f"sessions/{r.id}"]
            pod += np.bincount(np.asarray(g["state_pod"]).ravel(), minlength=3)
            phone += np.bincount(np.asarray(g["state_phone"]), minlength=3)
            y = read_targets(g)
            vd, ve, steps = vd + int(y["valid_delta"].sum()), ve + int(y["valid_eps"].sum()), steps + r.steps
    assert np.round(100 * pod / pod.sum(), 2).tolist() == [18.71, 0.0, 81.29]  # lost, held, fresh
    assert np.round(100 * phone / phone.sum(), 2).tolist() == [19.72, 1.02, 79.26]  # none, orientation only, full
    assert (round(100 * vd / (14 * steps), 2), round(100 * ve / (15 * steps), 2)) == (79.76, 81.63)


@pytest.mark.amass
def test_D5_the_chain_identity(kit12):
    _, a = kit12
    e9, d10, e10 = _deg(a["y_eps"][K, 9]), _deg(a["y_delta"][K, 10]), _deg(a["y_eps"][K, 10])
    assert (round(e9, 3), round(d10, 3), round(e10, 3)) == (-74.486, -113.013, 172.501)
    assert round(e9 + d10, 3) == -187.499 and abs(wrap_deg(e9 + d10) - e10) < 1e-3


@pytest.mark.amass
def test_D6_the_parts(idx):
    train = idx[idx["split"] == "train"]
    got = {}
    for p in (16, 4, 1):
        s = train[train["part"] >= p]
        subjects = (s["subset"] + "/" + s["subject"].astype(str)).nunique()
        n_windows = int(sum(len(windows(int(n))) for n in s["steps"]))
        got[p] = (len(s), subjects, round(s["length_s"].sum() / 3600, 1), n_windows)
    assert got == {16: (30, 10, 6.8, 2394), 4: (129, 43, 34.2, 12141), 1: (519, 173, 152.3, 54153)}
