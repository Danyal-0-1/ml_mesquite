"""The model kit (Step 3 brief §7.4; check 51.1): the contract, the dummy models, the loss references, and the tests
Danyal's code must pass. Each test of Danyal's code skips cleanly until its module exists (``pytest.importorskip``),
so the full check passes before Danyal starts:

* ``mesqsim/learn/loader.py``: ``WindowDataset(path, split, window_steps=576, loss_steps=320, seed=0)``;
* ``mesqsim/learn/models.py``: ``TCN(in_features, channels=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32),
  out_features=3)``;
* ``mesqsim/learn/loss.py``: ``cos_loss(cos_sin, theta, mask)`` and ``von_mises_nll(cos_sin, log_kappa, theta, mask)``;
* ``mesqsim/learn/train.py``: ``train(model, train_loader, val_loader, cfg)``.
"""

import json
import math
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from mesqsim.learn.contract import Model, SessionInputs, check_prediction, truncate
from mesqsim.learn.dummies import Cheat, Zeros

# ------------------------------------------------------------------------------------------- loss references
VM_CASES = [(0.0, 0.0), (0.3, 1.0), (-2.0, 1.0), (1.0, 50.0), (0.01, 1e4), (3.1, 1e5), (0.2, 1e7)]
"""(θ - μ, κ): includes large κ, where a naive log I₀(κ) overflows (I₀(1e4) is about 10^4340)."""


def vm_nll_reference(d, kappa):
    """-log of the von Mises density at θ - μ = d, by SciPy."""
    from scipy.stats import vonmises

    return float(-vonmises.logpdf(d, kappa)) if kappa > 0 else math.log(2 * math.pi)


def test_loss_references_agree_with_a_stable_formula():
    """The references themselves: SciPy's density against -κ cos d + log 2π + log i0e(κ) + κ (no overflow)."""
    from scipy.special import i0e

    for d, k in VM_CASES:
        stable = -k * math.cos(d) + math.log(2 * math.pi) + math.log(i0e(k)) + k
        assert abs(vm_nll_reference(d, k) - stable) < 1e-6 * max(1.0, abs(stable)), (d, k)


def test_danyals_von_mises_nll_matches_scipy():
    loss = pytest.importorskip("mesqsim.learn.loss")
    torch = pytest.importorskip("torch")
    mu = np.array([0.4, -1.0, 2.0, 0.0, -0.3, 1.0, 0.5])
    d = np.array([c[0] for c in VM_CASES])
    k = np.array([c[1] for c in VM_CASES])
    theta = mu + d
    cs = torch.tensor(np.stack([np.cos(mu), np.sin(mu)], axis=-1)[None])
    lk = torch.tensor(np.log(np.maximum(k, 1e-12))[None, :, None])
    mask = torch.ones(1, len(mu), dtype=torch.bool)
    for j in range(len(mu)):
        m = torch.zeros_like(mask)
        m[0, j] = True
        got = float(loss.von_mises_nll(cs, lk, torch.tensor(theta[None]), m))
        assert abs(got - vm_nll_reference(d[j], k[j])) < 1e-4 * max(1.0, abs(got)), (d[j], k[j], got)


def test_danyals_cos_loss():
    loss = pytest.importorskip("mesqsim.learn.loss")
    torch = pytest.importorskip("torch")
    theta = torch.tensor([[0.0, 1.0, -2.0]])
    cs = torch.stack([torch.cos(theta + 0.5), torch.sin(theta + 0.5)], dim=-1)
    got = float(loss.cos_loss(cs, theta, torch.ones_like(theta, dtype=torch.bool)))
    assert abs(got - (1.0 - math.cos(0.5))) < 1e-6


# ------------------------------------------------------------------------------------------- the networks
def _tcn(in_features):
    models = pytest.importorskip("mesqsim.learn.models")
    pytest.importorskip("torch")
    return models.TCN(in_features, channels=64, kernel_size=3, dilations=(1, 2, 4, 8, 16, 32), out_features=3)


def test_parameter_counts_by_hand():
    """TCN-J: 25 inputs; block 1 = 25·64·3 + 64 + 64·64·3 + 64 + (25·64 + 64), blocks 2-6 = 2 (64·64·3 + 64), head
    64·3 + 3: 18,880 + 5 × 24,704 + 195 = 142,595. TCN-P: 37 inputs: 21,952 + 123,520 + 195 = 145,667."""
    for n_in, want in ((25, 142_595), (37, 145_667)):
        assert sum(p.numel() for p in _tcn(n_in).parameters()) == want


def test_receptive_field_and_causality():
    torch = pytest.importorskip("torch")
    net = _tcn(25).eval()
    x = torch.randn(1, 25, 600)
    y = net(x)
    x2 = x.clone()
    x2[..., 400:] = torch.randn(1, 25, 200)  # change every input after t = 399
    assert torch.allclose(net(x2)[..., :400], y[..., :400])
    x3 = x.clone()
    x3[..., 399 - 253] += 1.0  # just outside the receptive field of step 399
    assert torch.allclose(net(x3)[..., 399], y[..., 399])
    x4 = x.clone()
    x4[..., 399 - 252] += 1.0  # just inside it
    assert not torch.allclose(net(x4)[..., 399], y[..., 399])


def _tiny_batches(torch, n_in=25, steps=300):
    """One batch in the guide's convention: ``x`` (B, C, T) float32, ``theta`` (B, T) rad, ``mask`` (B, T) bool; the
    network maps x to (B, 3, T): cos, sin and log κ."""
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, n_in, steps, generator=g)
    theta = torch.atan2(x[:, 1], x[:, 0])  # learnable from the inputs at the same step
    return [{"x": x, "theta": theta, "mask": torch.ones(2, steps, dtype=torch.bool)}]


def test_one_small_batch_can_be_overfitted_and_training_is_deterministic():
    """Danyal's ``train(model, train_loader, val_loader, cfg)`` with ``cfg = {"epochs", "lr", "seed"}``: on one batch
    the cos loss falls by 90 %, and two runs with the same seed end with identical weights (CPU)."""
    train = pytest.importorskip("mesqsim.learn.train")
    loss = pytest.importorskip("mesqsim.learn.loss")
    torch = pytest.importorskip("torch")
    batches = _tiny_batches(torch)
    cfg = {"epochs": 300, "lr": 3e-3, "seed": 0}

    def run():
        torch.manual_seed(0)
        net = _tcn(25)
        b = batches[0]
        before = float(loss.cos_loss(net(b["x"])[:, :2].transpose(1, 2), b["theta"], b["mask"]))
        train.train(net, batches, batches, cfg)
        after = float(loss.cos_loss(net(b["x"])[:, :2].transpose(1, 2), b["theta"], b["mask"]))
        return net, before, after

    a, before, after = run()
    b, _, _ = run()
    assert after < 0.1 * before
    assert all(torch.equal(p, q) for p, q in zip(a.state_dict().values(), b.state_dict().values(), strict=True))


# ------------------------------------------------------------------------------------------- the loader
def _loader_pair():
    loader = pytest.importorskip("mesqsim.learn.loader")
    root = os.environ.get("MESQ_TEST_DATASET")
    if not root:
        pytest.skip("set MESQ_TEST_DATASET to a dataset folder (e.g. $MESQ_OUTPUT_DIR/datasets/twin_v1)")
    a = loader.WindowDataset(root, "val", window_steps=576, loss_steps=320, seed=0)
    b = loader.WindowDataset(root, "val", window_steps=576, loss_steps=320, seed=0)
    return root, a, b


def test_loader_windows():
    """A fixed seed gives the same windows and batches."""
    _, a, b = _loader_pair()
    assert len(a) > 0 and len(a) == len(b)
    for i in range(min(5, len(a))):
        wa, wb = a[i], b[i]
        assert all(np.array_equal(np.asarray(wa[k]), np.asarray(wb[k])) for k in wa)  # a fixed seed, the same batch


def test_loader_windows_stay_in_one_session_of_the_split_and_are_causal():
    """Danyal's ``WindowDataset`` must expose ``windows``: one ``(session_id, start, end)`` per item, step indices into
    that session's arrays, end exclusive. Each window lies inside its session, the session is of the split, the
    window is ``window_steps`` long, and its loss region is its last ``loss_steps`` (so its inputs end where the loss
    does: causal)."""
    root, a, _ = _loader_pair()
    idx = pd.read_parquet(os.path.join(root, "index.parquet")).set_index("id")
    assert len(a.windows) == len(a)
    for sid, start, end in a.windows[:: max(1, len(a.windows) // 200)]:
        assert idx.loc[sid, "split"] == "val" and 0 <= start < end <= int(idx.loc[sid, "steps"])
        assert end - start == 576


def test_loader_masks_line_up_with_the_inputs():
    """Every item's ``mask`` has the window's time length, as its inputs do, and is False outside the loss region."""
    _, a, _ = _loader_pair()
    item = a[0]
    t = np.asarray(item["mask"]).shape[-1]
    assert t == 576 and np.asarray(item["x"]).shape[-1] == t
    assert not np.asarray(item["mask"])[..., : 576 - 320].any()


# ------------------------------------------------------------------------------------------- invariance and the phone
def _turned_features(turn_all: float = 0.0, pod: int | None = None, gamma: float = 0.0):
    """The short twin session's stored inputs, every pod and the phone turned by ``turn_all`` (C5), or pod ``pod``
    alone by ``gamma``, applied to the decoded reports as a turned world would deliver them."""
    from mesqsim.config import load_config
    from mesqsim.conventions import quat_mul, yaw_quat
    from mesqsim.learn.features import features
    from mesqsim.paths import REPO_ROOT
    from mesqsim.step0 import decode
    from mesqsim.twin import session_checks as sc

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    s = sc.short_session()
    cfg = load_config(REPO_ROOT / "configs" / "default.yaml")
    t = s.labels_steps["t_s"].to_numpy() + s.meta["plan"]["host_epoch_s"]
    series = decode(s.frames, s.session).copy()
    q = series[["qx", "qy", "qz", "qw"]].to_numpy().copy()
    sel = np.isfinite(q[:, 3]) & ((series["pod_id"].to_numpy() == pod) if pod is not None else True)
    q[sel] = quat_mul(yaw_quat(np.full(sel.sum(), turn_all if pod is None else gamma)), q[sel])
    series[["qx", "qy", "qz", "qw"]] = q
    phone = s.phone.copy()
    if pod is None and turn_all:
        h = turn_all / 2  # about the reported frame's +Y (ARKit's up), as a rotation within that frame
        qp = phone[["qx", "qy", "qz", "qw"]].to_numpy().copy()
        phone[["qx", "qy", "qz", "qw"]] = np.stack([np.cos(h) * qp[:, 0] + np.sin(h) * qp[:, 2],
                                                    np.cos(h) * qp[:, 1] + np.sin(h) * qp[:, 3],
                                                    np.cos(h) * qp[:, 2] - np.sin(h) * qp[:, 0],
                                                    np.cos(h) * qp[:, 3] - np.sin(h) * qp[:, 1]], axis=1)
    f = features(series, phone, t, cfg["geometry.clock_window_s"], cfg["geometry.clock_block_s"])
    return SessionInputs(t, f.x_pod, f.x_joint, f.x_phone, f.x_pod_phone, f.state_pod, f.state_phone)


def _tcn_j_input(inputs: SessionInputs, joint: int):
    """TCN-J's 25 inputs for one joint (§7.2): parent pod 8, child pod 8, relative rotation 6, joint type one-hot 3."""
    from mesqsim.body import JOINTS

    j = JOINTS[joint]
    onehot = np.zeros((inputs.steps, 3))
    onehot[:, ("hinge", "2dof", "3dof").index(j.jtype)] = 1.0
    return np.concatenate([inputs.x_pod[:, j.parent], inputs.x_pod[:, j.child], inputs.x_joint[:, joint], onehot],
                          axis=1).T[None].astype(np.float32)  # (1, 25, T)


@pytest.mark.amass
def test_danyals_network_is_blind_to_a_common_turn():
    """Turning every pod and the phone by one angle leaves every output identical (the inputs are, C5)."""
    torch = pytest.importorskip("torch")
    net = _tcn(25).eval()
    a, b = _turned_features(), _turned_features(turn_all=0.9)
    with torch.no_grad():
        ya, yb = net(torch.tensor(_tcn_j_input(a, 4))), net(torch.tensor(_tcn_j_input(b, 4)))
    assert torch.allclose(ya, yb, atol=1e-4)


@pytest.mark.amass
def test_a_trained_tcn_p_follows_one_pod_turned_alone():
    """§7.4: a trained TCN-P (a contract model, ``MESQ_TEST_TCNP=module:Class``, target "eps") shifts the turned pod's
    ε̂ by the turn and no other pod's: turning pod i's reports by Y(γ) makes ψ_i - γ its north, so ε_i moves by -γ;
    within 2° at γ = 30°, 90° and 180° (median over the session's last half)."""
    spec = os.environ.get("MESQ_TEST_TCNP")
    if not spec:
        pytest.skip("set MESQ_TEST_TCNP=module:Class to a trained per-pod model")
    from mesqsim.conventions import wrap_rad
    from mesqsim.learn.scorer import load_model

    model = load_model(spec)
    assert model.target == "eps"
    base = np.asarray(model.predict(_turned_features()).theta_rad)
    half = slice(base.shape[0] // 2, None)
    for gamma in np.deg2rad([30.0, 90.0, 180.0]):
        out = np.asarray(model.predict(_turned_features(pod=10, gamma=gamma)).theta_rad)
        shift = np.median(wrap_rad(out[half, 10] - base[half, 10] + gamma))
        assert abs(np.rad2deg(shift)) < 2.0, np.rad2deg(gamma)


# ------------------------------------------------------------------------------------------- the contract
@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A two-session dataset: one-minute programs of KIT subject 3, built by the dataset's own code."""
    from mesqsim.twin import session_checks as sc

    if not sc.available():
        pytest.skip("needs AMASS (KIT subject 3) and the SMPL+H model")
    import h5py

    from mesqsim.config import load_config
    from mesqsim.learn.store import DATASET_DEFAULTS, build_session, draw_conditions, write_session_group
    from mesqsim.paths import REPO_ROOT
    from mesqsim.twin.calibration import PRIORS
    from mesqsim.twin.config import load_twin_config

    root = tmp_path_factory.mktemp("tiny")
    (root / "shards").mkdir()
    cfgt, cfg1 = load_twin_config(), load_config(REPO_ROOT / "configs" / "default.yaml")
    post = pd.DataFrame([{p.key: (p.choices[0] if p.kind == "choice" else
                                  (math.sqrt(p.lo * p.hi) if p.kind == "log_uniform" else 0.5 * (p.lo + p.hi)))
                          for p in PRIORS}])  # fmt: skip
    rows = []
    with h5py.File(root / "shards" / "val_00000.h5", "w") as h5:
        for v, seed in enumerate((11, 12)):
            row = {"id": f"val_KIT_3_v{v}", "split": "val", "subset": "KIT", "subject": "3", "variant": v,
                   "composition": 0, "program_seed": 7, "seed": seed}  # two draws of one composition
            cond = draw_conditions(cfgt, post, PRIORS, seed, 1.5, DATASET_DEFAULTS["shares"] | {"protocol": 0.0})
            built = build_session(row, cond, cfgt, cfg1, length_min=1.0)
            write_session_group(h5, row["id"], built)
            rows.append(row | built["info"] | {"shard": "shards/val_00000.h5"})
    pd.DataFrame(rows).to_parquet(root / "index.parquet", index=False)
    return root


def _first_inputs(root):
    import h5py

    idx = pd.read_parquet(root / "index.parquet")
    r = idx.iloc[0]
    with h5py.File(root / r["shard"], "r") as h5:
        return SessionInputs.from_group(h5[f"sessions/{r['id']}"], source=f"{root / r['shard']}#{r['id']}")


@pytest.mark.amass
def test_zeros_meets_the_contract_and_scores_exactly_as_b0(tiny):
    from mesqsim.learn.scorer import score

    z = Zeros()
    assert isinstance(z, Model)
    inputs = _first_inputs(tiny)
    check_prediction(z, inputs, z.predict(inputs))
    a, b = score(z, tiny, "val"), score("B0", tiny, "val")
    for key in ("median_deg", "mean_deg", "p95_deg", "steps"):
        assert a["summary"][key] == b["summary"][key], key
    cols = ["stratum", "value", "unit", "n", "mean_deg", "median_deg", "p95_deg"]
    assert a["table"][cols].equals(b["table"][cols])
    assert a["summary"]["causality_error_rad"] == 0.0


@pytest.mark.amass
def test_cheat_fails_the_label_blindness_test(tiny, tmp_path):
    """Inference must run with the labels deleted: Zeros does, Cheat cannot."""
    import h5py

    blind = tmp_path / "blind.h5"
    shutil.copy(tiny / "shards" / "val_00000.h5", blind)
    with h5py.File(blind, "a") as h5:
        for sid in h5["sessions"]:
            for k in ("y_delta", "valid_delta", "y_eps", "valid_eps"):
                del h5[f"sessions/{sid}/{k}"]
    sid = pd.read_parquet(tiny / "index.parquet")["id"].iloc[0]
    with h5py.File(blind, "r") as h5:
        inputs = SessionInputs.from_group(h5[f"sessions/{sid}"], source=f"{blind}#{sid}")
    z, full = Zeros(), _first_inputs(tiny)
    assert np.array_equal(z.predict(inputs).theta_rad, z.predict(full).theta_rad)
    assert np.abs(Cheat().predict(full).theta_rad).max() > 0  # it reads the labels when it can
    with pytest.raises(KeyError):
        Cheat().predict(inputs)


@pytest.mark.amass
def test_stored_inputs_carry_no_label(tiny):
    inputs = _first_inputs(tiny)
    assert not any(k.startswith("y_") or k.startswith("valid_") for k in inputs.__dict__)
    assert "psi" not in json.dumps(inputs.attrs.get("conditions", ""))
    t = truncate(inputs, 10)
    assert t.steps == 11 and t.x_pod.shape[0] == 11
