"""RIGID and the chain (Step 3 brief §7.3; check 52.1): the 7° example, exact recovery, the vertical blind spot."""

import numpy as np

from mesqsim.body import JOINTS, ROOT_POD
from mesqsim.conventions import exp_map, quat_mul, wrap_rad, yaw_quat
from mesqsim.learn.rigid import chain, chain_kappa, rigid


def _stream(t, w_world, psi, r0=None):
    """Reports q = Y(-psi) R(t) of a body turning at the world rate ``w_world`` (constant)."""
    r = exp_map(np.outer(t, w_world))
    if r0 is not None:
        r = quat_mul(r, np.broadcast_to(r0, r.shape))
    return quat_mul(yaw_quat(np.full(t.size, -psi)), r)


def test_the_seven_degree_example():
    """ω = (1, 0, 0) rad/s, ψ_h = 12°, ψ_ph = 5°: C = cos 7° = 0.992546, S = sin 7° = 0.121869, ε̂_h = 7.000°."""
    t = np.linspace(0.0, 1.0, 33)
    w = np.array([1.0, 0.0, 0.0])
    f = rigid(t, _stream(t, w, np.deg2rad(5.0)), t, _stream(t, w, np.deg2rad(12.0)))
    assert abs(np.rad2deg(f.eps_rad) - 7.0) < 1e-9
    a, b = np.array([np.cos(np.deg2rad(-5.0)), np.sin(np.deg2rad(-5.0))]), \
        np.array([np.cos(np.deg2rad(-12.0)), np.sin(np.deg2rad(-12.0))])
    c, s = a @ b, b[0] * a[1] - b[1] * a[0]
    assert round(c, 6) == 0.992546 and round(s, 6) == 0.121869


def test_exact_on_noise_free_rigid_motion_and_blind_about_the_vertical():
    rng = np.random.default_rng(0)
    t = np.linspace(0.0, 4.0, 129)
    for _ in range(20):
        w = rng.normal(size=3)
        r0 = exp_map(rng.normal(size=3))
        psi_h, psi_ph = rng.uniform(-np.pi, np.pi, 2)
        f = rigid(t, _stream(t, w, psi_ph, r0), t, _stream(t, w, psi_h, r0))
        assert abs(wrap_rad(f.eps_rad - (psi_h - psi_ph))) < 1e-9 and f.vertical_rms < 1e-9
    f = rigid(t, _stream(t, [0, 0, 1.0], 0.3), t, _stream(t, [0, 0, 1.0], 1.1))
    assert f.kappa == 0.0  # a rotation about the vertical only: no horizontal part


def test_chain_adds_relative_headings_outward():
    rng = np.random.default_rng(1)
    psi = rng.uniform(-np.pi, np.pi, 15)
    psi_ph = 0.4
    delta = np.array([wrap_rad(psi[j.child] - psi[j.parent]) for j in JOINTS])
    eps = chain(wrap_rad(psi[ROOT_POD] - psi_ph), delta)
    assert np.allclose(wrap_rad(eps - (psi - psi_ph)), 0.0, atol=1e-12)
    k = chain_kappa(100.0, np.full(14, 100.0))
    assert np.isclose(k[ROOT_POD], 100.0) and np.isclose(k[1], 50.0) and np.isclose(k[0], 100.0 / 3)
    assert chain_kappa(0.0, np.full(14, 100.0))[0] == 0.0


def test_the_causal_track_follows_a_stepped_heading_and_holds_when_still():
    """32 Hz tracks of a pelvis rocking about a horizontal axis; ε steps from 10° to 40° at 60 s; still from 90 s."""
    from mesqsim.learn.rigid import TRACK_WINDOW_S, rigid_track

    t = np.arange(0, 160.0, 1 / 32)
    rock = 0.3 * np.sin(2 * np.pi * 0.9 * t)
    rock[t >= 90.0] = rock[np.searchsorted(t, 90.0)]
    body = exp_map(np.stack([rock, 0.5 * rock, np.zeros_like(t)], axis=1))
    eps_true = np.where(t < 60.0, np.deg2rad(10.0), np.deg2rad(40.0))
    q_ph = quat_mul(yaw_quat(np.full(t.size, -0.2)), body)  # ψ_ph = 0.2
    q_h = quat_mul(yaw_quat(-(0.2 + eps_true)), body)  # ψ_h = 0.2 + ε
    eps, kappa = rigid_track(q_ph, q_h)
    k = {x: np.searchsorted(t, x) for x in (50.0, 60.0 + TRACK_WINDOW_S + 1, 150.0)}
    assert abs(np.rad2deg(eps[k[50.0]]) - 10.0) < 1e-6 and kappa[k[50.0]] > 1e3
    assert abs(np.rad2deg(eps[k[91.0]]) - 40.0) < 1e-6  # once the window has left the step behind
    still = k[150.0]  # the window holds no rotation: the last confident estimate, held
    assert abs(np.rad2deg(eps[still]) - 40.0) < 1e-6 and kappa[still] >= 100.0
    assert np.array_equal(rigid_track(q_ph[:2000], q_h[:2000])[0], eps[:2000])  # causal
