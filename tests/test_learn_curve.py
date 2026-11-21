"""The learning curve and the sufficiency rule (Step 3 brief §6.3) on made-up curves with known answers."""

import numpy as np

from mesqsim.learn.curve import fit, predicted_gain, sufficient


def made_up(n, c=2.0, a=10.0, b=0.5):
    return c + a * np.power(n, -b)


def test_fit_recovers_a_known_power_law():
    n = np.array([4.0, 8.0, 16.0, 32.0, 64.0])
    f = fit(n, made_up(n))
    assert abs(f.c - 2.0) < 1e-4 and abs(f.a - 10.0) < 1e-3 and abs(f.b - 0.5) < 1e-4
    g_deg, g_rel = predicted_gain(f, 64.0)
    assert abs(g_deg - (made_up(64.0) - made_up(128.0))) < 1e-6 and abs(g_rel - g_deg / made_up(64.0)) < 1e-9


def test_not_sufficient_while_a_doubling_still_pays():
    n = np.array([4.0, 8.0, 16.0, 32.0, 64.0])  # one more doubling: 0.366° (11 %)
    seeds = np.stack([made_up(n) + d for d in (-0.05, 0.0, 0.05)])
    r = sufficient(n, seeds)
    assert not r["sufficient"] and not r["part1"] and r["gain_deg"] > 0.3


def test_relative_limit_binds_even_below_a_tenth_of_a_degree():
    n = np.array([256.0, 512.0, 1024.0, 2048.0, 4096.0])  # gain 0.046° but 2.1 % of 2.156°
    r = sufficient(n, np.stack([made_up(n) + d for d in (-0.06, 0.0, 0.06)]))
    assert 0.04 < r["gain_deg"] < 0.05 and r["gain_rel"] > 0.02 and not r["sufficient"]


def test_sufficient_when_the_doubling_is_within_the_seeds_spread():
    n = np.array([1024.0, 2048.0, 4096.0, 8192.0, 16384.0])  # gain 0.023° (1.1 %), seeds 0.12° apart
    r = sufficient(n, np.stack([made_up(n) + d for d in (-0.06, 0.0, 0.06)]))
    assert r["part1"] and r["part2"] and r["sufficient"]
    tight = sufficient(n, np.stack([made_up(n) + d for d in (-0.001, 0.0, 0.001)]))
    assert tight["part1"] and not tight["part2"] and not tight["sufficient"]  # the seeds agree more than it moves
