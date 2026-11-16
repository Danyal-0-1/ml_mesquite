"""Step 2 additions to the conventions module (rule 10): Cardan angles and von Mises statistics."""

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.special import i0e, i1e

from mesqsim.conventions import (
    KAPPA_MAX,
    matrix_to_cardan_rad,
    quat_normalize,
    quat_to_cardan_rad,
    quat_to_matrix,
    vm_halfwidth_rad,
    vm_kappa_from_resultant,
    vm_logpdf,
    vm_mean_resultant,
    vm_one_minus_resultant,
    vm_predict_kappa,
    vm_product,
    wrap_rad,
)

SEQS = ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX")


@pytest.mark.parametrize("seq", SEQS)
def test_cardan_matches_scipy_intrinsic(seq):
    rng = np.random.default_rng(1)
    ang = np.c_[rng.uniform(-3.1, 3.1, 500), rng.uniform(-1.5, 1.5, 500), rng.uniform(-3.1, 3.1, 500)]
    q = Rotation.from_euler(seq, ang).as_quat()  # upper case: intrinsic
    got = quat_to_cardan_rad(q, seq)
    np.testing.assert_allclose(wrap_rad(got - ang), 0.0, atol=1e-9)
    np.testing.assert_allclose(got, Rotation.from_quat(q).as_euler(seq), atol=1e-9)


def test_cardan_rejects_proper_euler_sequences():
    with pytest.raises(ValueError):
        matrix_to_cardan_rad(np.eye(3), "XYX")


def test_cardan_of_a_single_axis_rotation():
    q = Rotation.from_rotvec([0.0, 0.0, 0.7]).as_quat()
    np.testing.assert_allclose(quat_to_cardan_rad(q, "ZXY"), [0.7, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(quat_to_cardan_rad(q, "XYZ"), [0.0, 0.0, 0.7], atol=1e-12)
    np.testing.assert_allclose(matrix_to_cardan_rad(quat_to_matrix(q), "YZX"), [0.0, 0.7, 0.0], atol=1e-12)


def test_mean_resultant_is_bessel_ratio_and_inverse_roundtrips():
    k = np.array([0.0, 1e-6, 0.01, 0.4, 1.0, 3.0, 25.0, 400.0, 999.0, 1000.0, 1001.0, 3e4, 1e7])
    exact = np.where(k > 0, i1e(np.maximum(k, 1e-300)) / i0e(np.maximum(k, 1e-300)), 0.0)
    np.testing.assert_allclose(vm_mean_resultant(k), exact, atol=1e-14)
    np.testing.assert_allclose(vm_kappa_from_resultant(vm_mean_resultant(k)), k, rtol=1e-6, atol=1e-12)
    assert vm_kappa_from_resultant(0.0) == 0.0
    assert vm_kappa_from_resultant(1.0) == KAPPA_MAX


def test_asymptotic_one_minus_resultant_joins_the_exact_value():
    k = np.array([1000.0 - 1e-9, 1000.0])
    assert vm_one_minus_resultant(k[0]) == pytest.approx(vm_one_minus_resultant(k[1]), rel=1e-9)


def test_product_of_two_von_mises_is_their_resultant():
    mu, k = vm_product(np.deg2rad(10.0), 4.0, np.deg2rad(40.0), 2.0)
    z = 4.0 * np.exp(1j * np.deg2rad(10.0)) + 2.0 * np.exp(1j * np.deg2rad(40.0))
    assert mu == pytest.approx(np.angle(z)) and k == pytest.approx(abs(z))
    # a factor with kappa = 0 changes nothing; equal and opposite cancel to no information
    assert vm_product(0.3, 7.0, -2.0, 0.0) == pytest.approx((0.3, 7.0))
    assert vm_product(0.0, 5.0, np.pi, 5.0)[1] == pytest.approx(0.0, abs=1e-12)
    # the product density, normalised, is the von Mises with the resultant parameters
    x = np.linspace(-np.pi, np.pi, 2001)
    dens = np.exp(4.0 * np.cos(x - np.deg2rad(10.0)) + 2.0 * np.cos(x - np.deg2rad(40.0)))
    dens /= np.trapezoid(dens, x)
    np.testing.assert_allclose(dens, np.exp(vm_logpdf(x - mu, k)), rtol=1e-6)


def test_predict_rule_and_large_kappa_limit():
    q, dt = 2e-6, 30.0
    for k in (5.0, 50.0, 500.0):
        # exact rule: A(k') = A(k)·exp(-q dt / 2)
        assert vm_mean_resultant(vm_predict_kappa(k, q, dt)) == pytest.approx(
            vm_mean_resultant(k) * np.exp(-q * dt / 2), rel=1e-10)
    for k in (1e3, 1e5, 1e8):  # large kappa: the variances add, to 1 %
        assert vm_predict_kappa(k, q, dt) == pytest.approx(1.0 / (1.0 / k + q * dt), rel=0.01)
    assert vm_predict_kappa(37.0, q, 0.0) == 37.0 and vm_predict_kappa(0.0, q, 10.0) == 0.0


@pytest.mark.parametrize("level", [0.68, 0.95])
def test_interval_halfwidth_matches_scipy_and_probability(level):
    for k in (0.0, 5e-5, 0.3, 2.0, 17.0, 250.0, 4e3, 1e6, 3e8):
        h = float(vm_halfwidth_rad(k, level))
        if 1e-4 <= k <= 1e7:
            assert h == pytest.approx(stats.vonmises.ppf(0.5 + level / 2.0, k), abs=2e-4)
        if k < 1e4:
            mass = quad(lambda x, kk=k: np.exp(vm_logpdf(x, kk)), -h, h)[0]
            assert mass == pytest.approx(level, abs=2e-3)
    assert float(vm_halfwidth_rad(0.0, level)) == pytest.approx(np.pi * level)


def test_logpdf_is_normalised():
    for k in (0.0, 0.7, 30.0, 2e4):
        mass = quad(lambda x, kk=k: np.exp(vm_logpdf(x, kk)), -np.pi, np.pi, points=[0.0], limit=200)[0]
        assert mass == pytest.approx(1.0, abs=1e-6)


def test_random_unit_quaternions_cardan_roundtrip():
    q = quat_normalize(np.random.default_rng(3).normal(size=(200, 4)))
    for seq in SEQS:
        a = quat_to_cardan_rad(q, seq)
        back = Rotation.from_euler(seq, a).as_matrix()
        np.testing.assert_allclose(back, quat_to_matrix(q), atol=1e-9)
