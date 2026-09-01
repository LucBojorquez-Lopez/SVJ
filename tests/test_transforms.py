"""
tests/test_transforms.py
========================
Unit tests for the TRANSFORMS registry in src/observables.py.

Covers:
  - Registry completeness: every transform has the required fields with
    correct types.
  - affine_flip: forward/inverse roundtrip, correctness of the flip, custom
    'a' parameter, requires constraint.
  - log: forward/inverse roundtrip, correctness vs np.log/np.exp, requires
    constraint.
  - abs_value: forward equals np.abs, inverse is identity for non-negative
    inputs, requires is None (no constraint).
  - boxcox: fit/forward/inverse roundtrip on lognormal data, known-lambda
    roundtrip, fitted lambda is finite, requires constraint.

These tests require only numpy/scipy and observables.py — no PYTHIA binary.
"""

import numpy as np
import pytest
from observables import TRANSFORMS

_RNG = np.random.default_rng(42)

_REQUIRED_FIELDS = {'fit', 'forward', 'inverse', 'n_fitted', 'requires', 'desc'}


# ── Registry structure ─────────────────────────────────────────────────────────

class TestRegistry:

    def test_all_expected_transforms_present(self):
        assert {'affine_flip', 'log', 'abs_value', 'boxcox'}.issubset(TRANSFORMS)

    def test_all_required_fields_present(self):
        for name, spec in TRANSFORMS.items():
            missing = _REQUIRED_FIELDS - spec.keys()
            assert not missing, f"Transform '{name}' missing fields: {missing}"

    def test_n_fitted_is_non_negative_int(self):
        for name, spec in TRANSFORMS.items():
            assert isinstance(spec['n_fitted'], int), (
                f"'{name}' n_fitted is not int")
            assert spec['n_fitted'] >= 0, (
                f"'{name}' n_fitted is negative")

    def test_fit_forward_inverse_are_callable(self):
        for name, spec in TRANSFORMS.items():
            assert callable(spec['fit']),     f"'{name}' fit is not callable"
            assert callable(spec['forward']), f"'{name}' forward is not callable"
            assert callable(spec['inverse']), f"'{name}' inverse is not callable"

    def test_requires_is_callable_or_none(self):
        for name, spec in TRANSFORMS.items():
            assert spec['requires'] is None or callable(spec['requires']), (
                f"'{name}' requires is neither None nor callable")

    def test_requires_returns_lo_hi_pair(self):
        for name, spec in TRANSFORMS.items():
            if spec['requires'] is None:
                continue
            try:
                result = spec['requires']()
            except TypeError:
                result = spec['requires'](a=1.0)
            assert len(result) == 2, (
                f"'{name}' requires() must return (lo, hi), got {result}")
            lo, hi = result
            assert lo <= hi, f"'{name}' requires() lo > hi: {lo} > {hi}"

    def test_desc_is_non_empty_string(self):
        for name, spec in TRANSFORMS.items():
            assert isinstance(spec['desc'], str) and spec['desc'], (
                f"'{name}' desc is empty or not a string")


# ── affine_flip ────────────────────────────────────────────────────────────────

class TestAffineFlip:

    def test_forward_is_a_minus_x(self):
        x = np.array([0.1, 0.4, 0.9])
        y = TRANSFORMS['affine_flip']['forward'](x, (), a=1.0)
        np.testing.assert_allclose(y, 1.0 - x)

    def test_inverse_is_a_minus_y(self):
        y = np.array([0.3, 0.6, 0.8])
        x = TRANSFORMS['affine_flip']['inverse'](y, (), a=1.0)
        np.testing.assert_allclose(x, 1.0 - y)

    def test_roundtrip_default_a(self):
        x = np.linspace(0.01, 0.99, 50)
        y, params = TRANSFORMS['affine_flip']['fit'](x, a=1.0)
        x_back = TRANSFORMS['affine_flip']['inverse'](y, params, a=1.0)
        np.testing.assert_allclose(x_back, x, rtol=1e-12)

    def test_roundtrip_custom_a(self):
        a = 5.0
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y, params = TRANSFORMS['affine_flip']['fit'](x, a=a)
        x_back = TRANSFORMS['affine_flip']['inverse'](y, params, a=a)
        np.testing.assert_allclose(x_back, x, rtol=1e-12)

    def test_n_fitted_is_zero(self):
        assert TRANSFORMS['affine_flip']['n_fitted'] == 0

    def test_fit_returns_empty_params(self):
        x = np.array([0.2, 0.5])
        _, params = TRANSFORMS['affine_flip']['fit'](x)
        assert params == ()

    def test_requires_upper_bound_equals_a(self):
        lo, hi = TRANSFORMS['affine_flip']['requires'](a=1.0)
        assert hi == 1.0
        assert lo == -np.inf

    def test_requires_upper_bound_custom_a(self):
        lo, hi = TRANSFORMS['affine_flip']['requires'](a=np.pi)
        assert hi == pytest.approx(np.pi)

    def test_values_above_a_violate_requires(self):
        lo, hi = TRANSFORMS['affine_flip']['requires'](a=1.0)
        assert not (2.0 < hi)  # 2.0 is NOT < 1.0, so it violates the constraint


# ── log ───────────────────────────────────────────────────────────────────────

class TestLog:

    def test_forward_equals_np_log(self):
        x = np.array([0.5, 1.0, np.e, 10.0])
        y = TRANSFORMS['log']['forward'](x, ())
        np.testing.assert_allclose(y, np.log(x), rtol=1e-12)

    def test_inverse_equals_np_exp(self):
        y = np.array([-2.0, 0.0, 1.0, 3.0])
        x = TRANSFORMS['log']['inverse'](y, ())
        np.testing.assert_allclose(x, np.exp(y), rtol=1e-12)

    def test_roundtrip_positive_values(self):
        x = _RNG.lognormal(size=200)
        y, params = TRANSFORMS['log']['fit'](x)
        x_back = TRANSFORMS['log']['inverse'](y, params)
        np.testing.assert_allclose(x_back, x, rtol=1e-10)

    def test_n_fitted_is_zero(self):
        assert TRANSFORMS['log']['n_fitted'] == 0

    def test_fit_returns_empty_params(self):
        _, params = TRANSFORMS['log']['fit'](np.array([1.0, 2.0]))
        assert params == ()

    def test_requires_lower_bound_zero(self):
        lo, hi = TRANSFORMS['log']['requires']()
        assert lo == 0.0
        assert hi == np.inf

    def test_unity_maps_to_zero(self):
        y = TRANSFORMS['log']['forward'](np.array([1.0]), ())
        np.testing.assert_allclose(y, [0.0], atol=1e-12)


# ── abs_value ─────────────────────────────────────────────────────────────────

class TestAbsValue:

    def test_forward_equals_np_abs(self):
        x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = TRANSFORMS['abs_value']['forward'](x, ())
        np.testing.assert_allclose(y, np.abs(x))

    def test_forward_non_negative_output(self):
        x = _RNG.normal(size=300)
        y = TRANSFORMS['abs_value']['forward'](x, ())
        assert np.all(y >= 0)

    def test_inverse_preserves_magnitude(self):
        # The inverse reconstructs a signed sample by alternating signs, so it
        # is magnitude-preserving rather than an identity.  Valid only when the
        # underlying signed distribution is symmetric about 0.
        y = np.array([0.0, 0.5, 1.0, 2.5])
        x = TRANSFORMS['abs_value']['inverse'](y, ())
        np.testing.assert_allclose(np.abs(x), y)

    def test_inverse_alternates_signs(self):
        y = np.array([0.0, 0.5, 1.0, 2.5])
        x = TRANSFORMS['abs_value']['inverse'](y, ())
        np.testing.assert_allclose(x, [0.0, -0.5, 1.0, -2.5])

    def test_inverse_does_not_mutate_input(self):
        y     = np.array([0.0, 0.5, 1.0, 2.5])
        y_ref = y.copy()
        TRANSFORMS['abs_value']['inverse'](y, ())
        np.testing.assert_allclose(y, y_ref)

    def test_inverse_is_balanced_in_sign(self):
        # Roughly half the reconstructed sample must be negative, otherwise the
        # symmetry assumption the transform relies on would not be reproduced.
        y = np.abs(_RNG.normal(size=1000))
        x = TRANSFORMS['abs_value']['inverse'](y, ())
        assert np.isclose((x < 0).mean(), 0.5, atol=0.01)

    def test_n_fitted_is_zero(self):
        assert TRANSFORMS['abs_value']['n_fitted'] == 0

    def test_fit_returns_empty_params(self):
        _, params = TRANSFORMS['abs_value']['fit'](np.array([-1.0, 2.0]))
        assert params == ()

    def test_requires_is_none(self):
        assert TRANSFORMS['abs_value']['requires'] is None

    def test_positive_inputs_are_unchanged_by_forward(self):
        x = np.array([0.1, 1.0, 3.14])
        y = TRANSFORMS['abs_value']['forward'](x, ())
        np.testing.assert_allclose(y, x)


# ── boxcox ────────────────────────────────────────────────────────────────────

class TestBoxCox:

    def test_roundtrip_lognormal(self):
        x = _RNG.lognormal(mean=5.0, sigma=0.5, size=500)
        y, params = TRANSFORMS['boxcox']['fit'](x)
        x_back = TRANSFORMS['boxcox']['inverse'](y, params)
        np.testing.assert_allclose(x_back, x, rtol=1e-5)

    def test_forward_inverse_known_lambda(self):
        lam = 0.3
        x = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        y = TRANSFORMS['boxcox']['forward'](x, (lam,))
        x_back = TRANSFORMS['boxcox']['inverse'](y, (lam,))
        np.testing.assert_allclose(x_back, x, rtol=1e-8)

    def test_forward_inverse_lambda_half(self):
        lam = 0.5
        x = np.array([1.0, 4.0, 9.0, 16.0])
        y = TRANSFORMS['boxcox']['forward'](x, (lam,))
        x_back = TRANSFORMS['boxcox']['inverse'](y, (lam,))
        np.testing.assert_allclose(x_back, x, rtol=1e-8)

    def test_forward_inverse_lambda_near_zero(self):
        lam = 0.05
        x = np.array([0.5, 1.0, 2.0, 8.0])
        y = TRANSFORMS['boxcox']['forward'](x, (lam,))
        x_back = TRANSFORMS['boxcox']['inverse'](y, (lam,))
        np.testing.assert_allclose(x_back, x, rtol=1e-6)

    def test_n_fitted_is_one(self):
        assert TRANSFORMS['boxcox']['n_fitted'] == 1

    def test_fit_returns_one_finite_param(self):
        x = _RNG.lognormal(size=300)
        y, params = TRANSFORMS['boxcox']['fit'](x)
        assert len(params) == 1
        assert np.isfinite(params[0]), "Fitted lambda is not finite"

    def test_lambda_finite_for_varied_distributions(self):
        for mean, sigma in [(3.0, 0.3), (6.0, 0.8), (1.0, 1.5)]:
            x = _RNG.lognormal(mean=mean, sigma=sigma, size=300)
            _, (lam,) = TRANSFORMS['boxcox']['fit'](x)
            assert np.isfinite(lam), f"lambda not finite for mean={mean}, sigma={sigma}"

    def test_requires_positive(self):
        lo, hi = TRANSFORMS['boxcox']['requires']()
        assert lo == 0.0
        assert hi == np.inf

    def test_output_is_monotone(self):
        x = np.sort(_RNG.lognormal(size=200))
        y, _ = TRANSFORMS['boxcox']['fit'](x)
        assert np.all(np.diff(y) >= 0), "BoxCox forward must be monotone for sorted input"
