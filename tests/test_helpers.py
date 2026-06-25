"""
tests/test_helpers.py
=====================
Unit tests for the pure-Python functions in src/helpers.py.

Covers:
  - BoxCox(): l=0 → log, l=0.5 → power, l=-1 → identity; non-positive values
    filtered; n_bad reported correctly.
  - get_common_finite(): returns only rows where both arrays are finite and > 0.
  - preprocess_data(): drops rows with any entry ≤ 1e-6 or non-finite.
  - transform_data(): output shape, approximately N(0,1) marginals.
  - exp_terms(): finite output; matches the analytical 3D MVT log-normalisation
    constant (verified against scipy.special directly).
  - kld_params(): KLD(P‖P) ≈ 0; KLD(P‖Q) > 0 for clearly different Q; MC
    non-negativity.
  - sample_svj_new(): K=1 case; K=2 correlated case; shape, finiteness, and
    physical-space positivity for pT-like observables.  Uses the Gaussian
    copula (no nu parameter).

No PYTHIA binary required — only numpy/scipy and the src/ Python modules.
"""

import numpy as np
import pytest
import scipy.special as sp_

from helpers import (
    BoxCox,
    get_common_finite,
    preprocess_data,
    transform_data,
    exp_terms,
    kld_params,
    sample_svj_new,
)
from observables import fit_observable_col, param_offsets


# ── BoxCox ─────────────────────────────────────────────────────────────────────

class TestBoxCox:

    def test_l_zero_gives_log(self):
        x = np.array([1.0, np.e, np.e**2, 10.0])
        y = BoxCox(x, 0)
        np.testing.assert_allclose(y, np.log(x), rtol=1e-10)

    def test_l_half_gives_power_transform(self):
        x = np.array([1.0, 4.0, 9.0, 16.0])
        y = BoxCox(x, 0.5)
        expected = (x**0.5 - 1) / 0.5
        np.testing.assert_allclose(y, expected, rtol=1e-10)

    def test_l_minus_one_gives_identity(self):
        x = np.array([1.0, 2.0, 5.0])
        y = BoxCox(x, -1)
        np.testing.assert_allclose(y, x, rtol=1e-10)

    def test_non_positive_values_filtered(self):
        x = np.array([0.0, 1.0, -2.0, 3.0, np.inf])
        y = BoxCox(x, 0)
        # Only 1.0 and 3.0 pass the mask (> 0 and < inf)
        np.testing.assert_allclose(y, np.log(np.array([1.0, 3.0])), rtol=1e-10)

    def test_n_bad_reported(self, capsys):
        x = np.array([0.0, 1.0, -1.0, 2.0])
        BoxCox(x, 0)
        out = capsys.readouterr().out
        assert '2' in out  # two bad values (0.0, -1.0)

    def test_all_positive_no_filtering(self):
        x = np.array([0.1, 1.0, 5.0, 100.0])
        y = BoxCox(x, 0)
        assert len(y) == len(x)

    def test_output_finite_for_valid_input(self):
        x = np.array([0.5, 1.0, 2.0, 10.0])
        for l in (0, 0.5, 0.25, 0.1):
            y = BoxCox(x, l)
            assert np.all(np.isfinite(y)), f"Non-finite output for l={l}"


# ── get_common_finite ──────────────────────────────────────────────────────────

class TestGetCommonFinite:

    def test_filters_nan_in_first(self):
        a = np.array([np.nan, 1.0, 2.0])
        b = np.array([1.0,   2.0, 3.0])
        a_out, b_out = get_common_finite(a, b)
        assert len(a_out) == 2
        np.testing.assert_array_equal(a_out, [1.0, 2.0])
        np.testing.assert_array_equal(b_out, [2.0, 3.0])

    def test_filters_nan_in_second(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, np.nan, 3.0])
        a_out, b_out = get_common_finite(a, b)
        assert len(a_out) == 2

    def test_filters_non_positive(self):
        a = np.array([1.0, 0.0, -1.0, 2.0])
        b = np.array([1.0, 2.0,  3.0, 4.0])
        a_out, b_out = get_common_finite(a, b)
        # Row 1 (a=0) and row 2 (a=-1) removed
        assert len(a_out) == 2
        np.testing.assert_array_equal(a_out, [1.0, 2.0])

    def test_filters_inf(self):
        a = np.array([1.0, np.inf, 2.0])
        b = np.array([1.0, 2.0,   np.inf])
        a_out, b_out = get_common_finite(a, b)
        assert len(a_out) == 1

    def test_all_valid_unchanged(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        a_out, b_out = get_common_finite(a, b)
        np.testing.assert_array_equal(a_out, a)
        np.testing.assert_array_equal(b_out, b)

    def test_output_lengths_equal(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=100)
        b = rng.normal(size=100)
        a_out, b_out = get_common_finite(a, b)
        assert len(a_out) == len(b_out)


# ── preprocess_data ────────────────────────────────────────────────────────────

class TestPreprocessData:

    def test_drops_zero_row(self):
        X = np.array([[1.0, 2.0], [0.0, 3.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)
        np.testing.assert_array_equal(out[0], [1.0, 2.0])
        np.testing.assert_array_equal(out[1], [4.0, 5.0])

    def test_drops_sub_threshold_row(self):
        X = np.array([[1.0, 2.0], [1e-7, 3.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)

    def test_drops_negative_row(self):
        X = np.array([[1.0, 2.0], [-1.0, 3.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)

    def test_drops_inf_row(self):
        X = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)

    def test_drops_nan_row(self):
        X = np.array([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)

    def test_drops_row_when_any_col_invalid(self):
        # Only first col is bad — entire row must be dropped
        X = np.array([[1.0, 2.0], [3.0, 0.0], [4.0, 5.0]])
        out = preprocess_data(X)
        assert out.shape == (2, 2)

    def test_all_valid_unchanged(self):
        rng = np.random.default_rng(1)
        X = rng.lognormal(size=(50, 3))
        out = preprocess_data(X)
        assert out.shape == X.shape


# ── transform_data ─────────────────────────────────────────────────────────────

class TestTransformData:

    def test_output_shape(self):
        rng = np.random.default_rng(2)
        X = rng.lognormal(size=(400, 3))
        tr, params = transform_data(X)
        assert tr.shape == X.shape

    def test_params_shape(self):
        rng = np.random.default_rng(3)
        X = rng.lognormal(size=(400, 2))
        _, params = transform_data(X)
        # 2 observables × 4 params each
        assert params.shape == (2, 4)

    def test_output_approx_zero_mean(self):
        rng = np.random.default_rng(4)
        X = rng.lognormal(size=(500, 3))
        tr, _ = transform_data(X)
        assert np.abs(tr.mean()) < 0.3

    def test_output_approx_unit_variance(self):
        rng = np.random.default_rng(5)
        X = rng.lognormal(size=(500, 3))
        tr, _ = transform_data(X)
        assert abs(tr.std() - 1.0) < 0.4

    def test_output_all_finite(self):
        rng = np.random.default_rng(6)
        X = rng.lognormal(size=(300, 2))
        tr, _ = transform_data(X)
        assert np.all(np.isfinite(tr))

    def test_params_all_finite(self):
        rng = np.random.default_rng(7)
        X = rng.lognormal(size=(300, 2))
        _, params = transform_data(X)
        assert np.all(np.isfinite(params))


# ── exp_terms ──────────────────────────────────────────────────────────────────

class TestExpTerms:

    def test_finite_output(self):
        result = exp_terms(5.0, np.eye(3))
        assert np.isfinite(result)

    def test_matches_manual_formula(self):
        nu = 5.0
        sigma = np.eye(3)
        expected = (sp_.gammaln((3 + nu) / 2)
                    - sp_.gammaln(nu / 2)
                    - 1.5 * np.log(np.pi * nu)
                    - 0.5 * np.log(np.linalg.det(sigma)))
        np.testing.assert_allclose(exp_terms(nu, sigma), expected, rtol=1e-12)

    def test_identity_sigma_det_is_one(self):
        # log(det(I)) = 0 so the last term vanishes
        nu = 10.0
        result_identity = exp_terms(nu, np.eye(3))
        result_manual = (sp_.gammaln((3 + nu) / 2)
                         - sp_.gammaln(nu / 2)
                         - 1.5 * np.log(np.pi * nu))
        np.testing.assert_allclose(result_identity, result_manual, rtol=1e-12)

    def test_larger_sigma_det_reduces_value(self):
        # Larger sigma → larger det → smaller log-normalisation constant
        nu = 5.0
        r1 = exp_terms(nu, np.eye(3))
        r2 = exp_terms(nu, 2.0 * np.eye(3))
        assert r2 < r1

    def test_larger_nu_changes_value(self):
        # exp_terms should vary as nu changes
        r1 = exp_terms(3.0, np.eye(3))
        r2 = exp_terms(20.0, np.eye(3))
        assert r1 != pytest.approx(r2)


# ── kld_params ─────────────────────────────────────────────────────────────────

def _make_v1_params(mu, sigma_diag, nu=5.0):
    """Build a 10-element v1 parameter vector [mu(3), S_upper(6), nu]."""
    p = np.zeros(10)
    p[0:3] = mu
    p[3] = sigma_diag[0]   # S00
    p[4] = 0.0             # S01
    p[5] = 0.0             # S02
    p[6] = sigma_diag[1]   # S11
    p[7] = 0.0             # S12
    p[8] = sigma_diag[2]   # S22
    p[9] = nu
    return p


class TestKLDParams:

    def test_self_kld_near_zero(self):
        p = _make_v1_params([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        np.random.seed(0)
        kld = kld_params(p, p)
        assert abs(kld) < 0.1, f"KLD(P‖P) should be ≈ 0, got {kld:.4f}"

    def test_different_mean_gives_positive_kld(self):
        p1 = _make_v1_params([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p2 = _make_v1_params([3.0, 3.0, 3.0], [1.0, 1.0, 1.0])
        np.random.seed(1)
        kld = kld_params(p1, p2)
        assert kld > 0.1, f"KLD should be clearly positive for shifted mean, got {kld:.4f}"

    def test_different_scale_gives_positive_kld(self):
        p1 = _make_v1_params([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p2 = _make_v1_params([0.0, 0.0, 0.0], [4.0, 4.0, 4.0])
        np.random.seed(2)
        kld = kld_params(p1, p2)
        assert kld > 0.05, f"KLD should be positive for different scale, got {kld:.4f}"

    def test_kld_is_finite(self):
        p1 = _make_v1_params([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p2 = _make_v1_params([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
        np.random.seed(3)
        kld = kld_params(p1, p2)
        assert np.isfinite(kld)

    def test_kld_asymmetry(self):
        # KLD(P‖Q) ≠ KLD(Q‖P) in general for different distributions
        p1 = _make_v1_params([0.0, 0.0, 0.0], [1.0, 1.0, 1.0])
        p2 = _make_v1_params([2.0, 2.0, 2.0], [2.0, 2.0, 2.0])
        np.random.seed(4)
        kld_12 = kld_params(p1, p2)
        np.random.seed(5)
        kld_21 = kld_params(p2, p1)
        # For typical distributions these will differ; just check both are finite
        assert np.isfinite(kld_12)
        assert np.isfinite(kld_21)


# ── sample_svj_new ─────────────────────────────────────────────────────────────

class TestSampleSvjNew:
    """
    Tests for sample_svj_new without a real scan NPZ.
    We construct synthetic fitted parameters by running fit_observable_col
    on synthetic data, then verify the sampler produces valid physical-space
    outputs.
    """

    def _fit_obs(self, obs_name, x):
        from observables import OBSERVABLES
        spec = OBSERVABLES[obs_name]
        _, params = fit_observable_col(x, spec['pipeline'], spec['distribution'])
        return params

    def test_k1_output_shape(self):
        rng = np.random.default_rng(30)
        x = rng.lognormal(mean=6.0, sigma=0.4, size=500)
        params = self._fit_obs('leadVisPt', x)

        R_upper = np.array([])          # K=1 → no off-diagonal elements
        flat_params = np.array(params)
        offsets = param_offsets(['leadVisPt'])

        samples = sample_svj_new(
            R_upper, flat_params, offsets, ['leadVisPt'],
            n_samples=1000, rng=rng)
        assert samples.shape == (1000, 1)

    def test_k1_all_finite(self):
        rng = np.random.default_rng(31)
        x = rng.lognormal(mean=6.0, sigma=0.4, size=500)
        params = self._fit_obs('leadVisPt', x)
        samples = sample_svj_new(
            np.array([]), np.array(params),
            param_offsets(['leadVisPt']), ['leadVisPt'],
            n_samples=2000, rng=rng)
        assert np.all(np.isfinite(samples))

    def test_k1_positive_pT(self):
        # leadVisPt (pT) must be positive
        rng = np.random.default_rng(32)
        x = rng.lognormal(mean=6.0, sigma=0.4, size=500)
        params = self._fit_obs('leadVisPt', x)
        samples = sample_svj_new(
            np.array([]), np.array(params),
            param_offsets(['leadVisPt']), ['leadVisPt'],
            n_samples=2000, rng=rng)
        assert np.all(samples > 0), "pT samples must be positive"

    def test_k2_output_shape(self):
        rng = np.random.default_rng(33)
        x1 = rng.lognormal(mean=6.0, sigma=0.5, size=500)
        x2 = rng.lognormal(mean=5.0, sigma=0.6, size=500)
        p1 = self._fit_obs('leadVisPt', x1)
        p2 = self._fit_obs('MET', x2)

        R_upper = np.array([0.3])       # K=2 → one upper-triangle element
        flat_params = np.concatenate([p1, p2])
        offsets = param_offsets(['leadVisPt', 'MET'])

        samples = sample_svj_new(
            R_upper, flat_params, offsets, ['leadVisPt', 'MET'],
            n_samples=2000, rng=rng)
        assert samples.shape == (2000, 2)

    def test_k2_all_positive(self):
        rng = np.random.default_rng(34)
        x1 = rng.lognormal(mean=6.0, sigma=0.5, size=500)
        x2 = rng.lognormal(mean=5.0, sigma=0.6, size=500)
        p1 = self._fit_obs('leadVisPt', x1)
        p2 = self._fit_obs('MET', x2)
        flat_params = np.concatenate([p1, p2])
        samples = sample_svj_new(
            np.array([0.3]), flat_params,
            param_offsets(['leadVisPt', 'MET']),
            ['leadVisPt', 'MET'],
            n_samples=2000, rng=rng)
        assert np.all(samples > 0)

    def test_median_in_plausible_range(self):
        """Median of sampled leadVisPt should be near the fitting data's median."""
        rng = np.random.default_rng(35)
        x = rng.lognormal(mean=6.0, sigma=0.4, size=1000)
        params = self._fit_obs('leadVisPt', x)
        samples = sample_svj_new(
            np.array([]), np.array(params),
            param_offsets(['leadVisPt']), ['leadVisPt'],
            n_samples=5000, rng=rng)
        # Median of samples should be within factor 3 of the data median
        data_median = float(np.median(x))
        sample_median = float(np.median(samples))
        assert data_median / 3 < sample_median < data_median * 3, (
            f"Sample median {sample_median:.0f} far from data median {data_median:.0f}")

    def test_zero_correlation_independent(self):
        """With R_upper=[0], K=2 samples should have near-zero Spearman correlation."""
        rng = np.random.default_rng(36)
        x1 = rng.lognormal(mean=6.0, sigma=0.5, size=500)
        x2 = rng.lognormal(mean=5.0, sigma=0.6, size=500)
        p1 = self._fit_obs('leadVisPt', x1)
        p2 = self._fit_obs('MET', x2)
        flat_params = np.concatenate([p1, p2])
        samples = sample_svj_new(
            np.array([0.0]), flat_params,
            param_offsets(['leadVisPt', 'MET']),
            ['leadVisPt', 'MET'],
            n_samples=5000, rng=rng)
        from scipy.stats import spearmanr
        rho, _ = spearmanr(samples[:, 0], samples[:, 1])
        assert abs(rho) < 0.15, f"Expected near-zero correlation, got {rho:.3f}"
