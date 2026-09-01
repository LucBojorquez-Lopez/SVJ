"""
tests/test_observables.py
=========================
Unit tests for src/observables.py.

Covers:
  - OBSERVABLES / DISTRIBUTIONS registry completeness and internal consistency.
  - n_fitted_params(): correct parameter counts per observable.
  - param_offsets(): monotone, correct total, matches sum of n_fitted_params.
  - validate_scan_selection(): raises on no-pipeline observables and unknown names.
  - event_valid_mask(): correctly masks events that violate pipeline range
    requirements for boxcox, affine_flip, multi-observable, and edge cases.
  - fit_observable_col(): output approximates N(0,1); correct param count.
  - forward_observable_col(): consistent with fit on the same data.
  - inverse_observable_col(): roundtrip recovers original data to < 0.1% error;
    monotone order is preserved.
  - load_tsv(): parses header and numerical data correctly.

No PYTHIA binary required.
"""

import tempfile
import numpy as np
import pytest
import scipy.stats as st

from observables import (
    OBSERVABLES, TRANSFORMS, DISTRIBUTIONS, DEFAULT_SCAN,
    n_fitted_params, param_offsets,
    fit_observable_col, forward_observable_col, inverse_observable_col,
    event_valid_mask, validate_scan_selection, load_tsv,
)


# ── Seeded RNGs (deterministic) ────────────────────────────────────────────────

def _rng(seed=0):
    return np.random.default_rng(seed)


# ── OBSERVABLES registry ───────────────────────────────────────────────────────

class TestObservablesRegistry:
    _REQUIRED = {'col', 'pipeline', 'distribution', 'default_include', 'label', 'desc'}

    def test_all_required_fields_present(self):
        for name, spec in OBSERVABLES.items():
            missing = self._REQUIRED - spec.keys()
            assert not missing, f"'{name}' missing fields: {missing}"

    def test_pipeline_transforms_exist(self):
        for name, spec in OBSERVABLES.items():
            if spec['pipeline'] is None:
                continue
            for t_name, _ in spec['pipeline']:
                assert t_name in TRANSFORMS, (
                    f"'{name}' pipeline references unknown transform '{t_name}'")

    def test_distributions_exist(self):
        for name, spec in OBSERVABLES.items():
            if spec['distribution'] is None:
                continue
            assert spec['distribution'] in DISTRIBUTIONS, (
                f"'{name}' references unknown distribution '{spec['distribution']}'")

    def test_default_include_is_bool(self):
        for name, spec in OBSERVABLES.items():
            assert isinstance(spec['default_include'], bool), (
                f"'{name}' default_include is not bool")

    def test_default_scan_matches_flag(self):
        # DEFAULT_SCAN = non-derived observables where default_include is True.
        flagged = {n for n, s in OBSERVABLES.items()
                   if s['default_include'] and s['col'] is not None}
        assert set(DEFAULT_SCAN) == flagged

    def test_default_scan_no_derived_observables(self):
        # Derived observables (col=None) must never appear in DEFAULT_SCAN,
        # even if their default_include is True (it controls GUI inclusion only).
        for name in DEFAULT_SCAN:
            assert OBSERVABLES[name]['col'] is not None, (
                f"Derived observable '{name}' (col=None) must not appear in DEFAULT_SCAN")

    def test_default_scan_all_regressionable(self):
        for name in DEFAULT_SCAN:
            spec = OBSERVABLES[name]
            assert spec['pipeline'] is not None, (
                f"DEFAULT_SCAN '{name}' has pipeline=None")
            assert spec['distribution'] is not None, (
                f"DEFAULT_SCAN '{name}' has distribution=None")

    def test_derived_obs_have_no_pipeline(self):
        for name, spec in OBSERVABLES.items():
            if spec['col'] is None:
                assert spec['pipeline'] is None, (
                    f"Derived obs '{name}' should have pipeline=None")

    def test_all_distributions_have_required_fields(self):
        for dname, dspec in DISTRIBUTIONS.items():
            for field in ('dist', 'n_params', 'fit_init', 'input_range', 'desc'):
                assert field in dspec, f"Distribution '{dname}' missing '{field}'"


# ── n_fitted_params ────────────────────────────────────────────────────────────

class TestNFittedParams:

    def test_leadVisPt(self):
        # boxcox (1 fitted) + gennorm (3 params) = 4
        assert n_fitted_params('leadVisPt') == 4

    def test_MET(self):
        assert n_fitted_params('MET') == 4

    def test_jetThrust(self):
        # affine_flip (0) + boxcox (1) + gennorm (3) = 4
        assert n_fitted_params('jetThrust') == 4

    def test_dPhiMETclose(self):
        # abs_value (0) + boxcox (1) + gennorm (3) = 4
        assert n_fitted_params('dPhiMETclose') == 4

    def test_dPhiMETfar(self):
        # abs_value (0) + affine_flip (0) + boxcox (1) + gennorm (3) = 4
        assert n_fitted_params('dPhiMETfar') == 4

    def test_all_regressionable_positive(self):
        for name, spec in OBSERVABLES.items():
            if spec['pipeline'] is not None and spec['distribution'] is not None:
                assert n_fitted_params(name) > 0, (
                    f"'{name}' has n_fitted_params == 0")

    def test_matches_manual_sum(self):
        for name, spec in OBSERVABLES.items():
            if spec['pipeline'] is None or spec['distribution'] is None:
                continue
            manual = sum(TRANSFORMS[t]['n_fitted'] for t, _ in spec['pipeline'])
            manual += DISTRIBUTIONS[spec['distribution']]['n_params']
            assert n_fitted_params(name) == manual, (
                f"'{name}': n_fitted_params mismatch (expected {manual})")


# ── param_offsets ──────────────────────────────────────────────────────────────

class TestParamOffsets:

    def test_single_obs_first_offset_is_zero(self):
        offsets = param_offsets(['leadVisPt'])
        assert offsets[0] == 0

    def test_single_obs_last_offset_equals_n_fitted(self):
        offsets = param_offsets(['leadVisPt'])
        assert offsets[-1] == n_fitted_params('leadVisPt')

    def test_two_obs_offsets(self):
        names = ['leadVisPt', 'MET']
        offsets = param_offsets(names)
        assert offsets[0] == 0
        assert offsets[1] == n_fitted_params('leadVisPt')
        assert offsets[2] == n_fitted_params('leadVisPt') + n_fitted_params('MET')

    def test_length_is_n_obs_plus_one(self):
        names = DEFAULT_SCAN
        offsets = param_offsets(names)
        assert len(offsets) == len(names) + 1

    def test_strictly_increasing(self):
        offsets = param_offsets(DEFAULT_SCAN)
        assert np.all(np.diff(offsets) > 0), "param_offsets must be strictly increasing"

    def test_total_equals_sum_of_n_fitted(self):
        offsets = param_offsets(DEFAULT_SCAN)
        expected = sum(n_fitted_params(n) for n in DEFAULT_SCAN)
        assert offsets[-1] == expected


# ── validate_scan_selection ────────────────────────────────────────────────────

class TestValidateScanSelection:

    def test_default_scan_passes(self):
        validate_scan_selection(DEFAULT_SCAN)

    def test_single_regressionable_passes(self):
        validate_scan_selection(['leadVisPt'])
        validate_scan_selection(['MET', 'tau1', 'hemiMass1'])

    def test_none_pipeline_raises(self):
        with pytest.raises(ValueError, match='pipeline'):
            validate_scan_selection(['closeJetIsLead'])

    def test_none_distribution_derived_raises(self):
        with pytest.raises(ValueError):
            validate_scan_selection(['mass2/mass1'])

    def test_none_pipeline_nInvClose_raises(self):
        with pytest.raises(ValueError):
            validate_scan_selection(['nInvClose'])

    def test_unknown_observable_raises_key_error(self):
        with pytest.raises(KeyError):
            validate_scan_selection(['not_an_observable'])

    def test_error_message_names_bad_observables(self):
        with pytest.raises(ValueError, match='closeJetIsLead'):
            validate_scan_selection(['MET', 'closeJetIsLead'])


# ── event_valid_mask ───────────────────────────────────────────────────────────

class TestEventValidMask:

    def _col_map(self, *obs_names):
        return {OBSERVABLES[n]['col']: i for i, n in enumerate(obs_names)}

    def test_all_positive_pass_boxcox(self):
        x = _rng(0).lognormal(size=(100, 1))
        col_map = self._col_map('leadVisPt')
        mask, n_disc = event_valid_mask(x, ['leadVisPt'], col_map)
        assert mask.all()
        assert n_disc[0] == 0

    def test_zero_fails_boxcox(self):
        x = np.ones((20, 1)) * 2.0
        x[5, 0] = 0.0
        col_map = self._col_map('MET')
        mask, _ = event_valid_mask(x, ['MET'], col_map)
        assert not mask[5]
        assert mask.sum() == 19

    def test_negative_fails_boxcox(self):
        x = np.ones((10, 1)) * 3.0
        x[3, 0] = -1.0
        col_map = self._col_map('leadVisPt')
        mask, _ = event_valid_mask(x, ['leadVisPt'], col_map)
        assert not mask[3]
        assert mask.sum() == 9

    def test_jetThrust_at_one_fails(self):
        # affine_flip requires x < 1.0 (strict)
        x = np.array([[0.5], [1.0], [0.3]])
        col_map = self._col_map('jetThrust')
        mask, _ = event_valid_mask(x, ['jetThrust'], col_map)
        assert mask[0]
        assert not mask[1]
        assert mask[2]

    def test_jetThrust_above_one_fails(self):
        x = np.array([[0.9], [1.1]])
        col_map = self._col_map('jetThrust')
        mask, _ = event_valid_mask(x, ['jetThrust'], col_map)
        assert mask[0]
        assert not mask[1]

    def test_jetThrust_at_zero_passes(self):
        # thrust=0 → affine_flip → 1.0 > 0 → boxcox ok
        x = np.array([[0.0]])
        col_map = self._col_map('jetThrust')
        mask, _ = event_valid_mask(x, ['jetThrust'], col_map)
        assert mask[0]

    def test_n_discarded_counts_correctly(self):
        x = np.ones((10, 1)) * 2.0
        x[2, 0] = 0.0
        x[7, 0] = -0.5
        col_map = self._col_map('leadVisPt')
        mask, n_disc = event_valid_mask(x, ['leadVisPt'], col_map)
        assert mask.sum() == 8
        assert n_disc[0] == 2

    def test_multi_observable_independent_masking(self):
        # Column 0: leadVisPt (needs > 0)
        # Column 1: MET (needs > 0)
        n = 30
        x = np.ones((n, 2)) * 5.0
        x[3, 0] = 0.0   # event 3 fails leadVisPt
        x[15, 1] = -1.0  # event 15 fails MET
        col_map = {OBSERVABLES['leadVisPt']['col']: 0,
                   OBSERVABLES['MET']['col']: 1}
        mask, _ = event_valid_mask(x, ['leadVisPt', 'MET'], col_map)
        assert not mask[3]
        assert not mask[15]
        assert mask.sum() == 28

    def test_mask_is_boolean_array(self):
        x = _rng(1).lognormal(size=(50, 1))
        col_map = self._col_map('leadVisPt')
        mask, _ = event_valid_mask(x, ['leadVisPt'], col_map)
        assert mask.dtype == bool
        assert mask.shape == (50,)

    def test_derived_obs_skipped_silently(self):
        # 'mass2/mass1' has col=None → mask() should skip it without error
        x = _rng(2).lognormal(size=(10, 1))
        col_map = self._col_map('leadVisPt')
        # Add derived obs with col=None to selection
        mask, _ = event_valid_mask(x, ['leadVisPt', 'mass2/mass1'], col_map)
        assert mask.shape == (10,)


# ── fit_observable_col ─────────────────────────────────────────────────────────

class TestFitObservableCol:

    def test_output_approx_standard_normal_mean(self):
        x = _rng(10).lognormal(mean=5.0, sigma=0.5, size=500)
        y_std, _ = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert abs(y_std.mean()) < 0.2

    def test_output_approx_standard_normal_std(self):
        x = _rng(11).lognormal(mean=5.0, sigma=0.5, size=500)
        y_std, _ = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert abs(y_std.std() - 1.0) < 0.3

    def test_output_all_finite(self):
        x = _rng(12).lognormal(size=400)
        y_std, _ = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert np.all(np.isfinite(y_std))

    def test_output_in_typical_normal_range(self):
        x = _rng(13).lognormal(size=400)
        y_std, _ = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert np.percentile(np.abs(y_std), 99) < 5.0

    def test_correct_param_count_boxcox_gennorm(self):
        x = _rng(14).lognormal(size=300)
        _, params = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        # boxcox: 1 fitted; gennorm: 3 params → 4
        assert len(params) == 4

    def test_all_params_finite(self):
        x = _rng(15).lognormal(size=300)
        _, params = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert all(np.isfinite(p) for p in params)

    def test_affine_flip_boxcox_pipeline(self):
        # jetThrust: values in (0, 1)
        x = _rng(16).uniform(0.01, 0.99, size=300)
        y_std, params = fit_observable_col(
            x, [('affine_flip', {'a': 1.0}), ('boxcox', {})], 'gennorm')
        # 0 + 1 + 3 = 4 params
        assert len(params) == 4
        assert np.all(np.isfinite(y_std))

    def test_abs_affine_boxcox_pipeline(self):
        # dPhiMETfar: positive angles in (0, π)
        x = _rng(17).uniform(0.01, np.pi - 0.01, size=300)
        y_std, params = fit_observable_col(
            x, [('abs_value', {}), ('affine_flip', {'a': np.pi}), ('boxcox', {})], 'gennorm')
        # 0 + 0 + 1 + 3 = 4 params
        assert len(params) == 4
        assert np.all(np.isfinite(y_std))

    def test_output_shape_matches_input(self):
        x = _rng(18).lognormal(size=250)
        y_std, _ = fit_observable_col(x, [('boxcox', {})], 'gennorm')
        assert y_std.shape == x.shape


# ── forward / inverse roundtrip ────────────────────────────────────────────────

class TestForwardInverseRoundtrip:
    """
    The forward pipeline maps x → standard normal; the inverse maps
    uniform quantiles back to original space.  Since:
      fit_observable_col(x) → (y_std, params)
      u  = norm.cdf(y_std)   (≈ the uniform quantile of each x[i])
      inverse(u, params) → x_back
    the roundtrip should recover x to within numerical precision.
    """

    def _roundtrip(self, x, pipeline, dist_name):
        y_std, params = fit_observable_col(x, pipeline, dist_name)
        u = np.clip(st.norm.cdf(y_std), 1e-10, 1.0 - 1e-10)
        return inverse_observable_col(u, pipeline, dist_name, params), params

    def test_boxcox_gennorm(self):
        x = _rng(20).lognormal(mean=5.0, sigma=0.5, size=300)
        x_back, _ = self._roundtrip(x, [('boxcox', {})], 'gennorm')
        np.testing.assert_allclose(x_back, x, rtol=1e-3)

    def test_affine_flip_boxcox_gennorm(self):
        # Use lognormal-derived data so that 1-x is lognormally distributed;
        # boxcox maps lognormal → Gaussian, giving gennorm a well-conditioned fit.
        # Uniform data causes gennorm to converge to a degenerate large-beta solution.
        rng = np.random.default_rng(21)
        z = np.clip(rng.lognormal(mean=-2.0, sigma=0.5, size=400), 0.01, 0.98)
        x = 1.0 - z[:300]
        x_back, _ = self._roundtrip(
            x, [('affine_flip', {'a': 1.0}), ('boxcox', {})], 'gennorm')
        np.testing.assert_allclose(x_back, x, rtol=1e-3)

    def test_abs_affine_boxcox_gennorm(self):
        # This is the dPhiMETfar pipeline.  abs_value's inverse reconstructs a
        # signed sample by alternating signs, so the roundtrip recovers the
        # MAGNITUDE of each input, not the input itself.
        x = _rng(22).uniform(0.01, np.pi - 0.01, size=300)
        x_back, _ = self._roundtrip(
            x, [('abs_value', {}), ('affine_flip', {'a': np.pi}), ('boxcox', {})], 'gennorm')
        np.testing.assert_allclose(np.abs(x_back), x, rtol=1e-3)

    def test_forward_consistent_with_fit(self):
        """forward_observable_col with fitted params must equal fit output."""
        x = _rng(23).lognormal(size=200)
        pipeline = [('boxcox', {})]
        y_fit, params = fit_observable_col(x, pipeline, 'gennorm')
        y_fwd = forward_observable_col(x, pipeline, 'gennorm', params)
        np.testing.assert_allclose(y_fwd, y_fit, rtol=1e-10)

    def test_rank_order_preserved(self):
        """The forward pipeline is monotone → ranks of x and x_back must agree."""
        x = _rng(24).lognormal(size=200)
        x_back, _ = self._roundtrip(x, [('boxcox', {})], 'gennorm')
        np.testing.assert_array_equal(np.argsort(x), np.argsort(x_back))

    def test_all_outputs_finite(self):
        x = _rng(25).lognormal(size=200)
        x_back, _ = self._roundtrip(x, [('boxcox', {})], 'gennorm')
        assert np.all(np.isfinite(x_back))

    def test_all_back_positive_for_positive_pipeline(self):
        """leadVisPt is strictly positive; samples from inverse should be too."""
        x = _rng(26).lognormal(mean=6.0, sigma=0.4, size=200)
        x_back, _ = self._roundtrip(x, [('boxcox', {})], 'gennorm')
        assert np.all(x_back > 0)

    def test_different_distributions_all_roundtrip(self):
        """Test roundtrip for every distribution in the DISTRIBUTIONS registry."""
        for dist_name, dspec in DISTRIBUTIONS.items():
            lo, hi = dspec['input_range']
            lo = max(lo, 1e-3)
            if hi == np.inf:
                x = _rng(27).lognormal(size=200) + lo
            else:
                x = _rng(27).uniform(lo + 0.01, hi - 0.01, size=200)
            try:
                x_back, _ = self._roundtrip(x, [], dist_name)
                assert np.all(np.isfinite(x_back)), (
                    f"Non-finite output for distribution '{dist_name}'")
            except Exception as e:
                pytest.fail(f"Roundtrip failed for distribution '{dist_name}': {e}")


# ── load_tsv ───────────────────────────────────────────────────────────────────

class TestLoadTsv:

    def _write_tsv(self, content):
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.tsv', delete=False)
        f.write(content)
        f.close()
        return f.name

    def test_basic_shape(self):
        fname = self._write_tsv("# a\tb\tc\n1.0\t2.0\t3.0\n4.0\t5.0\t6.0\n")
        data, col_map = load_tsv(fname)
        assert data.shape == (2, 3)

    def test_col_map_correct(self):
        fname = self._write_tsv("# alpha\tbeta\tgamma\n0.0\t0.1\t0.2\n")
        _, col_map = load_tsv(fname)
        assert col_map == {'alpha': 0, 'beta': 1, 'gamma': 2}

    def test_values_correct(self):
        fname = self._write_tsv("# x\ty\n1.5\t2.5\n3.5\t4.5\n")
        data, col_map = load_tsv(fname)
        np.testing.assert_allclose(data[0], [1.5, 2.5])
        np.testing.assert_allclose(data[1], [3.5, 4.5])

    def test_indexing_via_col_map(self):
        fname = self._write_tsv("# col0\tcol1\n10.0\t20.0\n30.0\t40.0\n")
        data, col_map = load_tsv(fname)
        assert data[0, col_map['col0']] == pytest.approx(10.0)
        assert data[0, col_map['col1']] == pytest.approx(20.0)

    def test_single_row(self):
        fname = self._write_tsv("# x\n42.0\n")
        data, col_map = load_tsv(fname)
        # np.loadtxt returns a 0-D scalar for a 1×1 file; float() unwraps it
        assert float(data) == pytest.approx(42.0)
