"""
tests/test_val_utils.py
=======================
Unit tests for the pure-Python functions in src/run_regression/_val_utils.py.

Covers:
  - resolve_point(): simple arithmetic, chain-dependent derived expressions,
    circular/undefined dependency raises ValueError.
  - sample_interior_points(): correct count, fractional indices within bounds,
    physical values within axis ranges, raises for degenerate (single-point) axes.
  - js_per_obs(): near-zero for identical distributions, large for well-separated
    distributions, handles all-NaN columns gracefully.
  - mmd_rbf(): non-negative, near-zero for same-distribution samples, clearly
    larger for different-distribution samples, handles non-finite rows.
  - extract_obs_cols(): correct column selection and ordering.

No PYTHIA binary required.
"""

import numpy as np
import pytest

from _val_utils import (
    resolve_point,
    sample_interior_points,
    js_per_obs,
    mmd_rbf,
    extract_obs_cols,
)


# ── resolve_point ──────────────────────────────────────────────────────────────

class TestResolvePoint:

    def test_scan_point_present(self):
        result = resolve_point({'mZ': 1000.0}, {}, {})
        assert result['mZ'] == pytest.approx(1000.0)

    def test_fixed_params_present(self):
        result = resolve_point({}, {'alphaD': 0.4}, {})
        assert result['alphaD'] == pytest.approx(0.4)

    def test_scan_overrides_fixed(self):
        # scan_point values win over fixed_params when both have the same key
        result = resolve_point({'mRho': 25.0}, {'mRho': 20.0}, {})
        assert result['mRho'] == pytest.approx(25.0)

    def test_simple_derived_expression(self):
        result = resolve_point({'mRho': 20.0}, {}, {'mPi': 'mRho * 2'})
        assert result['mPi'] == pytest.approx(40.0)

    def test_chain_derived_expressions(self):
        # mPi depends on mRho; mDark depends on mPi
        result = resolve_point(
            {'mRho': 10.0}, {},
            {'mPi': 'mRho * 2', 'mDark': 'mPi + mRho'})
        assert result['mPi'] == pytest.approx(20.0)
        assert result['mDark'] == pytest.approx(30.0)

    def test_derived_uses_fixed_param(self):
        result = resolve_point({}, {'mRho': 15.0}, {'mPi': 'mRho / 3'})
        assert result['mPi'] == pytest.approx(5.0)

    def test_arithmetic_operators(self):
        result = resolve_point({'a': 6.0, 'b': 2.0}, {}, {
            'sum_': 'a + b',
            'diff': 'a - b',
            'prod': 'a * b',
            'quot': 'a / b',
        })
        assert result['sum_'] == pytest.approx(8.0)
        assert result['diff'] == pytest.approx(4.0)
        assert result['prod'] == pytest.approx(12.0)
        assert result['quot'] == pytest.approx(3.0)

    def test_circular_dependency_raises(self):
        with pytest.raises(ValueError, match='circular'):
            resolve_point({}, {}, {'a': 'b + 1', 'b': 'a + 1'})

    def test_undefined_variable_raises(self):
        with pytest.raises(ValueError):
            resolve_point({}, {}, {'x': 'undefined_var * 2'})

    def test_output_includes_all_inputs(self):
        result = resolve_point({'mZ': 500.0}, {'mRho': 20.0}, {'mPi': 'mRho * 2'})
        assert 'mZ' in result
        assert 'mRho' in result
        assert 'mPi' in result

    def test_empty_inputs(self):
        result = resolve_point({}, {}, {})
        assert result == {}


# ── sample_interior_points ─────────────────────────────────────────────────────

def _make_scan(axis_names, axis_ranges, n_points):
    return {
        'axis_names': axis_names,
        'axis_vals':  [np.linspace(lo, hi, n) for lo, hi, n in axis_ranges],
    }


class TestSampleInteriorPoints:

    def test_returns_correct_count(self):
        scan = _make_scan(['mZ'], [(500.0, 2000.0, 5)], 5)
        _, pts = sample_interior_points(scan, 20, np.random.default_rng(0))
        assert len(pts) == 20

    def test_frac_shape(self):
        scan = _make_scan(['mZ', 'rinv'], [(500.0, 2000.0, 4), (0.1, 0.7, 4)], 4)
        frac, _ = sample_interior_points(scan, 15, np.random.default_rng(1))
        assert frac.shape == (15, 2)

    def test_physical_values_within_axis_bounds(self):
        scan = _make_scan(['mZ', 'rinv'], [(500.0, 2000.0, 5), (0.1, 0.7, 5)], 5)
        _, pts = sample_interior_points(scan, 50, np.random.default_rng(2))
        for pt in pts:
            assert 500.0 < pt['mZ'] < 2000.0, f"mZ={pt['mZ']} out of bounds"
            assert 0.1 < pt['rinv'] < 0.7,    f"rinv={pt['rinv']} out of bounds"

    def test_fractional_indices_strictly_interior(self):
        n_pts = 4
        scan = _make_scan(['mZ'], [(500.0, 2000.0, n_pts)], n_pts)
        frac, _ = sample_interior_points(scan, 30, np.random.default_rng(3))
        # Fractional indices must lie in (0.5, n_pts - 1.5) = (0.5, 2.5)
        assert np.all(frac[:, 0] >= 0.5)
        assert np.all(frac[:, 0] <= n_pts - 1.5)

    def test_four_d_scan(self):
        scan = _make_scan(
            ['mZ', 'mRho', 'rinv', 'alphaD'],
            [(500.0, 2000.0, 4), (10.0, 30.0, 4), (0.05, 0.7, 4), (0.1, 0.8, 4)],
            4,
        )
        frac, pts = sample_interior_points(scan, 10, np.random.default_rng(4))
        assert len(pts) == 10
        assert frac.shape == (10, 4)

    def test_minimum_two_grid_points(self):
        # Axis with 2 points is degenerate (midpoint 0.5): should still work
        scan = _make_scan(['mZ'], [(500.0, 2000.0, 2)], 2)
        _, pts = sample_interior_points(scan, 5, np.random.default_rng(5))
        for pt in pts:
            assert pt['mZ'] == pytest.approx(1250.0)  # midpoint

    def test_single_point_axis_raises(self):
        scan = _make_scan(['mZ'], [(500.0, 500.0, 1)], 1)
        with pytest.raises(ValueError):
            sample_interior_points(scan, 5, np.random.default_rng(6))

    def test_reproducible_with_same_seed(self):
        scan = _make_scan(['mZ', 'rinv'], [(500.0, 2000.0, 5), (0.1, 0.7, 5)], 5)
        _, pts1 = sample_interior_points(scan, 10, np.random.default_rng(99))
        _, pts2 = sample_interior_points(scan, 10, np.random.default_rng(99))
        for p1, p2 in zip(pts1, pts2):
            assert p1 == p2


# ── js_per_obs ─────────────────────────────────────────────────────────────────

class TestJsPerObs:

    def test_identical_arrays_near_zero(self):
        rng = np.random.default_rng(10)
        A = rng.normal(0, 1, size=(5000, 2))
        js = js_per_obs(A, A, ['obs1', 'obs2'])
        assert np.all(js < 0.05), f"JS(A,A) should be near 0, got {js}"

    def test_well_separated_distributions_large(self):
        rng = np.random.default_rng(11)
        A = rng.normal(0,  1, size=(5000, 2))
        B = rng.normal(10, 1, size=(5000, 2))
        js = js_per_obs(A, B, ['obs1', 'obs2'])
        assert np.all(js > 0.5), f"JS for well-separated should be large, got {js}"

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(12)
        A = rng.normal(0, 1, size=(2000, 3))
        B = rng.normal(2, 1, size=(2000, 3))
        js = js_per_obs(A, B, ['a', 'b', 'c'])
        assert np.all(js >= 0.0)
        assert np.all(js <= 1.0)

    def test_output_length_matches_n_obs(self):
        rng = np.random.default_rng(13)
        A = rng.normal(size=(1000, 4))
        B = rng.normal(size=(1000, 4))
        js = js_per_obs(A, B, ['a', 'b', 'c', 'd'])
        assert js.shape == (4,)

    def test_all_nan_column_gives_nan(self):
        A = np.full((100, 1), np.nan)
        B = np.ones((100, 1))
        js = js_per_obs(A, B, ['obs'])
        assert np.isnan(js[0])

    def test_partially_nan_column_handled(self):
        rng = np.random.default_rng(14)
        A = rng.normal(size=(200, 1))
        A[::5, 0] = np.nan      # 20% NaN
        B = rng.normal(size=(200, 1))
        js = js_per_obs(A, B, ['obs'])
        assert np.isfinite(js[0])

    def test_monotone_with_increasing_separation(self):
        """JS should increase as the two distributions are moved further apart."""
        rng = np.random.default_rng(15)
        base = rng.normal(0, 1, size=(3000, 1))
        prev_js = 0.0
        for shift in (1.0, 3.0, 6.0):
            shifted = rng.normal(shift, 1, size=(3000, 1))
            js = float(js_per_obs(base, shifted, ['obs'])[0])
            assert js > prev_js, (
                f"JS should increase with separation; shift={shift}, js={js:.4f}")
            prev_js = js


# ── mmd_rbf ────────────────────────────────────────────────────────────────────

class TestMmdRbf:

    def test_non_negative(self):
        rng = np.random.default_rng(20)
        A = rng.normal(size=(300, 2))
        B = rng.normal(size=(300, 2))
        val = mmd_rbf(A, B, n_sub=150, rng=rng)
        assert val >= 0.0

    def test_same_data_exactly_zero(self):
        # When n_sub == len(A), both X and Y subsamples are full permutations
        # of the same rows → K(X,X) = K(Y,Y) = K(X,Y) → MMD = 0 exactly.
        rng = np.random.default_rng(21)
        A = rng.normal(size=(200, 2))
        val = mmd_rbf(A, A, n_sub=200, rng=np.random.default_rng(21))
        assert val == pytest.approx(0.0, abs=1e-10)

    def test_different_distributions_larger_than_same(self):
        rng = np.random.default_rng(22)
        A = rng.normal(0, 1, size=(500, 2))
        B = rng.normal(5, 1, size=(500, 2))
        mmd_same = mmd_rbf(A, A, n_sub=200, rng=np.random.default_rng(22))
        mmd_diff = mmd_rbf(A, B, n_sub=200, rng=np.random.default_rng(23))
        assert mmd_diff > mmd_same

    def test_clearly_different_gives_positive(self):
        rng = np.random.default_rng(24)
        A = rng.normal(0, 1, size=(500, 2))
        B = rng.normal(8, 1, size=(500, 2))
        val = mmd_rbf(A, B, n_sub=200, rng=rng)
        assert val > 0.01

    def test_infinite_rows_filtered(self):
        rng = np.random.default_rng(25)
        A = rng.normal(size=(200, 2))
        A[0, 0] = np.inf
        B = rng.normal(size=(200, 2))
        val = mmd_rbf(A, B, n_sub=100, rng=rng)
        assert np.isfinite(val)

    def test_all_infinite_returns_nan(self):
        A = np.full((100, 2), np.inf)
        B = np.ones((100, 2))
        val = mmd_rbf(A, B, n_sub=50, rng=np.random.default_rng(26))
        assert np.isnan(val)

    def test_scalar_output(self):
        rng = np.random.default_rng(27)
        A = rng.normal(size=(200, 3))
        B = rng.normal(size=(200, 3))
        val = mmd_rbf(A, B, n_sub=100, rng=rng)
        assert np.ndim(val) == 0


# ── extract_obs_cols ───────────────────────────────────────────────────────────

class TestExtractObsCols:

    def _make_col_map(self, names):
        return {n: i for i, n in enumerate(names)}

    def test_extracts_correct_columns(self):
        # Fake TSV data with 5 columns; extract the 'MET' and 'leadVisPt' columns
        from observables import OBSERVABLES
        col_names = ['leadVisPt', 'MET', 'tau1', 'tau2', 'hemiMass1']
        col_map = self._make_col_map(
            [OBSERVABLES[n]['col'] for n in col_names])
        data = np.arange(10.0).reshape(2, 5)
        result = extract_obs_cols(data, col_map, ['leadVisPt', 'MET'])
        assert result.shape == (2, 2)
        # 'leadVisPt' is column 0, 'MET' is column 1
        np.testing.assert_array_equal(result[:, 0], data[:, 0])  # leadVisPt
        np.testing.assert_array_equal(result[:, 1], data[:, 1])  # MET

    def test_order_follows_obs_selection(self):
        from observables import OBSERVABLES
        col_names = ['leadVisPt', 'MET', 'tau1']
        col_map = self._make_col_map(
            [OBSERVABLES[n]['col'] for n in col_names])
        data = np.array([[10.0, 20.0, 30.0]])
        # Request in reversed order
        result = extract_obs_cols(data, col_map, ['tau1', 'MET', 'leadVisPt'])
        np.testing.assert_array_equal(result[0], [30.0, 20.0, 10.0])

    def test_output_dtype_is_float(self):
        from observables import OBSERVABLES
        col_names = ['leadVisPt']
        col_map = self._make_col_map([OBSERVABLES['leadVisPt']['col']])
        data = np.array([[1, 2, 3]])  # integer array
        result = extract_obs_cols(data, col_map, ['leadVisPt'])
        assert result.dtype == float

    def test_single_observable(self):
        from observables import OBSERVABLES
        col_map = {OBSERVABLES['MET']['col']: 0}
        data = np.array([[42.0], [99.0]])
        result = extract_obs_cols(data, col_map, ['MET'])
        assert result.shape == (2, 1)
        np.testing.assert_array_equal(result[:, 0], [42.0, 99.0])
