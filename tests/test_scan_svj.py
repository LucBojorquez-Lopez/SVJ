"""
tests/test_scan_svj.py
======================
Unit tests for the pure-Python functions in src/run_regression/scan_svj.py.

Covers:
  - read_scan_cfg(): correctly parses [scan] (linear and log spacing),
    [fixed], [derived], and [simulation] sections from an INI config file.
  - resolve_point(): simple and chain-derived expressions; circular dependency
    raises ValueError.  (This is the scan_svj.py copy; identical behaviour
    to the _val_utils.py copy is verified independently here.)
  - _mvt_loglik(): returns a finite scalar; is lower for data that fits poorly.
  - fit_mvt_em(): R_upper has the correct length K*(K-1)//2; nu is positive
    and finite; uncorrelated data yields near-zero off-diagonal entries;
    correlated data recovers the embedded correlation.

No PYTHIA binary is required.
"""

import tempfile
import textwrap
import numpy as np
import pytest

from scan_svj import read_scan_cfg, resolve_point, _mvt_loglik, fit_mvt_em


# ── read_scan_cfg ──────────────────────────────────────────────────────────────

def _write_cfg(content):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.cfg', delete=False)
    f.write(textwrap.dedent(content))
    f.close()
    return f.name


class TestReadScanCfg:

    def test_linear_scan_axis(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 4
        """)
        cfg = read_scan_cfg(fname)
        vals = cfg.scan_axes['mZ']
        assert len(vals) == 4
        assert vals[0] == pytest.approx(500.0)
        assert vals[-1] == pytest.approx(2000.0)
        # Linear spacing
        np.testing.assert_allclose(vals, np.linspace(500, 2000, 4))

    def test_log_scan_axis(self):
        fname = _write_cfg("""
            [scan]
            mZ = 100, 1000, 3, log
        """)
        cfg = read_scan_cfg(fname)
        vals = cfg.scan_axes['mZ']
        assert len(vals) == 3
        assert vals[0] == pytest.approx(100.0)
        assert vals[-1] == pytest.approx(1000.0)
        np.testing.assert_allclose(vals, np.logspace(2, 3, 3))

    def test_multiple_scan_axes(self):
        fname = _write_cfg("""
            [scan]
            mZ   = 500, 2000, 3
            rinv = 0.1, 0.7, 4
        """)
        cfg = read_scan_cfg(fname)
        assert 'mZ' in cfg.scan_axes
        assert 'rinv' in cfg.scan_axes
        assert len(cfg.scan_axes['mZ']) == 3
        assert len(cfg.scan_axes['rinv']) == 4

    def test_fixed_params(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 2
            [fixed]
            alphaD = 0.4
            mRho   = 20.0
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.fixed_params['alphaD'] == pytest.approx(0.4)
        assert cfg.fixed_params['mRho'] == pytest.approx(20.0)

    def test_derived_exprs(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 2
            [derived]
            mPi = mRho * 2
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.derived_exprs['mPi'] == 'mRho * 2'

    def test_simulation_params_int(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 2
            [simulation]
            nEvent   = 5000
            nWorkers = 8
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.sim_params['nEvent'] == 5000
        assert cfg.sim_params['nWorkers'] == 8

    def test_simulation_params_float(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 2
            [simulation]
            some_float = 3.14
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.sim_params['some_float'] == pytest.approx(3.14)

    def test_empty_scan_gives_empty_axes(self):
        fname = _write_cfg("""
            [fixed]
            alphaD = 0.4
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.scan_axes == {}

    def test_axis_preserves_case(self):
        fname = _write_cfg("""
            [scan]
            mRho = 10, 30, 3
        """)
        cfg = read_scan_cfg(fname)
        assert 'mRho' in cfg.scan_axes  # not 'mrho'

    def test_missing_section_gives_empty_dict(self):
        fname = _write_cfg("""
            [scan]
            mZ = 500, 2000, 2
        """)
        cfg = read_scan_cfg(fname)
        assert cfg.fixed_params == {}
        assert cfg.derived_exprs == {}
        assert cfg.sim_params == {}


# ── resolve_point (scan_svj.py copy) ──────────────────────────────────────────

class TestResolvePoint:

    def test_scan_point_in_result(self):
        result = resolve_point({'mZ': 1500.0}, {}, {})
        assert result['mZ'] == pytest.approx(1500.0)

    def test_fixed_in_result(self):
        result = resolve_point({}, {'alphaD': 0.4}, {})
        assert result['alphaD'] == pytest.approx(0.4)

    def test_scan_overrides_fixed(self):
        result = resolve_point({'mRho': 25.0}, {'mRho': 20.0}, {})
        assert result['mRho'] == pytest.approx(25.0)

    def test_derived_computed(self):
        result = resolve_point({'mRho': 20.0}, {}, {'mPi': 'mRho * 2'})
        assert result['mPi'] == pytest.approx(40.0)

    def test_chain_derived(self):
        result = resolve_point(
            {'mRho': 10.0}, {},
            {'mPi': 'mRho * 2', 'mDark': 'mPi + mRho'})
        assert result['mPi'] == pytest.approx(20.0)
        assert result['mDark'] == pytest.approx(30.0)

    def test_circular_raises(self):
        with pytest.raises(ValueError):
            resolve_point({}, {}, {'a': 'b + 1', 'b': 'a + 1'})

    def test_undefined_raises(self):
        with pytest.raises(ValueError):
            resolve_point({}, {}, {'x': 'ghost_variable * 2'})

    def test_parentheses_in_expression(self):
        result = resolve_point({'a': 3.0, 'b': 4.0}, {}, {'c': '(a + b) * 2'})
        assert result['c'] == pytest.approx(14.0)


# ── _mvt_loglik ────────────────────────────────────────────────────────────────

class TestMvtLoglik:

    def test_finite_for_standard_normal_data(self):
        rng = np.random.default_rng(40)
        K, N = 3, 200
        X = rng.standard_normal((N, K))
        R_inv = np.eye(K)
        ll = _mvt_loglik(X, R_inv, 0.0, 5.0)
        assert np.isfinite(ll), f"log-likelihood is not finite: {ll}"

    def test_scalar_output(self):
        rng = np.random.default_rng(41)
        X = rng.standard_normal((100, 2))
        ll = _mvt_loglik(X, np.eye(2), 0.0, 5.0)
        assert np.ndim(ll) == 0

    def test_higher_nu_gives_different_value(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 3))
        R_inv = np.eye(3)
        ll_5  = _mvt_loglik(X, R_inv, 0.0, 5.0)
        ll_50 = _mvt_loglik(X, R_inv, 0.0, 50.0)
        assert ll_5 != pytest.approx(ll_50, rel=1e-3), (
            "_mvt_loglik should differ for nu=5 vs nu=50")

    def test_larger_log_det_reduces_likelihood(self):
        # Larger determinant of R → smaller log-likelihood for the same data
        rng = np.random.default_rng(43)
        X = rng.standard_normal((200, 2))
        R_I = np.eye(2)
        R_2 = 2.0 * np.eye(2)
        R_I_inv = np.eye(2)
        R_2_inv = 0.5 * np.eye(2)
        ll_I = _mvt_loglik(X, R_I_inv, np.log(np.linalg.det(R_I)), 5.0)
        ll_2 = _mvt_loglik(X, R_2_inv, np.log(np.linalg.det(R_2)), 5.0)
        assert ll_I > ll_2  # tighter fit to the data


# ── fit_mvt_em ─────────────────────────────────────────────────────────────────

def _sample_mvt(R, nu, N, rng):
    """Draw N samples from MVT(0, R, nu) using the normal-chi2 construction."""
    K = R.shape[0]
    L = np.linalg.cholesky(R)
    z = rng.standard_normal((N, K)) @ L.T
    v = rng.chisquare(nu, size=N)
    return z * np.sqrt(nu / v[:, None])


class TestFitMvtEm:

    def test_r_upper_length_k2(self):
        K = 2
        rng = np.random.default_rng(50)
        X = rng.standard_normal((500, K))
        R_upper, _ = fit_mvt_em(X, max_iter=50)
        assert len(R_upper) == K * (K - 1) // 2  # = 1

    def test_r_upper_length_k3(self):
        K = 3
        rng = np.random.default_rng(51)
        X = rng.standard_normal((500, K))
        R_upper, _ = fit_mvt_em(X, max_iter=50)
        assert len(R_upper) == K * (K - 1) // 2  # = 3

    def test_r_upper_length_k5(self):
        K = 5
        rng = np.random.default_rng(52)
        X = rng.standard_normal((500, K))
        R_upper, _ = fit_mvt_em(X, max_iter=50)
        assert len(R_upper) == K * (K - 1) // 2  # = 10

    def test_nu_positive(self):
        rng = np.random.default_rng(53)
        X = rng.standard_normal((500, 3))
        _, nu = fit_mvt_em(X, max_iter=50)
        assert nu > 0

    def test_nu_finite(self):
        rng = np.random.default_rng(54)
        X = rng.standard_normal((500, 3))
        _, nu = fit_mvt_em(X, max_iter=50)
        assert np.isfinite(nu)

    def test_uncorrelated_data_small_off_diagonal(self):
        # Draw uncorrelated t(5) data; fitted off-diagonals should be small
        K = 3
        rng = np.random.default_rng(55)
        X = _sample_mvt(np.eye(K), 5.0, 4000, rng)
        R_upper, _ = fit_mvt_em(X, max_iter=100)
        assert np.all(np.abs(R_upper) < 0.15), (
            f"Off-diagonal entries too large for uncorrelated data: {R_upper}")

    def test_correlated_data_recovers_correlation(self):
        K = 2
        rho_true = 0.7
        R_true = np.array([[1.0, rho_true], [rho_true, 1.0]])
        rng = np.random.default_rng(56)
        X = _sample_mvt(R_true, 5.0, 6000, rng)
        R_upper, _ = fit_mvt_em(X, max_iter=200)
        # R_upper[0] is the (0,1) correlation
        assert R_upper[0] > 0.55, (
            f"Expected correlation > 0.55, got {R_upper[0]:.3f}")
        assert R_upper[0] < 0.90, (
            f"Expected correlation < 0.90, got {R_upper[0]:.3f}")

    def test_negative_correlation_recovered(self):
        K = 2
        rho_true = -0.6
        R_true = np.array([[1.0, rho_true], [rho_true, 1.0]])
        rng = np.random.default_rng(57)
        X = _sample_mvt(R_true, 5.0, 6000, rng)
        R_upper, _ = fit_mvt_em(X, max_iter=200)
        assert R_upper[0] < -0.4, (
            f"Expected negative correlation, got {R_upper[0]:.3f}")

    def test_t_data_nu_finite_and_moderate(self):
        # Data from t(5) → nu should converge to a finite value (not 500)
        K = 3
        rng = np.random.default_rng(58)
        X = _sample_mvt(np.eye(K), 5.0, 5000, rng)
        _, nu = fit_mvt_em(X, max_iter=200)
        assert 2.0 < nu < 200.0, (
            f"nu={nu:.1f} out of expected range (2, 200) for t(5) data")

    def test_r_upper_values_in_neg1_pos1(self):
        K = 4
        rng = np.random.default_rng(59)
        X = rng.standard_normal((1000, K))
        R_upper, _ = fit_mvt_em(X, max_iter=50)
        assert np.all(R_upper >= -1.0)
        assert np.all(R_upper <= 1.0)

    def test_returns_tuple_of_two(self):
        rng = np.random.default_rng(60)
        X = rng.standard_normal((200, 2))
        result = fit_mvt_em(X, max_iter=10)
        assert len(result) == 2
