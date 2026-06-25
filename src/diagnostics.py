"""
src/diagnostics.py
==================
Diagnostic and validation plots for the SVJ interpolation framework.

Transform diagnostics
---------------------
    plot_observable_transforms(tsv_path, obs='default', n_events=None)

For each selected observable this produces a 3-panel figure:
  Left:   raw distribution histogram
  Middle: after all invertible transforms (but before distribution fit)
  Right:  after full pipeline (standard-normal mapped) with N(0,1) overlay

A summary line reports n_used / n_total (events passing range checks).

Production-validation plots
---------------------------
    plot_validation(npz_path, ...)
    show_validation(npz_path, ...)

For each observable: overlapping JS-distance histograms (baseline / interpolation /
nearest-grid) across all N3 validation points.  A final MMD panel shows the same
three-way comparison for the joint distribution.  Designed to be the primary
diagnostic after running validate_production.py.
"""

import sys
import numpy as np
import scipy.stats as st
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
from observables import (
    OBSERVABLES, TRANSFORMS, DISTRIBUTIONS, DEFAULT_SCAN,
    event_valid_mask, fit_observable_col, validate_scan_selection,
    load_tsv,
)


def plot_observable_transforms(
    tsv_path,
    obs='default',
    n_events=None,
    figsize_per_obs=(12, 3),
    bins=60,
):
    """
    Plot the raw, partially-transformed, and fully-transformed distributions
    for each selected observable.

    Parameters
    ----------
    tsv_path : str or Path
        Path to a TSV written by svj_regression (header line `# col0\\tcol1\\t...`).
    obs : str or list of str
        'default' → DEFAULT_SCAN; otherwise a list of observable names,
        or a comma-separated string.
    n_events : int or None
        If given, subsample to the first n_events rows after filtering.
    figsize_per_obs : (w, h)
        Figure size for each observable's 3-panel row.
    bins : int
        Number of histogram bins.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        One figure per observable.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for diagnostics.")

    # Parse obs selection
    if obs == 'default':
        obs_selection = list(DEFAULT_SCAN)
    elif isinstance(obs, str):
        obs_selection = [s.strip() for s in obs.split(',')]
    else:
        obs_selection = list(obs)
    validate_scan_selection(obs_selection)

    # Load TSV
    tsv_path = Path(tsv_path)
    data, col_map = load_tsv(tsv_path)
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D TSV, got shape {data.shape}.")

    n_total = len(data)

    # Range check + filter
    mask, n_disc_per_obs = event_valid_mask(data, obs_selection, col_map)
    X_valid = data[mask]
    n_used  = len(X_valid)
    n_disc_total = n_total - n_used

    print(f"Events: {n_used} / {n_total} pass all range checks "
          f"({n_disc_total} discarded)")
    if n_disc_per_obs.sum() > 0:
        for obs_name, nd in zip(obs_selection, n_disc_per_obs):
            if nd > 0:
                print(f"  {obs_name}: {nd} events discarded by range check")

    if n_events is not None:
        X_valid = X_valid[:n_events]
        print(f"Subsampling to {len(X_valid)} events.")

    figs = []
    for obs_name in obs_selection:
        obs_spec  = OBSERVABLES[obs_name]
        col_name  = obs_spec['col']
        col       = col_map[col_name]
        pipeline  = obs_spec['pipeline'] or []
        dist_name = obs_spec['distribution']
        label     = obs_spec.get('label', obs_name)

        x_raw = X_valid[:, col].copy()

        # Apply non-fitted transforms up to (but not including) the last step,
        # so panel 2 shows the distribution that the fitted transform sees.
        y = x_raw.copy()
        for t_name, t_fixed in pipeline[:-1] if len(pipeline) > 1 else []:
            t    = TRANSFORMS[t_name]
            y, _ = t['fit'](y, **t_fixed)

        if pipeline:
            last_t, last_fixed = pipeline[-1]
            y_pre_fit, _ = TRANSFORMS[last_t]['fit'](y, **last_fixed)
        else:
            y_pre_fit = y.copy()

        # Full pipeline fit → probit-mapped values + all fitted params
        y_std, all_params = fit_observable_col(x_raw, pipeline, dist_name)

        dist_spec   = DISTRIBUTIONS[dist_name]
        n_d         = dist_spec['n_params']
        dist_params = tuple(all_params[-n_d:])

        pipe_title = _fmt_pipeline_title(pipeline, all_params, dist_name)
        dist_label = _fmt_dist_label(dist_name, dist_params)

        with plt.rc_context(_PLOT_RCPARAMS):
            fig, axes = plt.subplots(1, 3, figsize=figsize_per_obs)

            # Two-line suptitle: observable name + fitted pipeline summary
            fig.suptitle(
                f'{label}   (n={len(x_raw):,})\n{pipe_title}',
                fontsize=10)

            # Panel 1: raw distribution
            ax = axes[0]
            ax.hist(x_raw, bins=bins, density=True,
                    alpha=0.7, color='steelblue', histtype='stepfilled')
            ax.hist(x_raw, bins=bins, density=True,
                    alpha=0.9, color='steelblue', histtype='step', linewidth=1.2)
            ax.set_xlabel(label, fontsize=10)
            ax.set_ylabel('Density', fontsize=9)
            ax.set_title('Raw', fontsize=10)

            # Panel 2: after all transforms, with fitted distribution overlaid
            ax = axes[1]
            ax.hist(y_pre_fit, bins=bins, density=True,
                    alpha=0.7, color='darkorange', histtype='stepfilled',
                    label='data')
            ax.hist(y_pre_fit, bins=bins, density=True,
                    alpha=0.9, color='darkorange', histtype='step', linewidth=1.2)
            xg = np.linspace(y_pre_fit.min(), y_pre_fit.max(), 400)
            ax.plot(xg, dist_spec['dist'].pdf(xg, *dist_params),
                    color='crimson', lw=1.8, label=dist_label)
            ax.set_title('After transforms', fontsize=10)
            ax.set_xlabel('Transformed value', fontsize=10)
            ax.legend(fontsize=7, framealpha=0.6)

            # Panel 3: probit-mapped (should be N(0,1))
            ax = axes[2]
            ax.hist(y_std, bins=bins, density=True,
                    alpha=0.7, color='seagreen', histtype='stepfilled',
                    label='data')
            ax.hist(y_std, bins=bins, density=True,
                    alpha=0.9, color='seagreen', histtype='step', linewidth=1.2)
            xn = np.linspace(-4, 4, 400)
            ax.plot(xn, st.norm.pdf(xn), 'k--', lw=1.5, label='N(0,1)')
            ax.set_title('Standard-normal mapped', fontsize=10)
            ax.set_xlabel('z', fontsize=10)
            ax.legend(fontsize=8, framealpha=0.6)

            fig.tight_layout(rect=[0, 0, 1, 0.88])
            figs.append(fig)

    return figs


def show_observable_transforms(tsv_path, obs='default', n_events=None, **kwargs):
    """Wrapper that calls plot_observable_transforms and shows all figures."""
    import matplotlib.pyplot as plt
    figs = plot_observable_transforms(tsv_path, obs=obs, n_events=n_events, **kwargs)
    for fig in figs:
        fig.show()
    plt.show()
    return figs


# ── Shared style (matches svj_explorer.py) ────────────────────────────────────

_PLOT_RCPARAMS = {
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'DejaVu Serif'],
    'axes.grid':       True,
    'grid.alpha':      0.3,
    'grid.linewidth':  0.5,
    'axes.axisbelow':  True,
    'axes.labelsize':  11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8,
}

# One color per comparison, used consistently across all panels.
_VAL_COLORS = {
    'baseline': 'steelblue',
    'interp':   'seagreen',
    'nearest':  'darkorange',
}
_VAL_LABELS = {
    'baseline': 'Baseline',
    'interp':   'Interpolation',
    'nearest':  'Nearest grid',
}

# ── Transform / distribution display helpers ──────────────────────────────────

_TRANSFORM_DISPLAY = {
    'boxcox':      'Box–Cox',   # Box–Cox
    'affine_flip': 'flip',
    'log':         'log',
    'abs_value':   '|·|',       # |·|
}

# Fitted parameter names per transform (positional, matching fit() return order).
_TRANSFORM_PARAM_NAMES = {
    'boxcox': ('λ',),           # λ
}

# Distribution parameter names (positional, matching scipy dist.fit() return order).
_DIST_PARAM_NAMES = {
    'gennorm': ('β', 'μ', 'σ'),          # β  μ  σ
    'norm':    ('μ', 'σ'),                     # μ  σ
    't':       ('ν', 'μ', 'σ'),           # ν  μ  σ
    'gamma':   ('α', 'loc', 'σ'),              # α  loc  σ
    'beta':    ('α', 'β', 'loc', 'scale'),     # α  β  loc  scale
    'fisk':    ('c', 'loc', 'σ'),                   # c  loc  σ
    'lognorm': ('s', 'loc', 'σ'),                   # s  loc  σ
}


def _fmt_dist_label(dist_name, dist_params):
    """
    Format a distribution + fitted params as a legend string.
    E.g. 'gennorm(β=1.874, μ=0.008, σ=1.023)'
    """
    pnames = _DIST_PARAM_NAMES.get(dist_name, ())
    inner  = ', '.join(
        f'{pnames[i] if i < len(pnames) else "p"+str(i)}={dist_params[i]:.3f}'
        for i in range(len(dist_params)))
    return f'{dist_name}({inner})'


def _fmt_pipeline_title(pipeline, all_params, dist_name):
    """
    Build a compact pipeline summary for use in figure titles.
    E.g. 'flip → Box–Cox(λ=0.215) → gennorm(β=2.134, μ=0.002, σ=0.987)'

    Parameters
    ----------
    pipeline   : list of (transform_name, fixed_params_dict)
    all_params : tuple  — fitted params from fit_observable_col
                          (transform params in order, then dist params)
    dist_name  : str
    """
    parts = []
    pos   = 0
    for t_name, t_fixed in (pipeline or []):
        n       = TRANSFORMS[t_name]['n_fitted']
        t_par   = all_params[pos:pos + n]
        pos    += n
        display = _TRANSFORM_DISPLAY.get(t_name, t_name)
        if n > 0:
            pnames = _TRANSFORM_PARAM_NAMES.get(t_name, ())
            inner  = ', '.join(
                f'{pnames[i] if i < len(pnames) else "p"+str(i)}={t_par[i]:.3f}'
                for i in range(n))
            parts.append(f'{display}({inner})')
        else:
            # Show non-default fixed params (e.g. affine_flip with a≠1)
            non_default = {k: v for k, v in t_fixed.items()
                           if not (t_name == 'affine_flip' and k == 'a'
                                   and abs(v - 1.0) < 1e-12)}
            if non_default:
                inner = ', '.join(f'{k}={v:.3g}' for k, v in non_default.items())
                parts.append(f'{display}({inner})')
            else:
                parts.append(display)

    dist_params = all_params[pos:]
    parts.append(_fmt_dist_label(dist_name, dist_params))
    return ' → '.join(parts)   # →


def _val_hist(ax, values, color, label, bins, range_):
    """
    Draw one density histogram for a validation metric (JS or MMD).

    Uses a lightly filled body plus a solid step outline so three overlapping
    histograms remain readable.  A dashed vertical line marks the mean, and
    the mean is appended to the legend label.
    """
    v = values[np.isfinite(values)]
    if len(v) == 0:
        return
    mu = float(np.mean(v))
    ax.hist(v, bins=bins, range=range_, density=True,
            color=color, alpha=0.25, histtype='stepfilled')
    ax.hist(v, bins=bins, range=range_, density=True,
            color=color, alpha=0.9,  histtype='step', linewidth=1.6,
            label=f'{label}  (mean={mu:.4f})')
    ax.axvline(mu, color=color, linestyle='--', linewidth=1.1, alpha=0.8)


def plot_validation(
    npz_path,
    bins=40,
    figsize=None,
    title=None,
    ncols=3,
):
    """
    Plot JS-distance histograms (one per observable) and an MMD histogram
    from a validation_production.npz output file.

    Each JS panel shows three overlapping histograms across the N3 validation
    points, one for each comparison:

      Baseline     JS(truth_1, truth_2)   — statistical noise floor
      Interpolation  JS(model,   truth_1) — quality of the scan interpolation
      Nearest grid   JS(nearest, truth_1) — naive nearest-grid-point alternative

    The final panel shows the same three comparisons for the joint MMD
    (all observables together).

    Interpretation guide
    --------------------
    - Baseline is the best achievable result given finite N2 statistics.
    - If Interpolation ≈ Baseline: the model is near the statistical limit.
    - If Interpolation < Nearest grid: the interpolation beats the naive
      alternative (the primary goal).
    - If Interpolation > Nearest grid: the interpolation is adding error;
      consider a denser grid or revisiting the transform choices.

    Parameters
    ----------
    npz_path : str or Path
        Path to the validation NPZ produced by validate_production.py
        (e.g. simulated/svj/validation_production.npz).
    bins : int
        Histogram bins per panel (default 40).
    figsize : (float, float) or None
        Figure size. Auto-sized from ncols/nrows when None.
    title : str or None
        Figure suptitle. Defaults to the filename and key run settings.
    ncols : int
        Subplot columns (default 3).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Call fig.savefig(...) to save, or just display in a notebook.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_validation.")

    npz = np.load(npz_path, allow_pickle=True)

    js_base  = np.array(npz['js_baseline'])    # (N3, n_obs)
    js_inter = np.array(npz['js_interp'])      # (N3, n_obs)
    js_near  = np.array(npz['js_nearest'])     # (N3, n_obs)
    mmd_base  = np.array(npz['mmd_baseline'])  # (N3,)
    mmd_inter = np.array(npz['mmd_interp'])    # (N3,)
    mmd_near  = np.array(npz['mmd_nearest'])   # (N3,)
    obs_names = [str(n) for n in npz['obs_names']]
    N3 = int(npz['N3'])
    N2 = int(npz['N2'])
    N1 = int(npz['N1'])

    n_obs    = len(obs_names)
    n_panels = n_obs + 1           # one JS panel per observable + one MMD panel
    nrows    = int(np.ceil(n_panels / ncols))

    if figsize is None:
        figsize = (5.2 * ncols, 3.8 * nrows)

    with plt.rc_context(_PLOT_RCPARAMS):
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = np.array(axes).flatten()

        # ── JS panels (one per observable) ────────────────────────────────────
        for i, obs_name in enumerate(obs_names):
            ax  = axes[i]
            lbl = OBSERVABLES.get(obs_name, {}).get('label', obs_name)

            col_base  = js_base[:,  i]
            col_inter = js_inter[:, i]
            col_near  = js_near[:,  i]

            # Shared x-range over all three populations so bars are comparable.
            all_vals = np.concatenate([col_base, col_inter, col_near])
            finite   = all_vals[np.isfinite(all_vals)]
            if len(finite) == 0:
                ax.set_title(f'{lbl}\n(all NaN)', fontsize=10)
                continue
            xlo = max(0.0, float(np.percentile(finite, 0.5)))
            xhi = float(np.percentile(finite, 99.5))
            rng = (xlo, xhi)

            # Draw in order: baseline first (widest), nearest, interp on top.
            _val_hist(ax, col_base,  _VAL_COLORS['baseline'],
                      _VAL_LABELS['baseline'], bins, rng)
            _val_hist(ax, col_near,  _VAL_COLORS['nearest'],
                      _VAL_LABELS['nearest'],  bins, rng)
            _val_hist(ax, col_inter, _VAL_COLORS['interp'],
                      _VAL_LABELS['interp'],   bins, rng)

            n_ok = int(np.sum(np.isfinite(col_inter)))
            ax.set_xlabel('JS distance', fontsize=10)
            ax.set_ylabel('Density',     fontsize=9)
            ax.set_title(f'{lbl}  (n={n_ok})', fontsize=10)
            ax.legend(framealpha=0.5)

        # ── MMD panel (last slot) ─────────────────────────────────────────────
        ax_mmd   = axes[n_obs]
        all_mmd  = np.concatenate([mmd_base, mmd_inter, mmd_near])
        finite_m = all_mmd[np.isfinite(all_mmd)]
        if len(finite_m) > 0:
            xlo_m = max(0.0, float(np.percentile(finite_m, 0.5)))
            xhi_m = float(np.percentile(finite_m, 99.5))
            rng_m = (xlo_m, xhi_m)
            _val_hist(ax_mmd, mmd_base,  _VAL_COLORS['baseline'],
                      _VAL_LABELS['baseline'], bins, rng_m)
            _val_hist(ax_mmd, mmd_near,  _VAL_COLORS['nearest'],
                      _VAL_LABELS['nearest'],  bins, rng_m)
            _val_hist(ax_mmd, mmd_inter, _VAL_COLORS['interp'],
                      _VAL_LABELS['interp'],   bins, rng_m)
        n_ok_mmd = int(np.sum(np.isfinite(mmd_inter)))
        ax_mmd.set_xlabel('MMD',     fontsize=10)
        ax_mmd.set_ylabel('Density', fontsize=9)
        ax_mmd.set_title(f'MMD — joint  (n={n_ok_mmd})', fontsize=10)
        ax_mmd.legend(framealpha=0.5)

        # Hide unused axes (if n_panels < nrows * ncols)
        for j in range(n_panels, len(axes)):
            axes[j].set_visible(False)

        if title is None:
            title = (f'{Path(npz_path).name} — '
                     f'N3={N3}  N2={N2}  N1={N1}')
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()

    return fig


def show_validation(npz_path, **kwargs):
    """
    Load a validation_production.npz and display the JS + MMD plot.

    Thin wrapper around plot_validation that also calls plt.show().
    All keyword arguments are forwarded to plot_validation.

    In a Jupyter notebook, use this function (not plot_validation) to avoid
    the figure being rendered twice: plot_validation returns the figure object,
    which Jupyter auto-displays as the cell output on top of plt.show().

    Example
    -------
    from diagnostics import show_validation
    show_validation('simulated/svj/validation_production.npz')
    """
    import matplotlib.pyplot as plt
    plot_validation(npz_path, **kwargs)
    plt.show()
