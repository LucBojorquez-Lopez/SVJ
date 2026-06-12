"""
src/diagnostics.py
==================
Diagnostic plots for the SVJ observable transform pipeline.

Main entry point:

    plot_observable_transforms(tsv_path, obs='default', n_events=None)

For each selected observable this produces a 3-panel figure:
  Left:   raw distribution histogram
  Middle: after all invertible transforms (but before distribution fit)
  Right:  after full pipeline (standard-normal mapped) with N(0,1) overlay

A summary line reports n_used / n_total (events passing range checks).
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
        Path to the TSV output of svj_regression (22-column format).
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
    data = np.loadtxt(tsv_path, comments='#')
    if data.ndim != 2 or data.shape[1] != 22:
        raise ValueError(
            f"Expected 22-column TSV, got shape {data.shape}.")

    n_total = len(data)

    # Range check + filter
    mask, n_disc_per_obs = event_valid_mask(data, obs_selection)
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
        obs_spec = OBSERVABLES[obs_name]
        col      = obs_spec['col']
        pipeline = obs_spec['pipeline'] or []
        dist_name = obs_spec['distribution']
        label    = obs_spec.get('label', obs_name)

        x_raw = X_valid[:, col].copy()

        # Build intermediate arrays: one per completed transform step
        intermediates = [('raw', x_raw.copy())]
        y = x_raw.copy()
        for t_name, t_fixed in pipeline[:-1] if len(pipeline) > 1 else []:
            t       = TRANSFORMS[t_name]
            y, _    = t['fit'](y, **t_fixed)
            intermediates.append((t_name, y.copy()))

        # The last transform is usually boxcox (fitted)
        if pipeline:
            last_t, last_fixed = pipeline[-1]
            t = TRANSFORMS[last_t]
            y_pre_fit, t_params = t['fit'](y, **last_fixed)
        else:
            y_pre_fit, t_params = y.copy(), ()
        intermediates.append(('pre-fit (after transforms)', y_pre_fit.copy()))

        # Full pipeline: fit dist + probit
        y_std, all_params = fit_observable_col(x_raw, pipeline, dist_name)

        # Extract dist params (last n_dist_params entries)
        dist_spec = DISTRIBUTIONS[dist_name]
        n_d = dist_spec['n_params']
        dist_params = tuple(all_params[-n_d:])

        fig, axes = plt.subplots(1, 3, figsize=figsize_per_obs)
        fig.suptitle(f"{obs_name}   (n={len(x_raw):,} events)", fontsize=11)

        # Panel 1: raw
        ax = axes[0]
        ax.hist(x_raw, bins=bins, density=True, alpha=0.7, color='steelblue')
        ax.set_xlabel(label)
        ax.set_title('Raw')
        ax.set_ylabel('Density')

        # Panel 2: after transforms (before dist fit)
        ax = axes[1]
        ax.hist(y_pre_fit, bins=bins, density=True, alpha=0.7, color='darkorange')
        # Overlay fitted distribution
        xg = np.linspace(y_pre_fit.min(), y_pre_fit.max(), 400)
        ax.plot(xg, dist_spec['dist'].pdf(xg, *dist_params),
                'r-', lw=1.5, label=f'fitted {dist_name}')
        ax.set_title(f"After transforms")
        ax.set_xlabel('Transformed value')
        ax.legend(fontsize=8)

        # Panel 3: standard-normal mapped
        ax = axes[2]
        ax.hist(y_std, bins=bins, density=True, alpha=0.7, color='seagreen',
                label='data')
        xn = np.linspace(-4, 4, 400)
        ax.plot(xn, st.norm.pdf(xn), 'k--', lw=1.5, label='N(0,1)')
        ax.set_title('Standard-normal mapped')
        ax.set_xlabel('z')
        ax.legend(fontsize=8)

        fig.tight_layout()
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
