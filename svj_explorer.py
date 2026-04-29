"""
svj_explorer.py
===============
Interactive SVJ distribution explorer for Jupyter notebooks.

Usage (two notebook cells):
    %matplotlib widget
    from svj_explorer import show
    show()

Requires: ipympl  (`pip install ipympl`)
"""

import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers

# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_distribution(mZ, mRho, rinv, alphaD, n_samples=100_000, rng=None):
    """
    Sample from the interpolated multivariate-t at (mZ, mRho, rinv, alphaD)
    and apply physical cuts:
        pT > 20 GeV   and   width = exp(logW) < 1

    Returns
    -------
    Y : np.ndarray, shape (n_pass, 3)
        Columns: (pT, logW, logMET) for events passing the cuts.
    """
    if rng is None:
        rng = np.random.default_rng()

    p     = helpers.interpolate_params(mZ, mRho, rinv, alphaD)
    mu    = p[0:3]
    Sigma = np.array([[p[3], p[4], p[5]],
                      [p[4], p[6], p[7]],
                      [p[5], p[7], p[8]]])
    nu    = p[9]

    L       = np.linalg.cholesky(Sigma)
    Z       = rng.standard_normal((n_samples, 3))
    v       = nu / 2
    U       = rng.gamma(shape=v, size=n_samples)
    Y_pre   = mu + np.sqrt(v / U)[:, None] * (Z @ L.T)

    mask     = (Y_pre[:, 0] > 20) & (np.exp(Y_pre[:, 1]) < 1)
    Y        = Y_pre[mask].copy()
    Y[:, 1]  = np.exp(Y[:, 1])   # logW  → width  (unitless)
    Y[:, 2]  = np.exp(Y[:, 2])   # logMET → MET   (GeV)
    return Y


# ── Fixed axis ranges (computed once from grid corners at import) ─────────────

def _compute_fixed_ranges(n_corner_samples=5_000):
    """
    Sample from each of the 16 grid corners and return per-axis (1%, 99%)
    ranges over the combined sample, to be used when axes are fixed.
    """
    mZ_lims     = [helpers._mZ_vals[0],     helpers._mZ_vals[-1]]
    mRho_lims   = [helpers._mRho_vals[0],   helpers._mRho_vals[-1]]
    rinv_lims   = [helpers._rinv_vals[0],   helpers._rinv_vals[-1]]
    alphaD_lims = [helpers._alphaD_vals[0], helpers._alphaD_vals[-1]]

    rng = np.random.default_rng(0)
    chunks = []
    for mZ_c in mZ_lims:
        for mRho_c in mRho_lims:
            for rinv_c in rinv_lims:
                for alphaD_c in alphaD_lims:
                    try:
                        s = sample_distribution(
                            mZ_c, mRho_c, rinv_c, alphaD_c,
                            n_samples=n_corner_samples, rng=rng)
                        if len(s) > 50:
                            chunks.append(s)
                    except Exception:
                        pass

    if not chunks:
        return [(0, 1000), (-5, 2), (0, 10)]

    all_s = np.vstack(chunks)
    return [(float(np.percentile(all_s[:, i], 1)),
             float(np.percentile(all_s[:, i], 99)))
            for i in range(3)]


_FIXED_RANGES = _compute_fixed_ranges()

# ── Labels ────────────────────────────────────────────────────────────────────

_LABELS_1D = [
    r'Leading jet $p_T$ (GeV)',
    r'Jet width',
    r'MET (GeV)',
]

_LABELS_2D = [
    (r'Leading jet $p_T$ (GeV)', r'Jet width'),
    (r'Leading jet $p_T$ (GeV)', r'MET (GeV)'),
    (r'Jet width',               r'MET (GeV)'),
]
_PAIRS_2D = [(0, 1), (0, 2), (1, 2)]


# ── Main entry point ──────────────────────────────────────────────────────────

def show(n_samples=10_000):
    """
    Launch the SVJ parameter explorer. Call from a Jupyter cell after
    running `%matplotlib widget`.

    Parameters
    ----------
    n_samples : int
        Number of points to draw from the distribution per update (default 10 000).
    """
    mZ_min,     mZ_max     = helpers._mZ_vals[0],     helpers._mZ_vals[-1]
    mRho_min,   mRho_max   = helpers._mRho_vals[0],   helpers._mRho_vals[-1]
    rinv_min,   rinv_max   = helpers._rinv_vals[0],   helpers._rinv_vals[-1]
    alphaD_min, alphaD_max = helpers._alphaD_vals[0], helpers._alphaD_vals[-1]

    slider_layout = widgets.Layout(width='520px')
    slider_style  = {'description_width': '110px'}

    w_mZ = widgets.FloatSlider(
        value=round((mZ_min + mZ_max) / 2 / 50) * 50,
        min=mZ_min, max=mZ_max, step=50.,
        description="mZ' (GeV)", continuous_update=False,
        style=slider_style, layout=slider_layout)

    w_mRho = widgets.FloatSlider(
        value=round((mRho_min + mRho_max) / 2 * 2) / 2,
        min=mRho_min, max=mRho_max, step=0.5,
        description='mRho (GeV)', continuous_update=False,
        style=slider_style, layout=slider_layout)

    w_rinv = widgets.FloatSlider(
        value=round((rinv_min + rinv_max) / 2, 2),
        min=rinv_min, max=rinv_max, step=0.01,
        description='rinv', continuous_update=False,
        style=slider_style, layout=slider_layout)

    w_alphaD = widgets.FloatSlider(
        value=round((alphaD_min + alphaD_max) / 2, 2),
        min=alphaD_min, max=alphaD_max, step=0.01,
        description='alphaD', continuous_update=False,
        style=slider_style, layout=slider_layout)

    w_mode = widgets.ToggleButtons(
        options=['1D marginals', '2D histograms', 'Both'],
        value='1D marginals',
        description='Plots:',
        style={'description_width': '50px', 'button_width': '130px'})

    w_axes = widgets.ToggleButtons(
        options=['Fixed', 'Auto'],
        value='Fixed',
        description='Axes:',
        style={'description_width': '50px', 'button_width': '80px'})

    w_info = widgets.HTML(value='')

    # ── Figure: always 2×3, rows toggled on/off ───────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.35)

    _rng = np.random.default_rng()

    def _draw(Y, mode, use_fixed):
        show_1d = mode in ('1D marginals', 'Both')
        show_2d = mode in ('2D histograms', 'Both')

        for ax in axes.flat:
            ax.cla()

        # ── top row: 1D histograms ──
        for i, ax in enumerate(axes[0]):
            if not show_1d:
                ax.set_visible(False)
                continue
            ax.set_visible(True)
            col  = Y[:, i]
            rng  = _FIXED_RANGES[i] if use_fixed else (col.min(), col.max())
            ax.hist(col, bins=80, range=rng, color='steelblue',
                    alpha=0.85, density=True)
            ax.set_xlabel(_LABELS_1D[i], fontsize=10)
            ax.set_ylabel('density', fontsize=9)
            if use_fixed:
                ax.set_xlim(rng)

        # ── bottom row: 2D histograms ──
        for n, ((xi, yi), ax) in enumerate(zip(_PAIRS_2D, axes[1])):
            if not show_2d:
                ax.set_visible(False)
                continue
            ax.set_visible(True)
            xlbl, ylbl = _LABELS_2D[n]
            x, y = Y[:, xi], Y[:, yi]
            xrng = _FIXED_RANGES[xi] if use_fixed else (x.min(), x.max())
            yrng = _FIXED_RANGES[yi] if use_fixed else (y.min(), y.max())
            ax.hist2d(x, y, bins=60, range=[xrng, yrng],
                      cmap='viridis', density=True)
            ax.set_xlabel(xlbl, fontsize=10)
            ax.set_ylabel(ylbl, fontsize=10)

        fig.canvas.draw_idle()

    def update(_=None):
        mZ     = w_mZ.value
        mRho   = w_mRho.value
        rinv   = w_rinv.value
        alphaD = w_alphaD.value
        mode   = w_mode.value
        use_fixed = (w_axes.value == 'Fixed')

        try:
            Y = sample_distribution(mZ, mRho, rinv, alphaD,
                                    n_samples=n_samples, rng=_rng)
        except ValueError as e:
            w_info.value = f'<span style="color:red">Error: {e}</span>'
            return

        n_pass = len(Y)
        pct    = 100 * n_pass / n_samples
        color  = 'green' if pct > 20 else 'orange' if pct > 5 else 'red'
        w_info.value = (
            f'<span style="color:{color}; font-weight:bold">'
            f'Events passing cuts: {n_pass} / {n_samples} ({pct:.1f}%)'
            f'</span>'
        )

        if n_pass < 20:
            w_info.value += '  — too few events to plot'
            return

        _draw(Y, mode, use_fixed)

    for w in [w_mZ, w_mRho, w_rinv, w_alphaD, w_mode, w_axes]:
        w.observe(update, names='value')

    controls = widgets.VBox([
        w_mZ, w_mRho, w_rinv, w_alphaD,
        widgets.HBox([w_mode, w_axes],
                     layout=widgets.Layout(margin='8px 0px')),
        w_info,
    ])

    display(controls)
    update()
