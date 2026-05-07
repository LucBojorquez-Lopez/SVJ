"""
svj_explorer.py
===============
Interactive SVJ distribution explorer for Jupyter notebooks.

Usage:
    %matplotlib widget
    from svj_explorer import show
    show()

Requires: ipympl  (`pip install ipympl`)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import ipywidgets as widgets
from IPython.display import display
import subprocess
import tempfile
import threading
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import helpers


# ── Config helpers ────────────────────────────────────────────────────────────

def _parse_cfg(path='svj_regression.cfg'):
    """Return dict of numeric values from the cfg file (comments stripped, expressions evaled)."""
    result = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.split('#')[0].strip()
                if '=' not in line:
                    continue
                key, _, val = line.partition('=')
                key = key.strip()
                try:
                    result[key] = eval(val.strip())
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return result


_CFG = _parse_cfg()
# mPi and LambdaDQCD scale with mRho; rinv2 = rinv; these are truly fixed:
_TRULY_FIXED = [
    ('mq',  'mq',  '{:.1f} GeV'),
    ('Brl', 'Brl', '{:.2f}'),
    ('jetR','jetR','{:.1f}'),
]


def _fixed_params_html(mRho, rinv):
    parts = []
    for key, label, fmt in _TRULY_FIXED:
        if key in _CFG:
            parts.append(f'{label}={fmt.format(_CFG[key])}')
    parts.append(f'rinv2={rinv:.2f}')
    parts.append(f'mPi={(8/15.5)*mRho:.2f} GeV')
    parts.append(f'ΛD={(5/15.5)*mRho:.2f} GeV')
    return ('<br><span style="color:#777; font-size:0.9em">Fixed: '
            + ',  '.join(parts) + '</span>')


def _make_validate_cfg(mZ, mRho, rinv, alphaD, n_events=100_000):
    """Generate cfg text for a validation run with the given slider parameters."""
    mq         = _CFG.get('mq',       4.0)
    Brl        = _CFG.get('Brl',      0.3)
    jetR       = _CFG.get('jetR',     1.0)
    nWorkers   = int(_CFG.get('nWorkers', 14))
    mPi        = (8  / 15.5) * mRho
    LambdaDQCD = (5  / 15.5) * mRho
    return (
        f'mZ         = {mZ}\n'
        f'mq         = {mq}\n'
        f'mPi        = {mPi}\n'
        f'mRho       = {mRho}\n'
        f'rinv       = {rinv}\n'
        f'rinv2      = {rinv}\n'
        f'Brl        = {Brl}\n'
        f'alphaD     = {alphaD}\n'
        f'nEvent     = {n_events}\n'
        f'jetR       = {jetR}\n'
        f'LambdaDQCD = {LambdaDQCD}\n'
        f'nWorkers   = {nWorkers}\n'
        f'save_tsv      = 1\n'
        f'jets_vis_only = 1\n'
        f'dijet_only    = 0\n'
        f'full_obs      = 1\n'
    )


def _load_true_data():
    """Load and preprocess the most recently simulated jets_default.tsv."""
    dataL = np.loadtxt('data/regression/jets_default.tsv')
    mask_cols = np.array([0, 1, 2, 4, 5, 7, 8, 11, 12, 13, 14, 15])
    data_to_fit = dataL[:, mask_cols].copy()
    data_to_fit[:, 4] = 1 - data_to_fit[:, 4]          # thrust → 1 - thrust
    return helpers.preprocess_data(data_to_fit)


# ── Observable metadata ───────────────────────────────────────────────────────

_OBS_NAMES = [
    'leadVisPt', 'leadWidth', 'MET', 'maxMuPt', 'inv_jetThrust',
    'hemiMass1', 'hemiMass2', 'e2c', 'e3c', 'tau1', 'tau2', 'tau3',
]

_OBS_LABELS = [
    r'Lead jet $p_T$ (GeV)',
    r'Lead jet width',
    r'MET (GeV)',
    r'Max $\mu$ $p_T$ (GeV)',
    r'$1 - $ jet thrust',
    r'Hemi-mass 1 (GeV)',
    r'Hemi-mass 2 (GeV)',
    r'$e_2^c$',
    r'$e_3^c$',
    r'$\tau_1$',
    r'$\tau_2$',
    r'$\tau_3$',
]

# ── Fixed axis ranges (computed once from grid corners at import) ─────────────

def _compute_fixed_ranges(n_corner_samples=3_000):
    mZ_lims     = [helpers._gn_mZ_vals[0],     helpers._gn_mZ_vals[-1]]
    mRho_lims   = [helpers._gn_mRho_vals[0],   helpers._gn_mRho_vals[-1]]
    rinv_lims   = [helpers._gn_rinv_vals[0],   helpers._gn_rinv_vals[-1]]
    alphaD_lims = [helpers._gn_alphaD_vals[0], helpers._gn_alphaD_vals[-1]]

    rng = np.random.default_rng(0)
    chunks = []
    for mZ_c in mZ_lims:
        for mRho_c in mRho_lims:
            for rinv_c in rinv_lims:
                for alphaD_c in alphaD_lims:
                    try:
                        corr, tf = helpers.interpolate_gennorm_params(
                            mZ_c, mRho_c, rinv_c, alphaD_c)
                        s = helpers.sample_svj(corr, tf,
                                               n_samples=n_corner_samples, rng=rng)
                        chunks.append(s)
                    except Exception:
                        pass

    if not chunks:
        return [(0.0, 1.0)] * 12

    all_s = np.vstack(chunks)
    ranges = []
    for i in range(12):
        col = all_s[:, i]
        col = col[np.isfinite(col)]
        if len(col) < 10:
            ranges.append((0.0, 1.0))
        else:
            ranges.append((float(np.percentile(col, 1)),
                           float(np.percentile(col, 99))))
    return ranges


_FIXED_RANGES = _compute_fixed_ranges()


# ── Main entry point ──────────────────────────────────────────────────────────

def show(n_samples=10_000):
    """
    Launch the two-feature SVJ parameter explorer.

    Select any two of the 12 observables to view their marginals and joint
    distribution while varying the four SVJ physics parameters via sliders.
    Press VALIDATE to run a full PYTHIA simulation at the current point and
    overlay the true distributions.

    Parameters
    ----------
    n_samples : int
        Number of model samples to draw per update (default 10 000).
    """
    mZ_min,     mZ_max     = helpers._gn_mZ_vals[0],     helpers._gn_mZ_vals[-1]
    mRho_min,   mRho_max   = helpers._gn_mRho_vals[0],   helpers._gn_mRho_vals[-1]
    rinv_min,   rinv_max   = helpers._gn_rinv_vals[0],   helpers._gn_rinv_vals[-1]
    alphaD_min, alphaD_max = helpers._gn_alphaD_vals[0], helpers._gn_alphaD_vals[-1]

    # ── Mutable closure state ─────────────────────────────────────────────────
    _state = {
        'true_data':     None,   # N×12 array after validation, else None
        'model_samples': None,   # last sampled model array
        'joint_mode':    'single',
        'ax_joint':      None,   # full-width axis (single mode)
        'ax_joint_est':  None,   # left half (split mode)
        'ax_joint_true': None,   # right half (split mode)
    }

    # ── Sliders ───────────────────────────────────────────────────────────────
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

    # ── Feature selectors ─────────────────────────────────────────────────────
    obs_opts = [(name, i) for i, name in enumerate(_OBS_NAMES)]

    w_xfeat = widgets.Dropdown(
        options=obs_opts, value=0,
        description='Feature X:', style={'description_width': '80px'},
        layout=widgets.Layout(width='220px'))

    w_yfeat = widgets.Dropdown(
        options=obs_opts, value=2,
        description='Feature Y:', style={'description_width': '80px'},
        layout=widgets.Layout(width='220px'))

    w_axes = widgets.ToggleButtons(
        options=['Fixed', 'Auto'], value='Fixed',
        description='Axes:',
        style={'description_width': '50px', 'button_width': '80px'})

    # ── Sample-count inputs ───────────────────────────────────────────────────
    int_style  = {'description_width': '90px'}
    int_layout = widgets.Layout(width='220px')

    w_nsamples = widgets.BoundedIntText(
        value=n_samples, min=100, max=500_000, step=1000,
        description='N model:', style=int_style, layout=int_layout)

    w_nvalidate = widgets.BoundedIntText(
        value=100_000, min=1_000, max=2_000_000, step=10_000,
        description='N validate:', style=int_style, layout=int_layout)

    # ── Validate button ───────────────────────────────────────────────────────
    w_validate = widgets.Button(
        description='VALIDATE',
        button_style='warning',
        layout=widgets.Layout(width='120px', height='36px'))

    w_info = widgets.HTML(value='')

    # ── All interactive widgets (for bulk enable/disable) ─────────────────────
    _all_widgets = [w_mZ, w_mRho, w_rinv, w_alphaD,
                    w_xfeat, w_yfeat, w_axes,
                    w_nsamples, w_nvalidate, w_validate]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 8))
    fig.canvas.header_visible = False
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax_xmarg = fig.add_subplot(gs[0, 0])
    ax_ymarg = fig.add_subplot(gs[0, 1])
    _state['ax_joint'] = fig.add_subplot(gs[1, :])

    _rng = np.random.default_rng()

    # ── Joint-axis layout management ──────────────────────────────────────────
    def _ensure_joint_layout(want_split):
        if want_split and _state['joint_mode'] == 'single':
            _state['ax_joint'].remove()
            _state['ax_joint']     = None
            _state['ax_joint_est'] = fig.add_subplot(gs[1, 0])
            _state['ax_joint_true']= fig.add_subplot(gs[1, 1])
            _state['joint_mode']   = 'split'
        elif not want_split and _state['joint_mode'] == 'split':
            _state['ax_joint_est'].remove()
            _state['ax_joint_true'].remove()
            _state['ax_joint_est']  = None
            _state['ax_joint_true'] = None
            _state['ax_joint']      = fig.add_subplot(gs[1, :])
            _state['joint_mode']    = 'single'

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(X, xi, yi, use_fixed):
        true_data = _state['true_data']
        _ensure_joint_layout(true_data is not None)

        ax_xmarg.cla()
        ax_ymarg.cla()

        xdata, ydata = X[:, xi], X[:, yi]
        mask = np.isfinite(xdata) & np.isfinite(ydata)
        xdata, ydata = xdata[mask], ydata[mask]
        xlbl, ylbl   = _OBS_LABELS[xi], _OBS_LABELS[yi]

        if len(xdata) < 20:
            w_info.value += '  <span style="color:orange"> — too few finite values to plot</span>'
            return

        if use_fixed:
            xrng = _FIXED_RANGES[xi]
            yrng = _FIXED_RANGES[yi]
        else:
            xrng = (float(np.percentile(xdata, 1)), float(np.percentile(xdata, 99)))
            yrng = (float(np.percentile(ydata, 1)), float(np.percentile(ydata, 99)))

        # prepare true-data slices up front (needed in both marginals and joint)
        tx = ty = None
        if true_data is not None:
            tx_raw = true_data[:, xi]
            ty_raw = true_data[:, yi]
            tmask  = np.isfinite(tx_raw) & np.isfinite(ty_raw)
            if tmask.sum() >= 20:
                tx, ty = tx_raw[tmask], ty_raw[tmask]

        # marginals
        ax_xmarg.hist(xdata, bins=80, range=xrng, color='steelblue', alpha=0.85,
                      density=True, histtype='step', linewidth=1.5,
                      label='model' if tx is not None else None)
        ax_xmarg.set_xlabel(xlbl, fontsize=10)
        ax_xmarg.set_ylabel('density', fontsize=9)
        if use_fixed:
            ax_xmarg.set_xlim(xrng)

        ax_ymarg.hist(ydata, bins=80, range=yrng, color='darkorange', alpha=0.85,
                      density=True, histtype='step', linewidth=1.5,
                      label='model' if ty is not None else None)
        ax_ymarg.set_xlabel(ylbl, fontsize=10)
        ax_ymarg.set_ylabel('density', fontsize=9)
        if use_fixed:
            ax_ymarg.set_xlim(yrng)

        if tx is not None:
            ax_xmarg.hist(tx, bins=80, range=xrng, color='crimson', alpha=0.85,
                          density=True, histtype='step', linewidth=1.5, label='true')
            ax_ymarg.hist(ty, bins=80, range=yrng, color='crimson', alpha=0.85,
                          density=True, histtype='step', linewidth=1.5, label='true')
            ax_xmarg.legend(fontsize=8, framealpha=0.5)
            ax_ymarg.legend(fontsize=8, framealpha=0.5)

        # joint
        if _state['joint_mode'] == 'single':
            ax_j = _state['ax_joint']
            ax_j.cla()
            ax_j.hist2d(xdata, ydata, bins=80, range=[xrng, yrng],
                        cmap='viridis', density=True)
            ax_j.set_xlabel(xlbl, fontsize=10)
            ax_j.set_ylabel(ylbl, fontsize=10)
        else:
            ax_est  = _state['ax_joint_est']
            ax_true = _state['ax_joint_true']
            ax_est.cla()
            ax_true.cla()
            b = 50
            # shared colour scale: compute both densities first, then find vmax
            H_est, _, _ = np.histogram2d(xdata, ydata, bins=b,
                                         range=[xrng, yrng], density=True)
            H_true, _, _ = np.histogram2d(tx, ty, bins=b,
                                          range=[xrng, yrng], density=True)
            vmax = max(H_est.max(), H_true.max())

            ax_est.hist2d(xdata, ydata, bins=b, range=[xrng, yrng],
                          cmap='viridis', density=True, vmin=0, vmax=vmax)
            ax_est.set_xlabel(xlbl, fontsize=10)
            ax_est.set_ylabel(ylbl, fontsize=10)
            ax_est.set_title('Estimated', fontsize=10)
            ax_true.hist2d(tx, ty, bins=b, range=[xrng, yrng],
                           cmap='viridis', density=True, vmin=0, vmax=vmax)
            ax_true.set_xlabel(xlbl, fontsize=10)
            ax_true.set_ylabel(ylbl, fontsize=10)
            ax_true.set_title('True (simulated)', fontsize=10)

        fig.canvas.draw_idle()

    # ── Update callbacks ──────────────────────────────────────────────────────
    def update(_=None):
        mZ     = w_mZ.value
        mRho   = w_mRho.value
        rinv   = w_rinv.value
        alphaD = w_alphaD.value
        xi     = w_xfeat.value
        yi     = w_yfeat.value
        use_fixed = (w_axes.value == 'Fixed')

        n = w_nsamples.value
        try:
            corr, tf = helpers.interpolate_gennorm_params(mZ, mRho, rinv, alphaD)
            X = helpers.sample_svj(corr, tf, n_samples=n, rng=_rng)
        except ValueError as e:
            w_info.value = f'<span style="color:red">Error: {e}</span>'
            return

        _state['model_samples'] = X

        val_note = ''
        if _state['true_data'] is not None:
            n_true = len(_state['true_data'])
            val_note = f'<br><span style="color:crimson; font-size:0.9em">▶ Validation: {n_true:,} true events</span>'

        w_info.value = (
            f'<span style="font-weight:bold; color:steelblue">'
            f'{n:,} samples — '
            f'mZ\'={mZ:.0f} GeV, mRho={mRho:.1f} GeV, '
            f'rinv={rinv:.2f}, alphaD={alphaD:.2f}'
            f'</span>'
            + _fixed_params_html(mRho, rinv)
            + val_note
        )

        _draw(X, xi, yi, use_fixed)

    def update_clear(_=None):
        """Called by slider changes — clears any validation overlay."""
        _state['true_data'] = None
        update()

    # ── Validation thread ─────────────────────────────────────────────────────
    def _on_validate(_b):
        mZ     = w_mZ.value
        mRho   = w_mRho.value
        rinv   = w_rinv.value
        alphaD = w_alphaD.value

        for w in _all_widgets:
            w.disabled = True

        n_val = w_nvalidate.value
        w_info.value = (
            '<span style="color:#c07000; font-weight:bold">'
            f'⏳ Running PYTHIA simulation ({n_val:,} events) — '
            f'mZ\'={mZ:.0f}, mRho={mRho:.1f}, rinv={rinv:.2f}, alphaD={alphaD:.2f} …'
            '</span>'
        )

        def _thread():
            tmp_path = None
            try:
                cfg_text = _make_validate_cfg(mZ, mRho, rinv, alphaD, n_val)
                with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.cfg', dir='.', delete=False) as f:
                    f.write(cfg_text)
                    tmp_path = f.name

                result = subprocess.run(
                    ['./svj_regression', tmp_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f'svj_regression exited {result.returncode}:\n'
                        + result.stderr[-600:])

                true_data = _load_true_data()
                _state['true_data'] = true_data

                n_true = len(true_data)
                w_info.value = (
                    f'<span style="font-weight:bold; color:steelblue">'
                    f'{n_samples:,} samples — '
                    f'mZ\'={mZ:.0f} GeV, mRho={mRho:.1f} GeV, '
                    f'rinv={rinv:.2f}, alphaD={alphaD:.2f}'
                    f'</span>'
                    + _fixed_params_html(mRho, rinv)
                    + f'<br><span style="color:crimson; font-size:0.9em">▶ Validation: {n_true:,} true events</span>'
                )

                X = _state['model_samples']
                if X is not None:
                    _draw(X, w_xfeat.value, w_yfeat.value, w_axes.value == 'Fixed')

            except Exception as e:
                w_info.value = f'<span style="color:red">Validation error: {e}</span>'
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                for w in _all_widgets:
                    w.disabled = False

        threading.Thread(target=_thread, daemon=True).start()

    # ── Wire up observers ─────────────────────────────────────────────────────
    for w in [w_mZ, w_mRho, w_rinv, w_alphaD]:
        w.observe(update_clear, names='value')
    for w in [w_xfeat, w_yfeat, w_axes, w_nsamples]:
        w.observe(update, names='value')
    w_validate.on_click(_on_validate)

    # ── Layout ────────────────────────────────────────────────────────────────
    controls = widgets.VBox([
        w_mZ, w_mRho, w_rinv, w_alphaD,
        widgets.HBox([w_xfeat, w_yfeat, w_axes, w_validate],
                     layout=widgets.Layout(margin='8px 0px')),
        widgets.HBox([w_nsamples, w_nvalidate],
                     layout=widgets.Layout(margin='0px 0px 8px 0px')),
        w_info,
    ])

    display(controls)
    update()
