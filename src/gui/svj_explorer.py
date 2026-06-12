"""
svj_explorer.py
===============
Interactive SVJ distribution explorer for Jupyter notebooks.

Supports both the old gennorm_scan.npz format (MVN copula, 12 fixed observables)
and the new svj_scan.npz format (MVT copula, dynamic observable selection from
src/observables.py).

Usage
-----
    %matplotlib widget
    from svj_explorer import show
    show()

    # or from the project root:
    %matplotlib widget
    import sys; sys.path.insert(0, 'src/gui')
    from svj_explorer import show
    show()

Requires: ipympl  (pip install ipympl)
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
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
import helpers
from observables import OBSERVABLES, DEFAULT_SCAN, event_valid_mask

_BINARY      = str(_HERE.parent / 'generate_events' / 'svj_regression')
_DEFAULT_CFG = str(_HERE.parent / 'generate_events' / 'svj_regression.cfg')


# ── Observable list — built from loaded NPZ or DEFAULT_SCAN ──────────────────

def _build_obs_list():
    """
    Return (base_names, all_names, all_labels, tsv_to_arr_idx).

    base_names : list[str]  — observables actually in the loaded NPZ
    all_names  : list[str]  — base + computable derived observables
    all_labels : list[str]  — LaTeX labels for all_names
    tsv_to_arr : dict[int→int] — TSV column index → position in base_names
    """
    try:
        helpers._build_gn_interp()
        base_names = list(helpers._gn_meta.get('obs_names', DEFAULT_SCAN))
    except Exception:
        base_names = list(DEFAULT_SCAN)

    tsv_to_arr = {OBSERVABLES[n]['col']: i for i, n in enumerate(base_names)
                  if OBSERVABLES[n].get('col') is not None}

    all_names  = list(base_names)
    all_labels = [OBSERVABLES[n].get('label', n) for n in base_names]

    # Append derived observables whose components are both in base_names
    for obs_name, spec in OBSERVABLES.items():
        dc = spec.get('derive_cols')
        if dc is None:
            continue
        num_col, den_col = dc
        if num_col in tsv_to_arr and den_col in tsv_to_arr:
            all_names.append(obs_name)
            all_labels.append(spec.get('label', obs_name))

    return base_names, all_names, all_labels, tsv_to_arr


_BASE_OBS, _OBS_NAMES, _OBS_LABELS, _TSV_TO_ARR = _build_obs_list()
_N_BASE = len(_BASE_OBS)
_N_OBS  = len(_OBS_NAMES)

# Precompute indices for hemiMass enforcement
_IDX_MASS1  = _OBS_NAMES.index('hemiMass1')  if 'hemiMass1'  in _OBS_NAMES else None
_IDX_MASS2  = _OBS_NAMES.index('hemiMass2')  if 'hemiMass2'  in _OBS_NAMES else None
_IDX_MRAT   = _OBS_NAMES.index('mass2/mass1') if 'mass2/mass1' in _OBS_NAMES else None


# ── Config helpers ────────────────────────────────────────────────────────────

def _parse_cfg(path=None):
    """Return dict of values from the cfg file (Python eval on each value)."""
    if path is None:
        path = _DEFAULT_CFG
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
    """Write a cfg block for a validation PYTHIA run."""
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


# ── Derived-observable computation ────────────────────────────────────────────

def _add_derived(X):
    """
    Append computable derived-observable columns to an (N, n_base) array.
    The set of derived columns is determined at module load by _TSV_TO_ARR.
    """
    eps = 1e-10
    derived = []
    for obs_name, spec in OBSERVABLES.items():
        dc = spec.get('derive_cols')
        if dc is None:
            continue
        num_col, den_col = dc
        if num_col in _TSV_TO_ARR and den_col in _TSV_TO_ARR:
            num_idx = _TSV_TO_ARR[num_col]
            den_idx = _TSV_TO_ARR[den_col]
            derived.append(X[:, num_idx] / np.maximum(X[:, den_idx], eps))
    if derived:
        return np.hstack([X] + [c[:, None] for c in derived])
    return X


# ── True-data loader ──────────────────────────────────────────────────────────

def _load_true_data():
    """
    Load and extract base + derived observables from
    data/regression/jets_default.tsv (written by VALIDATE).
    Returns (N, n_obs_all) array in original physical units.

    No range-check filtering is applied — we display all finite events
    so the true distribution is unbiased.  The GUI cut sliders let the
    user restrict the visible range interactively.
    """
    data = np.loadtxt('data/regression/jets_default.tsv', comments='#')
    if data.ndim != 2 or data.shape[1] != 22:
        raise ValueError(f"Expected 22-column TSV, got shape {data.shape}.")

    X = np.column_stack([data[:, OBSERVABLES[n]['col']] for n in _BASE_OBS])
    finite_mask = np.all(np.isfinite(X), axis=1)
    return _add_derived(X[finite_mask])


# ── Model sampling (format-agnostic) ─────────────────────────────────────────

def _sample_model(mZ, mRho, rinv, alphaD, n_samples, rng):
    """
    Draw n_samples from the interpolated SVJ model.  Handles both the old
    gennorm (Gaussian-copula) format and the new MVT format transparently.
    """
    result = helpers.interpolate_svj_params(mZ, mRho, rinv, alphaD)
    if len(result) == 5:
        # New format: (R_upper, nu, obs_p, param_offsets, obs_names)
        X = helpers.sample_svj_new(*result, n_samples=n_samples, rng=rng)
    else:
        # Old format: (corr_66, tf_12x4)
        X = helpers.sample_svj(*result, n_samples=n_samples, rng=rng)
    return _add_derived(X)


# ── Fixed axis ranges (computed once at import) ───────────────────────────────

def _compute_fixed_ranges(n_corner_samples=3_000):
    try:
        mZ_vals, mRho_vals, rinv_vals, alphaD_vals = helpers.gn_grid_bounds()
    except Exception:
        return [(0.0, 1.0)] * _N_OBS

    rng    = np.random.default_rng(0)
    chunks = []
    for mZ_c in [mZ_vals[0], mZ_vals[-1]]:
        for mRho_c in [mRho_vals[0], mRho_vals[-1]]:
            for rinv_c in [rinv_vals[0], rinv_vals[-1]]:
                for alphaD_c in [alphaD_vals[0], alphaD_vals[-1]]:
                    try:
                        chunks.append(
                            _sample_model(mZ_c, mRho_c, rinv_c, alphaD_c,
                                          n_corner_samples, rng))
                    except Exception:
                        pass

    if not chunks:
        return [(0.0, 1.0)] * _N_OBS

    all_s  = np.vstack(chunks)
    ranges = []
    for i in range(_N_OBS):
        col = all_s[:, i]
        col = col[np.isfinite(col)]
        ranges.append(
            (float(np.percentile(col, 1)), float(np.percentile(col, 99)))
            if len(col) >= 10 else (0.0, 1.0))
    return ranges


_FIXED_RANGES = _compute_fixed_ranges()


# ── Main entry point ──────────────────────────────────────────────────────────

def show(n_samples=10_000):
    """
    Launch the two-feature SVJ parameter explorer.

    Select any two observables to view their marginals and 2-D joint
    distribution while varying the four SVJ physics parameters via sliders.
    Press VALIDATE to run a full PYTHIA simulation at the current point and
    overlay the true distributions.

    Parameters
    ----------
    n_samples : int
        Number of model samples to draw per update (default 10 000).
    """
    try:
        mZ_vals, mRho_vals, rinv_vals, alphaD_vals = helpers.gn_grid_bounds()
    except FileNotFoundError as e:
        print(f"Could not load SVJ scan: {e}")
        print("Run scan_svj.py first, then restart the notebook.")
        return

    mZ_min,     mZ_max     = mZ_vals[0],     mZ_vals[-1]
    mRho_min,   mRho_max   = mRho_vals[0],   mRho_vals[-1]
    rinv_min,   rinv_max   = rinv_vals[0],   rinv_vals[-1]
    alphaD_min, alphaD_max = alphaD_vals[0], alphaD_vals[-1]

    # ── Mutable closure state ─────────────────────────────────────────────────
    _state = {
        'true_data':     None,
        'model_samples': None,
        'joint_mode':    'single',
        'ax_joint':      None,
        'ax_joint_est':  None,
        'ax_joint_true': None,
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
        options=obs_opts, value=min(2, _N_OBS - 1),
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

    _all_widgets = [w_mZ, w_mRho, w_rinv, w_alphaD,
                    w_xfeat, w_yfeat, w_axes,
                    w_nsamples, w_nvalidate, w_validate]

    # ── Cut widgets ───────────────────────────────────────────────────────────
    def _make_cut_slider(i):
        lo, hi = _FIXED_RANGES[i]
        span   = hi - lo if hi > lo else 1.0
        return widgets.FloatRangeSlider(
            value=[lo, hi], min=lo, max=hi, step=span / 200,
            description=_OBS_NAMES[i], continuous_update=False,
            readout_format='.3g',
            style={'description_width': '100px'},
            layout=widgets.Layout(width='340px'))

    w_cuts = [_make_cut_slider(i) for i in range(_N_OBS)]

    w_reset_cuts = widgets.Button(
        description='Reset cuts',
        layout=widgets.Layout(width='110px', height='28px', margin='4px 0px'))

    _all_widgets += w_cuts + [w_reset_cuts]

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
            _state['ax_joint']      = None
            _state['ax_joint_est']  = fig.add_subplot(gs[1, 0])
            _state['ax_joint_true'] = fig.add_subplot(gs[1, 1])
            _state['joint_mode']    = 'split'
        elif not want_split and _state['joint_mode'] == 'split':
            _state['ax_joint_est'].remove()
            _state['ax_joint_true'].remove()
            _state['ax_joint_est']  = None
            _state['ax_joint_true'] = None
            _state['ax_joint']      = fig.add_subplot(gs[1, :])
            _state['joint_mode']    = 'single'

    # ── Cut mask ──────────────────────────────────────────────────────────────
    def _cut_mask(arr):
        mask = np.ones(len(arr), dtype=bool)
        for i in range(_N_OBS):
            lo, hi         = w_cuts[i].value
            lo_def, hi_def = _FIXED_RANGES[i]
            if lo > lo_def:
                mask &= arr[:, i] >= lo
            if hi < hi_def:
                mask &= arr[:, i] <= hi
        return mask

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(X, xi, yi, use_fixed):
        true_data = _state['true_data']
        _ensure_joint_layout(true_data is not None)

        ax_xmarg.cla()
        ax_ymarg.cla()

        # Enforce hemiMass1 >= hemiMass2 when either axis is a mass column
        mass_idxs = {i for i in (_IDX_MASS1, _IDX_MASS2, _IDX_MRAT) if i is not None}
        if xi in mass_idxs or yi in mass_idxs:
            if _IDX_MASS1 is not None and _IDX_MASS2 is not None:
                X = X[X[:, _IDX_MASS1] >= X[:, _IDX_MASS2]]

        n_model_total = len(X)
        X = X[_cut_mask(X)]
        n_model_pass  = len(X)

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

        tx = ty = None
        n_true_total = n_true_pass = 0
        if true_data is not None:
            n_true_total = len(true_data)
            true_cut     = true_data[_cut_mask(true_data)]
            n_true_pass  = len(true_cut)
            tx_raw = true_cut[:, xi]
            ty_raw = true_cut[:, yi]
            tmask  = np.isfinite(tx_raw) & np.isfinite(ty_raw)
            if tmask.sum() >= 20:
                tx, ty = tx_raw[tmask], ty_raw[tmask]

        # Marginals
        b2 = 30
        ax_xmarg.hist(xdata, bins=b2, range=xrng, color='steelblue', alpha=0.85,
                      density=True, histtype='step', linewidth=1.5,
                      label='model' if tx is not None else None)
        ax_xmarg.set_xlabel(xlbl, fontsize=10)
        ax_xmarg.set_ylabel('density', fontsize=9)
        if use_fixed:
            ax_xmarg.set_xlim(xrng)

        ax_ymarg.hist(ydata, bins=b2, range=yrng, color='darkorange', alpha=0.85,
                      density=True, histtype='step', linewidth=1.5,
                      label='model' if ty is not None else None)
        ax_ymarg.set_xlabel(ylbl, fontsize=10)
        ax_ymarg.set_ylabel('density', fontsize=9)
        if use_fixed:
            ax_ymarg.set_xlim(yrng)

        if tx is not None:
            ax_xmarg.hist(tx, bins=b2, range=xrng, color='crimson', alpha=0.85,
                          density=True, histtype='step', linewidth=1.5, label='true')
            ax_ymarg.hist(ty, bins=b2, range=yrng, color='crimson', alpha=0.85,
                          density=True, histtype='step', linewidth=1.5, label='true')
            ax_xmarg.legend(fontsize=8, framealpha=0.5)
            ax_ymarg.legend(fontsize=8, framealpha=0.5)

        # Joint
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
            H_est,  _, _ = np.histogram2d(xdata, ydata, bins=b,
                                          range=[xrng, yrng], density=True)
            H_true, _, _ = np.histogram2d(tx,    ty,    bins=b,
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

        any_cut = any(
            w_cuts[i].value[0] > _FIXED_RANGES[i][0] or
            w_cuts[i].value[1] < _FIXED_RANGES[i][1]
            for i in range(_N_OBS))
        if any_cut:
            cut_html = (f'<br><span style="color:#555; font-size:0.85em">'
                        f'Cuts: model {n_model_pass:,}/{n_model_total:,}')
            if true_data is not None:
                cut_html += f',&nbsp; true {n_true_pass:,}/{n_true_total:,}'
            cut_html += '</span>'
            w_info.value += cut_html

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
            X = _sample_model(mZ, mRho, rinv, alphaD, n, _rng)
        except ValueError as e:
            w_info.value = f'<span style="color:red">Error: {e}</span>'
            return

        _state['model_samples'] = X

        val_note = ''
        if _state['true_data'] is not None:
            n_true = len(_state['true_data'])
            val_note = (f'<br><span style="color:crimson; font-size:0.9em">'
                        f'▶ Validation: {n_true:,} true events</span>')

        w_info.value = (
            f'<span style="font-weight:bold; color:steelblue">'
            f'{n:,} samples — '
            f"mZ'={mZ:.0f} GeV, mRho={mRho:.1f} GeV, "
            f'rinv={rinv:.2f}, alphaD={alphaD:.2f}'
            f'</span>'
            + _fixed_params_html(mRho, rinv)
            + val_note
        )
        _draw(X, xi, yi, use_fixed)

    def update_clear(_=None):
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
            f"mZ'={mZ:.0f}, mRho={mRho:.1f}, rinv={rinv:.2f}, alphaD={alphaD:.2f} …"
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
                    [_BINARY, tmp_path],
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    raise RuntimeError(
                        f'svj_regression exited {result.returncode}:\n'
                        + result.stderr[-600:])

                true_data = _load_true_data()
                _state['true_data'] = true_data

                n = w_nsamples.value
                n_true = len(true_data)
                w_info.value = (
                    f'<span style="font-weight:bold; color:steelblue">'
                    f'{n:,} samples — '
                    f"mZ'={mZ:.0f} GeV, mRho={mRho:.1f} GeV, "
                    f'rinv={rinv:.2f}, alphaD={alphaD:.2f}'
                    f'</span>'
                    + _fixed_params_html(mRho, rinv)
                    + f'<br><span style="color:crimson; font-size:0.9em">'
                      f'▶ Validation: {n_true:,} true events</span>'
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

    for i in range(_N_OBS):
        w_cuts[i].observe(update, names='value')

    def _on_reset_cuts(_b):
        for i in range(_N_OBS):
            w_cuts[i].unobserve(update, names='value')
        for i in range(_N_OBS):
            w_cuts[i].value = list(_FIXED_RANGES[i])
        for i in range(_N_OBS):
            w_cuts[i].observe(update, names='value')
        update()

    w_reset_cuts.on_click(_on_reset_cuts)

    # ── Layout ────────────────────────────────────────────────────────────────
    left_panel = widgets.VBox([
        w_mZ, w_mRho, w_rinv, w_alphaD,
        widgets.HBox([w_xfeat, w_yfeat, w_axes, w_validate],
                     layout=widgets.Layout(margin='8px 0px')),
        widgets.HBox([w_nsamples, w_nvalidate],
                     layout=widgets.Layout(margin='0px 0px 8px 0px')),
        w_info,
    ])

    cut_panel = widgets.VBox(
        [widgets.HTML(
            '<b style="font-size:0.9em">Cuts &nbsp;'
            '<span style="color:#888; font-weight:normal">'
            '(drag handles; reset = full range)</span></b>')] +
        w_cuts +
        [w_reset_cuts],
        layout=widgets.Layout(
            overflow_y='scroll', max_height='340px',
            border='1px solid #ccc', padding='4px 8px'))

    display(widgets.HBox(
        [left_panel, cut_panel],
        layout=widgets.Layout(align_items='flex-start', gap='20px')))
    update()
