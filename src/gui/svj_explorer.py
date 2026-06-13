"""
svj_explorer.py
===============
Interactive SVJ distribution explorer for Jupyter notebooks.

Loads svj_scan.npz (MVT copula, dynamic observable and parameter selection).
The slider set is built at runtime from the scan axes stored in the NPZ, so
adding or removing scan axes in scan_regression.cfg automatically updates the
explorer without code changes.

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

import json
import itertools
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

_HERE     = Path(__file__).resolve().parent
_REPO     = _HERE.parent.parent
sys.path.insert(0, str(_HERE.parent))
import helpers
from observables import OBSERVABLES, DEFAULT_SCAN, event_valid_mask, load_tsv

_BINARY      = str(_HERE.parent / 'generate_events' / 'svj_regression')
_DEFAULT_CFG = str(_HERE.parent / 'generate_events' / 'svj_regression.cfg')
_META_PATH   = _REPO / 'simulated' / 'svj' / 'svj_scan_meta.json'

# Per-parameter slider step sizes (used when building dynamic sliders)
_PARAM_STEPS = {
    'mZ': 50.0, 'mRho': 0.5, 'mq': 0.1,
    'rinv_pion': 0.01, 'rinv_rho': 0.01,
    'alphaD': 0.01, 'Brmu': 0.01,
    'jetR': 0.1, 'mPi': 0.1, 'LambdaDQCD': 0.1,
}

# Human-readable slider descriptions
_PARAM_LABELS = {
    'mZ':        "mZ' (GeV)",
    'mRho':      'mRho (GeV)',
    'mPi':       'mPi (GeV)',
    'mq':        'mq (GeV)',
    'rinv_pion': 'rinv_pion',
    'rinv_rho':  'rinv_rho',
    'alphaD':    'alphaD',
    'Brmu':      'Brmu',
    'jetR':      'jetR',
    'LambdaDQCD': 'ΛD (GeV)',
}

# Fallback derived-param expressions and fixed params used when no meta JSON exists
_DEFAULT_DERIVED = {
    'mPi':        'mRho * (8.0 / 15.5)',
    'LambdaDQCD': 'mRho * (5.0 / 15.5)',
    'rinv_rho':   'rinv_pion',
}
_DEFAULT_FIXED = {'mq': 4.0, 'Brmu': 0.3, 'jetR': 1.0}


# ── Scan meta loader ──────────────────────────────────────────────────────────

def _load_scan_meta():
    """Return the scan meta dict from svj_scan_meta.json, or an empty dict."""
    try:
        with open(_META_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _resolve_derived(scan_point, meta):
    """
    Given a dict of scan-axis values, return a complete param dict that also
    includes derived and fixed params from the scan meta.
    """
    derived_exprs = meta.get('derived_exprs', _DEFAULT_DERIVED)
    fixed_params  = meta.get('fixed_params',  _DEFAULT_FIXED)
    params = dict(fixed_params)
    params.update(scan_point)
    unresolved = dict(derived_exprs)
    for _ in range(len(unresolved) + 1):
        still_pending = {}
        for name, expr in unresolved.items():
            try:
                params[name] = float(eval(expr, {'__builtins__': {}}, dict(params)))
            except NameError:
                still_pending[name] = expr
            except Exception:
                pass
        if not still_pending:
            break
        unresolved = still_pending
    return params


# ── Observable list — built from loaded NPZ or DEFAULT_SCAN ──────────────────

def _build_obs_list():
    """
    Return (base_names, all_names, all_labels, name_to_arr).

    base_names  : list[str]      — observables in the loaded NPZ
    all_names   : list[str]      — base + computable derived observables
    all_labels  : list[str]      — display labels for all_names
    name_to_arr : dict[str→int]  — name → position in base_names
    """
    try:
        helpers._build_svj_interp()
        base_names = list(helpers._svj_meta.get('obs_names', DEFAULT_SCAN))
    except Exception:
        base_names = list(DEFAULT_SCAN)

    name_to_arr = {n: i for i, n in enumerate(base_names)}
    all_names   = list(base_names)
    all_labels  = [OBSERVABLES[n].get('label', n) for n in base_names]

    for obs_name, spec in OBSERVABLES.items():
        dc = spec.get('derive_cols')
        if dc is None:
            continue
        num_name, den_name = dc
        if num_name in name_to_arr and den_name in name_to_arr:
            all_names.append(obs_name)
            all_labels.append(spec.get('label', obs_name))

    return base_names, all_names, all_labels, name_to_arr


_BASE_OBS, _OBS_NAMES, _OBS_LABELS, _NAME_TO_ARR = _build_obs_list()
_N_BASE = len(_BASE_OBS)
_N_OBS  = len(_OBS_NAMES)

_IDX_MASS1 = _OBS_NAMES.index('hemiMass1')  if 'hemiMass1'  in _OBS_NAMES else None
_IDX_MASS2 = _OBS_NAMES.index('hemiMass2')  if 'hemiMass2'  in _OBS_NAMES else None
_IDX_MRAT  = _OBS_NAMES.index('mass2/mass1') if 'mass2/mass1' in _OBS_NAMES else None


# ── Config helpers ────────────────────────────────────────────────────────────

def _parse_cfg(path=None):
    """Return dict of values from a flat key=value cfg file."""
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


def _fixed_params_html(scan_point):
    """Return an HTML string listing the non-scan params for the given point."""
    meta   = _load_scan_meta()
    params = _resolve_derived(scan_point, meta)
    # Show everything that isn't a scan axis
    non_scan = {k: v for k, v in params.items() if k not in scan_point}
    parts = []
    for k, v in non_scan.items():
        label = _PARAM_LABELS.get(k, k)
        if isinstance(v, float):
            parts.append(f'{label}={v:.4g}')
        else:
            parts.append(f'{label}={v}')
    return ('<br><span style="color:#777; font-size:0.9em">Fixed/derived: '
            + ',  '.join(parts) + '</span>')


def _make_validate_cfg(scan_point, n_events=100_000):
    """Write a complete cfg block for a validation PYTHIA run."""
    meta   = _load_scan_meta()
    params = _resolve_derived(scan_point, meta)
    nWorkers = int(_CFG.get('nWorkers', 14))

    lines = ['# auto-generated validation config']
    for k, v in params.items():
        lines.append(f'{k} = {v}')
    lines += [
        f'nEvent        = {n_events}',
        f'nWorkers      = {nWorkers}',
        'save_tsv      = 1',
        'jets_vis_only = 1',
        'dijet_only    = 0',
    ]
    return '\n'.join(lines) + '\n'


# ── Derived-observable computation ────────────────────────────────────────────

def _add_derived(X):
    """Append computable derived columns to an (N, n_base) array."""
    eps     = 1e-10
    derived = []
    for obs_name, spec in OBSERVABLES.items():
        dc = spec.get('derive_cols')
        if dc is None:
            continue
        num_name, den_name = dc
        if num_name in _NAME_TO_ARR and den_name in _NAME_TO_ARR:
            num_idx = _NAME_TO_ARR[num_name]
            den_idx = _NAME_TO_ARR[den_name]
            derived.append(X[:, num_idx] / np.maximum(X[:, den_idx], eps))
    if derived:
        return np.hstack([X] + [c[:, None] for c in derived])
    return X


# ── True-data loader ──────────────────────────────────────────────────────────

def _load_true_data():
    """Load base + derived observables from simulated/tsv/jets_default.tsv."""
    data, col_map = load_tsv('simulated/tsv/jets_default.tsv')
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D TSV, got shape {data.shape}.")
    X = np.column_stack([data[:, col_map[OBSERVABLES[n]['col']]] for n in _BASE_OBS])
    finite_mask = np.all(np.isfinite(X), axis=1)
    return _add_derived(X[finite_mask])


# ── Model sampling ────────────────────────────────────────────────────────────

def _sample_model(scan_point, n_samples, rng):
    """Draw n_samples from the interpolated SVJ model at scan_point (dict)."""
    result = helpers.interpolate_svj_params(scan_point)
    X      = helpers.sample_svj_new(*result, n_samples=n_samples, rng=rng)
    return _add_derived(X)


# ── Fixed axis ranges (computed once at import) ───────────────────────────────

def _compute_fixed_ranges(n_corner_samples=3_000):
    try:
        grid_bounds = helpers.svj_grid_bounds()
    except Exception:
        return [(0.0, 1.0)] * _N_OBS

    rng        = np.random.default_rng(0)
    axis_names = list(grid_bounds.keys())
    corner_vals = [(grid_bounds[n][0], grid_bounds[n][-1]) for n in axis_names]
    chunks      = []

    for combo in itertools.product(*corner_vals):
        scan_point = dict(zip(axis_names, combo))
        try:
            chunks.append(_sample_model(scan_point, n_corner_samples, rng))
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

    Sliders are built dynamically from the scan axes stored in svj_scan.npz,
    so the explorer automatically reflects whatever parameters were scanned.
    Press VALIDATE to run a full PYTHIA simulation at the current point and
    overlay the true distributions.

    Parameters
    ----------
    n_samples : int
        Number of model samples to draw per update (default 10 000).
    """
    try:
        grid_bounds = helpers.svj_grid_bounds()   # dict[str, np.ndarray]
    except FileNotFoundError as e:
        print(f"Could not load SVJ scan: {e}")
        print("Run scan_svj.py first, then restart the notebook.")
        return

    axis_names = list(grid_bounds.keys())

    # ── Mutable closure state ─────────────────────────────────────────────────
    _state = {
        'true_data':     None,
        'model_samples': None,
        'joint_mode':    'single',
        'ax_joint':      None,
        'ax_joint_est':  None,
        'ax_joint_true': None,
    }

    # ── Dynamic parameter sliders ─────────────────────────────────────────────
    slider_layout = widgets.Layout(width='520px')
    slider_style  = {'description_width': '110px'}

    sliders = {}
    for name in axis_names:
        vals = grid_bounds[name]
        lo, hi = float(vals[0]), float(vals[-1])
        step   = _PARAM_STEPS.get(name, round((hi - lo) / 100, 6))
        label  = _PARAM_LABELS.get(name, name)
        mid    = round((lo + hi) / 2 / step) * step if step > 0 else (lo + hi) / 2
        mid    = max(lo, min(hi, mid))
        sliders[name] = widgets.FloatSlider(
            value=mid, min=lo, max=hi, step=step,
            description=label, continuous_update=False,
            style=slider_style, layout=slider_layout)

    def _get_scan_point():
        return {name: sliders[name].value for name in axis_names}

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

    _all_widgets = (list(sliders.values()) +
                    [w_xfeat, w_yfeat, w_axes, w_nsamples, w_nvalidate, w_validate])

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
        scan_point = _get_scan_point()
        xi        = w_xfeat.value
        yi        = w_yfeat.value
        use_fixed = (w_axes.value == 'Fixed')
        n         = w_nsamples.value

        try:
            X = _sample_model(scan_point, n, _rng)
        except ValueError as e:
            w_info.value = f'<span style="color:red">Error: {e}</span>'
            return

        _state['model_samples'] = X

        val_note = ''
        if _state['true_data'] is not None:
            n_true   = len(_state['true_data'])
            val_note = (f'<br><span style="color:crimson; font-size:0.9em">'
                        f'▶ Validation: {n_true:,} true events</span>')

        param_str = ',  '.join(
            f"{_PARAM_LABELS.get(k, k)}={v:.4g}" for k, v in scan_point.items())
        w_info.value = (
            f'<span style="font-weight:bold; color:steelblue">'
            f'{n:,} samples — {param_str}'
            f'</span>'
            + _fixed_params_html(scan_point)
            + val_note
        )
        _draw(X, xi, yi, use_fixed)

    def update_clear(_=None):
        _state['true_data'] = None
        update()

    # ── Validation thread ─────────────────────────────────────────────────────
    def _on_validate(_b):
        scan_point = _get_scan_point()

        for w in _all_widgets:
            w.disabled = True

        n_val     = w_nvalidate.value
        param_str = ',  '.join(
            f"{_PARAM_LABELS.get(k, k)}={v:.4g}" for k, v in scan_point.items())
        w_info.value = (
            '<span style="color:#c07000; font-weight:bold">'
            f'⏳ Running PYTHIA simulation ({n_val:,} events) — {param_str} …'
            '</span>'
        )

        def _thread():
            tmp_path = None
            try:
                cfg_text = _make_validate_cfg(scan_point, n_val)
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

                n       = w_nsamples.value
                n_true  = len(true_data)
                w_info.value = (
                    f'<span style="font-weight:bold; color:steelblue">'
                    f'{n:,} samples — {param_str}'
                    f'</span>'
                    + _fixed_params_html(scan_point)
                    + f'<br><span style="color:crimson; font-size:0.9em">'
                      f'▶ Validation: {n_true:,} true events</span>'
                )

                X = _state['model_samples']
                if X is not None:
                    _draw(X, w_xfeat.value, w_yfeat.value,
                          w_axes.value == 'Fixed')

            except Exception as e:
                w_info.value = f'<span style="color:red">Validation error: {e}</span>'
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                for w in _all_widgets:
                    w.disabled = False

        threading.Thread(target=_thread, daemon=True).start()

    # ── Wire up observers ─────────────────────────────────────────────────────
    for w in sliders.values():
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
    left_panel = widgets.VBox(
        list(sliders.values()) + [
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
