"""
src/observables.py
==================
Central registry of observable definitions, invertible transforms, and fitting
distributions for the SVJ regression pipeline.

To add a new transform
----------------------
Add an entry to TRANSFORMS with the four required fields:
  fit(x, **fixed_params)        → (y_array, fitted_params_tuple)
  forward(x, params, **fixed)   → y_array   (apply with known params)
  inverse(y, params, **fixed)   → x_array
  n_fitted  : int               (number of data-fitted params stored per observable)
  requires  : callable(**fixed) → (lo, hi)  or  None for no restriction
  desc      : str

To add a new fitting distribution
----------------------------------
Add an entry to DISTRIBUTIONS referencing a scipy.stats frozen-dist-like object.
  dist      : scipy.stats continuous_rv_generic
  n_params  : int   (number of params returned by dist.fit())
  fit_init  : callable(x) → (args_tuple, kwargs_dict) passed to dist.fit()
  input_range : (lo, hi) the distribution's support (informational)
  desc      : str

To add a new observable
-----------------------
Two-step process: C++ first, then Python.

  Step 1 — src/generate_events/svj_regression.cc
    a. Declare a local variable for the new quantity inside runWorker().
    b. Compute its value using the existing local variables (jet loop results,
       particle loop results, etc.) as needed.
    c. Add its name string to OBS_NAMES (must be unique; drives the TSV header).
    d. Append the variable to the data.push_back({...}) call at the end of
       runWorker(), in the same position as its OBS_NAMES entry.
    Recompile with `make svj_regression` from the project root.

  Step 2 — src/observables.py (this file)
    Add an entry to OBSERVABLES below with:
      col           : str   name matching OBS_NAMES in svj_regression.cc (None for derived)
      pipeline      : list of (transform_name, fixed_params_dict) or None
      distribution  : str key into DISTRIBUTIONS, or None
      default_include : bool   whether this observable is in DEFAULT_SCAN
      label         : str  LaTeX / display label for plots
      desc          : str  one-line physics description

Then optionally update DEFAULT_SCAN (or set default_include=True) to include it
in the default regression scan.
"""

import numpy as np
import scipy.stats as st
import scipy.special as sp
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════════
# Invertible transform registry
# ══════════════════════════════════════════════════════════════════════════════

def _affine_flip_fit(x, a=1.0):
    return np.asarray(a, dtype=float) - x, ()

def _affine_flip_fwd(x, params, a=1.0):
    return np.asarray(a, dtype=float) - x

def _affine_flip_inv(y, params, a=1.0):
    return np.asarray(a, dtype=float) - y

def _log_fit(x):
    return np.log(x), ()

def _log_fwd(x, params):
    return np.log(x)

def _log_inv(y, params):
    return np.exp(y)

def _abs_fit(x):
    return np.abs(x), ()

def _abs_fwd(x, params):
    return np.abs(x)

def _abs_inv(y, params):
    # Assumes y >= 0; sign information is lost.
    return y

def _boxcox_fit(x):
    y, lam = st.boxcox(x)
    return y, (float(lam),)

def _boxcox_fwd(x, params):
    return sp.boxcox(x, params[0])

def _boxcox_inv(y, params):
    return sp.inv_boxcox(y, params[0])


TRANSFORMS = {
    # ── a − x  (default a=1, giving the 1-thrust flip) ──────────────────────
    'affine_flip': {
        'fit':      _affine_flip_fit,
        'forward':  _affine_flip_fwd,
        'inverse':  _affine_flip_inv,
        'n_fitted': 0,
        'requires': lambda a=1.0: (-np.inf, float(a)),  # x must be strictly < a
        'desc':     'a − x  (no fitted params; default a=1 for thrust flip)',
    },
    # ── natural log ──────────────────────────────────────────────────────────
    'log': {
        'fit':      _log_fit,
        'forward':  _log_fwd,
        'inverse':  _log_inv,
        'n_fitted': 0,
        'requires': lambda: (0.0, np.inf),
        'desc':     'log(x)  (requires x > 0)',
    },
    # ── absolute value  (non-invertible; loses sign) ─────────────────────────
    'abs_value': {
        'fit':      _abs_fit,
        'forward':  _abs_fwd,
        'inverse':  _abs_inv,
        'n_fitted': 0,
        'requires': None,
        'desc':     '|x|  (non-invertible; use for signed angles)',
    },
    # ── Box–Cox power transform  (lambda fitted from data) ───────────────────
    'boxcox': {
        'fit':      _boxcox_fit,
        'forward':  _boxcox_fwd,
        'inverse':  _boxcox_inv,
        'n_fitted': 1,
        'requires': lambda: (0.0, np.inf),
        'desc':     'Box–Cox  (lambda fitted; requires x > 0)',
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Fitting distribution registry
# ══════════════════════════════════════════════════════════════════════════════
# fit_init(x) returns (args, kwargs) forwarded to dist.fit(x, *args, **kwargs).
# Positional args are initial guesses for shape parameters (NOT fixed values).
# Use 'f<n>' kwargs (e.g. floc=0) to FIX parameters during MLE.

DISTRIBUTIONS = {
    # ── Generalized normal  (beta, loc, scale) ───────────────────────────────
    'gennorm': {
        'dist':        st.gennorm,
        'n_params':    3,
        'fit_init':    lambda x: ((2.0,), {'loc':   float(np.mean(x)),
                                           'scale': max(float(np.std(x)), 1e-8)}),
        'input_range': (-np.inf, np.inf),
        'desc':        'Generalized normal; shape beta, loc, scale.  beta=2 → Gaussian.',
    },
    # ── Gaussian  (loc, scale) ────────────────────────────────────────────────
    'norm': {
        'dist':        st.norm,
        'n_params':    2,
        'fit_init':    lambda x: ((), {'loc':   float(np.mean(x)),
                                       'scale': max(float(np.std(x)), 1e-8)}),
        'input_range': (-np.inf, np.inf),
        'desc':        'Gaussian normal; loc, scale.',
    },
    # ── Student t  (df, loc, scale) ───────────────────────────────────────────
    't': {
        'dist':        st.t,
        'n_params':    3,
        'fit_init':    lambda x: ((5.0,), {'loc':   float(np.mean(x)),
                                           'scale': max(float(np.std(x)), 1e-8)}),
        'input_range': (-np.inf, np.inf),
        'desc':        'Student t; df, loc, scale.  Heavier tails than Gaussian.',
    },
    # ── Gamma  (a, loc, scale) ────────────────────────────────────────────────
    'gamma': {
        'dist':        st.gamma,
        'n_params':    3,
        'fit_init':    lambda x: ((), {'floc': 0.0}),
        'input_range': (0.0, np.inf),
        'desc':        'Gamma; shape a, loc fixed to 0, scale.  Requires x > 0.',
    },
    # ── Beta  (a, b, loc, scale) ──────────────────────────────────────────────
    'beta': {
        'dist':        st.beta,
        'n_params':    4,
        'fit_init':    lambda x: ((), {'floc': 0.0, 'fscale': 1.0}),
        'input_range': (0.0, 1.0),
        'desc':        'Beta; a, b, loc fixed to 0, scale fixed to 1.  Requires x in (0,1).',
    },
    # ── Fisk (log-logistic)  (c, loc, scale) ─────────────────────────────────
    'fisk': {
        'dist':        st.fisk,
        'n_params':    3,
        'fit_init':    lambda x: ((), {'floc': 0.0}),
        'input_range': (0.0, np.inf),
        'desc':        'Fisk (log-logistic); shape c, loc fixed to 0, scale.  Requires x > 0.',
    },
    # ── Log-normal  (s, loc, scale) ───────────────────────────────────────────
    'lognorm': {
        'dist':        st.lognorm,
        'n_params':    3,
        'fit_init':    lambda x: ((), {'floc': 0.0}),
        'input_range': (0.0, np.inf),
        'desc':        'Log-normal; shape s, loc fixed to 0, scale.  Requires x > 0.',
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# Observable registry  (TSV columns + derived ratios)
# ══════════════════════════════════════════════════════════════════════════════
# col  : observable name matching OBS_NAMES in svj_regression.cc.
#        None for derived observables (computed from other columns; GUI only).
# pipeline=None: TSV column exists but has no supported regression pipeline
#                (discrete, binary, angular, …); users may add one.

OBSERVABLES = {

    # ── leadVisPt: leading visible-pT jet transverse momentum ────────────────
    'leadVisPt': {
        'col':             'leadVisPt',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Lead jet $p_T$ (GeV)',
        'desc':            'Visible transverse momentum of the highest-pT jet.',
    },

    # ── leadWidth: leading jet angular width ──────────────────────────────────
    'leadWidth': {
        'col':             'leadWidth',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Lead jet width',
        'desc':            'pT-weighted mean angular spread of leading jet constituents.',
    },

    # ── MET: missing transverse energy proxy ──────────────────────────────────
    'MET': {
        'col':             'MET',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'MET (GeV)',
        'desc':            'Event missing ET: magnitude of negative visible pT sum.',
    },

    # ── maxElePt: maximum electron pT ─────────────────────────────────────────
    # Often zero in SVJ events (electrons are rare).  Including this observable
    # in the regression discards all events with maxElePt = 0.
    'maxElePt': {
        'col':             'maxElePt',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'Max $e$ $p_T$ (GeV)',
        'desc':            'Max final-state electron pT.  Often 0; including discards those events.',
    },

    # ── maxMuPt: maximum muon pT ──────────────────────────────────────────────
    'maxMuPt': {
        'col':             'maxMuPt',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Max $\mu$ $p_T$ (GeV)',
        'desc':            'Max final-state muon pT (from dark-rho leptonic decays).',
    },

    # ── jetThrust: jet thrust (stored as T; regressed as 1−T) ─────────────────
    # The raw TSV value is jetThrust ∈ [0,1].  The affine_flip maps it to
    # 1−thrust ∈ (0,1], which is then Box-Cox transformed.
    'jetThrust': {
        'col':             'jetThrust',
        'pipeline':        [('affine_flip', {'a': 1.0}), ('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$1 - $ jet thrust',
        'desc':            'Leading-jet thrust T; regressed as 1−T after flipping.',
    },

    # ── transSphericity: event-level transverse sphericity ────────────────────
    # S_T ∈ [0,1]; can be exactly 0 (perfectly pencil-like events), which
    # breaks Box-Cox.  Excluded by default; can be included if zero-events are
    # rare at your parameter point.
    'transSphericity': {
        'col':             'transSphericity',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'Transverse sphericity $S_T$',
        'desc':            'S_T = 2*lambda_min of the 2×2 sphericity tensor.  Can be 0.',
    },

    # ── hemiMass1: larger hemisphere invariant mass ───────────────────────────
    'hemiMass1': {
        'col':             'hemiMass1',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Hemi-mass 1 (GeV)',
        'desc':            'Larger invariant mass of the two event hemispheres (thrust split).',
    },

    # ── hemiMass2: smaller hemisphere invariant mass ──────────────────────────
    'hemiMass2': {
        'col':             'hemiMass2',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Hemi-mass 2 (GeV)',
        'desc':            'Smaller invariant mass of the two event hemispheres.',
    },

    # ── ptBal: pT balance ─────────────────────────────────────────────────────
    # ptBal = |pT_close + pT_far| / (pT_close + pT_far) ∈ [0,1]; can be 0.
    # Requires ≥ 2 jets; events with only 1 jet have ptBal = 0.
    'ptBal': {
        'col':             'ptBal',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$p_T$ balance',
        'desc':            'pT balance between the two jets closest/farthest to MET.  Can be 0.',
    },

    # ── dPhiMETdijet: Δφ(MET, dijet system) ──────────────────────────────────
    # Angle ∈ [0, π]; can be 0 if MET is aligned with the dijet.
    'dPhiMETdijet': {
        'col':             'dPhiMETdijet',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$\Delta\phi$(MET, dijet)',
        'desc':            'Azimuthal angle between MET and j1+j2 four-vector.  In [0,π].',
    },

    # ── e2c: 2-point energy correlator ────────────────────────────────────────
    'e2c': {
        'col':             'e2c',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$e_2^c$',
        'desc':            'E2C = Σ_{i<j} z_i z_j ΔR_ij (pT fractions, leading jet).',
    },

    # ── e3c: 3-point energy correlator ────────────────────────────────────────
    'e3c': {
        'col':             'e3c',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$e_3^c$',
        'desc':            'E3C = Σ_{i<j<k} z_i z_j z_k ΔR_ij ΔR_ik ΔR_jk (leading jet).',
    },

    # ── tau1/tau2/tau3: N-subjettiness ────────────────────────────────────────
    'tau1': {
        'col':             'tau1',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$\tau_1$',
        'desc':            'τ_1 (1-subjettiness); kT-seeded k-means axes, beta=1.',
    },
    'tau2': {
        'col':             'tau2',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$\tau_2$',
        'desc':            'τ_2 (2-subjettiness).',
    },
    'tau3': {
        'col':             'tau3',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$\tau_3$',
        'desc':            'τ_3 (3-subjettiness).',
    },

    # ── dPhiMETclose: signed Δφ(closest jet to MET, MET) ─────────────────────
    # Signed angle ∈ (−π, π]; abs_value maps to [0, π), then boxcox.
    'dPhiMETclose': {
        'col':             'dPhiMETclose',
        'pipeline':        [('abs_value', {}), ('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$|\Delta\phi|$(MET, close jet)',
        'desc':            'Signed Δφ(MET, closest jet); abs_value applied first.  Can be 0.',
    },

    # ── dPhiMETfar: signed Δφ(farthest jet to MET, MET) ──────────────────────
    'dPhiMETfar': {
        'col':             'dPhiMETfar',
        'pipeline':        [('abs_value', {}),('affine_flip', {'a':np.pi}), ('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$|\Delta\phi|$(MET, far jet)',
        'desc':            'Signed Δφ(MET, farthest jet); abs_value applied first.',
    },

    # ── nJets: number of jets passing selection ───────────────────────────────
    # Integer count ≥ 1.  BoxCox on integers is mathematically valid but unusual.
    'nJets': {
        'col':             'nJets',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$N_\mathrm{jets}$',
        'desc':            'Integer jet multiplicity.  BoxCox on counts is atypical but valid.',
    },

    # ── closeJetIsLead: is the closest-to-MET jet the leading-pT jet? ─────────
    # Binary (0 or 1).  Not suitable for continuous transforms.
    'closeJetIsLead': {
        'col':             'closeJetIsLead',
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'Close jet = lead',
        'desc':            'Binary: 1 if the jet closest to MET is also the leading-pT jet.',
    },

    # ── nInvClose: number of dark particles in the closest jet ────────────────
    # Integer count ≥ 0; often 0.  Not suitable for BoxCox without shifting.
    'nInvClose': {
        'col':             'nInvClose',
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'$N_\mathrm{dark}$ (close jet)',
        'desc':            'Dark pion/rho count in the closest-to-MET jet.  Often 0.',
    },

    # ── metPhi: azimuthal angle of MET ────────────────────────────────────────
    # Angle ∈ (−π, π].  abs_value maps to [0, π]; boxcox requires > 0,
    # so φ=0 events are discarded.
    'metPhi': {
        'col':             'metPhi',
        'pipeline':        [('abs_value', {}), ('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$|\phi_\mathrm{MET}|$',
        'desc':            'MET azimuthal angle; abs_value first.  φ=0 events discarded.',
    },

    # ── Derived: hemiMass2 / hemiMass1 ────────────────────────────────────────
    # Not a TSV column; computed from samples.  GUI-only; not regressionable.
    'mass2/mass1': {
        'col':             None,
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'$m_2 / m_1$',
        'desc':            'Derived: hemiMass2 / hemiMass1.  GUI display only.',
        'derive_cols':     ('hemiMass2', 'hemiMass1'),
    },

    # ── Derived: e3c / e2c ────────────────────────────────────────────────────
    'e3c/e2c': {
        'col':             None,
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'$e_3^c / e_2^c$',
        'desc':            'Derived: e3c / e2c.  GUI display only.',
        'derive_cols':     ('e3c', 'e2c'),
    },

    # ── Derived: tau2 / tau1 ──────────────────────────────────────────────────
    'tau2/tau1': {
        'col':             None,
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'$\tau_2 / \tau_1$',
        'desc':            'Derived: tau2 / tau1.  GUI display only.',
        'derive_cols':     ('tau2', 'tau1'),
    },

    # ── Derived: tau3 / tau2 ──────────────────────────────────────────────────
    'tau3/tau2': {
        'col':             None,
        'pipeline':        None,
        'distribution':    None,
        'default_include': False,
        'label':           r'$\tau_3 / \tau_2$',
        'desc':            'Derived: tau3 / tau2.  GUI display only.',
        'derive_cols':     ('tau3', 'tau2'),
    },
}

# Default set of observables used in the regression scan.
# Edit this list (or set default_include=True in the dict above) to change what
# gets regressed.  Every name here must have pipeline != None.
DEFAULT_SCAN = [name for name, spec in OBSERVABLES.items() if spec['default_include']]


# ══════════════════════════════════════════════════════════════════════════════
# TSV loader
# ══════════════════════════════════════════════════════════════════════════════

def load_tsv(path):
    """
    Load a TSV written by svj_regression and build a column-name→index map.

    The first line must be `# name0\\tname1\\t...` (the header written by the
    C++ binary from OBS_NAMES).  Subsequent lines are numerical data rows.

    Parameters
    ----------
    path : str or Path

    Returns
    -------
    data : np.ndarray, shape (N, n_cols)
    col_map : dict[str, int]
        Observable name → column index in data.
    """
    path = Path(path)
    col_map = {}
    with open(path) as fh:
        header = fh.readline().rstrip('\n')
    if header.startswith('#'):
        names = header.lstrip('#').strip().split('\t')
        col_map = {name: i for i, name in enumerate(names)}
    data = np.loadtxt(path, comments='#')
    return data, col_map


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline utility functions
# ══════════════════════════════════════════════════════════════════════════════

def n_fitted_params(obs_name):
    """
    Total number of data-fitted parameters for one observable
    (sum over transform n_fitted + distribution n_params).
    """
    obs  = OBSERVABLES[obs_name]
    pipe = obs['pipeline'] or []
    dist = obs['distribution']
    n  = sum(TRANSFORMS[t]['n_fitted'] for t, _ in pipe)
    n += DISTRIBUTIONS[dist]['n_params'] if dist else 0
    return n


def param_offsets(obs_selection):
    """
    Cumulative start indices into a flat parameter vector.

    Parameters
    ----------
    obs_selection : list of str

    Returns
    -------
    offsets : np.ndarray, shape (len(obs_selection) + 1,)
        offsets[i] is the start of obs_selection[i]'s parameters;
        offsets[-1] is the total number of parameters.
    """
    sizes   = [n_fitted_params(name) for name in obs_selection]
    return np.concatenate([[0], np.cumsum(sizes)]).astype(int)


def event_valid_mask(X_raw, obs_selection, col_map=None):
    """
    Per-event range check across all selected observables.

    For each observable and each step in its pipeline, verifies that the
    (partially-transformed) value satisfies the transform's required input range.
    Events failing any check for any observable are marked invalid.

    Parameters
    ----------
    X_raw : np.ndarray, shape (N, n_cols)
        Raw observable values in TSV column order.
    obs_selection : list of str
        Names from OBSERVABLES to check.
    col_map : dict[str, int] or None
        Mapping from observable name to column index in X_raw, as returned by
        load_tsv().  Pass None only if OBSERVABLES 'col' values are already ints
        (legacy usage).

    Returns
    -------
    mask : np.ndarray, shape (N,), dtype bool
        True for events that pass all range checks.
    n_discarded_per_obs : np.ndarray, shape (len(obs_selection),), dtype int
        Number of events discarded due to each observable (cumulative with mask
        from prior observables to reflect the marginal contribution of each).
    """
    mask = np.ones(len(X_raw), dtype=bool)
    n_disc = np.zeros(len(obs_selection), dtype=int)

    for col_idx, obs_name in enumerate(obs_selection):
        obs      = OBSERVABLES[obs_name]
        pipeline = obs['pipeline'] or []
        col_name = obs['col']
        if col_name is None:
            continue

        tsv_col  = col_map[col_name] if col_map is not None else col_name
        col_data  = X_raw[:, tsv_col].copy()
        col_mask  = np.ones(len(col_data), dtype=bool)

        for t_name, t_fixed in pipeline:
            t = TRANSFORMS[t_name]
            if t['requires'] is not None:
                lo, hi = t['requires'](**t_fixed)
                if lo > -np.inf:
                    col_mask &= col_data > lo
                if hi < np.inf:
                    col_mask &= col_data < hi

            # Advance col_data through non-fitted (fixed) transforms for the
            # next range check.  Fitted transforms (boxcox) are always last in
            # the pipeline so we stop after checking their input range.
            if t['n_fitted'] == 0:
                valid_now = col_mask & mask
                col_data[valid_now] = t['forward'](col_data[valid_now], (), **t_fixed)
            # else: fitted transform → input range just checked; stop advancing

        n_disc[col_idx] = int((mask & ~col_mask).sum())
        mask &= col_mask

    return mask, n_disc


def fit_observable_col(x_col, pipeline, dist_name):
    """
    Apply the full transform pipeline to a 1-D data array and fit the chosen
    distribution.  Returns standard-normal mapped values and all fitted params.

    Pipeline
    --------
    1. Apply each invertible transform in order, collecting fitted params.
    2. Fit the chosen distribution (MLE).
    3. Map: dist CDF → uniform → norm.ppf → standard normal.

    Parameters
    ----------
    x_col : np.ndarray, shape (N,)
        Pre-filtered observable values (all entries must satisfy range checks).
    pipeline : list of (str, dict)
        Ordered list of (transform_name, fixed_params).
    dist_name : str
        Key in DISTRIBUTIONS.

    Returns
    -------
    y_std : np.ndarray, shape (N,)
        Standard-normal mapped values (probit-transformed CDF outputs).
    fitted_params : tuple of float
        All data-fitted parameters: transform params (in pipeline order) then
        distribution params.
    """
    y = x_col.copy()
    fitted = []

    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        y, t_params = t['fit'](y, **t_fixed)
        fitted.extend(t_params)

    dist_spec = DISTRIBUTIONS[dist_name]
    args, kwargs = dist_spec['fit_init'](y)
    dist_fitted  = dist_spec['dist'].fit(y, *args, **kwargs)
    fitted.extend(dist_fitted)

    u     = np.clip(dist_spec['dist'].cdf(y, *dist_fitted), 1e-10, 1.0 - 1e-10)
    y_std = st.norm.ppf(u)

    return y_std, tuple(float(p) for p in fitted)


def forward_observable_col(x_col, pipeline, dist_name, fitted_params):
    """
    Apply forward pipeline with known params to map original-space values to
    standard-normal space.

    Parameters
    ----------
    x_col : np.ndarray
    pipeline, dist_name : as in fit_observable_col
    fitted_params : tuple/array of float
        Parameters in the same order as returned by fit_observable_col.

    Returns
    -------
    y_std : np.ndarray
    """
    params = list(fitted_params)
    y = x_col.copy()
    pos = 0

    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        n = t['n_fitted']
        t_params = tuple(params[pos:pos + n])
        pos += n
        y = t['forward'](y, t_params, **t_fixed)

    dist_spec   = DISTRIBUTIONS[dist_name]
    dist_params = tuple(params[pos:pos + dist_spec['n_params']])
    u    = np.clip(dist_spec['dist'].cdf(y, *dist_params), 1e-10, 1.0 - 1e-10)
    return st.norm.ppf(u)


def inverse_observable_col(u, pipeline, dist_name, fitted_params):
    """
    Apply inverse pipeline: uniform quantiles → original observable space.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Uniform values in (0, 1).  These are the MVT marginal CDF outputs.
    pipeline, dist_name, fitted_params : as in forward_observable_col.

    Returns
    -------
    x : np.ndarray, shape (N,)
        Values in the original observable space.
    """
    params = list(fitted_params)
    pos    = 0

    # Collect transform params in forward order
    t_params_list = []
    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        n = t['n_fitted']
        t_params_list.append((t_name, t_fixed, tuple(params[pos:pos + n])))
        pos += n

    dist_spec   = DISTRIBUTIONS[dist_name]
    dist_params = tuple(params[pos:pos + dist_spec['n_params']])

    # Distribution quantile: uniform → transform space
    y = dist_spec['dist'].ppf(u, *dist_params)

    # Inverse transforms in reverse order
    for t_name, t_fixed, t_params in reversed(t_params_list):
        t = TRANSFORMS[t_name]
        y = t['inverse'](y, t_params, **t_fixed)

    return y


def validate_scan_selection(obs_selection):
    """
    Raise ValueError if any observable in obs_selection cannot be regressed
    (pipeline is None or distribution is None).
    """
    bad = [n for n in obs_selection
           if OBSERVABLES[n]['pipeline'] is None or OBSERVABLES[n]['distribution'] is None]
    if bad:
        raise ValueError(
            f"Observable(s) {bad} have no regression pipeline defined.  "
            "Set pipeline and distribution in OBSERVABLES, or remove from selection.")
