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
    # Reconstruct the signed distribution by alternating signs.
    # Valid only when the underlying signed distribution is symmetric around 0
    # (sign and |y| are independent), which holds for all angular observables
    # currently using this transform.  Do not use for asymmetric quantities.
    y = np.asarray(y, dtype=float).copy()
    y[1::2] *= -1.0
    return y

def _boxcox_fit(x):
    y, lam = st.boxcox(x)
    return y, (float(lam),)

def _boxcox_fwd(x, params):
    return sp.boxcox(x, params[0])

def _boxcox_inv(y, params):
    return sp.inv_boxcox(y, params[0])

def _identity_fit(x):
    return np.asarray(x, dtype=float).copy(), ()

def _identity_fwd(x, params):
    return np.asarray(x, dtype=float).copy()

def _identity_inv(y, params):
    return np.asarray(y, dtype=float).copy()


TRANSFORMS = {
    # ── Identity / no-op  (use to inspect raw distributions) ────────────────
    'identity': {
        'fit':      _identity_fit,
        'forward':  _identity_fwd,
        'inverse':  _identity_inv,
        'n_fitted': 0,
        'requires': None,
        'desc':     'Identity (no-op).  Use pipeline=[("identity",{})] to inspect raw distributions.',
    },
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
    # ── absolute value  (invertible under symmetry assumption) ──────────────
    # The inverse alternates signs on the sample array.  Only correct when the
    # signed distribution is symmetric around 0 (sign ⊥ |x|).  True for all
    # angular observables currently using this transform.
    'abs_value': {
        'fit':      _abs_fit,
        'forward':  _abs_fwd,
        'inverse':  _abs_inv,
        'n_fitted': 0,
        'requires': None,
        'desc':     '|x|  (inverse alternates signs; assumes symmetric signed distribution)',
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
    # ── Burr Type XII  (c, d, loc, scale) ────────────────────────────────────
    # Heavy-tailed distribution with a mode at a positive x when c > 1; good
    # for pT-like spectra that rise from a hard threshold then fall off.
    # loc is fixed just below the data minimum rather than at 0: fixing at 0
    # forces MLE to put mass in [0, pT_cut] where there is no data, pushing
    # c → ≤1 (monotone decrease).  Anchoring loc near x.min() lets c > 1
    # freely and the distribution captures the rise-then-fall shape.
    'burr12': {
        'dist':        st.burr12,
        'n_params':    4,
        'fit_init':    lambda x: ((2.0, 2.0), {'floc': max(0.0, float(x.min()) - 1e-3)}),
        'input_range': (0.0, np.inf),
        'desc':        'Burr Type XII; shapes c, d, loc fixed just below data min, scale.',
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
    # Often zero in SVJ events (electrons are rare).  With point_mass enabled,
    # the zero-events are modelled as a discrete atom rather than discarded.
    # Set point_mass=None to revert to the original behaviour (discard zeros).
    'maxElePt': {
        'col':             'maxElePt',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'Max $e$ $p_T$ (GeV)',
        'desc':            'Max final-state electron pT.  Often 0; mixture model retains those events.',
        'point_mass':      {'value': 0.0, 'tol': 1e-10, 'symmetric': False, 'min_p0': 0.001},
    },

    # ── maxMuPt: maximum muon pT ──────────────────────────────────────────────
    # Zero when no muon is produced; fraction depends on Brmu.  The mixture
    # model captures this discrete atom.  Set point_mass=None to revert to
    # the original behaviour (discard zero-events before Box-Cox).
    'maxMuPt': {
        'col':             'maxMuPt',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Max $\mu$ $p_T$ (GeV)',
        'desc':            'Max final-state muon pT.  Often 0; mixture model retains those events.',
        'point_mass':      {'value': 0.0, 'tol': 1e-10, 'symmetric': False, 'min_p0': 0.001},
    },

    # ── jetThrust: jet thrust (stored as T; regressed as 1−T) ─────────────────
    # The raw TSV value is jetThrust ∈ [0,1].  The affine_flip maps it to
    # 1−thrust ∈ (0,1], which is then Box-Cox transformed.
    'jetThrust': {
        'col':             'jetThrust',
        'pipeline':        [('affine_flip', {'a': 1.0}), ('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Jet thrust',
        'desc':            'Leading-jet thrust T; regressed as 1−T after flipping.',
    },

    # ── transSphericity: event-level transverse sphericity ────────────────────
    # S_T ∈ [0,1]; can be exactly 0 (perfectly pencil-like events), which
    # breaks Box-Cox.  Excluded by default; can be included if zero-events are
    # rare at your parameter point.
    'transSphericity': {
        'col':             'transSphericity',
        'pipeline':        [('identity', {})],
        'distribution':    'beta',
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
        'default_include': False,
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
    # Symmetric around 0, so abs_value inverse (alternating signs) is valid.
    # Optional mixture at ±π (jet exactly opposite to MET): symmetric=True so
    # both raw +π and −π are captured as one point mass.  These events survive
    # the current pipeline (abs(±π)=π > 0) but can pile up near the upper
    # boundary.  Set point_mass=None to fit the full distribution as continuous.
    'dPhiMETclose': {
        'col':             'dPhiMETclose',
        'pipeline':        [('abs_value', {}),('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$|\Delta\phi|$(MET, close jet)',
        'desc':            'Signed Δφ(MET, closest jet); abs_value applied first.  Can be 0.',
        'point_mass':      {'value': np.pi, 'tol': 1e-1, 'symmetric': True, 'min_p0': 0.001},
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

    # ── HT: scalar pT sum of all passing jets ─────────────────────────────────
    'HT': {
        'col':             'HT',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$H_T$ (GeV)',
        'desc':            'Scalar sum of visible pT of all jets passing selection.',
    },

    # ── RT: MET / HT ──────────────────────────────────────────────────────────
    'RT': {
        'col':             'RT',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$R_T$',
        'desc':            'MET / H_T; key SVJ discriminant (Cohen et al. 2015).',
        'point_mass':      {'value': 1.0, 'tol': 1e-10, 'symmetric': False, 'min_p0': 0.01},
    },

    # ── Meff: effective mass ───────────────────────────────────────────────────
    'Meff': {
        'col':             'Meff',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$M_\mathrm{eff}$ (GeV)',
        'desc':            'Effective mass: H_T + MET.',
    },

    # ── leadJetMass: leading jet invariant mass (visible+muon constituents) ───
    'leadJetMass': {
        'col':             'leadJetMass',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'Lead jet mass (GeV)',
        'desc':            'Invariant mass of visible+muon constituents of the leading jet.',
    },

    # ── nConst: visible+muon constituent multiplicity of leading jet ───────────
    'nConst': {
        'col':             'nConst',
        'pipeline':        [('boxcox', {})],
        'distribution':    'gennorm',
        'default_include': True,
        'label':           r'$N_\mathrm{const}$',
        'desc':            'Visible+muon constituent multiplicity of the leading jet.',
    },

    # ── fInv: invisible pT fraction of leading jet ────────────────────────────
    # Invisible = neutrinos + dark pions/rhos, matched geometrically (ΔR < jetR)
    # to the visible-only jet axis.  fInv = inv_pT / (vis_pT + inv_pT).
    # fInv=0 (fully visible jet) is physically meaningful; the mixture model
    # captures this discrete atom.  Set point_mass=None for a pure continuous fit.
    'fInv': {
        'col':             'fInv',
        'pipeline':        [('identity', {})],
        'distribution':    'gennorm',
        'default_include': False,
        'label':           r'$f_\mathrm{inv}$',
        'desc':            'Invisible pT fraction of leading jet: (ν + dark pT) / total leading jet pT.',
        'point_mass':      {'value': 0.0, 'tol': 2e-2, 'symmetric': False, 'min_p0': 0.01},
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
DEFAULT_SCAN = [name for name, spec in OBSERVABLES.items()
                if spec['default_include'] and spec['col'] is not None]


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

def _check_point_mass(x, pm_spec):
    """
    Boolean mask: True where x matches the point-mass spec.

    Parameters
    ----------
    x : np.ndarray, shape (N,)
        Raw (pre-pipeline) observable values.
    pm_spec : dict or None
        The 'point_mass' entry from an OBSERVABLES spec.
        Keys: value (float), tol (float, default 1e-10),
              symmetric (bool, default False).
        None → all-False mask.
    """
    if pm_spec is None:
        return np.zeros(len(x), dtype=bool)
    val = float(pm_spec['value'])
    tol = float(pm_spec.get('tol', 1e-10))
    m   = np.abs(x - val) < tol
    if pm_spec.get('symmetric', False):
        m |= np.abs(x + val) < tol
    return m


def n_fitted_params(obs_name):
    """
    Total number of data-fitted parameters for one observable.

    Layout when 'point_mass' is not None:
      (p0, *transform_fitted_params_in_order, *dist_params)
    Layout otherwise:
      (*transform_fitted_params_in_order, *dist_params)
    """
    obs  = OBSERVABLES[obs_name]
    pipe = obs['pipeline'] or []
    dist = obs['distribution']
    n  = sum(TRANSFORMS[t]['n_fitted'] for t, _ in pipe)
    n += DISTRIBUTIONS[dist]['n_params'] if dist else 0
    if obs.get('point_mass') is not None:
        n += 1   # p0: point-mass fraction, stored as first parameter
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

        col_data_raw = col_data.copy()      # frozen raw values for point-mass check
        pm_spec      = obs.get('point_mass')

        for t_name, t_fixed in pipeline:
            t = TRANSFORMS[t_name]
            if t['requires'] is not None:
                lo, hi = t['requires'](**t_fixed)
                range_ok = np.ones(len(col_data), dtype=bool)
                if lo > -np.inf:
                    range_ok &= col_data > lo
                if hi < np.inf:
                    range_ok &= col_data < hi
                # Point-mass events are always checked against raw values and
                # bypass range restrictions (they are handled by the mixture model).
                pm_pass  = _check_point_mass(col_data_raw, pm_spec)
                col_mask &= (range_ok | pm_pass)

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


def fit_observable_col(x_col, pipeline, dist_name, point_mass=None, rng=None):
    """
    Apply the full transform pipeline to a 1-D data array and fit the chosen
    distribution.  Returns standard-normal mapped values and all fitted params.

    When point_mass is not None, a mixture model is used:
      - Events within tolerance of the boundary value are separated out.
      - p0 (their fraction) is prepended to fitted_params as the first element.
      - The continuous subset is transformed and fitted as usual.
      - Continuous events are mapped via the rescaled CDF: u = p0 + (1-p0)*F(y).
      - Point-mass events are mapped via a randomized PIT: u ~ Uniform(0, p0).
      - If p0 < point_mass['min_p0'], the mixture is skipped (p0 still stored).

    Parameters
    ----------
    x_col : np.ndarray, shape (N,)
        Pre-filtered observable values.
    pipeline : list of (str, dict)
        Ordered list of (transform_name, fixed_params).
    dist_name : str
        Key in DISTRIBUTIONS.
    point_mass : dict or None
        Spec from OBSERVABLES 'point_mass' key.  When not None, p0 is prepended
        to fitted_params.  Keys: value, tol, symmetric, min_p0.
    rng : numpy Generator or None
        RNG for the randomized PIT of point-mass events.  None → global state.

    Returns
    -------
    y_std : np.ndarray, shape (N,)
        Standard-normal mapped values.
    fitted_params : tuple of float
        Layout when point_mass is not None:
          (p0, *transform_params_in_pipeline_order, *dist_params)
        Layout when point_mass is None:
          (*transform_params_in_pipeline_order, *dist_params)
    """
    fitted = []

    # ── Point-mass separation ────────────────────────────────────────────────
    p0 = 0.0
    if point_mass is not None:
        pm_mask = _check_point_mass(x_col, point_mass)
        p0_raw  = float(pm_mask.sum()) / len(x_col) if len(x_col) > 0 else 0.0
        if p0_raw >= 1.0:
            raise ValueError(
                f"All {len(x_col)} events match the point mass "
                f"(value={point_mass['value']}, p0={p0_raw:.3f}).  "
                "Cannot fit a continuous distribution on an empty subset.")
        p0 = p0_raw   # store actual fraction; forward/inverse threshold at min_p0
        fitted.append(p0)

    pm_active = (point_mass is not None) and (p0 >= point_mass.get('min_p0', 0.01))

    # ── Continuous subset ─────────────────────────────────────────────────────
    # Always separate PM events from the pipeline to avoid e.g. boxcox(0) failing.
    if point_mass is not None:
        y = x_col[~pm_mask].copy()
    else:
        y = x_col.copy()

    # ── Transforms + distribution fit ────────────────────────────────────────
    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        y, t_params = t['fit'](y, **t_fixed)
        fitted.extend(t_params)

    dist_spec = DISTRIBUTIONS[dist_name]
    args, kwargs = dist_spec['fit_init'](y)
    dist_fitted  = dist_spec['dist'].fit(y, *args, **kwargs)
    fitted.extend(dist_fitted)

    # ── CDF → standard normal ─────────────────────────────────────────────────
    u_cont = np.clip(dist_spec['dist'].cdf(y, *dist_fitted), 1e-10, 1.0 - 1e-10)

    if pm_active:
        # Rescale continuous CDF into [p0, 1]
        u_rescaled = np.clip(p0 + (1.0 - p0) * u_cont, 1e-10, 1.0 - 1e-10)
        y_std  = np.empty(len(x_col))
        n_pm   = int(pm_mask.sum())
        _draw  = rng.uniform if rng is not None else np.random.uniform
        u_pm   = np.clip(_draw(0.0, p0, n_pm), 1e-10, p0 - 1e-10)
        y_std[pm_mask]  = st.norm.ppf(u_pm)
        y_std[~pm_mask] = st.norm.ppf(u_rescaled)
    elif point_mass is not None and pm_mask.any():
        # PM spec present but fraction below min_p0: place PM events at their
        # expected CDF rank (midpoint of the tiny [0, p0] interval).
        y_std = np.empty(len(x_col))
        y_std[~pm_mask] = st.norm.ppf(u_cont)
        y_std[pm_mask]  = st.norm.ppf(max(p0 / 2.0, 1e-10))
    else:
        y_std = st.norm.ppf(u_cont)

    return y_std, tuple(float(p) for p in fitted)


def forward_observable_col(x_col, pipeline, dist_name, fitted_params,
                           point_mass=None):
    """
    Apply forward pipeline with known params to map original-space values to
    standard-normal space.

    When point_mass is not None, p0 is read from fitted_params[0] and the
    mixture CDF is applied.  Point-mass events receive the deterministic
    midpoint value norm.ppf(p0/2) (unlike fit, which uses a randomized PIT).

    Parameters
    ----------
    x_col : np.ndarray
    pipeline, dist_name : as in fit_observable_col
    fitted_params : tuple/array of float
    point_mass : dict or None
        Spec from OBSERVABLES 'point_mass' key.

    Returns
    -------
    y_std : np.ndarray
    """
    params = list(fitted_params)
    pos    = 0

    p0 = 0.0
    if point_mass is not None:
        p0  = float(params[0])
        pos = 1

    pm_active = (point_mass is not None) and (p0 >= point_mass.get('min_p0', 0.01))

    # Separate PM events from pipeline input (prevents e.g. boxcox(0) failing)
    if point_mass is not None:
        pm_mask = _check_point_mass(x_col, point_mass)
        y = x_col[~pm_mask].copy()
    else:
        pm_mask = None
        y = x_col.copy()

    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        n = t['n_fitted']
        t_params = tuple(params[pos:pos + n])
        pos += n
        y = t['forward'](y, t_params, **t_fixed)

    dist_spec   = DISTRIBUTIONS[dist_name]
    dist_params = tuple(params[pos:pos + dist_spec['n_params']])
    u = np.clip(dist_spec['dist'].cdf(y, *dist_params), 1e-10, 1.0 - 1e-10)

    if pm_active:
        u_rescaled = np.clip(p0 + (1.0 - p0) * u, 1e-10, 1.0 - 1e-10)
        y_std = np.empty(len(x_col))
        y_std[pm_mask]  = st.norm.ppf(p0 / 2.0)   # deterministic midpoint
        y_std[~pm_mask] = st.norm.ppf(u_rescaled)
        return y_std
    elif point_mass is not None and pm_mask.any():
        y_std = np.empty(len(x_col))
        y_std[~pm_mask] = st.norm.ppf(u)
        y_std[pm_mask]  = st.norm.ppf(max(p0 / 2.0, 1e-10))
        return y_std
    else:
        return st.norm.ppf(u)


def inverse_observable_col(u, pipeline, dist_name, fitted_params,
                           point_mass=None):
    """
    Apply inverse pipeline: uniform quantiles → original observable space.

    When point_mass is not None and p0 >= min_p0, samples with u <= p0 are
    mapped to the boundary value directly; samples with u > p0 have their
    uniform quantile rescaled to (0,1) before passing through the standard
    inverse pipeline.

    Parameters
    ----------
    u : np.ndarray, shape (N,)
        Uniform values in (0, 1).  These are the copula marginal CDF outputs.
    pipeline, dist_name, fitted_params : as in forward_observable_col.
    point_mass : dict or None
        Spec from OBSERVABLES 'point_mass' key.

    Returns
    -------
    x : np.ndarray, shape (N,)
        Values in the original observable space.
    """
    params = list(fitted_params)
    pos    = 0

    p0 = 0.0
    if point_mass is not None:
        p0  = float(params[0])
        pos = 1

    pm_active = (point_mass is not None) and (p0 >= point_mass.get('min_p0', 0.01))

    # Collect transform params in forward order
    t_params_list = []
    for t_name, t_fixed in (pipeline or []):
        t = TRANSFORMS[t_name]
        n = t['n_fitted']
        t_params_list.append((t_name, t_fixed, tuple(params[pos:pos + n])))
        pos += n

    dist_spec   = DISTRIBUTIONS[dist_name]
    dist_params = tuple(params[pos:pos + dist_spec['n_params']])

    def _invert_continuous(u_c):
        y = dist_spec['dist'].ppf(u_c, *dist_params)
        for t_name, t_fixed, t_params in reversed(t_params_list):
            t = TRANSFORMS[t_name]
            y = t['inverse'](y, t_params, **t_fixed)
        return y

    if not pm_active:
        return _invert_continuous(u)

    # ── Mixture inverse ───────────────────────────────────────────────────────
    pm_u_mask = (u <= p0)
    x         = np.empty(len(u))

    # Point-mass events: return the boundary value (±value if symmetric)
    pm_val    = float(point_mass['value'])
    n_pm      = int(pm_u_mask.sum())
    if point_mass.get('symmetric', False) and n_pm > 0:
        signs        = np.where(np.random.random(n_pm) < 0.5, 1.0, -1.0)
        x[pm_u_mask] = signs * pm_val
    else:
        x[pm_u_mask] = pm_val

    # Continuous events: rescale u from [p0, 1] → (0, 1) then invert
    if (~pm_u_mask).any():
        u_rescaled      = np.clip((u[~pm_u_mask] - p0) / (1.0 - p0),
                                  1e-10, 1.0 - 1e-10)
        x[~pm_u_mask]  = _invert_continuous(u_rescaled)

    return x


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
