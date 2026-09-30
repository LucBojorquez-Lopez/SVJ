"""
normalisation.py -- turn shapes into rates, and rates into a significance.

The interpolated SVJ model and the background TSVs both give *shapes*: they say
what fraction of events land where, not how many events there are.  Everything
needed to attach an absolute number lives here.

Two independent normalisations meet:

  signal      sigma x BR(Z'->dark), measured with PYTHIA at the production grid
              masses and stored in signal/xsec.json.  Rescaled to any coupling
              by sigma ~ g_q^2 (see signal_sigma_pb).
  background  per-sample cross sections in background/samples.json, measured the
              same way, turned into a per-event weight in fb.

See docs/normalisation.md for the physics and the measurements behind both.
"""

import json
import warnings
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

SIGNAL_XSEC_JSON = _REPO_ROOT / 'signal' / 'xsec.json'
BACKGROUND_JSON = _REPO_ROOT / 'background' / 'samples.json'
BACKGROUND_TSV_DIR = Path('/eos/user/l/lbojorqu/svj/background/tsv')
# Three places a background cache can live, tried in this order.
#
#   REPO  a full cache someone dropped into their checkout.  Gitignored: it is
#         ~1.3 GB, far too large to version.
#   EOS   the canonical full cache, next to the shards.  Only reachable at CERN.
#   DEMO  a thinned cache that IS committed (~28 MB), so a fresh clone can drive
#         the GUI without either regenerating 17M events (~57 core-hours) or
#         having EOS access.
#
# The demo is last on purpose: anyone who has the real thing gets the real
# thing.  Its numbers are NOT publication-grade -- thinning inflates the MC
# error by roughly sqrt(thinning factor), which is why it is off by default for
# analysis (docs/normalisation.md M10).  load_background() records which file it
# used in LOADED_CACHE_PATH / LOADED_CACHE_IS_DEMO so callers can say so; the
# GUI shows a banner.
BACKGROUND_CACHE_REPO = _REPO_ROOT / 'background' / 'background_events.npz'
BACKGROUND_CACHE_EOS = Path(
    '/eos/user/l/lbojorqu/svj/background/background_events.npz')
BACKGROUND_CACHE_DEMO = _REPO_ROOT / 'background' / 'background_events_demo.npz'

LOADED_CACHE_PATH = None
LOADED_CACHE_IS_DEMO = False


def _resolve_cache():
    for c in (BACKGROUND_CACHE_REPO, BACKGROUND_CACHE_EOS,
              BACKGROUND_CACHE_DEMO):
        if c.exists():
            return c
    return BACKGROUND_CACHE_EOS      # for the error message


BACKGROUND_CACHE = _resolve_cache()

PB_TO_FB = 1.0e3

# Luminosity presets in fb^-1.  Run 2 delivered ~140; Run 3 targets ~300;
# HL-LHC ~3000.
LUMI_PRESETS = {'Run 2 (140)': 140.0, 'Run 3 (300)': 300.0, 'HL-LHC (3000)': 3000.0}


# ── Signal ────────────────────────────────────────────────────────────────────

_XSEC_CACHE = {}


def load_signal_xsec(path=None):
    """Load and cache signal/xsec.json."""
    path = Path(path or SIGNAL_XSEC_JSON)
    key = str(path)
    if key not in _XSEC_CACHE:
        doc = json.loads(path.read_text())
        pts = sorted(doc['points'], key=lambda r: r['mZ'])
        doc['_ln_mZ'] = np.log([r['mZ'] for r in pts])
        doc['_ln_sigma'] = np.log([r['sigma_pb'] for r in pts])
        _XSEC_CACHE[key] = doc
    return _XSEC_CACHE[key]


def _interp_ln_sigma(ln_mZ, doc):
    """
    PCHIP in (ln mZ, ln sigma).

    Measured against off-grid PYTHIA runs at 800/1400/2500/3500 GeV: PCHIP is
    accurate to 1.6% worst case, plain linear interpolation only to 4.4% and
    biased low throughout (docs/normalisation.md, M2).
    """
    try:
        from scipy.interpolate import PchipInterpolator
    except ImportError:                                   # pragma: no cover
        return np.interp(ln_mZ, doc['_ln_mZ'], doc['_ln_sigma'])
    if '_pchip' not in doc:
        doc['_pchip'] = PchipInterpolator(doc['_ln_mZ'], doc['_ln_sigma'])
    return doc['_pchip'](ln_mZ)


def br_dark(g_q, doc=None):
    """
    Dark branching ratio implied by g_q at the simulated width.

    Gamma(Z'->q qbar) per flavour is g_q^2 M / (4 pi), and the total width is
    forced to width_over_mass * M, so BR_SM = g_q^2 / (4 pi * width_over_mass)
    and BR_dark is whatever is left.  Goes negative above g_q_max_consistent,
    which is exactly where the configuration stops making sense.
    """
    doc = doc or load_signal_xsec()
    ref = doc['reference']
    br_sm = g_q ** 2 / (4.0 * np.pi * ref['width_over_mass'])
    return 1.0 - ref['n_sm_flavours'] * br_sm


def signal_sigma_pb(mZ, g_q=None, doc=None):
    """
    sigma x BR(Z' -> dark) in pb at this mZ' and coupling.

    With g_q=None the reference coupling is used, i.e. the cross section of the
    samples actually on disk.  Above reference['g_q_max_consistent'] the result
    is a benchmark extrapolation: the rate no longer corresponds to the 2.5%
    width the shapes were generated with.  Callers that surface the number to a
    user should say so -- see is_benchmark_only().
    """
    doc = doc or load_signal_xsec()
    ref = doc['reference']
    sigma = np.exp(_interp_ln_sigma(np.log(np.asarray(mZ, dtype=float)), doc))
    if g_q is None:
        return float(sigma) if np.isscalar(mZ) else sigma

    br_sm_ref = ref['sm_br_per_flavour']
    br_sm = g_q ** 2 / (4.0 * np.pi * ref['width_over_mass'])
    # sigma ~ BR_SM x BR_dark exactly (verified to 6 digits, M1).
    scale = (br_sm / br_sm_ref) * (br_dark(g_q, doc) / ref['br_dark'])
    out = sigma * max(scale, 0.0)
    return float(out) if np.isscalar(mZ) else out


def is_benchmark_only(g_q, doc=None):
    """True when g_q exceeds what the simulated width can support."""
    doc = doc or load_signal_xsec()
    return g_q > doc['reference']['g_q_max_consistent']


# ── Background ────────────────────────────────────────────────────────────────

def _read_tsv_fast(path):
    """Numeric rows of a TSV, without the header. pandas if present, else numpy."""
    try:
        import pandas as pd
        return pd.read_csv(path, sep='\t', comment='#', header=None,
                           dtype=np.float32).to_numpy()
    except ImportError:
        return np.loadtxt(path, comments='#', dtype=np.float32)


def build_background_cache(obs_names, tsv_dir=None, samples_json=None,
                           out=None, max_rows_per_sample=None, verbose=True):
    """
    Merge the per-shard background TSVs into one compact float32 NPZ.

    Slow (it parses every shard), so it is done once and the result committed to
    disk; the GUI only ever calls load_background().

    The per-event weight is sigma_fb / n_generated, where n_generated counts the
    events PYTHIA was asked for in the shards that actually landed -- not the
    nominal total in samples.json.  A sample missing 3 of 20 shards is thereby
    normalised correctly instead of coming out 15% low.  Rows dropped by the
    generator (no jets found) must NOT enter that denominator: they are real
    acceptance losses, and summing the surviving weights is what gives the
    post-acceptance cross section.
    """
    tsv_dir = Path(tsv_dir or BACKGROUND_TSV_DIR)
    samples = json.loads(Path(samples_json or BACKGROUND_JSON).read_text())
    out = Path(out or BACKGROUND_CACHE)

    chunks, weights, labels, per_sample = [], [], [], {}
    for name, spec in samples.items():
        shards = sorted(tsv_dir.glob(f'{name}_*.tsv'))
        if not shards:
            warnings.warn(f'background sample {name!r}: no shards found, skipping')
            continue
        n_gen = len(shards) * spec['n_event_per_job']
        if len(shards) != spec['n_jobs']:
            warnings.warn(
                f'background sample {name!r}: {len(shards)}/{spec["n_jobs"]} shards '
                f'present; normalising to the {n_gen:,} events actually generated')

        rows = []
        for s in shards:
            try:
                a = _read_tsv_fast(s)
            except Exception as exc:                      # a truncated shard
                warnings.warn(f'  unreadable shard {s.name}: {exc}')
                continue
            if a.ndim == 2 and len(a):
                rows.append(a)
        if not rows:
            continue
        X = np.vstack(rows)

        w = spec['sigma_pb'] * PB_TO_FB / n_gen           # fb per generated event
        n_before = len(X)
        if max_rows_per_sample and n_before > max_rows_per_sample:
            # Unbiased thinning: keep a random subset and scale the weight up by
            # exactly the factor discarded.  Raises the MC error, so it is off by
            # default -- the low-pT QCD slices can least afford it.
            keep = np.random.default_rng(0).choice(
                n_before, max_rows_per_sample, replace=False)
            X = X[keep]
            w *= n_before / max_rows_per_sample

        chunks.append(X.astype(np.float32))
        weights.append(np.full(len(X), w, dtype=np.float64))
        labels.append(np.full(len(X), len(per_sample), dtype=np.int16))
        per_sample[name] = {'n_shards': len(shards), 'n_generated': n_gen,
                            'n_rows': int(len(X)), 'n_rows_before_thin': int(n_before),
                            'weight_fb': float(w)}
        if verbose:
            print(f'  {name:20s} {len(shards):2d} shards  {len(X):8,d} rows  '
                  f'w={w:.4g} fb')

    if not chunks:
        raise RuntimeError(f'no background shards found under {tsv_dir}')

    np.savez_compressed(
        out,
        X=np.vstack(chunks), w_fb=np.concatenate(weights),
        sample_id=np.concatenate(labels),
        sample_names=np.array(list(per_sample), dtype=object),
        obs_names=np.array(list(obs_names), dtype=object),
        meta=np.array([json.dumps(per_sample)], dtype=object))
    if verbose:
        tot = sum(c.shape[0] for c in chunks)
        print(f'\n  wrote {out}  ({tot:,} events, '
              f'{out.stat().st_size / 1e6:.0f} MB)')
    return out


def load_background(path=None):
    """
    Return (X, w_fb, sample_id, sample_names, obs_names) from the cache.

    X is (N, n_obs) float32 in TSV column order; w_fb is the per-event weight in
    fb, so expected events at luminosity L [fb^-1] is L * w_fb.sum() over
    whichever rows pass a selection.
    """
    global LOADED_CACHE_PATH, LOADED_CACHE_IS_DEMO
    if path is None:
        path = _resolve_cache()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f'no background cache found. Tried:\n'
            f'  {BACKGROUND_CACHE_REPO}\n  {BACKGROUND_CACHE_EOS}\n'
            f'  {BACKGROUND_CACHE_DEMO}\n'
            f'Run build_background_cache() once after the Condor jobs finish '
            f'(see docs/normalisation.md), or pull the committed demo cache.')
    LOADED_CACHE_PATH = path
    LOADED_CACHE_IS_DEMO = (path.resolve() == BACKGROUND_CACHE_DEMO.resolve()
                            if BACKGROUND_CACHE_DEMO.exists() else False)
    z = np.load(path, allow_pickle=True)
    return (z['X'], z['w_fb'], z['sample_id'],
            list(z['sample_names']), list(z['obs_names']))


# ── Significance ──────────────────────────────────────────────────────────────

def counts(w_fb, mask, lumi_fb):
    """
    (N, err_N) for weighted events passing mask at this luminosity.

    err_N = L * sqrt(sum w^2) is the MC statistical error, which for weighted
    samples is NOT sqrt(N).  It matters here: the lowest QCD slice has an
    effective luminosity of 0.0007 fb^-1, so a signal region can rest on one or
    two events carrying enormous weight, and N alone would look far more
    precise than it is.
    """
    w = w_fb[mask]
    return lumi_fb * float(w.sum()), lumi_fb * float(np.sqrt((w ** 2).sum()))


# Bound on the PCHIP interpolation error between grid masses, as a fraction.
#
# Measured against independent PYTHIA runs at 800/1400/2500/3500 GeV, where the
# ratio measured/interpolated came out 1.0012, 0.9991, 0.9845, 0.9974 -- worst
# case 1.6%.  This is a BOUND, not a resolved measurement: those validation runs
# carry 0.83% MC error themselves, so three of the four deviations are within
# their own noise and only the 2500 GeV point is even marginally significant.
# Tightening it means more events per validation mass, not a cleverer spline.
SIGMA_INTERP_REL_BOUND = 0.016


def signal_sigma_rel_err_mc(mZ, doc=None):
    """
    Relative MC error on the tabulated sigma at this mass.

    Comes from the finite event count behind each grid point (n_event_per_point
    in signal/xsec.json).  Nearly flat at ~0.84% across the grid, but
    interpolated rather than assumed constant so a future re-run with uneven
    statistics stays correct.
    """
    doc = doc or load_signal_xsec()
    pts = sorted(doc['points'], key=lambda r: r['mZ'])
    m = np.array([r['mZ'] for r in pts])
    rel = np.array([r['err_pb'] / r['sigma_pb'] for r in pts])
    return float(np.interp(np.log(float(mZ)), np.log(m), rel))


def on_sigma_grid(mZ, doc=None, rtol=1e-6):
    """True when mZ coincides with a tabulated mass, where interpolation is exact."""
    doc = doc or load_signal_xsec()
    return any(abs(float(mZ) - r['mZ']) <= rtol * max(r['mZ'], 1.0)
               for r in doc['points'])


def signal_counts(sigma_pb, lumi_fb, n_pass, n_total, mZ=None, doc=None):
    """
    (S, components) from a cross section and a model-sample acceptance.

    S = sigma x L x eps, with eps = n_pass/n_total from the model sample.

    `components` breaks the uncertainty into pieces that are deliberately kept
    SEPARATE, because they are different kinds of thing and shrink for different
    reasons:

      acceptance    binomial error on eps.  A sample-size diagnostic -- it
                    shrinks by drawing more model samples and says nothing about
                    physics.  Its job is to make a hard selection with few
                    surviving draws visibly uncertain.
      sigma_mc      finite statistics behind the tabulated sigma.  Shrinks with
                    more events in signal/make_xsec.py.
      sigma_interp  PCHIP interpolation between grid masses; a bound, and
                    exactly zero when mZ sits on a tabulated mass.
      total         the three added in quadrature.  Approximate: sigma_interp is
                    a bound on a systematic, not a Gaussian sigma.

    NOT included, and much larger than any of the above: the LO K-factor.  The
    tabulated sigma is leading order, and NLO corrections for a Z' of this kind
    are tens of percent.  Nothing here estimates that, so `total` is a
    statistical-and-interpolation error, not a cross-section uncertainty.
    """
    comp = {'acceptance': 0.0, 'sigma_mc': 0.0, 'sigma_interp': 0.0,
            'total': 0.0, 'eps': 0.0}
    if n_total <= 0:
        return 0.0, comp
    eps = n_pass / n_total
    # sigma[pb] x PB_TO_FB -> fb, times L[fb^-1], gives a plain event count.
    s = sigma_pb * PB_TO_FB * lumi_fb * eps
    comp['eps'] = eps
    if s > 0:
        rel_acc = (np.sqrt(max(eps * (1.0 - eps), 0.0) / n_total) / eps
                   if eps > 0 else 0.0)
        rel_mc = signal_sigma_rel_err_mc(mZ, doc) if mZ is not None else 0.0
        rel_int = (0.0 if mZ is None or on_sigma_grid(mZ, doc)
                   else SIGMA_INTERP_REL_BOUND)
        comp['acceptance'] = s * rel_acc
        comp['sigma_mc'] = s * rel_mc
        comp['sigma_interp'] = s * rel_int
        comp['total'] = s * float(np.sqrt(rel_acc**2 + rel_mc**2 + rel_int**2))
    return float(s), comp


def background_diagnostics(w_fb, sample_id, sample_names, mask, lumi_fb):
    """
    The three numbers that say whether a background estimate can be believed.

    qcd_frac      fraction of B from QCD.  The dominant worry: QCD is the
                  component with no jet merging, no pileup, and MET that is
                  partly a jet-acceptance artefact and partly detector
                  mismeasurement (docs/normalisation.md M11).  Near 1.0 means
                  the answer rests almost entirely on the worst-modelled piece.
    mc_rel_err    sqrt(sum w^2)/sum w -- statistical only, and NOT sqrt(N) for
                  weighted events.
    worst_frac    largest single event's share of B.  Catches the case a small
                  relative error hides: B resting on one or two events.  The
                  |dPhi| region hit 10% here.
    n_eff         (sum w)^2 / sum w^2, the effective independent event count.
    """
    w = w_fb[mask]
    tot = float(w.sum())
    out = {'b': lumi_fb * tot, 'qcd_frac': 0.0, 'mc_rel_err': 0.0,
           'worst_frac': 0.0, 'n_eff': 0.0, 'n_rows': int(mask.sum())}
    if tot <= 0:
        return out
    sq = float((w ** 2).sum())
    out['mc_rel_err'] = np.sqrt(sq) / tot
    out['worst_frac'] = float(w.max()) / tot
    out['n_eff'] = tot ** 2 / sq
    qcd = [k for k, n in enumerate(sample_names) if str(n).startswith('qcd')]
    if qcd:
        out['qcd_frac'] = float(w[np.isin(sample_id[mask], qcd)].sum()) / tot
    return out


def significance(s, b, b_err=None, s_err=None):
    """
    Return dict with s_over_sqrt_b, asimov, and a flag for when to distrust them.

    s_over_sqrt_b is what was asked for and is the familiar number.  asimov is
    sqrt(2[(s+b)ln(1+s/b) - s]), which is the correct Poisson significance and
    reduces to s/sqrt(b) when s << b; the two diverge once the signal is not
    small, and s/sqrt(b) is the optimistic one there.

    unreliable is set when the background estimate rests on so little MC that
    the significance is not meaningful, regardless of which formula is used.
    """
    s = float(s)
    b = float(b)
    out = {'s': s, 'b': b, 'b_err': b_err, 'no_background': False}
    if b <= 0:
        # No background MC survived the selection.  S/sqrt(B) is formally
        # infinite, but that means "we cannot tell" rather than "infinitely
        # significant": the true background is unknown and merely unsampled,
        # and with a thinned cache this happens readily.  Callers must surface
        # `no_background` rather than printing inf, which reads as a result.
        out.update(s_over_sqrt_b=float('inf') if s > 0 else 0.0,
                   asimov=float('inf') if s > 0 else 0.0,
                   s_over_sqrt_b_err=float('nan'),
                   unreliable=True, no_background=True)
        return out
    out['s_over_sqrt_b'] = s / np.sqrt(b)
    out['asimov'] = float(np.sqrt(2.0 * ((s + b) * np.log1p(s / b) - s)))
    # Propagate both errors through Z = S/sqrt(B):
    #   dZ/Z = sqrt( (dS/S)^2 + (dB/(2B))^2 )
    # The B term carries the factor 1/2 because of the square root.
    rel = 0.0
    if s > 0 and s_err:
        rel += (s_err / s) ** 2
    if b > 0 and b_err:
        rel += (b_err / (2.0 * b)) ** 2
    out['s_over_sqrt_b_err'] = out['s_over_sqrt_b'] * np.sqrt(rel)
    # More than ~30% MC error on B means the number is noise-dominated.
    out['unreliable'] = bool(b_err is not None and b_err > 0.3 * b)
    return out
