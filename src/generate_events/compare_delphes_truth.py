#!/usr/bin/env python3
"""
compare_delphes_truth.py
=========================
Stage-0 sanity check: compare truth-level and Delphes-level observables
computed from the SAME Pythia events by svj_delphes_test.cc's unified
per-event driver (one paired TSV, columns suffixed _truth / _delphes).

This paired-event guarantee matters: svj_regression.cc and an independent
Delphes-only binary cannot be compared row-by-row -- svj_regression.cc's
usual nWorkers=16 config interleaves 16 different Pythia seeds in its
output, and each binary independently discards events with zero passing
jets, so two separately-generated TSVs' row indices do not correspond to the
same underlying event beyond, at best, a small and silently-decaying
prefix. svj_delphes_test.cc's paired driver sidesteps this entirely by
computing both sides from one event in one loop iteration.

Not part of the regression pipeline -- does not touch observables.py or
diagnostics.py beyond importing the existing, unmodified load_tsv() helper.
diagnostics.py's own plotting functions (plot_observable_transforms,
plot_validation*) are tightly coupled to the fitted-transform-pipeline and
validation-NPZ machinery (confirmed by reading src/diagnostics.py directly),
so there is no existing "overlay two raw samples" helper to reuse here --
this script draws its own minimal histograms instead.

The error-band histogram helper (_plot_hist_with_band) and the rcParams below
are a deliberate, exact copy of src/gui/svj_explorer.py's own
_plot_hist_with_band / plt.rcParams.update(...) -- not a reinterpretation.
svj_explorer.py's version is a private (leading-underscore) helper local to
the GUI module, not meant to be cross-imported, so it is mirrored here rather
than imported -- the same precedent diagnostics.py already set for its own
near-identical _PLOT_RCPARAMS copy.

Usage
-----
    # from the project root, after generating the paired TSV
    # (see docs/setup_delphes.md):
    python src/generate_events/compare_delphes_truth.py

    # custom path / output (three figures: marginal overlay, per-event diff,
    # per-event ratio):
    python src/generate_events/compare_delphes_truth.py \\
        --paired simulated/tsv/jets_delphes_paired.tsv \\
        --out simulated/tsv/truth_vs_delphes.png \\
        --out-diff simulated/tsv/truth_vs_delphes_diff.png \\
        --out-ratio simulated/tsv/truth_vs_delphes_ratio.png

    # find the 10 events where Delphes adds the most leadVisPt, and get a
    # dump_events list ready to paste into svj_delphes_test.cfg (see
    # plot_lego.py for the next step -- rendering their jet constituents):
    python src/generate_events/compare_delphes_truth.py --top-diff 10
"""

import sys
import argparse
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
from observables import load_tsv  # noqa: E402  (existing, unmodified helper)

# Base observable names -- the paired TSV has a "<name>_truth" and a
# "<name>_delphes" column for each of these.
SHARED_OBS = ["leadVisPt", "MET", "leadJetMass", "nConst", "nJets"]

# A third, independent reference series overlaid on specific marginal panels
# only (not part of the paired truth/delphes diff or ratio plots, since it
# has no Delphes-side counterpart to compare against): obs -> (column name,
# legend label, color). leadJetPt_truthfull is the leading jet re-clustered
# from ALL final-state truth particles (visible + invisible + dark pions/
# rhos), see svj_delphes_test.cc's buildFullTruthParticles() -- "what the
# true total leading jet pT is if you could see everything," for comparison
# against leadVisPt_truth (visible-only) and leadVisPt_delphes (reconstructed).
EXTRA_TRUTH_SERIES = {
    'leadVisPt': ('leadJetPt_truthfull', 'truth (+dark matter)', 'seagreen'),
}

# Expected sanity signature (see docs/setup_delphes.md "Verification"):
#   leadVisPt / leadJetMass -- broadened and mildly shifted by tracking/calo
#     resolution smearing.
#   MET -- a resolution tail not present in the crude truth "negative vector
#     sum of accepted-jet visible pT" proxy.
#   nConst -- systematically LOWER under Delphes: particle-flow candidates
#     merge/lose truth particles relative to infinite-resolution truth
#     constituents (calorimeter towers and tracking thresholds coarsen the
#     constituent count).
#   nJets -- the direct check for whether jets are migrating above
#     vis_jet_pt_min under Delphes; see the "zero-jet contingency" printout
#     below for the per-event version of this question.


# ── Exact copy of src/gui/svj_explorer.py's rcParams (not diagnostics.py's
#    near-duplicate _PLOT_RCPARAMS, which differs in legend.fontsize) ───────
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
    'legend.fontsize': 9,
}


def _summary(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "n=0 (all NaN -- no accepted jet on this side for any row)"
    q = np.percentile(x, [5, 50, 95])
    return f"n={len(x):6d}  mean={x.mean():10.3f}  std={x.std():9.3f}  " \
           f"p5={q[0]:9.3f}  median={q[1]:9.3f}  p95={q[2]:9.3f}"


# ── Histogram with multinomial error band ────────────────────────────────────
# Exact copy of svj_explorer.py's _plot_hist_with_band (src/gui/svj_explorer.py
# lines 251-275): each bin count is a binomial draw out of N total events with
# per-bin probability p_i = count_i / N, so sigma(p_i) = sqrt(p_i*(1-p_i)/N);
# dividing by bin width converts that into a sigma on the plotted density.
def _plot_hist_with_band(ax, data, bins, range_, color,
                         alpha_line=0.85, alpha_band=0.2, label=None):
    """
    Draw a density step histogram with a +/-1 sigma multinomial error band.

    sigma_density = sqrt(p_i * (1 - p_i) / N) / bin_width_i,  p_i = count_i / N.
    The band is rendered as a fill_between in step form, clipped below at 0.
    """
    counts, edges = np.histogram(data, bins=bins, range=range_)
    N       = max(counts.sum(), 1)
    widths  = np.diff(edges)
    density = counts / (N * widths)
    p       = counts / N
    err     = np.sqrt(np.maximum(p * (1.0 - p), 0.0) / N) / widths

    ax.hist(data, bins=edges, range=range_, color=color, alpha=alpha_line,
            density=True, histtype='step', linewidth=1.5, label=label)

    # Build step-form x/y arrays so fill_between matches the histogram outline.
    x_step = np.concatenate([[edges[0]], np.repeat(edges[1:-1], 2), [edges[-1]]])
    y_lo   = np.repeat(np.maximum(density - err, 0.0), 2)
    y_hi   = np.repeat(density + err, 2)
    ax.fill_between(x_step, y_lo, y_hi, color=color, alpha=alpha_band, linewidth=0)


def _shared_range(*arrays, lo_pct=1, hi_pct=99):
    """1st/99th-percentile range (svj_explorer.py's own convention), combined
    across all given arrays (NaNs ignored) so they share one set of bin edges."""
    los, his = [], []
    for a in arrays:
        a = a[np.isfinite(a)]
        if len(a) == 0:
            continue
        los.append(np.percentile(a, lo_pct))
        his.append(np.percentile(a, hi_pct))
    if not los:
        return (0.0, 1.0)
    return (min(los), max(his))


def compare(paired_path, out=None, out_diff=None, out_ratio=None, bins=30):
    data, cols = load_tsv(paired_path)

    missing = [f'{o}{suf}' for o in SHARED_OBS for suf in ('_truth', '_delphes')
               if f'{o}{suf}' not in cols]
    if missing:
        raise ValueError(f"Column(s) {missing} missing from {paired_path} -- "
                          f"was it generated by the current svj_delphes_test.cc?")

    print(f"Paired TSV: {paired_path}   ({len(data)} events processed)")
    print()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
        print("(matplotlib not available -- printing summary statistics only)")

    if plt is not None:
        fig, axes = plt.subplots(1, len(SHARED_OBS), figsize=(4.2 * len(SHARED_OBS), 3.6))
        fig_diff, axes_diff = plt.subplots(1, len(SHARED_OBS), figsize=(4.2 * len(SHARED_OBS), 3.6))
        fig_ratio, axes_ratio = plt.subplots(1, len(SHARED_OBS), figsize=(4.2 * len(SHARED_OBS), 3.6))
        if len(SHARED_OBS) == 1:
            axes, axes_diff, axes_ratio = [axes], [axes_diff], [axes_ratio]

    for i, obs in enumerate(SHARED_OBS):
        x_truth   = data[:, cols[f'{obs}_truth']]
        x_delphes = data[:, cols[f'{obs}_delphes']]

        print(f"{obs}")
        print(f"  truth   : {_summary(x_truth)}")
        print(f"  delphes : {_summary(x_delphes)}")

        # ── Zero-jet contingency: how often does each side have no accepted
        #    jet for this event? (nJets is never NaN, so use it as the mask
        #    source for all four other observables too, since they all share
        #    the same underlying "no accepted jet" condition per side.) ─────
        if obs != 'nJets':
            truth_zero   = ~np.isfinite(x_truth)
            delphes_zero = ~np.isfinite(x_delphes)
        else:
            truth_zero   = (data[:, cols['nJets_truth']] == 0)
            delphes_zero = (data[:, cols['nJets_delphes']] == 0)
        both_ok   = np.sum(~truth_zero & ~delphes_zero)
        truth_only_zero   = np.sum(truth_zero & ~delphes_zero)
        delphes_only_zero = np.sum(~truth_zero & delphes_zero)
        both_zero = np.sum(truth_zero & delphes_zero)
        print(f"  zero-jet contingency (truth / delphes): "
              f"both-ok={both_ok}  truth-only-zero={truth_only_zero}  "
              f"delphes-only-zero={delphes_only_zero}  both-zero={both_zero}")
        print()

        # ── Optional third reference series (e.g. leadVisPt's "+dark matter"
        #    truth-full channel) -- not required, so an older paired TSV
        #    without this column just silently skips it on this one panel.
        extra = EXTRA_TRUTH_SERIES.get(obs)
        x_extra, extra_label, extra_color = None, None, None
        if extra is not None:
            extra_col, extra_label, extra_color = extra
            if extra_col in cols:
                x_extra = data[:, cols[extra_col]]
                print(f"  {extra_label:8s}: {_summary(x_extra)}")
            else:
                print(f"  ({extra_col} not found -- regenerate the paired TSV "
                      f"with the current svj_delphes_test.cc for this series)")
        print()

        if plt is not None:
            with plt.rc_context(_PLOT_RCPARAMS):
                # ── Panel 1 (fig): marginal overlay, same as before, plus the
                #    optional third reference series if this obs has one ────
                ax = axes[i]
                range_arrays = [x_truth, x_delphes] + ([x_extra] if x_extra is not None else [])
                rng = _shared_range(*range_arrays)
                _plot_hist_with_band(ax, x_truth[np.isfinite(x_truth)], bins=bins,
                                     range_=rng, color='steelblue', label='truth')
                _plot_hist_with_band(ax, x_delphes[np.isfinite(x_delphes)], bins=bins,
                                     range_=rng, color='darkorange', label='delphes')
                if x_extra is not None:
                    _plot_hist_with_band(ax, x_extra[np.isfinite(x_extra)], bins=bins,
                                         range_=rng, color=extra_color, label=extra_label)
                ax.set_xlabel(obs)
                ax.set_ylabel('Density')
                ax.legend(framealpha=0.5)

                # ── Panel 2 (fig_diff): per-event difference, delphes - truth,
                #    masked to rows where BOTH sides have an accepted jet ────
                ax_d = axes_diff[i]
                pair_mask = np.isfinite(x_truth) & np.isfinite(x_delphes)
                diff = x_delphes[pair_mask] - x_truth[pair_mask]
                if len(diff) > 0:
                    rng_d = _shared_range(diff, lo_pct=0.5, hi_pct=99.5)
                    _plot_hist_with_band(ax_d, diff, bins=bins, range_=rng_d,
                                         color='mediumpurple',
                                         label=f'n={len(diff)}')
                    ax_d.axvline(0.0, color='black', linestyle=':',
                                linewidth=1.0, alpha=0.6)
                    ax_d.legend(framealpha=0.5)
                else:
                    ax_d.set_title('(no paired events)', fontsize=10)
                ax_d.set_xlabel(f'{obs}: delphes - truth (per event)')
                ax_d.set_ylabel('Density')

                # ── Panel 3 (fig_ratio): per-event ratio, delphes / truth,
                #    masked to paired rows where truth is also nonzero (a
                #    ratio against truth=0 is undefined, not just large) ────
                ax_r = axes_ratio[i]
                nonzero_mask = pair_mask & (x_truth != 0)
                ratio = x_delphes[nonzero_mask] / x_truth[nonzero_mask]
                if len(ratio) > 0:
                    rng_r = _shared_range(ratio, lo_pct=0.5, hi_pct=99.5)
                    _plot_hist_with_band(ax_r, ratio, bins=bins, range_=rng_r,
                                         color='seagreen',
                                         label=f'n={len(ratio)}')
                    ax_r.axvline(1.0, color='black', linestyle=':',
                                linewidth=1.0, alpha=0.6)
                    ax_r.legend(framealpha=0.5)
                else:
                    ax_r.set_title('(no paired nonzero-truth events)', fontsize=10)
                ax_r.set_xlabel(f'{obs}: delphes / truth (per event)')
                ax_r.set_ylabel('Density')

    if plt is not None:
        with plt.rc_context(_PLOT_RCPARAMS):
            fig.suptitle('Truth vs. Delphes (Stage 0) -- same events, +/-1sigma multinomial bands')
            fig.tight_layout()
            fig_diff.suptitle('Per-event difference (delphes - truth), paired events only')
            fig_diff.tight_layout()
            fig_ratio.suptitle('Per-event ratio (delphes / truth), paired nonzero-truth events only')
            fig_ratio.tight_layout()
        if out:
            fig.savefig(out, dpi=150, bbox_inches='tight')
            print(f"Saved marginal-overlay figure -> {out}")
        if out_diff:
            fig_diff.savefig(out_diff, dpi=150, bbox_inches='tight')
            print(f"Saved per-event-difference figure -> {out_diff}")
        if out_ratio:
            fig_ratio.savefig(out_ratio, dpi=150, bbox_inches='tight')
            print(f"Saved per-event-ratio figure -> {out_ratio}")
        if not out and not out_diff and not out_ratio:
            plt.show()


def print_top_diff(paired_path, obs, n=10, direction='high'):
    """
    Print the N paired events with the most extreme delphes-truth difference
    for one observable, plus a ready-to-paste eventIndex list for
    svj_delphes_test.cfg's `dump_events` key (see plot_lego.py).

    direction: 'high' (largest delphes-truth, i.e. Delphes adds the most),
               'low'  (most negative, i.e. Delphes subtracts the most),
               'abs'  (largest |delphes-truth| regardless of sign).
    """
    data, cols = load_tsv(paired_path)
    for suf in ('_truth', '_delphes'):
        if f'{obs}{suf}' not in cols:
            raise ValueError(f"'{obs}{suf}' not found in {paired_path}")

    x_truth   = data[:, cols[f'{obs}_truth']]
    x_delphes = data[:, cols[f'{obs}_delphes']]
    event_idx = data[:, cols['eventIndex']].astype(int)

    pair_mask = np.isfinite(x_truth) & np.isfinite(x_delphes)
    diff = x_delphes - x_truth

    if direction == 'high':
        order_key = -diff
    elif direction == 'low':
        order_key = diff
    elif direction == 'abs':
        order_key = -np.abs(diff)
    else:
        raise ValueError("direction must be 'high', 'low', or 'abs'")
    order_key = np.where(pair_mask, order_key, np.inf)  # push unpaired events to the end

    order = np.argsort(order_key)[:n]

    print(f"Top {len(order)} events by {obs} delphes-truth difference ({direction}):")
    print(f"{'eventIndex':>10}  {'truth':>12}  {'delphes':>12}  {'diff':>12}")
    for i in order:
        print(f"{event_idx[i]:>10}  {x_truth[i]:>12.3f}  {x_delphes[i]:>12.3f}  {diff[i]:>12.3f}")

    idx_list = ",".join(str(event_idx[i]) for i in order)
    print()
    print("Paste into svj_delphes_test.cfg's dump_events, then rebuild/rerun to")
    print("get the constituent dump plot_lego.py needs:")
    print(f"  dump_events = {idx_list}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired', default='simulated/tsv/jets_delphes_paired.tsv',
                         help='Path to the paired truth+Delphes TSV written by '
                              'svj_delphes_test.cc (default: %(default)s)')
    parser.add_argument('--out', default=None,
                         help='Save the marginal-overlay figure here instead of showing it')
    parser.add_argument('--out-diff', default=None,
                         help='Save the per-event-difference figure here instead of showing it')
    parser.add_argument('--out-ratio', default=None,
                         help='Save the per-event-ratio figure here instead of showing it')
    parser.add_argument('--bins', type=int, default=30,
                         help='Histogram bin count (default: %(default)s, matching '
                              "svj_explorer.py's own marginal-histogram bin count)")
    parser.add_argument('--top-diff', type=int, default=0, metavar='N',
                         help='Instead of plotting, print the N events with the most extreme '
                              'delphes-truth difference for --top-diff-obs, and a dump_events '
                              'list ready to paste into svj_delphes_test.cfg')
    parser.add_argument('--top-diff-obs', default='leadVisPt',
                         help='Observable to rank by for --top-diff (default: %(default)s)')
    parser.add_argument('--top-diff-dir', default='high', choices=['high', 'low', 'abs'],
                         help="'high' = Delphes adds the most (default), 'low' = Delphes "
                              "subtracts the most, 'abs' = largest magnitude either way")
    args = parser.parse_args()

    if args.top_diff > 0:
        print_top_diff(args.paired, args.top_diff_obs, n=args.top_diff,
                       direction=args.top_diff_dir)
        return

    compare(args.paired, out=args.out, out_diff=args.out_diff, out_ratio=args.out_ratio,
           bins=args.bins)


if __name__ == '__main__':
    main()
