#!/usr/bin/env python3
"""
plot_lego.py
============
Render eta-phi "lego" plots (3D bar charts, bar height = constituent pT) of
the leading jet's constituents for specific events, truth vs Delphes side by
side -- for visually investigating *why* a specific event's leadVisPt (or any
other jet observable) changed the way it did under Delphes reconstruction.

Reads the two constituent-dump TSVs svj_delphes_test.cc writes when its
`dump_events` cfg key is set (see docs/setup_delphes.md). Those files don't
exist for a normal run (dump_events empty by default), so this script is a
second step after:

    1. Run svj_delphes_test normally, then find interesting events:
       python src/generate_events/compare_delphes_truth.py --top-diff 10
    2. Paste the printed eventIndex list into svj_delphes_test.cfg's
       dump_events key and re-run svj_delphes_test (same fixed Pythia seed
       -> same events regenerate at the same indices, this time with their
       leading-jet constituents also written out).
    3. python src/generate_events/plot_lego.py

Each constituent is drawn as its own bar (not binned into a fixed grid --
these are individual truth particles / Delphes EFlow candidates, not
calorimeter cells), so this is closer to a raw "stem plot" of the jet's
actual constituent structure than a true calorimeter-tower lego plot. Phi is
recentered relative to the event's highest-pT constituent (wrapped into
(-pi, pi]) rather than plotted raw, so a jet that happens to straddle the
+-pi boundary doesn't get spuriously split across the edges of the plot.

Usage
-----
    # after the dump_events re-run above, plot every dumped event:
    python src/generate_events/plot_lego.py

    # only specific events, and save PNGs instead of showing interactively:
    python src/generate_events/plot_lego.py --events 4237,891 --out-dir simulated/tsv/lego
"""

import sys
import argparse
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SRC))
from observables import load_tsv  # noqa: E402  (existing, unmodified helper)

# Same font/label styling family as compare_delphes_truth.py / svj_explorer.py,
# trimmed to what a 3D axes actually uses (grid/axisbelow don't apply to bar3d).
_PLOT_RCPARAMS = {
    'font.family':     'serif',
    'font.serif':      ['Times New Roman', 'DejaVu Serif'],
    'axes.labelsize':  10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 9,
}


def _load_constituents(path):
    """Returns dict: eventIndex -> (eta array, phi array, pt array)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} does not exist yet -- set dump_events in "
            f"svj_delphes_test.cfg and re-run svj_delphes_test first "
            f"(see this script's module docstring).")

    data, cols = load_tsv(path)
    if data.size == 0:
        return {}
    if data.ndim == 1:
        data = data.reshape(1, -1)   # a single-constituent file loads as 1-D

    idx_col = data[:, cols['eventIndex']].astype(int)
    eta_col = data[:, cols['eta']]
    phi_col = data[:, cols['phi']]
    pt_col  = data[:, cols['pt']]

    out = {}
    for idx in np.unique(idx_col):
        m = idx_col == idx
        out[int(idx)] = (eta_col[m], phi_col[m], pt_col[m])
    return out


def _wrap_phi(phi, phi_ref):
    """Wrap phi values into (-pi, pi] relative to phi_ref, so a jet that
    happens to straddle the raw +-pi discontinuity isn't spuriously split
    across the two edges of the plot."""
    return ((phi - phi_ref + np.pi) % (2 * np.pi)) - np.pi


def _lego_panel(ax, eta, phi, pt, color, title, bar_width=0.08):
    if len(pt) == 0:
        ax.set_title(f'{title}\n(no accepted jet / no constituents)', fontsize=9)
        return

    phi_ref = phi[np.argmax(pt)]   # recenter on the highest-pT constituent
    dphi = _wrap_phi(phi, phi_ref)

    dx = dy = bar_width
    ax.bar3d(eta - dx / 2, dphi - dy / 2, np.zeros(len(pt)),
             dx, dy, pt, color=color, shade=True, alpha=0.85)
    ax.set_xlabel(r'$\eta$')
    ax.set_ylabel(r'$\Delta\phi$ (rel. to leading constituent)')
    ax.set_zlabel(r'$p_T$ (GeV)')
    ax.set_title(f'{title}  (n={len(pt)}, sum pT={pt.sum():.0f} GeV)', fontsize=9)


def plot_event(event_idx, truth_data, delphes_data, out=None):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

    eta_t, phi_t, pt_t = truth_data.get(event_idx, (np.array([]), np.array([]), np.array([])))
    eta_d, phi_d, pt_d = delphes_data.get(event_idx, (np.array([]), np.array([]), np.array([])))

    with plt.rc_context(_PLOT_RCPARAMS):
        fig = plt.figure(figsize=(10, 4.5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        _lego_panel(ax1, eta_t, phi_t, pt_t, 'steelblue', 'Truth')
        _lego_panel(ax2, eta_d, phi_d, pt_d, 'darkorange', 'Delphes')
        fig.suptitle(f'Leading-jet constituents -- event {event_idx}')
        fig.tight_layout()

    if out:
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved -> {out}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--truth', default='simulated/tsv/jets_delphes_constituents_truth.tsv',
                         help='Truth-side constituent dump (default: %(default)s)')
    parser.add_argument('--delphes', default='simulated/tsv/jets_delphes_constituents_delphes.tsv',
                         help='Delphes-side constituent dump (default: %(default)s)')
    parser.add_argument('--events', default=None,
                         help='Comma-separated event indices to plot (default: every '
                              'event index found in the constituent dump files)')
    parser.add_argument('--out-dir', default=None,
                         help='Save one PNG per event here instead of showing interactively')
    args = parser.parse_args()

    truth_data = _load_constituents(args.truth)
    delphes_data = _load_constituents(args.delphes)

    if args.events:
        event_list = [int(s) for s in args.events.split(',') if s.strip()]
    else:
        event_list = sorted(set(truth_data) | set(delphes_data))

    if not event_list:
        print("No events found in the constituent dump files -- did you set "
              "dump_events in svj_delphes_test.cfg and re-run svj_delphes_test?")
        return

    for idx in event_list:
        out = None
        if args.out_dir:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            out = str(Path(args.out_dir) / f'lego_event{idx}.png')
        plot_event(idx, truth_data, delphes_data, out=out)


if __name__ == '__main__':
    main()
