#!/usr/bin/env python3
"""
make_xsec.py -- build signal/xsec.json, the signal normalisation table.

Runs src/generate_events/svj_xsec at each mZ' on the production grid and
records sigma x BR(Z' -> dark).  See docs/normalisation.md for what the
numbers mean and how they are used.

    make svj_xsec && python3 signal/make_xsec.py

Cheap: the parton and hadron levels are off, so each mass is seconds.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / 'src' / 'generate_events' / 'svj_xsec'

# Must match the [scan] mZ axis of scan_prod_delphes.cfg.
MZ_GRID = np.logspace(np.log10(500.0), np.log10(4000.0), 8)

# Reference point: exactly what svj_regression_delphes.cc sets, so the measured
# sigma is the cross section of the samples already on disk.
SM_BR_PER_FLAVOUR = 1.0e-4
N_SM_FLAVOURS = 6
WIDTH_OVER_MASS = 0.025          # 4900023:mWidth = 0.025 * mZ, doForceWidth = on
N_EVENT = 4000                   # ~0.85% MC error on sigma


def run(mz, n_event=N_EVENT, sm_br=SM_BR_PER_FLAVOUR):
    """Return (sigma_pb, err_pb) at this mZ'."""
    out = subprocess.run([str(BIN), f'{mz:.6g}', str(n_event), f'{sm_br:g}'],
                         capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        parts = line.split('\t')
        if len(parts) == 5:
            return float(parts[1]), float(parts[2])
    raise RuntimeError(f'no result line for mZ={mz}:\n{out}')


def g_q_reference():
    """
    Coupling implied by the reference branching ratios.

    Gamma(Z' -> q qbar) for one flavour with vector coupling g_q is
    N_c g_q^2 M / (12 pi) = g_q^2 M / (4 pi).  Here that partial width is
    BR_per_flavour * Gamma_tot = SM_BR_PER_FLAVOUR * WIDTH_OVER_MASS * M.
    """
    return float(np.sqrt(SM_BR_PER_FLAVOUR * WIDTH_OVER_MASS * 4.0 * np.pi))


def g_q_max():
    """
    Largest g_q consistent with the width that was simulated.

    sigma ~ BR_SM * BR_dark with 6*BR_SM + BR_dark = 1, which peaks at
    BR_SM = 1/12, BR_dark = 1/2.  Beyond this the SM partial widths alone
    would exceed the forced total width.
    """
    return float(np.sqrt((1.0 / 12.0) * WIDTH_OVER_MASS * 4.0 * np.pi))


def main():
    if not BIN.exists():
        sys.exit(f"missing {BIN} -- run 'make svj_xsec' first")

    rows = []
    for mz in MZ_GRID:
        sigma, err = run(mz)
        rows.append({'mZ': float(mz), 'sigma_pb': sigma, 'err_pb': err})
        print(f'  mZ={mz:8.2f}  sigma={sigma:.6e} pb  (+-{100*err/sigma:.2f}%)')

    g_ref, g_max = g_q_reference(), g_q_max()
    br_dark = 1.0 - N_SM_FLAVOURS * SM_BR_PER_FLAVOUR
    doc = {
        'description':
            "sigma x BR(Z'->dark) for HiddenValley:ffbar2Zv at eCM=14 TeV, "
            "measured with PYTHIA 8.317 at the reference couplings below. "
            "Rescale with sigma_pb(mZ, g_q) in src/normalisation.py.",
        'eCM_GeV': 14000.0,
        'reference': {
            'sm_br_per_flavour': SM_BR_PER_FLAVOUR,
            'n_sm_flavours': N_SM_FLAVOURS,
            'br_dark': br_dark,
            'width_over_mass': WIDTH_OVER_MASS,
            'g_q': g_ref,
            'g_q_max_consistent': g_max,
            'comment':
                'g_q is inferred from Gamma_1flav = g_q^2 M/(4 pi), treating the '
                'coupling as purely vector. g_q_max_consistent is the ceiling at '
                'this forced width; above it the sample rate is benchmark-only.',
        },
        'n_event_per_point': N_EVENT,
        'interpolation': 'pchip in (ln mZ, ln sigma)',
        'points': rows,
    }
    out = REPO / 'signal' / 'xsec.json'
    out.write_text(json.dumps(doc, indent=1) + '\n')
    print(f'\n  g_q(reference) = {g_ref:.5f}   g_q(max at this width) = {g_max:.4f}')
    print(f'  wrote {out}')


if __name__ == '__main__':
    main()
