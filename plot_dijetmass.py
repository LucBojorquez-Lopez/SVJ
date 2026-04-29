#!/usr/bin/env python3
"""
Plot the dijet invariant mass for events with exactly two visible jets.
Reads data/regression/jets_kinematics.tsv produced by svj_regression.
Saves plot to dijetmass.png.
"""

import numpy as np
import matplotlib.pyplot as plt

TSV = 'data/regression/jets_kinematics.tsv'

d = np.loadtxt(TSV, comments='#')
# columns: n_jets, j1_px, j1_py, j1_pz, j1_E, j2_px, j2_py, j2_pz, j2_E

mask = d[:, 0] == 2
d2 = d[mask]
print(f"Total events (>=1 jet): {len(d)}")
print(f"2-jet events:           {len(d2)}")

j1_E  = d2[:, 4];  j1_px = d2[:, 1]; j1_py = d2[:, 2]; j1_pz = d2[:, 3]
j2_E  = d2[:, 8];  j2_px = d2[:, 5]; j2_py = d2[:, 6]; j2_pz = d2[:, 7]

E2  = (j1_E  + j2_E )**2
px2 = (j1_px + j2_px)**2
py2 = (j1_py + j2_py)**2
pz2 = (j1_pz + j2_pz)**2
mjj = np.sqrt(np.maximum(E2 - px2 - py2 - pz2, 0.0))

fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(mjj, bins=600, histtype='step', linewidth=1.5, color='steelblue')
ax.set_xlabel(r'$m_{jj}$ [GeV]', fontsize=13)
ax.set_ylabel('Events / bin', fontsize=13)
ax.set_title('Dijet invariant mass (2-jet events)', fontsize=13)
#ax.set_yscale('log')
plt.tight_layout()
plt.savefig('dijetmass.png', dpi=150)
print("Saved dijetmass.png")
