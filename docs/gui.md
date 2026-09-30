# Interactive GUI (Jupyter)

## Quick start

Create a notebook at the project root (notebooks are gitignored — the GUI is
driven from a scratch notebook per user) and run:

```python
%matplotlib widget
import sys
sys.path.insert(0, 'src/gui')
from svj_explorer import show
show()
```

The GUI loads the default scan automatically — `simulated/svj/working_example/`,
the same one `helpers` uses. Sliders, dropdowns and cut panels are all built
from whatever axes and observables that NPZ contains.

## Running it on lxplus

The GUI's home is a local checkout (see below), but it does run on lxplus for a
look at the plots. **VALIDATE will not work there** — it shells out to
`svj_regression` with `nWorkers=14`, which is the whole point of it locally and
antisocial on a shared login node ([lxplus.md](lxplus.md) section 2). Everything
else works: sliders, cuts, the background panel, S/sqrt(B).

**The kernel.** A venv that inherits the LCG view's packages, registered the
ordinary way. Built once with:

```bash
source /eos/user/l/lbojorqu/SVJ/setup_env.sh
python3 -m venv --system-site-packages /eos/user/l/lbojorqu/svj-venv
/eos/user/l/lbojorqu/svj-venv/bin/python -m ipykernel install --user \
    --name svj --display-name "SVJ (LCG_110 venv)"
```

Then add **one** variable to the generated
`~/.local/share/jupyter/kernels/svj/kernel.json`:

```json
"env": {"LD_LIBRARY_PATH":
        "/cvmfs/sft.cern.ch/lcg/releases/gcc/13.1.0-b3d18/x86_64-el9/lib64"}
```

Two traps are worth recording, because each produced a confusing failure:

- **The LCG view's python cannot be used directly as a VS Code interpreter.**
  Invoked bare it cannot even `import numpy` — it depends on `PYTHONPATH` from
  the view's `setup.sh`. The venv fixes this permanently: `--system-site-packages`
  puts the view's packages on the path via `pyvenv.cfg`, so the venv python works
  with no environment setup at all.
- **`scipy.optimize` still needs LCG's gcc 13 runtime.** Its compiled
  `_highspy` links `GLIBCXX_3.4.30`; bare, the loader picks the system
  `/lib64/libstdc++.so.6` and it dies with `ImportError`. That is the same
  gcc-version trap as the Delphes link failure recorded in the `Makefile`, and
  it is the only reason the `env` block above exists. One directory is enough.

An earlier attempt pointed `argv` at a shell script that sourced
`setup_env.sh`. That works under `jupyter lab` but **VS Code's Jupyter
extension does not reliably surface script-based kernelspecs**, so the kernel
picker came up empty. A plain venv interpreter avoids the whole problem, and
needs no running server: pick **SVJ (LCG_110 venv)** from the kernel list.
Reload the VS Code window if a freshly installed kernel does not appear.

**Launching, option A — SSH tunnel:**

```bash
ssh -L 8888:localhost:8888 <user>@lxplus.cern.ch
source /eos/user/l/lbojorqu/SVJ/setup_env.sh
cd /eos/user/l/lbojorqu/SVJ
jupyter lab --no-browser --port=8888
```

Then open the printed `127.0.0.1:8888/?token=...` URL locally.

**Option B — SWAN** (`swan.cern.ch`): pick an LCG release matching
`setup_env.sh`'s `LCG_VIEW`, then open the notebook from `/eos/user/...`. No
tunnelling, but you get SWAN's own kernel rather than the `svj` one, so the
environment is only as close as the LCG versions are.

**Which scan to load.** The 17-observable truth fit at
`/eos/user/l/lbojorqu/svj/prod/truth/fit17` is the one to use. It is the only
fit containing both `hemiMass2` and `dPhiMETclose`, so it is the only one where
the mass swap and the |Δφ| fold are actually active — on the default
11-observable `working_example` both are inert. Note the numbers there are
*plumbing, not physics*: signal is truth level while background is Delphes.

**Cost of the first cell.** Loading the 1.3 GB background cache and computing
the fixed plot ranges takes a couple of minutes and a few GB of RAM, because
`_prepare` promotes the background to float64. On a shared login node that is
worth being aware of; if it is too heavy, `build_background_cache` accepts
`max_rows_per_sample` to thin it — at the cost documented in
[normalisation.md](normalisation.md) M10.

## GUI controls

| Control | Description |
|---------|-------------|
| **Parameter sliders** | One slider per scan axis, built dynamically from the loaded NPZ — adding or removing axes in `scan_regression.cfg` automatically updates the GUI |
| **Feature X / Y dropdowns** | Choose any two observables for the joint plot |
| **Fixed / Auto toggle** | Fixed axes use percentile-1/99 ranges from grid corners; Auto rescales to the current sample |
| **N model** | Number of model samples drawn from the interpolated distribution |
| **N validate** | Events for the VALIDATE PYTHIA run |
| **VALIDATE button** | Runs `svj_regression` at the current slider point and overlays the true distribution in crimson |
| **Cuts panel** | Per-observable range sliders to filter both model and true events; **Reset cuts** restores all |

## Where the GUI runs, and why that matters for VALIDATE

**The GUI is used on a local copy of the repository, not on lxplus.** Nothing
here has been tested on lxplus, so treat running it there as unexplored rather
than supported.

That distinction matters for one setting. The VALIDATE button shells out to
`svj_regression` at the current slider point, and `_make_validate_cfg` takes
`nWorkers` from the cfg with a **default of 14** — i.e. it deliberately uses the
whole local machine, because an interactive button should return quickly and the
machine is not shared.

This is the exact opposite of the rule in [lxplus.md](lxplus.md) section 2
("-j4, not -j$(nproc)"), which exists because lxplus login nodes are shared and
a 12-worker job there triggered an automated CPU-pressure warning. Both are
right in their own place. If the GUI is ever run on lxplus, `nWorkers` must be
lowered in the cfg first — but the fix belongs in the cfg for that deployment,
not in the default.


Moving any physics slider clears the validation overlay (since the true data is now at a different point).

## `show()` arguments

```python
show(n_samples=10_000, scan_dir=None)
```

| Argument | Default | Description |
|----------|---------|-------------|
| `n_samples` | 10 000 | Initial number of model samples per draw |
| `scan_dir` | None | Path to a directory containing `svj_scan.npz` and `svj_scan_meta.json`. When `None`, `helpers.DEFAULT_SCAN_DIR` (`simulated/svj/working_example/`) is used. |

To load a different scan — for instance the larger 6-axis one:

```python
show(scan_dir='simulated/svj/')          # 6 axes, 16 observables
show(scan_dir='path/to/my_scan/')        # any scan you produced yourself
```

Both `svj_scan.npz` and `svj_scan_meta.json` must be present in that directory.

## Observable coverage

The GUI dropdown includes all observables that are in the loaded NPZ plus any
derived observables (ratios) whose components are present. The set is determined
automatically when the module is imported.

## Physical constraints applied to model draws

The interpolator fits each observable's marginal separately and joins them with
a Gaussian copula, so it does not know about constraints the generator enforces
exactly. Two exist, and model draws violate both. `_physicalise()` repairs each
by **projecting the draw back onto the surface the true data occupies** — never
by discarding it, which would throw away draws the model considered likely and
bias the signal acceptance.

| constraint | why draws violate it | repair |
|---|---|---|
| `dPhiMETclose`, `dPhiMETfar` are folded to \|.\| | stored signed in (-pi, pi]; the fitting pipeline takes `abs_value` first and labels them \|.\| | take the absolute value |
| `hemiMass1 >= hemiMass2` | `svj_observables_common.h:280` swaps them, so it holds exactly in all simulated data; the copula can draw the pair inverted | swap the pair |

Both act on **base** columns before derived ratios are computed, so
`mass2/mass1` uses the corrected values. Measured on the 16-observable scan,
11.7% of model draws had a negative `dPhiMETclose` before folding and 0.0%
after.

Why this matters: unfolded, a cut of `> 0.4` on `dPhiMETclose` reads as an
angular selection but actually discards the entire negative half — it halved the
background while leaving the QCD fraction at 99% (docs/normalisation.md M9).
And \|dPhi\|(MET, closest jet) is *the* discriminant between real and
mismeasured MET: QCD sits at 0.1-0.7, Z(->nunu) at 1.6-2.8.

The mass ordering used to be repaired in `_draw`, but only when a mass feature
happened to be plotted — which made the signal acceptance depend on the chosen
plot axes. It is now unconditional.

Note that both constraints are **inert on the default 11-observable scan**
(`working_example` has neither `dPhiMETclose` nor `hemiMass2`) and on the
16-observable scan only the fold applies. Both become live with the
28-observable Delphes fit.

## Background and S/sqrt(B)

The third figure row shows the Standard-Model background: one summed weighted
density over all 17 samples, from the cache described in
[normalisation.md](normalisation.md). It is **static** — no physics slider
changes it — so it is loaded once per session and only the cut mask is
re-applied per redraw. The background also appears as a dashed green overlay on
both marginal panels.

The weighting is not optional. Per-sample weights span nine orders of magnitude
(1398 fb down to 9.5e-7 fb per event), so an unweighted histogram would show
whichever sample has the most *rows* rather than the most *rate*.

### The Rate & significance panel

`g_q` and `L` are deliberately separated from the physics sliders, because
neither changes any plotted distribution — they affect only the counter. (The
background panel is a weighted *density*, so it does not rescale with
luminosity either.) Their observers therefore call only the significance
update, not a full resample-and-redraw.

| control | default | meaning |
|---|---|---|
| `g_q` | **0.1618** | Z' coupling to SM quarks. Default is the ceiling consistent with the 2.5% width the samples were generated at; above it the readout says **benchmark-only** (see normalisation.md M4). |
| `L` | 140 fb^-1 | integrated luminosity. Run 2 ~140, Run 3 ~300, HL-LHC ~3000. |

The readout reports, in order: sigma, g_q, BR_dark; the signal acceptance
`eps_S` and `S`; `B` with its MC error and effective event count; **S/sqrt(B)
with a propagated uncertainty** and the Asimov significance beside it; then
three trust indicators.

`S` uses the acceptance of the **full** model sample with only the cut panel
applied, so it does not move when the plotted features change.

### Reading the trust indicators

They matter more than the significance itself, because S/sqrt(B) looks equally
confident whether or not the background behind it is believable.

| indicator | meaning | when to worry |
|---|---|---|
| **QCD %** | share of B from QCD | QCD is the worst-modelled component (LO, no merging, no pileup, MET partly a jet-acceptance artefact). Amber above 70%, red above 90%. At `MET > 200` it is 99.6%. |
| **MC err** | `sqrt(sum w^2)/sum w` | statistical only, and *not* `sqrt(N)` for weighted events. Red above 30%. |
| **worst event** | largest single event's share of B | the failure a small MC error hides. Red above 5%. The `MET>600 & lVpT>800 & \|dPhi\|>1` region reaches 10%. |

Where the background is most trustworthy — and why it is not simply "where the
error bar is smallest" — is worked through in
[normalisation.md](normalisation.md) M8-M11. Short version: **MET >~ 800 GeV**,
because that is where QCD stops dominating, not because the statistics are
better there (they are worse).

### The S uncertainty, broken down

The three contributions to `S` are shown **separately** rather than summed,
because they are different kinds of thing and shrink for different reasons:

| contribution | typical | what shrinks it |
|---|---|---|
| **acceptance** | 1-2% | binomial error on eps from the model sample. Draw more samples (`N model`). Purely a sample-size diagnostic — it says nothing about physics, and exists so that a hard selection with few surviving draws is visibly uncertain rather than silently precise. |
| **sigma_MC** | 0.83% | finite statistics behind the tabulated sigma. Raise `N_EVENT` in `signal/make_xsec.py`. |
| **interp** | 1.6%, or **0** on a tabulated mass | a *bound* on the PCHIP interpolation between grid masses, not a resolved measurement — see below. Reads "(exact: on a tabulated mass)" at the grid points, which in practice means only the slider endpoints 500 and 4000 GeV. |

`total` adds them in quadrature, which is approximate: `interp` is a bound on a
systematic, not a Gaussian sigma.

**Why `interp` is a bound.** The four validation runs at 800/1400/2500/3500 GeV
gave measured/interpolated ratios of 1.0012, 0.9991, 0.9845 and 0.9974. But
those runs carry 0.83% MC error themselves, so three of the four deviations sit
inside their own noise and only the 2500 GeV point is even marginally
significant. 1.6% is therefore the worst case observed, applied flat off-grid —
conservative near a tabulated mass, where the true interpolation error is
negligible. Tightening it needs more events per validation mass, not a better
spline.

**Not included, and larger than all three: the LO K-factor.** The tabulated
sigma is leading order, and NLO corrections for a Z' of this kind are tens of
percent (for comparison, ttbar's LO-to-NNLO K-factor is ~1.6). Nothing here
estimates it, so the quoted `S` error is a statistical-and-interpolation error,
not a cross-section uncertainty. The readout says so.

`B`'s error is MC statistical only, and is *not* sqrt(N) — for weighted events
it is `L x sqrt(sum w^2)`. Everything in
[backgrounds.md](backgrounds.md#what-this-will-get-wrong) — LO normalisations,
no jet merging, no pileup, a parametrised detector — sits on top of it and is
not in any error bar on this panel.
