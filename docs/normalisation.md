# Normalisation: from shapes to S/sqrt(B)

The interpolated SVJ model and the background TSVs both produce **shapes** —
they say what fraction of events lands where, not how many events there are.
S/sqrt(B) needs *rates*. This document is how the rates are obtained, what was
measured to justify them, and what the resulting number does and does not mean.

Companion docs: [backgrounds.md](backgrounds.md) (what the SM samples are and
why), [raw-store.md](raw-store.md) (the signal event store), [gui.md](gui.md)
(where this surfaces).

---

## The short version

```
sigma(mZ', g_q) = sigma_ref(mZ')          # measured, signal/xsec.json
                  x (g_q / 0.005605)^2    # coupling, sigma ~ g_q^2
                  x (BR_dark / 0.9994)    # what is left for the dark sector

S = sigma x L x eps(cuts)                 # eps = fraction of model samples passing
B = sum_samples w_i x n_i(cuts)           # w_i in fb, background/samples.json
```

Implemented in [`src/normalisation.py`](../src/normalisation.py). The cross
sections come from `src/generate_events/svj_xsec.cc` (`make svj_xsec`), which
reproduces the signal's PYTHIA setup exactly for everything that touches the
production rate and switches off everything that does not.

---

## Why the mass matters and the branching ratios do not

A natural first guess is that mZ' changes both the production `qq -> Z'` and the
decay `Z' -> qq / chi chi`. Only the first is true.

**Branching ratios are mass-independent here, for two separate reasons.**
First, in `svj_regression_delphes.cc` they are *inputs*, not outputs:
`4900023:oneChannel/addChannel` fixes them at 1e-4 per SM flavour and 0.9994
dark, and `doForceWidth = on` pins the total width at 2.5% of mZ'. PYTHIA never
computes them. Second, even in a model where PYTHIA did compute them, every
partial width goes as `Gamma = N_c g^2 M / (12 pi)` — linear in M — so the
*ratio* is flat. Thresholds could break that, but the dark quark is
`mq = LambdaDQCD (mPiOverLambda/5.5)^2 <= 2 GeV` against `mZ' >= 500 GeV`.
Nothing is near threshold anywhere on the grid.

So mass enters in exactly one place: **the production cross section, through
parton luminosity**. Making a heavier resonance needs higher-x quarks, and the
PDFs fall steeply there.

---

## Measurement log

### M1 — sigma factorises exactly as `C(mZ') x BR_SM x BR_dark`

Varying the per-flavour SM branching ratio at mZ' = 1219 GeV, with
`BR_dark = 1 - 6 BR_SM` following along, against the prediction
`sigma(BR) = sigma(1e-4) x (BR/1e-4) x (BR_dark/0.9994)`:

| BR_SM | measured [pb] | predicted [pb] | ratio |
|---|---|---|---|
| 1e-4 | 3.694492e-03 | 3.694492e-03 | 1.000000 |
| 2e-4 | 7.384548e-03 | 7.384548e-03 | 1.000000 |
| 5e-4 | 1.842810e-02 | 1.842810e-02 | 1.000000 |
| 1e-3 | 3.674530e-02 | 3.674530e-02 | 1.000000 |

Six-digit agreement. Note that plain proportionality to `BR_SM` alone would be
off by 0.54% at the last row; the `BR_dark` factor is what closes it. This is
the licence to rescale to any coupling without re-simulating: **the shapes on
disk are valid for every (g_q, BR_dark); only the normalisation moves.**

### M2 — the reference grid, and how to interpolate between its masses

`signal/xsec.json`, 4000 events per mass (~0.85% MC error), at the reference
couplings:

| mZ' [GeV] | 500 | 673 | 906 | 1219 | 1641 | 2208 | 2972 | 4000 |
|---|---|---|---|---|---|---|---|---|
| sigma [pb] | 1.189e-1 | 4.015e-2 | 1.270e-2 | 3.704e-3 | 9.655e-4 | 2.123e-4 | 3.758e-5 | 4.669e-6 |

**sigma falls 25 000x across the grid**, roughly `sigma ~ mZ'^-4.8`.

Interpolation was checked against independent PYTHIA runs at four off-grid
masses (ratio measured/interpolated; MC error 0.85%):

| mZ' | linear in (ln m, ln s) | PCHIP | cubic |
|---|---|---|---|
| 800 | 1.0103 | 1.0012 | 1.0011 |
| 1400 | 1.0157 | 0.9991 | 0.9989 |
| 2500 | 1.0181 | 0.9845 | 0.9843 |
| 3500 | 1.0438 | 0.9974 | 0.9902 |

Linear is biased low throughout (the curve is convex in log-log) and reaches
4.4% at the top. **PCHIP is used**: worst case 1.6%, monotone, no overshoot.

### M3 — the samples on disk assume g_q ~ 0.0056, not 0.25

Inverting `Gamma(Z' -> q qbar)_1flav = g_q^2 M / (4 pi)` with
`Gamma_1flav = BR_SM x Gamma_tot = 1e-4 x 0.025 M`:

**g_q(reference) = 0.005605.**

The standard SVJ benchmark (Cohen-Lisanti-Lou; CMS EXO-19-020) is
**g_q = 0.25** — about 45x larger, so ~2000x more signal. Worth knowing before
any yield is quoted: these samples are a very weakly coupled Z'.

### M4 — a hard ceiling at 417x, from the width we simulated

g_q = 0.25 is **not self-consistent** with this sample. At that coupling the SM
partial widths alone are ~3% of the mass, so `BR_dark = 0.9994` would demand a
total width ~50x the mass — not a particle.

Holding `Gamma/M = 2.5%` as simulated, maximise `BR_SM x BR_dark` subject to
`6 BR_SM + BR_dark = 1`. The optimum is `BR_SM = 1/12`, `BR_dark = 1/2`:

- **g_q(max) = 0.1618**
- **417x the reference rate** — verified numerically at three masses, all
  giving 416.9x, and `BR_dark = 0.50000` at the optimum as predicted.

Above g_q = 0.1618 the rate no longer corresponds to the width the shapes were
generated with. `normalisation.is_benchmark_only()` flags this, and the GUI
marks that slider region rather than silently extrapolating.

### M5 — sigma/sigma_inclusive is the wrong anchor

Considered as a reference scale, and rejected. With
`sigma_inel(14 TeV) ~ 80 mb = 8e10 pb`, at mZ' ~ 1.2 TeV:

| coupling | sigma [pb] | sigma/sigma_inel |
|---|---|---|
| reference (0.0056) | 3.70e-3 | 4.6e-14 |
| ceiling (0.1618) | 1.54 | 1.9e-11 |
| benchmark (0.25) | 7.35 | 9.2e-11 |

A ratio of 1e-8 would mean ~800 pb at 1 TeV — comparable to ttbar, excluded long
ago. The interesting range is 1e-14 to 1e-10, which makes the ratio a poor knob:
it is fine for intuition, but g_q is what the field quotes and what limits are
set on.

### M6 — expected yields, and where the grid runs out

At 140 fb^-1, produced events before any acceptance:

| mZ' [GeV] | 500 | 1219 | 2208 | 4000 |
|---|---|---|---|---|
| N at g_q = 0.0056 | 16 630 | 517 | 30 | **0.66** |
| N at g_q = 0.1618 (ceiling) | 6.9e6 | 2.2e5 | 1.3e4 | 275 |

At the reference coupling the top of the mass grid produces **less than one
event** in all of Run 2 — the search is hopeless there regardless of selection.
At the ceiling it becomes viable across the whole grid. This is worth keeping in
view while sliding: a large S/sqrt(B) at 4 TeV and low g_q is an artefact of
dividing two numbers that are both essentially zero.

### M7 — 0.7% of missing grid points cost 22% of the interpolable volume

Measured while deciding whether to fit before the gap-fill finished. A cell of
the 6-D grid is usable only if all `2^6 = 64` corners are present:

| stream | points | cells interpolable |
|---|---|---|
| truth | 16 368 / 16 384 (99.90%) | 3801 / 3969 (**95.8%**) |
| Delphes | 16 269 / 16 384 (99.30%) | 3101 / 3969 (**78.1%**) |

A 99.3% complete point grid is a **78% complete interpolator**. The lesson
generalises: in six dimensions, point-completeness badly overstates coverage,
and scattered gaps are far more expensive than clustered ones.

The missing points concentrate at high mZ' (truth: all at mZ' >= 2972;
Delphes: 79 of 115 there) while spreading uniformly over the other five axes —
a walltime signature rather than a physics corner, since a physics failure would
cluster in `LambdaDQCD`/`mPiOverLambda` too. Gap-fill clusters `14812863`
(truth, 3 slices) and `14812864` (Delphes, 22 slices) were submitted on
2026-09-07; `scan_svj.py` skips points that already have finite params, so a
nearly-complete slice exits in minutes.

### M8 — the background is 99% QCD in any inclusive MET selection

First run of the loader over 321 of 340 shards (thinned to 20k rows per sample,
so the errors below are pessimistic). Expected events at 140 fb^-1:

| selection | B | MC error | QCD fraction |
|---|---|---|---|
| none | 2.03e11 | 0.7% | 95.1% |
| MET > 200 | 1.49e9 | 5.1% | 99.5% |
| MET > 400 | 1.11e7 | 13.5% | 97.5% |
| MET > 600 | 1.86e5 | 22.2% | 80.9% |

Not a bug: it is what [backgrounds.md](backgrounds.md) predicted. QCD's cross
section is enormous and an inclusive MET cut cannot suppress it; real searches
remove it with an angular cut, not a harder MET cut.

**Corrected 2026-09-07:** this entry originally said QCD's MET "is entirely
mismeasurement". That is wrong -- see M11. `MET` here is jet-based missing H_T,
and it is non-zero at truth level.

Two structural checks that the samples are behaving:

- **Each pT-hat slice has a hard MET ceiling** at roughly 1.5x its upper
  pT-hat bound (qcd_pt100_200 maxes at 312 GeV, qcd_pt200_400 at 461, and so
  on). The slicing is working; the high-MET tail rests on the high-pT slices,
  which is where the statistics are thinnest.
- **Real-MET samples order correctly** by slice: Z(->nunu) MET medians run
  118 / 230 / 446 / 871 GeV across the four slices, W(->lnu) lower at
  86 / 153 / 288 / 557 (the charged lepton carries away energy), ttbar 53.

### M9 — |dPhi|(MET, closest jet) is the discriminant, and it is stored SIGNED

Median |dPhi|(MET, closest jet), the variable that separates real MET from
mismeasured MET:

| QCD slices | 0.72, 0.49, 0.32, 0.22, 0.16, 0.11 |
|---|---|
| **Z(->nunu) slices** | **2.78, 2.17, 1.82, 1.63** |
| W(->lnu) slices | 1.10, 0.57, 0.29, 0.15 |

Exactly the expected physics: QCD's fake MET points *along* a jet, so dPhi is
small; a Z recoiling against a jet puts MET *opposite* it, near pi.

**But `dPhiMETclose` is stored signed, in (-pi, pi].** The observable registry
applies `abs_value` as the first pipeline step and labels it `|dPhi|`, so the
fitted quantity is the absolute value — while the raw TSV column, and hence the
GUI's cut slider, is the signed one. A naive `dPhiMETclose > 0.4` cut therefore
discards the entire negative half rather than making an angular selection: it
halved B (1.16e8 -> 5.9e7) while leaving the QCD fraction at 99%, which is the
signature of cutting on sign rather than on physics. Cutting on `abs()` gives
the intended behaviour. This caught me out while writing M8 and would catch a
GUI user out the same way; the cut panel needs to apply `abs()` for any
observable whose pipeline starts with `abs_value`.

### M10 — do not thin the background cache

`build_background_cache(max_rows_per_sample=...)` thins unbiasedly and scales
the weights up to compensate, but the variance cost is real: at MET > 600 the
MC error was 22% with 20k rows per sample, and `significance()` flags anything
past 30% as unreliable. The samples that would be thinned hardest are the
low-pT QCD slices, which have the worst effective luminosity to begin with
(0.0007 fb^-1). Thinning is therefore **off by default**; the option exists only
for machines that cannot hold the full cache.

Reading all 321 shards takes ~64 s, so the cache is built once and loaded
instantly thereafter.

### M11 — `MET` is jet-based missing H_T, and QCD's is NOT all detector

Prompted by the question "assuming Delphes were perfect, does the QCD error
still matter?". Checking the definition first turned out to matter more than
answering it. From `svj_observables_common.h`:

```cpp
for (const auto& jet : raw_jets) {
    if (std::fabs(jet.eta()) >= ETA_MAX) continue;   // jets outside acceptance dropped
    ...
      if (c.user_index() >= TAG_INV) continue;       // invisible constituents excluded
    ...
    if (vis_pt < VIS_PT_MIN) continue;               // jets under vis_jet_pt_min dropped
    evt_vis_px += vis_px;
}
double met = std::sqrt(evt_vis_px*evt_vis_px + evt_vis_py*evt_vis_py);
```

So `MET` is the imbalance of **reconstructed jets above threshold and inside
|eta|** -- missing H_T, not calorimeter MET. It is non-zero whenever energy
escapes the jet collection: jets below `vis_jet_pt_min`, jets outside
`ETA_MAX`, or soft unclustered radiation. Excluding invisible constituents
*inside* jets is what makes it the right variable for SVJ.

Measured on qcd_pt400_800, truth (2000 events, local) against Delphes (50k):

| quantile | truth | Delphes |
|---|---|---|
| median | 25.1 | 54.9 |
| 90% | 102.1 | 165.1 |
| 99% | 455.0 | 434.2 |
| frac MET > 100 | **10.40%** | 23.68% |

**Truth QCD MHT is not zero.** Its fake MET therefore has three sources, only
one of which is the detector:

1. **jet acceptance / threshold / unclustered energy** — present at truth
   level, purely a definition effect;
2. **detector mismeasurement** — only the truth-to-Delphes *difference*;
3. real neutrinos from heavy-flavour decay — small, already in truth.

This weakens the attribution behind the "MET >~ 800" recommendation in M8: the
QCD fraction remains the component modelled worst (LO, no merging, no pileup),
but describing it as "a readout of the calorimeter parametrisation" overstated
it. The high-MET comparison above is *not* yet conclusive — 2.5% of 2000 truth
events is 50 events and the >600 GeV bin holds 3 — which is why the truth QCD
stream was generated (cluster `14816219`, 6 samples x 20 shards, output
`/eos/user/l/lbojorqu/svj/background/tsv_truth`). Redo this table from it
before drawing any conclusion about the tail.

---

## Background weights

From `background/samples.json`, per sample: `w = sigma_fb / n_generated` in fb,
so expected events at luminosity L is `L x sum(w_i)` over rows passing a
selection.

Two subtleties, both handled in `build_background_cache()`:

1. **`n_generated` counts shards that actually landed**, not the nominal total.
   A sample missing 3 of 20 shards would otherwise come out 15% low.
2. **Rows the generator dropped (no jets found) must not enter that
   denominator.** They are real acceptance losses; summing the surviving
   weights is exactly what yields the post-acceptance cross section. Only
   events PYTHIA was *asked* to generate belong in the denominator.

---

## What the number means, and when to distrust it

`normalisation.significance()` returns both `s_over_sqrt_b` and the Asimov
significance `sqrt(2[(s+b)ln(1+s/b) - s])`. They agree when S << B (at
S=50, B=2500: 1.000 vs 0.997) and diverge once the signal is not small, with
S/sqrt(B) the optimistic one. S/sqrt(B) is displayed because it is what was
asked for and what people recognise; the Asimov value is the one to trust when
they disagree.

**The MC error on B is reported alongside, and it is not sqrt(N).** For weighted
events it is `L x sqrt(sum w^2)`. This matters concretely: the lowest QCD slice
has an effective luminosity of 0.0007 fb^-1, so at 140 fb^-1 each generated
event stands for ~200 000 real ones, and a tight signal region can rest on one
or two events carrying enormous weight. `significance()` sets `unreliable` when
the MC error exceeds 30% of B — at which point the number is noise, however
precise it looks.

Beyond MC statistics, everything in
[backgrounds.md](backgrounds.md#what-this-will-get-wrong) still applies: LO
normalisations with K-factors of 1.2-2, no jet merging, no pileup, and a
parametrised detector whose fake-MET tail is the least trustworthy part of the
whole estimate. **This is a tool for watching how S/sqrt(B) responds as the
physics sliders move, not for setting a limit.**

---

## Open: the fit and interpolation choice

Everything above uses the **incumbent** pipeline: per-observable Box-Cox to
`gennorm` marginals, a Gaussian copula for correlations, and **linear**
interpolation of the fitted parameters across the physics grid. That is a
deliberate hold, not a conclusion — S/sqrt(B) had to be built on something
stable, and the incumbent is what the GUI already trusted.

Three candidates remain unsettled, and the evidence for each is the measurement
log in [symbolic-regression.md](symbolic-regression.md):

| candidate | status |
|---|---|
| **gennorm + linear** (incumbent) | in use here. Note M-series finding that the shipped linear interpolator scored *worse* than nearest-grid on JS (0.0842 vs 0.0758) and MMD (0.0208 vs 0.0122) — the interpolation, not the marginal, is the weak part. |
| **Design A** (sparse basis over physics axes) | per-observable hybrid gave -13.2% vs linear. Explicitly deferred by the user. |
| **Design B** (learned quantile function, "Q everywhere") | -31.5% mean JS vs `gennorm` with fold and atom handling. Stage 3 (mapping the 9 quantile coefficients over physics) not done. |
| **Design B hybrid** | per-observable selection bought only 0.5% over Q-everywhere, and 5 of 26 picks disagreed between selection and evaluation sets. User noted a possible preference for the hybrid once that selection noise is reduced. |

Nothing in `normalisation.py` depends on which wins: it consumes samples and
weights, not fit parameters. Swapping the interpolator changes the *signal*
shapes and hence `eps_S`, but no rate machinery.

---

## Which background cache gets loaded

`load_background()` tries three locations in order, and records the choice in
`LOADED_CACHE_PATH` / `LOADED_CACHE_IS_DEMO`:

| order | path | in git? | notes |
|---|---|---|---|
| 1 | `background/background_events.npz` | no (gitignored) | a full cache dropped into a checkout |
| 2 | `/eos/user/l/lbojorqu/svj/background/background_events.npz` | no | canonical, 1.27 GB, CERN only |
| 3 | `background/background_events_demo.npz` | **yes**, ~27 MB | thinned: 20k rows per sample, 340k events |

The demo is last so that anyone holding the real cache gets the real cache. It
exists so a fresh clone can drive the background panel and S/sqrt(B) without
EOS access or regenerating 17M events (~57 core-hours). Its **total** B is
correct — thinning scales the weights up by exactly the factor discarded — but
its MC error is inflated by roughly sqrt(50), so it is a demonstration, not a
measurement. The GUI shows a red banner whenever it is in use.

### The zero-background case

With a thinned cache an empty selection is easy to reach, so it is handled
explicitly rather than left to arithmetic. `significance()` sets
`no_background=True` when B <= 0 and the GUI prints

```
S/sqrtB = undetermined (no background events pass the cuts)
```

rather than `inf`. That matters: formally S/sqrt(0) is infinite, but it means
"the background here is unsampled, so this is unknown", not "infinitely
significant" — and printed as `inf` it reads as a result. The Asimov value is
suppressed in that state for the same reason.

Note also that the cut sliders are bounded by the **model's** percentile-1/99
range, so a single cut often cannot empty the background at all: the demo
background reaches MET = 3641 GeV while the 4-axis example's MET slider caps
near 1130.
