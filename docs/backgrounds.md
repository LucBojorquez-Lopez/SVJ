# Standard-Model backgrounds with PYTHIA + Delphes

A first, deliberately rough background estimate: generate the dominant SM
processes with the PYTHIA we already have, push them through the same Delphes
card the signal uses, and compute the same 28 observables. The product is a set
of **observable densities per background, with relative normalisations** — not a
sensitivity estimate.

Scope is set honestly up front: this is LO matrix elements plus parton shower,
no jet merging, no pileup, and a simple detector card. Normalisations are good
to roughly a factor of two, shapes better than that for most observables and
poor for one that matters (see "What this will get wrong").

---

## A short glossary

Because this doc uses collider-simulation vocabulary the rest of the repository
does not.

| term | meaning here |
|---|---|
| **matrix element (ME)** | the exact quantum-mechanical calculation for a specific few-particle process, e.g. `qqbar -> Zg`. Accurate, but only for a fixed, small number of outgoing particles. |
| **parton shower (PS)** | the approximate, probabilistic dressing of that hard process with many additional soft/collinear gluons and quarks. Cheap, and where most of an event's particles come from. |
| **LO / NLO** | leading / next-to-leading order in perturbation theory. PYTHIA's built-in processes are LO. NLO changes normalisations by ~20-100% (the "K-factor"). |
| **pT-hat** | the transverse momentum of the hard scattering itself, before showering. PYTHIA's main handle for selecting how energetic an event is. |
| **cross section (sigma)** | the intrinsic rate of a process, in picobarns (pb). Expected events = sigma x luminosity. |
| **luminosity (L)** | how much data, in inverse femtobarns (fb^-1). LHC Run 2 delivered ~140 fb^-1; HL-LHC targets ~3000 fb^-1. |
| **branching ratio (BR)** | the fraction of decays going to a particular final state, e.g. BR(Z -> neutrinos) = 0.20. |
| **MET** | missing transverse energy: the momentum imbalance that signals invisible particles. The whole point of an SVJ search, and the axis every background is judged on. |
| **merging** | stitching together ME calculations with different numbers of hard jets so the jet-multiplicity spectrum is right. PYTHIA supports it but needs an external ME generator. |
| **pileup** | extra proton-proton collisions in the same bunch crossing, which degrade MET resolution. |

---

## Why these processes

A semi-visible-jet signal is *jets plus genuine MET*. A background matters if it
can produce that, so each entry below is chosen for a specific mechanism, not
because it is large in general.

| # | background | PYTHIA processes | why it enters |
|---|---|---|---|
| 1 | **Z(->nunu) + jets** | `WeakBosonAndParton:qqbar2gmZg`, `qg2gmZq` | **Irreducible.** Neutrinos are genuinely invisible, so a Z recoiling against a jet gives exactly the signature with no mismeasurement required. Usually the leading background in any MET+jets search. |
| 2 | **W(->lnu) + jets** | `WeakBosonAndParton:qqbar2Wg`, `qg2Wq` | The neutrino gives real MET; enters whenever the charged lepton is missed — out of acceptance, too soft, or a hadronically-decaying tau. |
| 3 | **QCD multijet** | `HardQCD:all` | Enormous cross section, so even a small probability of apparent MET fills the low-MET region and leaks upward. **Not purely a detector effect** — see normalisation.md M11: `MET` is jet-based missing H_T and is already non-zero at truth level (median 25 GeV for pT-hat 400-800) from jets below threshold, jets outside eta acceptance and unclustered energy. Delphes adds mismeasurement on top of that. |
| 4 | **ttbar** | `Top:gg2ttbar`, `Top:qqbar2ttbar` | Real MET from W -> lnu in the top decays, plus high jet multiplicity and b-jets. |
| 5 | **single top** | `Top:qq2tq(t:W)`, `Top:ffbar2tqbar(s:W)` | Same mechanism as ttbar at ~1/3 the rate. |
| 6 | **diboson** | `WeakDoubleBoson:all` (`ffbar2WW`, `ffbar2ZW`, `ffbar2gmZgmZ`) | Small, but WZ/ZZ with one Z -> nunu is kinematically signal-like, so it survives cuts better than its cross section suggests. |

**Deliberately excluded, and why**

- **`SoftQCD` / minimum bias** — no hard scale; cannot produce the jets a signal
  region requires. Only relevant as pileup, which this setup does not model.
- **`PromptPhoton`** — not a background to a MET search (photons are visible).
  It is the standard *control region* for calibrating the Z -> nunu estimate
  from data, so worth adding later for that purpose, not now.
- **`WeakSingleBoson:ffbar2gmZ` / `ffbar2W`** — inclusive V production with no
  ME-level parton. The recoil jet would come entirely from the shower, which is
  exactly the regime the shower approximates worst. `WeakBosonAndParton` puts
  the jet in the matrix element instead. **This is the single most important
  process choice here.**
- **`Top:gmgm2ttbar`, `ggm2ttbar`** — photon-initiated, negligible at a hadron
  collider.
- **tW associated single-top production** — physically relevant at roughly the
  size of t-channel, but **PYTHIA has no such process**. A genuine gap in this
  setup; it needs an external ME generator.

---

## Measured cross sections

Not quoted from literature — measured with this exact PYTHIA build at
`Beams:eCM = 14000` (matching the signal), so they are internally consistent
with everything else in this repository.

| sample | pT-hat [GeV] | sigma [pb] | L_eff at 1M events [fb^-1] | weight/event [fb] |
|---|---|---|---|---|
| QCD | 100-200 | 1.398e6 | **0.00072** | 1398 |
| QCD | 200-400 | 6.655e4 | 0.015 | 66.6 |
| QCD | 400-800 | 2393 | 0.42 | 2.39 |
| QCD | 800-1500 | 56.6 | 17.7 | 0.057 |
| QCD | 1500-3000 | 0.817 | 1224 | 8.2e-4 |
| QCD | 3000+ | 9.5e-4 | 1.05e6 | 9.5e-7 |
| Z(->nunu)+jet | 100-200 | 128.9 | 7.8 | 0.129 |
| Z(->nunu)+jet | 200-400 | 12.62 | 79 | 0.0126 |
| Z(->nunu)+jet | 400-800 | 0.706 | 1417 | 7.1e-4 |
| Z(->nunu)+jet | 800+ | 0.0231 | 4.3e4 | 2.3e-5 |
| W(->lnu)+jet | 100-200 | 468.1 | 2.1 | 0.468 |
| W(->lnu)+jet | 200-400 | 45.5 | 22 | 0.0455 |
| W(->lnu)+jet | 400-800 | 2.65 | 377 | 0.00265 |
| W(->lnu)+jet | 800+ | 0.090 | 1.1e4 | 9.0e-5 |
| ttbar | inclusive | 604.9 | 1.7 | 0.605 |
| single top | inclusive | 189.7 | 5.3 | 0.190 |
| diboson | inclusive | 110.1 | 9.1 | 0.110 |

Sanity checks that these are believable: W/Z ~ 3.6 at matched pT-hat, as
expected from the relative couplings and branching fractions; ttbar at 605 pb LO
against a known NNLO value near 1000 pb at 14 TeV, i.e. a K-factor of ~1.6,
which is the right size for LO.

**The last two columns are the real content of this table.** `L_eff` is the
luminosity a million generated events actually represents. Note the top row:
the low-pT QCD slice corresponds to **0.0007 fb^-1**, so at even 140 fb^-1 each
generated event stands for ~200 000 real ones. No amount of CPU fixes that —
it is why real analyses estimate QCD from data rather than from simulation.

---

## Every other choice, and why

**Generator: PYTHIA 8.317 alone.** Already built, already integrated, no new
dependency, and it reuses the entire existing pipeline. The cost is LO+PS with
no merging (see limitations).

**Detector: `svj_regression_delphes` with `svj_delphes_particles.tcl`** — the
same binary and the same card as the signal Delphes stream. Consistency matters
more than absolute realism: signal and background must be smeared identically or
any comparison is meaningless. It is also mandatory rather than optional here,
because QCD's entire contribution *is* a detector effect.

**Centre of mass: 14 TeV**, matching `svj_regression.cc`. Not because 14 TeV is
what ATLAS has run at (Run 2 was 13, Run 3 is 13.6) but because the signal was
generated there and the two must match.

**Observables: unchanged.** `svj_observables_common.h` contains zero references
to the Hidden Valley — verified — so it is already process-agnostic. A
background generator is `svj_regression_delphes.cc` with `setupPythia()`
replaced and nothing else touched, which also guarantees signal and background
observables are computed by identical code.

**Decay steering:** `23:onMode = off` + `23:onIfAny = 12 14 16` for Z -> nunu;
`24:onMode = off` + `24:onIfAny = 11..16` for W -> lnu; `WeakZ0:gmZmode = 2` to
select the Z rather than the photon interference term, which otherwise dominates
at low mass and is not what "Z+jets" means.

**Weighting — verified, not assumed.** Restricting the Z decay modes changes
`sigmaGen` by a factor 0.2008 against a PDG BR(Z -> invisible) of 0.200, so
**PYTHIA rescales the cross section by the open branching fraction itself** and
no manual BR correction should be applied. This was checked directly rather than
taken on faith, because applying it twice would be a silent factor-5 error.

**pT-hat slices, not event-by-event biasing.** Both are available
(`PhaseSpace:bias2Selection` exists). Slices win here for one concrete reason:
a slice has a *single* cross section, so the weight is one number per sample and
can live in metadata. Biasing produces a different weight for every event, which
would require a weight column in the TSV and in the raw store — a change to code
shared with the signal path, for no benefit at this stage.

**Slice boundaries 100 / 200 / 400 / 800 / 1500 / 3000 GeV.** Roughly one per
decade-and-a-half of cross section, keeping the weight ratio between adjacent
slices near 20-40x. Starting at 100 rather than 0 avoids both the low-pT
divergence and a vast sample that cannot produce a signal-region event.

**One million events per sample, 17 samples.** ~47 core-hours at the measured
Delphes rate of ~100 events/s — about **5% of the signal scan's 934 core-hours**,
so this is cheap and can be scaled up if the tails turn out thin.

**Reuse `condor/tsv_delphes.sub`.** The `tsv` workflow in `condor/svj_job.sh`
already runs `svj_regression_delphes` with per-job `seed_offset` and merges
shards by `merge_svj_tsv.sh`. Background generation is that workflow with a
different cfg per sample; no new Condor machinery.

---

## What this will get wrong

Stated plainly, because these determine how far the result can be pushed.

1. **QCD's fake MET is the weakest part, and QCD is often the largest
   background.** Its entire contribution comes from mismeasuring jets, so it is
   a direct readout of the calorimeter response model. The Delphes card is a
   simple parametrisation; ATLAS's fake-MET tail depends on detailed jet
   response, cracks, dead material and the actual MET reconstruction algorithm.
   Expect the shape to be qualitatively right and the tail normalisation to be
   wrong, possibly by a lot.
2. **No jet merging.** V+jets with 2 or more hard jets comes from the shower.
   Since SVJ selections cut on jet multiplicity and topology, this biases exactly
   what the analysis uses. `WeakBosonAndParton` fixes the *first* jet only.
3. **LO normalisations.** K-factors of 1.2-2 across these processes; relative
   ratios between different processes are correspondingly uncertain.
4. **No pileup.** Pileup degrades MET resolution, which is precisely the axis
   that separates signal from QCD.
5. **No tW single top**, as noted.
6. **Low-pT QCD statistics are irreducibly poor** (`L_eff` = 0.0007 fb^-1).

---

## If not PYTHIA + Delphes

Roughly in order of increasing fidelity and effort.

**MadGraph5_aMC@NLO -> LHE -> PYTHIA -> Delphes.** The standard phenomenology
chain. MadGraph computes multi-parton matrix elements (V+0,1,2,3 jets) and
writes Les Houches events; PYTHIA showers them with CKKW-L or MLM merging to
avoid double counting. This build already supports the input side
(`Beams:frameType = 4`, `Beams:LHEF`) and the merging side
(`Merging:doKTMerging`, `doPTLundMerging`, `doCutBasedMerging`,
`doUNLOPSTree`, `Merging:Process`, `TMS`, `nJetMax`) — **only MadGraph itself is
missing.** This is the correct fix for limitation 2 and the natural next step if
the rough estimate looks promising.

**Sherpa.** ME generation, showering and merging in one program, with merging
particularly well developed. Fewer moving parts than the MadGraph chain, at the
cost of another large dependency to build.

**POWHEG-BOX.** NLO matched to parton shower for specific processes (ttbar,
single top, V+jet). The standard choice when the *normalisation* of one
background must be right, rather than the multi-jet shape.

**Herwig.** An independent shower and hadronisation model. Its real use is as a
*systematic*: running the same analysis through PYTHIA and Herwig brackets the
modelling uncertainty.

**ATLAS full simulation (Athena + Geant4), or AtlFast3.** The only way to get
the detector genuinely right, and the only thing acceptable in a real ATLAS
result. Requires ATLAS membership, the Athena release, and orders of magnitude
more CPU. AtlFast3 is the fast parametrisation that ATLAS itself uses when full
Geant4 is too expensive.

**Existing public samples instead of generating.** ATLAS and CMS Open Data both
publish simulated background samples with official normalisations. If the goal
is realistic background *shapes* rather than a self-consistent private study,
this can be far cheaper and considerably more accurate than anything generated
here — worth checking before scaling this campaign up.

**Data-driven estimation.** What an actual analysis does: measure each
background in a control region and transfer it to the signal region with a
simulation-derived ratio. Simulation then only needs to model the *ratio*, which
is far more forgiving than modelling the absolute rate. If this work is ever
aimed at a real limit rather than at studying observable separation, this is the
route — and it changes what the simulation is for.

---

## Generation outcome (2026-09-07)

339 of 340 shards landed on the first pass plus one gap-fill. Two failure modes,
both understood:

- **19 shards** died instantly with `unknown workflow 'background'` — they
  launched within seconds of `condor_submit` and read a stale `svj_job.sh` from
  the EOS cache, before the version adding the `background)` case was visible on
  the worker. Nothing wrong with the config; resubmitted and fine. After editing
  `svj_job.sh`, wait a moment before submitting.
- **`diboson_4`** hit the Delphes `SimpleCalorimeter::Process()` segfault
  documented in [raw-store.md](raw-store.md#the-delphes-simplecalorimeter-segfault-2026-09-07).
  Because the shard index seeds PYTHIA, re-running that shard reproduces the
  crash exactly; it was resubmitted as shard 20 for a fresh seed.

Cross sections in `samples.json` are unaffected by either — they were measured
separately, before any of this.
