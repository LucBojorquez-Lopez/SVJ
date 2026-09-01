// svj_delphes_test.cc
// ============================================================================
// Stage 0 smoke test for a DELPHES detector-level stream, run alongside (never
// touching) the existing truth-level pipeline in svj_regression.cc.
//
// WHAT THIS IS
// ------------
// A deliberately minimal, single-threaded, single-physics-point driver that,
// for EVERY Pythia8 event, computes the same small observable subset TWICE:
//   (a) from truth final-state particles, clustered exactly the way
//       svj_regression.cc does it, and
//   (b) from Delphes-reconstructed particle-flow ("EFlow") candidates,
//       via an embedded (in-process, no ROOT file ever written) Delphes
//       module chain restricted to PARTICLE-LEVEL RECONSTRUCTION ONLY --
//       tracking efficiency + momentum smearing, calorimeter granularity +
//       energy smearing, particle-flow merging. No jet-finder, no MissingET
//       module, no b-tagging, no pileup: see svj_delphes_particles.tcl for
//       the exact module chain.
// Both (a) and (b) are computed from the IDENTICAL underlying Pythia event,
// in the same loop iteration, and written side by side in one combined TSV
// row (columns suffixed _truth / _delphes). This guarantees exact per-event
// correspondence by construction -- no seed-matching or discard-desync
// guesswork required, which running svj_regression.cc and this binary as two
// independent processes could never guarantee (svj_regression.cc's usual
// nWorkers=16 config alone means 16 different Pythia seeds are interleaved
// in its output, and each binary independently discards events with zero
// passing jets, so two separately-run TSVs' row indices do not correspond to
// the same event beyond, at best, a small and silently-decaying prefix).
//
// `jetR` stays a plain scan-config parameter, never baked into a Delphes
// .tcl card -- both the truth-side and Delphes-side clustering below use the
// SAME jet_def built from the same jetR, via the same clusterAndExtract()
// helper, applied to two different input particle lists.
//
// PER-EVENT CONSTITUENT DUMP (`dump_events` cfg key)
// ----------------------------------------------------
// The main per-event loop only ever computes/writes scalar summary
// quantities (leadVisPt, MET, ...) -- it never keeps a jet's full
// constituent list around, since that's unnecessary weight for every one of
// nEvent events. To investigate a handful of specific events in detail
// (e.g. the largest leadVisPt outliers, via plot_lego.py), set the cfg key
// `dump_events` to a comma-separated list of event indices (the `eventIndex`
// column already written for every row) and re-run: Pythia8 with a fixed
// seed is fully deterministic, so re-running regenerates the exact same
// event at the same iEvent, and this time the requested indices' leading-jet
// constituents (eta, phi, pt -- both truth and Delphes side) are ALSO
// written to constitTruthFile / constitDelphesFile, one row per constituent.
// Leaving `dump_events` empty (the default) skips this entirely -- a normal
// run touches neither of those two files.
//
// WHAT THIS IS NOT
// ----------------
// Not a scan worker (single-threaded, one physics point, no grid, no
// checkpointing), not a faithful port of all ~27 svj_regression.cc
// observables (only a small representative subset: leadVisPt, MET,
// leadJetMass, nConst, nJets -- enough to sanity-check that Delphes smearing
// looks physically sane before investing in anything bigger), and not wired
// into scan_svj.py / observables.py / the GUI in any way.
//
// WHY THE PHYSICS SETUP BELOW IS A COPY, NOT A #include
// ------------------------------------------------------
// readConfig/cfgDouble/cfgInt/cfgStr/rs/setupPythia are an intentional,
// line-for-line copy of the equivalent code in svj_regression.cc, and the
// truth-side particle selection in buildTruthParticles() below is the same
// TAG_VIS/TAG_MUON logic svj_regression.cc uses (inlined, without the full
// tag-enum machinery, since this driver only needs jet kinematics, not
// per-constituent substructure). The alternative (#include-ing
// svj_regression.cc, which already defines its own main()) creates fragile
// build coupling between the truth and Delphes binaries. This duplication is
// exactly the "drift risk" flagged in the Stage-0 plan (docs/setup_delphes.md)
// as the reason a later, separately approved refactor should share this code
// between both workers -- it is not an oversight.
//
// VERIFIED DELPHES CALL CONVENTION -- READ BEFORE CHANGING ANYTHING BELOW
// ------------------------------------------------------------------------
// Delphes' actual per-module reconstruction logic (ParticlePropagator's
// propagation, each Efficiency's pass/fail roll, each MomentumSmearing's
// smear, SimpleCalorimeter's tower simulation, EFlowMerger's merge -- i.e.
// the entire physics content of a Delphes run) only executes via ROOT's
// TTask sub-task recursion, which is triggered by calling
// InitTask()/ProcessTask()/FinishTask() -- NOT the similarly-named bare
// Init()/Process()/Finish(). This was verified directly against this
// project's actual installed copy (not a remote "latest docs" guess):
//   delphes3.5.1/modules/Delphes.cc      -- Delphes::Process() is a literal
//                                            empty no-op ({}); Delphes::Init()
//                                            only BUILDS the module list
//                                            (via Add(task) for each entry in
//                                            the .tcl card's ExecutionPath),
//                                            it never runs any module itself.
//   delphes3.5.1/external/ExRootAnalysis/ExRootTask.cc
//                                       -- InitTask()/ProcessTask()/FinishTask()
//                                          call ExecuteTask(option), ROOT
//                                          TTask's own method that runs this
//                                          task's Init()/Process()/Finish()
//                                          AND THEN recurses into every
//                                          sub-task added by Delphes::Init()
//                                          above -- this recursion is the only
//                                          place any module's real Process()
//                                          (e.g. ParticlePropagator::Process())
//                                          actually runs.
// Calling bare Init()/Process()/Finish() instead would compile, link, and run
// without any error or crash -- it would just silently do nothing per event,
// which is a much worse failure mode than a build error. Do not "simplify"
// the calls below without re-reading this note.
// ============================================================================

#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"

#include "classes/DelphesClasses.h"
#include "classes/DelphesFactory.h"
#include "modules/Delphes.h"
#include "ExRootAnalysis/ExRootConfReader.h"
#include "ExRootAnalysis/ExRootTreeWriter.h"

#include "TDatabasePDG.h"
#include "TParticlePDG.h"

#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <set>
#include <memory>
#include <string>
#include <cmath>
#include <limits>
#include <filesystem>
#include <map>
#include <iostream>
#include <iomanip>

using namespace Pythia8;

// ── Config-file reader ───────────────────────────────────────────────────
// Copied verbatim from svj_regression.cc (see file header for why).
static std::map<std::string,std::string> readConfig(const std::string& path) {
  std::map<std::string,std::string> cfg;
  std::ifstream f(path);
  if (!f.is_open()) return cfg;
  std::string line;
  while (std::getline(f, line)) {
    auto start = line.find_first_not_of(" \t");
    if (start == std::string::npos || line[start] == '#') continue;
    auto eq = line.find('=');
    if (eq == std::string::npos) continue;
    std::string key = line.substr(start, eq - start);
    std::string val = line.substr(eq + 1);
    auto kend = key.find_last_not_of(" \t");
    if (kend != std::string::npos) key = key.substr(0, kend + 1);
    auto vstart = val.find_first_not_of(" \t");
    if (vstart == std::string::npos) continue;
    val = val.substr(vstart);
    auto comment = val.find('#');
    if (comment != std::string::npos) val = val.substr(0, comment);
    auto vend = val.find_last_not_of(" \t\r\n");
    if (vend != std::string::npos) val = val.substr(0, vend + 1);
    cfg[key] = val;
  }
  return cfg;
}

static double cfgDouble(const std::map<std::string,std::string>& cfg,
                        const std::string& key, double def) {
  auto it = cfg.find(key);
  return (it != cfg.end()) ? std::stod(it->second) : def;
}
static int cfgInt(const std::map<std::string,std::string>& cfg,
                  const std::string& key, int def) {
  auto it = cfg.find(key);
  return (it != cfg.end()) ? std::stoi(it->second) : def;
}
static std::string cfgStr(const std::map<std::string,std::string>& cfg,
                           const std::string& key, const std::string& def) {
  auto it = cfg.find(key);
  return (it != cfg.end()) ? it->second : def;
}

// Parse a comma-separated list of event indices (e.g. "4237,891,55") into a
// set for fast per-event membership checks. Empty/whitespace-only string ->
// empty set (the default -- no events dumped).
static std::set<int> parseEventIndexList(const std::string& s) {
  std::set<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    auto start = tok.find_first_not_of(" \t");
    if (start == std::string::npos) continue;
    auto end = tok.find_last_not_of(" \t");
    tok = tok.substr(start, end - start + 1);
    if (!tok.empty()) out.insert(std::stoi(tok));
  }
  return out;
}

// ── Physics parameters ───────────────────────────────────────────────────
// Same defaults as svj_regression.cfg's defaults, minus everything Stage 0
// doesn't need (nWorkers/seedOffset/jetsVisOnly/dijetOnly/tsvKinFile -- no
// threading, no dijet-only filter, no separate kinematics TSV here).
static double mZ         = 2000.0;
static double mq         =    4.0;
static double mPi        =    8.0;
static double mRho       =   15.5;
static double rinv_pion  =    0.3;   // dark pion invisible BR
static double rinv_rho   =    0.3;   // dark rho invisible BR
static double Brmu       =    0.3;   // dark rho -> mu+mu- BR fraction (of visible decays)
static double alphaD     =    0.4;
static int    nEvent     = 3000;     // Stage-0 default: an order of magnitude below the usual 50000
static double jetR       =    1.0;
static double LambdaDQCD =    5.0;
static double visJetPtMin = 20.0;    // min visible pT (GeV) for a jet to be kept -- same acceptance
                                     // proxy svj_regression.cc uses, for apples-to-apples comparison
static std::string tsvFile     = "simulated/tsv/jets_delphes_paired.tsv";
static std::string delphesCard = "src/generate_events/svj_delphes_particles.tcl";

// ── Per-event constituent dump (for investigating specific events, e.g. via
//    plot_lego.py) -- empty by default, so a normal run touches neither of
//    these files. See file header "PER-EVENT CONSTITUENT DUMP" section.
static std::string dumpEventsStr        = "";   // comma-separated event indices, e.g. "4237,891,55"
static std::string constitTruthFile     = "simulated/tsv/jets_delphes_constituents_truth.tsv";
static std::string constitDelphesFile   = "simulated/tsv/jets_delphes_constituents_delphes.tsv";

static void rs(Pythia& p, const std::string& key, double val) {
  std::ostringstream oss;
  oss << key << " = " << val;
  p.readString(oss.str());
}

// Copied verbatim from svj_regression.cc's setupPythia() -- same HiddenValley
// SVJ physics model (Z' -> dark quarks -> dark pions/rhos -> visible SM decays
// + invisible dark/neutrino decays). See svj_regression.cc for the physics
// commentary; nothing here has been changed.
static bool setupPythia(Pythia& pythia, int seed) {
  pythia.readString("Beams:eCM = 14000.");
  pythia.readString("HiddenValley:ffbar2Zv = on");

  rs(pythia, "4900023:m0",   mZ);
  rs(pythia, "4900023:mMin", mZ - 0.05*mZ);
  rs(pythia, "4900023:mMax", mZ + 0.05 * mZ);
  rs(pythia, "4900023:mWidth", 0.025 * mZ);
  pythia.readString("4900023:doForceWidth = on");

  pythia.readString("4900023:oneChannel = 1 0.9994 102 4900101 -4900101");
  pythia.readString("4900023:addChannel = 1 0.0001 102 1 -1");
  pythia.readString("4900023:addChannel = 1 0.0001 102 2 -2");
  pythia.readString("4900023:addChannel = 1 0.0001 102 3 -3");
  pythia.readString("4900023:addChannel = 1 0.0001 102 4 -4");
  pythia.readString("4900023:addChannel = 1 0.0001 102 5 -5");
  pythia.readString("4900023:addChannel = 1 0.0001 102 6 -6");
  pythia.readString("4900023:onMode = off");
  pythia.readString("4900023:onIfAny = 4900101 4900102");

  pythia.readString("HiddenValley:Ngauge = 3");
  pythia.readString("HiddenValley:nFlav = 2");
  pythia.readString("HiddenValley:spinFv = 0");
  pythia.readString("HiddenValley:FSR = on");
  pythia.readString("HiddenValley:fragment = on");
  pythia.readString("HiddenValley:setLambda = on");
  pythia.readString("HiddenValley:alphaOrder = 0");
  rs(pythia, "HiddenValley:Lambda",   LambdaDQCD);
  rs(pythia, "HiddenValley:pTminFSR", 1.1 * LambdaDQCD);
  pythia.readString("HiddenValley:probVector = 0.75");

  rs(pythia, "HiddenValley:alphaFSR", alphaD);
  rs(pythia, "4900101:m0", mq);
  rs(pythia, "4900102:m0", mq);
  rs(pythia, "4900111:m0", mPi);
  rs(pythia, "4900211:m0", mPi);

  pythia.readString("51:m0 = 0.0");
  pythia.readString("51:isResonance = false");
  pythia.readString("51:mayDecay = false");

  rs(pythia, "4900113:m0", mRho);
  rs(pythia, "4900213:m0", mRho);

  pythia.readString("53:m0 = 0.0");
  pythia.readString("53:isResonance = false");
  pythia.readString("53:mayDecay = off");

  for (int id : {4900001,4900002,4900003,4900004,4900005,4900006,
                 4900011,4900012,4900013,4900014,4900015,4900016}) {
    std::ostringstream oss;
    oss << id << ":m0 = 50000";
    pythia.readString(oss.str());
  }

  {
    std::ostringstream vis, inv;
    vis << 1. - rinv_pion;
    inv << rinv_pion;
    pythia.readString("4900111:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("4900111:addChannel = 1 " + vis.str() + " 91 4 -4");
    pythia.readString("4900211:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("4900211:addChannel = 1 " + vis.str() + " 91 4 -4");
    pythia.readString("-4900211:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("-4900211:addChannel = 1 " + vis.str() + " 91 4 -4");
  }
  {
    // Brmu is the fraction of the VISIBLE dark-rho decays going to mu+mu-.
    // rho_mu_br   = Brmu * (1 - rinv_rho)  -> mu+mu-
    // rho_bott_br = (1 - Brmu) * (1 - rinv_rho)  -> bb-bar
    // These always sum to (1 - rinv_rho), together with rinv_rho giving total = 1.
    double rho_mu_br   = Brmu * (1.0 - rinv_rho);
    double rho_bott_br = (1.0 - Brmu) * (1.0 - rinv_rho);
    std::ostringstream lep, bott, inv2;
    inv2 << rinv_rho;
    bott << rho_bott_br;
    lep  << rho_mu_br;

    pythia.readString("4900113:onMode = off");
    pythia.readString("4900213:onMode = off");
    pythia.readString("-4900213:onMode = off");
    pythia.readString("-4900213:addChannel = 1 " + lep.str()  + " 91 13 -13");
    pythia.readString("4900213:addChannel = 1 "  + lep.str()  + " 91 13 -13");
    pythia.readString("4900113:addChannel = 1 "  + lep.str()  + " 91 13 -13");
    pythia.readString("-4900213:addChannel = 1 " + bott.str() + " 91 5 -5");
    pythia.readString("4900213:addChannel = 1 "  + bott.str() + " 91 5 -5");
    pythia.readString("4900113:addChannel = 1 "  + bott.str() + " 91 5 -5");
    pythia.readString("-4900213:addChannel = 1 " + inv2.str() + " 91 53 -53");
    pythia.readString("4900213:addChannel = 1 "  + inv2.str() + " 91 53 -53");
    pythia.readString("4900113:addChannel = 1 "  + inv2.str() + " 91 53 -53");
  }

  // Suppress all PYTHIA output
  pythia.readString("Print:quiet = on");
  pythia.readString("Init:showChangedSettings = off");
  pythia.readString("Init:showChangedParticleData = off");
  pythia.readString("Next:numberCount = 0");
  pythia.readString("Stat:showProcessLevel = off");
  pythia.readString("Stat:showErrors = off");

  pythia.readString("Random:setSeed = on");
  {
    std::ostringstream oss;
    oss << "Random:seed = " << seed;
    pythia.readString(oss.str());
  }

  return pythia.init();
}

// ── Observable names (Stage-0 paired subset -- see file header) ─────────
// eventIndex (the iEvent this row came from -- re-running with a fixed seed
// regenerates the same event at the same index, which is what makes
// `dump_events` work) plus _truth / _delphes suffixed pairs, so one TSV row
// holds both computations for the same event. Same "# name\tname\t..."
// header convention svj_regression.cc uses, so src/observables.py::load_tsv()
// reads this file completely unmodified (it is column-name-agnostic).
// leadJetPt_truthfull: leading jet pT re-clustered from ALL final-state
// truth particles (visible + invisible + dark), see buildFullTruthParticles()
// -- a third, independent reference channel (not paired with a Delphes
// counterpart the way the _truth/_delphes columns are), so it only carries
// the single pT value, not the full leadJetMass/nConst/nJets tuple.
static const std::vector<std::string> OBS_NAMES = {
    "eventIndex",
    "leadVisPt_truth", "MET_truth", "leadJetMass_truth", "nConst_truth", "nJets_truth",
    "leadVisPt_delphes", "MET_delphes", "leadJetMass_delphes", "nConst_delphes", "nJets_delphes",
    "leadJetPt_truthfull",
};

// ── Shared jet-kinematics extraction ──────────────────────────────────────
// Clusters a list of 4-vectors and extracts the small Stage-0 observable
// subset, plus the leading jet's own constituent list (eta, phi, pt -- used
// only when dumping a specific event via dump_events; see file header).
// Used IDENTICALLY for both truth-particle input and Delphes EFlow-candidate
// input -- the two sides are computed by the literal same code, so any
// difference between truth and Delphes results comes only from the
// difference in the input particle list, never from the two sides happening
// to implement slightly different jet-kinematics math.
//
// When no jet passes the |eta|/pT acceptance (nJets == 0), leadPt/met/
// leadMass/nConst are set to NaN rather than 0 -- "no accepted jet" is not
// the same physical statement as "an accepted jet with these exact values",
// and leaving them undefined lets compare_delphes_truth.py mask them out of
// the per-event difference rather than silently contaminating it with
// spurious zero-vs-zero agreement.
//
// Constituent (eta, phi, pt) triples are extracted here, inside this
// function, rather than returning the leading fastjet::PseudoJet itself:
// PseudoJet::constituents() needs its originating ClusterSequence (the local
// `cs` below) to still be alive, which it no longer would be once this
// function returns and `cs` goes out of scope -- pulling the plain values
// out now avoids a dangling-ClusterSequence bug entirely.
struct JetObs { double leadPt, met, leadMass, nConst; int nJets; };
struct ClusterResult {
  JetObs obs;
  std::vector<std::array<double,3>> leadConstituents;  // {eta, phi, pt}; empty if obs.nJets==0
};

static ClusterResult clusterAndExtract(const std::vector<fastjet::PseudoJet>& particles,
                                       const fastjet::JetDefinition& jet_def,
                                       double eta_max, double vis_pt_min) {
  fastjet::ClusterSequence cs(particles, jet_def);
  std::vector<fastjet::PseudoJet> jets =
      fastjet::sorted_by_pt(cs.inclusive_jets(1.0));

  double evt_px = 0.0, evt_py = 0.0;
  double lead_pt = -1.0, lead_mass = 0.0, lead_nconst = 0.0;
  int n_jets = 0;
  int lead_idx = -1;

  for (size_t k = 0; k < jets.size(); ++k) {
    const auto& jet = jets[k];
    if (std::fabs(jet.eta()) >= eta_max) continue;
    if (jet.pt() < vis_pt_min) continue;

    ++n_jets;
    evt_px += jet.px();
    evt_py += jet.py();

    if (jet.pt() > lead_pt) {
      lead_pt     = jet.pt();
      lead_mass   = jet.m() > 0 ? jet.m() : 0.0;
      lead_nconst = (double)jet.constituents().size();
      lead_idx    = (int)k;
    }
  }

  std::vector<std::array<double,3>> leadConstituents;
  if (lead_idx >= 0) {
    for (const auto& c : jets[lead_idx].constituents())
      leadConstituents.push_back({c.eta(), c.phi(), c.pt()});
  }

  if (n_jets < 1) {
    double nan = std::numeric_limits<double>::quiet_NaN();
    return {{nan, nan, nan, nan, 0}, {}};
  }

  // Same "negative vector sum of accepted-jet visible pT" MET proxy
  // svj_regression.cc uses, for apples-to-apples comparability with truth.
  double met = std::sqrt(evt_px*evt_px + evt_py*evt_py);
  return {{lead_pt, met, lead_mass, lead_nconst, n_jets}, leadConstituents};
}

// ── Truth-side particle selection ────────────────────────────────────────
// Same TAG_VIS/TAG_MUON selection svj_regression.cc uses for jet clustering
// (visible final-state particles + muons; neutrinos [pid 12/14/16] and
// dark-sector pions/rhos [pid 51/53] excluded), inlined without the full
// tag-enum machinery since this driver only needs jet kinematics, not
// per-constituent substructure.
static std::vector<fastjet::PseudoJet> buildTruthParticles(Pythia8::Event& event) {
  std::vector<fastjet::PseudoJet> particles;
  particles.reserve(event.size());

  for (int i = 0; i < event.size(); ++i) {
    const Particle& p = event[i];
    if (!p.isFinal()) continue;

    int aid = p.idAbs();
    int id  = p.id();
    bool isDark = (id == 51 || id == -51 || id == 53 || id == -53);
    bool isInv  = (aid == 12 || aid == 14 || aid == 16);
    if (isDark || isInv) continue;   // excluded from visible clustering

    particles.emplace_back(p.px(), p.py(), p.pz(), p.e());
  }
  return particles;
}

// ── Full-truth particle selection (visible + invisible + dark) ──────────
// Same final-state loop as buildTruthParticles(), but WITHOUT the
// isDark/isInv exclusion -- every final-state particle (neutrinos and
// dark-sector pions/rhos included) is clustered. Re-clustering from scratch
// on this unfiltered list (rather than geometrically adding invisible
// momentum onto the already-identified visible jet) means the leading jet
// found here can, in principle, have a different axis/constituents than
// leadVisPt_truth's leading jet, since invisible momentum can shift which
// jet ends up "leading" -- this answers "what is the true total leading
// jet pT if you could see everything," not "how much invisible momentum
// sits inside the same visible jet."
static std::vector<fastjet::PseudoJet> buildFullTruthParticles(Pythia8::Event& event) {
  std::vector<fastjet::PseudoJet> particles;
  particles.reserve(event.size());

  for (int i = 0; i < event.size(); ++i) {
    const Particle& p = event[i];
    if (!p.isFinal()) continue;
    particles.emplace_back(p.px(), p.py(), p.pz(), p.e());
  }
  return particles;
}

// ── Pythia8 event -> Delphes candidate conversion ────────────────────────
// Ported from this project's installed delphes3.5.1/readers/DelphesPythia8.cpp
// (its free function ConvertInput()), trimmed to what Stage 0 needs:
//   - Only populates stableParticleOutputArray. ConvertInput() also fills
//     allParticleOutputArray (every particle, unconditionally) and
//     partonOutputArray (non-final light quarks/gluon/tau) -- neither is read
//     by any module in svj_delphes_particles.tcl's ExecutionPath (its only
//     entry point, ParticlePropagator, reads Delphes/stableParticles alone),
//     so they are omitted here rather than exported and left unused.
//   - Mother/daughter links (M1/M2/D1/D2) are still copied even though no
//     chained module currently uses them, to stay a faithful subset of the
//     verified working conversion rather than a reinterpretation of it.
//
// IMPORTANT, CORRECTED CLAIM: this function used to rely on
// `TDatabasePDG::Instance()->GetParticle(id)` returning null for the
// Hidden-Valley dark-sector bookkeeping codes (pid 51/53) to silently drop
// them. That assumption was checked directly against the installed ROOT
// (`root -b -q -e 'TDatabasePDG::Instance()->GetParticle(51)...'`) and is
// WRONG: ROOT's default particle database has an unrelated "technicolor
// pion" placeholder (pi_tech0 / pi'_tech0) sitting on the same numeric
// codes 51/53 (a generic/reserved BSM PDG slot, reused by an unrelated
// model), so GetParticle(51)/GetParticle(53) return a valid, non-null,
// charge-0 TParticlePDG -- they were never actually being filtered out
// here. They ended up having zero effect on Delphes' output anyway, but via
// an unrelated, fragile coincidence further downstream (charge=0 means they
// never become tracked candidates, and since 51/53 aren't listed in
// svj_delphes_particles.tcl's EnergyFraction table they fall back to its
// default fraction of 0.0, which SimpleCalorimeter::Process() uses to skip
// them before any calorimeter tower is even created) -- not because of
// anything guaranteed by this function. Excluding them explicitly here,
// the same way buildTruthParticles() already does, makes the exclusion
// robust by construction instead of by accident.
static const int kDarkPionRho[] = {51, -51, 53, -53};

static void fillDelphesInput(Pythia8::Event& event, DelphesFactory* factory,
                             TObjArray* stableParticleOutputArray) {
  TDatabasePDG* pdg = TDatabasePDG::Instance();

  for (int i = 1; i < event.size(); ++i) {
    Pythia8::Particle& p = event[i];

    // HepMC-convention status: 1 == final-state stable particle (the same
    // population Pythia8::Particle::isFinal() selects elsewhere in this
    // project; confirmed against the verified ConvertInput() source).
    if (p.statusHepMC() != 1) continue;

    // Explicit dark-sector exclusion (see corrected note above) -- mirrors
    // buildTruthParticles()'s isDark check exactly, rather than depending on
    // TDatabasePDG happening to not recognise these codes.
    bool isDark = false;
    for (int code : kDarkPionRho) if (p.id() == code) { isDark = true; break; }
    if (isDark) continue;

    TParticlePDG* pdgParticle = pdg->GetParticle(p.id());
    if (!pdgParticle) continue;   // genuinely unrecognised codes, if any

    Candidate* cand = factory->NewCandidate();
    cand->PID    = p.id();
    cand->Status = 1;
    cand->M1 = p.mother1()   - 1;  cand->M2 = p.mother2()   - 1;
    cand->D1 = p.daughter1() - 1;  cand->D2 = p.daughter2() - 1;
    cand->Charge = Int_t(pdgParticle->Charge() / 3.0);
    cand->Mass   = p.m();
    cand->Momentum.SetPxPyPzE(p.px(), p.py(), p.pz(), p.e());
    cand->Position.SetXYZT(p.xProd(), p.yProd(), p.zProd(), p.tProd());

    stableParticleOutputArray->Add(cand);
  }
}

int main(int argc, char* argv[]) {
  auto cfgPath = (std::filesystem::path(argv[0]).parent_path() / "svj_delphes_test.cfg").string();
  if (argc > 1) cfgPath = argv[1];

  auto cfg = readConfig(cfgPath);
  if (cfg.empty() && argc > 1)
    std::cerr << "Warning: could not read config file: " << cfgPath << "\n";

  mZ          = cfgDouble(cfg, "mZ",             mZ);
  mq          = cfgDouble(cfg, "mq",             mq);
  mPi         = cfgDouble(cfg, "mPi",            mPi);
  mRho        = cfgDouble(cfg, "mRho",           mRho);
  rinv_pion   = cfgDouble(cfg, "rinv_pion",      rinv_pion);
  rinv_rho    = cfgDouble(cfg, "rinv_rho",       rinv_rho);
  Brmu        = cfgDouble(cfg, "Brmu",           Brmu);
  alphaD      = cfgDouble(cfg, "alphaD",         alphaD);
  nEvent      = cfgInt   (cfg, "nEvent",         nEvent);
  jetR        = cfgDouble(cfg, "jetR",           jetR);
  LambdaDQCD  = cfgDouble(cfg, "LambdaDQCD",     LambdaDQCD);
  visJetPtMin = cfgDouble(cfg, "vis_jet_pt_min", visJetPtMin);
  tsvFile     = cfgStr   (cfg, "tsv_file",       tsvFile);
  delphesCard = cfgStr   (cfg, "delphes_card",   delphesCard);
  dumpEventsStr      = cfgStr(cfg, "dump_events",             dumpEventsStr);
  constitTruthFile   = cfgStr(cfg, "constituents_truth_file",   constitTruthFile);
  constitDelphesFile = cfgStr(cfg, "constituents_delphes_file", constitDelphesFile);
  std::set<int> dumpEvents = parseEventIndexList(dumpEventsStr);

  if (rinv_rho < 0.0 || rinv_rho >= 1.0 || Brmu < 0.0 || Brmu > 1.0) {
    std::cerr << "Error: rinv_rho=" << rinv_rho << " Brmu=" << Brmu
              << " -- require 0 <= rinv_rho < 1 and 0 <= Brmu <= 1\n";
    return 1;
  }

  // ── Pythia8 setup (single worker, fixed seed -- Stage 0 is a one-shot,
  //    reproducible smoke test, not a scan worker) ─────────────────────────
  Pythia pythia;
  if (!setupPythia(pythia, /*seed=*/1)) {
    std::cerr << "Error: Pythia8 initialization failed.\n";
    return 1;
  }

  // ── Delphes setup (once, before the event loop) ──────────────────────────
  // Headless: no ROOT output file is ever created. Verified safe against the
  // installed delphes3.5.1/modules/Delphes.cc (Init() only requires a
  // non-null ExRootTreeWriter*; none of the five chained modules touch its
  // file) and delphes3.5.1/external/ExRootAnalysis/ExRootTreeWriter.cc
  // (Clear() only iterates its branch set, which stays empty since NewBranch
  // is never called here).
  //
  // Heap-allocated (not stack) to mirror the verified-working reference
  // pattern in delphes3.5.1/readers/DelphesPythia8.cpp exactly -- these are
  // ROOT TNamed/TTask-derived objects with their own global bookkeeping
  // (e.g. TTask's task-tree registration), so there is no reason to risk a
  // stack-lifetime subtlety the reference implementation doesn't test either.
  ExRootConfReader* confReader = new ExRootConfReader;
  confReader->ReadFile(delphesCard.c_str());

  Delphes* modularDelphes = new Delphes("Delphes");
  modularDelphes->SetConfReader(confReader);

  ExRootTreeWriter* treeWriter = new ExRootTreeWriter(nullptr, "Delphes");
  modularDelphes->SetTreeWriter(treeWriter);

  DelphesFactory* factory = modularDelphes->GetFactory();
  TObjArray* stableParticleOutputArray = modularDelphes->ExportArray("stableParticles");

  // IMPORTANT: InitTask(), not Init() -- see file header. Init() alone would
  // build the module list but never call any individual module's own Init().
  modularDelphes->InitTask();

  const double ETA_MAX = 2.5;
  const fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, jetR);

  std::vector<std::vector<double>> data;   // one row per event, columns = OBS_NAMES
  int nProcessed = 0;

  // Constituent-dump output (only opened/written if dump_events is non-empty
  // -- a normal run touches neither of these files). "# eventIndex\teta\tphi\tpt",
  // one row per leading-jet constituent of a requested event.
  std::unique_ptr<std::ofstream> fConstitTruth, fConstitDelphes;
  if (!dumpEvents.empty()) {
    auto openDump = [](const std::string& path) {
      auto dir = std::filesystem::path(path).parent_path();
      if (!dir.empty()) std::filesystem::create_directories(dir);
      auto f = std::make_unique<std::ofstream>(path);
      (*f) << "#\teventIndex\teta\tphi\tpt\n" << std::scientific << std::setprecision(6);
      return f;
    };
    fConstitTruth   = openDump(constitTruthFile);
    fConstitDelphes = openDump(constitDelphesFile);
  }

  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) continue;
    ++nProcessed;
    bool dumpThisEvent = dumpEvents.count(iEvent) > 0;

    // ── Truth side: same particle selection + clustering svj_regression.cc
    //    uses, on the raw Pythia event ───────────────────────────────────
    std::vector<fastjet::PseudoJet> truthParticles = buildTruthParticles(pythia.event);
    ClusterResult truthResult = clusterAndExtract(truthParticles, jet_def, ETA_MAX, visJetPtMin);
    const JetObs& truthObs = truthResult.obs;

    // ── Full-truth side (visible + invisible + dark, re-clustered from
    //    scratch -- see buildFullTruthParticles()) ───────────────────────
    std::vector<fastjet::PseudoJet> fullTruthParticles = buildFullTruthParticles(pythia.event);
    ClusterResult fullTruthResult = clusterAndExtract(fullTruthParticles, jet_def, ETA_MAX, visJetPtMin);
    double leadJetPtTruthFull = fullTruthResult.obs.leadPt;

    // ── Delphes side: same event, reconstructed ─────────────────────────
    treeWriter->Clear();
    modularDelphes->Clear();

    fillDelphesInput(pythia.event, factory, stableParticleOutputArray);

    // IMPORTANT: ProcessTask(), not Process() -- Delphes::Process() is a
    // verified no-op; ProcessTask() is what actually recurses into every
    // configured module (ParticlePropagator, tracking efficiency/smearing,
    // ECal/HCal, EFlowMerger, ...) and runs their real per-event logic.
    modularDelphes->ProcessTask();

    TObjArray* eflow = modularDelphes->ImportArray("EFlowMerger/eflow");

    std::vector<fastjet::PseudoJet> delphesParticles;
    delphesParticles.reserve(eflow->GetEntriesFast());
    for (int i = 0; i < eflow->GetEntriesFast(); ++i) {
      Candidate* cand = static_cast<Candidate*>(eflow->At(i));
      const TLorentzVector& mom = cand->Momentum;
      delphesParticles.emplace_back(mom.Px(), mom.Py(), mom.Pz(), mom.E());
    }
    ClusterResult delphesResult = clusterAndExtract(delphesParticles, jet_def, ETA_MAX, visJetPtMin);
    const JetObs& delphesObs = delphesResult.obs;

    // Every processed event gets a row -- no jet-count-based discard here.
    // NaN sentinels (from clusterAndExtract, per side) mark "this side had
    // no accepted jet" so compare_delphes_truth.py can mask correctly rather
    // than the two sides' discard patterns silently desynchronising the
    // event correspondence this whole driver exists to guarantee.
    data.push_back({(double)iEvent,
                    truthObs.leadPt, truthObs.met, truthObs.leadMass,
                    truthObs.nConst, (double)truthObs.nJets,
                    delphesObs.leadPt, delphesObs.met, delphesObs.leadMass,
                    delphesObs.nConst, (double)delphesObs.nJets,
                    leadJetPtTruthFull});

    // ── Constituent dump, only for explicitly requested event indices ────
    if (dumpThisEvent) {
      auto writeConstits = [&](std::ofstream& f,
                               const std::vector<std::array<double,3>>& constits) {
        for (const auto& c : constits)
          f << iEvent << "\t" << c[0] << "\t" << c[1] << "\t" << c[2] << "\n";
      };
      if (fConstitTruth)   writeConstits(*fConstitTruth,   truthResult.leadConstituents);
      if (fConstitDelphes) writeConstits(*fConstitDelphes, delphesResult.leadConstituents);
    }
  }

  modularDelphes->FinishTask();

  // ── Write TSV (same "# name\tname\t..." header convention as
  //    svj_regression.cc, so src/observables.py::load_tsv() reads it
  //    completely unmodified) ────────────────────────────────────────────
  auto tsvDir = std::filesystem::path(tsvFile).parent_path();
  if (!tsvDir.empty()) std::filesystem::create_directories(tsvDir);

  std::ofstream fOut(tsvFile);
  fOut << "#";
  for (size_t i = 0; i < OBS_NAMES.size(); ++i) fOut << "\t" << OBS_NAMES[i];
  fOut << "\n";
  fOut << std::scientific << std::setprecision(6);
  for (const auto& row : data) {
    for (size_t i = 0; i < row.size(); ++i) {
      if (i > 0) fOut << "\t";
      fOut << row[i];
    }
    fOut << "\n";
  }

  std::cerr << "Wrote " << nProcessed << " / " << nEvent
            << " events -> " << tsvFile << "\n";
  if (!dumpEvents.empty()) {
    std::cerr << "Dumped constituents for " << dumpEvents.size() << " requested event(s) -> "
              << constitTruthFile << ", " << constitDelphesFile << "\n";
  }

  delete modularDelphes;
  delete confReader;
  delete treeWriter;

  return 0;
}
