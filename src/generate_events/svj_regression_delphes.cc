// svj_regression_delphes.cc
// ============================================================================
// DELPHES production binary: the Delphes-level counterpart to svj_regression.cc.
// Writes a plain, single-stream TSV (same OBS_NAMES-header convention as
// svj_regression.cc -- NOT the paired _truth/_delphes format svj_delphes_test.cc
// uses for its own, different diagnostic purpose) so it works unmodified with
// observables.py / diagnostics.py / scan_svj.py / validate_fit.py.
//
// Computes the FULL ~27-observable tuple (same OBS_NAMES/order as
// svj_regression.cc) via the SAME shared computeSvjObservables()
// (svj_observables_common.h) svj_regression.cc uses -- writing every
// observable, not a hand-picked subset, means DEFAULT_SCAN (or any --obs
// selection) can grow in src/observables.py without this binary silently
// falling out of sync and dropping columns scan_svj.py expects. nInvClose,
// fInv and closeJetIsLead are truth-only concepts (see
// svj_observables_common.h's header comment) and always read 0 here, since
// invis_ptcls is always empty for Delphes-eflow input -- everything else
// (including dPhiMETclose/dPhiMETfar, which depend only on jets + MET, not
// invisibles) is a genuine Delphes-level value.
// Adding a new observable means editing svj_observables_common.h, then
// adding its name + a field to this binary's OBS_NAMES/push_back (exactly the
// same two-step pattern already documented for svj_regression.cc in
// docs/extending-observables.md).
//
// Architecture mirrors svj_delphes_test.cc exactly (same verified Delphes API
// usage, same particle-level-only module chain, same dark-sector exclusion
// fix) -- see that file's header for the InitTask/ProcessTask/FinishTask
// correctness note, which applies identically here. Physics setup
// (readConfig/cfgDouble/.../setupPythia) is again an intentional copy, not a
// #include, for the same reason documented there.
//
// Single-threaded, single physics point, like svj_delphes_test.cc -- the
// `gRandom` data race in Delphes' MomentumSmearing/SimpleCalorimeter modules
// (called directly, not through any thread-safety wrapper) rules out
// in-process multithreading. All scan-level parallelism comes from
// scan_svj.py's outer ProcessPoolExecutor, exactly as already documented for
// the truth binary (nWorkers=1, "outer loop handles parallelism").
// ============================================================================

#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"
#include "svj_observables_common.h"

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
#include <string>
#include <cmath>
#include <filesystem>
#include <map>
#include <iostream>
#include <iomanip>

using namespace Pythia8;

// ── Config-file reader (copied from svj_regression.cc; see file header) ──
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

// ── Physics parameters (same defaults as svj_regression.cfg) ────────────
static double mZ         = 2000.0;
static double mq         =    4.0;
static double mPi        =    8.0;
static double mRho       =   15.5;
static double rinv_pion  =    0.3;
static double rinv_rho   =    0.3;
static double Brmu       =    0.3;
static double alphaD     =    0.4;
static int    nEvent     = 50000;
static double jetR       =    1.0;
static double LambdaDQCD =    5.0;
static int    seedOffset =    0;    // seed = 1 + seedOffset (single-threaded -> one seed)
static int    jetsVisOnly = 1;
static int    dijetOnly   = 0;
static double visJetPtMin = 20.0;
static std::string tsvFile     = "simulated/tsv/jets_delphes.tsv";
static std::string delphesCard = "src/generate_events/svj_delphes_particles.tcl";

static void rs(Pythia& p, const std::string& key, double val) {
  std::ostringstream oss;
  oss << key << " = " << val;
  p.readString(oss.str());
}

// Copied verbatim from svj_regression.cc's setupPythia() -- same HiddenValley
// SVJ physics model. See svj_regression.cc for the physics commentary.
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

// ── Observable names (full set, same order as svj_regression.cc) ────────
// ADD NEW OBSERVABLES HERE (two steps -- the shared computation itself lives
// in svj_observables_common.h, used identically by svj_regression.cc):
//   1. Add the field to SvjObservables and compute it in computeSvjObservables()
//      (svj_observables_common.h) -- skip if it's already computed there.
//   2. Add the name string to OBS_NAMES below and the matching field to the
//      data.push_back({...}) in main()'s event loop, in the same position.
static const std::vector<std::string> OBS_NAMES = {
    "leadVisPt", "leadWidth", "MET",
    "maxElePt",  "maxMuPt",
    "jetThrust", "transSphericity",
    "hemiMass1", "hemiMass2",
    "ptBal",     "dPhiMETdijet",
    "e2c",       "e3c",
    "tau1",      "tau2",   "tau3",
    "dPhiMETclose", "dPhiMETfar", "nJets",
    "closeJetIsLead", "nInvClose", "metPhi",
    "HT", "RT", "Meff",
    "leadJetMass", "nConst", "fInv",
};

// ── Pythia8 event -> Delphes candidate conversion ────────────────────────
// Identical to svj_delphes_test.cc's fillDelphesInput() (see that file's
// header for the full verification trail: ported from this project's
// installed delphes3.5.1/readers/DelphesPythia8.cpp, explicit dark-sector
// exclusion rather than relying on TDatabasePDG returning null for pid 51/53
// -- confirmed by direct query that it does NOT, see
// docs -- and the associated project memory).
static const int kDarkPionRho[] = {51, -51, 53, -53};

static void fillDelphesInput(Pythia8::Event& event, DelphesFactory* factory,
                             TObjArray* stableParticleOutputArray) {
  TDatabasePDG* pdg = TDatabasePDG::Instance();

  for (int i = 1; i < event.size(); ++i) {
    Pythia8::Particle& p = event[i];

    if (p.statusHepMC() != 1) continue;

    bool isDark = false;
    for (int code : kDarkPionRho) if (p.id() == code) { isDark = true; break; }
    if (isDark) continue;

    TParticlePDG* pdgParticle = pdg->GetParticle(p.id());
    if (!pdgParticle) continue;

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

// ── Delphes EFlow candidates -> shared SvjJetInputs form ─────────────────
// All eflow candidates are "visible" by construction (invisible truth
// particles never entered Delphes at all -- see fillDelphesInput above), so
// there is no TAG_INV/TAG_DARK population here; invis_ptcls stays empty,
// which naturally makes computeSvjObservables()'s nInvClose/fInv evaluate to
// their trivial (zero) values rather than needing a separate code path.
// Muons are still tagged TAG_MUON (vs TAG_VIS for everything else) for
// symmetry with the truth-side classification, even though no current
// observable in computeSvjObservables() actually distinguishes the two.
static SvjJetInputs buildDelphesInputs(TObjArray* eflow) {
  SvjJetInputs inputs;
  inputs.particles.reserve(eflow->GetEntriesFast());

  for (int i = 0; i < eflow->GetEntriesFast(); ++i) {
    Candidate* cand = static_cast<Candidate*>(eflow->At(i));
    const TLorentzVector& mom = cand->Momentum;
    int aid = std::abs(cand->PID);

    if (aid == 11) {
      double ept = std::sqrt(mom.Px()*mom.Px() + mom.Py()*mom.Py());
      if (ept > inputs.max_ele_pt) inputs.max_ele_pt = ept;
    }
    if (aid == 13) {
      double mpt = std::sqrt(mom.Px()*mom.Px() + mom.Py()*mom.Py());
      if (mpt > inputs.max_mu_pt) inputs.max_mu_pt = mpt;
    }

    double px = mom.Px(), py = mom.Py();
    inputs.s_xx += px*px; inputs.s_xy += px*py; inputs.s_yy += py*py;
    inputs.s_pt2 += px*px + py*py;

    int ptag = (aid == 13) ? TAG_MUON : TAG_VIS;
    fastjet::PseudoJet pj(mom.Px(), mom.Py(), mom.Pz(), mom.E());
    pj.set_user_index(ptag);
    inputs.particles.push_back(pj);
  }
  return inputs;
}

int main(int argc, char* argv[]) {
  auto cfgPath = (std::filesystem::path(argv[0]).parent_path() / "svj_regression_delphes.cfg").string();
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
  seedOffset  = cfgInt   (cfg, "seed_offset",    seedOffset);
  jetsVisOnly = cfgInt   (cfg, "jets_vis_only",  jetsVisOnly);
  dijetOnly   = cfgInt   (cfg, "dijet_only",     dijetOnly);
  visJetPtMin = cfgDouble(cfg, "vis_jet_pt_min", visJetPtMin);
  tsvFile     = cfgStr   (cfg, "tsv_file",       tsvFile);
  delphesCard = cfgStr   (cfg, "delphes_card",   delphesCard);

  if (rinv_rho < 0.0 || rinv_rho >= 1.0 || Brmu < 0.0 || Brmu > 1.0) {
    std::cerr << "Error: rinv_rho=" << rinv_rho << " Brmu=" << Brmu
              << " -- require 0 <= rinv_rho < 1 and 0 <= Brmu <= 1\n";
    return 1;
  }

  // Single-threaded, single seed -- see file header for why (gRandom race).
  Pythia pythia;
  if (!setupPythia(pythia, /*seed=*/1 + seedOffset)) {
    std::cerr << "Error: Pythia8 initialization failed.\n";
    return 1;
  }

  // ── Delphes setup (verified pattern, see svj_delphes_test.cc's header) ──
  ExRootConfReader* confReader = new ExRootConfReader;
  confReader->ReadFile(delphesCard.c_str());

  Delphes* modularDelphes = new Delphes("Delphes");
  modularDelphes->SetConfReader(confReader);

  ExRootTreeWriter* treeWriter = new ExRootTreeWriter(nullptr, "Delphes");
  modularDelphes->SetTreeWriter(treeWriter);

  DelphesFactory* factory = modularDelphes->GetFactory();
  TObjArray* stableParticleOutputArray = modularDelphes->ExportArray("stableParticles");

  // IMPORTANT: InitTask(), not Init() -- see svj_delphes_test.cc's header.
  modularDelphes->InitTask();

  std::vector<std::vector<double>> data;
  int nProcessed = 0, nKept = 0;

  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) continue;
    ++nProcessed;

    treeWriter->Clear();
    modularDelphes->Clear();

    fillDelphesInput(pythia.event, factory, stableParticleOutputArray);

    // IMPORTANT: ProcessTask(), not Process() -- see svj_delphes_test.cc's header.
    modularDelphes->ProcessTask();

    TObjArray* eflow = modularDelphes->ImportArray("EFlowMerger/eflow");
    SvjJetInputs inputs = buildDelphesInputs(eflow);

    SvjObservables obs;
    bool ok = computeSvjObservables(inputs, jetR, visJetPtMin, /*etaMax=*/2.5,
                                    jetsVisOnly != 0, dijetOnly != 0, obs, nullptr);
    if (!ok) continue;
    ++nKept;

    data.push_back({obs.leadVisPt, obs.leadWidth, obs.MET,
                    obs.maxElePt, obs.maxMuPt, obs.jetThrust,
                    obs.transSphericity, obs.hemiMass1, obs.hemiMass2,
                    obs.ptBal, obs.dPhiMETdijet,
                    obs.e2c, obs.e3c, obs.tau1, obs.tau2, obs.tau3,
                    obs.dPhiMETclose, obs.dPhiMETfar, obs.nJets,
                    obs.closeJetIsLead, obs.nInvClose, obs.metPhi,
                    obs.HT, obs.RT, obs.Meff,
                    obs.leadJetMass, obs.nConst, obs.fInv});
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

  std::cerr << "Wrote " << nKept << " / " << nProcessed
            << " kept events (of " << nEvent << " requested) -> " << tsvFile << "\n";

  delete modularDelphes;
  delete confReader;
  delete treeWriter;

  return 0;
}
