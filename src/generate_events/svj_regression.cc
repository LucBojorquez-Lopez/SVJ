#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"
#include "svj_observables_common.h"
#include <sstream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <filesystem>
#include <thread>
#include <map>
#include <iostream>
#include <iomanip>

using namespace Pythia8;

// Read "key = value" pairs from a config file; lines starting with # are comments.
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

// Physics parameters
static double mZ        = 2000.0;
static double mq        =    4.0;
static double mPi       =    8.0;
static double mRho      =   15.5;
static double rinv_pion  =    0.3;   // dark pion invisible BR
static double rinv_rho   =    0.3;   // dark rho invisible BR
static double Brmu      =    0.3;   // dark rho → mu+mu- BR fraction (of visible decays)
static double alphaD    =    0.4;
static int    nEvent    = 100000;
static double jetR      =    1.0;
static double LambdaDQCD =   5.0;
static int    nWorkers  =   10;
static int    seedOffset =   0;   // added to worker seeds: seed = workerID+1 + seedOffset*nWorkers
                                  // set per-job by run_svj_tsv.sh for batch TSV generation
static int    saveTSV   =    1;   // write raw TSV; set to 0 during scan
static int    jetsVisOnly = 1;    // 1 = store visible jet 4-momenta in jets_kinematics.tsv; 0 = full jet (incl. invisible)
static int    dijetOnly  =  0;   // 1 = only keep events with >= 2 jets (dijet topology)
static double visJetPtMin = 20.0; // min visible pT (GeV) for a jet to be kept
static std::string tsvFile    = "simulated/tsv/jets_default.tsv";
static std::string tsvKinFile = "simulated/tsv/jets_kinematics.tsv";

static void rs(Pythia& p, const std::string& key, double val) {
  std::ostringstream oss;
  oss << key << " = " << val;
  p.readString(oss.str());
}

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
    // rho_mu_br   = Brmu * (1 - rinv_rho)  → mu+mu-
    // rho_bott_br = (1 - Brmu) * (1 - rinv_rho)  → bb-bar
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

// ── Observable names ─────────────────────────────────────────────────────────
// ADD NEW OBSERVABLES HERE (step 1 of 3 -- observable computation itself lives
// in the SHARED svj_observables_common.h, used identically by
// svj_regression_delphes.cc, so a new observable is computed once, not twice):
//   1. Add the field to SvjObservables and compute it in computeSvjObservables()
//      (svj_observables_common.h).
//   2. Add the name string to OBS_NAMES below (append at end to preserve order).
//   3. Append the new SvjObservables field to the data.push_back({...}) in
//      runWorker() below, in the same position as its OBS_NAMES entry.
// The Python side (src/observables.py) reads column names from the TSV header at
// runtime — no integer indices to synchronise.
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

// ── Worker: collect per-event observables ─────────────────────────────────────
static void runWorker(int workerID, int nEvtWorker,
                      std::vector<std::vector<double>>& data,
                      std::vector<JetKin>& jetData) {
  Pythia pythia;
  if (!setupPythia(pythia, workerID + 1 + seedOffset * nWorkers)) return;

  Event& event = pythia.event;

  for (int iEvent = 0; iEvent < nEvtWorker; ++iEvent) {
    if (!pythia.next()) continue;

    // ── Classify final-state particles into the shared SvjJetInputs form
    //    (TAG_VIS/TAG_MUON particles for clustering; neutrinos/dark particles
    //    kept separately for the geometric nInvClose/fInv matching inside
    //    computeSvjObservables()) -- this classification step is inherently
    //    truth-specific (raw Pythia8::Particle -> tag), so it stays here
    //    rather than in the shared header; svj_regression_delphes.cc has its
    //    own analogous classification from Delphes candidates instead. ─────
    SvjJetInputs inputs;
    inputs.particles.reserve(event.size());

    for (int i = 0; i < event.size(); ++i) {
      const Particle& p = event[i];
      if (!p.isFinal()) continue;

      int aid = p.idAbs();
      int id  = p.id();

      int ptag;
      if (id == 51 || id == -51 || id == 53 || id == -53)
        ptag = TAG_DARK;
      else if (aid == 12 || aid == 14 || aid == 16)
        ptag = TAG_INV;
      else if (aid == 13)
        ptag = TAG_MUON;
      else
        ptag = TAG_VIS;

      // Max electron and muon pT (from final-state particles)
      if (aid == 11) {
        double ept = std::sqrt(p.px()*p.px() + p.py()*p.py());
        if (ept > inputs.max_ele_pt) inputs.max_ele_pt = ept;
      }
      if (aid == 13) {
        double mpt = std::sqrt(p.px()*p.px() + p.py()*p.py());
        if (mpt > inputs.max_mu_pt) inputs.max_mu_pt = mpt;
      }
      // Event-level transverse sphericity tensor: all visible+muon particles
      if (ptag < TAG_INV) {
        double px = p.px(), py = p.py();
        inputs.s_xx += px*px; inputs.s_xy += px*py; inputs.s_yy += py*py;
        inputs.s_pt2 += px*px + py*py;
      }

      if (ptag < TAG_INV) {
        // Visible + muon only: feed into jet clustering so jet axes are
        // determined purely by detectable particles.
        fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
        pj.set_user_index(ptag);
        inputs.particles.push_back(pj);
      } else {
        // Neutrinos (TAG_INV) + dark pions/rhos (TAG_DARK): store kinematics
        // for geometric matching into the visible-only jet cones below.
        double px = p.px(), py = p.py(), pz = p.pz();
        double pt = std::sqrt(px*px + py*py);
        double pm = std::sqrt(px*px + py*py + pz*pz);
        double eta = (pm > std::fabs(pz)) ? 0.5*std::log((pm+pz)/(pm-pz))
                                          : (pz > 0 ? 1e9 : -1e9);
        inputs.invis_ptcls.push_back({eta, std::atan2(py, px), pt, ptag});
      }
    }

    SvjObservables obs;
    JetKin jk;
    bool ok = computeSvjObservables(inputs, jetR, visJetPtMin, /*etaMax=*/2.5,
                                    jetsVisOnly != 0, dijetOnly != 0, obs, &jk);
    if (!ok) continue;

    // ADD NEW OBSERVABLE COMPUTATIONS in svj_observables_common.h, then
    // append to push_back below in the same position as OBS_NAMES.
    data.push_back({obs.leadVisPt, obs.leadWidth, obs.MET,
                    obs.maxElePt, obs.maxMuPt, obs.jetThrust,
                    obs.transSphericity, obs.hemiMass1, obs.hemiMass2,
                    obs.ptBal, obs.dPhiMETdijet,
                    obs.e2c, obs.e3c, obs.tau1, obs.tau2, obs.tau3,
                    obs.dPhiMETclose, obs.dPhiMETfar, obs.nJets,
                    obs.closeJetIsLead, obs.nInvClose, obs.metPhi,
                    obs.HT, obs.RT, obs.Meff,
                    obs.leadJetMass, obs.nConst, obs.fInv});
    jetData.push_back(jk);
  }
}

int main(int argc, char* argv[]) {

  auto cfgPath = (std::filesystem::path(argv[0]).parent_path() / "svj_regression.cfg").string();
  if (argc > 1) cfgPath = argv[1];

  auto cfg = readConfig(cfgPath);
  if (cfg.empty() && argc > 1)
    std::cerr << "Warning: could not read config file: " << cfgPath << "\n";

  mZ         = cfgDouble(cfg, "mZ",         mZ);
  mq         = cfgDouble(cfg, "mq",         mq);
  mPi        = cfgDouble(cfg, "mPi",        mPi);
  mRho       = cfgDouble(cfg, "mRho",       mRho);
  rinv_pion  = cfgDouble(cfg, "rinv_pion",  rinv_pion);
  rinv_rho   = cfgDouble(cfg, "rinv_rho",   rinv_rho);
  Brmu       = cfgDouble(cfg, "Brmu",       Brmu);
  alphaD     = cfgDouble(cfg, "alphaD",     alphaD);
  nEvent     = cfgInt   (cfg, "nEvent",     nEvent);
  jetR       = cfgDouble(cfg, "jetR",       jetR);
  LambdaDQCD = cfgDouble(cfg, "LambdaDQCD", LambdaDQCD);
  nWorkers   = cfgInt   (cfg, "nWorkers",    nWorkers);
  seedOffset = cfgInt   (cfg, "seed_offset", seedOffset);
  saveTSV    = cfgInt   (cfg, "save_tsv",    saveTSV);
  jetsVisOnly = cfgInt  (cfg, "jets_vis_only", jetsVisOnly);
  dijetOnly   = cfgInt  (cfg, "dijet_only",    dijetOnly);
  visJetPtMin = cfgDouble(cfg, "vis_jet_pt_min", visJetPtMin);
  tsvFile     = cfgStr  (cfg, "tsv_file",      tsvFile);
  tsvKinFile  = cfgStr  (cfg, "tsv_kin_file",  tsvKinFile);

  if (rinv_rho < 0.0 || rinv_rho >= 1.0 || Brmu < 0.0 || Brmu > 1.0) {
    std::cerr << "Error: rinv_rho=" << rinv_rho << " Brmu=" << Brmu
              << " — require 0 <= rinv_rho < 1 and 0 <= Brmu <= 1\n";
    return 1;
  }

  int base      = nEvent / nWorkers;
  int remainder = nEvent % nWorkers;

  std::vector<std::vector<std::vector<double>>> results(nWorkers);
  std::vector<std::vector<JetKin>> jetResults(nWorkers);
  std::vector<std::thread> threads;
  threads.reserve(nWorkers);

  for (int w = 0; w < nWorkers; ++w) {
    int nEvtWorker = base + (w < remainder ? 1 : 0);
    threads.emplace_back(runWorker, w, nEvtWorker,
                         std::ref(results[w]), std::ref(jetResults[w]));
  }
  for (auto& t : threads) t.join();

  // Optionally write raw TSVs (skipped during scan via save_tsv = 0)
  if (saveTSV) {
    auto tsvDir = std::filesystem::path(tsvFile).parent_path();
    if (!tsvDir.empty()) std::filesystem::create_directories(tsvDir);
    auto tsvKinDir = std::filesystem::path(tsvKinFile).parent_path();
    if (!tsvKinDir.empty()) std::filesystem::create_directories(tsvKinDir);

    std::ofstream fOut(tsvFile);
    fOut << "#";
    for (size_t i = 0; i < OBS_NAMES.size(); ++i)
      fOut << "\t" << OBS_NAMES[i];
    fOut << "\n";
    fOut << std::scientific << std::setprecision(6);
    for (const auto& rows : results)
      for (const auto& r : rows) {
        for (size_t i = 0; i < r.size(); ++i) {
          if (i > 0) fOut << "\t";
          fOut << r[i];
        }
        fOut << "\n";
      }

    std::ofstream fJets(tsvKinFile);
    fJets << std::scientific << std::setprecision(6);
    fJets << "# n_jets\tj1_px\tj1_py\tj1_pz\tj1_E\tj2_px\tj2_py\tj2_pz\tj2_E\n";
    for (const auto& rows : jetResults)
      for (const auto& jk : rows)
        fJets << jk.n_jets  << "\t"
              << jk.j1_px   << "\t" << jk.j1_py << "\t"
              << jk.j1_pz   << "\t" << jk.j1_E  << "\t"
              << jk.j2_px   << "\t" << jk.j2_py << "\t"
              << jk.j2_pz   << "\t" << jk.j2_E  << "\n";
  }

  return 0;
}
