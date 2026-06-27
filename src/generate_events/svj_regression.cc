#include "Pythia8/Pythia.h"
#include "fastjet/ClusterSequence.hh"
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

// user_index tags
static const int TAG_VIS  = 0;
static const int TAG_MUON = 1;
static const int TAG_INV  = 2;  // neutrinos (pid 12/14/16)
static const int TAG_DARK = 3;  // dark pions/rhos (pid 51/53)

// Per-event jet kinematics (visible 4-momentum for leading and subleading jet)
struct JetKin {
  int    n_jets;
  double j1_px, j1_py, j1_pz, j1_E;
  double j2_px, j2_py, j2_pz, j2_E;
};

// Invisible final-state particle (neutrino or dark) stored for geometric matching
// into visible-only jets after clustering.
struct InvisPtcl { double eta, phi, pt; int tag; };

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
// ADD NEW OBSERVABLES HERE (step 1 of 2):
//   1. Add the name string to OBS_NAMES (append at end to preserve existing order).
//   2. Compute the observable in runWorker() and append it to data.push_back({...}).
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
  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, jetR);

  const double VIS_PT_MIN = visJetPtMin;
  const double ETA_MAX    =  2.5;
  const double PI         = std::acos(-1.0);

  for (int iEvent = 0; iEvent < nEvtWorker; ++iEvent) {
    if (!pythia.next()) continue;

    std::vector<fastjet::PseudoJet> particles;
    particles.reserve(event.size());
    // Invisible particles (neutrinos + dark) stored for geometric matching
    // into the visible-only jet cones after clustering.
    std::vector<InvisPtcl> invis_ptcls;

    // Tracked during particle loop
    double max_ele_pt = 0.0, max_mu_pt = 0.0;
    double s_xx = 0, s_xy = 0, s_yy = 0, s_pt2 = 0;  // sphericity tensor sums

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
        if (ept > max_ele_pt) max_ele_pt = ept;
      }
      if (aid == 13) {
        double mpt = std::sqrt(p.px()*p.px() + p.py()*p.py());
        if (mpt > max_mu_pt) max_mu_pt = mpt;
      }
      // Event-level transverse sphericity tensor: all visible+muon particles
      if (ptag < TAG_INV) {
        double px = p.px(), py = p.py();
        s_xx += px*px; s_xy += px*py; s_yy += py*py;
        s_pt2 += px*px + py*py;
      }

      if (ptag < TAG_INV) {
        // Visible + muon only: feed into jet clustering so jet axes are
        // determined purely by detectable particles.
        fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
        pj.set_user_index(ptag);
        particles.push_back(pj);
      } else {
        // Neutrinos (TAG_INV) + dark pions/rhos (TAG_DARK): store kinematics
        // for geometric matching into the visible-only jet cones below.
        double px = p.px(), py = p.py(), pz = p.pz();
        double pt = std::sqrt(px*px + py*py);
        double pm = std::sqrt(px*px + py*py + pz*pz);
        double eta = (pm > std::fabs(pz)) ? 0.5*std::log((pm+pz)/(pm-pz))
                                          : (pz > 0 ? 1e9 : -1e9);
        invis_ptcls.push_back({eta, std::atan2(py, px), pt, ptag});
      }
    }

    fastjet::ClusterSequence cs(particles, jet_def);
    std::vector<fastjet::PseudoJet> raw_jets =
      fastjet::sorted_by_pt(cs.inclusive_jets(1.0));

    double evt_vis_px = 0, evt_vis_py = 0;
    int    n_jets = 0;
    double lead_vis_pt = -1.0, sub_vis_pt = -1.0;  // visible-pT ordering for new observables
    double lead_cut_pt = -1.0, sub_cut_pt = -1.0;  // ordering pT for kinematics TSV
    double lead_width  =  0.0;
    double j1_vis_px = 0, j1_vis_py = 0;   // leading visible-pT jets (for dijet phi)
    double j2_vis_px = 0, j2_vis_py = 0;
    double j1_px=0, j1_py=0, j1_pz=0, j1_E=0;
    double j2_px=0, j2_py=0, j2_pz=0, j2_E=0;
    std::vector<std::array<double,4>> lead_constits;          // visible constituents of leading jet (substructure)
    std::vector<std::array<double,4>> all_vis_constits;       // visible constituents of ALL passing jets (hemispheres)
    std::vector<std::pair<double,double>> jet_vis_vecs;       // (vis_px, vis_py) per passing jet
    std::vector<std::pair<double,double>> jet_axes;           // (eta, phi) of visible jet axis per passing jet
    std::vector<int> jet_inv_counts;                          // dark particle count per passing jet (geometric)
    int lead_vis_idx = 0;

    for (const auto& jet : raw_jets) {
      if (std::fabs(jet.eta()) >= ETA_MAX) continue;

      double vis_px = 0, vis_py = 0, vis_pz = 0, vis_E = 0;
      double width_num = 0, width_den = 0;

      for (const auto& c : jet.constituents()) {
        if (c.user_index() >= TAG_INV) continue;   // muons (TAG_MUON) count as visible

        vis_px += c.px();
        vis_py += c.py();
        vis_pz += c.pz();
        vis_E  += c.e();

        double dR = jet.delta_R(c);
        width_num += c.pt() * dR;
        width_den += c.pt();
      }

      double vis_pt = std::sqrt(vis_px*vis_px + vis_py*vis_py);
      if (vis_pt < VIS_PT_MIN) continue;

      evt_vis_px += vis_px;
      evt_vis_py += vis_py;
      ++n_jets;

      jet_vis_vecs.emplace_back(vis_px, vis_py);
      jet_axes.emplace_back(jet.eta(), jet.phi());
      // Count dark pions/rhos (TAG_DARK) within this jet's cone via geometric
      // matching — jet.constituents() no longer contains invisible particles.
      int n_dark = 0;
      {
        double ja_eta = jet.eta(), ja_phi = jet.phi();
        for (const auto& ip : invis_ptcls) {
          if (ip.tag != TAG_DARK) continue;
          double deta = ip.eta - ja_eta;
          double dphi = ip.phi - ja_phi;
          if (dphi >  PI) dphi -= 2*PI;
          if (dphi < -PI) dphi += 2*PI;
          if (std::sqrt(deta*deta + dphi*dphi) < jetR) ++n_dark;
        }
      }
      jet_inv_counts.push_back(n_dark);

      // Accumulate all visible constituents for event-level hemisphere masses
      for (const auto& c2 : jet.constituents()) {
        if (c2.user_index() >= TAG_INV) continue;
        all_vis_constits.push_back({c2.px(), c2.py(), c2.pz(), c2.e()});
      }

      // Track leading visible-pT jet for regression + substructure observables
      if (vis_pt > lead_vis_pt) {
        j2_vis_px = j1_vis_px; j2_vis_py = j1_vis_py;
        sub_vis_pt  = lead_vis_pt;
        j1_vis_px   = vis_px;  j1_vis_py = vis_py;
        lead_vis_pt = vis_pt;
        lead_vis_idx = (int)jet_vis_vecs.size() - 1;
        lead_width  = (width_den > 0) ? width_num / width_den : 0.0;
        // Save visible constituents of leading jet for substructure (E2C, E3C, tauN)
        lead_constits.clear();
        for (const auto& c2 : jet.constituents()) {
          if (c2.user_index() >= TAG_INV) continue;
          lead_constits.push_back({c2.px(), c2.py(), c2.pz(), c2.e()});
        }
      } else if (vis_pt > sub_vis_pt) {
        j2_vis_px = vis_px; j2_vis_py = vis_py;
        sub_vis_pt = vis_pt;
      }

      // 4-momentum and ordering for kinematics TSV: visible or full jet
      double cut_pt = jetsVisOnly ? vis_pt : jet.pt();
      double kx = jetsVisOnly ? vis_px : jet.px();
      double ky = jetsVisOnly ? vis_py : jet.py();
      double kz = jetsVisOnly ? vis_pz : jet.pz();
      double ke = jetsVisOnly ? vis_E  : jet.e();

      if (cut_pt > lead_cut_pt) {
        j2_px = j1_px; j2_py = j1_py; j2_pz = j1_pz; j2_E = j1_E;
        sub_cut_pt = lead_cut_pt;
        j1_px = kx; j1_py = ky; j1_pz = kz; j1_E = ke;
        lead_cut_pt = cut_pt;
      } else if (cut_pt > sub_cut_pt) {
        j2_px = kx; j2_py = ky; j2_pz = kz; j2_E = ke;
        sub_cut_pt = cut_pt;
      }
    }

    if (n_jets < 1) continue;
    if (dijetOnly && n_jets < 2) continue;

    double met     = std::sqrt(evt_vis_px*evt_vis_px + evt_vis_py*evt_vis_py);
    double met_phi = std::atan2(-evt_vis_py, -evt_vis_px);

    // ── H_T, R_T, M_eff ──────────────────────────────────────────────────────
    double HT = 0.0;
    for (const auto& jvv : jet_vis_vecs)
      HT += std::sqrt(jvv.first*jvv.first + jvv.second*jvv.second);
    double RT   = (HT > 0) ? met / HT : 0.0;
    double Meff = HT + met;

    // ── Transverse sphericity S_T = 2*lambda_min of the 2x2 sphericity tensor ──
    // Tensor S^{ab} = sum(p_a*p_b) / sum(pT^2), trace = 1, so S_T = 2*lambda_min.
    double spher = 0.0;
    if (s_pt2 > 0) {
      double a = s_xx/s_pt2, b = s_xy/s_pt2, c = s_yy/s_pt2;
      double disc = 1.0 - 4.0*(a*c - b*b);
      spher = 2.0 * (disc > 0 ? (1.0 - std::sqrt(disc)) / 2.0 : 0.5);
    }

    // ── Jet thrust (leading jet) and event-level hemisphere masses ──
    // jetThrust: T = max_{nhat} sum_i |pT_i . nhat| / sum_i |pT_i|  over leading-jet constituents.
    // hemiMass1/2: split ALL visible event particles along the event-level transverse thrust axis.
    double thrust = 0.0, hemi_mass1 = 0.0, hemi_mass2 = 0.0;
    if (!lead_constits.empty()) {
      double pt_sum = 0;
      for (auto& c : lead_constits)
        pt_sum += std::sqrt(c[0]*c[0] + c[1]*c[1]);
      if (pt_sum > 0) {
        for (auto& ac : lead_constits) {
          double anorm = std::sqrt(ac[0]*ac[0] + ac[1]*ac[1]);
          if (anorm == 0) continue;
          double nx = ac[0]/anorm, ny = ac[1]/anorm;
          double T_cand = 0;
          for (auto& c : lead_constits)
            T_cand += std::fabs(c[0]*nx + c[1]*ny);
          T_cand /= pt_sum;
          if (T_cand > thrust) thrust = T_cand;
        }
      }
    }
    if (!all_vis_constits.empty()) {
      // Event-level transverse thrust axis for hemisphere splitting
      double pt_sum_ev = 0;
      for (auto& c : all_vis_constits)
        pt_sum_ev += std::sqrt(c[0]*c[0] + c[1]*c[1]);

      double tx = 1.0, ty = 0.0, best_T = -1.0;
      if (pt_sum_ev > 0) {
        for (auto& ac : all_vis_constits) {
          double anorm = std::sqrt(ac[0]*ac[0] + ac[1]*ac[1]);
          if (anorm == 0) continue;
          double nx = ac[0]/anorm, ny = ac[1]/anorm;
          double T_cand = 0;
          for (auto& c : all_vis_constits)
            T_cand += std::fabs(c[0]*nx + c[1]*ny);
          T_cand /= pt_sum_ev;
          if (T_cand > best_T) { best_T = T_cand; tx = nx; ty = ny; }
        }
      }

      // Split all visible event particles into hemispheres along event thrust axis
      double h1_px=0, h1_py=0, h1_pz=0, h1_E=0;
      double h2_px=0, h2_py=0, h2_pz=0, h2_E=0;
      for (auto& c : all_vis_constits) {
        if (c[0]*tx + c[1]*ty >= 0) {
          h1_px += c[0]; h1_py += c[1]; h1_pz += c[2]; h1_E += c[3];
        } else {
          h2_px += c[0]; h2_py += c[1]; h2_pz += c[2]; h2_E += c[3];
        }
      }
      auto massOf = [](double px, double py, double pz, double E) -> double {
        double m2 = E*E - px*px - py*py - pz*pz;
        return m2 > 0 ? std::sqrt(m2) : 0.0;
      };
      hemi_mass1 = massOf(h1_px, h1_py, h1_pz, h1_E);
      hemi_mass2 = massOf(h2_px, h2_py, h2_pz, h2_E);
      if (hemi_mass1 < hemi_mass2) std::swap(hemi_mass1, hemi_mass2);
    }

    // ── pT balance: |pT_close + pT_far| / (pT_close + pT_far) ──
    // close/far = jet with smallest/largest delta-phi to MET, over all passing jets.
    double pt_bal = 0.0;
    if (n_jets >= 2) {
      double min_dphi = 4.0, max_dphi = -1.0;
      int close_idx = 0, far_idx = 0;
      for (int jj = 0; jj < (int)jet_vis_vecs.size(); ++jj) {
        double jphi = std::atan2(jet_vis_vecs[jj].second, jet_vis_vecs[jj].first);
        double dphi = std::fabs(jphi - met_phi);
        if (dphi > PI) dphi = 2*PI - dphi;
        if (dphi < min_dphi) { min_dphi = dphi; close_idx = jj; }
        if (dphi > max_dphi) { max_dphi = dphi; far_idx   = jj; }
      }
      double vx = jet_vis_vecs[close_idx].first  + jet_vis_vecs[far_idx].first;
      double vy = jet_vis_vecs[close_idx].second + jet_vis_vecs[far_idx].second;
      double vmag = std::sqrt(vx*vx + vy*vy);
      double pt_c = std::sqrt(jet_vis_vecs[close_idx].first*jet_vis_vecs[close_idx].first
                            + jet_vis_vecs[close_idx].second*jet_vis_vecs[close_idx].second);
      double pt_f = std::sqrt(jet_vis_vecs[far_idx].first*jet_vis_vecs[far_idx].first
                            + jet_vis_vecs[far_idx].second*jet_vis_vecs[far_idx].second);
      if (pt_c + pt_f > 0) pt_bal = vmag / (pt_c + pt_f);
    }

    // ── delta-phi(MET, dijet): azimuthal angle between MET and j1+j2 ──
    double dphi_met_dijet = 0.0;
    if (n_jets >= 2) {
      double dijet_phi = std::atan2(j1_vis_py + j2_vis_py, j1_vis_px + j2_vis_px);
      double dphi = std::fabs(dijet_phi - met_phi);
      if (dphi > PI) dphi = 2*PI - dphi;
      dphi_met_dijet = dphi;
    }

    // ── delta-phi(closest/furthest jet to MET, MET) ──
    // Signed: dphi = jet_phi - met_phi, wrapped to (-pi, pi].
    // Close/far determined by |dphi|; stored values are signed.
    double dphi_met_close = 0.0, dphi_met_far = 0.0;
    int close_idx = 0;
    if (n_jets >= 1) {
      double min_abs = 4.0, max_abs = -1.0;
      for (int jj = 0; jj < (int)jet_vis_vecs.size(); ++jj) {
        double jphi = std::atan2(jet_vis_vecs[jj].second, jet_vis_vecs[jj].first);
        double dphi = jphi - met_phi;
        if (dphi >  PI) dphi -= 2*PI;
        if (dphi < -PI) dphi += 2*PI;
        double adphi = std::fabs(dphi);
        if (adphi < min_abs) { min_abs = adphi; dphi_met_close = dphi; close_idx = jj; }
        if (adphi > max_abs) { max_abs = adphi; dphi_met_far   = dphi; }
      }
    }

    // ── is closest jet the leading-pT jet? / dark particle count in closest jet ──
    double close_jet_is_lead = 0.0;
    double n_inv_close       = 0.0;
    if (n_jets >= 1) {
      close_jet_is_lead = (close_idx == lead_vis_idx) ? 1.0 : 0.0;
      n_inv_close       = (double)jet_inv_counts[close_idx];
    }

    // ── f_inv: invisible pT fraction of leading jet (geometric matching) ──────
    // Sums pT of all invisible particles (neutrinos + dark) within the leading
    // jet's cone; denominator is the full jet pT (vis + invisible).
    double f_inv = 0.0;
    if (n_jets >= 1) {
      double la_eta = jet_axes[lead_vis_idx].first;
      double la_phi = jet_axes[lead_vis_idx].second;
      double inv_pt = 0.0;
      for (const auto& ip : invis_ptcls) {
        double deta = ip.eta - la_eta;
        double dphi = ip.phi - la_phi;
        if (dphi >  PI) dphi -= 2*PI;
        if (dphi < -PI) dphi += 2*PI;
        if (std::sqrt(deta*deta + dphi*dphi) < jetR) inv_pt += ip.pt;
      }
      f_inv = (lead_vis_pt + inv_pt > 0) ? inv_pt / (lead_vis_pt + inv_pt) : 0.0;
    }

    // ── Energy Correlators and N-subjettiness (leading visible-pT jet) ──
    // z_i = pT_i / sum_pT  (pT fractions, visible+muon constituents only)
    // E2C = sum_{i<j}   z_i z_j dR_ij                         (beta=1, R=1)
    // E3C = sum_{i<j<k} z_i z_j z_k * dR_ij * dR_ik * dR_jk  (product of 3 angles)
    // tau_N = sum_i z_i * min_k(dR(i, axis_k))                (kT-seeded k-means axes, beta=1)
    double e2c = 0, e3c = 0, tau1 = 0, tau2 = 0, tau3 = 0;
    {
      int nc = (int)lead_constits.size();
      if (nc > 0) {
        std::vector<double> pt_c(nc), eta_c(nc), phi_c(nc), z_c(nc);
        double pt_sum_c = 0;
        for (int ii = 0; ii < nc; ++ii) {
          double px = lead_constits[ii][0], py = lead_constits[ii][1],
                 pz = lead_constits[ii][2];
          double pt = std::sqrt(px*px + py*py);
          double p  = std::sqrt(px*px + py*py + pz*pz);
          pt_c[ii]  = pt;
          phi_c[ii] = std::atan2(py, px);
          eta_c[ii] = (p > std::fabs(pz))
                    ? 0.5*std::log((p+pz)/(p-pz)) : (pz > 0 ? 1e9 : -1e9);
          pt_sum_c += pt;
        }
        if (pt_sum_c > 0)
          for (int ii = 0; ii < nc; ++ii) z_c[ii] = pt_c[ii] / pt_sum_c;

        // Precompute pairwise dR (shared by E2C, E3C, and subjettiness axis search)
        std::vector<std::vector<double>> dRmat(nc, std::vector<double>(nc, 0.0));
        for (int ii = 0; ii < nc; ++ii) {
          for (int jj = ii+1; jj < nc; ++jj) {
            double deta = eta_c[ii] - eta_c[jj];
            double dphi = phi_c[ii] - phi_c[jj];
            if (dphi >  PI) dphi -= 2*PI;
            if (dphi < -PI) dphi += 2*PI;
            dRmat[ii][jj] = dRmat[jj][ii] = std::sqrt(deta*deta + dphi*dphi);
          }
        }

        // E2C
        for (int ii = 0; ii < nc; ++ii)
          for (int jj = ii+1; jj < nc; ++jj)
            e2c += z_c[ii] * z_c[jj] * dRmat[ii][jj];

        // E3C: product of all three pairwise angles
        for (int ii = 0; ii < nc; ++ii)
          for (int jj = ii+1; jj < nc; ++jj)
            for (int kk = jj+1; kk < nc; ++kk)
              e3c += z_c[ii] * z_c[jj] * z_c[kk]
                   * dRmat[ii][jj] * dRmat[ii][kk] * dRmat[jj][kk];

        // tau_N: kT-seeded k-means N-subjettiness (beta=1).
        // For N>1, two seeds are tried: the standard kT exclusive_jets(N) seed, and a
        // warm-start seeded with the previous level's converged axes plus the Nth kT jet.
        // At initialisation the warm-start satisfies tau_N <= tau_{N-1} (the extra axis
        // can only reduce each constituent's minimum distance), so k-means converges to
        // a result <= tau_{N-1} — no post-hoc clamping required.
        std::vector<fastjet::PseudoJet> cpjs;
        cpjs.reserve(nc);
        for (auto& c : lead_constits)
          cpjs.emplace_back(c[0], c[1], c[2], c[3]);

        fastjet::JetDefinition kt_def(fastjet::kt_algorithm, 1.0);
        fastjet::ClusterSequence cs_sub(cpjs, kt_def);

        // Run k-means from given axes (modified in-place); returns converged tau.
        auto run_kmeans = [&](int N,
                              std::vector<double>& ax_eta,
                              std::vector<double>& ax_phi) -> double {
          for (int iter = 0; iter < 100; ++iter) {
            std::vector<int> asgn(nc, 0);
            for (int ii = 0; ii < nc; ++ii) {
              double min_d = 1e30;
              for (int k = 0; k < N; ++k) {
                double deta = eta_c[ii] - ax_eta[k];
                double dphi = phi_c[ii] - ax_phi[k];
                if (dphi >  PI) dphi -= 2*PI;
                if (dphi < -PI) dphi += 2*PI;
                double d = std::sqrt(deta*deta + dphi*dphi);
                if (d < min_d) { min_d = d; asgn[ii] = k; }
              }
            }
            std::vector<double> spx(N,0), spy(N,0), spz(N,0);
            for (int ii = 0; ii < nc; ++ii) {
              int k = asgn[ii];
              spx[k] += lead_constits[ii][0];
              spy[k] += lead_constits[ii][1];
              spz[k] += lead_constits[ii][2];
            }
            double max_shift = 0.0;
            for (int k = 0; k < N; ++k) {
              double pt2 = spx[k]*spx[k] + spy[k]*spy[k];
              double p2  = pt2 + spz[k]*spz[k];
              if (pt2 < 1e-20 || p2 < 1e-20) continue;
              double p = std::sqrt(p2);
              double new_eta = (p > std::fabs(spz[k]))
                               ? 0.5*std::log((p+spz[k])/(p-spz[k]))
                               : (spz[k] > 0 ? 1e9 : -1e9);
              double new_phi = std::atan2(spy[k], spx[k]);
              double deta = new_eta - ax_eta[k];
              double dphi = new_phi - ax_phi[k];
              if (dphi >  PI) dphi -= 2*PI;
              if (dphi < -PI) dphi += 2*PI;
              double shift = std::sqrt(deta*deta + dphi*dphi);
              if (shift > max_shift) max_shift = shift;
              ax_eta[k] = new_eta;
              ax_phi[k] = new_phi;
            }
            if (max_shift < 1e-6) break;
          }
          double tau = 0;
          for (int ii = 0; ii < nc; ++ii) {
            double min_d = 1e30;
            for (int k = 0; k < N; ++k) {
              double deta = eta_c[ii] - ax_eta[k];
              double dphi = phi_c[ii] - ax_phi[k];
              if (dphi >  PI) dphi -= 2*PI;
              if (dphi < -PI) dphi += 2*PI;
              double d = std::sqrt(deta*deta + dphi*dphi);
              if (d < min_d) min_d = d;
            }
            tau += z_c[ii] * min_d;
          }
          return tau;
        };

        // tau1: single kT seed; ax1e/ax1p hold converged axis for tau2 warm-start
        std::vector<double> ax1e, ax1p;
        if (nc >= 1) {
          auto s1 = cs_sub.exclusive_jets(1);
          ax1e = {s1[0].eta()};
          ax1p = {s1[0].phi()};
          tau1 = run_kmeans(1, ax1e, ax1p);
        }

        // tau2: kT seed AND warm-start [ax1_opt, kT-2nd]; ax2e/ax2p hold best axes
        std::vector<double> ax2e, ax2p;
        if (nc >= 2) {
          auto s2 = cs_sub.exclusive_jets(2);

          std::vector<double> e2a = {s2[0].eta(), s2[1].eta()};
          std::vector<double> p2a = {s2[0].phi(), s2[1].phi()};
          double tau2a = run_kmeans(2, e2a, p2a);

          // Warm-start: ax1 optimum + kT-2nd jet.  Initial tau <= tau1 by construction.
          std::vector<double> e2b = {ax1e[0], s2[1].eta()};
          std::vector<double> p2b = {ax1p[0], s2[1].phi()};
          double tau2b = run_kmeans(2, e2b, p2b);

          if (tau2a <= tau2b) { tau2 = tau2a; ax2e = e2a; ax2p = p2a; }
          else                { tau2 = tau2b; ax2e = e2b; ax2p = p2b; }
        }

        // tau3: kT seed AND warm-start [ax2_opt_0, ax2_opt_1, kT-3rd]
        if (nc >= 3) {
          auto s3 = cs_sub.exclusive_jets(3);

          std::vector<double> e3a = {s3[0].eta(), s3[1].eta(), s3[2].eta()};
          std::vector<double> p3a = {s3[0].phi(), s3[1].phi(), s3[2].phi()};
          double tau3a = run_kmeans(3, e3a, p3a);

          // Warm-start: both ax2 optima + kT-3rd jet.  Initial tau <= tau2 by construction.
          std::vector<double> e3b = {ax2e[0], ax2e[1], s3[2].eta()};
          std::vector<double> p3b = {ax2p[0], ax2p[1], s3[2].phi()};
          double tau3b = run_kmeans(3, e3b, p3b);

          tau3 = std::min(tau3a, tau3b);
        }
      }
    }

    // ── Leading jet invariant mass and constituent multiplicity ──────────────
    double lead_jet_mass = 0.0;
    double n_const       = 0.0;
    if (!lead_constits.empty()) {
      double mpx = 0, mpy = 0, mpz = 0, mE = 0;
      for (const auto& c : lead_constits) {
        mpx += c[0]; mpy += c[1]; mpz += c[2]; mE += c[3];
      }
      double m2 = mE*mE - mpx*mpx - mpy*mpy - mpz*mpz;
      lead_jet_mass = (m2 > 0) ? std::sqrt(m2) : 0.0;
      n_const = (double)lead_constits.size();
    }

    // ADD NEW OBSERVABLE COMPUTATIONS ABOVE THIS LINE, then append to push_back below.
    // Keep push_back values in the same order as OBS_NAMES.
    data.push_back({lead_vis_pt, lead_width, met,
                    max_ele_pt, max_mu_pt, thrust,
                    spher, hemi_mass1, hemi_mass2,
                    pt_bal, dphi_met_dijet,
                    e2c, e3c, tau1, tau2, tau3,
                    dphi_met_close, dphi_met_far, (double)n_jets,
                    close_jet_is_lead, n_inv_close, met_phi,
                    HT, RT, Meff,
                    lead_jet_mass, n_const, f_inv});
    jetData.push_back({n_jets,
                       j1_px, j1_py, j1_pz, j1_E,
                       j2_px, j2_py, j2_pz, j2_E});
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
