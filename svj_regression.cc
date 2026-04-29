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

// user_index tags
static const int TAG_VIS  = 0;
static const int TAG_MUON = 1;
static const int TAG_INV  = 2;

// Per-event jet kinematics (visible 4-momentum for leading and subleading jet)
struct JetKin {
  int    n_jets;
  double j1_px, j1_py, j1_pz, j1_E;
  double j2_px, j2_py, j2_pz, j2_E;
};

// Physics parameters
static double mZ        = 2000.0;
static double mq        =    4.0;
static double mPi       =    8.0;
static double mRho      =   15.5;
static double rinv      =    0.3;
static double rinv2     =    0.3;
static double Brl       =    0.3;
static double alphaD    =    0.4;
static int    nEvent    = 100000;
static double jetR      =    1.0;
static double LambdaDQCD =   5.0;
static int    nWorkers  =   10;
static int    saveTSV   =    1;   // write raw TSV; set to 0 during scan
static int    jetsVisOnly = 1;    // 1 = store visible 4-momenta in jets TSV; 0 = full jet 4-momenta

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
    vis << 1. - rinv;
    inv << rinv;
    pythia.readString("4900111:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("4900111:addChannel = 1 " + vis.str() + " 91 4 -4");
    pythia.readString("4900211:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("4900211:addChannel = 1 " + vis.str() + " 91 4 -4");
    pythia.readString("-4900211:oneChannel = 1 " + inv.str() + " 0 51 -51");
    pythia.readString("-4900211:addChannel = 1 " + vis.str() + " 91 4 -4");
  }
  {
    std::ostringstream lep, bott, inv2;
    inv2 << rinv2;
    bott << 1. - Brl - rinv2;
    lep  << Brl;

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

// ── 3×3 symmetric linear algebra (analytic / no external deps) ───────────────

static double det3(const double s[3][3]) {
  return s[0][0]*(s[1][1]*s[2][2] - s[1][2]*s[1][2])
        -s[0][1]*(s[0][1]*s[2][2] - s[1][2]*s[0][2])
        +s[0][2]*(s[0][1]*s[1][2] - s[1][1]*s[0][2]);
}

// Inverse of symmetric 3×3 via cofactors (result stored in inv[][]).
static void inv3(const double s[3][3], double r[3][3]) {
  double d = det3(s);
  r[0][0] =  (s[1][1]*s[2][2] - s[1][2]*s[1][2]) / d;
  r[0][1] = r[1][0] = -(s[0][1]*s[2][2] - s[1][2]*s[0][2]) / d;
  r[0][2] = r[2][0] =  (s[0][1]*s[1][2] - s[1][1]*s[0][2]) / d;
  r[1][1] =  (s[0][0]*s[2][2] - s[0][2]*s[0][2]) / d;
  r[1][2] = r[2][1] = -(s[0][0]*s[1][2] - s[0][1]*s[0][2]) / d;
  r[2][2] =  (s[0][0]*s[1][1] - s[0][1]*s[0][1]) / d;
}

// Mahalanobis distance squared: (x-mu)^T Sinv (x-mu)
static double mahal3(const double* x, const double* mu,
                     const double si[3][3]) {
  double d0 = x[0]-mu[0], d1 = x[1]-mu[1], d2 = x[2]-mu[2];
  return d0*(si[0][0]*d0 + si[0][1]*d1 + si[0][2]*d2)
        +d1*(si[1][0]*d0 + si[1][1]*d1 + si[1][2]*d2)
        +d2*(si[2][0]*d0 + si[2][1]*d1 + si[2][2]*d2);
}

// ── Multivariate-t EM fit (p = 3) ────────────────────────────────────────────
// Returns fitted (mu, Sigma, nu) where Sigma is the scatter matrix
// (not the covariance; for a t-dist, Cov = nu/(nu-2)*Sigma when nu>2).
//
// Output 10 scalars in order:
//   mu[0..2],  Sigma[00 01 02 11 12 22],  nu

struct MVTResult {
  double mu[3];
  double Sigma[3][3];
  double nu;
  int    n_used;
};

static MVTResult fitMVT(const std::vector<std::array<double,3>>& data,
                         int maxIter = 200, double tol = 1e-6) {
  const int p = 3;
  int n = (int)data.size();

  // ── initialise from sample mean & covariance ──
  double mu[3] = {0,0,0};
  for (auto& x : data)
    for (int i=0; i<3; i++) mu[i] += x[i];
  for (int i=0; i<3; i++) mu[i] /= n;

  double Sigma[3][3] = {};
  for (auto& x : data) {
    double d[3] = {x[0]-mu[0], x[1]-mu[1], x[2]-mu[2]};
    for (int i=0; i<3; i++)
      for (int j=0; j<3; j++)
        Sigma[i][j] += d[i]*d[j];
  }
  for (int i=0; i<3; i++)
    for (int j=0; j<3; j++)
      Sigma[i][j] /= n;

  double nu = 10.0;
  double prevLL = -1e30;

  std::vector<double> w(n), delta(n);

  for (int iter = 0; iter < maxIter; iter++) {

    // ── E-step: Mahalanobis distances and weights ──
    double Sinv[3][3];
    inv3(Sigma, Sinv);
    double logdet = std::log(std::fabs(det3(Sigma)));

    double wsum = 0.0;
    for (int k = 0; k < n; k++) {
      delta[k] = mahal3(data[k].data(), mu, Sinv);
      w[k]     = (nu + p) / (nu + delta[k]);
      wsum    += w[k];
    }

    // ── M-step: mu ──
    double mu_new[3] = {0,0,0};
    for (int k = 0; k < n; k++)
      for (int i = 0; i < 3; i++)
        mu_new[i] += w[k] * data[k][i];
    for (int i = 0; i < 3; i++) mu_new[i] /= wsum;

    // ── M-step: Sigma ──
    double Sigma_new[3][3] = {};
    for (int k = 0; k < n; k++) {
      double d[3] = {data[k][0]-mu_new[0],
                     data[k][1]-mu_new[1],
                     data[k][2]-mu_new[2]};
      for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
          Sigma_new[i][j] += w[k] * d[i] * d[j];
    }
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        Sigma_new[i][j] /= n;

    // ── Precompute delta_new for nu search ──
    double Sinv_new[3][3];
    inv3(Sigma_new, Sinv_new);
    double logdet_new = std::log(std::fabs(det3(Sigma_new)));
    for (int k = 0; k < n; k++)
      delta[k] = mahal3(data[k].data(), mu_new, Sinv_new);

    // ── M-step: nu via golden-section over log(nu) in [-4.6, 6.9] ──
    // i.e., nu in [~0.01, ~1000]
    auto negLL = [&](double log_nu) -> double {
      double nu_t = std::exp(log_nu);
      double ll = n * (std::lgamma((nu_t+p)/2.) - std::lgamma(nu_t/2.)
                       - (p/2.) * std::log(nu_t))
                 - (n/2.) * logdet_new;
      for (int k = 0; k < n; k++)
        ll -= ((nu_t+p)/2.) * std::log1p(delta[k] / nu_t);
      return -ll;
    };

    double a = -4.6, b = 6.9;
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;
    for (int gs = 0; gs < 60; gs++) {
      double c = b - phi*(b-a);
      double dd = a + phi*(b-a);
      if (negLL(c) < negLL(dd)) b = dd;
      else                       a = c;
    }
    double nu_new = std::exp((a+b) / 2.0);

    // ── Convergence check via log-likelihood ──
    double ll = n * (std::lgamma((nu_new+p)/2.) - std::lgamma(nu_new/2.)
                     - (p/2.) * std::log(nu_new))
               - (n/2.) * logdet_new;
    for (int k = 0; k < n; k++)
      ll -= ((nu_new+p)/2.) * std::log1p(delta[k] / nu_new);

    for (int i = 0; i < 3; i++) mu[i] = mu_new[i];
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        Sigma[i][j] = Sigma_new[i][j];
    nu = nu_new;

    if (std::fabs(ll - prevLL) < tol) break;
    prevLL = ll;
  }

  MVTResult res;
  for (int i = 0; i < 3; i++) res.mu[i] = mu[i];
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      res.Sigma[i][j] = Sigma[i][j];
  res.nu = nu;
  res.n_used = n;
  return res;
}

// ── Worker: collect per-event observables ────────────────────────────────────
// data[] layout (16 values per event):
//   0  leadVisPt     1  leadWidth     2  MET
//   3  maxElePt      4  maxMuPt
//   5  jetThrust     6  transSphericity
//   7  hemiMass1     8  hemiMass2
//   9  ptBal         10 dPhiMETdijet
//   11 e2c           12 e3c
//   13 tau1          14 tau2          15 tau3
static void runWorker(int workerID, int nEvtWorker,
                      std::vector<std::array<double,16>>& data,
                      std::vector<JetKin>& jetData) {
  Pythia pythia;
  if (!setupPythia(pythia, workerID + 1)) return;

  Event& event = pythia.event;
  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, jetR);

  const double VIS_PT_MIN = 20.0;
  const double ETA_MAX    =  2.5;
  const double PI         = std::acos(-1.0);

  for (int iEvent = 0; iEvent < nEvtWorker; ++iEvent) {
    if (!pythia.next()) continue;

    std::vector<fastjet::PseudoJet> particles;
    particles.reserve(event.size());

    // Tracked during particle loop
    double max_ele_pt = 0.0, max_mu_pt = 0.0;
    double s_xx = 0, s_xy = 0, s_yy = 0, s_pt2 = 0;  // sphericity tensor sums

    for (int i = 0; i < event.size(); ++i) {
      const Particle& p = event[i];
      if (!p.isFinal()) continue;

      int aid = p.idAbs();
      int id  = p.id();

      int ptag;
      if (aid == 12 || aid == 14 || aid == 16 ||
          id  == 51 || id  == -51 ||
          id  == 53 || id  == -53)
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
      if (ptag != TAG_INV) {
        double px = p.px(), py = p.py();
        s_xx += px*px; s_xy += px*py; s_yy += py*py;
        s_pt2 += px*px + py*py;
      }

      fastjet::PseudoJet pj(p.px(), p.py(), p.pz(), p.e());
      pj.set_user_index(ptag);
      particles.push_back(pj);
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
    std::vector<std::array<double,4>> lead_constits;          // visible constituents of leading jet
    std::vector<std::pair<double,double>> jet_vis_vecs;       // (vis_px, vis_py) per passing jet

    for (const auto& jet : raw_jets) {
      if (std::fabs(jet.eta()) >= ETA_MAX) continue;

      double vis_px = 0, vis_py = 0, vis_pz = 0, vis_E = 0;
      double width_num = 0, width_den = 0;

      for (const auto& c : jet.constituents()) {
        if (c.user_index() == TAG_INV) continue;   // muons (TAG_MUON) count as visible

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

      // Track leading visible-pT jet for regression + thrust + dijet observables
      if (vis_pt > lead_vis_pt) {
        j2_vis_px = j1_vis_px; j2_vis_py = j1_vis_py;
        sub_vis_pt  = lead_vis_pt;
        j1_vis_px   = vis_px;  j1_vis_py = vis_py;
        lead_vis_pt = vis_pt;
        lead_width  = (width_den > 0) ? width_num / width_den : 0.0;
        // Save visible constituents for thrust and hemisphere masses
        lead_constits.clear();
        for (const auto& c2 : jet.constituents()) {
          if (c2.user_index() == TAG_INV) continue;
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

    double met = std::sqrt(evt_vis_px*evt_vis_px + evt_vis_py*evt_vis_py);

    // ── Transverse sphericity S_T = 2*lambda_min of the 2x2 sphericity tensor ──
    // Tensor S^{ab} = sum(p_a*p_b) / sum(pT^2), trace = 1, so S_T = 2*lambda_min.
    double spher = 0.0;
    if (s_pt2 > 0) {
      double a = s_xx/s_pt2, b = s_xy/s_pt2, c = s_yy/s_pt2;
      double disc = 1.0 - 4.0*(a*c - b*b);
      spher = 2.0 * (disc > 0 ? (1.0 - std::sqrt(disc)) / 2.0 : 0.5);
    }

    // ── Jet thrust and hemisphere masses (leading visible-pT jet) ──
    // Thrust: T = max_{nhat} sum_i |pT_i . nhat| / sum_i |pT_i|
    // Optimal nhat lies along one of the constituent pT directions (O(n^2) search).
    double thrust = 0.0, hemi_mass1 = 0.0, hemi_mass2 = 0.0;
    if (!lead_constits.empty()) {
      double pt_sum = 0;
      for (auto& c : lead_constits)
        pt_sum += std::sqrt(c[0]*c[0] + c[1]*c[1]);

      double tx = 1.0, ty = 0.0;
      if (pt_sum > 0) {
        for (auto& ac : lead_constits) {
          double anorm = std::sqrt(ac[0]*ac[0] + ac[1]*ac[1]);
          if (anorm == 0) continue;
          double nx = ac[0]/anorm, ny = ac[1]/anorm;
          double T_cand = 0;
          for (auto& c : lead_constits)
            T_cand += std::fabs(c[0]*nx + c[1]*ny);
          T_cand /= pt_sum;
          if (T_cand > thrust) { thrust = T_cand; tx = nx; ty = ny; }
        }
      }

      // Split constituents into hemispheres along thrust axis
      double h1_px=0, h1_py=0, h1_pz=0, h1_E=0;
      double h2_px=0, h2_py=0, h2_pz=0, h2_E=0;
      for (auto& c : lead_constits) {
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
      double met_phi = std::atan2(-evt_vis_py, -evt_vis_px);
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
      double met_phi   = std::atan2(-evt_vis_py, -evt_vis_px);
      double dphi = std::fabs(dijet_phi - met_phi);
      if (dphi > PI) dphi = 2*PI - dphi;
      dphi_met_dijet = dphi;
    }

    // ── Energy Correlators and N-subjettiness (leading visible-pT jet) ──
    // z_i = pT_i / sum_pT  (pT fractions, visible+muon constituents only)
    // E2C = sum_{i<j}   z_i z_j dR_ij                         (beta=1, R=1)
    // E3C = sum_{i<j<k} z_i z_j z_k * dR_ij * dR_ik * dR_jk  (product of 3 angles)
    // tau_N = sum_i z_i * min_k(dR(i, axis_k))                (exclusive kt axes, beta=1)
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

        // tau_N: exclusive kt reclustering of visible constituents to N subjets
        std::vector<fastjet::PseudoJet> cpjs;
        cpjs.reserve(nc);
        for (auto& c : lead_constits)
          cpjs.emplace_back(c[0], c[1], c[2], c[3]);

        fastjet::JetDefinition kt_def(fastjet::kt_algorithm, 1.0);
        fastjet::ClusterSequence cs_sub(cpjs, kt_def);

        // Returns tau_N using dRmat for constituent-to-axis distances
        auto compute_tau = [&](int N) -> double {
          if (nc < N) return 0.0;
          auto axes = cs_sub.exclusive_jets(N);
          // Build eta/phi for each axis
          std::vector<double> ax_eta(N), ax_phi(N);
          for (int k = 0; k < N; ++k) {
            ax_eta[k] = axes[k].eta();
            ax_phi[k] = axes[k].phi();  // FastJet phi in [-pi, pi]
          }
          double tau = 0;
          for (int ii = 0; ii < nc; ++ii) {
            double min_dR = 1e30;
            for (int k = 0; k < N; ++k) {
              double deta = eta_c[ii] - ax_eta[k];
              double dphi = phi_c[ii] - ax_phi[k];
              if (dphi >  PI) dphi -= 2*PI;
              if (dphi < -PI) dphi += 2*PI;
              double d = std::sqrt(deta*deta + dphi*dphi);
              if (d < min_dR) min_dR = d;
            }
            tau += z_c[ii] * min_dR;
          }
          return tau;
        };

        tau1 = compute_tau(1);
        tau2 = compute_tau(2);
        tau3 = compute_tau(3);
      }
    }

    data.push_back({lead_vis_pt, lead_width, met,
                    max_ele_pt, max_mu_pt, thrust,
                    spher, hemi_mass1, hemi_mass2,
                    pt_bal, dphi_met_dijet,
                    e2c, e3c, tau1, tau2, tau3});
    jetData.push_back({n_jets,
                       j1_px, j1_py, j1_pz, j1_E,
                       j2_px, j2_py, j2_pz, j2_E});
  }
}

int main(int argc, char* argv[]) {

  std::string cfgPath = "svj_regression.cfg";
  if (argc > 1) cfgPath = argv[1];

  auto cfg = readConfig(cfgPath);
  if (cfg.empty() && argc > 1)
    std::cerr << "Warning: could not read config file: " << cfgPath << "\n";

  mZ         = cfgDouble(cfg, "mZ",         mZ);
  mq         = cfgDouble(cfg, "mq",         mq);
  mPi        = cfgDouble(cfg, "mPi",        mPi);
  mRho       = cfgDouble(cfg, "mRho",       mRho);
  rinv       = cfgDouble(cfg, "rinv",       rinv);
  rinv2      = cfgDouble(cfg, "rinv2",      rinv2);
  Brl        = cfgDouble(cfg, "Brl",        Brl);
  alphaD     = cfgDouble(cfg, "alphaD",     alphaD);
  nEvent     = cfgInt   (cfg, "nEvent",     nEvent);
  jetR       = cfgDouble(cfg, "jetR",       jetR);
  LambdaDQCD = cfgDouble(cfg, "LambdaDQCD", LambdaDQCD);
  nWorkers   = cfgInt   (cfg, "nWorkers",   nWorkers);
  saveTSV    = cfgInt   (cfg, "save_tsv",    saveTSV);
  jetsVisOnly = cfgInt  (cfg, "jets_vis_only", jetsVisOnly);

  if (Brl + rinv2 > 1.0) {
    std::cerr << "Error: Brl + rinv2 = " << Brl + rinv2
              << " > 1 (dark rho BRs unphysical)\n";
    return 1;
  }

  int base      = nEvent / nWorkers;
  int remainder = nEvent % nWorkers;

  std::vector<std::vector<std::array<double,16>>> results(nWorkers);
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
    std::filesystem::create_directories("data/regression");

    std::ofstream fOut("data/regression/jets_default.tsv");
    fOut << "# leadVisPt\tleadWidth\tMET"
            "\tmaxElePt\tmaxMuPt"
            "\tjetThrust\ttransSphericity"
            "\themiMass1\themiMass2"
            "\tptBal\tdPhiMETdijet"
            "\te2c\te3c"
            "\ttau1\ttau2\ttau3\n";
    fOut << std::scientific << std::setprecision(6);
    for (const auto& rows : results)
      for (const auto& r : rows)
        fOut << r[0]  << "\t" << r[1]  << "\t" << r[2]  << "\t"
             << r[3]  << "\t" << r[4]  << "\t" << r[5]  << "\t"
             << r[6]  << "\t" << r[7]  << "\t" << r[8]  << "\t"
             << r[9]  << "\t" << r[10] << "\t"
             << r[11] << "\t" << r[12] << "\t"
             << r[13] << "\t" << r[14] << "\t" << r[15] << "\n";

    std::ofstream fJets("data/regression/jets_kinematics.tsv");
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

  // Aggregate workers, apply transform, filter
  std::vector<std::array<double,3>> fitData;
  for (const auto& rows : results) {
    for (const auto& r : rows) {
      double pT = r[0], width = r[1], met = r[2];
      if (width <= 0.0 || met <= 0.0) continue;
      fitData.push_back({pT, std::log(width), std::log(met)});
    }
  }

  if ((int)fitData.size() < 10) {
    std::cerr << "Error: only " << fitData.size()
              << " events passed filter — cannot fit.\n";
    std::cout << "RESULT: nan nan nan nan nan nan nan nan nan nan\n";
    return 1;
  }

  MVTResult res = fitMVT(fitData);
  std::cerr << "Fit used " << res.n_used << " events, nu = " << res.nu << "\n";

  // Print 10 fitted parameters to stdout.
  // Format: RESULT: mu0 mu1 mu2 S00 S01 S02 S11 S12 S22 nu
  // (mu0=mean pT, mu1=mean log(width), mu2=mean log(MET);
  //  S** are upper-triangle of the 3x3 scatter matrix;
  //  nu is the degrees-of-freedom parameter)
  std::cout << std::setprecision(10) << std::scientific
            << "RESULT:"
            << " " << res.mu[0]
            << " " << res.mu[1]
            << " " << res.mu[2]
            << " " << res.Sigma[0][0]
            << " " << res.Sigma[0][1]
            << " " << res.Sigma[0][2]
            << " " << res.Sigma[1][1]
            << " " << res.Sigma[1][2]
            << " " << res.Sigma[2][2]
            << " " << res.nu
            << "\n";

  return 0;
}
