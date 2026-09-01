// svj_observables_common.h
// ============================================================================
// Shared observable-computation code, extracted from svj_regression.cc so
// both the truth-level binary (svj_regression.cc) and the Delphes production
// binary (svj_regression_delphes.cc) compute the SAME ~27 observables from
// the literal same code -- adding a new observable means editing this file
// once, not twice, and truth/Delphes definitions can't silently drift apart.
//
// Header-only (all functions `inline`) so neither binary's build needs any
// Makefile changes beyond an extra #include -- each binary still compiles
// from a single translation unit, matching the existing one-file-per-binary
// Makefile pattern (see docs/setup.md).
//
// DESIGN: computeSvjObservables() takes an already-classified SvjJetInputs
// (a list of TAG_VIS/TAG_MUON-tagged particles to cluster, plus whatever
// truth-level invisible-particle info is available for the geometric
// nInvClose/fInv matching -- empty for Delphes, which has no invisible
// candidates by construction). Building that classified input FROM a raw
// Pythia8::Event (truth) or FROM Delphes EFlow candidates is necessarily
// different between the two binaries and stays in each binary's own file
// (buildTruthParticles()-style adapters) -- only the classification-to-
// observables computation below is shared. This mirrors exactly the
// pattern already used in svj_delphes_test.cc's clusterAndExtract().
//
// nInvClose/fInv (and, less fundamentally, closeJetIsLead) are truth-only
// concepts -- they require geometric matching to invisible dark-sector
// particles, which by definition don't exist in a reconstructed detector
// output. Passing an empty invis_ptcls list (as the Delphes binary does)
// naturally makes them evaluate to 0/trivial rather than requiring a
// separate code path -- callers that don't want them can simply not expose
// those fields in their own OBS_NAMES/output.
// ============================================================================

#pragma once

#include "fastjet/ClusterSequence.hh"

#include <vector>
#include <array>
#include <cmath>
#include <algorithm>

// user_index tags for classified input particles
static const int TAG_VIS  = 0;
static const int TAG_MUON = 1;
static const int TAG_INV  = 2;  // neutrinos (pid 12/14/16) -- truth only
static const int TAG_DARK = 3;  // dark pions/rhos (pid 51/53) -- truth only

// Invisible final-state particle (neutrino or dark), for geometric matching
// into visible-only jet cones. Truth-only -- always empty for Delphes.
struct InvisPtcl { double eta, phi, pt; int tag; };

// Per-event jet kinematics (visible 4-momentum for leading and subleading
// jet) -- optional output, used by svj_regression.cc for jets_kinematics.tsv.
struct JetKin {
  int    n_jets;
  double j1_px, j1_py, j1_pz, j1_E;
  double j2_px, j2_py, j2_pz, j2_E;
};

// Classified inputs computeSvjObservables() needs. Built differently by each
// binary's own adapter, consumed identically here.
struct SvjJetInputs {
  std::vector<fastjet::PseudoJet> particles;   // TAG_VIS/TAG_MUON, user_index set
  std::vector<InvisPtcl> invis_ptcls;          // empty if no truth-level invisible info
  double max_ele_pt = 0.0;
  double max_mu_pt  = 0.0;
  double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0, s_pt2 = 0.0;  // sphericity tensor sums
};

// Full observable tuple, same order/definitions as svj_regression.cc's
// original OBS_NAMES. Zero-initialised so an unused field (e.g. nInvClose
// when invis_ptcls is empty) reads as a well-defined 0, not garbage.
struct SvjObservables {
  double leadVisPt = 0, leadWidth = 0, MET = 0;
  double maxElePt = 0, maxMuPt = 0;
  double jetThrust = 0, transSphericity = 0;
  double hemiMass1 = 0, hemiMass2 = 0;
  double ptBal = 0, dPhiMETdijet = 0;
  double e2c = 0, e3c = 0;
  double tau1 = 0, tau2 = 0, tau3 = 0;
  double dPhiMETclose = 0, dPhiMETfar = 0, nJets = 0;
  double closeJetIsLead = 0, nInvClose = 0, metPhi = 0;
  double HT = 0, RT = 0, Meff = 0;
  double leadJetMass = 0, nConst = 0, fInv = 0;
};

// Clusters `in.particles`, computes the full SvjObservables tuple, and
// returns whether the event should be kept (false if it should be
// discarded -- n_jets < 1, or dijetOnly requested and n_jets < 2 -- mirroring
// svj_regression.cc's original `if (n_jets < 1) continue;` / dijetOnly check
// exactly). jetKinOut, if non-null, is filled with the leading/subleading
// jet 4-momenta (svj_regression.cc-specific; pass nullptr if not needed).
inline bool computeSvjObservables(const SvjJetInputs& in,
                                  double jetR, double visJetPtMin, double etaMax,
                                  bool jetsVisOnly, bool dijetOnly,
                                  SvjObservables& out, JetKin* jetKinOut = nullptr) {
  const double PI = std::acos(-1.0);
  const double VIS_PT_MIN = visJetPtMin;
  const double ETA_MAX = etaMax;

  fastjet::JetDefinition jet_def(fastjet::antikt_algorithm, jetR);
  fastjet::ClusterSequence cs(in.particles, jet_def);
  std::vector<fastjet::PseudoJet> raw_jets =
      fastjet::sorted_by_pt(cs.inclusive_jets(1.0));

  double evt_vis_px = 0, evt_vis_py = 0;
  int    n_jets = 0;
  double lead_vis_pt = -1.0, sub_vis_pt = -1.0;
  double lead_cut_pt = -1.0, sub_cut_pt = -1.0;
  double lead_width  =  0.0;
  double j1_vis_px = 0, j1_vis_py = 0;
  double j2_vis_px = 0, j2_vis_py = 0;
  double j1_px=0, j1_py=0, j1_pz=0, j1_E=0;
  double j2_px=0, j2_py=0, j2_pz=0, j2_E=0;
  std::vector<std::array<double,4>> lead_constits;
  std::vector<std::array<double,4>> all_vis_constits;
  std::vector<std::pair<double,double>> jet_vis_vecs;
  std::vector<std::pair<double,double>> jet_axes;
  std::vector<int> jet_inv_counts;
  int lead_vis_idx = 0;

  for (const auto& jet : raw_jets) {
    if (std::fabs(jet.eta()) >= ETA_MAX) continue;

    double vis_px = 0, vis_py = 0, vis_pz = 0, vis_E = 0;
    double width_num = 0, width_den = 0;

    for (const auto& c : jet.constituents()) {
      if (c.user_index() >= TAG_INV) continue;

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

    int n_dark = 0;
    {
      double ja_eta = jet.eta(), ja_phi = jet.phi();
      for (const auto& ip : in.invis_ptcls) {
        if (ip.tag != TAG_DARK) continue;
        double deta = ip.eta - ja_eta;
        double dphi = ip.phi - ja_phi;
        if (dphi >  PI) dphi -= 2*PI;
        if (dphi < -PI) dphi += 2*PI;
        if (std::sqrt(deta*deta + dphi*dphi) < jetR) ++n_dark;
      }
    }
    jet_inv_counts.push_back(n_dark);

    for (const auto& c2 : jet.constituents()) {
      if (c2.user_index() >= TAG_INV) continue;
      all_vis_constits.push_back({c2.px(), c2.py(), c2.pz(), c2.e()});
    }

    if (vis_pt > lead_vis_pt) {
      j2_vis_px = j1_vis_px; j2_vis_py = j1_vis_py;
      sub_vis_pt  = lead_vis_pt;
      j1_vis_px   = vis_px;  j1_vis_py = vis_py;
      lead_vis_pt = vis_pt;
      lead_vis_idx = (int)jet_vis_vecs.size() - 1;
      lead_width  = (width_den > 0) ? width_num / width_den : 0.0;
      lead_constits.clear();
      for (const auto& c2 : jet.constituents()) {
        if (c2.user_index() >= TAG_INV) continue;
        lead_constits.push_back({c2.px(), c2.py(), c2.pz(), c2.e()});
      }
    } else if (vis_pt > sub_vis_pt) {
      j2_vis_px = vis_px; j2_vis_py = vis_py;
      sub_vis_pt = vis_pt;
    }

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

  if (jetKinOut) {
    *jetKinOut = JetKin{n_jets, j1_px, j1_py, j1_pz, j1_E, j2_px, j2_py, j2_pz, j2_E};
  }

  if (n_jets < 1) return false;
  if (dijetOnly && n_jets < 2) return false;

  double met     = std::sqrt(evt_vis_px*evt_vis_px + evt_vis_py*evt_vis_py);
  double met_phi = std::atan2(-evt_vis_py, -evt_vis_px);

  double HT = 0.0;
  for (const auto& jvv : jet_vis_vecs)
    HT += std::sqrt(jvv.first*jvv.first + jvv.second*jvv.second);
  double RT   = (HT > 0) ? met / HT : 0.0;
  double Meff = HT + met;

  double spher = 0.0;
  if (in.s_pt2 > 0) {
    double a = in.s_xx/in.s_pt2, b = in.s_xy/in.s_pt2, c = in.s_yy/in.s_pt2;
    double disc = 1.0 - 4.0*(a*c - b*b);
    spher = 2.0 * (disc > 0 ? (1.0 - std::sqrt(disc)) / 2.0 : 0.5);
  }

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

  double dphi_met_dijet = 0.0;
  if (n_jets >= 2) {
    double dijet_phi = std::atan2(j1_vis_py + j2_vis_py, j1_vis_px + j2_vis_px);
    double dphi = std::fabs(dijet_phi - met_phi);
    if (dphi > PI) dphi = 2*PI - dphi;
    dphi_met_dijet = dphi;
  }

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

  double close_jet_is_lead = 0.0;
  double n_inv_close       = 0.0;
  if (n_jets >= 1) {
    close_jet_is_lead = (close_idx == lead_vis_idx) ? 1.0 : 0.0;
    n_inv_close       = (double)jet_inv_counts[close_idx];
  }

  double f_inv = 0.0;
  if (n_jets >= 1) {
    double la_eta = jet_axes[lead_vis_idx].first;
    double la_phi = jet_axes[lead_vis_idx].second;
    double inv_pt = 0.0;
    for (const auto& ip : in.invis_ptcls) {
      double deta = ip.eta - la_eta;
      double dphi = ip.phi - la_phi;
      if (dphi >  PI) dphi -= 2*PI;
      if (dphi < -PI) dphi += 2*PI;
      if (std::sqrt(deta*deta + dphi*dphi) < jetR) inv_pt += ip.pt;
    }
    f_inv = (lead_vis_pt + inv_pt > 0) ? inv_pt / (lead_vis_pt + inv_pt) : 0.0;
  }

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

      for (int ii = 0; ii < nc; ++ii)
        for (int jj = ii+1; jj < nc; ++jj)
          e2c += z_c[ii] * z_c[jj] * dRmat[ii][jj];

      for (int ii = 0; ii < nc; ++ii)
        for (int jj = ii+1; jj < nc; ++jj)
          for (int kk = jj+1; kk < nc; ++kk)
            e3c += z_c[ii] * z_c[jj] * z_c[kk]
                 * dRmat[ii][jj] * dRmat[ii][kk] * dRmat[jj][kk];

      std::vector<fastjet::PseudoJet> cpjs;
      cpjs.reserve(nc);
      for (auto& c : lead_constits)
        cpjs.emplace_back(c[0], c[1], c[2], c[3]);

      fastjet::JetDefinition kt_def(fastjet::kt_algorithm, 1.0);
      fastjet::ClusterSequence cs_sub(cpjs, kt_def);

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

      std::vector<double> ax1e, ax1p;
      if (nc >= 1) {
        auto s1 = cs_sub.exclusive_jets(1);
        ax1e = {s1[0].eta()};
        ax1p = {s1[0].phi()};
        tau1 = run_kmeans(1, ax1e, ax1p);
      }

      std::vector<double> ax2e, ax2p;
      if (nc >= 2) {
        auto s2 = cs_sub.exclusive_jets(2);

        std::vector<double> e2a = {s2[0].eta(), s2[1].eta()};
        std::vector<double> p2a = {s2[0].phi(), s2[1].phi()};
        double tau2a = run_kmeans(2, e2a, p2a);

        std::vector<double> e2b = {ax1e[0], s2[1].eta()};
        std::vector<double> p2b = {ax1p[0], s2[1].phi()};
        double tau2b = run_kmeans(2, e2b, p2b);

        if (tau2a <= tau2b) { tau2 = tau2a; ax2e = e2a; ax2p = p2a; }
        else                { tau2 = tau2b; ax2e = e2b; ax2p = p2b; }
      }

      if (nc >= 3) {
        auto s3 = cs_sub.exclusive_jets(3);

        std::vector<double> e3a = {s3[0].eta(), s3[1].eta(), s3[2].eta()};
        std::vector<double> p3a = {s3[0].phi(), s3[1].phi(), s3[2].phi()};
        double tau3a = run_kmeans(3, e3a, p3a);

        std::vector<double> e3b = {ax2e[0], ax2e[1], s3[2].eta()};
        std::vector<double> p3b = {ax2p[0], ax2p[1], s3[2].phi()};
        double tau3b = run_kmeans(3, e3b, p3b);

        tau3 = std::min(tau3a, tau3b);
      }
    }
  }

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

  out.leadVisPt = lead_vis_pt;   out.leadWidth = lead_width;   out.MET = met;
  out.maxElePt  = in.max_ele_pt; out.maxMuPt   = in.max_mu_pt;
  out.jetThrust = thrust;        out.transSphericity = spher;
  out.hemiMass1 = hemi_mass1;    out.hemiMass2 = hemi_mass2;
  out.ptBal     = pt_bal;        out.dPhiMETdijet = dphi_met_dijet;
  out.e2c = e2c; out.e3c = e3c;
  out.tau1 = tau1; out.tau2 = tau2; out.tau3 = tau3;
  out.dPhiMETclose = dphi_met_close; out.dPhiMETfar = dphi_met_far;
  out.nJets = (double)n_jets;
  out.closeJetIsLead = close_jet_is_lead; out.nInvClose = n_inv_close;
  out.metPhi = met_phi;
  out.HT = HT; out.RT = RT; out.Meff = Meff;
  out.leadJetMass = lead_jet_mass; out.nConst = n_const; out.fInv = f_inv;

  return true;
}
