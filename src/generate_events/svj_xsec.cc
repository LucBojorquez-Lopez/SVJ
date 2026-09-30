// svj_xsec.cc -- measure sigma x BR for HV Z' production as a function of mZ'.
//
// Mirrors setupPythia() in svj_regression_delphes.cc exactly for everything
// that touches the PRODUCTION cross section (beams, Z' mass/width window, the
// SM and dark branching ratios, onIfAny).  Everything after the Z' decays --
// shower, hadronisation, HV fragmentation -- is switched off, because none of
// it changes sigma and all of it is slow.
#include "Pythia8/Pythia.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>

using namespace Pythia8;

static void rs(Pythia& p, const char* key, double v) {
  std::ostringstream oss; oss << key << " = " << v;
  p.readString(oss.str());
}

int main(int argc, char* argv[]) {
  if (argc < 2) { std::cerr << "usage: svj_xsec <mZ> [nEvent] [smBR] [partonLevel]\n"; return 1; }
  double mZ      = std::atof(argv[1]);
  int    nEvent  = (argc > 2) ? std::atoi(argv[2]) : 2000;
  // Per-flavour SM branching ratio.  The repo's signal uses 1e-4.
  double smBR    = (argc > 3) ? std::atof(argv[3]) : 1e-4;
  bool   doPS    = (argc > 4) ? (std::atoi(argv[4]) != 0) : false;
  double darkBR  = 1.0 - 6.0 * smBR;

  Pythia pythia;
  pythia.readString("Print:quiet = on");
  pythia.readString("Beams:eCM = 14000.");
  pythia.readString("HiddenValley:ffbar2Zv = on");

  rs(pythia, "4900023:m0",     mZ);
  rs(pythia, "4900023:mMin",   mZ - 0.05 * mZ);
  rs(pythia, "4900023:mMax",   mZ + 0.05 * mZ);
  rs(pythia, "4900023:mWidth", 0.025 * mZ);
  pythia.readString("4900023:doForceWidth = on");

  { std::ostringstream o; o << "4900023:oneChannel = 1 " << darkBR
                            << " 102 4900101 -4900101"; pythia.readString(o.str()); }
  for (int f = 1; f <= 6; ++f) {
    std::ostringstream o; o << "4900023:addChannel = 1 " << smBR
                            << " 102 " << f << " -" << f; pythia.readString(o.str());
  }
  pythia.readString("4900023:onMode = off");
  pythia.readString("4900023:onIfAny = 4900101 4900102");

  pythia.readString("HiddenValley:Ngauge = 3");
  pythia.readString("HiddenValley:nFlav = 2");
  pythia.readString("HiddenValley:spinFv = 0");
  rs(pythia, "4900101:m0", 2.0);
  rs(pythia, "4900102:m0", 2.0);

  if (!doPS) {
    pythia.readString("PartonLevel:all = off");   // sigma is fixed at process level
    pythia.readString("HadronLevel:all = off");
  }
  pythia.readString("Random:setSeed = on");
  pythia.readString("Random:seed = 12345");

  if (!pythia.init()) { std::cerr << "init failed for mZ=" << mZ << "\n"; return 2; }
  for (int i = 0; i < nEvent; ++i) pythia.next();

  double sigma_mb  = pythia.info.sigmaGen();     // PYTHIA reports mb
  double sigma_pb  = sigma_mb * 1.0e9;
  double err_pb    = pythia.info.sigmaErr() * 1.0e9;
  std::cout << std::scientific << std::setprecision(6)
            << mZ << "\t" << sigma_pb << "\t" << err_pb << "\t"
            << smBR << "\t" << (doPS ? 1 : 0) << "\n";
  return 0;
}
