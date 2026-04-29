from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
pp   = PdfPages('PLOT.pdf')
tmp1 = plt.figure(1)
tmp1.set_size_inches(8.00,6.00)
plot = open('PLOT-0.dat')
plot = [line.split() for line in plot]
valx = [float(x[0]) for x in plot]
valy = [float(x[1]) for x in plot]
plt.plot( valx, valy, '-', color='red', label=r"HV gluons")
plot = open('PLOT-1.dat')
plot = [line.split() for line in plot]
valx = [float(x[0]) for x in plot]
valy = [float(x[1]) for x in plot]
plt.plot( valx, valy, '-', color='olive', label=r"HV hadrons")
plot = open('PLOT-2.dat')
plot = [line.split() for line in plot]
valx = [float(x[0]) for x in plot]
valy = [float(x[1]) for x in plot]
plt.plot( valx, valy, '-', color='blue', label=r"muons")
plt.xlim( -5.000e-01, 4.050e+01)
plt.ylim( 0.000e+00, 1.491e+02)
plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,3))
plt.legend(frameon=False,loc='best')
plt.title(r"HV True Multiplicities")
plt.xlabel(r"$\mathrm{Multiplicity}$")
plt.ylabel(r"$N_{\mathrm{events}}$")
pp.savefig(tmp1,bbox_inches='tight')
plt.clf()
pp.close()
