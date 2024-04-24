"""
Anomalous Hall conductivity, energy
===================================

Here we computed sigma resolved by k-point and pair of band indices
(n and m).  This allows us then to compute the contributions of various
electron-hole pairs to the sigma.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

all_x = []
all_y = []
for i in range(10):
    comp = db.do_mesh([16, 16, 16], to_compute = ["A"], shift_k = "random")

    comp.replace("hbaromega", {"value":[0.5, 1.0, 1.5, 2.0], "units":wf.Units(eV = 1)})
    
    comp["eta"] = 0.3
    comp.new("Ax", comp["A"][:,:,:,0], "A")
    comp.new("Ay", comp["A"][:,:,:,1], "A")

    comp.evaluate("sigma_oknm <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               Ax_knm * Ay_kmn")
    result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")
    result = result.real / 100.0

    comp.evaluate("dife_knm <= E_km - E_kn")

    all_x.append(np.real(comp["dife"]))
    all_y.append(np.real(result))
    
fig, axs = plt.subplots(1, 4, figsize = (4*4.0, 4.0))
for i in range(4):
    ax = axs[i]
    x = np.ravel(np.array(all_x)[:,:,:])
    y = np.ravel(np.array(all_y)[:,i,:,:,:])
    h, hxedg, hyedg = np.histogram2d(x, y, bins = 101, weights = y, range = [[-1.0, 3.0], [-2.5, 2.5]])
    col_range = np.max(np.abs(h))
    ax.pcolormesh(hxedg, hyedg, h.T, vmin = (-1.0)*col_range, vmax = col_range, cmap = "bwr")
    ax.set_xlim(hxedg[0], hxedg[-1])
    ax.set_ylim(hyedg[0], hyedg[-1])
    ax.set_title("$\hbar \omega$ = " + str(comp["hbaromega"][i]) + " eV")
    ax.set_xlabel(r"$E_{km} - E_{kn}$ (eV)")
    if i == 0:
        ax.set_ylabel(r"$\sigma_{\rm xy}$ contribution")
fig.tight_layout()
fig.savefig("fig_ahc_analysis_energy.pdf")
