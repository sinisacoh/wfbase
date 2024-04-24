"""
Anomalous Hall conductivity, orbital
====================================

Here we computed sigma resolved by orbital decomposition of electron
and hole states. This allows us then to compute the contributions of various
orbital characters to the sigma.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

all_mat = []

for i in range(5):
    comp = db.do_mesh([16, 16, 16], to_compute = ["psi", "A"], shift_k = "random", reorder_orbitals = True)

    comp.compute_orbital_character("O")

    comp.replace("hbaromega", {"value":[0.5, 1.0, 1.5, 2.0], "units":wf.Units(eV = 1)})
    
    comp["eta"] = 0.3
    comp.new("Ax", comp["A"][:,:,:,0], "A")
    comp.new("Ay", comp["A"][:,:,:,1], "A")

    comp.evaluate("sigma_opr <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               Ax_knm * Ay_kmn * O_knp * O_kmr")

    result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")

    result = result.real / 100.0
    
    all_mat.append(np.real(result))
    
all_mat = np.mean(np.array(all_mat), axis = 0)

fig, axs = plt.subplots(1, 4, figsize = (4*5.0, 5.2))
for i in range(4):
    mat = all_mat[i]
    orb_lab = list(map(lambda s: "$" + s + "$", comp["orbitallabels"]))

    maxval = np.max(np.abs(mat))
    axs[i].matshow(mat, vmin = -maxval, vmax = maxval, cmap = "seismic")
    axs[i].set_xticks(list(range(mat.shape[0])))
    axs[i].set_xticklabels(orb_lab, rotation = 45)
    axs[i].set_yticks(list(range(mat.shape[1])))
    axs[i].set_yticklabels(orb_lab)
    axs[i].set_aspect("equal")
    axs[i].set_title("$\hbar \omega$ = " + str(comp["hbaromega"][i]) + " eV")

fig.tight_layout()
fig.savefig("fig_ahc_analysis_orbital.pdf")
