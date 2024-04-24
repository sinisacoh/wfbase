"""
Inter- and intra-band optical conductivity (all metals)
=======================================================

The same calculation of inter-band and intra-band optical conductivity,
this time done for all transition metals.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

cases = """sc ti v  cr mn fe co ni cu zn
           y  zr nb mo tc ru rh pd ag cd
           XX hf ta w  re os ir pt au hg"""

wf.download_data_if_needed()

fig, axs = plt.subplots(3, 10, figsize = (20, 6))
row = 0
for line in cases.split("\n"):
    column = 0
    for ll in line.split(" "):
        if ll == "":
            continue
        if ll == "XX":
            axs[row][column].remove()
            column += 1
            continue
        
        db = wf.load("data/" + ll + "_bcc.wf")

        if ll in ["fe", "co", "ni"]:
            comp = db.do_mesh([10, 10, 10], shift_k = "random", to_compute = ["A", "dEdk"])
        else:
            comp = db.do_mesh([10, 10, 10], shift_k = "random", to_compute = ["A", "dEdk"], doublet_indices = True)
        comp.compute_photon_energy("hbaromega", 0.5, 3.5, 51)
        comp.compute_hbar_velocity("hbarv")
        comp.new("kbtemp", "0.05 * eV")
        comp.compute_occupation("f", "E", "ef", "kbtemp")
        comp.compute_occupation_derivative("dfdE", "E", "ef", "kbtemp")
        comp.replace("eta", "0.2 * eV")
        ev_str = """
        sigma~intra_oij <= (j / (numk * volume)) * (-dfdE_knN) * (hbarv_knNnNi * hbarv_knNnNj) / (hbaromega_o + j*eta)
        sigma~inter_oij <= (j / (numk * volume)) * ((f_kmM - f_knN) * hbarv_knNmMi * hbarv_kmMnNj) /((E_kmM - E_knN) * (E_kmM - E_knN - hbaromega_o - j*eta))
        chi_oij <= j * (sigma~intra_oij + sigma~inter_oij) / hbaromega_o
        """
        if ll in ["fe", "co", "ni"]:
            ev_str = ev_str.replace("M", "").replace("N", "")
        comp.evaluate(ev_str, "m != n")
        result, result_latex = comp.compute_in_SI("chi", "e^2 / epszero")
        eps_xx = 1.0 + result[:,0,0]
        
        ax = axs[row][column]
        ax.plot(comp["hbaromega"], np.real(eps_xx), "k-")
        ax.plot(comp["hbaromega"], np.imag(eps_xx), "k:")
        ax.axhline(0.0, c = "g", lw = 0.5)
        if row == 2:
            ax.set_xlabel(r"$\hbar \omega$ (eV)")
        if column == 0:
            ax.set_ylabel(r"$\epsilon^{\rm rel}_{\rm xx}$")
        ax.set_ylim(-40, 40)
        ax.set_title(ll.title() + " (bcc)")
        column += 1
    row += 1
fig.tight_layout()
fig.savefig("fig_sigma_all.pdf")
    
