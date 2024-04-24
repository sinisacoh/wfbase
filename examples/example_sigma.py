"""
Inter- and intra-band optical conductivity
==========================================

Calculation of inter-band and intra-band optical conductivity
of gold, as a function of frequency.  For demonstration purposes,
we are using here a relatively large smearing of 0.2 eV and a small
number of k-points (10^3).

In this example, we use the doublet-notation, so that bands are indexed
with pair (*n*, *N*) and not a single *n* index as usual.  Therefore index
*n* goes over 9 values, while *N* goes over 2 values.  (Without doublet
notation single index *n* would go over 9*2=18 values.)  You can find more
details in the description of parameter *doublet_indices* in the
function :func:`do_mesh <wfbase.DatabaseWf.do_mesh>`.

Our bands in this example are doubly-degenerate, due to the presence of the
combined symmetry of inversion and time-reversal.  Therefore, we need to
exclude from the inter-band calculation below all terms where (*n*, *N*)
and (*m*, *M*) correspond to the same doublet.  In other words, we need
to exclude terms for which *n* is equal to *m*.  This is achieved by the
second parameter (*conditions*) sent to the function
:func:`evaluate <wfbase._ComputatorWf.evaluate>`.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/au_fcc.wf")

comp = db.do_mesh([10, 10, 10],
                  shift_k = "random",
                  to_compute = ["A", "dEdk"],
                  doublet_indices = True)

comp.compute_photon_energy("hbaromega", 0.5, 3.5, 51)
comp.compute_hbar_velocity("hbarv")
comp.new("kbtemp", "0.05 * eV")

comp.compute_occupation("f", "E", "ef", "kbtemp")
comp.compute_occupation_derivative("dfdE", "E", "ef", "kbtemp")

comp.replace("eta", "0.2 * eV")

# Chapter 6, Eqs 25 and 26
# https://www.sciencedirect.com/science/article/abs/pii/S1572093406020063
comp.evaluate("""
sigma~intra_oij <= (j / (numk * volume)) * \
    (-dfdE_knN) * (hbarv_knNnNi * hbarv_knNnNj) / (hbaromega_o + j*eta)
sigma~inter_oij <= (j / (numk * volume)) * \
    ((f_kmM - f_knN) * hbarv_knNmMi * hbarv_kmMnNj) /((E_kmM - E_knN) * (E_kmM - E_knN - hbaromega_o - j*eta))
chi_oij <= j * (sigma~intra_oij + sigma~inter_oij) / hbaromega_o
""", "m != n")

wf.render_latex(comp.report(), "fig_sigma_latex.png")

result, result_latex = comp.compute_in_SI("chi", "e^2 / epszero")
wf.render_latex(result_latex, "fig_sigma_chi_latex.png")

eps_xx = 1.0 + result[:,0,0]

fig, ax = plt.subplots(figsize = (4.0, 3.0))
ax.plot(comp["hbaromega"], np.real(eps_xx), "k-")
ax.plot(comp["hbaromega"], np.imag(eps_xx), "k:")
ax.axhline(0.0, c = "g", lw = 0.5)
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel(r"$\epsilon^{\rm rel}_{\rm xx}$")
ax.set_title("Dielectric constant of Au (fcc)")
ax.set_ylim(-30, 20)
fig.tight_layout()
fig.savefig("fig_sigma.pdf")
