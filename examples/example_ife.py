"""
Spin part of inverse Faraday effect
===================================

Calculation of the spin part of the inverse Faraday effect.
Again, here for demonstration purposes, we are using a rather
small k-point sampling and large smearing *eta*.

More details about the spin part of the inverse Faraday
effect can be found in this paper,

https://doi.org/10.1103/PhysRevB.107.214432
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/au_fcc.wf")

comp = db.do_mesh([10, 10, 10], to_compute = ["A", "S"])

# increase smearing
comp["eta"] = 0.2

comp.compute_photon_energy("hbaromega", 1.0, 3.0, 31)
comp.compute_optical_offdiagonal_polarization("L", "hbaromega", "x - i y")

comp.evaluate("""
I~elea_as <= (1/numk) *     L_knma *  S_kmrs * #L_knra / ((E_km - E_kn - hbaromega_a -  j*eta)*(E_kr - E_kn - hbaromega_a + j*eta))
I~b_as    <= (1/numk) *    #L_krna *  S_kmrs *  L_krna / ((E_km - E_kn + hbaromega_a -  j*eta)*(E_kr - E_kn + hbaromega_a + j*eta))
I~c_as    <= (1/numk) * 2 * S_knms *  L_kmra * #L_knra / ((E_km - E_kn               + 2j*eta)*(E_kr - E_kn - hbaromega_a + j*eta))
I~d_as    <= (1/numk) * 2 * S_knms * #L_krma *  L_krna / ((E_km - E_kn               + 2j*eta)*(E_kr - E_kn + hbaromega_a + j*eta))
""", "E_kn < ef, E_km > ef, E_kr > ef")

comp.evaluate("""
I~hole_as <= (-1) * (1/numk) *     L_knma * #L_koma *  S_kons / ((E_km - E_kn - hbaromega_a -  j*eta)*(E_km - E_ko - hbaromega_a + j*eta))
I~f_as    <= (-1) * (1/numk) *    #L_kmna *  L_kmoa *  S_kons / ((E_km - E_kn + hbaromega_a -  j*eta)*(E_km - E_ko + hbaromega_a + j*eta))
I~g_as    <= (-1) * (1/numk) * 2 * S_knms * #L_koma *  L_kona / ((E_km - E_kn               + 2j*eta)*(E_km - E_ko - hbaromega_a + j*eta))
I~h_as    <= (-1) * (1/numk) * 2 * S_knms *  L_kmoa * #L_knoa / ((E_km - E_kn               + 2j*eta)*(E_km - E_ko + hbaromega_a + j*eta))
I_as <= Real(I~elea_as + I~b_as + I~c_as + I~d_as + I~hole_as + I~f_as + I~g_as + I~h_as)
""", "E_kn < ef , E_ko < ef, E_km > ef")

wf.render_latex(comp.report(), "fig_ife_latex.png")

print("Units of ife are: ", comp.get_units("I"))

fig, ax = plt.subplots(figsize = (6.0, 4.5))
ax.plot(comp["hbaromega"], comp["I"][:, 2].real, "k-")
ax.set_ylim(0.0)
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel("IFE coefficient " + r"($\mu_{\rm B}$ / atom)" )
ax.set_title("Inverse Faraday effect due to spin in Au (fcc)")
fig.tight_layout()
fig.savefig("fig_ife.pdf")

fig, ax = plt.subplots(figsize = (6.0, 4.5))
for x in ["I~elea", "I~b", "I~c", "I~d", "I~hole", "I~f", "I~g", "I~h"]:
    ax.plot(comp["hbaromega"], np.log10(np.abs(comp[x][:, 2].real)), "-", label = x.split("~")[1])
ax.legend()
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel("log (IFE coefficient) " + r"($\mu_{\rm B}$ / atom)" )
ax.set_title("Various contributions")
fig.tight_layout()
fig.savefig("fig_ife_contrib.pdf")
