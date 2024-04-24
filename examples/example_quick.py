"""
Example source code
===================

Here is the complete source code of the calculation described above.
Many more examples can be found :ref:`here <examples>`.
"""

import wfbase as wf

wf.download_data_if_needed()

# loads the database
db = wf.load("data/au_fcc.wf")

# calculates needed quantities on a grid of k-points
comp = db.do_mesh([16, 16, 16])

# changes the default values of some of the parameters
comp["eta"] = 0.2
comp.compute_photon_energy("hbaromega", 1.0, 3.0, 51)

# evaluates interband optical conductivity
comp.evaluate(
    "sigma_oij <= (j/(numk*volume)) * A_knmi*A_kmnj * (E_kn - E_km)/(E_km - E_kn - hbaromega_o - j*eta)",
    "E_km > ef, E_kn < ef"
)

# renders the string above in LaTeX
wf.render_latex(comp.get_latex("sigma"), "fig_quick_latex.png")

# converts to SI units and multiplies with e^2/hbar
result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (4.0, 3.0))
ax.plot(comp["hbaromega"], result[:, 0, 0].real * 1.0E-5, "k-")
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel(r"$\sigma_{\rm xx}$ $\left(\displaystyle\frac{10^5}{\Omega {\rm m}}\right)$")
ax.set_ylim(0.0)
fig.tight_layout()
fig.savefig("fig_quick.pdf")
