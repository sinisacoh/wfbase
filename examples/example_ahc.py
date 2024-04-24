"""
Anomalous Hall conductivity
===========================

Calculation of an off-diagonal optical conductivity of iron
as a function of frequency.  Zero frequency limit corresponds
to the anomalous Hall conductivity.  For demonstration purposes,
this calculation uses a rather large smearing *eta* parameter
and a relatively small k-point mesh.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

# show information about the database file
db.info()

# interpolate quantities to a 16^3 k-mesh
comp = db.do_mesh([16, 16, 16], to_compute = ["A"])

# change the default value of photon energy
comp.compute_photon_energy("hbaromega", 0.0, 5.0, 51)

# change the default value of constant eta
comp["eta"] = 0.3

# this is equation 12.11 for sigma, as written in
# the user_guide.pdf of Wannier90 https://wannier.org/support/
comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj")

print("Units of sigma are: ", comp.get_units("sigma"))

# prints to the screen information about quantity sigma evaluated above
comp.info("sigma", show_code = True)

# parse equation into a LaTeX and plot it to a file
wf.render_latex(comp.get_latex("sigma"), "fig_ahc_latex.png")

# the following line will work in iTerm2 and similar terminals
# that support imgcat
wf.display_in_terminal("fig_ahc_latex.png")
# the following line should work in any terminal
#wf.display_in_separate_window("fig_ahc_latex.png")

# compute in SI and multiply with e^2/hbar
result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")

# plot latex'ed information about the sigma converted to SI units
#wf.render_latex(result_latex, "fig_ahc_si_latex.png")

fig, ax = plt.subplots(figsize = (4.0, 3.0))
ax.plot(comp["hbaromega"], result[:, 1, 0].real / 100.0, "k-")
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel(r"$\sigma_{\rm yx}$ (S/cm)")
ax.set_title("Off-diagonal optical conductivity in Fe (bcc)")
fig.tight_layout()
fig.savefig("fig_ahc.pdf")
