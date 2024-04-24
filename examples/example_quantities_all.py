"""
Quantities
==========

Placeholder.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf

db = wf.load("data/au_fcc.wf")

# create computator with all basic quantities
comp = db.do_mesh(to_compute = ["psi", "A", "S", "dEdk"])

# get some additional quantities
comp.compute_occupation("f")
comp.compute_occupation_derivative("dfdE")
comp.compute_kronecker("d", "E", 1)
comp.compute_identity("one", 3)
comp.compute_photon_energy("hbaromega")
comp.compute_orbital_character("O")
comp.compute_optical_offdiagonal("L", "hbaromega")
comp.compute_optical_offdiagonal_polarization("R", "hbaromega", "x + i y")
comp.compute_hbar_velocity("hbarv")

# evaluate some user-specified quantity
comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
         Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
         A_knmi * A_kmnj")

# now get all the information about it
comp.info(print_to_screen = False, show_code = True)



   
