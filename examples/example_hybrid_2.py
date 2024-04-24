"""
Alternative, equivalent, calculation  (WfBase and numpy routines)
=================================================================

Calculation of the quantity by directly using both regular numpy
routines and WfBase.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt
from opt_einsum import contract as opteinsum

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")
comp = db.do_mesh([16, 16, 16], to_compute = ["A"])

# computation using wfbase's evaluate function
comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj")

# user could instead perform a part of the calculation using regular numpy arrays
# and then add those to the computator above and then run wfbase evaluate.
# As a demonstration, let us compute one of the intermediate objects from
# the equation above:
f = comp["f"]
deltaf = f[:, :, None] - f[:, None, :]
# now we can put this variable back into the computator
comp.new("deltaf", deltaf)
# and now we can use this new quantity in the computation below
comp.evaluate("alternative_oij <= (j / (numk * volume)) * deltaf_kmn * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj")

difference = np.max(np.abs(comp["sigma"] - comp["alternative"]))
print("Difference between two approaches is: ", difference)

assert difference < 1.0E-10
