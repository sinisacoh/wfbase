"""
Alternative, equivalent, calculation  (only using numpy routines)
=================================================================

All quantities in WfBase are simply numpy arrays, and one can
directly do the computation with the numpy arrays, without
having to use the evaluate routine from WfBase.  This approach
is demonstrated in the example below.  The approach
from WfBase is more compact and easier to debug and analyze.
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

# the user can achieve the same thing by directly
# accessing quantities and using standard numpy operations.
#
# these are simply numpy arrays...
f = comp["f"]
E = comp["E"]
ho = comp["hbaromega"]
A = comp["A"]
# and now we can build up the same expression as above
dif_f = f[:, :, None] - f[:, None, :]
dif_E = E[:, :, None] - E[:, None, :]
dif_E_denom = dif_E[None, :, :, :] - ho[:, None, None, None] - 1.0j * comp["eta"]
fraction = np.real(dif_E[None, :, :, :] / dif_E_denom[:, :, :, :])
sigma = opteinsum("kmn, okmn, knmi, kmnj -> oij", dif_f, fraction, A, A)
sigma = sigma * 1.0j / (comp["numk"] * comp["volume"])

difference = np.max(np.abs(comp["sigma"] - sigma))
print("Difference between two approaches is: ", difference)

assert difference < 1.0E-10
