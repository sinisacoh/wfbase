"""
Brute-forced sums
=================

Compares calculation without brute-forced sums (default, using numpy
vectorization) and with brute-forced sums (using Numba).  To compare
the speed of the two approaches, see the following 
:ref:`example <sphx_glr_all_examples_example_timing.py>`.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

comp = db.do_mesh([8, 8, 8], shift_k = "random", to_compute = ["A"])
comp.compute_photon_energy("hbaromega", 0.0, 5.0, 51)
comp["eta"] = 0.1

# this calculation uses numpy vectorizations
comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj")

# the extra flag at the end of this function call
# ensures that now we use brute force for loops in Numba instead of numpy vectorization
comp.evaluate("alter_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj", brute_force_sums = True)
    
difference = np.max(np.abs(comp["sigma"] - comp["alter"]))
print("Difference between two approaches is: ", difference)

assert difference < 1.0E-10
