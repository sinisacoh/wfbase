"""
Alternative, equivalent, calculation  (brute-forced sums using numba)
=====================================================================

When you call :func:`evaluate <wfbase._ComputatorWf.evaluate>` and set parameter *brute_force_sums*
to *True*, then WfBase under the hood will dynamically construct a Numba (not numpy)
python code based on your input to the evaluate(...) function call.  You can
access this Numba python code directly, and even run it on your own,
modify it if you want, and so on.

The example below shows you how to access this Numba python code
and how to run it.

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
               A_knmi * A_kmnj", brute_force_sums = True)

# the following line will display the dynamically generated
# Numba python code used to evaluate the expression above
comp.info("sigma", show_code = True)

# the output of the info command above is given
# by the function evaluate_directly below
def evaluate_directly(__object):
    from numba import njit
    import numpy as np

    __object_numk = __object["numk"]
    __object_volume = __object["volume"]
    __object_f = __object["f"]
    __size_k = __object.get_shape("f")[0]
    __size_m = __object.get_shape("f")[1]
    __size_n = __object.get_shape("f")[1]
    __object_E = __object["E"]
    __object_hbaromega = __object["hbaromega"]
    __size_o = __object.get_shape("hbaromega")[0]
    __object_eta = __object["eta"]
    __object_A = __object["A"]
    __size_i = __object.get_shape("A")[3]
    __size_j = __object.get_shape("A")[3]

    @njit
    def _tmp_func__000(__value):
        for k in range(__size_k):
            for n in range(__size_n):
                for m in range(__size_m):
                    for o in range(__size_o):
                        for i in range(__size_i):
                            for j in range(__size_j):
                                __value[o,i,j] += (((1.0j) / (__object_numk * __object_volume)) * (__object_f[k,m] - \
                                                  __object_f[k,n]) * (((__object_E[k,m] - __object_E[k,n]) / (__object_E[k,m] - \
                                                  __object_E[k,n] - __object_hbaromega[o] - ((1.0j) * __object_eta)))).real * \
                                                  __object_A[k,n,m,i] * __object_A[k,m,n,j])
    __value = np.zeros((__size_o,__size_i,__size_j), dtype = complex)
    _tmp_func__000(__value)
    return __value

# now call the function we got printed using info
sigma_alt = evaluate_directly(comp)

difference = np.max(np.abs(comp["sigma"] - sigma_alt))
print("Difference between two approaches is: ", difference)

assert difference < 1.0E-10

