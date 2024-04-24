"""
Alternative, equivalent, calculation  (only using numpy routines)
=================================================================

WfBase under the hood dynamically constructs a numpy python code
based on your input to the evaluate(...) function call.  You can
access this numpy python code directly, and even run it on your own,
modify it if you want, and so on.

The example below shows you how to access this numpy python code
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
               A_knmi * A_kmnj")

# the following line will display the dynamically generated
# numpy python code used to evaluate the expression above
comp.info("sigma", show_code = True)

# the output of the info command above is given
# by the function evaluate_directly below
def evaluate_directly(__object):
    import numpy as np
    from opt_einsum import contract as opteinsum

    _orig_shp = {}
    _f = []
    _s = {}
    __brod00 = np.copy(__object["f"])
    __brod00 = __brod00[:,:,None] - (__object["f"])[:,None,:]
    __brod01 = np.copy(__object["E"])
    __brod01 = __brod01[:,:,None] - (__object["E"])[:,None,:]
    __brod02 = np.array(complex(1.0j))
    __brod02 = __brod02 * (__object["eta"])
    __brod03 = np.copy(__object["E"])
    __brod03 = __brod03[:,:,None,None] - (__object["E"])[:,None,:,None]
    __brod03 = __brod03.squeeze(axis = (3,))
    __brod03 = __brod03[:,:,:,None] - (__object["hbaromega"])[None,None,None,:]
    __brod03 = __brod03 - (__brod02)
    __brod04 = np.copy(__brod01)
    __brod04 = __brod04[:,:,:,None] * ((1.0/(__brod03)))[:,:,:,:]
    __mult00 = opteinsum(",,,kmn,kmno,knmi,kmnj->oij",\
                         complex(1.0j),\
                         (1.0/(__object["numk"])),\
                         (1.0/(__object["volume"])),\
                         __brod00,\
                         np.real(__brod04),\
                         __object["A"],\
                         __object["A"])
    __value = __mult00
    return __value


# now call the function we got printed using info
sigma_alt = evaluate_directly(comp)

difference = np.max(np.abs(comp["sigma"] - sigma_alt))
print("Difference between two approaches is: ", difference)

assert difference < 1.0E-10

