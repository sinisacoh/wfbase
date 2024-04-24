"""
Information about the quantities
================================

This example prints to the screen information about the
quantities stored in the computator object *comp*, as
well as information about the newly computed quantity
*sigma*.

See :ref:`quantities` for more details.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf

wf.download_data_if_needed()

# loads the database file
db = wf.load("data/au_fcc.wf")

# creates a computator on a 8x8x8 k-mesh of points 
comp = db.do_mesh([8, 8, 8])

# prints information about all quantities currently in comp
comp.info()

# this computes new quantity called "sigma"
comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
               Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
               A_knmi * A_kmnj")

# prints information about newly calculated quantity "sigma"
comp.info("sigma", show_code = True)
