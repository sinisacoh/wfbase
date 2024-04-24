"""
Information about the database file
===================================

This example prints to the screen information about the DFT
calculation, and processing, used to generate
the database entry stored in file data/au_fcc.wf

See :ref:`database` for more details.

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf

wf.download_data_if_needed()

db = wf.load("data/au_fcc.wf")

db.info()
