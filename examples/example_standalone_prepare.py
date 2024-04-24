"""
Prepare input for DFT calculations
==================================

This example creates input files for DFT calculations (Quantum ESPRESSO and Wannier90).
Script *go* in the *run_dft* folder, created by the example script below, gives
you information on how to run these DFT calculations.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

db.create_dft_input_files("run_dft")

print("""You should now take input files in run_dft folder and run
this with your installation of Quantum ESPRESSO and Wannier90. See
script run_dft/go for more details on how to do this.""")
