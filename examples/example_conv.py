"""
Simple k-mesh convergence
=========================

Demonstration of converging calculation with respect to the number
of k-points in the mesh. As shown in the previous example
:ref:`sphx_glr_all_examples_example_timing.py`, instead of running
a single calculation on a large k-mesh (which takes a lot of memory,
as all quantities need to be computed on this large k-mesh), the
calculation is faster if it is split into many smaller calculations,
each with a randomly shifted grid.  (If you prefer to do a calculation
with a single shot, without splitting it up into smaller randomly shifted
grids, then try using :func:`evaluate <wfbase._ComputatorWf.evaluate>` with parameter
*brute_force_sums* set to *True*.  To see how to use this example, take
a look at example :ref:`sphx_glr_all_examples_example_brute_force.py`.)

The calculation proceeds until the value of optical conductivity does
not change too much upon the addition of more k-points.

With the smearing *eta* set to 0.1 eV, the calculation below needs
71 steps to converge to the relative error of around 0.5 percent.
This means that in total one needs 71*8^3 k-points, which is around
36,000 k-points.  At smaller smearing one will need to use even more
k-points.

While DFT calculations used to create this database are reasonably
well-converged, in principle there is no apriori way of knowing whether
the physical quantity you are computing is well-converged with respect
to these parameters.  Therefore, in principle, one would need to
check to make sure one has a well-converged calculation with respect
to: the energy cut used in the plane-wave DFT calculation
used in the creation of the database file data/fe_bcc.wf, smearing parameter,
number of k-points in the coarse and fine meshes, frozen and disentanglement
windows used in the Wannier90 calculation, structural relaxation,
dependence on the exchange-correlation functional, and so on.

However, to change the underlying DFT parameters, you will have to redo
the DFT calculations themselves, as these are currently not stored in
the database.  See the following two examples of how to perform these
calculations.  First, example :ref:`sphx_glr_all_examples_example_standalone_prepare.py`
will show you how to prepare input files for Quantum ESPRESSO and Wannier90.
Second, example :ref:`sphx_glr_all_examples_example_standalone_recalculate.py`
will show you how to perform WfBase calculations once you get output
from Quantum ESPRESSO and Wannier90.
"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import numpy as np
import pylab as plt

wf.download_data_if_needed()

db = wf.load("data/fe_bcc.wf")

max_steps = 200

all_values = []
for i in range(max_steps):

    # interpolate quantities to a k-mesh with random shift of k-points
    comp = db.do_mesh([8, 8, 8], shift_k = "random", to_compute = ["A"])

    comp.compute_photon_energy("hbaromega", 0.0, 5.0, 51)
    comp["eta"] = 0.1

    comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
                   Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
                   A_knmi * A_kmnj")
    
    result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")
    all_values.append(result[:, 1, 0].real / 100.0)

    # check if convergence has been achieved
    if len(all_values) > 3:
        res0 = np.mean(all_values[:  ], axis = 0)
        res1 = np.mean(all_values[:-3], axis = 0)
        rdif = np.max(np.abs(res0 - res1))/np.max(np.abs(res0))
        if rdif < 5.0E-3:
            break

    if i == max_steps - 1:
        print("Convergence not achieved in " + str(max_steps) + " steps.")
        exit()
        
result = np.mean(all_values, axis = 0)
            
fig, ax = plt.subplots()
ax.plot(comp["hbaromega"], result, "k-", lw = 1.5)
for i in range(len(all_values)):
    average_up_to_i = np.mean(all_values[:i + 1], axis = 0)
    ax.plot(comp["hbaromega"], average_up_to_i, "g-", lw = 0.5, zorder = -1000)
ax.set_xlabel(r"$\hbar \omega$ (eV)")
ax.set_ylabel(r"$\sigma_{\rm yx}$ (S/cm)")
ax.set_title("Off-diagonal optical conductivity in Fe (bcc)")
fig.tight_layout()
fig.savefig("fig_conv.pdf")
    
