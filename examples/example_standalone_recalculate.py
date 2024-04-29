"""
Use the output of DFT calculations
==================================

In this example, we use DFT input files prepared in the previous
:ref:`example <sphx_glr_all_examples_example_standalone_prepare.py>`.
We then modified various parameters in those input files and redid the DFT
calculations (Quantum ESPRESSO and Wannier90).  The example script
below reads in the output from Quantum ESPRESSO and Wannier90 and
loads those calculations into WfBase.

If you want to redo these calculations on your own, you will need to
download the output of the Quantum ESPRESSO and Wannier90 calculations
:download:`run_dft_output.tar.gz <misc/run_dft_output.tar.gz>` and unpack
it in the same folder from which you are running this example script.

The *modified* calculation below has only 4x4x4 coarse k-mesh (instead
of 8x8x8 coarse k-mesh in the *original* calculation).  Furthermore,
the *modified* calculation has fewer empty states.

The *modified* and *original* results are quite different, which
demonstrates the importance of using dense enough k-meshes
in the DFT calculations (and/or enough empty states).

"""

# Copyright under GNU General Public License 2024
# by Sinisa Coh (see gpl-wfbase.txt)

import wfbase as wf
import pylab as plt
import wannierberri as wberri

def main():

    wf.download_data_if_needed()

    fig, ax = plt.subplots()
    for s in range(2):
        if s == 0:
            # regular calulation from wfbase database
            db = wf.load("data/fe_bcc.wf")
            label = "Original"
        elif s == 1:
            # reads in output of DFT calculations using WannierBerri
            # The following line uses WannierBerri syntax.  See here for
            # more details on how to use WannierBerri https://wannier-berri.org
            system = wberri.System_w90("run_dft_output/x", berry = True, spin = False)
            db = wf.load_from_wannierberri(system, global_fermi_level_ev = 18.3776)
            label = "Modified"
        
        comp = db.do_mesh([16, 16, 16], to_compute = ["A"])
        comp.compute_photon_energy("hbaromega", 0.0, 5.0, 51)
        comp["eta"] = 0.3
        comp.evaluate("sigma_oij <= (j / (numk * volume)) * (f_km - f_kn) * \
        Real((E_km - E_kn) / (E_km - E_kn - hbaromega_o - j*eta)) * \
        A_knmi * A_kmnj")
        result, result_latex = comp.compute_in_SI("sigma", "e^2 / hbar")
        
        ax.plot(comp["hbaromega"], result[:, 1, 0].real / 100.0, "-", label = label)

    ax.legend()
    ax.set_xlabel(r"$\hbar \omega$ (eV)")
    ax.set_ylabel(r"$\sigma_{\rm yx}$ (S/cm)")
    ax.set_title("Off-diagonal optical conductivity in Fe (bcc)")
    fig.tight_layout()
    fig.savefig("fig_standalone_recalculate.pdf")

if __name__ == "__main__":
    main()
