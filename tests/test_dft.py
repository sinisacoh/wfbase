import numpy as np
import sys
sys.path.append('../')
import wfbase as wf

import pytest

from common import *

def test_on_dft(threshold):
    
    db = wf.load('au_fcc.wf')

    comp = db.do_mesh([3, 3 , 3], [0.002837,0.003431,0.0012415], ["A", "S"])
    nbnd = comp["numwann"]    
    numfreq = 11
    comp.compute_photon_energy("omega", 1.0, 3.0, numfreq)
    comp.compute_optical_offdiagonal("L", "omega")
    # reshape things a bit
    comp["E"] = comp["E"].reshape((-1, nbnd//2, 2))
    comp["A"] = comp["A"].reshape((-1, nbnd//2, 2, nbnd//2, 2, 3))
    comp["L"] = comp["L"].reshape((-1, nbnd//2, 2, nbnd//2, 2, 3, numfreq))
    comp["S"] = comp["S"][:,:,:,2].reshape((-1, nbnd//2, 2, nbnd//2, 2))
    # compute optical matrix element
    comp.new("P", {"value": comp["L"][:,:,:,:,:,0,:] + 1.0j*comp["L"][:,:,:,:,:,1,:],
                            "units": comp.get("L", "units")})
    comp.compute_occupation("f", "E", "ef")
    comp.replace("eta", {"value": 0.1, "units": wf.Units(eV = 1, Ang = 0, muB = 0)})
    comp.compute_kronecker("d", "E", 1)

    use_str = """
    elec_a <=                         (1/numk) *     P_knNmMa * S_kmMmO * #P_knNmOa * f_knN * (1.0 - f_kmM) * (1.0 - f_kmO)  / ((E_kmM - E_knN - omega_a)^2 + eta^2)
    hole_a <=  (-1.0) *               (1/numk) *     P_knNmMa * #P_knOmMa * S_knOnN * f_knN * f_knO         * (1.0 - f_kmM)  / ((E_kmM - E_knN - omega_a)^2 + eta^2)
    
    ndra_a <=            (1 - d_mo) * (1/numk) *     P_knNmMa * S_kmMoO * #P_knNoOa * f_knN * (1.0 - f_kmM) * (1.0 - f_koO)  / ((E_kmM - E_knN - omega_a - 1.0j*eta)*(E_koO - E_knN - omega_a + 1.0j*eta))
    ndrb_a <=                         (1/numk) *     #P_koOnNa * S_kmMoO * P_koOnNa * f_knN * (1.0 - f_kmM) * (1.0 - f_koO)  / ((E_kmM - E_knN + omega_a - 1.0j*eta)*(E_koO - E_knN + omega_a + 1.0j*eta))
    ndrc_a <=                         (1/numk) * 2 * S_knNmM * P_kmMoOa * #P_knNoOa * f_knN * (1.0 - f_kmM) * (1.0 - f_koO)  / ((E_kmM - E_knN + 2.0j*eta)*(E_koO - E_knN - omega_a + 1.0j*eta))
    ndrd_a <=                         (1/numk) * 2 * S_knNmM * #P_koOmMa * P_koOnNa * f_knN * (1.0 - f_kmM) * (1.0 - f_koO)  / ((E_kmM - E_knN + 2.0j*eta)*(E_koO - E_knN + omega_a + 1.0j*eta))

    ndre_a <=  (-1.0) *  (1 - d_no) * (1/numk) *     P_knNmMa * #P_koOmMa * S_koOnN * f_knN * (1.0 - f_kmM) * f_koO          / ((E_kmM - E_knN - omega_a - 1.0j*eta)*(E_kmM - E_koO - omega_a + 1.0j*eta))
    ndrf_a <=  (-1.0) *               (1/numk) *     #P_kmMnNa * P_kmMoOa * S_koOnN * f_knN * (1.0 - f_kmM) * f_koO          / ((E_kmM - E_knN + omega_a - 1.0j*eta)*(E_kmM - E_koO + omega_a + 1.0j*eta))
    ndrg_a <=  (-1.0) *               (1/numk) * 2 * S_knNmM * #P_koOmMa * P_koOnNa * f_knN * (1.0 - f_kmM) * f_koO          / ((E_kmM - E_knN + 2.0j*eta)*(E_kmM - E_koO - omega_a + 1.0j*eta))
    ndrh_a <=  (-1.0) *               (1/numk) * 2 * S_knNmM * P_kmMoOa * #P_knNoOa * f_knN * (1.0 - f_kmM) * f_koO          / ((E_kmM - E_knN + 2.0j*eta)*(E_kmM - E_koO + omega_a + 1.0j*eta))

    ife_a  <= elec_a + hole_a + ndra_a + ndrb_a + ndrc_a + ndrd_a + ndre_a + ndrf_a + ndrg_a + ndrh_a 
    """

    five_evaluate(comp, use_str, ["ife", "ndrf"])
    
