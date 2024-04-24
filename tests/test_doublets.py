import numpy as np
import sys
sys.path.append('../')
import wfbase as wf

import pytest

from common import *

def test_that_doublets_behave_well(threshold):

    rrr = 2.31
    
    db = wf.load('au_fcc.wf')

    vals = []
    
    for cs in [False, True]:
    
        comp = db.do_mesh([4, 3, 5], shift_k = [0.002837,0.003431,0.0012415], doublet_indices = cs)
        comp.compute_kronecker("d", "E", 1)
        comp.compute_optical_offdiagonal("W", "hbaromega")
        comp.compute_optical_offdiagonal_polarization("Q", "hbaromega", "x+iy")
        comp.compute_hbar_velocity("V")
        comp.compute_occupation("h", "W", 0.3)
        comp.compute_photon_energy("hbaralpha", emin = 0.5, emax = 3.0, steps = 31)

        if cs == True:
            comp["d"] = comp["d"]*2 + 3 * rrr
            
        use_str = []
        for k in comp.all_quantities():
            if comp[k].dtype.type is np.str_:
                continue

            #print(k, np.sum(comp[k]), np.prod(comp.get_shape(k)))
            
            one_str = "( " + k
            if "indices_info" in comp.all_quantity_keys(k):
                one_str += "_" + comp.get(k, "indices_info")["canonical_names"]

                #print(k, comp.get(k, "indices_info")["canonical_names"], comp[k].shape)
            
            one_str += " + " + str(rrr) + " "
        
            units = str(comp.get(k, "units"))
            
            if units != "1":
                units = units.split()
                units = " * ".join(units)
                one_str += " * " + units
            
            one_str += ")"
                    
            use_str.append(one_str)

        use_str  = "total_ki <= " + " * ".join(use_str)
            
        comp.evaluate(use_str)
        vals.append(comp.get("total", "value"))

    for i in range(1, len(vals)):
        assert (abs((vals[i] - vals[0]) / np.max(np.abs(vals[0]))) < threshold).all()
