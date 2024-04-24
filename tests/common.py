import numpy as np
import sys
sys.path.append('../')
import re
from copy import deepcopy

from wfbase import _ComputatorWf, Units

import pytest

RAND = np.random.RandomState(0)

def rnd_matrix(sz, seed = None):
    MIN, MAX = -1.0, 1.0
    if seed is None:
        rand_use = RAND
    else:
        rand_use = np.random.RandomState(seed)
    return rand_use.uniform(MIN, MAX, size=sz) + 1j*rand_use.uniform(MIN, MAX, size=sz)
    
@pytest.fixture
def comp_test():
    return direct_comp_test()

def direct_comp_test(seed = None):
    quantities = {
        'A'  : {"value": 2.4312 - 1.1231j,                  "units": Units(eV = 0, Ang = 0, muB = 0)},
        'B'  : {"value": rnd_matrix((3), seed),             "units": Units(eV = 0, Ang = 0, muB = 0)},
        'BB' : {"value": rnd_matrix((5), seed),             "units": Units(eV = 0, Ang = 0, muB = 0)},
        'C'  : {"value": rnd_matrix((3, 5), seed),          "units": Units(eV = 0, Ang = 0, muB = 0)},
        'CC' : {"value": rnd_matrix((5, 5), seed),          "units": Units(eV = 0, Ang = 0, muB = 0)},
        'D'  : {"value": rnd_matrix((4, 2, 5), seed),       "units": Units(eV = 0, Ang = 0, muB = 0)},
        'DD' : {"value": rnd_matrix((4, 4, 5), seed),       "units": Units(eV = 0, Ang = 0, muB = 0)},
        'E'  : {"value": rnd_matrix((7, 5, 3, 6), seed),    "units": Units(eV = 0, Ang = 0, muB = 0)},
        'EE' : {"value": rnd_matrix((7, 5, 5, 7), seed),    "units": Units(eV = 0, Ang = 0, muB = 0)},
        'F'  : {"value": rnd_matrix((5, 3, 4, 3, 7), seed), "units": Units(eV = 0, Ang = 0, muB = 0)},
        'FF' : {"value": rnd_matrix((5, 5, 4, 3, 7), seed), "units": Units(eV = 0, Ang = 0, muB = 0)},
        'FFF': {"value": rnd_matrix((3, 3, 2, 2, 3), seed), "units": Units(eV = 0, Ang = 0, muB = 0)},
        'G'  : {"value": rnd_matrix((4, 2, 2, 5), seed),    "units": Units(eV = 0, Ang = 0, muB = 0)},
        'H'  : {"value": rnd_matrix((4, 2, 7, 4), seed),    "units": Units(eV = 0, Ang = 0, muB = 0)},
        'I'  : {"value": rnd_matrix((2, 2), seed),          "units": Units(eV = 0, Ang = 0, muB = 0)},
        'J'  : {"value": rnd_matrix((3, 3), seed),          "units": Units(eV = 0, Ang = 0, muB = 0)},
        'Y'  : {"value": 4.2124 - 3.1411j,                  "units": Units(eV = 0, Ang = 0, muB = 0)},
        'Z'  : {"value": 2.4312 - 1.1231j,                  "units": Units(eV = 1, Ang = 1, muB = 1)},
    }
    
    return _ComputatorWf(quantities)

@pytest.fixture
def comp_test_same_len():
    return direct_comp_test_same_len()

def direct_comp_test_same_len(seed = None):
    # it is hard coded in test_latex_backwards.py that A and B are only constants here
    # it is hard coded in test_latex_backwards.py and test_automated_rnd_expr.py that matrices have shapes 3,...
    quantities = {
        'A' : {"value":  2.4312 - 1.1231j,                 "units": Units(eV = 0, Ang = 0, muB = 0)},
        'B' : {"value": -8.8282 - 6.3801j,                 "units": Units(eV = 0, Ang = 0, muB = 0)},
        'C' : {"value": rnd_matrix((3), seed),             "units": Units(eV = 0, Ang = 0, muB = 0)},
        'D' : {"value": rnd_matrix((3, 3), seed),          "units": Units(eV = 0, Ang = 0, muB = 0)},
        'E' : {"value": rnd_matrix((3, 3, 3), seed),       "units": Units(eV = 0, Ang = 0, muB = 0)},
        'F' : {"value": rnd_matrix((3, 3, 3, 3), seed),    "units": Units(eV = 0, Ang = 0, muB = 0)},
    }
    
    return _ComputatorWf(quantities)

@pytest.fixture
def threshold():
    return 1.0E-10

def three_evaluate(x, use_str, conditions = "", cores = None, divide_by_max = False, skip_brute = False):
    if cores is None:
        xc = deepcopy(x)
        custom  = xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = True)
        xc = deepcopy(x)
        custom2 = xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = False)
        if skip_brute == False:
            xc = deepcopy(x)
            custom3 = xc.evaluate(use_str, conditions, brute_force_sums = True)    
        if divide_by_max == False:
            assert np.all(np.abs((custom - custom2) / custom) < 1.0E-7)
            if skip_brute == False:
                assert np.all(np.abs((custom - custom3) / custom) < 1.0E-7)
        else:
            assert np.all(np.abs((custom - custom2) / np.max(np.abs(custom))) < 1.0E-7)
            if skip_brute == False:
                assert np.all(np.abs((custom - custom3) / np.max(np.abs(custom))) < 1.0E-7)
        return custom
    else:
        xc = deepcopy(x)
        xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = True)
        custom = xc.get_as_dictionary(cores)
        xc = deepcopy(x)
        xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = False)    
        custom2 = xc.get_as_dictionary(cores)
        if skip_brute == False:
            xc = deepcopy(x)
            xc.evaluate(use_str, conditions, brute_force_sums = True)    
            custom3 = xc.get_as_dictionary(cores)
        for core in cores:
            if divide_by_max == False:
                assert np.all(np.abs((custom[core] - custom2[core]) / custom[core]) < 1.0E-7)
                if skip_brute == False:
                    assert np.all(np.abs((custom[core] - custom3[core]) / custom[core]) < 1.0E-7)
            else:
                assert np.all(np.abs((custom[core] - custom2[core]) / np.max(np.abs(custom[core]))) < 1.0E-7)
                if skip_brute == False:
                    assert np.all(np.abs((custom[core] - custom3[core]) / np.max(np.abs(custom[core]))) < 1.0E-7)
        return custom, xc
    
def five_evaluate(x, use_str, cores, conditions = "", divide_by_max = False):
    xc = deepcopy(x)
    xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = True , optimize_recomputation = True )
    num_opt_rec = xc._from_last_evaluation_num_used_stored
    custom = xc.get_as_dictionary(cores)
    #
    xc = deepcopy(x)
    xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = True , optimize_recomputation = False)
    custom2 = xc.get_as_dictionary(cores)
    #
    xc = deepcopy(x)
    xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = False, optimize_recomputation = True )
    custom3 = xc.get_as_dictionary(cores)
    #
    xc = deepcopy(x)
    xc.evaluate(use_str, conditions, brute_force_sums = False, optimize_divisions = False, optimize_recomputation = False)
    custom4 = xc.get_as_dictionary(cores)
    #
    xc = deepcopy(x)
    xc.evaluate(use_str, conditions, brute_force_sums = True)
    custom5 = xc.get_as_dictionary(cores)

    for custom_alt in [custom2, custom3, custom4, custom5]:
        for core in cores:
            if divide_by_max == False:
                assert np.all(np.abs((custom[core] - custom_alt[core]) / custom[core]) < 1.0E-7)
            else:
                assert np.all(np.abs((custom[core] - custom_alt[core]) / np.max(np.abs(custom[core]))) < 1.0E-7)
                    
    return custom, num_opt_rec
            
#def check_error(match):
#    if match == "":
#        print("Must provide match!!")
#        assert False
#    if "^" in match or "/" in match or "*" in match:
#        return pytest.raises(ValueError, match = re.escape(match))
#    else:
#        return pytest.raises(ValueError, match = match)

def error_testing(exc_info, match):
    cmpto = exc_info.value.args[0].replace("*"*81, "").replace("\n", " ").replace("***  ", "").strip()
    assert exc_info.type is ValueError    
    assert cmpto == match
