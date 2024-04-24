import numpy as np
import sys
sys.path.append('../')
from wfbase import Units, _ComputatorWf

import pytest

from common import *

def test_change_dummy_indices_simple(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    use_str = '_ijklmno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    custom  = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()

def test_change_order_terms_in_products(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    use_str = '_ijklmno <= ( A * B_i * C_ij * D_klj ) / (E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    use_str = '_ijklmno <= ( A * B_i * D_klj * C_ij  ) / (E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    use_str = '_ijklmno <= ( A * B_i * D_klj * C_ij  ) / (F_jikom * E_mjin )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
def test_change_dummy_indices_in_parantheses(comp_test, threshold):
    x = comp_test

    use_str = '_ijklmno <= (A * B_i * (C_ij + D_klj + C_ij) ) / ((E_mjin + F_jikom + D_klj) * C_ij * D_klj )'
    correct = three_evaluate(x, use_str)
        
    custom  = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()
    
def test_change_order_terms_in_parantheses(comp_test, threshold):
    x = comp_test

    use_str = '_ijklmno <= (A * B_i * (-C_ij + D_klj + C_ij) ) / ((E_mjin - F_jikom + D_klj) * C_ij * D_klj )'
    correct = three_evaluate(x, use_str)
        
    use_str = '_ijklmno <= (A * B_i * (C_ij + D_klj - C_ij) ) / (( D_klj + E_mjin - F_jikom) * C_ij * D_klj )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_ijklmno <= (A * B_i * (D_klj + C_ij - C_ij) ) / ((-F_jikom + D_klj + E_mjin) * C_ij * D_klj )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_ijklmno <= (A * B_i * (D_klj + C_ij - C_ij) ) / ((-F_jikom + D_klj + E_mjin) * C_ij * D_klj )'
    custom  = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()

    
def test_change_minus_signs(comp_test, threshold):
    x = comp_test

    use_str = '_ijklmno <= (A * B_i * (-C_ij - D_klj - C_ij) ) / ((E_mjin - F_jikom + D_klj) * C_ij * D_klj )'
    correct = three_evaluate(x, use_str)
        
    use_str = '_ijklmno <= (A * B_i * ( C_ij + D_klj + C_ij) ) / ((-E_mjin + F_jikom - D_klj) * C_ij * D_klj )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    
def test_repeated_indices_simple(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 2, 7), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            correct[i,j,l,m] +=  x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["G"][k,l,l,j] * x["H"][k,l,m,k])
    
    use_str = "_ijlm <= (A * B_i * C_ij * D_klj ) / (E_mjin * G_kllj * H_klmk )"
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    
def test_repeated_indices_in_parantheses(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 2, 7, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for r in range(4):
                                correct[i,j,l,m,r] +=  x["A"] * x["B"][i] * (x["C"][i,j] - x["D"][k,l,j]) * x["H"][k,l,m,r] / ((-x["E"][m,j,i,n] + x["G"][k,l,l,j]) * x["H"][k,l,m,k])
    
    use_str = '_ijlmr <= (A * B_i * (C_ij - D_klj) * H_klmr ) / ((-E_mjin + G_kllj) * H_klmk )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    custom = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("k", "w").replace("l", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()

def test_conjugate(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  x["A"] * np.conjugate(x["B"][i] + x["C"][i,j]) * np.conjugate(x["D"][k,l,j]) / (x["E"][m,j,i,n] * np.conjugate(np.conjugate(x["F"][j,i,k,o,m])))
    
    use_str = '_ijklmno <= (A * #(B_i + C_ij) * #D_klj ) / (E_mjin * ##F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    custom  = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()
        
def test_constants(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  x["A"] * (x["B"][i] + 3.0j * x["C"][i,j]) * x["D"][k,l,j] / (x["E"][m,j,i,n] * (7.3j + 3.0 + 3.j + 4. + 9 + 1j) * x["F"][j,i,k,o,m])
    
    use_str = '_ijklmno <= (A * (B_i + 3.j * C_ij) * D_klj ) / (E_mjin * (7.3j + 3.0 + 3.j + 4. + 9 + 1j) * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    custom  = three_evaluate(x, use_str.replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()
    
def test_trace(comp_test, threshold):
    x = comp_test

    correct = 0.0
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct +=  x["A"] * (x["B"][i] + 3.0 * x["C"][i,j]) * x["D"][k,l,j] / (x["E"][m,j,i,n] * 7.3 * x["F"][j,i,k,o,m])
    
    use_str = '_ <= (A * (B_i + 3.0 * C_ij) * D_klj ) / (E_mjin * 7.3 * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    custom  = three_evaluate(x, use_str.replace("j","z").replace("i","a").replace("m","b").replace("o", "w").replace("k", "t"))
    assert (abs((custom - correct) / correct) < threshold).all()
    
    
def test_no_einsum(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    use_str = '_ijklmno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '-A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str, skip_brute = True)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    
def test_check_units(comp_test, threshold):
    x = comp_test
    
    quantities={}
    keys = x.all_quantities()
    for k in keys:
        quantities[k] = {"value": x.get(k, "value"), "units": x.get(k, "units")}
    
    quantities["A"]["units"] = Units(eV = 1, Ang = 0, muB = 0)
    quantities["B"]["units"] = Units(eV = 0, Ang = 2, muB = 0)
    quantities["C"]["units"] = Units(eV = 0, Ang = 0, muB = 3)
    quantities["D"]["units"] = Units(eV = 2, Ang = 2, muB = 2)
    quantities["E"]["units"] = Units(eV = 0, Ang = 2, muB = 0)
    quantities["F"]["units"] = Units(eV = 0, Ang = 2, muB = 0)
    
    x = _ComputatorWf(quantities)

    use_str = 'Q_ijklmno <= -A * B_i * (C_ij ^ 3) * (D_klj + D_klj)  / ( -E_mjin + F_jikom + B_i)'
    x.evaluate(use_str)
    result_units = x.get("Q", "units")
    
    assert (result_units._check_units_the_same(Units(eV = 3, Ang = 2, muB = 11)))


def test_check_if_update_numbers(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    use_str = '_ijklmno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)

    x['E'] = rnd_matrix((7, 5, 3, 6)) * 3.9

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

     

def test_ambiguous_indices(comp_test, threshold):
    x = comp_test

    correct = np.zeros((5), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[j] +=  x["B"][i] + x["A"] * x["B"][i] * x["C"][i,j]
    
    use_str = '_j <= B_i + A * B_i * C_ij'
    print(three_evaluate(x, use_str))
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_j <= #(B_i + A * B_i * C_ij)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct.conjugate()) / correct.conjugate()) < threshold).all()

    use_str = '_j <= -(B_i + A * B_i * C_ij)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct*(-1.0)) / (correct*(-1.0))) < threshold).all()

    correct = np.zeros((5), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[j] +=  x["A"] + x["A"] * x["B"][i] * x["C"][i,j]
    
    use_str = '_j <= A + A * B_i * C_ij'
    print(three_evaluate(x, use_str))
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    correct = np.zeros((5), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[j] +=  x["B"][i] * x["C"][i,j]
    correct = correct + 3.0
            
    use_str = '_j <= 1.0 + B_i * C_ij'
    print(three_evaluate(x, use_str))
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

def test_parsing_conjugates_meaning(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[m] += np.conjugate(x["E"][m,j,i,n]) + 3.0j + x["B"][i] + x["A"]

    use_str = '_m <=  #E_mjin + 3.0j + B_i + A'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    
    correct = np.zeros((7), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[m] += np.conjugate(x["E"][m,j,i,n] + 3.0j) + x["B"][i] + x["A"]

    use_str = '_m <=  #(E_mjin + 3.0j) + B_i + A'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    correct = np.zeros((7), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[m] += np.conjugate(x["E"][m,j,i,n]) / 3.0j + x["B"][i] + x["A"]

    use_str = '_m <=  #E_mjin/3.0j + B_i + A'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
