import numpy as np
import sys
sys.path.append('../')
from wfbase import Units

import pytest

from common import *

def test_simple_diagonal(comp_test, threshold):
    x = comp_test
    
    correct = np.zeros((7,7,3), dtype = complex)
    for i in range(5):
        for j in range(5):
            for k in range(4):
                for l in range(4):
                    for m in range(7):
                        for n in range(7):
                            for o in range(3):
                                if np.real(x["CC"][i,j]) > -0.3 and np.real(x["DD"][k,l,j] < 0.5) and i != j and l != k:
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["BB"][i] * x["CC"][i,j] * x["DD"][k,l,j] / (x["EE"][m,j,i,n] * x["FF"][j,i,k,o,m])
                                    
    use_str = '_mno <= -A * BB_i * CC_ij * DD_klj  / ( EE_mjin * FF_jikom )'
    conditions = "CC_ij > -0.3 , DD_klj < 0.5, i != j, l != k"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

def test_simple_diagonal_other_side(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7,7,3), dtype = complex)
    for i in range(5):
        for j in range(5):
            for k in range(4):
                for l in range(4):
                    for m in range(7):
                        for n in range(7):
                            for o in range(3):
                                if np.real(x["CC"][i,j]) > -0.3 and np.real(x["DD"][k,l,j] < 0.5) and i != j and m != n:
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["BB"][i] * x["CC"][i,j] * x["DD"][k,l,j] / (x["EE"][m,j,i,n] * x["FF"][j,i,k,o,m])

    use_str = '_mno <= -A * BB_i * CC_ij * DD_klj  / ( EE_mjin * FF_jikom )'
    conditions = "CC_ij > -0.3 , DD_klj < 0.5, i != j, m != n"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

        
def test_soft_division(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3), dtype = complex)
    for i in range(5):
        for j in range(5):
            for k in range(3):
                for l in range(5):
                    if np.real(x["CC"][i,j]) > -0.3 and j != l:
                        correct[k] +=  (-1.0)*x["A"] * x["B"][k] / ( x["CC"][i,j] - x["CC"][i,l] + 0)

    use_str = '_k <= -A * B_k / (CC_ij - CC_il + 0)'
    conditions = "CC_ij > -0.3 , j != l"

    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

        
def test_soft_division_mult(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3), dtype = complex)
    for i in range(5):
        for j in range(2):
            for k in range(4):
                for n in range(5):
                    for m in range(3):
                        if np.real(x["D"][k,j,i]) > -0.3 and i != n:
                            correct[m] +=  (-1.0) * x["A"] * x["B"][m] / ( x["D"][k,j,i] - x["D"][k,j,n] + 0)
                                                                     
    use_str = '_m <= -A * B_m / (D_kji - D_kjn + 0)'
    conditions = "D_kji > -0.3 , i != n"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

        
def test_soft_division_mult_2(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3), dtype = complex)
    for i in range(5):
        for j in range(2):
            for k in range(4):
                for r in range(4):
                    for p in range(4):
                        for n in range(5):
                            for m in range(3):
                                if np.real(x["D"][k,j,i]) > -0.3 and i != n and k != r:
                                    correct[m] +=  (-1.0) * x["A"] * x["B"][m] / ( x["D"][p,j,i] - x["D"][p,j,n] + x["D"][k,j,i] - x["D"][r,j,i] + 0)
                                                                     
    use_str = '_m <= -A * B_m / (D_pji - D_pjn + D_kji - D_rji + 0)'
    conditions = "D_kji > -0.3 , i != n, k != r"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

def test_diagonals_using_broadcast(comp_test, threshold):
    x = comp_test

    # make sure that slice _s does something
    tmp = np.copy(x["C"])
    tmp[:, 3] += -10.0
    x.replace("C", tmp)

    # put zeros at place you need to divide
    tmp = np.copy(x["FF"])
    for i in range(5):
        tmp[i,i,:,:,:] = 0.0
    x.replace("FF", tmp)
    
    correct = np.zeros((4,7), dtype = complex)
    for i in range(5):
        for I in range(5):
            for j in range(5):
                for J in range(5):
                    for k in range(4):
                        for l in range(3):
                            for m in range(7):
                                if np.real(x["C"][l,j]) > -0.3 and i != j:
                                    correct[k,m] += x["BB"][i] + ((3.0 + x["FF"][I,J,k,l,m]) / (x["C"][l,j]*0 + x["FF"][i,j,k,l,m]) )

    use_str = '_km <= BB_i + ((3 + FF_IJklm) / (C_lj*0 + FF_ijklm))'
    conditions = "C_lj > -0.3, i != j"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

    def replace_all(xx): return xx.replace("i","z").replace("j","x").replace("l", "v").replace("I", "y").replace("J", "W").replace("k","r").replace("m","a")
    
    custom  = three_evaluate(x, replace_all(use_str), replace_all(conditions), divide_by_max = True)
    assert (abs((custom - correct) / correct) < threshold).all()
    
def test_diagonals_using_broadcast_multiple(comp_test, threshold):
    tot_return = 0
    for i in range(30):
        x = direct_comp_test(i)
        ret = do_one(x, threshold)
        if ret == True:
            tot_return += 1
    assert tot_return > 0
            
def do_one(x, threshold):
    
    # put zeros at place you need to divide
    tmp = np.copy(x["FFF"])
    for i in range(3):
        tmp[i,i,:,:,:] = 0.0
    for i in range(3):
        tmp[i,:,:,:,i] = 0.0
#    for i in range(3):
#        tmp[:,i,:,:,i] = 0.0
    for i in range(2):
        tmp[:,:,i,i,:] = 0.0
    x.replace("FFF", tmp)
    
    correct = np.zeros((3, 2), dtype = complex)
    for a in range(3):
        for b in range(3):
            for c in range(2):
                for d in range(2):
                    for e in range(3):
                        for A in range(3):
                            for B in range(3):
                                for C in range(2):
                                    if np.real(x["I"][c, d]) > -0.3 and np.real(x["J"][A, b]) > -0.2 and a != b and a != e and c != d:
                                        correct[B, C] += x["I"][c, d] * x["J"][A, b] + (x["I"][C, d] + x["J"][a, B]) / (x["I"][c, d]*0.0 + x["FFF"][a, b, c, d, e])

    if np.sum(np.abs(correct)) < threshold:
        return False
                                        
    use_str = '_BC <= I_cd * J_Ab + (I_Cd + J_aB) / (I_cd * 0 + FFF_abcde)'
    conditions = "I_cd > -0.3, J_Ab > -0.2, a != b, a != e, c != d"
    custom  = three_evaluate(x, use_str, conditions, divide_by_max = True)
    assert (abs((custom - correct) / np.max(np.abs(correct))) < threshold).all()

    def replace_all(xx): return xx.replace("a","z").replace("b","x").replace("c", "v").replace("d", "y").replace("e", "W").replace("A","r").replace("B","a").replace("C","t")
    
    custom  = three_evaluate(x, replace_all(use_str), replace_all(conditions), divide_by_max = True)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    return True
