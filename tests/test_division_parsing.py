import numpy as np
import sys
sys.path.append('../')
from wfbase import Units

import pytest

from common import *

def test_parsing_ampersand(comp_test, threshold):
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
    
    use_str = '_ijklmno <= -A * B_i * C_ij * D_klj / ( E_mjin * F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    use_str = '_ijklmno <= -A * B_i * C_ij * (D_klj / E_mjin) * (1.0 / F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_ijklmno <= -A * (D_klj / E_mjin) * B_i * C_ij * (1.0 / F_jikom )'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_ijklmno <= (D_klj / E_mjin) * B_i * C_ij * (1.0 / F_jikom ) * (-1.0) * (1/(1/A))'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

def test_parsing_ampersand_with_addition(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  ((-1.0)*x["A"] * x["B"][i] * (x["C"][i,j] + x["D"][k,l,j]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])) + x["B"][i]

    use_str = '_ijklmno <= (-A * B_i * (C_ij + D_klj) / ( E_mjin + F_jikom )) + B_i'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    use_str = '_ijklmno <= (-A * B_i * C_ij / ( E_mjin + F_jikom )) + (-A * B_i * D_klj / ( E_mjin + F_jikom )) + B_i'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


def test_parsing_ampersand_with_complicated_division(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  ((-1.0)*x["A"] * x["B"][i] * ((x["C"][i,j] + x["D"][k,l,j]/(x["A"] + x["B"][i]))/x["E"][m,j,i,n]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m] / x["B"][i]))

    use_str = '_ijklmno <= -A * B_i * ((C_ij + D_klj / (A + B_i)) / E_mjin) / ( E_mjin + F_jikom / B_i ) '
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


def test_parsing_multiple_divisions(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] / ( x["B"][i] / (x["C"][i,j] / ( x["D"][k,l,j] * (x["E"][m,j,i,n] / x["F"][j,i,k,o,m]))))
                                
    use_str = '_ijklmno <= -A / (B_i / (C_ij / ( D_klj * ( E_mjin / F_jikom ))))'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()



def test_parsing_complicated_2(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct[i,j,k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] + x["B"][i] * x["C"][i,j] / x["D"][k,l,j] - x["A"] * x["E"][m,j,i,n] + x["F"][j,i,k,o,m] *  x["C"][i,j] /  (x["A"] +  x["B"][i]) 

    use_str = '_ijklmno <= -A * B_i  + B_i * C_ij /  D_klj - A * E_mjin + F_jikom * C_ij / (A + B_i)  '
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


def test_division_with_unary_sign(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3), dtype = complex)
    for i in range(3):
        correct[i] += x["A"] / (-x["A"]) + x["B"][i]

    use_str = '_i <= A / (-A) + B_i'

    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    
