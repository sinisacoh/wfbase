import numpy as np
import sys
sys.path.append('../')
from wfbase import Units

import pytest

from common import *

def test_simple_condition(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7,6,3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["C"][i,j]) > -0.3 and np.real(x["D"][k,l,j] < 0.5):
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])

    use_str = '_mno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    conditions = "C_ij > -0.3 , D_klj < 0.5"
    custom  = three_evaluate(x, use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()

def test_simple_condition_too_large(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7,6,3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["C"][i,j]) > 10.0 and np.real(x["D"][k,l,j] < 0.5):
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * (x["C"][i,j] + 1.0) * x["D"][k,l,j] / x["E"][m,j,i,n] + x["F"][j,i,k,o,m] + 3.0

    use_str = '_mno <= -A * B_i * (C_ij + 1.0)* D_klj  / E_mjin + F_jikom'
    conditions = "C_ij > 10.0 , D_klj < 0.5"
    with pytest.raises(ValueError) as exc_info:
        custom  = three_evaluate(x, use_str, conditions)
    error_testing(exc_info, "Condition on index i is so restrictive that it removes all elements.")
    
def test_simple_condition_too_small(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7,6,3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["C"][i,j]) > -10.0 and np.real(x["D"][k,l,j] < 0.5):
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])

    use_str = '_mno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    conditions = "C_ij > -10.0 , D_klj < 0.5"
    custom  = x.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
    
def test_simple_condition_other(comp_test, threshold):
    x = comp_test

    correct = np.zeros((7,6,3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["C"][i,j]) > -0.3 and np.real(x["D"][k,l,j] < 0.5):
                                    correct[m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] + x["E"][m,j,i,n] * x["F"][j,i,k,o,m] * x["C"][o,j]

    use_str = '_mno <= -A * B_i * C_ij * D_klj  +  E_mjin * F_jikom * C_oj'
    conditions = "C_ij > -0.3 , D_klj < 0.5"
    custom  = x.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
    

def test_simple_condition_edge_case(comp_test, threshold):
    x = comp_test

    y = deepcopy(x)
    y.new("W", {"value": np.array([0,1,2,3,4,5])})
    correct = np.array(4 + 5)
    use_str = '_ <= W_i'
    conditions = "W_i > 3"
    custom  = y.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    y = deepcopy(x)
    y.new("W", {"value": np.array([0,1,2,3,4,5])})
    correct = np.array(5)
    use_str = '_ <= W_i'
    conditions = "W_i > 4"
    custom  = y.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
    

    y = deepcopy(x)
    y.new("W", {"value": np.array([0,1,2,3,4,5])})
    correct = np.array(3 + 4)
    use_str = '_ <= W_i'
    conditions = "W_i > 2, W_i < 5"
    custom  = y.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
    
    y = deepcopy(x)
    y.new("W", {"value": np.array([0,1,2,3,4,5])})
    use_str = 'W_i'
    conditions = "W_i > 2, W_i < 5"
    with pytest.raises(ValueError) as exc_info:
        custom  = y.evaluate(use_str, conditions)
    error_testing(exc_info, "You specified greater/lesser condition involving indices i but now you are not summing over index i.")

def test_simple_condition_edge_case_2d(comp_test, threshold):
    for combinations in [-2.5, 0.5, 2.5, 10.5, 30, 49.5]:
        x = deepcopy(comp_test)
        
        # 5x6
        x.new("W", {"value": np.array([[ 0, 1, 2, 3, 4, 5],
                                                [ 0,10,20,30, 4,50],
                                                [ 0,10, 2,30,40, 5],
                                                [10,10,20, 3,40,50],
                                                [ 0, 1,20,30,40, 5],
                                                ])})
        # 5x4
        x.new("R", {"value": np.array([[ 0, 1, 2, 3],
                                                [20,30, 4,50],
                                                [ 0, 2, 3, 5],
                                                [10, 3,40,50],
                                                [ 5,30,40, 5],
                                                ])})
        correct = np.zeros((4), dtype = complex)
        for i in range(5):
            for j in range(6):
                for k in range(4):
                    if np.real(x["W"][i,j]) > combinations:
                        correct[k] +=  x["W"][i,j] + x["R"][i,k]

        use_str = '_k <= W_ij + R_ik'
        conditions = "W_ij > " + str(combinations)
        custom  = x.evaluate(use_str, conditions)
        assert (abs((custom - correct) / correct) < threshold).all()
        
def test_more_complicated_expression_with_condition(comp_test, threshold):
    x = comp_test

    correct = np.zeros((4, 2, 7, 6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["C"][i,j]) > 0.4:
                                    correct[k,l,m,n,o] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    
    use_str = '_klmno <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )'
    conditions = "C_ij > 0.4"
    custom  = x.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
        
def test_more_complicated_expression_with_condition_2(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 7), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            if np.real(x["D"][k,l,j]) > 0.2:
                                correct[i,m] +=  x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["G"][k,l,l,j] * x["H"][k,l,m,k])
    
    use_str = "_im <= (A * B_i * C_ij * D_klj ) / (E_mjin * G_kllj * H_klmk )"
    conditions = "D_klj > 0.2"
    custom  = x.evaluate(use_str, conditions)
    assert (abs((custom - correct) / correct) < threshold).all()
        
