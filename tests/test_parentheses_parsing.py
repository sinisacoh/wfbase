import numpy as np
import sys
sys.path.append('../')
from wfbase import Units

import pytest

from common import *

def test_parsing_order(comp_test, threshold):
    x = comp_test

    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.real(x["B"][i] + x["C"][i,j]) / (x["E"][m,j,i,n] + x["B"][i])
    use_str = '_ijn <= Real(B_i + C_ij)/(E_mjin + B_i)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()
    

    
    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.real((x["B"][i] + x["C"][i,j]) / (x["E"][m,j,i,n] + x["B"][i]))
    use_str = '_ijn <= Real((B_i + C_ij)/(E_mjin + B_i))'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    
    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.conjugate(x["B"][i] + x["C"][i,j]) / (x["E"][m,j,i,n] + x["B"][i])
    use_str = '_ijn <= #(B_i + C_ij)/(E_mjin + B_i)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    
    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.conjugate(x["B"][i] * x["C"][i,j] * x["E"][m,j,i,n] * x["B"][i])
    use_str = '_ijn <= #(B_i * C_ij * E_mjin * B_i)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    
    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.conjugate(x["B"][i]) * x["C"][i,j] * x["E"][m,j,i,n] * x["B"][i]
    use_str = '_ijn <= #B_i * C_ij * E_mjin * B_i'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    
    correct = np.zeros((3, 5, 6), dtype = complex)
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct[i,j,n] += np.conjugate((x["B"][i] + x["C"][i,j]) / (x["E"][m,j,i,n] + x["B"][i]))
    use_str = '_ijn <= #((B_i + C_ij)/(E_mjin + B_i))'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    
    correct = np.zeros((3), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[i] += (-1.0)*(x["B"][i] + x["C"][i,j])
    use_str = '_i <= -(B_i + C_ij)'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()

    
    
    correct = np.zeros((3), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[i] += (-1.0)*(x["B"][i]) + x["C"][i,j]                
    use_str = '_i <= -(B_i) + C_ij'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()



    correct = np.zeros((3), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct[i] += np.power(x["B"][i] + x["C"][i,j] + x["A"], x["Y"])
    use_str = '_i <= (B_i + C_ij + A)^Y'
    custom = three_evaluate(x, use_str)
    assert (abs((custom - correct) / correct) < threshold).all()


    

    
    
