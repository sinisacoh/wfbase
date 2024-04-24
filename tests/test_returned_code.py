import numpy as np
import sys
sys.path.append('../')
from wfbase import Units, _ComputatorWf
import random
from pprint import pprint
import pytest
from common import *
from test_automated_rnd_expr import get_automated_expression

np.set_printoptions(linewidth = 100000)

def test_returned_code(comp_test_same_len, threshold):
    for i in range(30):
        print("STEP", i)
        
        random.seed(i)
        orig_x = direct_comp_test_same_len(i)
        
        xc = deepcopy(orig_x)
        
        how_many_lines = 1
        how_many_conditions = 1
        how_many_diagonals = 1
        
        use_str, new_cores, cond_all, correct, xc = get_automated_expression(xc, how_many_lines, how_many_conditions, how_many_diagonals, print_code = True,  max_num_operations = [10, 40])
        cores = ["ZZA"]
        custom, xc = three_evaluate(xc, use_str, cond_all, cores = cores, divide_by_max = True)

        assert np.all(np.abs((custom["ZZA"] - correct["A"]) / np.max(np.abs(correct["A"]))) < threshold)
        
        code = xc.get("ZZA", "exec")["code"]
        code += "__value = evaluate_directly(__object)\n"
        code_dic = {"__object": xc}
        exec(code, code_dic)
        from_exec_result = code_dic["__value"]

        assert np.all(np.abs((from_exec_result - correct["A"]) / np.max(np.abs(correct["A"]))) < threshold)
        
