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

def test_latex(comp_test_same_len, threshold):
    for i in range(30):
        print("STEP", i)
        
        random.seed(i)
        orig_x = direct_comp_test_same_len(i)
        
        xc = deepcopy(orig_x)
        
        how_many_lines = 1
        how_many_conditions = 0
        how_many_diagonals = 0
        
        use_str, new_cores, cond_all, correct, xc = get_automated_expression(xc, how_many_lines, how_many_conditions, how_many_diagonals, print_code = True,  max_num_operations = [10, 40])
        print(use_str)
        cores = ["ZZA"]
        custom, xc = three_evaluate(xc, use_str, cores = cores, divide_by_max = True)

        for k in cores:
            kk = k.replace("ZZ", "")
            assert np.all(np.abs((custom[k] - correct[kk]) / np.max(np.abs(correct[kk]))) < threshold)
            
        # process latex back to python
            
        latex = str(xc.get_latex("ZZA"))

        print(latex)
        
        if latex.count(r"\Leftarrow") != 1:
            assert False

        sp = latex.split(r"\Leftarrow")

        out_inds = sp[0].replace(r"\mathrm{ZZA}", "").replace("_", "").replace("{", "").replace("}", "").replace(" ", "")

        latex = sp[1]

        latex = latex.replace(r"\displaystyle", "")
        latex = latex.replace(r" \, ", " * ")

        latex = latex.replace(r"\left(", "( ").replace(r"\right)", " )")
        
        latex = latex.strip()

        if latex[:5 + 1] != r"\sum_{":
            sum_indices = ""
        else:
            end = find_pair_parantheses(latex, 5, "{")
            sum_indices = latex[5 + 1: end].strip()
            latex = latex[end + 1:]

        while r"\frac{" in latex:
            start = latex.index(r"\frac{") + 5
            end = find_pair_parantheses(latex, start, "{")
            start2 = end + 1
            end2 = find_pair_parantheses(latex, start2, "{")
            latex = latex[:start - 5] + " (( " + latex[start + 1: end] + " )/( " + latex[start2 + 1: end2] + " )) " + latex[end2 + 2:]

        while r"{\rm Re} (" in latex:
            start = latex.index(r"{\rm Re} (") + 9
            end = find_pair_parantheses(latex, start, "(")
            latex = latex[:start - 9] + " np.real( " + latex[start + 1: end] + " ) " + latex[end + 1:]

        while r"{\rm Im} (" in latex:
            start = latex.index(r"{\rm Im} (") + 9
            end = find_pair_parantheses(latex, start, "(")
            latex = latex[:start - 9] + " np.imag( " + latex[start + 1: end] + " ) " + latex[end + 1:]

        while r"\overline{" in latex:
            start = latex.index(r"\overline{") + 9
            end = find_pair_parantheses(latex, start, "{")
            latex = latex[:start - 9] + " np.conjugate( " + latex[start + 1: end] + " ) " + latex[end + 1:]
            
        while r"_{" in latex:
            start = latex.index(r"_{") + 1
            end = find_pair_parantheses(latex, start, "{")
            end2 = latex[:start].rindex(" ")
            tensor_name = latex[end2 + 1: start - 1]
            latex = latex[:end2] + " x[\"" + tensor_name.strip() + "\"][" + ",".join(latex[start + 1: end].strip()) + "] " + latex[end + 1:]

        while r")^{ 2 }" in latex:
            end = latex.index(r")^{ 2 }")
            start = find_pair_parantheses_backwards(latex, end, ")")
            latex = latex[:start] + " np.power(" + latex[start + 1: end] + ", 2.0) " + latex[end + 7:]

#        while r"^{ 2 }" in latex:
#            end = latex.index(r"^{ 2 }")
#            start = latex[:end].rfind(" ")
#            latex = latex[:start] + " np.power(" + latex[start + 1: end] + ", 2.0) " + latex[end + 6:]

        latex = latex.replace("A", " x[\"A\"] ")
        latex = latex.replace("B", " x[\"B\"] ")

        latex = latex.replace(" i ", " *1.0j ")
        if latex[-2:] == " i":
            latex = latex[:-2] + " *1.0j"
        if latex[:2] == "i ":
            latex = "1.0j * " + latex[2:]
        
        while "  " in latex:
            latex = latex.replace("  ", " ")
        latex = latex.strip()


        # now create python code
        code = ""
        if out_inds == "":
            code = "comp_in_latex = 0.0" + "\n"
        else:
            code = "comp_in_latex = np.zeros(tuple([3]*" + str(len(out_inds)) + "), dtype = complex)" + "\n"
        sum_inds = list(sorted(set(out_inds + sum_indices)))
        prefix = ""
        for i in range(len(sum_inds)):
            code += prefix + "for " + sum_inds[i] + " in range(3):" + "\n"
            prefix += "    "
        code += prefix + "comp_in_latex"
        if out_inds != "":
            code += "[" + ",".join(out_inds) + "]"
        code += " += "
        code += latex
        code += "\n"

        print(code)

        comp_in_latex = {}
        code_dic = {"comp_in_latex": comp_in_latex, "np": np, "x": deepcopy(orig_x)}
        exec(code, code_dic)
        comp_in_latex = code_dic["comp_in_latex"]

        print("comp_in_latex")
        print(comp_in_latex)
        print("correct[A]")
        print(correct["A"])
        
        assert np.all(np.abs((comp_in_latex - correct["A"]) / np.max(np.abs(correct["A"]))) < threshold)
        

def find_pair_parantheses(s, ind, para):
    if para == "(":
        opp = ")"
    elif para == "[":
        opp = "]"
    elif para == "{":
        opp = "}"
    else:
        raise ValueError("Wrong parantheses")

    if s[ind] != para:
        raise ValueError("This is not parantheses")

    nest = 0
    for i in range(ind + 1, len(s)):
        if s[i] == para:
            nest += 1
            continue
        if s[i] == opp:
            if nest < 0:
                raise ValueError("Can't make a match!")
            if nest == 0:
                return i
            nest -= 1
            continue

    raise ValueError("Can't find a match")

        
def find_pair_parantheses_backwards(s, ind, para):
    if para == ")":
        opp = "("
    elif para == "]":
        opp = "["
    elif para == "}":
        opp = "{"
    else:
        raise ValueError("Wrong parantheses")

    if s[ind] != para:
        raise ValueError("This is not parantheses")

    nest = 0
    for i in range(ind - 1, -1, -1):
        if s[i] == para:
            nest += 1
            continue
        if s[i] == opp:
            if nest < 0:
                raise ValueError("Can't make a match!")
            if nest == 0:
                return i
            nest -= 1
            continue

    raise ValueError("Can't find a match")

        
