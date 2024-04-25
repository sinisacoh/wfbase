import numpy as np
import sys
sys.path.append('../')
from wfbase import Units, _ComputatorWf
import random
from pprint import pprint

import pytest

from common import *

np.set_printoptions(linewidth = 100000)

# so that error is raised if divide by zero
np.seterr(all = 'raise')

def test_automated_single_line_expression(comp_test_same_len, threshold):
    for i in range(20):
        print("STEP", i)
        
        random.seed(i)
        orig_x = direct_comp_test_same_len(i)
        
        xc = deepcopy(orig_x)
        
        how_many_lines = 1
        how_many_conditions = 0
        how_many_diagonals = 0

        use_str, new_cores, cond_all, correct, xc = get_automated_expression(xc, how_many_lines, how_many_conditions, how_many_diagonals, print_code = True,  max_num_operations = [30, 60])
        use_str = use_str.replace("ZZA <=", "_ <=").replace("ZZA_", "_")
        correct = correct["A"]

        print(use_str)
        print("correct")
        pprint(correct)
        
        custom = three_evaluate(xc, use_str, divide_by_max = True)        
        print("custom")
        pprint(custom)
        assert np.all(np.abs((custom - correct) / np.max(np.abs(correct))) < threshold)
    
def test_automated_multiple_line_expression_short(comp_test_same_len, threshold):
    for i in range(25):
        print("STEP", i)

        random.seed(i)
        orig_x = direct_comp_test_same_len(i)

        xc = deepcopy(orig_x)

        how_many_lines = 2
        how_many_conditions = 1
        how_many_diagonals = 1

        use_str, new_cores, cond_all, correct, xc = get_automated_expression(xc, how_many_lines, how_many_conditions, how_many_diagonals, print_code = True,  max_num_operations = [2, 5],
                                                                             inds = ["a","b","c","d","e","f"], max_left_allowed_inds = 3, max_cond_inds = 2)
        
        print(use_str)
        print(cond_all)
        print("correct")
        pprint(correct)
        
        custom, num_opt_rec = five_evaluate(xc, use_str, new_cores, cond_all, divide_by_max = True)
        print("custom")
        pprint(custom)
        for k in new_cores:
            kk = k.replace("ZZ", "")
            assert np.all(np.abs((custom[k] - correct[kk]) / np.max(np.abs(correct[kk]))) < threshold)

        
def test_automated_multiple_line_expression_long(comp_test_same_len, threshold):
    # breaks at i = 1126 because there is near cancellation in ZZD between these guys: (ZZB_dbb - ZZB_fch)
    # hard to make completely robust tester with so many operations.
    for i in range(25):
        print("STEP", i)

        random.seed(i)
        orig_x = direct_comp_test_same_len(i)

        xc = deepcopy(orig_x)

        how_many_lines = random.randint(2, 5)
        how_many_conditions = random.randint(0, 2)
        how_many_diagonals = random.randint(0, 2)

        use_str, new_cores, cond_all, correct, xc = get_automated_expression(xc, how_many_lines, how_many_conditions, how_many_diagonals, print_code = True,  max_num_operations = [10, 14])
        
        print(use_str)
        print(cond_all)
        print("correct")
        pprint(correct)
        
        custom, num_opt_rec = five_evaluate(xc, use_str, new_cores, cond_all, divide_by_max = True)
        print("custom")
        pprint(custom)
        for k in new_cores:
            kk = k.replace("ZZ", "")
            assert np.all(np.abs((custom[k] - correct[kk]) / np.max(np.abs(correct[kk]))) < threshold)

        
def get_automated_expression(x, how_many_lines, how_many_conditions, how_many_diagonals, print_code = False, max_num_operations = [10, 14], inds = ["a","b","c","d","e","f","g","h"], max_left_allowed_inds = 4, max_cond_inds = 4):
    while True:            
        terms = x.all_quantities()
        dimens = {}
        for k in terms:
            dimens[k] = len(x.get_shape(k))

        # only allow these indices on the left of <=
        allowed_left_inds = random.sample(inds, max_left_allowed_inds)            
        allowed_cond_inds = list(sorted(set(inds).difference(allowed_left_inds)))

        # precompute which indices will appear in conditions
        all_conds_inds = []
        for l in range(how_many_conditions):
            all_conds_inds.append(random.sample(allowed_cond_inds, random.randint(1, max_cond_inds)))
        #
        cond_terms = []
        cond_inds = []
        cond_opers = []
        cond_limit = []
        for w in range(how_many_conditions):
            allowed_tensor_terms = []
            for k in x.all_quantities():
                if len(x.get_shape(k)) == len(all_conds_inds[w]):
                    allowed_tensor_terms.append(k)
                
            cond_terms.append(random.sample(allowed_tensor_terms, 1)[0])
            cond_inds.append(all_conds_inds[w])
            cond_opers.append(random.choices(["<", ">"], k = 1)[0])
            cond_limit.append("%f"%(random.random()*0.2 - 0.1))                
        all_cond_code = []
        for j in range(how_many_conditions):
            all_cond_code.append(cond_terms[j] + "_" + "".join(cond_inds[j]) + " " + cond_opers[j] + " " + cond_limit[j])
        all_cond_code_py = []
        for j in range(how_many_conditions):
            all_cond_code_py.append("x[\"" + cond_terms[j] + "\"]" + "[" + ",".join(cond_inds[j]) + "]" + " " + cond_opers[j] + " " + cond_limit[j])

        # set some of the rows/columns so that all elements are False, so that we are actually testing conditions
        for w in range(how_many_conditions):
            _term = cond_terms[w]
            _ndim = dimens[_term]
            _val = x[_term]
            if _ndim >= 1:
                if random.randint(0, 10) > 3:
                    _val[random.randint(0, _val.shape[0] - 1)] += 2.0
            if _ndim >= 2:
                if random.randint(0, 10) > 3:
                    _val[:, random.randint(0, _val.shape[1] - 1)] += 2.0
            if _ndim >= 3:
                if random.randint(0, 10) > 3:
                    _val[:, :, random.randint(0, _val.shape[2] - 1)] += 2.0
            if _ndim >= 4:
                if random.randint(0, 10) > 3:
                    _val[:, :, :, random.randint(0, _val.shape[3] - 1)] += 2.0
            x.replace(_term, _val)

        # store which pairs of indices are allowed to appear in the diagonals
        allowed_diagonals = np.zeros((len(inds), len(inds)), dtype = bool)
        allowed_diagonals[:, :] = True
        for _i in range(allowed_diagonals.shape[0]):
            allowed_diagonals[_i, _i] = False
            
        code = ""
        use_str = ""            
        code += r"correct = {}" + "\n\n"
        new_cores = []
        all_conds_indices_used = []
        all_used_inds = []
        all_casee = []
        for j in range(how_many_lines):
            casee = chr(ord("A") + j)
            all_casee.append(casee)
            one_code, one_use_str, out_inds, apply_conds_this_line, used_inds = do_one(x, terms, dimens, inds, allowed_left_inds, all_conds_inds, all_cond_code_py, prefix = casee, max_num_operations = max_num_operations)

            allowed_diag_left  = list(sorted(set(out_inds)))
            allowed_diag_right = list(sorted(set(used_inds).difference(out_inds)))
            allowed_diag_left_ord  = sorted([inds.index(_i) for _i in allowed_diag_left ])
            allowed_diag_right_ord = sorted([inds.index(_i) for _i in allowed_diag_right])
            _tmp = np.zeros((len(inds), len(inds)), dtype = bool)
            _tmp[:, :] = True
            for _i in allowed_diag_left_ord:
                for _j in allowed_diag_right_ord:
                    _tmp[_i, _j] = False
                    _tmp[_j, _i] = False
            for _i in range(_tmp.shape[0]):
                _tmp[_i, _i] = False
            allowed_diagonals = np.logical_and(allowed_diagonals, _tmp)
            all_used_inds.append(used_inds)
            
            terms.append("ZZ" + casee)
            dimens["ZZ" + casee] = len(out_inds)
            
            code +=one_code + "\n"
            use_str += one_use_str + "\n"
            
            new_cores.append("ZZ" + casee)
            
            for ww in apply_conds_this_line:
                all_conds_indices_used.append(ww)

        str_all_used_inds = ""
        for _s in all_used_inds:
            str_all_used_inds += "".join(_s)
        not_used_inds = list(sorted(set(inds).difference(set(str_all_used_inds))))
        not_used_inds  = sorted([inds.index(_i) for _i in not_used_inds])
        for _i in not_used_inds:
            allowed_diagonals[_i, :] = False
            allowed_diagonals[:, _i] = False
        allowed_diagonal_pairs = []
        for _i in range(allowed_diagonals.shape[0]):
            for _j in range(_i, allowed_diagonals.shape[0]):
                if allowed_diagonals[_i, _j] == True:
                    allowed_diagonal_pairs.append(inds[_i] + inds[_j])
        use_pairs = random.sample(allowed_diagonal_pairs, how_many_diagonals)
        use_pairs_str = []
        for up in use_pairs:
            use_pairs_str.append(up[0] + " != " + up[1])
        for w in range(len(all_used_inds)):
            tmp_txt = []
            for q in range(len(use_pairs)):
                if use_pairs[q][0] in all_used_inds[w] and use_pairs[q][1] in all_used_inds[w]:
                    tmp_txt.append(use_pairs_str[q])
            if len(tmp_txt) == 0:
                code = code.replace("DIAGONAL_" + all_casee[w], " True")
            else:
                code = code.replace("DIAGONAL_" + all_casee[w], " and ".join(tmp_txt))
                    
        all_conds_indices_used = list(sorted(set(all_conds_indices_used)))
        
        cond_all = []
        for ww in all_conds_indices_used:
            cond_all.append(all_cond_code[ww])
        cond_all = ", ".join(cond_all + use_pairs_str)
            
        correct = {}
        code_dic = {"correct": correct, "np": np, "x": x}
            
        all_ok = True
        try:
            exec(code, code_dic)
            correct = code_dic["correct"]
            for k in correct.keys():
                if not np.all(np.isfinite(correct[k])):
                    all_ok = False
                    print("Result is not finite, let me try another combination.")
                if np.max(np.abs(correct[k])) < 1.0E-8:
                    all_ok = False
                    print("Result is close to zero, let me try another combination.")
        except ZeroDivisionError:
            all_ok = False
            print("Dividing by zero, let me try something else.")
        except FloatingPointError:
            all_ok = False
            print("Float dividing by zero, let me try something else.")
            
        if all_ok == True:
            break

    if print_code == True:
        print(code)

    use_str = use_str.strip()
        
    return use_str, new_cores, cond_all, correct, x
            

def do_one(x, terms, dimens, inds, allowed_left_inds, all_conds_inds, all_cond_code_py, prefix = "", max_num_operations = [10, 14]):
    out_inds = random.sample(allowed_left_inds, random.randint(0, 3))

    used_inds = []
    inds_weights = None
    expr, expr_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    all_funcs   = [rnd_sum_l, rnd_sum_r, rnd_min_l, rnd_min_r, rnd_mul_l, rnd_mul_r, rnd_div_l, rnd_div_r, rnd_sign, rnd_conj, rnd_par, rnd_real, rnd_imag, rnd_power_2]
    all_funcs_w = [        3,         3,         3,         3,         7,         7,         5,         5,        2,        2,       7,        1,        1,           1]
    min_num_func_calls = random.randint(max_num_operations[0], max_num_operations[1])
    tot_num_func_calls = 0
    just_had_division = False
    while True:
        func = random.choices(all_funcs, weights = all_funcs_w, k = 1)[0]
        use_func = func
        if just_had_division == True:
            if use_func in [rnd_mul_r, rnd_div_l, rnd_div_r]:
                while True:
                    new_func = random.choices(all_funcs, weights = all_funcs_w, k = 1)[0]
                    if new_func not in [rnd_mul_r, rnd_div_l, rnd_div_r]:
                        use_func = new_func
                        break
            if use_func in [rnd_sum_r, rnd_min_r, rnd_conj, rnd_par]:
                just_had_division = False
        expr, expr_py, used_inds = use_func(expr, expr_py, dimens, terms, inds, used_inds, inds_weights)
        if use_func == rnd_power_2:
            irem = all_funcs.index(rnd_power_2)
            all_funcs.pop(irem)
            all_funcs_w.pop(irem)
        
        if use_func in [rnd_div_l, rnd_div_r]:
            just_had_division = True
        tot_num_func_calls += 1

        # make sure that expression is consistent with all conditions
        which_would_want_to_get = ""
        indices_consistent_with_conditions = True
        apply_conds_this_line = []
        for ii, ca in enumerate(all_conds_inds):
            how_many_appear = 0
            tmp = []
            for qq in ca:
                if qq in used_inds:
                    how_many_appear += 1
                else:
                    tmp.append(qq)
            if how_many_appear != 0 and how_many_appear != len(ca):
                which_would_want_to_get += "".join(tmp)
                indices_consistent_with_conditions = False
            if how_many_appear == len(ca):
                apply_conds_this_line.append(ii)
                
        # make sure that all indices on the left of <= also appear on the right of <=
        left_also_on_right = True
        for oi in out_inds:
            if oi not in used_inds:
                left_also_on_right = False
                which_would_want_to_get += oi
                
        if tot_num_func_calls >= min_num_func_calls and indices_consistent_with_conditions == True and left_also_on_right == True:
            break

        # give larger weight to those that are missing
        inds_weights = []
        which_would_want_to_get = list(sorted(set(list(which_would_want_to_get))))
        for ww in range(len(inds)):
            if ww in which_would_want_to_get:
                inds_weights.append(5)
            else:
                inds_weights.append(1)

    num_conditions = len(apply_conds_this_line)

    used_cond_py = []
    for ii in apply_conds_this_line:
        used_cond_py.append(all_cond_code_py[ii])
    used_cond_py = " and ".join(used_cond_py)
    
    code = ""
    if prefix == "":
        use_correct = "correct"
    else:
        use_correct = "correct[\"" + prefix + "\"]"
    
    if len(out_inds) > 0:
        code += use_correct + r" = np.zeros((" + ("3,"*len(out_inds))[:-1] + r"), dtype = complex)" + "\n"
    else:
        code += use_correct + r" = 0.0j" + "\n"
    for i in range(len(used_inds)):
        code += "    "*i + "for " + used_inds[i] + " in range(3):" + "\n"
    
    if num_conditions > 0:
        code += "    "*(len(used_inds)) + "if " + used_cond_py + " and DIAGONAL_" + prefix + ":" + "\n" 
        extra_spaces = "    "
    else:
        code += "    "*(len(used_inds)) + "if " + used_cond_py + "DIAGONAL_" + prefix + ":" + "\n" 
        extra_spaces = "    "
        
    if len(out_inds) > 0:
        code += extra_spaces + "    "*(len(used_inds)) + use_correct + r"[" + ",".join(out_inds) + "] += "
    else:
        code += extra_spaces + "    "*(len(used_inds)) + use_correct + r" += "        
    code += expr_py + "\n"

    if len(out_inds) == 0:
        if prefix != "":
            use_str = "ZZ" + prefix + " <= " + expr
        else:
            use_str ="_ <= " + expr
    else:
        if prefix != "":
            use_str = "ZZ" + prefix + "_" + "".join(out_inds) + " <= " + expr
        else:
            use_str = "_" + "".join(out_inds) + " <= " + expr

    use_str = use_str.strip()
            
    return code, use_str, out_inds, apply_conds_this_line, used_inds

def rnd_term(dimens, terms, inds, used_inds, inds_weights):
    rr = random.randint(0, 10)
    if rr == 0:
        return "3.123", "3.123", used_inds
    elif rr == 1:
        return "2.383j", "2.383j", used_inds
    elif rr == 2:
        return "(4.294 - 5.321j)", "(4.294 - 5.321j)", used_inds
    else:
        tt  = random.sample(terms, 1)[0]
        di  = dimens[tt]
        ret = tt
        #
        not_from_x = False
        if len(tt) > 2:
            if tt[:2] == "ZZ":
                not_from_x = True
        if not_from_x == False:
            ret_py = "x[\"" + tt + "\"]"
        else:
            ret_py = tt.replace("ZZ", "")
            ret_py = "correct[\"" + ret_py + "\"]"
        if di > 0:
            ci = random.choices(inds, weights = inds_weights, k = di)
            ret    = ret    + "_" + "".join(ci)
            ret_py = ret_py + "[" + ",".join(ci) + "]"
            used_inds = list(sorted(set(used_inds + ci)))
        return ret, ret_py, used_inds

def rnd_sum_r(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return expr + " + " + other, expr_py + " + " + other_py, used_inds

def rnd_sum_l(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return other + " + " + expr, other_py + " + " + expr_py, used_inds

def rnd_min_r(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return expr + " - " + other, expr_py + " - " + other_py, used_inds
    
def rnd_min_l(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return other + " - " + expr, other_py + " - " + expr_py, used_inds

def rnd_mul_r(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return expr + " * " + other, expr_py + " * " + other_py, used_inds
    
def rnd_mul_l(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return other + " * " + expr, other_py + " * " + expr_py, used_inds
    
def rnd_div_r(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return expr + " / " + other, expr_py + " / " + other_py, used_inds

def rnd_div_l(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    other, other_py, used_inds = rnd_term(dimens, terms, inds, used_inds, inds_weights)
    return other + " / (" + expr + ")", other_py + " / (" + expr_py + ")", used_inds

def rnd_sign(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    if random.randint(0, 1) == 0:
        return "-" + expr, "(-1.0) * " + expr_py, used_inds
    else:
        return "+" + expr, expr_py, used_inds

def rnd_conj(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    return "#(" + expr + ")", "np.conjugate(" + expr_py + ")", used_inds

def rnd_par(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    return "(" + expr + ")" , "(" + expr_py + ")", used_inds

def rnd_power_2(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    return "(" + expr + ")^2" , "np.power(" + expr_py + ", 2.0)", used_inds

def rnd_real(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    return "Real(" + expr + ")" , "np.real(" + expr_py + ")", used_inds

def rnd_imag(expr, expr_py, dimens, terms, inds, used_inds, inds_weights):
    return "Imag(" + expr + ")" , "np.imag(" + expr_py + ")", used_inds

    
