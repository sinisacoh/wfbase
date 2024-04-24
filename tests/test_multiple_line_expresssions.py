import numpy as np
import sys
sys.path.append('../')
from wfbase import Units

import pytest

from common import *

def test_do_variables_get_updated(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    correct2 = np.zeros((3), dtype = complex)
    for n in range(6):
        for i in range(3):
            correct2[i] +=  correct1[n,i] + x["A"] + x["B"][i]
                                
    use_str = """
    res_ni <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )
    qwe_i <= res_ni + A + B_i
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)
    
def test_do_fancy_arrows_work(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((3, 5), dtype = complex)
    for i in range(3):
        for j in range(5):
            correct1[i,j] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j]
                                
    use_str = """
    res_ni <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )
    qwe_i <= res_ni + A + B_i
    res_ij <<= -A * B_i * C_ij
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

def test_do_fancy_arrows_work2(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
    correct2 = np.zeros((3), dtype = complex)
    for n in range(6):
        for i in range(3):
            correct2[i] +=  correct1[n,i] + x["A"] + x["B"][i]
    for i in range(3):
        for j in range(5):
            for m in range(7):
                for n in range(6):
                    correct1[n,i] +=  correct2[i] * x["E"][m,j,i,n]
                                
    use_str = """
    res_ni <= -A * B_i * C_ij * D_klj  / ( E_mjin * F_jikom )
    qwe_i <= res_ni + A + B_i
    res_ni <+= qwe_i * E_mjin
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

def test_optimizations_recomputation(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
    correct2 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[n,i,k] +=  (-1.0)*x["A"] * (x["B"][i] + x["C"][i,j] + x["D"][k,l,j]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
                                
    use_str = """
    res_nik <= -A * B_i * C_ij * D_klj  / ( E_mjin + F_jikom )
    qwe_nik <= -A * (B_i + C_ij + D_klj)  / ( E_mjin + F_jikom )
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 1 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)
    
def test_optimizations_recomputation2(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] + x["E"][m,j,i,n] + x["F"][j,i,k,o,m]
    correct2 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] + (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
                                
    use_str = """
    res_nik <= -A * B_i * C_ij * D_klj + E_mjin + F_jikom
    qwe_nik <= -A * B_i * C_ij * D_klj + (E_mjin * F_jikom)
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 1 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

def test_optimizations_recomputation2_changed_in_the_middle(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] + x["E"][m,j,i,n] + x["F"][j,i,k,o,m]
    correct2 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[n,i,k] +=  (-1.0)*x["A"] * (x["B"][i] + 2.3 + 0.4j) * x["C"][i,j] * x["D"][k,l,j] + (x["E"][m,j,i,n] * x["F"][j,i,k,o,m])
                                
    use_str = """
    res_nik <= -A * B_i * C_ij * D_klj + E_mjin + F_jikom
    B_i <<= B_i + 2.3 + 0.4j
    qwe_nik <= -A * B_i * C_ij * D_klj + (E_mjin * F_jikom)
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            # Now this should be zero because you changed B in the second line in evaluate
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

    
def test_optimizations_recomputation2_changed_in_the_middle_einsum(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] * x["E"][m,j,i,n] * x["F"][j,i,k,o,m]
    correct2 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[n,i,k] +=  (-1.0)*x["A"] * (x["B"][i] + 2.3 + 0.4j) * x["C"][i,j] * x["D"][k,l,j] * x["E"][m,j,i,n] * x["F"][j,i,k,o,m]
                                
    use_str = """
    res_nik <= -A * B_i * C_ij * D_klj * E_mjin * F_jikom
    B_i <<= B_i + 2.3 + 0.4j
    qwe_nik <= -A * B_i * C_ij * D_klj * E_mjin * F_jikom
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            # Now this should be zero because you changed B in the second line in evaluate
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

    
def test_change_indices_in_broadcast(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
    correct2 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[n,i,k] +=  (-1.0)*x["A"] * (x["B"][i] + x["C"][i,j] + x["D"][k,l,j]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
                                
    use_str = """
    res_nik <= -A *  B_i * C_ij * D_klj   / ( E_mjin + F_jikom )
    qwe_tzk <= -A * (B_z + C_zr + D_klr)  / ( E_mrzt + F_rzkym )
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 1)
    
def test_change_indices_in_broadcast_with_condition(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["E"][m,j,i,n]) < 0.0:
                                    correct1[o,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
    correct2 = np.zeros((3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[o,k] +=  (-1.0)*x["A"] * (x["B"][i] + x["C"][i,j] + x["D"][k,l,j]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
                                
    use_str = """
    res_ok <= -A *  B_i * C_ij * D_klj   / ( E_mjin + F_jikom )
    qwe_ok <= -A * (B_c + C_cb + D_klb)  / ( E_abcd + F_bckoa )
    """
    condition = "E_mjin < 0"
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"], condition)
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            # this should not be zero because, eventhough indices are the same, the condition is different
            num_opt_rec["broadcast_jumbled_indices"] == 0)

def test_change_indices_in_broadcast_with_condition_atsymbol(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                if np.real(x["E"][m,j,i,n]) < 0.3:
                                    correct1[o,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
    correct2 = np.zeros((3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct2[o,k] +=  (-1.0)*x["A"] * (x["B"][i] + x["C"][i,j] + x["D"][k,l,j]) / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
                                
    use_str = """
    r~es_ok <= -A *  B_i * C_ij * D_klj   / ( E_mjin + F_jikom )
    q~we_ok <= -A * (B_c + C_cb + D_klb)  / ( E_abcd + F_bckoa )
    """
    x.new("cond~ition", 0.3)
    condition = "E_mjin < cond~ition"
    custom, num_opt_rec = five_evaluate(x, use_str, ["r~es", "q~we"], condition)
    assert (abs((custom["r~es"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["q~we"] - correct2) / correct2) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            # this should not be zero because, eventhough indices are the same, the condition is different
            num_opt_rec["broadcast_jumbled_indices"] == 0)

def test_change_indices_in_einsum(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] * x["E"][m,j,i,n] * x["F"][j,i,k,o,m]
                                
    use_str = """
    res_nik <= -A *  B_i * C_ij * D_klj * E_mjin * F_jikom
    qwe_nik <= -A *  B_i * C_ij * D_klj * E_mjin * F_jikom
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct1) / correct1) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 1 and \
            num_opt_rec["einsum_jumbled_indices"]    == 0 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

    
def test_change_indices_in_einsum2(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] * x["E"][m,j,i,n] * x["F"][j,i,k,o,m]
                                
    use_str = """
    res_nik <= -A *  B_i * C_ij * D_klj * E_mjin * F_jikom
    qwe_uwk <= -A *  B_w * C_wt * D_klt * E_ytwu * F_twkoy
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct1) / correct1) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 1 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 0)

    
def test_change_indices_in_einsum3(comp_test, threshold):
    x = comp_test

    correct1 = np.zeros((6, 3, 4), dtype = complex)
    for i in range(3):
        for j in range(5):
            for k in range(4):
                for l in range(2):
                    for m in range(7):
                        for n in range(6):
                            for o in range(3):
                                correct1[n,i,k] +=  (-1.0)*x["A"] * x["B"][i] * x["C"][i,j] * x["D"][k,l,j] / (x["E"][m,j,i,n] + x["F"][j,i,k,o,m])
                                
    use_str = """
    res_nik <= -A *  B_i * C_ij * D_klj / ( E_mjin + F_jikom )
    qwe_yzw <= -A *  B_z * C_zj * D_wpj / ( E_hjzy + F_jzwoh )
    """
    custom, num_opt_rec = five_evaluate(x, use_str, ["res", "qwe"])    
    assert (abs((custom["res"] - correct1) / correct1) < threshold).all()
    assert (abs((custom["qwe"] - correct1) / correct1) < threshold).all()
    assert (num_opt_rec["einsum_simple"]             == 0 and \
            num_opt_rec["einsum_jumbled_indices"]    == 1 and \
            num_opt_rec["broadcast_simple"]          == 0 and \
            num_opt_rec["broadcast_jumbled_indices"] == 1)
    
