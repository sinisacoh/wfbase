import numpy as np
import sys
sys.path.append('../')
from wfbase import Units
import re

import pytest

from common import *

def test_two_divisions(comp_test, threshold):
    x = comp_test    
    with pytest.raises(ValueError) as exc_info:
        use_str = "_i <= B_i - A / B_i / C_ij"
        three_evaluate(x, use_str)
    error_testing(exc_info, "Not allowing terms like A / B / C or similar, as it might be ambiguous. Use parantheses to clarify what you mean.")

def test_order_division(comp_test, threshold):
    x = comp_test    
    with pytest.raises(ValueError) as exc_info:
        use_str = "_i <= A + A / B_i * C_ij"
        three_evaluate(x, use_str)
    error_testing(exc_info, "Not allowing terms like A / B * C or similar, as it might be ambiguous. Use parantheses to clarify what you mean.")
        
def test_two_powers(comp_test, threshold):
    x = comp_test    
    with pytest.raises(ValueError) as exc_info:
        use_str = "_ij <= B_i ^ A ^ A + C_ij"
        three_evaluate(x, use_str)
    error_testing(exc_info, "Don't allow things like A^B^C. Use parentheses instead. For example (A^B)^C.")

def test_missing_indices(comp_test, threshold):
    x = comp_test    
    with pytest.raises(ValueError) as exc_info:
        use_str = "_ij <= B_i + C_ii"
        three_evaluate(x, use_str)
    error_testing(exc_info, "Index \"j\" appears on the left of assignment operator (<=, <<=, <+=) but this index does not appear on the right.")

def test_missing_indices_2(comp_test, threshold):
    x = comp_test    
    with pytest.raises(ValueError) as exc_info:
        use_str = "_ijm <= B_i + C_ij"
        three_evaluate(x, use_str)
    error_testing(exc_info, "Index \"m\" appears on the left of assignment operator (<=, <<=, <+=) but this index does not appear on the right.")

def test_not_allowed_names(comp_test, threshold):
    x = comp_test
    x.new("ABC~DEF", "1.0 * eV")

    with pytest.raises(ValueError) as exc_info:
        x.new("ABCDEF", "1.0 * eV")
    error_testing(exc_info, "Quantity \"ABCDEF\" does not already exists, but a similarly named quantity (ignoring the tilde symbol, ~) does exist! This is not allowed as these are too similar. Pick a more unique name.")
    
    with pytest.raises(ValueError) as exc_info:
        x.new("A~BCDEF", "1.0 * eV")
    error_testing(exc_info, "Quantity \"A~BCDEF\" does not already exists, but a similarly named quantity (ignoring the tilde symbol, ~) does exist! This is not allowed as these are too similar. Pick a more unique name.")


    
