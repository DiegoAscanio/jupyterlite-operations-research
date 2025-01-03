import pytest
import numpy as np

from simplex_bounded_variables import\
    _build_initial_tableau,\
    _find_k_who_enters,\
    _compute_c_hat_J,\
    _compute_π,\
    _compute_γ_1_for_non_basic_entering_from_its_lower_bound,\
    _compute_γ_2_for_non_basic_entering_from_its_lower_bound,\
    _compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds,\
    _compute_Δ_k_for_non_basic_entering_from_its_lower_bound

parameters_and_expected_results_for_tests = {
    'A': np.array([ # 1st test
        [2, 1,  1, 1, 0],
        [1, 1, -1, 0, 1]
    ]),
    'b': np.array([
        10,
        4
    ]),
    'c': np.array([-2, -4, -1, 0, 0]), # c
    'I': [3, 4],
    'J_1': [0, 1, 2],
    'J_2': [],
    'lower_bounds': np.array([0, 0, 1, 0, 0]), 
    'upper_bounds': np.array([4, 6, 4, np.inf, np.inf]),
    'expected_tableau': np.array([
        [2, 4,  1, 0, 0, -1],  
        [2, 1,  1, 1, 0,  9], 
        [1, 1, -1, 0, 1,  5]  
    ]),

    'expected_k_to_enter_iteration_1': 1, # 2nd test
    'expected_c_hat_k_iteration_1': 4,
    'expected_proceed_iteration_1': True,

    'J_1_iteration_3': [0, 4], # 3rd test
    'J_2_iteration_3': [1],  
    'c_hat_J_1_iteration_3': np.array([3, 1]),  
    'c_hat_J_2_iteration_3': np.array([5]),  
    'expected_k_to_enter_iteration_3': 0,
    'expected_c_hat_k_iteration_3': 3,
    'expected_proceed_iteration_3': True,
}

test_build_initial_tableau_inputs_and_expected_results = [
    'A', 'b', 'c', 'I', 'J_1', 'J_2', 'lower_bounds', 'upper_bounds', 'expected_tableau'
]
argnames = ', '.join(test_build_initial_tableau_inputs_and_expected_results)
argvalues = [tuple(
    parameters_and_expected_results_for_tests[key] for key in test_build_initial_tableau_inputs_and_expected_results
)]

@pytest.mark.parametrize(argnames, argvalues)
def test_build_initial_tableau(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds, expected_tableau):
    tableau = _build_initial_tableau(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds)
    assert np.allclose(tableau, expected_tableau)

test_find_k_who_enters_basis_iteration_1_inputs_and_expected_results = list(
    set(test_build_initial_tableau_inputs_and_expected_results) - 
    set(
        ['b', 'lower_bounds', 'upper_bounds', 'expected_tableau']
    ) | 
    set(
        ['expected_k_to_enter_iteration_1', 'expected_c_hat_k_iteration_1', 'expected_proceed_iteration_1']
    )
)
argnames = ', '.join(test_find_k_who_enters_basis_iteration_1_inputs_and_expected_results)
argvalues = [tuple(
    parameters_and_expected_results_for_tests[key] for key in test_find_k_who_enters_basis_iteration_1_inputs_and_expected_results
)]

@pytest.mark.parametrize(argnames, argvalues)
def test_find_k_who_enters_basis_iteration_1(A, c, I, J_1, J_2, expected_k_to_enter_iteration_1, expected_c_hat_k_iteration_1, expected_proceed_iteration_1):
    π = _compute_π(A, c, I)
    c_hat_J_1 = _compute_c_hat_J(π, A, c, J_1)
    c_hat_J_2 = _compute_c_hat_J(π, A, c, J_2)
    k_to_enter_iteration_1, c_hat_k_iteration_1, proceed_iteration_1 = _find_k_who_enters(c_hat_J_1, c_hat_J_2, J_1, J_2)
    assert k_to_enter_iteration_1 == expected_k_to_enter_iteration_1 and\
           c_hat_k_iteration_1 == expected_c_hat_k_iteration_1 and\
           proceed_iteration_1 == expected_proceed_iteration_1

test_find_k_who_enters_basis_iteration_3_inputs_and_expected_results = [
    'c_hat_J_1_iteration_3',
    'c_hat_J_2_iteration_3',
    'J_1_iteration_3',
    'J_2_iteration_3',
    'expected_k_to_enter_iteration_3',
    'expected_c_hat_k_iteration_3',
    'expected_proceed_iteration_3'
]
argnames = ', '.join(test_find_k_who_enters_basis_iteration_3_inputs_and_expected_results)
argvalues = [tuple(
    parameters_and_expected_results_for_tests[key] for key in test_find_k_who_enters_basis_iteration_3_inputs_and_expected_results
)]

@pytest.mark.parametrize(argnames, argvalues)
def test_find_k_who_enters_basis_iteration_3(c_hat_J_1_iteration_3, c_hat_J_2_iteration_3, J_1_iteration_3, J_2_iteration_3, expected_k_to_enter_iteration_3, expected_c_hat_k_iteration_3, expected_proceed_iteration_3):
    J_1, J_2 = J_1_iteration_3, J_2_iteration_3
    c_hat_J_1, c_hat_J_2 = c_hat_J_1_iteration_3, c_hat_J_2_iteration_3
    k_to_enter_iteration_3, c_hat_k_iteration_3, proceed_iteration_3 = _find_k_who_enters(c_hat_J_1, c_hat_J_2, J_1, J_2)
    assert k_to_enter_iteration_3 == expected_k_to_enter_iteration_3 and\
           c_hat_k_iteration_3 == expected_c_hat_k_iteration_3 and\
           proceed_iteration_3 == expected_proceed_iteration_3

def test_compute_γ_1_for_non_basic_entering_from_its_lower_bound():
    assert _compute_γ_1_for_non_basic_entering_from_its_lower_bound(
        [3, 4],
        np.array([
            1,
            1
        ]),
        np.array([
            9,
            5
        ]),
        np.array([0, 0, 1, 0, 0])
    ) == (1, 5.0)

def test_compute_γ_2_for_non_basic_entering_from_its_lower_bound_y_k_geq_0():
    assert _compute_γ_2_for_non_basic_entering_from_its_lower_bound(
        [3, 4],
        np.array([
            1,
            1
        ]),
        np.array([
            9,
            5
        ]),
        np.array([4, 6, 4, np.inf, np.inf])
    ) == (0, np.inf)

def test_compute_γ_2_for_non_basic_entering_from_its_lower_bound_any_y_k_lt_0():
    assert _compute_γ_2_for_non_basic_entering_from_its_lower_bound(
        [3, 1],
        np.array([
             2,
            -1
        ]),
        np.array([
            4,
            5
        ]),
        np.array([4, 6, 4, np.inf, np.inf])
    ) == (1, 1)

def test_compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds():
    assert _compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds(
        k = np.intp(2),
        lower_bounds = np.array([0, 0, 0, 0, 0]),
        upper_bounds = np.array([7, 10, 1, 5, 3])
    ) == (None, 1)

def test_compute_Δ_k_for_non_basic_entering_from_its_lower_bound():
    k = np.intp(3)
    y_k = np.array([
        -1,
         2
    ])
    x_I = np.array([
        5,
        9
    ])
    lower_bounds = np.array([0, 0, 0, 0, 0])
    upper_bounds = np.array([7, 10, 1, 5, 3])
    assert _compute_Δ_k_for_non_basic_entering_from_its_lower_bound(
        I = [0, 1],
        k = k,
        y_k = y_k,
        x_I = x_I,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds
    ) == (0, 2.0)

'''
    As tests for computing γ_1, γ_2, γ_3 and Δ_k are similar for the
    case of entering from the upper bound, we will not repeat them 
    here, we'll only make adaptations at the original source code 
    for the necessary functions at simplex_bounded_variables.py
    to make everything work as expected.
'''
