from typing import Any, Tuple, List
import numpy as np
from copy import deepcopy
import pdb

from utils import lexicographic_negative, lexicographic_positive

_solution_types_map = {
    1: 'Optimal unique solution',
    2: 'Optimal multiple solutions',
    3: 'Unbounded solution',
    -1: 'Infeasible solution',
    -2: 'Entered in loop'
}

def _compute_x_from_x_I(n, x_I, I, J_1, J_2, lower_bounds, upper_bounds) -> np.ndarray:
    x = np.zeros(n)
    x[I] = x_I
    x[J_1] = lower_bounds[J_1]
    x[J_2] = upper_bounds[J_2]
    return x

def _compute_A_I(A: np.ndarray, I: list) -> np.ndarray:
    A_I = A[:, I]
    return A_I

def _compute_J(n: int, I: list) -> list:
    J = np.setdiff1d(np.arange(n), I)
    return list(J)

def _compute_A_J(A: np.ndarray, J: list) -> np.ndarray:
    A_J = A[:, J]
    return A_J

def _compute_π(A: np.ndarray, c: np.ndarray, I: list) -> np.ndarray:
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    c_I = c[I]
    π = c_I.dot(A_I_inv)
    return π

def _compute_X_I(
    A: np.ndarray,
    b: np.ndarray,
    I: list,
    J_1: list,
    J_2: list,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> np.ndarray:
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    A_J_1 = _compute_A_J(A, J_1)
    A_J_2 = _compute_A_J(A, J_2)
    x_J_1 = lower_bounds[J_1]
    x_J_2 = upper_bounds[J_2]
    X_I = A_I_inv.dot(b) - A_I_inv.dot(A_J_1).dot(x_J_1) - A_I_inv.dot(A_J_2).dot(x_J_2)
    return X_I

def _compute_A_I_x_I_π_and_z_0(
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    I: list,
    J_1 : list,
    J_2: list,
    lower_bounds : np.ndarray,
    upper_bounds : np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # A_I should be a submatrix from A with columns indexed (and sorted) by I
    A_I = _compute_A_I(A, I)
    x_I = _compute_X_I(A, b, I, J_1, J_2, lower_bounds, upper_bounds)
    π = _compute_π(A, c, I)
    c_hat_J_1 = _compute_c_hat_J(π, A, c, J_1)
    c_hat_J_2 = _compute_c_hat_J(π, A, c, J_2)
    z_0 = π.dot(b) - c_hat_J_1.dot(lower_bounds[J_1]) - c_hat_J_2.dot(upper_bounds[J_2])
    return A_I, x_I, π, z_0

def _compute_c_hat_J(π: np.ndarray, A: np.ndarray, c: np.ndarray, J: list) -> np.ndarray:
    A_J = A[:, J]
    c_hat_J = π.dot(A_J) - c[J]
    return c_hat_J

def _find_k_who_enters(
    c_hat_J_1: np.ndarray,
    c_hat_J_2: np.ndarray,
    J_1: list,
    J_2: list
) -> Tuple[np.intp, np.float64, list, np.bool_]:
    """
    Finds the index of the variable that enters the basis.
    Args:
        c_hat_J_1: the reduced costs of the non-basic variables at their lower bounds.
        c_hat_J_2: the reduced costs of the non-basic variables at their upper bounds.
        J_1: the set of indices of the non-basic variables that are at their lower bound.
        J_2: the set of indices of the non-basic variables that are at their upper bound.
    Returns:
        k: the index of the variable that enters the basis.
        c_hat_k: the reduced cost of the variable that enters the basis.
        J_k: the set where the variable k belongs to.
        proceed: whether the algorithm should proceed.
    """
    # initialize c_hat_k
    c_hat_k = np.max(
        [np.max(c_hat_J_1), -np.max(c_hat_J_2)]
    ) if len(c_hat_J_1) > 0 and len(c_hat_J_2) > 0 else np.max(
        c_hat_J_1
    ) if len(c_hat_J_1) > 0 else -np.max(c_hat_J_2)
    k = 0 # just to initialize
    # an aux_c_hat_J_1 with -infinities where c_hat_J_1 is leq 0
    aux_c_hat_J_1 = c_hat_J_1[np.where(c_hat_J_1 > 0)]
    # an aux_c_hat_J_2 with infinities where c_hat_J_2 is geq 0
    aux_c_hat_J_2 = c_hat_J_2[np.where(c_hat_J_2 < 0)]

    # ok, let's remake our logic

    if len(aux_c_hat_J_1) > 0 and len(aux_c_hat_J_2) > 0:
        k_1 = np.argmax(
            aux_c_hat_J_1
        )
        # then update k_1 to reflect the actual index in J_1
        # properly
        k_1 = list(c_hat_J_1).index(aux_c_hat_J_1[k_1])

        k_2 = np.argmax(
            aux_c_hat_J_2
        )
        # then update k_2 to reflect the actual index in J_2
        # properly
        k_2 = list(c_hat_J_2).index(aux_c_hat_J_2[k_2])

        # these aux_c_hat_J were created as loop prevention measures
        # then the algorithm should follow as originally intended
        candidate_indices = [k_1, k_2]
        candidate_values = [c_hat_J_1[k_1], -c_hat_J_2[k_2]]
        better_improvement = np.max(candidate_values)
        better_index = candidate_values.index(better_improvement)
        k = candidate_indices[better_index]
        c_hat_k = better_improvement
    elif len(aux_c_hat_J_1) > 0:
        # if J_2 is empty we should only consider the reduced costs of J_1
        k = np.argmax(
            aux_c_hat_J_1
        )
        # then update k to reflect the actual index in J_1
        # properly
        k = list(c_hat_J_1).index(aux_c_hat_J_1[k])
        c_hat_k = c_hat_J_1[k]
    elif len(aux_c_hat_J_2) > 0:
        # if J_1 is empty we should only consider the reduced costs of J_2
        k = np.argmax(
            aux_c_hat_J_2
        )
        # then update k to reflect the actual index in J_2
        # properly
        k = list(c_hat_J_2).index(aux_c_hat_J_2[k])
        c_hat_k = -c_hat_J_2[k]
    J_k = J_1 if c_hat_k in c_hat_J_1 else J_2
    proceed = c_hat_k > 0
    return J_k[k], c_hat_k, J_k, proceed

def _compute_γ_1_for_non_basic_entering_from_its_lower_bound(
    I : list,
    y_k : np.ndarray,
    x_I : np.ndarray,
    lower_bounds : np.ndarray,
) -> Tuple [np.intp, np.float64]:
    """
        Computes γ_1 for non-basic variable entering from its lower bound.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
               to its lower bound.
            γ_1: the value of γ_1.
    """
    γ_1_candidate_values = np.where(y_k > 0, (x_I - lower_bounds[I]) / y_k, np.inf)
    r = np.argmin(γ_1_candidate_values)
    γ_1 = γ_1_candidate_values[r]
    return r, γ_1

def _compute_γ_2_for_non_basic_entering_from_its_lower_bound(
    I : list,
    y_k : np.ndarray,
    x_I : np.ndarray,
    upper_bounds : np.ndarray,
) -> Tuple [np.intp, np.float64]:
    """
        Computes γ_2 for non-basic variable entering from its lower bound.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
               to its lower bound.
            γ_2: the value of γ_2.
    """
    γ_2_candidate_values = np.where(y_k < 0, (upper_bounds[I] - x_I) / -y_k, np.inf)
    r = np.argmin(γ_2_candidate_values)
    γ_2 = γ_2_candidate_values[r]
    return r, γ_2

def _compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds(
    k: np.intp,
    upper_bounds: np.ndarray,
    lower_bounds: np.ndarray,
) -> Tuple [ None, np.float64 ]:
    """
        Computes γ_3 for non-basic variable entering either from its 
        lower or upper bounds, as the same function is used for both
        cases.
        Args:
            k: the index of the non-basic variable that enters the basis.
            upper_bounds: the upper bounds of the variables.
            lower_bounds: the lower bounds of the variables.
        Returns:
            r: None.
            γ_3: the value of γ_3.
    """
    γ_3 = upper_bounds[k] - lower_bounds[k]
    return None, γ_3

def _compute_Δ_k_for_non_basic_entering_from_its_lower_bound(
    I: List[int],
    k: np.intp,
    y_k: np.ndarray,
    x_I: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> Tuple [ int | None, np.float64, np.bool_, int | None ]:
    """
        Computes Δ_k for non-basic variable entering from its lower bound
        that corresponds to the increment that x_k will have when entering
        the basis. We also return the index of the basic variable that
        leaves the basis - if appropriate.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
            upper_bounds: the upper bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
            Δ_k: the value of Δ_k.
            proceed: if Δ_k is less than infinity.
            bound: the bound to which the basic variable leaves the basis.
                   0 for lower bound, 1 for upper bound.
                   None if Δ_k is γ_3
    """
    r_γ_1, γ_1 = _compute_γ_1_for_non_basic_entering_from_its_lower_bound(I, y_k, x_I, lower_bounds)
    r_γ_2, γ_2 = _compute_γ_2_for_non_basic_entering_from_its_lower_bound(I, y_k, x_I, upper_bounds)
    r_γ_3, γ_3 = _compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds(k, upper_bounds, lower_bounds)
    candidate_values = np.array([γ_1, γ_2, γ_3])
    candidate_indices = [ r_γ_1, r_γ_2, r_γ_3 ]
    Δ_k = np.min(candidate_values)
    r: int = candidate_indices[
        list(candidate_values).index(Δ_k)
    ]
    bound = 0 if Δ_k == γ_1 else 1 if Δ_k == γ_2 else None
    return r, Δ_k, Δ_k < np.inf, bound

def _compute_γ_1_for_non_basic_entering_from_its_upper_bound(
    I : list,
    y_k : np.ndarray,
    x_I : np.ndarray,
    lower_bounds : np.ndarray,
) -> Tuple [np.intp, np.float64]:
    """
        Computes γ_1 for non-basic variable entering from its upper bound.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
               to its lower bound.
            γ_1: the value of γ_1.
    """
    γ_1_candidate_values = np.where(y_k < 0, (x_I - lower_bounds[I]) / -y_k, np.inf)
    r = np.argmin(γ_1_candidate_values)
    γ_1 = γ_1_candidate_values[r]
    return r, γ_1

def _compute_γ_2_for_non_basic_entering_from_its_upper_bound(
    I : list,
    y_k : np.ndarray,
    x_I : np.ndarray,
    upper_bounds : np.ndarray,
) -> Tuple [np.intp, np.float64]:
    """
        Computes γ_2 for non-basic variable entering from its lower bound.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
               to its lower bound.
            γ_2: the value of γ_2.
    """
    γ_2_candidate_values = np.where(y_k > 0, (upper_bounds[I] - x_I) / y_k, np.inf)
    r = np.argmin(γ_2_candidate_values)
    γ_2 = γ_2_candidate_values[r]
    return r, γ_2

def _compute_Δ_k_for_non_basic_entering_from_its_upper_bound(
    I: List[int],
    k: np.intp,
    y_k: np.ndarray,
    x_I: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray
) -> Tuple [ int | None, np.float64, np.bool_, int | None ]:
    """
        Computes Δ_k for non-basic variable entering from its upper bound
        ---------------------------- k ∈ J_2 ----------------------------
        that corresponds to the increment that x_k will have when entering
        the basis. We also return the index of the basic variable that
        leaves the basis - if appropriate.
        Args:
            I: the set of indices of the basic variables.
            y_k: the k-th column of the inverse of the basis matrix.
            x_I: the values of the basic variables.
            lower_bounds: the lower bounds of the variables.
            upper_bounds: the upper bounds of the variables.
        Returns:
            r: the index of the basic variable that leaves the basis
               to its lower bound - if appropriate.
            Δ_k: the value of Δ_k.
            proceed: if Δ_k is less than infinity.
            bound: the bound to which the basic variable leaves the basis.
            1 for upper bound, 0 for lower bound, None if Δ_k is γ_3
    """
    r_γ_1, γ_1 = _compute_γ_1_for_non_basic_entering_from_its_upper_bound(I, y_k, x_I, lower_bounds)
    r_γ_2, γ_2 = _compute_γ_2_for_non_basic_entering_from_its_upper_bound(I, y_k, x_I, upper_bounds)
    r_γ_3, γ_3 = _compute_γ_3_for_non_basic_entering_either_from_its_lower_or_upper_bounds(k, upper_bounds, lower_bounds)
    candidate_values = np.array([γ_1, γ_2, γ_3])
    candidate_indices = [ r_γ_1, r_γ_2, r_γ_3 ]
    Δ_k = np.min(candidate_values)
    r: int = candidate_indices[
        list(candidate_values).index(Δ_k)
    ]
    bound = 0 if Δ_k == γ_1 else 1 if Δ_k == γ_2 else None
    return r, Δ_k, Δ_k < np.inf, bound

def _basic_variables_degenerate_at_their_lower_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    lower_bounds : np.ndarray
) -> list:
    """
        Returns the basic_variables_degenerate_at_their_lower_bounds set
        with the indices of the basic variables that are degenerate at
        their lower bounds.
        Args:
            I: the set of indices of the basic variables.
            A: the matrix of the linear system.
            b: the right-hand side vector of the linear system.
            lower_bounds: the lower bounds of the variables.
        Returns
            degenerate_in_lower_bounds_indices: the indices of the basic
            variables that are degenerate at their lower bounds.
    """
    basic_variables_values = np.linalg.solve(A[:, I], b)
    degenerate_in_lower_bounds_indices, *_ = np.where(
        np.isclose(basic_variables_values, lower_bounds[I])
    )
    return list(degenerate_in_lower_bounds_indices)

def _basic_variables_degenerate_at_their_upper_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    upper_bounds : np.ndarray
) -> list:
    """
        Returns the basic_variables_degenerate_at_their_upper_bounds set
        with the indices of the basic variables that are degenerate at
        their upper bounds.
        Args:
            I: the set of indices of the basic variables.
            A: the matrix of the linear system.
            b: the right-hand side vector of the linear system.
            upper_bounds: the upper bounds of the variables.
        Returns:
            degenerate_in_upper_bounds_indices: the indices of the basic
            variables that are deg
    """
    basic_variables_values = np.linalg.solve(A[:, I], b)
    degenerate_in_upper_bounds_indices, *_ = np.where(
        np.isclose(basic_variables_values, upper_bounds[I])
    )
    return list(degenerate_in_upper_bounds_indices)

def _build_s_1_set_x_k_enters_from_lower_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    upper_bounds : np.ndarray,
    k : np.intp
) -> list:
    '''
        S_1 is the set of basic variables indices i for which y_ik < 0
        and that are at their upper bounds
        Args:
            I : basic variables set
            A : constraints matrix
            B : right-side matrix
            upper_bounds : upper_bounds for variables
            k : entering variable index
        Returns:
            S_1 : list of basic variables i that are at their upper bound where y_ik < 0
    '''
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    y_k = A_I_inv @ A[:, k]
    degenerate_at_upper_bounds = _basic_variables_degenerate_at_their_upper_bounds(I, A, b, upper_bounds)
    print(degenerate_at_upper_bounds)
    print(y_k)
    s_1 = []
    for d in degenerate_at_upper_bounds:
        if y_k[d] < 0:
            s_1.append(d)
    return s_1

def _build_s_2_set_x_k_enters_from_lower_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    lower_bounds : np.ndarray,
    k : np.intp
) -> list:
    '''
        S_2 is the set of basic variables indices i for which y_ik < 0
        and that are at their lower bounds
        Args:
            I : basic variables set
            A : constraints matrix
            B : right-side matrix
            lower_bounds : lower_bounds for variables
            k : entering variable index
        Returns:
            S_2 : list of basic variables i that are at their lower bounds where y_ik > 0
    '''
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    y_k = A_I_inv @ A[:, k]
    degenerate_at_lower_bounds = _basic_variables_degenerate_at_their_lower_bounds(I, A, b, lower_bounds)
    print(degenerate_at_lower_bounds)
    print(y_k)
    s_2 = []
    for d in degenerate_at_lower_bounds:
        if y_k[d] > 0:
            s_2.append(d)
    return s_2

def _build_s_1_set_x_k_enters_from_upper_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    lower_bounds : np.ndarray,
    k : np.intp
) -> list:
    '''
        S_1 is the set of basic variables indices i for which y_ik < 0
        and that are at their lower bounds
        Args:
            I : basic variables set
            A : constraints matrix
            B : right-side matrix
            lower_bounds : lower_bounds for variables
            k : entering variable index
        Returns:
            S_1 : list of basic variables i that are at their lower bounds where y_ik < 0
    '''
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    y_k = A_I_inv @ A[:, k]
    degenerate_at_lower_bounds = _basic_variables_degenerate_at_their_lower_bounds(I, A, b, lower_bounds)
    s_1 = []
    for d in degenerate_at_lower_bounds:
        if y_ki[d] < 0:
            s_1.append(d)
    return s_1

def _build_s_2_set_x_k_enters_from_upper_bounds(
    I : list,
    A : np.ndarray,
    b : np.ndarray,
    upper_bounds : np.ndarray,
    k : np.intp
) -> list:
    '''
        S_2 is the set of basic variables indices i for which y_ik > 0
        and that are at their upper bounds
        Args:
            I : basic variables set
            A : constraints matrix
            B : right-side matrix
            upper_bounds : upper_bounds for variables
            k : entering variable index
        Returns:
            S_2 : list of basic variables i that are at their upper bounds where y_ik > 0
    '''
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    y_k = A_I_inv @ A[:, k]
    degenerate_at_upper_bounds = _basic_variables_degenerate_at_their_upper_bounds(I, A, b, upper_bounds)
    s_2 = []
    for d in degenerate_at_upper_bounds:
        if y_ki > 0:
            s_2.append(d)
    return s_2

def _is_strongly_feasible(
    A: np.ndarray,
    b: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    I: list
) -> bool:
    """
        Evaluate if the partition I, J_1, J_2 is strongly feasible.
        Args:
            A: the matrix of the linear system.
            b: the right-hand side vector of the linear system.
            lower_bounds: the lower bounds of the variables.
            upper_bounds: the upper bounds of the variables.
            I: the set of indices of the basic variables.
            J_1: the set of indices of the non-basic variables that are at their
                 lower bound.
            J_2: the set of indices of the non-basic variables that are at their
                 upper bound.
        Returns:
            strongly_feasible: whether the partition I, J_1, J_2 is strongly feasible.
            (True or False)
    """
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    degenerate_in_lower_bounds_indices = _basic_variables_degenerate_at_their_lower_bounds(I, A, b, lower_bounds)
    degenerate_in_upper_bounds_indices = _basic_variables_degenerate_at_their_upper_bounds(I, A, b, upper_bounds)
    # the partition I, J_1, J_2 is strongly feasible if and only if
    # degenerate basic varibles at their lower bounds are strictly
    # lexico-positive
    strongly_feasible = np.allclose(
        list(
            map(
                lambda x: lexicographic_positive(x),
                A_I_inv[degenerate_in_lower_bounds_indices, :]
            )
        ), True
    )
    # and if the degenerate basic variables at their upper bounds are
    # strictly lexico-negative
    strongly_feasible = strongly_feasible and np.allclose(
        list(
            map(
                lambda x: lexicographic_negative(x),
                A_I_inv[degenerate_in_upper_bounds_indices, :]
            )
        ), True
    )
    return strongly_feasible

def _find_r_to_leave_for_k_entering_from_lower_bound_cycle_proof(
    S_1 : list,
    S_2 : list,
    I : list,
    y_k : np.ndarray,
    A_I_inv : np.ndarray
) -> int | None:
    degenerate_basic_variables = list(set(S_1) | set(S_2))
    # we need to handle it as list to sort appropriately
    r_candidates_to_leave = [
        tuple(A_I_inv [d, :] / y_k[d]) for d in degenerate_basic_variables
    ]

    # we make a copy of r_candidates_to_leave - a list of tuples
    lexicographic_rule = r_candidates_to_leave.copy()
    # and sort the copy, as tuple sorting in python is lexicographic
    lexicographic_rule.sort()

    # now we're interested at the minimum value of the lexicographically ordered
    # so the first element of lexicographic_rule is the one we're interested at
    lexicographically_minimum_row = lexicographic_rule[0]

    # the index is the index of r_candidates_to_leave that contains the minimum
    # row
    # now we find the row in A_I_inv that contains the lexicographically minimum
    A_I_inv_aux = np.copy(A_I_inv)
    for d in degenerate_basic_variables:
        A_I_inv_aux [d, :] = A_I_inv_aux [d, :] / y_k[d]
    # non eficcient search for the r-index to leave as np.where failed :/
    for r_index, row in enumerate(A_I_inv_aux):
        if tuple(row) == lexicographically_minimum_row:
            break
    return r_index
def _find_r_to_leave_for_k_entering_from_upper_bound_cycle_proof(
    S_1 : list,
    S_2 : list,
    y_k : np.ndarray,
    A_I_inv : np.ndarray
) -> int | None:
    degenerate_basic_variables = list(set(S_1) | set(S_2))
    # we need to handle it as list to sort appropriately
    r_candidates_to_leave = [
        tuple(A_I_inv [d, :] / y_k[d]) for d in degenerate_basic_variables
    ]

    # we make a copy of r_candidates_to_leave - a list of tuples
    lexicographic_rule = r_candidates_to_leave.copy()
    # and sort the copy, as tuple sorting in python is lexicographic
    lexicographic_rule.sort()

    # now we're interested at the maximum value of the lexicographically ordered
    # so the last element of lexicographic_rule is the one we're interested at
    lexicographically_maximum_row = lexicographic_rule[-1]

    # the index is the index of r_candidates_to_leave that contains the maximum
    # row
    # now we find the row in A_I_inv that contains the lexicographically minimum
    A_I_inv_aux = np.copy(A_I_inv)

    # non eficcient search for the r-index to leave as np.where failed :/
    for r_index, row in enumerate(A_I_inv_aux):
        if tuple(row) == lexicographically_maximum_row:
            break
    return r_index

def _remove_from_lower_bounds_and_add_to_upper_bounds(
    lower_bounds_set: list, upper_bounds_set: list, k: np.intp
) -> tuple[list, list]:
    """
    Updates the sets of non-basic variables at their lower and upper bounds.
    Args:
        lower_bounds_set: the set of indices of the non-basic variables that are at
            their lower bound.
        upper_bounds_set: the set of indices of the non-basic variables that are at
            their upper bound.
        k: the variable that should leave lower_bounds_set and enter 
            upper_bounds_set.
    Returns:
        lower_bounds_set: the updated set of indices of the non-basic variables at
            their lower bounds.
        upper_bounds_set: the updated set of indices of the non-basic variables at
            their upper bounds.
    """
    leaving_variable = k
    lower_bounds_set = list(set(lower_bounds_set) - {leaving_variable})
    upper_bounds_set = list(set(upper_bounds_set) | {leaving_variable})
    return lower_bounds_set, upper_bounds_set

def _remove_from_upper_bounds_and_add_to_lower_bounds(
    lower_bounds_set: list, upper_bounds_set: list, k: np.intp
) -> tuple[list, list]:
    """
    Updates the sets of non-basic variables at their lower and upper bounds.
    Args:
        lower_bounds_set: the set of indices of the non-basic variables that are at
            their lower bound.
        upper_bounds_set: the set of indices of the non-basic variables that are at
            their upper bound.
        k: the variable that should leave upper_bounds and enter lower_bounds.
    Returns:
        lower_bounds_set: the updated set of indices of the non-basic variables at
            their lower bounds.
        upper_bounds_set: the updated set of indices of the non-basic variables at
            their upper bounds.
    """
    leaving_variable = k
    upper_bounds_set = list(set(upper_bounds_set) - {leaving_variable})
    lower_bounds_set = list(set(lower_bounds_set) | {leaving_variable})
    return lower_bounds_set, upper_bounds_set

def _update_I_and_J(I: list, J: list, k: np.intp, r: np.intp) -> tuple[list, list]:
    """
    Updates the sets of basic and non-basic variables.
    Args:
        I: the set of indices of the basic variables.
        J: the set of indices of the non-basic variables.
        k: the index of the variable that enters the basis.
        r: the index of the variable that leaves the basis.
    Returns:
        I: the updated set of indices of the basic variables.
        J: the updated set of indices of the non-basic variables.
    """
    I = deepcopy(I)
    J = deepcopy(J)

    # as we have two sets of non-basic variables, indices should be handled
    # in a different way that we did in the previous methods
    # we'll use to_enter_value and to_enter_indices as they are different
    # entities now
    to_enter_index = J.index(k)
    to_enter_value = k

    to_exit = deepcopy(I[r])

    I[r] = to_enter_value
    J[to_enter_index] = to_exit
    return I, J

def _build_initial_tableau(
        A : np.ndarray,
        b : np.ndarray,
        c : np.ndarray,
        I : list,
        J_1 : list,
        J_2 : list,
        lower_bounds : np.ndarray,
        upper_bounds : np.ndarray,
        debug : bool = False
):
    """
    Builds the initial tableau for the simplex algorithm for bounded variables.
    Args:
        A: the matrix of the linear system.
        b: the right-hand side vector of the linear system.
        c: the cost vector of the linear system.
        I: the set of indices of the basic variables.
        J_1: the set of indices of the non-basic variables that are at their
             lower bound.
        J_2: the set of indices of the non-basic variables that are at their
             upper bound.
        lower_bounds: the lower bounds of the variables.
        upper_bounds: the upper bounds of the variables.
        debug: whether to log debug information.
    Returns:
        tableau: the initial tableau.
    """
    m, n = A.shape
    # we expect that α + β + γ == n
    # if this is not the case, we should raise an error
    if len(I) + len(J_1) + len(J_2) != n:
        raise ValueError("The sets of basic and non-basic variables are not disjoint.")

    α = len(I)
    β = len(J_1)
    γ = len(J_2)

    # compute necessary variables
    A_I, x_I, π, z_0 = _compute_A_I_x_I_π_and_z_0(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds)

    tableau = np.zeros((m + 1, α + β + γ + 1))
    # 1. add reduced costs from J_1 into the first row of the tableau
    tableau[0, J_1] = _compute_c_hat_J(π, A, c, J_1)
    # 2. add reduced costs from J_2 into the first row of the tableau
    tableau[0, J_2] = _compute_c_hat_J(π, A, c, J_2)
    # 3. add z_0 - ẑ - into the first row of the tableau at the last column
    tableau[0, -1] = z_0

    # 4. add identity matrix into the 2nd to (α + 1)-th rows of the tableau
    #    corresponding to the columns of the basic variables
    tableau[1 : α + 1, I] = np.eye(α)
    # 5. add A_I_inv @ A_J_1 into the 2nd to (α + 1)-th rows of the tableau
    #    corresponding to the columns of the non-basic variables at their
    #    lower bounds J_1
    A_I_inv = np.linalg.inv(A_I)
    tableau[1 : α + 1, J_1] = A_I_inv.dot(A[:, J_1])
    # 6. add A_I_inv @ A_J_2 into the 2nd to (α + 1)-th rows of the tableau
    #    corresponding to the columns of the non-basic variables at their
    #    upper bounds J_2
    tableau[1 : α + 1, J_2] = A_I_inv.dot(A[:, J_2])
    # 7. add x_I - b̂ - into the 2nd to (α + 1)-th rows of the tableau
    #    into the last column of the tableau
    tableau[1 : α + 1, -1] = x_I

    # the tableau is now built
    return tableau

def _pivot_tableau(T, r, k):
    """
    Pivots the tableau T.
    Args:
        T: the tableau of the linear system.
        r: the index of the variable that leaves the basis.
        k: the index of the variable that enters the basis.
    Returns:
        T: the pivoted tableau.
        pivot_operations : the pivot operations that were applied to the tableau.
    """
    pivot_operations = {
        'divide': None,
        'subtract': []
    }
    r = r + 1 # this is needed as the first row of T is the cost row
    pivot_area = T[:, :-1]
    m, _ = pivot_area.shape
    pivot_element = pivot_area[r, k]
    # divide the pivot row by the pivot element
    pivot_area[r, :] = pivot_area[r, :] / pivot_element
    pivot_operations['divide'] = (r - 1, pivot_element)
    # apply linear combination to the other rows
    rows = list(range(r)) + list(range(r + 1, m))
    for row in rows:
        pivot_operations['subtract'].append((row, pivot_area[row, k], r - 1))
        pivot_area[row,:] -= pivot_area[row,k] * pivot_area[r, :]
    return T, pivot_operations


def _simplex_step_1(
    T : np.ndarray, 
    I : list,
    J_1 : list,
    J_2 : list
) -> Tuple[np.intp | None, np.float64, list, np.bool_]:
    """
    Executes the first step of the simplex algorithm for bounded variables.
    Args:
    T : the tableau of the linear system.
    I : the set of indices of the basic variables.
    J_1 : the set of indices of the non-basic variables that are at their
    J_2 : the set of indices of the non-basic variables that are at their
    Returns:
    k : the index of the variable that enters the basis.
    c_hat_k : the reduced cost of the variable that enters the basis.
    k_set: the set where the variable k belongs to.
    proceed : whether the algorithm should proceed.
    """
    c_hat_J_1 = T[0, J_1]
    c_hat_J_2 = T[0, J_2]
    k, c_hat_k, k_set, proceed = _find_k_who_enters(
        c_hat_J_1, c_hat_J_2, J_1, J_2
    )

    return (k, c_hat_k, k_set, proceed) if proceed else\
           (None, c_hat_k, k_set, proceed)

pipeline_memory = {
}

def _store_at_pipeline_memory(T, I, J_1, J_2):
    pipeline_memory["T"] = T
    pipeline_memory["I"] = I
    pipeline_memory["J_1"] = J_1
    pipeline_memory["J_2"] = J_2

def _retrieve_from_pipeline_memory():
    return pipeline_memory["T"], pipeline_memory["I"], pipeline_memory["J_1"], pipeline_memory["J_2"]

def _simplex_step_2_k_enters_from_lower_bounds(I, J_1, J_2, k, T, lower_bounds, upper_bounds):
    # necessary variables
    A = np.copy(T[1:, :-1])
    A_I_inv = A[:, I]
    b = np.copy(T[1:, -1])
    b_hat = np.copy(T[1:, -1])
    c_hat_k = T[0, k]
    z_hat = T[0, -1]
    y_k = A_I_inv.dot(A[:, k])

    # 1. compute r, Δ_k
    r, Δ_k, proceed, bound = _compute_Δ_k_for_non_basic_entering_from_its_lower_bound(
        I, k, y_k, T[1:, -1], lower_bounds, upper_bounds
    )
    # 2. early return if Δ_k is infinite and then stop the algorithm
    #    as the problem is unbounded
    if not proceed:
        return T, I, J_1, J_2, {'divide': None, 'subtract': []}, False

    # 3. update variables that will hold this value till the end of the iteration
    x_k = lower_bounds[k] + Δ_k
    b_hat = b_hat - y_k * Δ_k
    z_hat = z_hat - c_hat_k * Δ_k
    # update x_I (b_hat) and z_hat at the tableau
    T[1:, -1] = b_hat
    T[0, -1] = z_hat

    # 3.1 early return if Δ_k is γ_3 (u_k - l_k) as no modifications will be made
    #     at the basis
    if Δ_k == (upper_bounds[k] - lower_bounds[k]):
        # remove k from lower_bounds and add it to upper_bounds
        J_1, J_2 = _remove_from_lower_bounds_and_add_to_upper_bounds(J_1, J_2, k)
        # return the updated tableau
        return T, I, J_1, J_2, {'divide': None, 'subtract': []}, True

    # 4. check if candidates r and k produces a strongly feasible partition
    candidate_I = I.copy()
    candidate_J_1 = J_1.copy()
    candidate_I, candidate_J_1 = _update_I_and_J(candidate_I, candidate_J_1, k, r)
    # 5. if the partition is not strongly feasible, we should find the
    #    index of the basic variable that leaves the basis through the
    #    lexico-rule cycle-proof method
    if not _is_strongly_feasible(
        A,
        b,
        lower_bounds,
        upper_bounds,
        candidate_I
        ):
        S_1 = _build_s_1_set_x_k_enters_from_lower_bounds(candidate_I, A, b, lower_bounds, k)
        S_2 = _build_s_2_set_x_k_enters_from_lower_bounds(candidate_I, A, b, upper_bounds, k)
        print(I[r], k)
        r = _find_r_to_leave_for_k_entering_from_lower_bound_cycle_proof(
            S_1, S_2, candidate_I, y_k, A_I_inv
        ) # the lexico-rule will properly select a valid r to leave
    # 6. Now that we have a valid r, we can update the basis
    J_1_before = J_1.copy()
    I, J_1 = _update_I_and_J(I, J_1, k, r)
    # 6.1 and then update the sets of non-basic variables at their bounds
    # considering the bound that r leaved to
    if bound == 1: # r leaved to upper bounds
        # so we should take it from lower (J_1) to upper (J_2) but k ain't no index
        # as it is the variable itself that entered the basis, so the basic variable
        # which leaved the basis and should go upper bound can be obtained by the difference
        # between J_1 and J_1_before sets as after the update, this difference represents
        # the basic variable that joined the basis
        lower_to_upper = (set(J_1) - set(J_1_before)).pop()
        J_1, J_2 = _remove_from_lower_bounds_and_add_to_upper_bounds(J_1, J_2, lower_to_upper)
    # else, r leaved to lower bounds and the previous I_and_J update already
    # took care of the bounds

    # 7. update b_hat at r index to reflect the new value of x_k that just
    #    entered the basis
    b_hat[r] = x_k
    # 8. update the tableau with the new values of b_hat
    T[1:, -1] = b_hat
    # 9. pivot the tableau at the k-th column and r-th row
    T, pivot_operations = _pivot_tableau(T, r, k)
    # return the updated tableau
    return T, I, J_1, J_2, pivot_operations, True


def _simplex_step_2_pipeline(proceed, *args, **kwargs):
    if proceed:
        T, I, J_1, J_2, pivot_operations, proceed = _simplex_step_2_k_enters_from_lower_bounds(*args, **kwargs)
        _store_at_pipeline_memory(T, I, J_1, J_2)
        return T, I, J_1, J_2, pivot_operations, proceed
    return *_retrieve_from_pipeline_memory(), {'divide': None, 'subtract': []}, False

def _simplex_step_3_k_enters_from_upper_bounds(I, J_1, J_2, k, T, lower_bounds, upper_bounds):
    # necessary variables
    A = np.copy(T[1:, :-1])
    A_I_inv = A[:, I]
    b = np.copy(T[1:, -1])
    b_hat = np.copy(T[1:, -1])
    c_hat_k = T[0, k]
    z_hat = T[0, -1]
    y_k = A_I_inv.dot(A[:, k])

    # 1. compute r, Δ_k
    r, Δ_k, proceed, bound = _compute_Δ_k_for_non_basic_entering_from_its_upper_bound(
        I, k, y_k, T[1:, -1], lower_bounds, upper_bounds
    )
    # 2. early return if Δ_k is infinite and then stop the algorithm
    #    as the problem is unbounded
    if not proceed:
        return T, I, J_1, J_2, {'divide': None, 'subtract': []}, False

    # 3. update variables that will hold this value till the end of the iteration
    x_k = upper_bounds[k] - Δ_k
    b_hat = b_hat + y_k * Δ_k
    z_hat = z_hat + c_hat_k * Δ_k
    # update x_I (b_hat) and z_hat at the tableau
    T[1:, -1] = b_hat
    T[0, -1] = z_hat

    # 3.1 early return if Δ_k is γ_3 (u_k - l_k) as no modifications will be made
    #     at the basis
    if Δ_k == (upper_bounds[k] - lower_bounds[k]):
        # remove k from upper_bounds and add it to lower_bounds
        J_1, J_2 = _remove_from_upper_bounds_and_add_to_lower_bounds(J_1, J_2, k)
        # return the updated tableau
        return T, I, J_1, J_2, {'divide': None, 'subtract': []}, False

    # 4. check if candidates r and k produces a strongly feasible partition
    candidate_I = I.copy()
    candidate_J_2 = J_2.copy()
    candidate_I, candidate_J_2 = _update_I_and_J(candidate_I, candidate_J_2, k, r)

    # 5. if the partition is not strongly feasible, we should find the
    #    index of the basic variable that leaves the basis through the
    #    lexico-rule cycle-proof method
    if not _is_strongly_feasible(
        A,
        b,
        lower_bounds,
        upper_bounds,
        candidate_I
        ):
        S_1 = _build_s_1_set_x_k_enters_from_upper_bounds(candidate_I, A, b, lower_bounds, k)
        S_2 = _build_s_1_set_x_k_enters_from_upper_bounds(candidate_I, A, b, upper_bounds, k)
        r = _find_r_to_leave_for_k_entering_from_upper_bound_cycle_proof(
            S_1, S_2, y_k, A_I_inv
        ) # the lexico-rule will properly select a valid r to leave
    # 6. Now that we have a valid r, we can update the basis
    J_2_before = J_2.copy()
    I, J_2 = _update_I_and_J(I, J_2, k, r)
    # 6.1 and then update the sets of non-basic variables at their bounds
    # considering the bound that r leaved to
    if bound == 0: # r leaved to lower
        # so we should take it from upper (J_2) to lower (J_1) but k ain't no index
        # as it is the variable itself that entered the basis, so the basic variable
        # which leaved the basis and should go lower bound can be obtained by the difference
        # between J_2 and J_2_before sets as after the update, this difference represents
        # the basic variable that joined the basis
        upper_to_lower = (set(J_2) - set(J_2_before)).pop()
        J_1, J_2 = _remove_from_upper_bounds_and_add_to_lower_bounds(J_1, J_2, upper_to_lower)
    # else, r leaved to upper bounds and the previous I_and_J update already
    # took care of the bounds

    # 7. update b_hat at r index to reflect the new value of x_k that just
    #    entered the basis
    b_hat[r] = x_k
    # 8. update the tableau with the new values of b_hat
    T[1:, -1] = b_hat
    # 9. pivot the tableau at the k-th column and r-th row
    T, pivot_operations = _pivot_tableau(T, r, k)
    # return the updated tableau
    return T, I, J_1, J_2, pivot_operations, True

def _simplex_step_3_pipeline(proceed, *args, **kwargs):
    if proceed:
        T, I, J_1, J_2, pivot_operations, proceed = _simplex_step_3_k_enters_from_upper_bounds(*args, **kwargs)
        _store_at_pipeline_memory(T, I, J_1, J_2)
        return T, I, J_1, J_2, pivot_operations, proceed
    return *_retrieve_from_pipeline_memory(), {'divide': None, 'subtract': []}, False

def _simplex_main_loop(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds):
    tableau_steps = {}

    _, n = A.shape
    T = _build_initial_tableau(
        A, b, c, I, J_1, J_2, lower_bounds, upper_bounds
    )
    # store values at pipeline memory for the case that the I basis is already
    # an optimal basis
    _store_at_pipeline_memory(T, I, J_1, J_2)

    # main loop
    solution_found = False
    optimal = False
    unbouded = False
    iterations = 0

    tableau_steps[iterations] = {
        'previous_T': np.copy(T),
        'T': np.copy(T),
        'previous_I': deepcopy(I),
        'previous_J_1': deepcopy(J_1),
        'previous_J_2': deepcopy(J_2),
        'I': deepcopy(I),
        'J_1': deepcopy(J_1),
        'J_2': deepcopy(J_2),
        'to_enter': None,
        'to_leave': None,
        'c_hat_J_1': None,
        'c_hat_J_2': None,
        'c_hat_k': None,
        'pivot_operations': {
            'divide': None,
            'subtract': []
        }
    }
    while not solution_found:
        # 1. rebuild the step functions map to reflect changes made at the non-basic
        # variables set on previous iterations
        step_functions = {
            tuple(J_1): _simplex_step_2_pipeline,
            tuple(J_2): _simplex_step_3_pipeline
        }
        # 2. find k non basic candidate to enter the basis
        k, c_hat_k, k_set, proceed = _simplex_step_1(T, I, J_1, J_2)
        # 2.1 set optimal flag if optimal solution was found
        optimal = not proceed
        # 3. update solution_found flag
        solution_found = not proceed
        # 4. build args for the next step
        args = (I, J_1, J_2, k, np.copy(T), lower_bounds, upper_bounds)
        # 5. call next step through the pipeline (to avoid computing r and Δ_k
        #    if a solution was found in step 1)
        T, I, J_1, J_2, pivot_operations, proceed = step_functions[
            tuple(k_set)
        ](proceed, *args)
        # 5.1 set unbounded flag if unbounded solution was found
        unbouded = not proceed
        # 6. update solution_found flag again
        solution_found = not proceed
        # 7. update iterations counter
        iterations += 1
        # 8. store tableau at the steps dictionary
        tableau_steps[iterations] = {
            'previous_T': np.copy(tableau_steps[iterations - 1]['T']),
            'T': np.copy(T),
            'previous_I': deepcopy(tableau_steps[iterations - 1]['I']),
            'previous_J_1': deepcopy(tableau_steps[iterations - 1]['J_1']),
            'previous_J_2': deepcopy(tableau_steps[iterations - 1]['J_2']),
            'I': deepcopy(I),
            'J_1': deepcopy(J_1),
            'J_2': deepcopy(J_2),
            'to_enter': k,
            'to_leave': next(iter(set(tableau_steps[iterations - 1]['I']) - set(I)), None),
            'c_hat_J_1': np.copy(T[0, J_1]),
            'c_hat_J_2': np.copy(T[0, J_2]),
            'c_hat_k': c_hat_k,
            'pivot_operations': pivot_operations
        }
    # 9. evaluate solution found
    solution_type = None
    if optimal:
        solution_type = 1 if c_hat_k != 0 else 2
    elif unbouded:
        solution_type = 3
    # 10. Add necessary data to the final step
    tableau_steps[iterations]['C'] = c
    tableau_steps[iterations]['B'] = b
    tableau_steps[iterations]['A'] = A
    tableau_steps[iterations]['solution_type'] = solution_type
    tableau_steps[iterations]['lower_bounds'] = lower_bounds
    tableau_steps[iterations]['upper_bounds'] = upper_bounds
    # 11. return final values
    I_star = I
    z_star = T[0, -1]
    x_star = np.zeros(n)
    x_star[J_1] = lower_bounds[J_1]
    x_star[J_2] = upper_bounds[J_2]
    x_star[I_star] = T[1:, -1]
    return z_star, x_star, I_star, solution_type, iterations, tableau_steps

def simplex(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds):
    """
    Solves the linear programming problem using the simplex algorithm for bounded variables.
    Args:
        A: the constraint matrix.
        b: the right-hand side vector.
        c: the cost vector.
        I: the set of indices of the basic variables.
        J_1: the set of indices of the non-basic variables that are at their lower bound.
        J_2: the set of indices of the non-basic variables that are at their upper bound.
        lower_bounds: the lower bounds of the variables.
        upper_bounds: the upper bounds of the variables.
    Returns:
        z_star: the optimal value of the objective function.
        x_star: the optimal values of the variables.
        I_star: the optimal set of indices of the basic variables.
        solution_type: the type of the solution.
        iterations: the number of iterations.
        tableau_steps: the steps of the simplex algorithm.
    """
    return _simplex_main_loop(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds)

def _repr_row_pivot_operations(row_pivot_operations: tuple, I : list) -> str:
    if row_pivot_operations is None:
        return ''
    r, pivot_row_multiplier = row_pivot_operations
    return f'\\\\(R_{{X_{{{I[r]}}}}} \\leftarrow \\frac{{R_{{X_{{{I[r]}}}}}}}{{{pivot_row_multiplier:.3f}}}\\\\)'

def _repr_column_pivot_operations(column_pivot_operations: list, labels : list, I : list) -> list:
    column_pivot_operations_list = []
    for i, pivot_column_multiplier, r in column_pivot_operations:
        if pivot_column_multiplier >= 0:
            column_pivot_operations_list.append(
                    f'\\\\(R_{{{labels[i]}}} \\leftarrow R_{{{labels[i]}}} - {pivot_column_multiplier:.3f}R_{{{I[r]}}}\\\\)'
            )
        else:
            column_pivot_operations_list.append(
                    f'\\\\(R_{{{labels[i]}}} \\leftarrow R_{{{labels[i]}}} + {-pivot_column_multiplier:.3f}R_{{{I[r]}}}\\\\)'
            )
    return column_pivot_operations_list

def _markdown_pivot_operations(to_enter, to_leave, I: list, pivot_operations: dict) -> str:
    labels = ['\\text{{Cost}}'] + [
        f'X_{{{i}}}' for i in I
    ]
    pivot_operations_str  = '### Pivot Operations\n\n'
    pivot_operations_str += 'Variable to enter: \\\\(' + (str(to_enter) or '') + '\\\\)\n'
    pivot_operations_str += 'Variable to leave: \\\\(' + (str(to_leave) or '') + '\\\\)\n'
    pivot_operations_str += '#### Row Operations:\n\n'
    pivot_operations_str += '1. ' + _repr_row_pivot_operations(pivot_operations['divide'],  I) + '\n'
    for i, column_pivot_operation in enumerate(_repr_column_pivot_operations(pivot_operations['subtract'], labels, I)):
        pivot_operations_str += f'{i + 1}. {column_pivot_operation}\n'
    return pivot_operations_str

def _markdown_T(T: np.ndarray, I: list) -> str:
    table_header = '|   |' + ' | '.join([f'\\\\(X_{{{i}}}\\\\)' for i in range(T.shape[1] - 1)]) + ' | RHS |\n'
    table_header += '|---|' + '---|' * (T.shape[1] - 1) + '---|\n'

    row_names = ['Cost'] + [f'\\\\(X_{{{i}}}\\\\)' for i in I]

    table_body = ''
    for i, row in enumerate(T):
        table_body += f'| {row_names[i]} | ' + ' | '.join([f'{value:.3f}' for value in row]) + ' |\n'

    return table_header + table_body

def _repr_arr_tex(bounds) -> str:
    _bounds = []
    # replace all np.inf in bounds for '∞'
    for bound in bounds:
        if bound == np.inf:
            _bounds.append('∞')
        elif bound == -np.inf:
            _bounds.append('-∞')
        else:
            _bounds.append(f'{bound:.2f}')
    return '[ ' + ',\\ '.join(_bounds) + ' ]'

def _markdown_final_step(last_step : dict, iteration) -> str:
    solution_type = _solution_types_map[last_step['solution_type']]
    n = len(last_step['C'])
    I_star = last_step['I']
    J_1 = last_step['J_1']
    J_2 = last_step['J_2']
    c_hat_J_1 = last_step['c_hat_J_1']
    c_hat_J_2 = last_step['c_hat_J_2']
    z_star = last_step['T'][0, -1]
    x_star = np.zeros(n)
    x_star[J_1] = last_step['lower_bounds'][J_1]
    x_star[J_2] = last_step['upper_bounds'][J_2]
    x_star[I_star] = last_step['T'][1:, -1]


    A = last_step['A']
    B = last_step['B']
    C = last_step['C']
    restrictions_matrix_latex_format = '\\begin{bmatrix}\n'
    for i, row in enumerate(A):
        restrictions_matrix_latex_format += ' & '.join([f'{value:.3f}' for value in row]) + ' \\\\\n'
    restrictions_matrix_latex_format += '\\end{bmatrix}'
    c_latex_format = '\\begin{bmatrix}\n' + ' & '.join([f'{value:.3f}' for value in C]) + '\n\\end{bmatrix}'
    b_latex_format = '\\begin{bmatrix}\n' + '\\\\\n'.join([f'{value:.3f}' for value in B]) + '\n\\end{bmatrix}'
    x_matrix_latex_format = '\\begin{bmatrix}\n' + '\\\\\n'.join([f'X_{{{i}}}' for i, _ in enumerate(C)]) + '\n\\end{bmatrix}'

    statement = f'''\\\\[
    \\begin{{aligned}}
    & \\text{{Minimize}} & C^{{T}} \\cdot X \\\\
    & \\text{{Subject to}} & A \\cdot X & = B \\\\
    & & L \\leq  X  \\leq U &
    \\end{{aligned}}
    \\\\]
    where: 
    \\\\[
    \\begin{{aligned}}
    & & A = {restrictions_matrix_latex_format} \\\\
    & & B = {b_latex_format} \\\\
    & & C^{{T}} = {c_latex_format} \\\\
    & & X = {x_matrix_latex_format} \\\\
    & & L = {_repr_arr_tex(last_step['lower_bounds'])} \\\\
    & & U = {_repr_arr_tex(last_step['upper_bounds'])}
    \\end{{aligned}}
    \\\\]\n\n'''
    markdown  = f'\n\n## Solution for \n\n'
    markdown += statement
    markdown += f'Solution Type: {solution_type}\n\n'
    markdown += f'Optimal Solution: \\\\(X^{{*}} = {_repr_arr_tex(x_star)}\\\\)\n\n'
    markdown += f'Optimal Value: \\\\(Z^{{*}} = {z_star:.2f}\\\\)\n\n'
    markdown += f'Optimal Basis: \\\\({I_star}\\\\)\n\n'
    markdown += f'Final non-basic variables set at lower bounds \\\\(J_1\\\\): \\\\({J_1}\\\\)\n\n'
    markdown += f'Final non-basic variables set at upper bounds \\\\(J_2\\\\): \\\\({J_2}\\\\)\n\n'
    markdown += f'\\\\(\\hat{{C}}_{{J_1}}\\\\): \\\\({c_hat_J_1.tolist()}\\\\)\n\n'
    markdown += f'\\\\(\\hat{{C}}_{{J_2}}\\\\): \\\\({c_hat_J_2.tolist()}\\\\)\n\n'
    markdown += f'Number of Iterations: {iteration + 1}\n\n'

    return markdown

def markdown_repr_T(tableau_steps) -> str:
    markdown = ''
    for iteration in tableau_steps:
        # skip first iteration
        if iteration == 0:
            continue
        markdown += f'\n\n## Iteration {iteration}\n\n'
        markdown += 'Previous Tableau:\n\n'
        markdown += _markdown_T(
            tableau_steps[iteration]['previous_T'],
            tableau_steps[iteration]['previous_I']
        )
        markdown += _markdown_pivot_operations(
            tableau_steps[iteration]['to_enter'],
            tableau_steps[iteration]['to_leave'],
            tableau_steps[iteration]['previous_I'],
            tableau_steps[iteration]['pivot_operations']
        )
        markdown += '\n\nComputed Tableau:\n\n'
        markdown += _markdown_T(
            tableau_steps[iteration]['T'],
            tableau_steps[iteration]['I']
        )
        markdown += '\n\n\\\\(J_1\\\\) (lower bounds variables): ' + str(tableau_steps[iteration]['J_1']) + '\n'
        markdown += '\n\n\\\\(J_2\\\\) (upper bounds variables): ' + str(tableau_steps[iteration]['J_2']) + '\n\n'
    markdown += _markdown_final_step(tableau_steps[iteration], iteration)
    return markdown
