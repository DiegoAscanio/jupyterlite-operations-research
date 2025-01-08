from typing import Any, Tuple, List
import numpy as np
from copy import deepcopy
import pdb

from utils import lexicographic_negative, lexicographic_positive

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
    # if J_2 is empty we should only consider the reduced costs of J_1
    if len(J_2) == 0:
        k = np.argmax(
            np.where(c_hat_J_1 > 0, c_hat_J_1, -np.inf)
        )
        c_hat_k = c_hat_J_1[k]
    # elif J_1 is empty we should only consider the reduced costs of J_2
    elif len(J_1) == 0:
        k = np.argmax(
            np.where(c_hat_J_2 < 0, c_hat_J_2, np.inf)
        )
        c_hat_k = c_hat_J_2[k]
    # otherwise we should consider the reduced costs of both J_1 and J_2
    # and choose the one with the highest reduced cost
    else:
        k_1 = np.argmax(
            np.where(c_hat_J_1 > 0, c_hat_J_1, -np.inf)
        )
        k_2 = np.argmax(
            np.where(c_hat_J_2 < 0, c_hat_J_2, -np.inf)
        )
        candidate_indices = [k_1, k_2]
        candidate_values = [c_hat_J_1[k_1], -c_hat_J_2[k_2]]
        better_improvement = np.max(candidate_values)
        better_index = candidate_values.index(better_improvement)
        k = candidate_indices[better_index]
        c_hat_k = better_improvement
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
_build_s_1_set = _basic_variables_degenerate_at_their_lower_bounds

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
_build_s_2_set = _basic_variables_degenerate_at_their_upper_bounds

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
    y_k : np.ndarray,
    A_I_inv : np.ndarray
) -> int | None:
    degenerate_basic_variables = list(set(S_1) | set(S_2))
    # we need to handle it as list to sort appropriately
    r_candidates_to_leave: list = list(
        map(
            lambda x: tuple(x),
            A_I_inv [degenerate_basic_variables, :] / y_k[degenerate_basic_variables]
        )
    )

    # we make a copy of r_candidates_to_leave - a list of tuples
    lexicographic_rule = r_candidates_to_leave.copy()
    # and sort the copy, as tuple sorting in python is lexicographic
    lexicographic_rule.sort()

    # now we're interested at the minimum value of the lexicographically ordered
    # so the first element of lexicographic_rule is the one we're interested at
    lexicographically_minimum_row = lexicographic_rule[0]

    # the index is the index of r_candidates_to_leave that contains the minimum
    # row
    return r_candidates_to_leave.index(lexicographically_minimum_row)

def _find_r_to_leave_for_k_entering_from_upper_bound_cycle_proof(
    S_1 : list,
    S_2 : list,
    y_k : np.ndarray,
    A_I_inv : np.ndarray
) -> int | None:
    degenerate_basic_variables = list(set(S_1) | set(S_2))
    # we need to handle it as list to sort appropriately
    r_candidates_to_leave: list = list(
        map(
            lambda x: tuple(x),
            A_I_inv [degenerate_basic_variables, :] / y_k[degenerate_basic_variables]
        )
    )

    # we make a copy of r_candidates_to_leave - a list of tuples
    lexicographic_rule = r_candidates_to_leave.copy()
    # and sort the copy, as tuple sorting in python is lexicographic
    lexicographic_rule.sort()

    # now we're interested at the maximum value of the lexicographically ordered
    # so the last element of lexicographic_rule is the one we're interested at
    lexicographically_maximum_row = lexicographic_rule[-1]

    # the index is the index of r_candidates_to_leave that contains the minimum
    # row
    return r_candidates_to_leave.index(lexicographically_maximum_row)

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
    leaving_variable = lower_bounds_set[k]
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
    leaving_variable = upper_bounds_set[k]
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
    """
    r = r + 1 # this is needed as the first row of T is the cost row
    pivot_area = T[:, :-1]
    m, _ = pivot_area.shape
    pivot_element = pivot_area[r, k]
    # divide the pivot row by the pivot element
    pivot_area[r, :] = pivot_area[r, :] / pivot_element
    # apply linear combination to the other rows
    rows = list(range(r)) + list(range(r + 1, m))
    for row in rows:
        pivot_area[row,:] -= pivot_area[row,k] * pivot_area[r, :]
    return T


def _simplex_step_1(
    T : np.ndarray, 
    I : list,
    J_1 : list,
    J_2 : list
) -> Tuple[np.intp, np.float64, list, np.bool_]:
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

    return k, c_hat_k, k_set, proceed

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
        return T, I, J_1, J_2, False

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
        return T, I, J_1, J_2, True

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
        S_1 = _build_s_1_set(candidate_I, A, b, lower_bounds)
        S_2 = _build_s_2_set(candidate_I, A, b, upper_bounds)
        r = _find_r_to_leave_for_k_entering_from_lower_bound_cycle_proof(
            S_1, S_2, y_k, A_I_inv
        ) # the lexico-rule will properly select a valid r to leave
    # 6. Now that we have a valid r, we can update the basis
    I, J_1 = _update_I_and_J(I, J_1, k, r)
    # 6.1 and then update the sets of non-basic variables at their bounds
    # considering the bound that r leaved to
    if bound == 1: # r leaved to upper bounds
        J_1, J_2 = _remove_from_lower_bounds_and_add_to_upper_bounds(J_1, J_2, k)
    # else, r leaved to lower bounds and the previous I_and_J update already
    # took care of the bounds

    # 7. update b_hat at r index to reflect the new value of x_k that just
    #    entered the basis
    b_hat[r] = x_k
    # 8. update the tableau with the new values of b_hat
    T[1:, -1] = b_hat
    # 9. pivot the tableau at the k-th column and r-th row
    T = _pivot_tableau(T, r, k)
    # return the updated tableau
    return T, I, J_1, J_2, True


def _simplex_step_2_pipeline(proceed, *args, **kwargs):
    if proceed:
        T, I, J_1, J_2, proceed = _simplex_step_2_k_enters_from_lower_bounds(*args, **kwargs)
        _store_at_pipeline_memory(T, I, J_1, J_2)
        return T, I, J_1, J_2, proceed
    return *_retrieve_from_pipeline_memory(), False

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
        return T, I, J_1, J_2, False

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
        return T, I, J_1, J_2, True

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
        S_1 = _build_s_1_set(candidate_I, A, b, lower_bounds)
        S_2 = _build_s_2_set(candidate_I, A, b, upper_bounds)
        r = _find_r_to_leave_for_k_entering_from_upper_bound_cycle_proof(
            S_1, S_2, y_k, A_I_inv
        ) # the lexico-rule will properly select a valid r to leave
    # 6. Now that we have a valid r, we can update the basis
    I, J_2 = _update_I_and_J(I, J_2, k, r)
    # 6.1 and then update the sets of non-basic variables at their bounds
    # considering the bound that r leaved to
    if bound == 0: # r leaved to lower
        J_1, J_2 = _remove_from_upper_bounds_and_add_to_lower_bounds(J_1, J_2, k)
    # else, r leaved to upper bounds and the previous I_and_J update already
    # took care of the bounds

    # 7. update b_hat at r index to reflect the new value of x_k that just
    #    entered the basis
    b_hat[r] = x_k
    # 8. update the tableau with the new values of b_hat
    T[1:, -1] = b_hat
    # 9. pivot the tableau at the k-th column and r-th row
    T = _pivot_tableau(T, r, k)
    # return the updated tableau
    return T, I, J_1, J_2, True

def _simplex_step_3_pipeline(proceed, *args, **kwargs):
    if proceed:
        T, I, J_1, J_2, proceed = _simplex_step_3_k_enters_from_upper_bounds(*args, **kwargs)
        _store_at_pipeline_memory(T, I, J_1, J_2)
        return T, I, J_1, J_2, proceed
    return *_retrieve_from_pipeline_memory(), False

def _simplex_main_loop(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds):
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
        args = (I, J_1, J_2, k, T, lower_bounds, upper_bounds)
        print(f'Iteration {iterations}')
        print("I, J_1, J_2, k, T, lower_bounds, upper_bounds")
        print(args)
        print()
        # 5. call next step through the pipeline (to avoid computing r and Δ_k
        #    if a solution was found in step 1)
        T, I, J_1, J_2, proceed = step_functions[
            tuple(k_set)
        ](proceed, *args)
        # 5.1 set unbounded flag if unbounded solution was found
        unbouded = not proceed
        # 6. update solution_found flag again
        solution_found = not proceed
        # 7. update iterations counter
        iterations += 1
    # 7. evaluate solution found
    solution_type = None
    if optimal:
        solution_type = 1 if c_hat_k != 0 else 2
    elif unbouded:
        solution_type = 3
    # 8. return final values
    I_star = I
    z_star = T[0, -1]
    x_star = np.zeros(n)
    x_star[I_star] = T[1:, -1]
    return z_star, x_star, I_star, solution_type, iterations
