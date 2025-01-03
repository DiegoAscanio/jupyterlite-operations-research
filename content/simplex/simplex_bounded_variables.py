from typing import Any, Tuple, List
import numpy as np
from copy import deepcopy
import pdb

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
) -> Tuple[np.intp, np.float64, np.bool_]:
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
    proceed = c_hat_k > 0
    return k, c_hat_k, proceed

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
) -> Tuple [ int | None, np.float64 ]:
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
               to its lower bound - if appropriate.
            Δ_k: the value of Δ_k.
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
    return r, Δ_k

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
) -> Tuple [ int | None, np.float64 ]:
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
    return r, Δ_k

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
    to_enter = deepcopy(J[k])
    to_exit = deepcopy(I[r])
    I[r] = to_enter
    J[k] = to_exit
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
