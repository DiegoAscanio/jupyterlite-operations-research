from typing import Any
import numpy as np
from copy import deepcopy
import pdb

def _compute_x_from_x_I(n, I, x_I):
    x = np.zeros(n)
    x[I] = x_I
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

def _compute_X_I(A_I: np.ndarray, b: np.ndarray) -> np.ndarray:
    X_I = np.linalg.inv(A_I).dot(b)
    return X_I

def _compute_A_I_x_I_π_and_z_0(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # A_I should be a submatrix from A with columns indexed (and sorted) by I
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    # C_I should be a submatrix from c with columns indexed (and sorted) by I
    c_I = c[I]

    x_I = _compute_X_I(A_I, b)
    π = c_I.dot(A_I_inv)
    z_0 = π.dot(b)
    return A_I, x_I, π, z_0

def _compute_c_hat_J(π: np.ndarray, A: np.ndarray, c: np.ndarray, J: list) -> np.ndarray:
    A_J = A[:, J]
    c_hat_J = π.dot(A_J) - c[J]
    return c_hat_J

def _find_k_who_enters_to_keep_dual_feasibility(c_hat_J: np.ndarray) -> tuple[np.intp, bool]:
    """
    Finds the index of the variable that enters the basis and returns if
    the simplex method shouuld continue. When c_hat_J[k] <= 0, the method
    found an optimal, unique and finite solution and, therefore, should stop.
    Args:
        c_hat_J: the vector of the reduced costs of the non-basic variables.
    Returns:
        k: the index of the variable that enters the basis.
        proceed_: a boolean indicating if the simplex method should proceed.
    """
    k: np.intp = np.argmax(c_hat_J)
    return k, c_hat_J[k] > 0

def _find_r_who_leaves( I : list, x_I : np.ndarray) -> np.intp:
    """
    Finds the index of the variable that leaves the basis.
    for the dual simplex method.
    Args:
        I: the set of indices of the basic variables.
        x_I: the vector of the values of the basic variables.
    Returns:
        r : the index of the variable that leaves the basis.
    """
    b_bar = x_I
    mask = b_bar < 0
    b_bar[~mask] = np.inf
    return np.argmin(b_bar)

def _find_k_who_enters(
    A_I_inv : np.ndarray,
    A_J : np.ndarray,
    c_hat_J : np.ndarray,
    r : np.intp
) -> tuple[np.intp, bool]:
    """
    Finds the index of the variable that enters the basis
    for the dual simplex method.
    Args:
        A_I_inv: the inverse of the matrix of coefficients of the basic variables.
        A_J: the matrix of coefficients of the non-basic variables.
        c_hat_J: the vector of the reduced costs of the non-basic variables.
        r: the index of the variable that leaves the basis.
    Returns:
        k: the index of the variable that enters the basis.
        dual_unbounded: a boolean indicating if the dual problem is unbounded.
    """
    k = np.nan # default for unbounded case
    y_j = A_I_inv @ A_J
    y_rj = y_j[r, :]
    dual_unbounded = bool(np.all(y_rj >= 0))

    # early return if the dual problem is unbounded
    if dual_unbounded:
        return k, dual_unbounded

    # otherwise, find the index of the variable that enters the basis
    mask = y_rj < 0
    ratios = c_hat_J / y_rj
    ratios[~mask] = np.inf
    k = np.argmin(ratios)
    return k, dual_unbounded

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

_is_dual_feasible = lambda c_hat_J: np.all(c_hat_J <= 0)

def _dual_feasible_basis(
    A : np.ndarray,
    b : np.ndarray,
    c : np.ndarray,
    I : list
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list ]:
    """
    Finds a dual feasible basis for the given linear programming problem.
    Args:
        A: the matrix of coefficients of the constraints.
        b: the vector of the right-hand side of the constraints.
        c: the vector of coefficients of the objective function.
        I: the set of indices of the basic variables.
    Returns:
        A_aux: the A matrix with new variables and restrictions
               that make it dual feasible.
        b_aux: the b vector with new variables and restrictions
        c_aux: the c vector with new variables and restrictions
        I : the set of indices of the basic variables
            after the dual feasible basis is found.
        J : the set of indices of the non-basic variables
    """
    A_aux = np.copy(A)
    b_aux = np.copy(b)
    c_aux = np.copy(c)
    m, n = A.shape
    J = _compute_J(n, I)
    _, _, π, _ = _compute_A_I_x_I_π_and_z_0(A, b, c, I)
    c_hat_J = _compute_c_hat_J(π, A, c, J)
    dualfeasible = _is_dual_feasible(c_hat_J)
    # return immediately if the basis is already dual feasible
    if dualfeasible:
        return A_aux, b_aux, c_aux, I, J

    # otherwise, we create a new restriction and add a new slack variable
    # into the A_aux matrix to find a dual feasible basis
    # such restriction is \sum_{j \in J} x_j \leq M
    new_restriction_row = np.zeros((1, n))
    new_restriction_row[0, J] = 1 # \sum_{j \in J} x_j
    new_restriction_column = np.zeros((m + 1, 1))
    new_restriction_column[-1, -1] = 1 # slack variable
    A_aux = np.vstack((A_aux, new_restriction_row))
    A_aux = np.hstack((A_aux, new_restriction_column))
    # adding big M rhs into b_aux array
    big_M = np.abs(np.max(b) * 2**10)
    b_aux = np.hstack((b_aux, big_M))
    # adding new slack variable into c_aux array
    c_aux = np.hstack((c_aux, 0))  # slack variable has zero cost
    # and adding it into the new basis
    I = list(I) + [n]
    # we need to compute the new c_hat_J_aux to find a dual feasible basis
    A_J_aux = _compute_A_J(A_aux, J)
    A_I_aux, x_I_aux, π_aux, z_0_aux = _compute_A_I_x_I_π_and_z_0(A_aux, b_aux, c_aux, I)
    c_hat_J_aux = _compute_c_hat_J(π_aux, A_aux, c_aux, J)

    # then we select the max from c_hat_J_aux >= 0 to enter the basis
    k, _ = _find_k_who_enters_to_keep_dual_feasibility(c_hat_J_aux)

    # we'll remove the new slack variable from the basis
    r = I.index(n)

    # and update the basis
    I, J = _update_I_and_J(I, J, k, r)

def _tableau (A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, J: list, π : np.ndarray) -> np.ndarray:
    """
    Builds the tableau for the simplex method.
    Args:
        A: the matrix of coefficients of the constraints.
        b: the vector of the right-hand side of the constraints.
        c: the vector of coefficients of the objective function.
        I: the set of indices of the basic variables.
        J: the set of indices of the non-basic variables.
        π: the vector of dual variables (shadow prices).
    Returns:
        tableau: the tableau for the simplex method.
    """
    m, n = A.shape
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    A_J = _compute_A_J(A, J)
    c_hat_J = _compute_c_hat_J(π, A, c, J)

    tableau = np.zeros((m + 1, n + 1))
    tableau[0, I] = 0
    tableau[0, J] = c_hat_J
    tableau[0, -1] = π @ b # z_0
    tableau[1:, I] = np.eye(m)
    tableau[1:, J] = A_I_inv @ A_J
    tableau[1:, -1] = A_I_inv @ b
    return tableau

def _compute_w_from_I_when_primal_is_optimal(
    A_I_inv: np.ndarray,
    c_I: np.ndarray
) -> np.ndarray:
    """
    Computes the dual solution w from the primal solution x_I.
    Args:
        A_I_inv: the inverse of the matrix of coefficients of the basic variables.
        c_I: the vector of coefficients of the objective function for the basic variables.
    Returns:
        w: the vector of dual variables (shadow prices).
    """
    return c_I @ A_I_inv

def dual_simplex(A, b, c, I):
    """
    Solves the LP problem using the dual simplex method.
    Assuming that the initial basis is dual feasible.
    Args:
        A: the matrix of coefficients of the constraints.
        b: the vector of the right-hand side of the constraints.
        c: the vector of coefficients of the objective function (minimize).
        I: the set of indices of the basic variables.
    Returns:
        z_star: the optimal value of the objective function,
        x_star: the optimal solution,
        w_star: the optimal dual solution,
        I_star: optimal basis indices,
        iters: number of iterations
        type: 1 - optimal finite solution found,
              2 - multiple optimal solutions found,
              3 - primal infeasible, dual unbounded.
        steps: a dictionary with information for each step
    """
    m, n = A.shape
    iters = 0
    solution_type = 0
    z_star = np.nan
    x_star = np.full(n, np.nan)
    w_star = np.full(m, np.nan)
    I_star = np.full(n, np.nan)
    solution_found = False
    J = _compute_J(n, I)
    steps = {}
    while not solution_found:
        # assuming that the initial basis is dual feasible
        # 1. Compute A_I, A_I_inv, J, A_J, x_I, π, and z_0
        A_I, x_I, π, z_0 = _compute_A_I_x_I_π_and_z_0(A, b, c, I)
        A_I_inv = np.linalg.inv(A_I)

        # compute tableau for the current iteration
        T = _tableau(A, b, c, I, J, π)

        # 2. Compute A_J, c_hat_J
        A_J = _compute_A_J(A, J)
        c_hat_J = _compute_c_hat_J(π, A, c, J)
        # 3. update solution found and if it is the case, break
        solution_found = np.all( c_hat_J <= 0) and np.all(x_I >= 0)
        if solution_found:
            solution_type = 1 if np.all(c_hat_J < 0) else 2
            break
        # 4. Else, find the variable r to leave the basis
        #    and k to replace it
        r = _find_r_who_leaves(I, x_I)
        k, dual_unbounded = _find_k_who_enters(
            A_I_inv = A_I_inv,
            A_J = A_J,
            c_hat_J = c_hat_J,
            r = r
        )
        # 5. If the dual problem is unbounded, break
        if dual_unbounded:
            solution_type = 3
            solution_found = True
            break

        # Otherwise
        # 6. store current step
        steps[iters] = {
            'T': T,
            'to_enter': J[k],
            'to_leave': I[r]
        }

        # 7. update I and J
        I, J = _update_I_and_J(I, J, k, r)

        # 8. Increment the iterations count
        iters += 1
    # store last step
    steps[iters] = {
        'T': T,
        'to_enter': np.nan,
        'to_leave': np.nan
    }

    # evaluate the final solution
    if solution_type in [1, 2]:
        z_star = z_0
        x_star = _compute_x_from_x_I(n, I, x_I)
        w_star = _compute_w_from_I_when_primal_is_optimal(A_I_inv, c[I])
        I_star = I

    return z_star, x_star, w_star, I_star, iters, solution_type, steps
