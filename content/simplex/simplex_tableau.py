'''
    This is a simplex method implementation trhough tableau method
    where it is assumed that the given problem is in standard form,
    the objective function is to be minimized and the constraints are
    of the form Ax = b where A is a m x n matrix, x is a n x 1 vector.
    It is also assumed that the initial basis is feasible.
'''

from typing import Dict, Tuple, Literal
from copy import deepcopy
import numpy as np

def _compute_c_hat_J(π: np.ndarray, A: np.ndarray, c: np.ndarray, J: list) -> np.ndarray:
    A_J = A[:, J]
    c_hat_J = π.dot(A_J) - c[J]
    return c_hat_J

def find_J_from_I(A, I):
    _, n = A.shape
    # np 1d difference
    J = np.setdiff1d(np.arange(n), I)
    return list(J)

def _initial_tableau(A, B, C, I):
    '''
        This function computes the initial tableau for the simplex tableau
        method. The initial tableau is a matrix of shape (m + 1, n + 1) where
        the first row is the cost vector, the next m rows are the constraint
        matrix and the last column is the right hand side of the constraints.

        Arguments:
            A : np.ndarray : m x n matrix
            B : np.ndarray : m x 1 matrix
            C : np.ndarray : 1 x n matrix
    '''
    m, n = A.shape

    # the simplex tableau consists of a matrix T shaped as
    # [[m + 1, n + 1]] where the first row is the cost vector
    # the next m rows are the constraint matrix and the last
    # column is the right hand side of the constraints

    # start a zeroes matrix of shape (m + 1, n + 1) to serve
    # as the tableau
    T = np.zeros((m + 1, n + 1))

    # the cost vector is determined by ĉ_j = πA_j - c_j where π
    # is the cost vector of the basic variables (all zeroes at the
    # beginning) and c_j is the cost vector of the non-basic variables
    J = find_J_from_I(A, I)
    c_hat_J = _compute_c_hat_J(np.zeros(m), A, C, J)
    T[0, J] = c_hat_J
    T[0, -1] = 0
    # put the constraint matrix in the tableau
    T[1:, :-1] = A
    # put the right hand side of the constraints in the tableau
    T[1:, -1] = B.flatten()

    return T

def _find_k_to_enter(T: np.ndarray) -> Tuple[np.intp, np.float64]:
    '''
        This function finds the index of the variable to enter the basis
        by finding the index of the most negative value in the cost vector.
        Arguments:
            T : np.ndarray : m x n matrix
        Returns:
            k : int : Index of the variable to enter the basis,
            c_hat_k: float: value of the maximum value in the cost vector
    '''
    c_hat = T[0, :-1]
    k = np.argmax(c_hat)
    return k, c_hat[k]

def _find_r_to_leave(T: np.ndarray, k: np.intp) -> Tuple[np.intp, np.ndarray, np.ndarray]:
    '''
        This function finds the index of the variable to leave the basis
        by finding the index of the minimum ratio of the right hand side
        to the entering variable in the tableau.
        Arguments:
            T : np.ndarray : m x n matrix
            k : int : Index of the variable to enter the basis
        Returns:
            r : int : Index of the variable to leave the basis
            ratios : np.ndarray : Ratios of the right hand side to the entering variable
            y_k : np.ndarray : Values of the entering variable in the tableau
    '''
    y_k = T[1:, k]
    b = T[1:, -1]
    ratios: np.ndarray = np.where(
        y_k > 0, b / y_k, np.inf
    )
    r = np.argmin(ratios) + 1
    return r, ratios, y_k

def _handle_pivot_row(T: np.ndarray, r: np.intp, k: np.intp) -> Tuple[np.ndarray, tuple]:
    '''
        This function handles the pivot row operation by dividing
        the pivot row by the pivot element.
        Arguments:
            T : np.ndarray : m x n matrix
            r : int : Index of the variable to leave the basis
            k : int : Index of the variable to enter the basis
        Returns:
            T : np.ndarray : m x n matrix
    '''
    row_pivot_operations = (r, np.copy(T[r, k]))
    T[r, :] = T[r, :] / T[r, k]
    return T, row_pivot_operations

def _handle_pivot_column(T: np.ndarray, r: np.intp, k: np.intp) -> Tuple[np.ndarray, list]:
    '''
        This function handles the pivot column operation by
        performing the row operations to make the pivot element
        zero in all other rows except the pivot row.
        Arguments:
            T : np.ndarray : m x n matrix
            r : int : Index of the variable to leave the basis
            k : int : Index of the variable to enter the basis
        Returns:
            T : np.ndarray : m x n matrix
    '''
    m, _ = T.shape
    column_pivot_operations = []
    for i in range(m):
        if i == r:
            continue
        column_pivot_operations.append(
            (i, np.copy(T[i, k]), r)
        )
        T[i, :] = T[i, :] - T[i, k] * T[r, :]
    return T, column_pivot_operations

def _apply_pivot(T: np.ndarray, r: np.intp, k: np.intp) -> Tuple[np.ndarray, dict]:
    '''
        This function applies the pivot operation to the tableau
        by performing the pivot row and pivot column operations.
        Arguments:
            T : np.ndarray : m x n matrix
            r : int : Index of the variable to leave the basis
            k : int : Index of the variable to enter the basis
        Returns:
            T : np.ndarray : m x n matrix
    '''
    T, row_pivot_operations = _handle_pivot_row(T, r, k)
    T, column_pivot_operations = _handle_pivot_column(T, r, k)
    return T, {
        'row_pivot_operations': row_pivot_operations,
        'column_pivot_operations': column_pivot_operations
    }

def _evaluate_c_hat_k(c_hat_k: np.float64) -> Tuple[np.bool, Literal[1, 2] | None]:
    '''
        This functions evaluates the value of the cost vector at the
        entering variable. If the value is positive, it means that the
        no optimal solution has been found and the algorithm should
        continue. If the value is negative, it means that the optimal
        solution has been found and the algorithm should stop, returning
        the type of solution: 1 for single optimal solution and 2 for
        multiple optimal solutions.
        Arguments:
            c_hat_k : float : Value of the cost vector at the entering variable
        Returns:
            proceed : bool : Boolean value indicating whether the algorithm should
            continue or not
            solution_type : int : Type of solution
    '''
    proceed  = c_hat_k > 0
    solution_type = None if proceed else 1 if c_hat_k < 0 else 2
    return proceed, solution_type

def _evaluate_y_k(y_k: np.ndarray) -> Tuple[np.bool, Literal[3] | None]:
    '''
        This function evaluates the values of the entering variable in the
        tableau. If all values are negative, it means that the problem is
        unbounded and the algorithm should stop, returning the type of solution
        as 3. Otherwise, the algorithm should continue.
        Arguments:
            y_k : np.ndarray : Values of the entering variable in the tableau
        Returns:
            proceed : bool : Boolean value indicating whether the algorithm should
            continue or not
            solution_type : int : Type of solution
    '''
    proceed = np.any(y_k > 0)
    solution_type = None if not proceed else 3
    return proceed, solution_type

def _evaluate_solution_by_c_hat_k_and_y_k(c_hat_k: np.float64, y_k: np.ndarray) -> int:
    '''
        This function evaluates the solution by the values of the cost vector
        at the entering variable and the values of the entering variable in the
        tableau. If the cost vector value is positive, it means that the optimal
        solution has been found and the algorithm should stop, returning the type
        of solution as 1. If the cost vector value is negative and all values of
        the entering variable are negative, it means that the problem is unbounded
        and the algorithm should stop, returning the type of solution as 3. Otherwise,
        the algorithm should continue.
        Arguments:
            c_hat_k : float : Value of the cost vector at the entering variable
            y_k : np.ndarray : Values of the entering variable in the tableau
        Returns:
            solution_type : int : Type of solution
    '''
    _, solution_type_hat_k = _evaluate_c_hat_k(c_hat_k)
    _, solution_type_y_k = _evaluate_y_k(y_k)
    return max(
        solution_type_hat_k or 0,
        solution_type_y_k or 0
    )

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
    to_enter = J[k]
    to_exit = I[r]
    I[r] = to_enter
    J[k] = to_exit
    return I, J

def simplex_tableau(
    A : np.ndarray,
    B : np.ndarray,
    C : np.ndarray,
    I : list
    ) -> Dict:
    '''
        This function solves the linear programming problem
        min cx subject to Ax = b, x >= 0
        using the simplex method through tableau method, assuming
        that the initial basis is feasible as well as the problem
        Arguments:
            A : np.ndarray : m x n matrix
            B : np.ndarray : m x 1 matrix
            C : np.ndarray : 1 x n matrix
            I : list : List of indices of the initial basis
        Returns:
            Solution : Dict : Dictionary containing the
            optimal solution, the optimal value, the
            optimal basis, the solution type  and the tableau
            at each iteration step.
    '''
    # define necessary variables
    m, _ = A.shape
    c_hat_k = np.max(C)
    y_k = np.zeros(m)
    proceed = True
    iteration = 0
    J = find_J_from_I(A, I)
    solution = {}
    tableau_steps = {}
    # 1. build the initial tableau
    T = _initial_tableau(A, B, C, I)
    tableau_steps[iteration] = {
        'previous_T': np.copy(T),
        'T': np.copy(T),
        'previous_I': deepcopy(I),
        'previous_J': deepcopy(J),
        'I': deepcopy(I),
        'J': deepcopy(J),
        'to_enter': None,
        'to_leave': None,
        'c_hat_j': None,
        'c_hat_k': None,
        'y_k': None,
        'ratios': None,
        'pivot_operations': None
    }
    # 2. And then proceed to solve the problem
    while proceed:
        # 2.1 find the index of the variable to enter the basis
        k, c_hat_k = _find_k_to_enter(T)
        # 2.2 find the index of the variable to leave the basis
        r, ratios, y_k = _find_r_to_leave(T, k)
        # 2.3 check if the optimal solution has been found when
        # computing the entering variable
        proceed, _ = _evaluate_c_hat_k(c_hat_k)
        # 2.4 check if the problem is unbounded when computing
        # the leaving variable
        proceed, _ = _evaluate_y_k(y_k)
        # 2.5 if no optimal solution has been found, apply the pivot
        # operation and continue
        if proceed:
            # 0. store the previous tableau
            previous_T = np.copy(T)
            # 1. apply the pivot operation
            T, pivot_operations = _apply_pivot(T, r, k)
            # 2. update the sets of basic and non-basic variables
            previous_I, previous_J = deepcopy(I), deepcopy(J)
            I, J = _update_I_and_J(I, J, k, r - 1)
            # 3. increment the iteration
            iteration += 1
            # 4. store the tableau at this iteration
            tableau_steps[iteration] = {
                'previous_T': previous_T,
                'T': np.copy(T),
                'previous_I': deepcopy(previous_I),
                'previous_J': deepcopy(previous_J),
                'I': deepcopy(I),
                'J': deepcopy(J),
                'to_enter': f'X_{previous_J[k]}',
                'to_leave': f'X_{previous_I[r - 1]}',
                'c_hat_j': np.copy(T[0, :-1]),
                'c_hat_k': np.max(T[0, :-1]),
                'y_k': np.copy(y_k),
                'ratios': np.copy(ratios),
                'pivot_operations': deepcopy(pivot_operations)
            }
    # After solving the problem evaluate the solution
    solution_type = _evaluate_solution_by_c_hat_k_and_y_k(c_hat_k, y_k)
    # build the solution dictionary
    solution = {
        'solution_type': solution_type,
        'Z_star': T[0, -1],
        'X_star': T[0, :-1],
        'I_star': I,
        'tableaus': tableau_steps
    }
    return solution

def _repr_row_pivot_operations(row_pivot_operations: tuple, labels: list) -> str:
    if row_pivot_operations is None:
        return ''
    r, pivot_row_multiplier = row_pivot_operations
    return f'\\\\\\\\(R_{{{labels[r]}}} = \\\\frac{{R_{{{labels[r]}}}}}{{{pivot_row_multiplier}}}\\\\\\\\)'

def _repr_column_pivot_operations(column_pivot_operations: list, labels : list) -> list:
    column_pivot_operations_list = []
    for i, pivot_column_multiplier, r in column_pivot_operations:
        if pivot_column_multiplier >= 0:
            column_pivot_operations_list.append(
                f'\\\\\\\\(R_{{{labels[i]}}} = R_{{{labels[i]}}} - {pivot_column_multiplier}R_{{{labels[r]}}}\\\\\\\\)'
            )
        else:
            column_pivot_operations_list.append(
                f'\\\\\\\\(R_{{{labels[i]}}} = R_{{{labels[i]}}} + {-pivot_column_multiplier}R_{{{labels[r]}}}\\\\\\\\)'
            )
    return column_pivot_operations_list

def _markdown_pivot_operations(to_enter, to_leave, I: list, pivot_operations: dict) -> str:
    labels = ['C'] + [
        f'X_{i}' for i in I
    ]
    pivot_operations_str  = '### Pivot Operations\n\n'
    pivot_operations_str += 'Variable to enter: \\\\\\\\(' + (to_enter or '') + '\\\\\\\\)\n'
    pivot_operations_str += 'Variable to leave: \\\\\\\\(' + (to_leave or '') + '\\\\\\\\)\n'
    pivot_operations_str += '#### Row Operations:\n\n'
    pivot_operations_str += '1. ' + _repr_row_pivot_operations(pivot_operations['row_pivot_operations'], labels) + '\n'
    for i, column_pivot_operation in enumerate(_repr_column_pivot_operations(pivot_operations['column_pivot_operations'], labels)):
        pivot_operations_str += f'{i + 1}. {column_pivot_operation}\n'
    return pivot_operations_str

def _markdown_T(T: np.ndarray, I: list) -> str:
    table_header = '|   |' + ' | '.join([f'\\\\\\\\(X_{i}\\\\\\\\)' for i in range(T.shape[1] - 1)]) + ' | RHS |\n'
    table_header += '|---|' + '---|' * (T.shape[1] - 1) + '---|\n'

    row_names = ['Cost'] + [f'\\\\\\\\(X_{i}\\\\\\\\)' for i in I]

    table_body = ''
    for i, row in enumerate(T):
        table_body += f'| {row_names[i]} | ' + ' | '.join([f'{value:.3f}' for value in row]) + ' |\n'
        
    return table_header + table_body

def markdown_repr_T(tableau_steps) -> str:
    markdown = ''
    for iteration in tableau_steps:
        # skip the first iteration
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
    return markdown

def simplex(A : np.ndarray, B : np.ndarray, C : np.ndarray, I : list) -> Dict:
    '''
        This function solves the linear programming problem
        min cx subject to Ax = b, x >= 0
        using the simplex method through tableau method, assuming
        that the initial basis is feasible as well as the problem
        Arguments:
            A : np.ndarray : m x n matrix
            B : np.ndarray : m x 1 matrix
            C : np.ndarray : 1 x n matrix
            I : list : List of indices of the initial basis
        Returns:
            Solution : Dict : Dictionary containing the
            optimal solution, the optimal value, the
            optimal basis, the solution type  and the tableau
            at each iteration step.
    '''
    solution = simplex_tableau(A, B, C, I)
    return solution

