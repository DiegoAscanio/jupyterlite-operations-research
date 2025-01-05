
from typing import Dict, Tuple
import numpy as np
import pdb

def _initialization_step(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I: list
) -> np.ndarray:
    # variables
    m, _ = A.shape
    A_I = A[:, I]
    A_I_inv = np.linalg.inv(A_I)
    C_I = C[I]
    T = np.zeros((m + 1, m + 1))
    B_prime = A_I_inv @ B

    # fill T
    # 1. w
    T[0, :-1] = C_I @ A_I_inv
    # 2. RHS
    T[0, -1] = C_I @ B_prime
    T[1:, -1] = B_prime.flatten()
    # 3. Basis inverse
    T[1:, :-1] = A_I_inv

    return T

def _J_from_I(I : list, n : np.intp) -> list:
    return [i for i in range(n) if i not in I]

def _update_I_and_J(I : list, J : list, k : np.intp, r: np.intp) -> Tuple[list, list]:
    I_new = I.copy()
    J_new = J.copy()
    aux = I_new[r]
    I_new[r] = J_new[k]
    J_new[k] = aux
    return I_new, J_new

def _find_k_to_enter(
    w : np.ndarray,
    A : np.ndarray,
    C : np.ndarray,
    J : list,
) -> Tuple[np.intp, np.float64, np.ndarray]:
    c_hat_J = np.round(w @ A[:, J] - C[J], 6) # rounding to ensure that the first max will be the selected one
    k = np.argmax(c_hat_J)
    c_hat_k = c_hat_J[k]
    return k, c_hat_k, c_hat_J

def _find_r_to_leave_cycle_allowed(B_prime: np.ndarray, y_k: np.ndarray) -> Tuple[np.intp, np.ndarray]:
    ratios = np.where(y_k > 0, B_prime / y_k, np.inf)
    return np.argmin(ratios), ratios

def _find_r_to_leave_cycle_proof(
    A : np.ndarray,
    A_I_inv : np.ndarray,
    B_prime: np.ndarray,
    y_k: np.ndarray,
    k : np.intp
):
    _, n = A.shape
    columns = iter(set(range(n)) - set([k]))
    ratios = np.where(y_k > 0, B_prime / y_k, np.inf)
    aux_ratios = np.copy(ratios)

    # start lexicographic rule
    minimum_value = np.min(ratios)
    minimum_values_indices, *_ = np.where(ratios == minimum_value)
    candidate_indices_to_leave = np.copy(minimum_values_indices)

    # as in revised tableau we try to compute r to leave anyway
    # even after an optimal was reached, when this happens, all
    # y_k become leq 0, so ratios we'll become an array of inf
    # then, candidate_indices we'll have a length of I, as all
    # ratios are infinity. So that's why we added this or
    # to the implementation. Not the most elegant solution.
    # this one would be to refactor main step into a pipeline
    # where find_r_to_leave would be decorated through a pipeline
    # function who would receive proceed from _find_k_to_enter as 
    # a parameter and only runs if proceed is True.
    # but as we're losing a lot of time trying to prevent cycles
    # and fixing the bugs that naturally happens when doing so,
    # we'll leave this solution as it is. We should prefer simplicity
    # over complexity and this is simple enough to guarantee the proper
    # execution of our code.
    singleton = len(candidate_indices_to_leave) == 1 or minimum_value == np.inf
    # we can set r to the first element of candidate_indices_to_leave as
    # if it is a singleton, it will be the only element in the array
    # but if it is not, r will be updated inside the loop until we find
    # a singleton set
    r = candidate_indices_to_leave[0]
    while not singleton:
        c = next(columns)
        y_c = A_I_inv @ A[:, c]
        aux_ratios = np.where(y_k > 0, y_c / y_k, np.inf)
        minimum_value = np.min(aux_ratios)
        minimum_values_indices = np.where(aux_ratios == minimum_value)
        singleton = len(minimum_values_indices) == 1
        r = candidate_indices_to_leave[np.argmin(aux_ratios)]
        # As our A_I matrix is non-singular, we are guaranteed to find a singleton
        # so, the last computed r is guaranteed to come from a singleton set
        # as the last computed aux_ratios will produce a singleton min
        # safely retrieve it as a variable to leave the basis
    return r, ratios

def _find_r_to_leave(
    A : np.ndarray,
    A_I_inv : np.ndarray,
    B_prime,
    y_k: np.ndarray,
    k : np.intp,
    cycle_proof = True
):
    return _find_r_to_leave_cycle_proof(
        A,
        A_I_inv,
        B_prime,
        y_k,
        k
    ) if cycle_proof else _find_r_to_leave_cycle_allowed(
        B_prime, y_k
    )
    

def _row_operation(T_prime: np.ndarray, r: np.intp, I : list) -> Tuple[np.ndarray, tuple]:
    pivot = T_prime[r, -1]
    toleave = I[r - 1] # minus one as we're receiving an r-index appropriate for T_prime
    T_prime[r, :] = T_prime[r, :] / pivot # normalize row
    return T_prime, (toleave, pivot)

def _column_operation(T_prime: np.ndarray, r: np.intp, I : list) -> Tuple[np.ndarray, list]:
    toleave = I[r - 1] # minus one as we're receiving an r-index appropriate for T_prime
    column_operations = []
    m, _ = T_prime.shape
    for i in range(m):
        if i != r:
            multiplier = T_prime[i, -1]
            T_prime[i, :] = T_prime[i, :] - multiplier * T_prime[r, :]
            column_operations.append((i, multiplier, toleave))
    return T_prime, column_operations

def _revised_pivot(T_prime: np.ndarray, k : np.intp, r : np.intp, I: list, J: list) -> Tuple[np.ndarray, list, list, dict]:
    # increase r in one as y_k vector occupies the last column in T_prime from
    # the second row below
    r = r + 1

    T_prime, row_operation = _row_operation(T_prime, r, I)
    T_prime, column_operations = _column_operation(T_prime, r, I)
    I, J = _update_I_and_J(I, J, k, r - 1) # minus one as we're receiving an r-index appropriate for T_prime
    pivot_operations = {
        'row_pivot_operations': row_operation,
        'column_pivot_operations': column_operations
    }
    return T_prime, I, J, pivot_operations

def _build_X_I(A_I_inv: np.ndarray, B : np.ndarray) -> np.ndarray:
    return A_I_inv @ B

def _build_X(A: np.ndarray, A_I_inv: np.ndarray, B: np.ndarray, I: list) -> np.ndarray:
    _, n = A.shape
    x = np.zeros(n)
    x[I] = _build_X_I(A_I_inv, B)
    return x

def _main_step(
    T: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I: list,
    J: list,
    cycle_proof = True
) -> Tuple[Dict, bool]:
    # variables
    proceed = True
    step_operations = {}
    step_operations['I'] = I.copy()
    step_operations['J'] = J.copy()
    step_operations['previous_T'] = np.copy(T)
    A_I_inv = T[1:, :-1]
    step_operations['A_I_inv'] = A_I_inv

    # 1. Find k to enter
    w = T[0, :-1]
    k, c_hat_k, c_hat_J = _find_k_to_enter(w, A, C, J)
    y_k = A_I_inv @ A[:, J[k]]
    step_operations['k'] = k
    step_operations['J[k]'] = J[k]
    step_operations['c_hat_k'] = c_hat_k
    step_operations['c_hat_J'] = c_hat_J
    step_operations['y_k'] = y_k

    # 2. Find r to leave
    # 2.1 Retrieve B_prime
    B_prime = T[1:, -1]
    # 2.2 Now find r
    r, ratios = _find_r_to_leave(
        A,
        A_I_inv,
        B_prime,
        y_k,
        J[k],
        cycle_proof
    )
    step_operations['r'] = r
    step_operations['ratios'] = ratios

    # 3. Optimality checks
    step_operations['solution_type'] = 1 if c_hat_k < 0 else 2 if c_hat_k == 0 else 3 if np.all(y_k <= 0) else None
    # proceed if and only if we're not at an optimal solution
    proceed = step_operations['solution_type'] is None

    # 4. Pivot
    if proceed:
        # 4.1 Create a T_prime first
        T_prime = np.hstack((np.copy(T), np.zeros((T.shape[0], 1))))
        T_prime[0, -1] = c_hat_k
        T_prime[1:, -1] = y_k.flatten()
        # 4.2 Pivot
        step_operations['T_prime'] = np.copy(T_prime)
        T_prime, I, J, pivot_operations = _revised_pivot(T_prime, k, r, I, J)
        step_operations['Computed I'] = I.copy()
        step_operations['Computed J'] = J.copy()
        step_operations['pivot_operations'] = pivot_operations
        # 4.3 Update T
        T = T_prime[:, :-1]
    step_operations['X_I'] = _build_X_I(A_I_inv, B)
    step_operations['X'] = _build_X(A, A_I_inv, B, I)
    step_operations['T'] = np.copy(T)
    step_operations['proceed'] = proceed
    return step_operations, proceed
    
def revised_simplex_tableau(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    I: list,
    max_iter: int = 100,
    cycle_proof = True
) -> Dict:
    # variables
    m, n = A.shape
    proceed = True
    iterations = 0
    revised_tableau_steps = {}
    I = I.copy()
    J = _J_from_I(I, n)
    
    # 1. perform initialization step
    T = _initialization_step(A, B, C, I)

    # 2. then the main steps
    while proceed:
        step_operations, proceed = _main_step(T, A, B, C, I, J, cycle_proof)
        revised_tableau_steps[iterations] = step_operations
        iterations += 1
        proceed = proceed and iterations < max_iter
        # update T, I, J
        T = step_operations['T']
        I = step_operations.get('Computed I', I)
        J = step_operations.get('Computed J', J)
    
    # Add C, B and A to the final step
    revised_tableau_steps[iterations - 1]['C'] = C
    revised_tableau_steps[iterations - 1]['B'] = B
    revised_tableau_steps[iterations - 1]['A'] = A

    # if exceeded max_iters we assume that
    # simplex entered in loop
    if iterations >= max_iter:
        revised_tableau_steps[iterations - 1]['solution_type'] = -2
    return revised_tableau_steps

def _repr_row_pivot_operations(row_pivot_operations: tuple, labels: list, I : list) -> str:
    if row_pivot_operations is None:
        return ''
    r, pivot_row_multiplier = row_pivot_operations
    return f'\\\\(R_{{{labels[I.index(r) + 1]}}} \\leftarrow \\frac{{R_{{{labels[I.index(r) + 1]}}}}}{{{pivot_row_multiplier:.3f}}}\\\\)'

def _repr_column_pivot_operations(column_pivot_operations: list, labels : list, I : list) -> list:
    column_pivot_operations_list = []
    for i, pivot_column_multiplier, r in column_pivot_operations:
        if pivot_column_multiplier >= 0:
            column_pivot_operations_list.append(
                    f'\\\\(R_{{{labels[i]}}} \\leftarrow R_{{{labels[i]}}} - {pivot_column_multiplier:.3f}R_{{{labels[I.index(r) + 1]}}}\\\\)'
            )
        else:
            column_pivot_operations_list.append(
                    f'\\\\(R_{{{labels[i]}}} \\leftarrow R_{{{labels[i]}}} + {-pivot_column_multiplier:.3f}R_{{{labels[I.index(r) + 1]}}}\\\\)'
            )
    return column_pivot_operations_list

def _markdown_pivot_operations(to_enter, to_leave, I: list, pivot_operations: dict) -> str:
    labels = ['Z'] + [
        f'X_{{{i}}}' for i in I
    ]
    pivot_operations_str  = '### Pivot Operations\n\n'
    pivot_operations_str += 'Variable to enter: \\\\(' + (to_enter or '') + '\\\\)\n'
    pivot_operations_str += 'Variable to leave: \\\\(' + (to_leave or '') + '\\\\)\n'
    pivot_operations_str += '#### Row Operations:\n\n'
    pivot_operations_str += '1. ' + _repr_row_pivot_operations(pivot_operations['row_pivot_operations'], labels, I) + '\n'
    for i, column_pivot_operation in enumerate(_repr_column_pivot_operations(pivot_operations['column_pivot_operations'], labels, I)):
        pivot_operations_str += f'{i + 1}. {column_pivot_operation}\n'
    return pivot_operations_str

def _markdown_T_prime(T_prime: np.ndarray, k: np.intp, I: list, J : list) -> str:
    columns_length = T_prime.shape[1] - 2
    table_header = '|   |' + ' | '.join([
        ' ' if i != columns_length // 2 else '\\\\(A_I^{-1}\\\\)' for i in range(columns_length)
    ]) + f' | RHS | \\\\(X_{{{J[k]}}}\\\\)\n'
    table_header += '|---|' + '---|' * (T_prime.shape[1] - 1) + '---|\n'

    row_names = ['Z'] + [f'\\\\(X_{{{i}}}\\\\)' for i in I]

    table_body = ''
    for i, row in enumerate(T_prime):
        table_body += f'| {row_names[i]} | ' + ' | '.join([f'{value:.3f}' for value in row]) + ' |\n'
        
    return table_header + table_body

def _markdown_T(T: np.ndarray, I: list) -> str:
    columns_length = T.shape[1] - 1
    table_header = '|   |' + ' | '.join([
        ' ' if i != columns_length // 2 else '\\\\(A_I^{-1}\\\\)' for i in range(columns_length)
    ]) + ' | RHS |\n'
    table_header += '|---|' + '---|' * (T.shape[1] - 1) + '---|\n'

    row_names = ['Z'] + [f'\\\\(X_{{{i}}}\\\\)' for i in I]

    table_body = ''
    for i, row in enumerate(T):
        table_body += f'| {row_names[i]} | ' + ' | '.join([f'{value:.3f}' for value in row]) + ' |\n'
        
    return table_header + table_body

_solution_types_map = {
    1: 'Optimal unique solution',
    2: 'Optimal multiple solutions',
    3: 'Unbounded solution',
    -1: 'Infeasible solution',
    -2: 'Entered in loop'
}

def _markdown_final_step(last_step : dict, iteration) -> str:
    solution_type = _solution_types_map[last_step['solution_type']]
    n = len(last_step['C'])
    I_star = last_step['I']
    J = last_step['J']
    c_hat_J = last_step['c_hat_J']
    z_star = last_step['T'][0, -1]
    x_star = np.zeros(n)
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
    & & X & \\geq 0
    \\end{{aligned}}
    \\\\]
    where: 
    \\\\[
    \\begin{{aligned}}
    & & A = {restrictions_matrix_latex_format} \\\\
    & & B = {b_latex_format} \\\\
    & & C^{{T}} = {c_latex_format} \\\\
    & & X = {x_matrix_latex_format}
    \\end{{aligned}}
    \\\\]\n\n'''
    markdown  = f'\n\n## Solution for \n\n'
    markdown += statement
    markdown += f'Solution Type: {solution_type}\n\n'
    markdown += f'Optimal Solution: \\\\(X^{{*}} = {[f'{x:.3f}' for x in x_star]}\\\\)\n\n'
    markdown += f'Optimal Value: \\\\(Z^{{*}} = {z_star:.3f}\\\\)\n\n'
    markdown += f'Optimal Basis: \\\\({I_star}\\\\)\n\n'
    markdown += f'Final non-basic variables set J: \\\\({J}\\\\)\n\n'
    markdown += f'\\\\(\\hat{{C}}_{{J}}\\\\): \\\\({c_hat_J.tolist()}\\\\)\n\n'
    markdown += f'Number of Iterations: {iteration + 1}\n\n'

    return markdown

def markdown_repr_T(tableau_steps) -> str:
    markdown = ''
    for iteration in tableau_steps:
        previous_I = tableau_steps[iteration]['I']
        previous_J = tableau_steps[iteration]['J']
        previous_T = tableau_steps[iteration]['previous_T']
        pivot_operations = tableau_steps[iteration].get('pivot_operations', None)

        markdown += f'\n\n## Iteration {iteration}\n\n'
        markdown += 'Starting Tableau:\n\n'
        markdown += _markdown_T(
            previous_T,
            previous_I
        )

        if pivot_operations:
            markdown += '\n\nExtended Tableau\n\n'
            markdown += _markdown_T_prime(
                tableau_steps[iteration]['T_prime'],
                tableau_steps[iteration]['k'],
                tableau_steps[iteration]['I'],
                tableau_steps[iteration]['J']
            )
            markdown += f'\n\n\\\\(Ĉ_J = \\ \\\\){repr(tableau_steps[iteration]['c_hat_J'])}\n'
            markdown += f'\n\n\\\\(J = {{{tableau_steps[iteration]['J']}}}\\\\)\n\n'
            markdown += _markdown_pivot_operations(
                f'X_{{{previous_J[tableau_steps[iteration]['k']]}}}',
                f'X_{{{previous_I[tableau_steps[iteration]['r']]}}}',
                tableau_steps[iteration]['I'],
                tableau_steps[iteration]['pivot_operations']
            )
            markdown += '\n\nComputed Tableau:\n\n'
            markdown += _markdown_T(
                tableau_steps[iteration]['T'],
                tableau_steps[iteration]['Computed I']
            )
    markdown += _markdown_final_step(tableau_steps[iteration], iteration)
    return markdown
