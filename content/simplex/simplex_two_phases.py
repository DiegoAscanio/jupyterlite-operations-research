from types import SimpleNamespace
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
    J = np.setdiff1d(np.arange(n), I).tolist()
    return J

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

def _find_k_who_enters(c_hat_J: np.ndarray) -> tuple[np.intp, bool]:
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

def _find_r_who_leaves(I: list, A_I_inv: np.ndarray, x_I: np.ndarray, A_k: np.ndarray, debug = False) -> tuple[np.intp, np.ndarray, bool, dict]:
    """
    Finds the index of the variable that leaves the basis and returns if
    the simplex method shouuld continue. When all elements of A_k are
    non-positive, the method found an unbounded solution and, therefore,
    should stop.
    Args:
        I: list of basic variables
        A_I_inv: the inverse of the matrix of the basic variables.
        x_I: the vector of the basic variables.
        A_k: the vector of the coefficients of the entering variable.
        debug: a boolean variable to print the steps of the simplex method.
    Returns:
        r: the index of the variable that leaves the basis.
        y_k: the vector of the coefficients of the entering variable.
        proceed_: a boolean indicating if the simplex method should proceed.
                  if y_k[r] <= 0, the solution is unbounded and, therefre,
                  the method should stop.
        debug_info: a dictionary with debug information.
    """
    debug_info = {}

    # 1. compute y_k
    y_k = A_I_inv.dot(A_k)

    ratios = np.where(y_k > 0, x_I / y_k, np.inf)

    # 2. Variable index to leave the basis
    r = np.argmin(ratios)

    # 3. Append debug information
    if debug:
        debug_info['description'] = 'Finding the variable that leaves the basis'
        debug_info['y_k'] = y_k
        debug_info['ratios'] = ratios
        debug_info['r'] = r
        debug_info['I_r'] = I[r]

    # 6. Return the variable index to leave the basis and if the simplex method should proceed
    return r, y_k, np.any(y_k > 0), debug_info

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
    print(f'''
    k: {k}
    J[k]: {J[k]}
    r: {r}
    I[r]: {I[r]}
    Before J: {J}
    Before I: {I}
    ''')
    I = deepcopy(I)
    J = deepcopy(J)
    to_enter = deepcopy(J[k])
    to_exit = deepcopy(I[r])
    I[r] = to_enter
    J[k] = to_exit
    print(f'''
    k: {k}
    J[k]: {J[k]}
    r: {r}
    I[r]: {I[r]}
    After J: {J}
    After I: {I}
    ''')
    return I, J

def _simplex_find_feasible_initial_basis(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, debug = False) -> tuple[Any, np.ndarray, np.ndarray, np.ndarray, bool, int, dict]:
    """
    Finds a feasible initial basis for the 2-phase simplex method.
    This is the first phase of the 2-phase simplex method.
    Args:
        A: the matrix of coefficients of the constraints.
        b: the vector of the right-hand side of the constraints.
        c: the vector of coefficients of the objective function.
        I: the set of indices of the basic variables.
        debug: a boolean variable to print the steps of the simplex method.
    Returns:
        I: the set of indices of the basic variables.
        A_I: the coefficients matrix of the basic variables.
        A: the coefficients matrix of the constraints after the first phase.
        b: the vector of the right-hand side of the constraints after the first phase.
        c: the vector of coefficients of the objective function after the first phase.
        feasible: a boolean indicating if the original problem is feasible.
        iterations_count: the number of iterations needed to find a feasible basis.
        debug_info: a dictionary with debug information.
    """
    # Create copies of the input matrices so we don't modify the original ones
    A = np.copy(A)
    sanitized_A = np.copy(A)
    b = np.copy(b)
    sanitized_b = np.copy(b)
    c = np.copy(c)
    I = deepcopy(I)

    # Initialize auxiliary variables
    debug_info = {}
    m, n = A.shape
    A_I = _compute_A_I(A, I)
    x_I = _compute_X_I(A_I, b)

    # Check if the initial basis is already feasible
    if np.all(x_I > 0):
        return I, A_I, A, b, True, 0, debug_info

    # If not, perform the first phase of the 2-phase simplex method
    """
    Perform first phase - steps:
        1. Create artificial variables
        2. Add them to the coefficients matrix A
        3. Create the artificial objective function φ of the form min 1.X
        4. Solve the auxiliary problem through the simplex method
        The original problem is feasible if φ* = 0, otherwise it is infeasible.
    """
    # 1. Create the artificial variables
    x_I_indices_lesser_than_zero = np.where(x_I < 0)[0]

    artificial_variables = np.array([
        [1 if i == j else 0 for i in range(m)] for j in x_I_indices_lesser_than_zero
    ]).T
    artificial_variables_indices = np.arange(n, n + len(x_I_indices_lesser_than_zero))
    # 2. Add the artificial variables to the coefficients matrix A
    A_artificial = np.hstack((np.copy(A), artificial_variables))
    # 3. Create the artificial objective function
    c_artificial = np.zeros(A_artificial.shape[1])
    c_artificial[n:] = 1
    # 4. Solve the auxiliary problem
    # 4.1. Replace the indices of the artificial variables in the basic variables set
    I_artificial = np.copy(I)
    I_artificial[x_I_indices_lesser_than_zero] = np.arange(n, n + len(x_I_indices_lesser_than_zero))
    # 4.2. call the simplex method
    φ_star, x_star, I_star, A_I_star, A, iterations_count, solution_type, first_phase_debug_info = _simplex_with_feasible_initial_basis(
        A_artificial, b, c_artificial, I_artificial, debug
    )
    # 4.3. Determine original problem feasibility
    feasible = φ_star == 0
    # 3. Update the debug information
    debug_info['first phase'] = {
        'first_phase_debug_info': first_phase_debug_info,
        'φ_star': φ_star,
        'x_star': x_star,
        'I_star': I_star,
        'A_I_star': A_I_star,
        'A': A,
        'A_artificial': A_artificial, 
        'b': b,
        'c_artificial': c_artificial,
        'I_artificial': I_artificial,
        'iterations_count': iterations_count,
        'solution_type': solution_type
    }

    # 4. Analyze the resulting basis
    # If the resulting basis contains some of the artificial variables and the original problem is feasible,
    # we'll try to replace the artificial basic variables by the original non-basic variables

    # 4.1 Check if the original problem is feasible
    if not feasible:
        return I, A_I, A, b, feasible, iterations_count, debug_info

    # 4.2 Check if there are artificial variables are in the basis
    I_restricted_to_artificial_variables_in_I = np.intersect1d(I_star,artificial_variables_indices)

    # If there aren't any artificial variables in the basis, return I_star, A_I_star, A, feasible, debug_info
    artificial_variables_in_basis = len(I_restricted_to_artificial_variables_in_I) > 0
    if not artificial_variables_in_basis:
        return I_star, A_I_star, A, b, feasible, iterations_count, debug_info

    # Otherwise chose an original non-basic variable to replace the artificial basic variable
    # until there are no artificial variables in the basis or if not possible, to suppress
    # the constraints that contain the artificial basic variables which could not be replaced
    counter = 1

    # define a nested function in here to select only artificial variables
    # in the basis for replacement. The criteria for picking an artificial
    # variable to be replaced is the one that has the lowest index in the
    # artificial variables indices list
    def _pick_artificial_variable_in_basis_to_leave(I_star, artificial_variables_indices):
        return np.intp(I_star.index(np.min(np.intersect1d(I_star, artificial_variables_indices))))

    # define a nested function in here to select only non-basic variables
    # to enter the basis for replacement. The criteria for picking a non-basic
    # variable to enter the basis is the one that has the maximum ĉ_j value
    # and is the leftermost in the list of non-basic variables
    def _pick_non_basic_variable_to_enter(
        A_artificial: np.ndarray,
        b_artificial: np.ndarray,
        c_artificial: np.ndarray,
        I: list,
        J: list,
        c_hat_J: np.ndarray,
        r: np.intp,
    ) -> np.intp | None:
        # Here we want to force the selection of a non-basic variable to replace
        # an artificial variable in the basis, even if it worsens the objective
        # function value, as we're interested in finding a feasible basis with 
        # no artificial variables for phase 2 of the 2-phase simplex method
        aux_I = I.copy()
        aux_J = J.copy()
        aux_c_hat_J = np.copy(c_hat_J)
        k = None
        proceed = len(aux_J) > 0
        valid_non_basic_variable_found = False
        while proceed:
            # we try first to find a non-basic variable to enter with the best
            # c_hat value
            k, _ = _find_k_who_enters(aux_c_hat_J)
            # we then check if it is possible to pivot the artificial variable
            # column with the k-th non-basic variable column
            print(f'''
            Inside job:
            J = {aux_J}
            c_hat_J = {aux_c_hat_J}
            k = {k}
            r = {r}
            J[k] = {aux_J[k]}
            I[r] (to leave) = {aux_I[r]}
            A_artificial[r] = {A_artificial[r]}
            A_artificial[r, J[k]] = {A_artificial[r, aux_J[k]]}
            ''')
            if A_artificial[r, aux_J[k]] == 0:
                # if it is not possible, we'll remove k from the non-basic variables
                # and try again
                aux_J.pop(k)
                aux_c_hat_J = np.delete(aux_c_hat_J, k)
            else:
                # then, if it is possible we'll check that the resulting X_I satisfies
                # non-negativity constraints
                # 1. Backup aux_I and aux_J
                old_aux_I = aux_I.copy()
                old_aux_J = aux_J.copy()
                # 2. Update aux_I and aux_J
                aux_I, aux_J = _update_I_and_J(aux_I, aux_J, k, r)
                # 3. Compute X_I
                A_I = _compute_A_I(A_artificial, aux_I)
                # 3.1 If the A_I matrix is singular, we we'll fill the X_I with -inf
                # so we'll force the selection of another non-basic variable
                if np.linalg.det(A_I) == 0:
                    x_I = np.full(A_I.shape[0], -np.inf)
                else: # otherwise, we'll compute the X_I
                    x_I = _compute_X_I(A_I, b_artificial)
                # 4. If any X_I is negative, we'll remove k from the non-basic variables
                # and try again
                if np.any(x_I < 0):
                    aux_I = old_aux_I # restore aux_I
                    aux_J = old_aux_J # restore aux_J
                    aux_J.pop(k)
                    aux_c_hat_J = np.delete(aux_c_hat_J, k)
                # 5. Otherwise, it impplies that we found a non-basic variable to enter
                # the basis that allows us to pivot the artificial variable column with
                # the k-th non-basic variable column while keeping the non-negativity
                # constraints satisfied. So, we can stop the loop
                else:
                    proceed = False
                    valid_non_basic_variable_found = True
            # 6. If there are no more non-basic variables to try, we should stop
            proceed = proceed and len(aux_J) > 0
        # if we've found a variable capable of entering the basis without violating
        # no constraints, old_aux_J will store the value from J that must enter the
        # basis. That happens because we iterate through a copy of J (and make a
        # backup copy of it - old_aux_J) in order to find non basic variables 
        # capable of joining the basis. lines 294 ~ 321 contains this implementation
        k = J.index(old_aux_J[k]) if valid_non_basic_variable_found else None
        return k

    # Compute J for the original non-basic variables
    J = list(
        set(_compute_J(n, I)) - set(I_star)
    )
    J = list(
        set(range(n)) - set(I_star)
    )

    # We're still in phase 1, so we need to compute things taking into
    # account the artificial problem. That's why we will define again
    # A_artificial, c_artificial, I_to_clean and b_artificial, because
    # explicit is better than implicit
    
    A_artificial = np.copy(A_artificial)
    b_artificial = np.copy(b)
    I_to_clean = list(I_star).copy()
    c_artificial = np.copy(c_artificial)

    # The while loop below should run only if there are artificial variables
    # in the basis

    debug_info['constraints_removed'] = []
    while artificial_variables_in_basis:
        # 1. compute A_I, J, A_J, x_I, π
        A_I, x_I, π, _ = _compute_A_I_x_I_π_and_z_0(A_artificial, b_artificial, c_artificial, I_to_clean)
        # 2. compute c_hat_J
        c_hat_J = _compute_c_hat_J(π, A_artificial, c_artificial, J)
        # 3. pick an artificial variable in the basis to leave
        r = _pick_artificial_variable_in_basis_to_leave(I_to_clean, artificial_variables_indices)
        # 4. find a non-basic variable to enter
        k = _pick_non_basic_variable_to_enter(A_artificial, b_artificial, c_artificial, I_to_clean, J, c_hat_J, r)
        # 5. if no non-basic variable was found, we should suppress the constraint
        # that contains the artificial basic variable that could not be replaced
        if k is None:
            # 5.1. remove the constraint that contains the artificial basic variable
            A_artificial = np.delete(A_artificial, r, axis=0) # from artificial
            sanitized_A = np.delete(sanitized_A, r, axis=0) # as well from original 
            b_artificial = np.delete(b_artificial, r) # from artificial
            sanitized_b = np.delete(sanitized_b, r) # as well from original
            # 5.2 add to debug info that some constraint was removed
            debug_info['constraints_removed'].append(deepcopy(I_to_clean[r]))
            # 5.3. update I_to_clean
            I_to_clean.pop(r)
        else: # otherwise, we should pivot k and r variables
            # 6. update I and J
            I_to_clean, J = _update_I_and_J(I_to_clean, J, k, r)
            print(f'''
            After updates:
            I = {I_to_clean}
            J = {J}
            ''')
            # 7. remove artificial variables from J
            J = list(set(J) - set(artificial_variables_indices))
        # 8.1 update the artificial variables indices
        artificial_variables_indices = np.intersect1d(artificial_variables_indices, I_to_clean)
        # 8.2 update the artificial variables in the basis
        artificial_variables_in_basis = len(artificial_variables_indices) > 0
        
    # return the updated basis
    I_sanitized = I_to_clean.copy()
    A_I_sanitized = _compute_A_I(A, I_sanitized)
    return I_sanitized, A_I_sanitized, sanitized_A, sanitized_b, feasible, iterations_count, debug_info

def _simplex_with_feasible_initial_basis(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, debug = False) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, dict]:    
    """
    Solves the linear programming minimization problem where:
        A is the matrix of coefficients of the constraints,
        b is the vector of the right-hand side of the constraints,
        c is the vector of coefficients of the objective function,
        I is the set of indices of the basic variables, considering that the
        initial basis is feasible,
        debug is a boolean variable to print the steps of the simplex method.
    It returns:
        z_star: the optimal value of the objective function,
        x_star: the optimal solution,
        I_star: the set of indices of the basic variables in the optimal solution,
        A_I: the matrix of coefficients of the basic variables in the optimal solution,
        A: the matrix of coefficients of the constraints after the method,
        iterations_count: the number of iterations needed to reach the optimal solution,
        solution_type: the type of the solution
            1 - optimal finite solution found,
            2 - multiple optimal solutions found,
            3 - unbounded solution,
            -1 - infeasible solution,
        debug_info: a dictionary with debug information.
    """
    # Create copies of the input matrices so we don't modify the original ones
    A = np.copy(A)
    b = np.copy(b)
    c = np.copy(c)
    I = deepcopy(I)
    # Initialize auxiliary variables
    k = None
    r = None
    y_k = None
    old_I = None
    old_J = None
    debug_info_from_r_who_leaves = None
    solution_found = False
    debug_info = {}
    debug_info['title'] = 'Simplex method with feasible initial basis for granted'
    _, n = A.shape
    iterations_count = 0
    # Perform the simplex method
    J = _compute_J(n, I)
    while not solution_found:
        # 1. Compute A_I, A_I_inv, J, A_J, x_I, π, and z_0
        A_I, x_I, π, z_0 = _compute_A_I_x_I_π_and_z_0(A, b, c, I)
        A_I_inv = np.linalg.inv(A_I)
        A_J = _compute_A_J(A, J)
        old_I, old_J = deepcopy(I), deepcopy(J)
        # 2. Compute c_hat_J
        c_hat_J = _compute_c_hat_J(π, A, c, J)
        # 3. Find the variable to enter the basis
        k, c_hat_k_gt_0 = _find_k_who_enters(c_hat_J)
        # If c_hat_k <= 0, a solution was found
        if not c_hat_k_gt_0:
            solution_found = True
            # solution type must be 1 if all c_hat_J < 0, otherwise 2
            solution_type = 1 if np.all(c_hat_J < 0) else 2
            continue # advance to the end of the loop and exit in sequence
        # 4. Find the variable to leave the basis
        r, y_k, y_k_gt_0, debug_info_from_r_who_leaves = _find_r_who_leaves(I, A_I_inv, x_I, A_J[:, k], debug)
        # If y_k_r <= 0, the solution is unbounded
        if not y_k_gt_0:
            solution_found = True
            # our problem is unbounded, so our solution type is 3
            solution_type = 3
            continue # advance to the end of the loop and exit in sequence
        # 5. Update I and J
        I, J = _update_I_and_J(I, J, k, r)
        # 6. Append debug information for all values computed in this iteration
        h = f'debug_info_iteration_{iterations_count:02d}'
        debug_info[h] = {
            'A_I': A_I,
            'A_I_inv': A_I_inv,
            'J': J,
            'A_J': A_J,
            'x_I': x_I,
            'x': _compute_x_from_x_I(n, I, x_I),
            'π': π,
            'z_0': z_0,
            'c_hat_J': c_hat_J,
            'k': k,
            'r': r,
            'debug_info_from_r_who_leaves': debug_info_from_r_who_leaves,
            'y_k': y_k,
            'previous I': old_I,
            'previous J': old_J,
            'computed I': I,
            'computed J': J,
        }
        # 7. Increment the iterations count
        iterations_count += 1

    # 8. Append debug information for all values computed in the last iteration
    h = f'debug_info_iteration_{iterations_count:02d}'
    debug_info[h] = {
        'A_I': A_I,
        'A_I_inv': A_I_inv,
        'J': J,
        'A_J': A_J,
        'x_I': x_I,
        'x': _compute_x_from_x_I(n, I, x_I),
        'π': π,
        'z_0': z_0,
        'c_hat_J': c_hat_J,
        'k': k,
        'r': r,
        'debug_info_from_r_who_leaves': debug_info_from_r_who_leaves,
        'y_k': y_k,
        'previous I': old_I,
        'previous J': old_J,
        'computed I': I,
        'computed J': J,
    }
    # Assuming that this method takes a feasible initial I basis for granted,
    # it can only returns solution of types 1, 2, or 3. Now we should return
    # the expected values
    z_star = z_0
    x_star = x_I
    I_star = I
    A_I_star = A_I
    return z_star, x_star, I_star, A_I_star, A, iterations_count, solution_type, debug_info

def simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: np.ndarray, debug = False) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, dict]:    
    """
    Solves the linear programming minimization problem through the 2-phases simplex where:
        A is the matrix of coefficients of the constraints,
        b is the vector of the right-hand side of the constraints,
        c is the vector of coefficients of the objective function,
        I is the set of indices of the basic variables, feasible or not,
        debug is a boolean variable to print the steps of the simplex method.
    It returns:
        z_star: the optimal value of the objective function,
        x_star: the optimal solution,
        I_star: the set of indices of the basic variables in the optimal solution,
        A_I: the matrix of coefficients of the basic variables in the optimal solution,
        A: the matrix of coefficients of the constraints after the method,
        iterations_count: the number of iterations needed to reach the optimal solution,
        solution_type: the type of the solution
            1 - optimal finite solution found,
            2 - multiple optimal solutions found,
            3 - unbounded solution,
            -1 - infeasible solution,
        debug_info: a dictionary with debug information.
    """
    # Create copies of the input matrices so we don't modify the original ones
    A = np.copy(A)
    b = np.copy(b)
    c = np.copy(c)
    I = np.copy(I)
    # Find a feasible initial basis
    I, A_I, A_artificial, b, feasible, iterations_count, debug_info = _simplex_find_feasible_initial_basis(A, b, c, I, debug)
    # If the original problem is infeasible, return the expected values
    if not feasible:
        return 0, np.zeros(A.shape[1]), I, A_I, A, iterations_count, -1, debug_info
    # Otherwise, solve the problem with the feasible initial basis granted:
    # the second phase of the 2-phases simplex method
    z_star, x_star, I_star, A_I_star, A, iterations_count_second_phase, solution_type, debug_info_second_phase = _simplex_with_feasible_initial_basis(A, b, c, I, debug)
    # Append the debug information from the second phase to the debug information from the first phase
    debug_info['second phase'] = debug_info_second_phase
    return z_star, x_star, I_star, A_I_star, A, iterations_count + iterations_count_second_phase, solution_type, debug_info