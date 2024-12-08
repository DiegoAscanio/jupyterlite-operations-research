from types import SimpleNamespace
import numpy as np

def _compute_A_I(A: np.ndarray, I: np.ndarray) -> np.ndarray:
    A_I = A[:, I]
    return A_I

def _compute_J(n: int, I: np.ndarray) -> np.ndarray:
    J = np.setdiff1d(np.arange(n), I)
    return J

def _compute_A_J(A: np.ndarray, J: np.ndarray) -> np.ndarray:
    A_J = A[:, J]
    return A_J

def _compute_X_I(A_I: np.ndarray, b: np.ndarray) -> np.ndarray:
    X_I = np.linalg.inv(A_I).dot(b)
    return X_I

def _compute_A_I_x_I_π_and_z_0(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # A_I should be a submatrix from A with columns indexed (and sorted) by I
    A_I = _compute_A_I(A, I)
    A_I_inv = np.linalg.inv(A_I)
    # C_I should be a submatrix from c with columns indexed (and sorted) by I
    c_I = c[I]

    x_I = _compute_X_I(A_I, b)
    π = c_I.dot(A_I_inv)
    z_0 = π.dot(b)
    return A_I, x_I, π, z_0

def _compute_c_hat_J(π: np.ndarray, A: np.ndarray, c: np.ndarray, J: np.ndarray) -> np.ndarray:
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

def _find_r_who_leaves(A_I_inv: np.ndarray, x_I: np.ndarray, A_k: np.ndarray, debug = False) -> tuple[np.intp, np.ndarray, bool, dict]:
    """
    Finds the index of the variable that leaves the basis and returns if
    the simplex method shouuld continue. When all elements of A_k are
    non-positive, the method found an unbounded solution and, therefore,
    should stop.
    Args:
        A_I_inv: the inverse of the matrix of the basic variables.
        x_I: the vector of the basic variables.
        A_k: the vector of the coefficients of the entering variable.
    Returns:
        r: the index of the variable that leaves the basis.
        y_k: the vector of the coefficients of the entering variable.
        proceed_: a boolean indicating if the simplex method should proceed.
                  if y_k[r] <= 0, the solution is unbounded and, therefre,
                  the method should stop.
        debug_info: a dictionary with debug information.
    """
    debug_info = {}
    y_k = A_I_inv.dot(A_k)
    ratios = x_I / y_k if y_k > 0 else np.inf
    r = np.argmin(ratios)
    if debug:
        debug_info['y_k'] = y_k
        debug_info['ratios'] = ratios
        debug_info['r'] = r
    return r, y_k, y_k[r] > 0, debug_info

def _update_I_and_J(I: np.ndarray, J: np.ndarray, k: np.intp, r: np.intp) -> tuple[np.ndarray, np.ndarray]:
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
    I[r] = k
    J[k] = r
    return I, J

def _find_feasible_initial_basis(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: np.ndarray, debug = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, int, dict]:
    """
    Finds a feasible initial basis for the 2-phase simplex method.
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
        feasible: a boolean indicating if the original problem is feasible.
        iterations_count: the number of iterations needed to find a feasible basis.
        debug_info: a dictionary with debug information.
    """
    # Initialize the variables
    debug_info = {}
    m, n = A.shape
    A_I = _compute_A_I(A, I)
    x_I = _compute_X_I(A_I, b)
    if x_I >= 0: # The initial basis is already feasible
        return I, A_I, A, True, 0, debug_info
    """
    Perform first phase - steps:
        1. Create artificial variables
        2. Add them to the coefficients matrix A
        3. Create the artificial objective function φ of the form min 1.X
        4. Solve the auxiliary problem through the simplex method
        The original problem is feasible if φ* = 0, otherwise it is infeasible.
    """
    # Create the artificial variables
    x_I_indices_lesser_than_zero = np.where(x_I < 0)[0]
    artificial_variables = np.array([
        [1 if i == j else 0 for i in range(m)] for j in x_I_indices_lesser_than_zero
        ])
    artificial_variables_indices = np.arange(n, n + len(x_I_indices_lesser_than_zero))
    # Add the artificial variables to the coefficients matrix A
    A_artificial = np.hstack((A, artificial_variables))
    # Create the artificial objective function
    c_artificial = np.zeros(A_artificial.shape[1])
    c_artificial[n:] = 1
    # Solve the auxiliary problem
    # 1. Replace the indices of the artificial variables in the basic variables set
    I_artificial = np.copy(I)
    I_artificial[x_I_indices_lesser_than_zero] = np.arange(n, n + len(x_I_indices_lesser_than_zero))
    # 2. call the simplex method
    φ_star, x_star, I_star, A_I_star, A, iterations_count, solution_type, first_phase_debug_info = simplex(
            A_artificial, b, c_artificial, I_artificial, debug
            )
    feasible = φ_star == 0
    # 3. Update the debug information
    debug_info['first phase'] = first_phase_debug_info
    # 4. Analyze the resulting basis
    # If the resulting basis contains some of the artificil variables and the original problem is feasible,
    # we'll try to replace the artificial basic variables by the original non-basic variables
    # 4.1 Check if the original problem is feasible
    if not feasible:
        return I, A_I, A, feasible, iterations_count, debug_info
    # 4.2 Check if there are artificial variables are in the basis
    I_artificial_variables = np.intersect1d(I_star, artificial_variables_indices)
    # If there aren't any artificial variables in the basis, return I_star, A_I_star, A, feasible, debug_info
    artificial_variables_in_basis = len(I_artificial_variables) > 0
    if not artificial_variables_in_basis:
        return I_star, A_I_star, A, feasible, iterations_count, debug_info
    # Otherwise chose an original non-basic variable to replace the artificial basic variable
    # until there are no artificial variables in the basis or if not possible, to suppress
    # the constraints that contain the artificial basic variables which could not be replaced
    while artificial_variables_in_basis:
        # Compute A_I, A_I_inv, x_I, π, and z_0
        A_I, x_I, π, _ = _compute_A_I_x_I_π_and_z_0(A_artificial, b, c_artificial, I_artificial_variables)
        A_I_inv = np.linalg.inv(A_I)
        # Compute J for the original non-basic variables
        J = _compute_J(n, I)
        # Compute c_hat_J
        c_hat_J = _compute_c_hat_J(π, A_artificial, c_artificial, J)
        # Find the variable to enter the basis
        k, _ = _find_k_who_enters(c_hat_J)
        # Find the variable to leave the basis
        A_k = A_artificial[:, k]
        r, can_leave, debug_info[f'trying to make x_{r} to leave'] = _find_r_who_leaves(A_I_inv, x_I, A_k, debug)
        if can_leave:
            # if some variable can leave the basis, update I and J
            I_star, J = _update_I_and_J(I_artificial_variables, J, k, r)
        else:
            # If no variable can leave the basis, suppress the constraint that
            # contains some artificial basic variable.
            # If no variable can leave, we are considering the ratio x_I / y_k to be infinite
            # so we can suppress the constraint that contains one of the artificial basic variables
            # represented by the first index of I_artificial_variables, as this constraint is
            # redundant and can be removed without affecting the feasible region
            to_supress = I_artificial_variables[0]
            # find the row of A where to_supress is non-zero
            constraint_to_supress = np.where(A[:, to_supress] != 0)[0]
            # suppress the constraint
            A = np.delete(A, constraint_to_supress, axis = 0)
            b = np.delete(b, constraint_to_supress)
            # delete the artificial variable from the basis I_artificial_variables and from the I_star
            I_star = np.delete(I_star, np.where(I_star == to_supress))

        # Update the artificial variables in the basis
        I_artificial_variables = np.intersect1d(I_star, artificial_variables_indices)
        artificial_variables_in_basis = len(I_artificial_variables) > 0
    # after all artificial variables are removed from the basis, we need to recompute A_I
    # from the modifications made during artificial variables removal
    A_I = _compute_A_I(A, I_star)
    # and return the updated basis
    return I_star, A_I, A, feasible, iterations_count, debug_info

_do_nothing = lambda *args, **kwargs: None

simplex_steps = {
        'initial_basis': _find_feasible_initial_basis,
        'compute_A_I_x_I_π_and_z_0': {
            True: _compute_A_I_x_I_π_and_z_0,
            False: _do_nothing
            },
        'compute_A_J': _compute_A_J,
        'compute_c_hat_J': _compute_c_hat_J,
        'find_k_who_enters': _find_k_who_enters,
        'find_r_who_leaves': {
            True: _find_r_who_leaves,
            False: _do_nothing
            },
        'update_I_and_J': {
            True: _update_I_and_J,
            False: _do_nothing
            }
        }

second_phase_steps = [ 'compute_A_I_x_I_π_and_z_0', 'compute_A_J', 'compute_c_hat_J', 'find_k_who_enters', 'find_r_who_leaves', 'update_I_and_J' ]

first_phase = simplex_steps['initial_basis']

def simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: np.ndarray, debug = False) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, dict]:    
    """
    Solves the linear programming minimization problem using the 2-phase simplex 
    method where:
        A is the matrix of coefficients of the constraints,
        b is the vector of the right-hand side of the constraints,
        c is the vector of coefficients of the objective function,
        I is the set of indices of the basic variables,
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
    # Initialize the variables
    solution_found = False
    debug_info = {}
    _, n = A.shape
    J = _compute_J(n, I)
    iterations_count = 0
    # First phase: find a feasible initial basis
    I, A_I, A, feasible, iterations_count, first_phase_debug_info = first_phase(A, b, c, I, debug)
    debug_info['first phase'] = first_phase_debug_info
    if not feasible:
        return 0, np.zeros(A.shape[1]), I, A_I, A, iterations_count, -1, debug_info
    # Second phase: find the optimal solution
    second_phase_debug_info = {}
    while not solution_found:
        second_phase_debug_info[f'iteration_{iterations_count}'] = {}
        # instantiate an iterator for second phase at each iteration
        second_phase_steps_iter = iter(second_phase_steps)
        # 1. Compute A_I, x_I, π, z_0, A_J, c_hat_J
        step = next(second_phase_steps_iter)
        A_I, x_I, π, z_0 = simplex_steps[step][feasible](A, b, c, I)
        # 2. Compute A_J
        step = next(second_phase_steps_iter)
        A_J = simplex_steps[step](A, J)
        # 3. Compute c_hat_J
        step = next(second_phase_steps_iter)
        c_hat_J = simplex_steps[step](π, A, c, J)
        # 2. Find the variable that enters the basis
        step = next(second_phase_steps_iter)
        k, c_hat_k_gt_0 = simplex_steps[step](c_hat_J)
        # Then, I should run the next steps if and only if no solution was found
        solution_found = not c_hat_k_gt_0
        # 3. Find the variable that leaves the basis
        step = next(second_phase_steps_iter)
        r, y_k, y_k_r_gt_0, second_phase_debug_info[f'iteration_{iterations_count}'][f'trying to make x_{r} to leave'] = simplex_steps[step][not solution_found](A_I, x_I, A_J[:, k], debug)
        solution_found = not y_k_r_gt_0
        # 4. Update I and J
        step = next(second_phase_steps_iter)
        I, J = simplex_steps[step][not solution_found](I, J, k, r)
        # 5. Update the debug information
        second_phase_debug_info[f'iteration_{iterations_count:02d}'] += {
                'A_I': A_I,
                'x_I': x_I,
                'π': π,
                'z_0': z_0,
                'A_J': A_J,
                'c_hat_J': c_hat_J,
                'k': k,
                'r': r,
                'y_k': y_k
        }
        # 6. Update the number of iterations
        iterations_count += 1
    # Evaluate the solution
    if np.any(np.isclose(c_hat_J, 0)):
        solution_type = 2 # multiple optimal solutions
    elif y_k < 0:
        solution_type = 3 # unbounded solution
    else:
        solution_type = 1 # optimal finite solution
    z_star = z_0
    x_star = np.zeros(n)
    x_star[I] = x_I
    return z_star, x_star, I, A_I, A, iterations_count, solution_type, debug_info
