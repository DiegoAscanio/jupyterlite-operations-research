from types import SimpleNamespace
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

def _find_r_who_leaves(I: list, A_I_inv: np.ndarray, x_I: np.ndarray, A_k: np.ndarray, debug = False, artificial_indices = np.array([])) -> tuple[np.intp, np.ndarray, bool, dict]:
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
        artificial_indices: the indices of the artificial variables so we can
        prioritize them to leave the basis when appropriate.
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

    # 2. Try to remove (firstly) the artificial variables from the basis
    artificials_in_base = np.isin(I, artificial_indices)
    ratios = np.where((y_k > 0) & artificials_in_base, x_I / y_k, np.inf)

    # 3. If all ratios were infinite, no artificial variable can leave the basis, so we try to remove any variable
    if np.all(np.isinf(ratios)):
        ratios = np.where(y_k > 0, x_I / y_k, np.inf)

    # 4. Variable index to leave the basis
    r = np.argmin(ratios)

    # 5. Append debug information
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
    I = deepcopy(I)
    J = deepcopy(J)
    to_enter = deepcopy(J[k])
    to_exit = deepcopy(I[r])
    I[r] = to_enter
    J[k] = to_exit
    return I, J

def _simplex_find_feasible_initial_basis(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, debug = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, int, dict]:
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
        feasible: a boolean indicating if the original problem is feasible.
        iterations_count: the number of iterations needed to find a feasible basis.
        debug_info: a dictionary with debug information.
    """
    # Create copies of the input matrices so we don't modify the original ones
    A = np.copy(A)
    b = np.copy(b)
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
    A_artificial = np.hstack((A, artificial_variables))
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
    # Compute J for the original non-basic variables
    J = _compute_J(n, I)
    while artificial_variables_in_basis:
        # Compute A_I, A_I_inv, x_I, π, and z_0
        A_I, x_I, π, _ = _compute_A_I_x_I_π_and_z_0(A_artificial, b, c_artificial, I_star)
        A_I_inv = np.linalg.inv(A_I)
        # Compute c_hat_J
        c_hat_J = _compute_c_hat_J(π, A_artificial, c_artificial, J)
        # Find the variable to enter the basis
        k, _ = _find_k_who_enters(c_hat_J)
        # Find the variable to leave the basis
        A_k = A_artificial[:, k]
        r, _, can_leave, debug_info_from_r_who_leaves = _find_r_who_leaves(I_star, A_I_inv, x_I, A_k, debug, I_restricted_to_artificial_variables_in_I)
        if can_leave:
            # if some variable can leave the basis, update I and J
            print(I_star, J, J[k], I_star[r], sep=' | ')
            I_star, J = _update_I_and_J(I_star, J, k, r)
            print(I_star, J)
        else:
            # If no variable can leave, we are considering the ratio x_I / y_k to be infinite
            # so we can suppress the constraint that contains one of the artificial basic variables
            # represented by the first index of I_restricted_to_artificial_variables_in_I, as this constraint is
            # redundant and can be removed without affecting the feasible region. This can be done if and only
            # if the rank of A remains the same before and after the removal of the constraint.
            valid_suppression = False
            artificial_variables_iterator = iter(I_restricted_to_artificial_variables_in_I)
            to_supress = next(artificial_variables_iterator, None)
            while not valid_suppression and to_supress is not None:
                # find the row of A where to_supress is non-zero
                constraint_to_supress = np.where(A[:, to_supress] != 0)[0]
                # suppress the constraint
                A_reduced = np.delete(A, constraint_to_supress, axis = 0)
                # check if the rank of A remains the same
                valid_suppression = np.linalg.matrix_rank(A_reduced) == np.linalg.matrix_rank(A)
                to_supress = next(artificial_variables_iterator)

            A = A_reduced
            A_artificial = np.delete(A_artificial, constraint_to_supress, axis = 0)
            b = np.delete(b, constraint_to_supress)
            # delete the artificial variable from the basis I_restricted_to_artificial_variables_in_I and from the I_with_artificial_variables
            I_star = np.delete(I_star, np.where(I_star == to_supress))
        counter += 1
        # Update the artificial variables in the basis
        I_restricted_to_artificial_variables_in_I = np.intersect1d(I_star, artificial_variables_indices)
        artificial_variables_in_basis = len(I_restricted_to_artificial_variables_in_I) > 0
    # after all artificial variables are removed from the basis, we need to recompute A_I
    # from the modifications made during artificial variables removal
    A_I = _compute_A_I(A, I_star)
    # and return the updated basis
    return I_star, A_I, A, b, feasible, iterations_count, debug_info

def _simplex_with_feasible_initial_basis(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, debug = False, artificial_indices=[]) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, dict]:    
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
        r, y_k, y_k_gt_0, debug_info_from_r_who_leaves = _find_r_who_leaves(I, A_I_inv, x_I, A_J[:, k], debug, artificial_indices)
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
