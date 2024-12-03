import numpy as np

def _compute_A_I_x_I_π_and_z_0(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    # A_I should be a submatrix from A with columns indexed (and sorted) by I
    A_I = A[:, I]
    A_I_inv = np.linalg.inv(A_I)
    # C_I should be a submatrix from c with columns indexed (and sorted) by I
    c_I = c[I]

    x_I = A_I_inv.dot(b)
    π = c_I.dot(A_I_inv)
    z_0 = π.dot(b)
    return A_I, x_I, π, z_0

def _compute_c_hat_J(π: np.ndarray, A: np.ndarray, c: np.ndarray, J: list) -> np.ndarray:
    A_J = A[:, J]
    c_hat_J = π.dot(A_J) - c[J]
    return c_hat_J

def simplex(A: np.ndarray, b: np.ndarray, c: np.ndarray, I: list, debug = False) -> tuple[float, np.ndarray, np.ndarray, int, int]:
    """
    Solves the linear programming minimization problem using the simplex method
    where:
        A is the matrix of coefficients of the constraints,
        b is the vector of the right-hand side of the constraints,
        c is the vector of coefficients of the objective function,
        I is the set of indices of the basic variables,
        debug is a boolean variable to print the steps of the simplex method.
    It returns:
        z_star: the optimal value of the objective function,
        x_star: the optimal solution,
        I_star: the set of indices of the basic variables in the optimal solution,
        iterations_count: the number of iterations needed to reach the optimal solution,
        solution_type: the type of the solution
            1 - optimal finite solution found,
            2 - multiple optimal solutions found,
            3 - unbounded solution,
            -1 - infeasible solution.
    """
    # Initialize the variables
    m, n = A.shape
    iterations_count = 0
    J = list(set(range(n)) - set(I))
    solution_found = False
    solution_type = 0
    k = 0
    r = 0
    x_trivial = np.zeros(n)
    
    # start the method
    while not solution_found:
        old_I = np.copy(I)
        old_J = np.copy(J)
        # step 1 - compute x_I, π, and z_0
        A_I, x_I, π, z_0 = _compute_A_I_x_I_π_and_z_0(A, b, c, I)
        # step 2 - compute c_hat_J
        c_hat_J = _compute_c_hat_J(π, A, c, J)
        # step 3 - find which variable to enter the basis
        k = np.argmax(c_hat_J)
        # step 4 - check if c_hat_J[k] leq 0 so the solution is optimal
        if c_hat_J[k] <= 0:
            solution_found = True
            solution_type = 1
        else: # step 5 - find which variable to leave the basis
            y_k = np.linalg.inv(A_I).dot(A[:, k])
            # step 6 - if all y_k leq 0 then the solution is unbounded
            if np.all(y_k <= 0):
                solution_found = True
                solution_type = 3
            else: # step 7 - find the variable to leave the basis through the minimum ratio test for all y_k > 0
                ratios = np.where(y_k > 0, x_I / y_k, np.inf)
                r = np.argmin(ratios)
                # step 8 - update the indices of the basic variables and non-basic variables
                to_enter = J[k]
                to_exit = I[r]
                I[r] = to_enter
                J[k] = to_exit
        # step 9 - find if system is still feasible, that is, if Ax = b
        # or if all restraints are still respected
        current_x = np.copy(x_trivial)
        current_x[old_I] = x_I
        if not np.allclose(A.dot(current_x), b):
            solution_found = True
            solution_type = -1
        
        # print the steps of the simplex method
        if debug:
            print(f"After Iteration {iterations_count + 1}:")
            print(f"Basic variables indices before: {old_I}")
            print(f"Non-basic variables indices before: {old_J}")
            print(f"Basic variables indices after: {I}")
            print(f"Non-basic variables indices after: {J}")
            print(f"x_I = {x_I}")
            print(f"A_I =\n{A_I}")
            print(f"A_I^-1 =\n{np.linalg.inv(A_I)}")
            print(f"A_J =\n{A[:, J]}")
            print(f"π = {π}")
            print(f"z_0 = {z_0}")
            print(f"c_hat_J = {c_hat_J}")
            print(f"Variable to enter the basis: x_{I[r]}")
            print(f"Variable to leave the basis: x_{J[k]}")
            print()
        # increment the iterations count
        iterations_count += 1
    # If a solution is found and it is optimal, check for multiple optimal solutions
    if solution_type == 1:
        if np.isclose(c_hat_J, 0).any(): # if some c_hat_J[j] = 0 then there are multiple optimal solutions
            solution_type = 2
    # Compute the optimal value of the objective function
    x_star = np.copy(x_trivial)
    x_star[I] = x_I
    z_star = c.dot(x_star)
    return z_star, x_star, I, iterations_count, solution_type