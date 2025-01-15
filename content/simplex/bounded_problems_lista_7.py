import numpy as np
from scipy.optimize import linprog
from simplex_bounded_variables import _simplex_main_loop, markdown_repr_T, _markdown_final_step, _markdown_pivot_operations

problems = {
    '5.12': {
        'A': np.array([
            [ 2, -1,  1, -2, 1, 0,  0],
            [-1,  2, -1,  1, 0, 1,  0],
            [ 2,  1, -1,  0, 0, 0, -1]
        ]),
        'b': np.array([
            6,
            8,
            2
        ]),
        'c': np.array([
            1, 2, 3, -1, 0, 0, 0
        ]),
        'lower_bounds': np.array([0, 1, 0, 2, 0, 0, 0]),
        'upper_bounds': np.array([3, 4, 8, 5, np.inf, np.inf, np.inf]),
        'I': [4, 5, 1],
        'J_1': [0, 6, 2, 3],
        'J_2': []
    },
    '5.13': {
        'A': np.array([
            [ 3, 1, 1, 1, 0, 0],
            [-1, 1, 0, 0, 1, 0],
            [ 0, 1, 2, 0, 0, 1]
        ]),
        'b': np.array([
            12,
            4,
            8
        ]),
        'c': np.array([
            -2, -1, -3, 0, 0, 0
        ]),
        'lower_bounds': np.array([0, 0, 0, 0, 0, 0]),
        'upper_bounds': np.array([3, 5, 4, np.inf, np.inf, np.inf]),
        'I': [3, 4, 5],
        'J_1': [0, 1, 2],
        'J_2': []
    },
    '5.14': {
        'A': np.array([
            [1, 3,  1, 1,  0],
            [2, 1, -1, 0, -1]
        ]),
        'b': np.array([
            8,
            3
        ]),
        'c': np.array([
            -2, -3, 2, 0, 0
        ]),
        'lower_bounds': np.array([-np.inf, -2, 2, 0, 0]),
        'upper_bounds': np.array([4, 3, np.inf, np.inf, np.inf]),
        'I': [3, 0],
        'J_1': [1, 2, 4],
        'J_2': []
    }
}

# add bounds for scipy
indices = ['5.12', '5.13', '5.14']
for index in indices:
    problems[index]['scipy_bounds'] = [
        (lb, ub) for lb, ub in zip(problems[index]['lower_bounds'], problems[index]['upper_bounds'])
    ]

# ok now we solve the problems
for index in indices:
    for key in problems[index].keys():
        locals()[key] = problems[index][key]
    print(f'Problem {index}')
    print('Simplex method')
    return_values = _simplex_main_loop(A, b, c, I, J_1, J_2, lower_bounds, upper_bounds)
    print('Scipy method')
    res = linprog(c, A_eq=A, b_eq=b, bounds=scipy_bounds, method='highs')
    print(res)
    print()
    print()

import pyperclip
steps = return_values[-1]
iterations = return_values[-2]
m = markdown_repr_T(steps)
pyperclip.copy(m)
