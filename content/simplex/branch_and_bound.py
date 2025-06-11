import numpy as np
from simplex_two_phases import simplex
from copy import deepcopy
import pdb

try:
    from graphviz import Digraph
except ImportError:
    class Digraph:
        def __init__(self, comment=''):
            self.comment = comment
            self.nodes = []
            self.edges = []

        def node(self, name, label, shape='box'):
            self.nodes.append((name, label, shape))

        def edge(self, from_node, to_node, label=''):
            self.edges.append((from_node, to_node, label))

        def render(self):
            return "Graph rendering not implemented for this mock class."

def generate_dot(node):
    lines = ["digraph G {", "node [shape=box];"]

    def recurse(n):
        if n is None:
            return
        name = f"N{n['name']}"
        z_inf = n.get("z_inf", "None")
        x = n.get("x", None)
        x_str = str(np.round(x, 2)) if x is not None else "N/A"
        label = f"{name}\\nP: {n['P']['name']}\\nz_inf: {z_inf}\\nx: {x_str}"
        lines.append(f'"{name}" [label="{label}"];')

        if 'left' in n and n['left']:
            left_name = f"N{n['left']['name']}"
            var = n.get("branch_variable", "?")
            bound = n.get("lower_bound", "?")
            lines.append(f'"{name}" -> "{left_name}" [label="x_{var} <= {bound}"];')
            recurse(n['left'])

        if 'right' in n and n['right']:
            right_name = f"N{n['right']['name']}"
            var = n.get("branch_variable", "?")
            bound = n.get("upper_bound", "?")
            lines.append(f'"{name}" -> "{right_name}" [label="x_{var} >= {bound}"];')
            recurse(n['right'])

    recurse(node)
    lines.append("}")
    return "\n".join(lines)

def draw_bnb_tree(node, graph=None):
    if graph is None:
        graph = Digraph(comment='Árvore BnB')

    if node is None:
        return graph

    node_name = f"N{node['name']}"

    # Pega informações com verificação de existência
    z_inf = node.get('z_inf')
    z_inf_str = f"{z_inf:.2f}" if z_inf is not None and not np.isnan(z_inf) and not np.isinf(z_inf) else str(z_inf)

    x_vals = node.get('x')
    x_str = np.round(x_vals, 2) if x_vals is not None else "N/A"

    P_name = node['P']['name'] if 'P' in node and 'name' in node['P'] else 'N/A'

    label = f"{node_name}\\nP: {P_name}\\nz_inf: {z_inf_str}\\nx: {x_str}"

    graph.node(node_name, label, shape='box')

    # Processar filhos
    if 'left' in node and node['left']:
        var = node.get('branch_variable', '?')
        bound = node.get('lower_bound', '?')
        edge_label = f"x_{var} ≤ {bound}"
        draw_bnb_tree(node['left'], graph)
        graph.edge(node_name, f"N{node['left']['name']}", label=edge_label)

    if 'right' in node and node['right']:
        var = node.get('branch_variable', '?')
        bound = node.get('upper_bound', '?')
        edge_label = f"x_{var} ≥ {bound}"
        draw_bnb_tree(node['right'], graph)
        graph.edge(node_name, f"N{node['right']['name']}", label=edge_label)

    return graph

def gap(z_inf, z_star):
    if z_star == np.inf:
        return np.inf
    return (z_inf - z_star) / z_star

def select_current_problem(L, estrategy = 'dfs'):
    if estrategy == 'dfs':
        return L.pop()
    elif estrategy == 'bfs':
        return L.pop(0)
    elif estrategy == 'best':
        return L.pop(np.argmax([-p['z_inf'] for p in L]))
    else:
        raise ValueError("Unknown strategy: {}".format(estrategy))

def check_integrality(x : np.ndarray, integrality):
    x_slice = x[integrality == 1]
    return np.all(np.isclose(x_slice, np.round(x_slice)))

def least_close_to_integer_index(x, integrality):
    """
    Find the index of the variable that is least close to being an integer.
    Arguments:
    x -- numpy array of variable values
    integrality -- numpy array indicating which variables should be integers
    Returns:
    index -- the index of the variable that is least close to an integer
    """
    x_slice = x[integrality == 1]
    fractional_parts = np.abs(x_slice - np.round(x_slice))
    max_fractional_part = np.max(fractional_parts)
    x_copy = np.copy(x)
    x_copy = np.abs(x_copy - np.round(x_copy))
    x_copy[integrality != 1] = np.inf  # Ignore non-integer variables
    index, *_ = np.where(x_copy == max_fractional_part)
    return index[0] if len(index) > 0 else None

def breadth_first_search(branch_and_bound_tree, target):
    from collections import deque
    root = branch_and_bound_tree

    if root is None:
        return None

    queue = deque([root])

    while queue:
        current_node = queue.popleft()

        # Check if the current node's name matches the target
        if current_node['name'] == target:
            return current_node

        # Add left and right children to the queue
        if current_node['left'] is not None:
            queue.append(current_node['left'])
        if current_node['right'] is not None:
            queue.append(current_node['right'])
    return None  # Target not found in the tree

def branch_and_bound(c, A, b, integrality = None, epsilon = 1e-3, estrategy = 'dfs'):
    _, n = A.shape
    if integrality is None:
        integrality = np.ones(n)
    z_star = np.inf  # Initialize with positive infinity
    best_z_inf = -np.inf  # Initialize with negative infinity
    x_star = None
    P = {
        'name': 0,
        'c': c,
        'A': A,
        'b': b,
        'z_inf': -np.inf,  # Initialize with negative infinity
        'z_star': np.inf,  # Initialize with positive infinity
        'integrality': integrality
    }
    L = [P]
    iters = 0
    snapshots = {}
    branch_and_bound_tree = {
        'name': P['name'],
        'branch_variable': None,
        'lower_bound': -np.inf,
        'upper_bound': np.inf,
        'z_inf': -np.inf,
        'x': None,
        'P': None,
        'left': None,
        'right': None
    }
    while len(L) > 0 and gap(best_z_inf, z_star) > epsilon:
        # 1. Select the current problem based on the strategy
        current_L = deepcopy(L)  # Make a copy of L to snapshot it
        P_current = select_current_problem(L, estrategy)
        root_node = breadth_first_search(branch_and_bound_tree, P_current['name'])
        root_node['P'] = P_current

        # 2. Solve the linear relaxation
        c = P_current['c']
        A = P_current['A']
        m, n = A.shape
        b = P_current['b']
        I = np.arange( n - m, n)
        integrality = P_current['integrality']
        z_current, x_solution, I_star, A_I, A, _, solution_type, debug_info = simplex(
            A, b, c, I
        )

        snapshots[iters] = {
            'z_LR': z_current,
            'x_LR': x_solution,
            'L': current_L
        }
        iters += 1 # Increment the iteration counter

        # 1. If the current problem is infeasible, skip this branch
        if solution_type == -1:
            continue 
        x_current = np.zeros(n)
        x_current[I_star] = x_solution
        # update root node values for bnb tree
        root_node['z_inf'] = z_current
        root_node['x'] = x_current
        P_current['z_inf'] = z_current
        # 2. if z_current is gt than the best_z_inf found untill now
        # we have a thighter inferior bound, so we update it
        if z_current > best_z_inf:
            best_z_inf = z_current
        # 4. Check if the current solution is worse than the best known solution
        # for the current problem, assuming we're handling minimization
        if z_current >= z_star:
            continue
        # 5. Check if the current solution is integral and better than the best
        # known solution
        if check_integrality(x_current, integrality):
            if z_current < z_star:
                z_star = z_current
                x_star = x_current
            # prune all active nodes that have a z_inf greater than the current z_star
            active_nodes = iter(
                deepcopy(L)  # Use deepcopy to avoid modifying iterator while changing L
            )
            P_active = next(active_nodes, None)
            while P_active is not None:
                if P_active['z_inf'] >= z_star:
                    L.remove(P_active)
                P_active = next(active_nodes, None)
            continue # go back to the beginning of the loop
        # 6. If the solution is not integral, branch on the variable that is least
        # close to being an integer
        index = least_close_to_integer_index(x_current, integrality)
        # we'll assume index is always valid
        lower_partition = np.floor(x_current[index])
        upper_partition = np.ceil(x_current[index])
        # we'll divide P_current into two new problems
        # 1. Lower partition problem
        P_lower = {
            'name': 2 * P_current['name'] + 1,  # Unique name for the new problem
            'z_inf': -np.inf,  # Initialize with negative infinity
            'z_star': np.inf,  # Initialize with positive infinity
            'integrality': np.hstack((integrality, [0]))
        }
        # add one new slack variable for the new constraint
        # x[index] <= lower_partition
        P_lower['c'] = np.hstack([c, np.zeros(1)])
        # add the new constraint to A and b
        P_lower['A'] = np.hstack([A, np.zeros((m, 1))])
        P_lower['A'] = np.vstack([P_lower['A'], np.zeros(n + 1)])
        P_lower['A'][-1, index] = 1
        P_lower['A'][-1, -1] = 1  # slack variable
        P_lower['b'] = np.hstack([b, lower_partition])
        L.append(P_lower)
        # 2. Upper partition problem
        P_upper = {
            'name': 2 * P_current['name'] + 2,  # Unique name for the new problem
            'z_inf': -np.inf,  # Initialize with negative infinity
            'z_star': np.inf,  # Initialize with positive infinity
            'integrality': np.hstack((integrality, [0]))
        }
        # add one new excess variable for the new constraint
        # x[index] >= upper_partition
        P_upper['c'] = np.hstack([c, np.zeros(1)])
        # add the new constraint to A and b
        P_upper['A'] = np.hstack([A, np.zeros((m, 1))])
        P_upper['A'] = np.vstack([P_upper['A'], np.zeros(n + 1)])
        P_upper['A'][-1, index] = 1
        P_upper['A'][-1, -1] = -1  # excess variable
        P_upper['b'] = np.hstack([b, upper_partition])
        L.append(P_upper)

        # Update the branch and bound tree structure
        root_node['branch_variable'] = index
        root_node['lower_bound'] = lower_partition
        root_node['upper_bound'] = upper_partition
        root_node['left'] = {
            'name': P_lower['name'],
            'left': None,
            'right': None
        }

        root_node['right'] = {
            'name': P_upper['name'],
            'left': None,
            'right': None
        }

    return z_star, x_star, iters, snapshots, branch_and_bound_tree

A = np.array([
    [5, 2, 3, 2, 2, 1]
])

b = np.array([
    7
])

c = np.array([-10, -6, -7, -2, -1, 0])

integrality = np.array([1, 1, 1, 1, 1, 0])  # x1 and x2 are integers
