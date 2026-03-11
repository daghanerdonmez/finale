import dimod
from SteinerTree import SteinerTree

# Convert a Steiner Tree instance to dimod.BinaryQuadraticModel instance
# Following the QUBO formulation of: https://arxiv.org/pdf/2603.04089

def steiner_to_bqm_Li_et_al(
    problem: SteinerTree,
    constraint_weight: float,
) -> dimod.BinaryQuadraticModel:
    linear = {}
    quadratic = {}
    vartype = dimod.BINARY
    offset = 0.0

    add_H_A(problem, quadratic)
    offset += add_H_1(problem, linear, constraint_weight)
    offset += add_H_2(problem, linear, constraint_weight)

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


def add_H_A(
    problem: SteinerTree,
    quadratic: dict,
) -> None:

    n = len(problem.nodes)

    for i in problem.nodes:
        for j in problem.nodes:

            weight = problem.check_edge(i, j)

            if weight == 0:
                continue

            for k in problem.terminals:
                for s in range(n):
                    key_1 = (i, s, k)
                    key_2 = (j, s+1, k)
                    key = (key_1, key_2)
                    quadratic[key] = quadratic.get(key, 0) + weight


def add_H_1(
        problem: SteinerTree,
        linear: dict, 
        constraint_weight: float
) -> float:
    if not problem.terminals:
        raise ValueError("At least one terminal is required.")
    
    root_node = problem.terminals[0]

    first_target = problem.terminals[0]
    root_var = (root_node, 0, first_target)

    # For binary x, (1 - x)^2 = 1 - x.
    linear[root_var] = linear.get(root_var, 0.0) - constraint_weight
    return constraint_weight

def add_H_2(
        problem: SteinerTree,
        linear: dict,
        constraint_weight: float
) -> float:
    if not problem.terminals:
        raise ValueError("At least one terminal is required.")
    
    n = len(problem.nodes)
    S = n
    
    total_offset_to_be_returned = 0.0

    for terminal in problem.terminals:
        var_name = (terminal, S, terminal)
        linear[var_name] = linear.get(var_name, 0.0) - constraint_weight
        total_offset_to_be_returned += constraint_weight

    return total_offset_to_be_returned
