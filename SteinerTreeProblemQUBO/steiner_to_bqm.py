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
    add_H_3(problem, quadratic, constraint_weight)
    add_H_4(problem, linear, quadratic, constraint_weight)
    add_H_5(problem, quadratic, constraint_weight)
    offset += add_H_6(problem, linear, quadratic, constraint_weight)

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
                    var_name_1 = (i, s, k)
                    var_name_2 = (j, s+1, k)
                    key = (var_name_1, var_name_2)
                    quadratic[key] = quadratic.get(key, 0) + weight


def add_H_1(
        problem: SteinerTree,
        linear: dict, 
        constraint_weight: float
) -> float:
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
    n = len(problem.nodes)
    S = n
    
    total_offset_to_be_returned = 0.0

    for terminal in problem.terminals:
        var_name = (terminal, S, terminal)
        linear[var_name] = linear.get(var_name, 0.0) - constraint_weight
        total_offset_to_be_returned += constraint_weight

    return total_offset_to_be_returned

def add_H_3(
        problem: SteinerTree,
        quadratic: dict,
        constraint_weight: float,
) -> None:
    n = len(problem.nodes)
    S = n

    for terminal in problem.terminals:
        for s in range(S+1):
            for i in problem.nodes:
                for j in problem.nodes:
                    # WE ARE DOUBLE COUNTING HERE BECAUSE EQ. 12 IN THE PAPER DID SO AS WELL
                    # BUT I DON'T KNOW IF IT WAS INTENTIONAL OR IF IT IS LOGICAL.
                    # OPEN FOR EXPERIMENTATION
                    if i == j:
                        continue
                    var_name_1 = (i, s, terminal)
                    var_name_2 = (j, s, terminal)
                    key = (var_name_1, var_name_2)
                    quadratic[key] = quadratic.get(key, 0.0) + constraint_weight

def add_H_4(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> None:
    n = len(problem.nodes)
    S = n

    for terminal in problem.terminals:
        for s in range(S):
            for i in problem.nodes:
                var_name = (i, s, terminal)
                linear[var_name] = linear.get(var_name, 0.0) + constraint_weight
            
            for i in problem.nodes:
                for j in problem.nodes:
                    var_name_1 = (i, s, terminal)
                    var_name_2 = (j, s+1, terminal)
                    key = (var_name_1, var_name_2)
                    quadratic[key] = quadratic.get(key, 0.0) - constraint_weight

def add_H_5(
        problem: SteinerTree,
        quadratic: dict,
        constraint_weight: float
) -> None:
    n = len(problem.nodes)
    S = n

    for terminal in problem.terminals:
        for i in problem.nodes:
            if i not in problem.terminals:
                for s in range(S):
                    var_name_1 = (i, s, terminal)
                    var_name_2 = (i, s+1, terminal)
                    key = (var_name_1, var_name_2)
                    quadratic[key] = quadratic.get(key, 0.0) + constraint_weight


