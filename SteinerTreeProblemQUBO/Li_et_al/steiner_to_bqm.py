import dimod
from SteinerTree import SteinerTree
from typing import List, Tuple

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

def add_H_6(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float
) -> float:
    total_offset = 0.0
    n = len(problem.nodes)
    S = n

    for terminal in problem.terminals:
        coincidence_vars = []

        for other_terminal in problem.terminals:
            if terminal == other_terminal:
                continue

            for i in problem.nodes:
                for s in range(S+1):
                    var_name_1 = (i, s, terminal)
                    var_name_2 = (i, s, other_terminal)
                    z_var = ("h6_z", i, s, terminal, other_terminal)
                    coincidence_vars.append(z_var)

                    # Enforce z_var = var_name_1 * var_name_2.
                    linear[z_var] = linear.get(z_var, 0.0) + 3 * constraint_weight

                    key = (var_name_1, var_name_2)
                    quadratic[key] = quadratic.get(key, 0.0) + constraint_weight

                    key = (var_name_1, z_var)
                    quadratic[key] = quadratic.get(key, 0.0) - 2 * constraint_weight

                    key = (var_name_2, z_var)
                    quadratic[key] = quadratic.get(key, 0.0) - 2 * constraint_weight

        if not coincidence_vars:
            continue

        max_slack_value = len(coincidence_vars) - 1
        slack_terms = []
        bit_index = 0

        while (2**bit_index) <= max_slack_value:
            bit_weight = 2**bit_index
            slack_var = ("h6_y", terminal, bit_index)
            slack_terms.append((slack_var, bit_weight))
            bit_index += 1

        total_offset += _add_squared_sum_equals_one_with_slack(
            coincidence_vars,
            slack_terms,
            linear,
            quadratic,
            constraint_weight,
        )

    return total_offset


def _add_squared_sum_equals_one_with_slack(
        coincidence_vars: list,
        slack_terms: List[Tuple[object, int]],
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    total_offset = constraint_weight

    for coincidence_var in coincidence_vars:
        linear[coincidence_var] = linear.get(coincidence_var, 0.0) - constraint_weight

    for slack_var, bit_weight in slack_terms:
        linear[slack_var] = linear.get(slack_var, 0.0) + constraint_weight * (
            bit_weight ** 2 + 2 * bit_weight
        )

    for index, coincidence_var_1 in enumerate(coincidence_vars):
        for coincidence_var_2 in coincidence_vars[index + 1:]:
            key = (coincidence_var_1, coincidence_var_2)
            quadratic[key] = quadratic.get(key, 0.0) + 2 * constraint_weight

    for index, (slack_var_1, bit_weight_1) in enumerate(slack_terms):
        for slack_var_2, bit_weight_2 in slack_terms[index + 1:]:
            key = (slack_var_1, slack_var_2)
            quadratic[key] = quadratic.get(key, 0.0) + (
                2 * constraint_weight * bit_weight_1 * bit_weight_2
            )

    for coincidence_var in coincidence_vars:
        for slack_var, bit_weight in slack_terms:
            key = (coincidence_var, slack_var)
            quadratic[key] = quadratic.get(key, 0.0) - 2 * constraint_weight * bit_weight

    return total_offset
