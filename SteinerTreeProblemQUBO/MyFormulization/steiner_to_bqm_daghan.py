import dimod
from SteinerTree import SteinerTree
from typing import List, Tuple

def steiner_to_bqm_Li_et_al(
    problem: SteinerTree,
    constraint_weight: float,
) -> dimod.BinaryQuadraticModel:
    linear = {}
    quadratic = {}
    vartype = dimod.BINARY
    offset = 0.0

    add_H_cost(problem, linear)

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


def add_H_cost(
        problem: SteinerTree,
        linear: dict
) -> None:
    for edge in problem.edges:
        var_name = (edge[0], edge[1])
        linear[var_name] = linear.get(var_name, 0.0) + edge[2]