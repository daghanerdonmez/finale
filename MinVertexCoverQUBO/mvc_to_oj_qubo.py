import dimod
from MinVertexCoverQUBO.MinVertexCover import MinVertexCover
from typing import Tuple


def mvc_to_oj_qubo(
    problem: MinVertexCover,
    penalty: float,
) -> Tuple[dict, float]:
    if penalty <= 0:
        raise ValueError("penalty must be positive")

    linear = {
        problem.variable_label(vertex): problem.weights[vertex]
        for vertex in problem.vertices
    }
    quadratic = {}
    offset = penalty * len(problem.edges)

    for u, v in problem.edges:
        u_var = problem.variable_label(u)
        v_var = problem.variable_label(v)

        linear[u_var] = linear.get(u_var, 0.0) - penalty
        linear[v_var] = linear.get(v_var, 0.0) - penalty

        key = tuple(sorted((u_var, v_var)))
        quadratic[key] = quadratic.get(key, 0.0) + penalty

    bqm = dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)
    return bqm.to_qubo()
