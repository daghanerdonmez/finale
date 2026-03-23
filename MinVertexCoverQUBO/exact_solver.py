import pulp

from MinVertexCoverQUBO.MinVertexCover import MinVertexCover


def solve(problem: MinVertexCover) -> dict:
    model = pulp.LpProblem("minimum_vertex_cover", pulp.LpMinimize)
    variables = {
        vertex: pulp.LpVariable(
            f"x__{vertex}",
            lowBound=0,
            upBound=1,
            cat=pulp.LpBinary,
        )
        for vertex in problem.vertices
    }

    model += pulp.lpSum(problem.weights[vertex] * variables[vertex] for vertex in problem.vertices)

    for u, v in problem.edges:
        model += variables[u] + variables[v] >= 1, f"cover_{u}_{v}"

    solver = pulp.PULP_CBC_CMD(msg=False)
    status = model.solve(solver)
    if status != pulp.LpStatusOptimal:
        raise RuntimeError(f"exact solver failed with status {pulp.LpStatus[status]}")

    cover = [
        vertex
        for vertex in problem.vertices
        if pulp.value(variables[vertex]) is not None and pulp.value(variables[vertex]) > 0.5
    ]
    cover_weight = problem.cover_weight(cover)
    independent_set = [vertex for vertex in problem.vertices if vertex not in set(cover)]

    return {
        "cover": cover,
        "cover_weight": cover_weight,
        "optimal_energy": cover_weight,
        "independent_set": independent_set,
        "independent_set_weight": sum(problem.weights[vertex] for vertex in independent_set),
        "total_weight": sum(problem.weights.values()),
    }
