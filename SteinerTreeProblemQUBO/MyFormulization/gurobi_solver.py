import gurobipy as gp
from gurobipy import GRB
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def solve_ilp(problem: SteinerTree):
    """
    Solve the Steiner Tree problem using the compact ILP formulation
    with integer flow variables f_uv, edge selection x_e.
    """
    root = problem.terminals[0]
    M = len(problem.terminals) - 1  # max flow = |K| - 1

    model = gp.Model("SteinerTree_ILP")
    model.setParam("OutputFlag", 0)

    # ---- Variables ----

    # x_e: edge selected
    x = {}
    for a, b, w in problem.edges:
        x[a, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{b}")

    # f_uv: integer flow on each directed arc
    f = {}
    for a, b, _ in problem.edges:
        f[a, b] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"f_{a}_{b}")
        f[b, a] = model.addVar(vtype=GRB.INTEGER, lb=0, ub=M, name=f"f_{b}_{a}")

    model.update()

    # ---- Objective ----
    model.setObjective(
        gp.quicksum(w * x[a, b] for a, b, w in problem.edges),
        GRB.MINIMIZE,
    )

    # ---- Constraints ----

    # Flow conservation: for every node v
    for v in problem.nodes:
        outflow = gp.LinExpr()
        inflow = gp.LinExpr()
        for a, b, _ in problem.edges:
            if a == v:
                outflow += f[a, b]
                inflow += f[b, a]
            if b == v:
                outflow += f[b, a]
                inflow += f[a, b]

        if v == root:
            d_v = -M
        elif v in problem.terminals:
            d_v = 1
        else:
            d_v = 0

        model.addConstr(inflow - outflow == d_v, name=f"flow_{v}")

    for a, b, _ in problem.edges:
        # f_uv + f_vu <= M * y_e
        model.addConstr(f[a, b] + f[b, a] <= M * x[a, b],
                        name=f"use_ub_{a}_{b}")

    # ---- Solve ----
    model.optimize()

    if model.status == GRB.OPTIMAL:
        selected_edges = []
        for a, b, w in problem.edges:
            if x[a, b].X > 0.5:
                selected_edges.append((a, b, w))
        return {
            "cost": model.ObjVal,
            "edges": selected_edges,
            "status": "OPTIMAL",
            "model": model,
        }
    else:
        return {
            "cost": None,
            "edges": [],
            "status": model.status,
            "model": model,
        }
