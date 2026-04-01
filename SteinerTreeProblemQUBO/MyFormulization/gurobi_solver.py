import gurobipy as gp
from gurobipy import GRB
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def solve_ilp(problem: SteinerTree):
    """
    Solve the Steiner Tree problem using the compact ILP formulation
    with integer flow variables f_uv, edge selection x_e, usage y_e,
    and direction-activation a_uv.
    """
    root = problem.terminals[0]
    M = len(problem.terminals) - 1  # max flow = |K| - 1

    model = gp.Model("SteinerTree_ILP")

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

    # y_e: edge actually used by flow
    y = {}
    for a, b, _ in problem.edges:
        y[a, b] = model.addVar(vtype=GRB.BINARY, name=f"y_{a}_{b}")

    # a_uv: direction-activation binaries
    a_var = {}
    for a, b, _ in problem.edges:
        a_var[a, b] = model.addVar(vtype=GRB.BINARY, name=f"a_{a}_{b}")
        a_var[b, a] = model.addVar(vtype=GRB.BINARY, name=f"a_{b}_{a}")

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
            d_v = M
        elif v in problem.terminals:
            d_v = -1
        else:
            d_v = 0

        model.addConstr(outflow - inflow == d_v, name=f"flow_{v}")

    for a, b, _ in problem.edges:
        # a_uv + a_vu = x_e
        model.addConstr(a_var[a, b] + a_var[b, a] == x[a, b],
                        name=f"dir_{a}_{b}")

        # f_uv <= M * a_uv
        model.addConstr(f[a, b] <= M * a_var[a, b],
                        name=f"fcap_{a}_{b}")
        model.addConstr(f[b, a] <= M * a_var[b, a],
                        name=f"fcap_{b}_{a}")

        # f_uv + f_vu <= M * y_e
        model.addConstr(f[a, b] + f[b, a] <= M * y[a, b],
                        name=f"use_ub_{a}_{b}")
        # f_uv + f_vu >= y_e
        model.addConstr(f[a, b] + f[b, a] >= y[a, b],
                        name=f"use_lb_{a}_{b}")

        # x_e <= y_e
        model.addConstr(x[a, b] <= y[a, b],
                        name=f"sel_{a}_{b}")

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
