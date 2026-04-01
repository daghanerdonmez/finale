import math
import gurobipy as gp
from gurobipy import GRB
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def solve_ilp_binary(problem: SteinerTree):
    """
    Solve the Steiner Tree problem using the binarized ILP formulation.
    Flow variables are replaced by their binary expansion:
        f_uv = sum_{b=0}^{B-1} 2^b * z_{uv,b}
    where B = ceil(log2(|K|)).
    """
    root = problem.terminals[0]
    M = len(problem.terminals) - 1  # max flow = |K| - 1
    B = math.ceil(math.log2(len(problem.terminals)))

    model = gp.Model("SteinerTree_BinaryILP")

    # ---- Variables ----

    # x_e: edge selected
    x = {}
    for a, b, w in problem.edges:
        x[a, b] = model.addVar(vtype=GRB.BINARY, name=f"x_{a}_{b}")

    # z_{uv,bit}: flow bits for each directed arc
    z = {}
    for a, b, _ in problem.edges:
        for bit in range(B):
            z[a, b, bit] = model.addVar(vtype=GRB.BINARY, name=f"z_{a}_{b}_{bit}")
            z[b, a, bit] = model.addVar(vtype=GRB.BINARY, name=f"z_{b}_{a}_{bit}")

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

    # Helper: reconstruct flow as linear expression from bits
    def flow_expr(u, v):
        return gp.quicksum(2**bit * z[u, v, bit] for bit in range(B))

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
                outflow += flow_expr(a, b)
                inflow += flow_expr(b, a)
            if b == v:
                outflow += flow_expr(b, a)
                inflow += flow_expr(a, b)

        if v == root:
            d_v = M
        elif v in problem.terminals:
            d_v = -1
        else:
            d_v = 0

        model.addConstr(outflow - inflow == d_v, name=f"flow_{v}")

    for a, b, _ in problem.edges:
        f_ab = flow_expr(a, b)
        f_ba = flow_expr(b, a)

        # a_uv + a_vu = x_e
        model.addConstr(a_var[a, b] + a_var[b, a] == x[a, b],
                        name=f"dir_{a}_{b}")

        # f_uv <= M * a_uv
        model.addConstr(f_ab <= M * a_var[a, b],
                        name=f"fcap_{a}_{b}")
        model.addConstr(f_ba <= M * a_var[b, a],
                        name=f"fcap_{b}_{a}")

        # f_uv + f_vu <= M * y_e
        model.addConstr(f_ab + f_ba <= M * y[a, b],
                        name=f"use_ub_{a}_{b}")
        # f_uv + f_vu >= y_e
        model.addConstr(f_ab + f_ba >= y[a, b],
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

        # Extract flow values for inspection
        flows = {}
        for a, b, _ in problem.edges:
            val_ab = sum(2**bit * z[a, b, bit].X for bit in range(B))
            val_ba = sum(2**bit * z[b, a, bit].X for bit in range(B))
            if val_ab > 0.5:
                flows[a, b] = round(val_ab)
            if val_ba > 0.5:
                flows[b, a] = round(val_ba)

        return {
            "cost": model.ObjVal,
            "edges": selected_edges,
            "flows": flows,
            "status": "OPTIMAL",
            "model": model,
        }
    else:
        return {
            "cost": None,
            "edges": [],
            "flows": {},
            "status": model.status,
            "model": model,
        }
