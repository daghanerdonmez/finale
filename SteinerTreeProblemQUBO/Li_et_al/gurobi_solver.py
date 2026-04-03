import gurobipy as gp
from gurobipy import GRB
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def solve_ilp_li_et_al(problem: SteinerTree, non_edge_penalty: float = 10000.0):
    """
    Linearized ILP version of the Li et al. formulation.

    This tries to follow the paper as written:

        H_A = sum_{i,j,k,s} w_ij X^k_{i,s} X^k_{j,s+1}

    with
        w_ii = 0
        w_ij = edge weight if (i,j) in E
        w_ij = non_edge_penalty if (i,j) not in E

    Constraints implemented:
        1) X[k1, root, 0] = 1
        2) X[k, k, S] = 1
        3) sum_i X[k,i,s] <= 1
        4) sum_i X[k,i,s] <= sum_i X[k,i,s+1]
        5) sum_{k,i notin K,s} X[k,i,s] * X[k,i,s+1] = 0
        6) For each k: sum_{i,s,k'!=k} X[k,i,s] * X[k',i,s] >= 1

    Warning:
    This is the paper's path-based formulation, not a correct compact Steiner ILP.
    """

    K = list(problem.terminals)
    V = list(problem.nodes)
    n = len(V)
    S = n  # paper says typically S = n
    if 0 in V:
        root = 0
    elif len(K) > 0:
        root = K[0]
    else:
        raise ValueError("No terminals found, cannot choose a root.")

    # Build full weight matrix exactly as in the paper:
    # w_ii = 0, edge weight if edge exists, otherwise non_edge_penalty
    edge_weight = {}
    for a, b, w in problem.edges:
        edge_weight[a, b] = w
        edge_weight[b, a] = w

    W = {}
    for i in V:
        for j in V:
            if i == j:
                W[i, j] = 0.0
            elif (i, j) in edge_weight:
                W[i, j] = float(edge_weight[i, j])
            else:
                W[i, j] = float(non_edge_penalty)

    model = gp.Model("Li_et_al_exact_linearized")
    model.setParam("OutputFlag", 0)

    # ------------------------------------------------------------------
    # Variables
    # ------------------------------------------------------------------

    # X[k,i,s]
    X = model.addVars(
        [(k, i, s) for k in K for i in V for s in range(S + 1)],
        vtype=GRB.BINARY,
        name="X"
    )

    # Z[k,i,j,s] = X[k,i,s] * X[k,j,s+1]
    # IMPORTANT: full i,j over all nodes, as in paper objective
    Z = model.addVars(
        [(k, i, j, s) for k in K for i in V for j in V for s in range(S)],
        vtype=GRB.BINARY,
        name="Z"
    )

    # U[k,i,s] = X[k,i,s] * X[k,i,s+1] for non-terminal i
    non_terminals = [i for i in V if i not in K]
    U = model.addVars(
        [(k, i, s) for k in K for i in non_terminals for s in range(S)],
        vtype=GRB.BINARY,
        name="U"
    )

    # R[k,kp,i,s] = X[k,i,s] * X[kp,i,s] for k != kp
    R = model.addVars(
        [(k, kp, i, s)
         for k in K for kp in K if kp != k
         for i in V for s in range(S + 1)],
        vtype=GRB.BINARY,
        name="R"
    )

    model.update()

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    model.setObjective(
        gp.quicksum(
            W[i, j] * Z[k, i, j, s]
            for k in K
            for i in V
            for j in V
            for s in range(S)
        ),
        GRB.MINIMIZE
    )

    # ------------------------------------------------------------------
    # Constraints from paper
    # ------------------------------------------------------------------

    # 1) Root constraint: X[k1, 0, 0] = 1
    # paper uses node 0 as root and k1 as first target
    if root not in V:
        raise ValueError("Paper assumes root node 0, but node 0 is not in the graph.")
    if K[0] not in K:
        raise ValueError("Terminal list is empty.")
    model.addConstr(X[K[0], root, 0] == 1, name="root")

    # 2) Terminal end: X[k, k, S] = 1 for all k
    for k in K:
        if k not in V:
            raise ValueError(f"Terminal {k} is not a node in the graph.")
        model.addConstr(X[k, k, S] == 1, name=f"terminal_end_{k}")

    # 3) At most one node per time step
    for k in K:
        for s in range(S + 1):
            model.addConstr(
                gp.quicksum(X[k, i, s] for i in V) <= 1,
                name=f"one_node_{k}_{s}"
            )

    # 4) Path continuity: occupancy cannot disappear and then reappear
    for k in K:
        for s in range(S):
            model.addConstr(
                gp.quicksum(X[k, i, s] for i in V) <= gp.quicksum(X[k, i, s + 1] for i in V),
                name=f"continuity_{k}_{s}"
            )

    # 5) No consecutive stay at non-terminal nodes:
    # sum_{k, i notin K, s} U[k,i,s] = 0
    if len(non_terminals) > 0:
        model.addConstr(
            gp.quicksum(U[k, i, s] for k in K for i in non_terminals for s in range(S)) == 0,
            name="no_stay_nonterminal"
        )

    # 6) Steiner-point overlap: for each k, at least one overlap with another path
    for k in K:
        model.addConstr(
            gp.quicksum(
                R[k, kp, i, s]
                for kp in K if kp != k
                for i in V
                for s in range(S + 1)
            ) >= 1,
            name=f"overlap_{k}"
        )

    # ------------------------------------------------------------------
    # Linearization constraints
    # ------------------------------------------------------------------

    # Z[k,i,j,s] = X[k,i,s] * X[k,j,s+1]
    for k in K:
        for i in V:
            for j in V:
                for s in range(S):
                    model.addConstr(Z[k, i, j, s] <= X[k, i, s], name=f"Zub1_{k}_{i}_{j}_{s}")
                    model.addConstr(Z[k, i, j, s] <= X[k, j, s + 1], name=f"Zub2_{k}_{i}_{j}_{s}")
                    model.addConstr(
                        Z[k, i, j, s] >= X[k, i, s] + X[k, j, s + 1] - 1,
                        name=f"Zlb_{k}_{i}_{j}_{s}"
                    )

    # U[k,i,s] = X[k,i,s] * X[k,i,s+1]
    for k in K:
        for i in non_terminals:
            for s in range(S):
                model.addConstr(U[k, i, s] <= X[k, i, s], name=f"Uub1_{k}_{i}_{s}")
                model.addConstr(U[k, i, s] <= X[k, i, s + 1], name=f"Uub2_{k}_{i}_{s}")
                model.addConstr(
                    U[k, i, s] >= X[k, i, s] + X[k, i, s + 1] - 1,
                    name=f"Ulb_{k}_{i}_{s}"
                )

    # R[k,kp,i,s] = X[k,i,s] * X[kp,i,s]
    for k in K:
        for kp in K:
            if kp == k:
                continue
            for i in V:
                for s in range(S + 1):
                    model.addConstr(R[k, kp, i, s] <= X[k, i, s], name=f"Rub1_{k}_{kp}_{i}_{s}")
                    model.addConstr(R[k, kp, i, s] <= X[kp, i, s], name=f"Rub2_{k}_{kp}_{i}_{s}")
                    model.addConstr(
                        R[k, kp, i, s] >= X[k, i, s] + X[kp, i, s] - 1,
                        name=f"Rlb_{k}_{kp}_{i}_{s}"
                    )

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # Extract used transitions
        used_transitions = []
        for k in K:
            for i in V:
                for j in V:
                    for s in range(S):
                        if Z[k, i, j, s].X > 0.5:
                            used_transitions.append((k, i, j, s, W[i, j]))

        # Extract actual graph edges used (ignore self-transitions and non-edges)
        selected_edges = set()
        for _, i, j, _, _ in used_transitions:
            if i != j and (i, j) in edge_weight:
                selected_edges.add(tuple(sorted((i, j))))

        edge_list = []
        tree_cost = 0.0
        for a, b in sorted(selected_edges):
            w = edge_weight[a, b]
            edge_list.append((a, b, w))
            tree_cost += w

        # Extract all variable values
        x_all = []
        for k in K:
            for i in V:
                for s in range(S + 1):
                    x_all.append((k, i, s, int(round(X[k, i, s].X))))

        z_all = []
        for k in K:
            for i in V:
                for j in V:
                    for s in range(S):
                        z_all.append((k, i, j, s, int(round(Z[k, i, j, s].X))))

        u_all = []
        for k in K:
            for i in non_terminals:
                for s in range(S):
                    u_all.append((k, i, s, int(round(U[k, i, s].X))))

        r_all = []
        for k in K:
            for kp in K:
                if kp == k:
                    continue
                for i in V:
                    for s in range(S + 1):
                        r_all.append((k, kp, i, s, int(round(R[k, kp, i, s].X))))

        return {
            "status": "OPTIMAL",
            "objective": model.ObjVal,
            "tree_cost": tree_cost,
            "edges": edge_list,
            "used_transitions": used_transitions,
            "x_all": x_all,
            "z_all": z_all,
            "u_all": u_all,
            "r_all": r_all,
            "model": model,
        }

    return {
        "status": model.status,
        "objective": None,
        "tree_cost": None,
        "edges": [],
        "used_transitions": [],
        "model": model,
    }