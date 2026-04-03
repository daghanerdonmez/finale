import time
from SteinerTreeProblemQUBO.random_problem_generator import generate_erdos_renyi_steiner_tree
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp as my_ilp
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver_binary import solve_ilp_binary as my_ilp_binary
from SteinerTreeProblemQUBO.Li_et_al.gurobi_solver import solve_ilp_li_et_al as li_ilp


if __name__ == "__main__":

    problem = generate_erdos_renyi_steiner_tree(
                        node_count=8,
                        terminal_count=4,
                        edge_probability=0.3,
                        weight_range=(1, 100),
                        seed=1,
                    )
    print(f"Nodes: {problem.nodes}")
    print(f"Terminals: {problem.terminals}")
    print(f"Edges: {problem.edges}")
    print()

    # --- My ILP ---
    print("=" * 60)
    print("My formulation — ILP (integer flow)")
    print("=" * 60)
    t0 = time.time()
    r1 = my_ilp(problem)
    t1 = time.time() - t0
    print(f"Status:  {r1['status']}")
    print(f"Cost:    {r1['cost']}")
    print(f"Time:    {t1:.4f}s")
    print(f"Edges:   {r1['edges']}")
    print()

    # --- My Binary ILP ---
    print("=" * 60)
    print("My formulation — Binary ILP (binary flow bits)")
    print("=" * 60)
    t0 = time.time()
    r2 = my_ilp_binary(problem)
    t2 = time.time() - t0
    print(f"Status:  {r2['status']}")
    print(f"Cost:    {r2['cost']}")
    print(f"Time:    {t2:.4f}s")
    print(f"Edges:   {r2['edges']}")
    print()

    # --- Li et al. ILP ---
    print("=" * 60)
    print("Li et al. formulation — linearized ILP")
    print("=" * 60)
    t0 = time.time()
    r3 = li_ilp(problem)
    t3 = time.time() - t0
    print(f"Status:    {r3['status']}")
    print(f"Objective: {r3['objective']}  (total path cost, may double-count)")
    print(f"Tree cost: {r3['tree_cost']}")
    print(f"Time:      {t3:.4f}s")
    print(f"Edges:     {r3['edges']}")
    print("Paths:")
    from collections import defaultdict
    paths = defaultdict(list)
    for k, i, s, val in r3["x_all"]:
        if val == 1:
            paths[k].append((s, i))
    for k in problem.terminals:
        steps = ", ".join(f"s{s}:{node}" for s, node in sorted(paths[k]))
        print(f"  terminal {k}: {steps}")
    print()
    print("X variables = 1:")
    for k, i, s, val in r3["x_all"]:
        if val == 1:
            print(f"  X[{k}, {i}, {s}]")
    print("Z variables = 1:")
    for k, i, j, s, val in r3["z_all"]:
        if val == 1:
            print(f"  Z[{k}, {i}, {j}, {s}]")
    print("U variables = 1:")
    for k, i, s, val in r3["u_all"]:
        if val == 1:
            print(f"  U[{k}, {i}, {s}]")
    print("R variables = 1:")
    for k, kp, i, s, val in r3["r_all"]:
        if val == 1:
            print(f"  R[{k}, {kp}, {i}, {s}]")
    print()

    # --- Summary ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  My ILP:        cost={r1['cost']:<10}  time={t1:.4f}s")
    print(f"  My Binary ILP: cost={r2['cost']:<10}  time={t2:.4f}s")
    print(f"  Li et al. ILP: cost={r3['tree_cost']:<10}  time={t3:.4f}s")
    if r1['cost'] is not None and r3['tree_cost'] is not None:
        match = abs(r1['cost'] - r3['tree_cost']) < 1e-6
        print(f"  All agree on tree cost: {match}")
