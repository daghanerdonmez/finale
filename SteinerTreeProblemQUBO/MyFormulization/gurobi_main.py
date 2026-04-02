import time
from SteinerTreeProblemQUBO.random_problem_generator import generate_random_steiner_tree
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver_binary import solve_ilp_binary


if __name__ == "__main__":

    problem = generate_random_steiner_tree(10, (10, 100), 3, 0.3, 20)
    print("SteinerTree object created")
    print(f"Nodes: {problem.nodes}")
    print(f"Terminals: {problem.terminals}")
    print(f"Edges: {problem.edges}")
    print()

    # --- ILP (integer flow) ---
    print("=" * 50)
    print("Solving with ILP (integer flow variables)")
    print("=" * 50)
    start = time.time()
    result_ilp = solve_ilp(problem)
    ilp_time = time.time() - start
    print(f"Status: {result_ilp['status']}")
    print(f"Optimal cost: {result_ilp['cost']}")
    print(f"Solution time: {ilp_time:.4f} seconds")
    print("Selected edges:")
    for u, v, w in result_ilp["edges"]:
        print(f"  ({u}, {v}) weight={w}")
    print()

    # --- Binarized ILP ---
    print("=" * 50)
    print("Solving with binarized ILP (binary flow bits)")
    print("=" * 50)
    start = time.time()
    result_bin = solve_ilp_binary(problem)
    bin_time = time.time() - start
    print(f"Status: {result_bin['status']}")
    print(f"Optimal cost: {result_bin['cost']}")
    print(f"Solution time: {bin_time:.4f} seconds")
    print("Selected edges:")
    for u, v, w in result_bin["edges"]:
        print(f"  ({u}, {v}) weight={w}")
    print("Flow values:")
    for (u, v), val in result_bin["flows"].items():
        print(f"  {u} -> {v}: {val}")
    print()

    # --- Comparison ---
    if result_ilp["cost"] is not None and result_bin["cost"] is not None:
        match = abs(result_ilp["cost"] - result_bin["cost"]) < 1e-6
        print(f"Both formulations agree: {match}")
