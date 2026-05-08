"""
New models geometric benchmark - Segment 1: run Gurobi on every
(n, k_terminals, k_neighbors, seed) instance and save optimal costs to JSON.
Run this locally where Gurobi is available, then commit/push the JSON so
Segment 2 can read it on the cluster.

Uses generate_geometric_steiner_tree with knn connectivity.
The varying density parameter is K_NEIGHBORS (number of nearest neighbors),
which plays the same role as edge_probability in the sparsity benchmark.
"""
import json
import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_geometric_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp

# ── Parameters ──────────────────────────────────────────────────────
NODE_COUNT_LIST = [4, 6, 8, 10, 12]
TERMINAL_COUNT_LIST = [2, 3, 4, 5, 6]
K_NEIGHBORS_LIST = [3, 5, 8]       # knn connectivity; analogous to edge_probability
NUM_INSTANCES_PER_COMBO = 10       # seeds 0 .. NUM_INSTANCES_PER_COMBO - 1
MAX_WEIGHT = 100
# ────────────────────────────────────────────────────────────────────


def _make_key(k_neighbors, node_count, terminal_count, seed):
    return f"geo|knn={k_neighbors}|n={node_count}|k={terminal_count}|seed={seed}"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        os.path.dirname(__file__), "logs", "gurobi_new_models_geo_results"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir, f"new_models_geo_gurobi_{timestamp}.json"
    )

    results = {
        "_meta": {
            "timestamp": timestamp,
            "node_count_list": NODE_COUNT_LIST,
            "terminal_count_list": TERMINAL_COUNT_LIST,
            "k_neighbors_list": K_NEIGHBORS_LIST,
            "num_instances_per_combo": NUM_INSTANCES_PER_COMBO,
            "max_weight": MAX_WEIGHT,
            "generator": "geometric_knn",
        },
        "instances": {},
    }

    total = sum(
        NUM_INSTANCES_PER_COMBO
        for knn in K_NEIGHBORS_LIST
        for n in NODE_COUNT_LIST
        for k in TERMINAL_COUNT_LIST
        if k <= n
    )
    done = 0

    for knn in K_NEIGHBORS_LIST:
        for n in NODE_COUNT_LIST:
            for k in TERMINAL_COUNT_LIST:
                if k > n:
                    continue
                for seed in range(NUM_INSTANCES_PER_COMBO):
                    done += 1
                    key = _make_key(knn, n, k, seed)
                    print(f"[{done}/{total}] {key}", flush=True)

                    problem = generate_geometric_steiner_tree(
                        node_count=n,
                        terminal_count=k,
                        max_weight=MAX_WEIGHT,
                        connectivity="knn",
                        k=knn,
                        seed=seed,
                    )

                    t0 = time.time()
                    r_ilp = solve_ilp(problem)
                    t_ilp = time.time() - t0

                    results["instances"][key] = {
                        "family": "geometric_knn",
                        "seed": seed,
                        "node_count": n,
                        "terminal_count": k,
                        "k_neighbors": knn,
                        "num_nodes": len(problem.nodes),
                        "num_edges": len(problem.edges),
                        "terminals": problem.terminals,
                        "edges": [(a, b, w) for a, b, w in problem.edges],
                        "ilp_cost": r_ilp["cost"],
                        "ilp_time": round(t_ilp, 4),
                        "ilp_edges": [(a, b, w) for a, b, w in r_ilp["edges"]],
                    }

            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGurobi results saved to {results_path}")


if __name__ == "__main__":
    main()
