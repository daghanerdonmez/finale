"""
Feasibility comparison - Segment 1: run Gurobi on every (n, k, p, seed) instance
and save the optimal costs to a JSON file.  Run this locally where Gurobi is
available, then commit/push the JSON so Segment 2 can read it on the cluster.

For each (node_count n, terminal_count k, extra_edge_probability p) we build
NUM_INSTANCES_PER_COMBO problems with seeds 0..NUM_INSTANCES_PER_COMBO-1
using the sparsity generator.
"""
import json
import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.sparsity_problem_generator import (
    generate_sparsity_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp

# ── Parameters ──────────────────────────────────────────────────────
NODE_COUNT_LIST = [3,4,5,6,7]
TERMINAL_COUNT_LIST = [2,3,4,5]

# Separate experiments per density.  Each p in this list produces its own
# set of instances (different random graphs because p is part of the seed).
EDGE_PROBABILITY_LIST = [0.1, 0.3, 0.6]

NUM_INSTANCES_PER_COMBO = 5   # seeds 0 .. NUM_INSTANCES_PER_COMBO - 1
MAX_WEIGHT = 100
# ────────────────────────────────────────────────────────────────────


def _make_key(p, node_count, terminal_count, seed):
    return f"sparsity|p={p}|n={node_count}|k={terminal_count}|seed={seed}"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(
        os.path.dirname(__file__), "logs", "gurobi_feasibility_results"
    )
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(
        results_dir, f"feasibility_gurobi_{timestamp}.json"
    )

    results = {
        "_meta": {
            "timestamp": timestamp,
            "node_count_list": NODE_COUNT_LIST,
            "terminal_count_list": TERMINAL_COUNT_LIST,
            "edge_probability_list": EDGE_PROBABILITY_LIST,
            "num_instances_per_combo": NUM_INSTANCES_PER_COMBO,
            "max_weight": MAX_WEIGHT,
            "generator": "sparsity",
        },
        "instances": {},
    }

    # Count only combos where k <= n.
    total = 0
    for p in EDGE_PROBABILITY_LIST:
        for n in NODE_COUNT_LIST:
            for k in TERMINAL_COUNT_LIST:
                if k <= n:
                    total += NUM_INSTANCES_PER_COMBO
    done = 0

    for p in EDGE_PROBABILITY_LIST:
        for n in NODE_COUNT_LIST:
            for k in TERMINAL_COUNT_LIST:
                if k > n:
                    continue
                for seed in range(NUM_INSTANCES_PER_COMBO):
                    done += 1
                    key = _make_key(p, n, k, seed)
                    print(f"[{done}/{total}] {key}", flush=True)

                    problem = generate_sparsity_steiner_tree(
                        node_count=n,
                        terminal_count=k,
                        extra_edge_probability=p,
                        weight_range=(1, MAX_WEIGHT),
                        seed=seed,
                    )

                    t0 = time.time()
                    r_ilp = solve_ilp(problem)
                    t_ilp = time.time() - t0

                    results["instances"][key] = {
                        "family": "sparsity",
                        "seed": seed,
                        "node_count": n,
                        "terminal_count": k,
                        "edge_probability": p,
                        "num_nodes": len(problem.nodes),
                        "num_edges": len(problem.edges),
                        "terminals": problem.terminals,
                        "edges": [(a, b, w) for a, b, w in problem.edges],
                        "ilp_cost": r_ilp["cost"],
                        "ilp_time": round(t_ilp, 4),
                        "ilp_edges": [(a, b, w) for a, b, w in r_ilp["edges"]],
                    }

            # flush intermittently so partial progress survives crashes
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGurobi results saved to {results_path}")


if __name__ == "__main__":
    main()
