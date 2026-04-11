"""
Segment 1: Run Gurobi solvers and save optimal costs to a JSON file.
Run this locally where Gurobi is available, then commit/push the results.
"""
import json
import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_geometric_steiner_tree,
    generate_erdos_renyi_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver_binary import solve_ilp_binary

# ── Parameters ──────────────────────────────────────────────────────
SEED_START = 1
SEED_END = 10

NODE_COUNT_LIST = [4, 6, 8, 10, 12]
TERMINAL_COUNT = 3
MAX_WEIGHT = 100

# Geometric generator
GEO_CONNECTIVITY = "knn"
GEO_K_LIST = [3, 5, 8]

# Erdos-Renyi generator
ER_EDGE_PROB_LIST = [0.1, 0.3, 0.6]
# ────────────────────────────────────────────────────────────────────


def _make_key(family, seed, node_count, param):
    """Unique string key for each instance."""
    return f"{family}|seed={seed}|n={node_count}|param={param}"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(os.path.dirname(__file__), "logs", "gurobi_results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, f"gurobi_{timestamp}.json")

    results = {
        "_meta": {
            "timestamp": timestamp,
            "seed_start": SEED_START,
            "seed_end": SEED_END,
            "node_count_list": NODE_COUNT_LIST,
            "terminal_count": TERMINAL_COUNT,
            "max_weight": MAX_WEIGHT,
            "geo_connectivity": GEO_CONNECTIVITY,
            "geo_k_list": GEO_K_LIST,
            "er_edge_prob_list": ER_EDGE_PROB_LIST,
        },
        "instances": {},
    }

    num_seeds = SEED_END - SEED_START + 1
    num_per_seed = len(NODE_COUNT_LIST) * (len(GEO_K_LIST) + len(ER_EDGE_PROB_LIST))
    total = num_seeds * num_per_seed
    done = 0

    for seed in range(SEED_START, SEED_END + 1):
        for node_count in NODE_COUNT_LIST:
            # ── Geometric ──
            for geo_k in GEO_K_LIST:
                done += 1
                key = _make_key("geometric", seed, node_count, f"k={geo_k}")
                print(f"[{done}/{total}] {key}", flush=True)

                problem = generate_geometric_steiner_tree(
                    node_count=node_count,
                    terminal_count=TERMINAL_COUNT,
                    max_weight=MAX_WEIGHT,
                    connectivity=GEO_CONNECTIVITY,
                    k=geo_k,
                    seed=seed,
                )

                t0 = time.time()
                r_ilp = solve_ilp(problem)
                t_ilp = time.time() - t0

                t0 = time.time()
                r_bin = solve_ilp_binary(problem)
                t_bin = time.time() - t0

                results["instances"][key] = {
                    "family": "geometric",
                    "seed": seed,
                    "node_count": node_count,
                    "param": f"k={geo_k}",
                    "num_nodes": len(problem.nodes),
                    "num_edges": len(problem.edges),
                    "terminals": problem.terminals,
                    "ilp_cost": r_ilp["cost"],
                    "ilp_time": round(t_ilp, 4),
                    "ilp_edges": [(a, b, w) for a, b, w in r_ilp["edges"]],
                    "bin_cost": r_bin["cost"],
                    "bin_time": round(t_bin, 4),
                    "bin_edges": [(a, b, w) for a, b, w in r_bin["edges"]],
                }

            # ── Erdos-Renyi ──
            for er_p in ER_EDGE_PROB_LIST:
                done += 1
                key = _make_key("erdos_renyi", seed, node_count, f"p={er_p}")
                print(f"[{done}/{total}] {key}", flush=True)

                problem = generate_erdos_renyi_steiner_tree(
                    node_count=node_count,
                    terminal_count=TERMINAL_COUNT,
                    edge_probability=er_p,
                    weight_range=(1, MAX_WEIGHT),
                    seed=seed,
                )

                t0 = time.time()
                r_ilp = solve_ilp(problem)
                t_ilp = time.time() - t0

                t0 = time.time()
                r_bin = solve_ilp_binary(problem)
                t_bin = time.time() - t0

                results["instances"][key] = {
                    "family": "erdos_renyi",
                    "seed": seed,
                    "node_count": node_count,
                    "param": f"p={er_p}",
                    "num_nodes": len(problem.nodes),
                    "num_edges": len(problem.edges),
                    "terminals": problem.terminals,
                    "ilp_cost": r_ilp["cost"],
                    "ilp_time": round(t_ilp, 4),
                    "ilp_edges": [(a, b, w) for a, b, w in r_ilp["edges"]],
                    "bin_cost": r_bin["cost"],
                    "bin_time": round(t_bin, 4),
                    "bin_edges": [(a, b, w) for a, b, w in r_bin["edges"]],
                }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nGurobi results saved to {results_path}")


if __name__ == "__main__":
    main()
