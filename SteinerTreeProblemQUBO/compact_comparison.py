import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_geometric_steiner_tree,
    generate_erdos_renyi_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp
from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver_binary import solve_ilp_binary
from SteinerTreeProblemQUBO.MyFormulization.oj_solver import solve_with_sqa

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

# OJ solver
CONSTRAINT_WEIGHT = MAX_WEIGHT
OJ_VERSION = 2
OJ_BATCH_SIZE = 100
OJ_MAX_READS = 10000
OJ_NUM_SWEEPS = 2000
OJ_TROTTER = 8
# ────────────────────────────────────────────────────────────────────


def _run_one(problem, log):
    """Solve a single instance with all three methods and write results to log."""
    # --- Gurobi ILP ---
    t0 = time.time()
    r_ilp = solve_ilp(problem)
    t_ilp = time.time() - t0
    log.write(f"  Gurobi ILP        | cost: {r_ilp['cost']:<10} | time: {t_ilp:.4f}s\n")

    # --- Gurobi Binary ILP ---
    t0 = time.time()
    r_bin = solve_ilp_binary(problem)
    t_bin = time.time() - t0
    log.write(f"  Gurobi Binary ILP | cost: {r_bin['cost']:<10} | time: {t_bin:.4f}s\n")

    # --- OJ SQA (incremental batches) ---
    optimal_cost = r_ilp["cost"]
    t0 = time.time()
    total_reads = 0
    best_oj_cost = float("inf")
    matched_at = None

    while total_reads < OJ_MAX_READS:
        r_oj = solve_with_sqa(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            version=OJ_VERSION,
            num_reads=OJ_BATCH_SIZE,
            show_stats=False,
            show_progress=False,
            num_sweeps=OJ_NUM_SWEEPS,
            trotter=OJ_TROTTER,
        )
        total_reads += OJ_BATCH_SIZE
        oj_cost = r_oj["best_energy_with_offset"]
        if oj_cost < best_oj_cost:
            best_oj_cost = oj_cost

        if optimal_cost is not None and abs(best_oj_cost - optimal_cost) < 1e-6:
            matched_at = total_reads
            break

    t_oj = time.time() - t0

    if matched_at is not None:
        log.write(f"  OJ SQA            | cost: {best_oj_cost:<10.1f} | time: {t_oj:.4f}s | matched at {matched_at} reads\n")
    else:
        log.write(f"  OJ SQA            | cost: {best_oj_cost:<10.1f} | time: {t_oj:.4f}s | no match in {total_reads} reads\n")
    log.flush()


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"comparison_{timestamp}.txt")

    with open(log_path, "w") as log:
        log.write(f"Steiner Tree Comparison  |  {timestamp}\n")
        log.write(f"node_counts: {NODE_COUNT_LIST}  terminals: {TERMINAL_COUNT}  max_weight: {MAX_WEIGHT}\n")
        log.write(f"OJ params: constraint_weight={CONSTRAINT_WEIGHT}  version={OJ_VERSION}  batch={OJ_BATCH_SIZE}  max_reads={OJ_MAX_READS}  num_sweeps={OJ_NUM_SWEEPS}  trotter={OJ_TROTTER}\n")
        log.write(f"Geometric params: connectivity={GEO_CONNECTIVITY}  k_values={GEO_K_LIST}\n")
        log.write(f"Erdos-Renyi params: edge_probabilities={ER_EDGE_PROB_LIST}\n")
        log.write(f"Seeds: {SEED_START}-{SEED_END}\n")
        log.write("=" * 70 + "\n\n")

        num_seeds = SEED_END - SEED_START + 1
        num_per_seed = len(NODE_COUNT_LIST) * (len(GEO_K_LIST) + len(ER_EDGE_PROB_LIST))
        total = num_seeds * num_per_seed
        done = 0

        for seed in range(SEED_START, SEED_END + 1):
            for node_count in NODE_COUNT_LIST:
                # ── Geometric (loop over k values) ──
                for geo_k in GEO_K_LIST:
                    done += 1
                    print(f"[{done}/{total}] seed={seed}  n={node_count}  Geometric k={geo_k}", flush=True)
                    log.write(f"Seed {seed} | n={node_count} | Geometric ({GEO_CONNECTIVITY}, k={geo_k})\n")
                    problem_geo = generate_geometric_steiner_tree(
                        node_count=node_count,
                        terminal_count=TERMINAL_COUNT,
                        max_weight=MAX_WEIGHT,
                        connectivity=GEO_CONNECTIVITY,
                        k=geo_k,
                        seed=seed,
                    )
                    log.write(f"  |V|={len(problem_geo.nodes)}  |E|={len(problem_geo.edges)}  terminals={problem_geo.terminals}\n")
                    _run_one(problem_geo, log)
                    log.write("\n")

                # ── Erdos-Renyi (loop over edge probabilities) ──
                for er_p in ER_EDGE_PROB_LIST:
                    done += 1
                    print(f"[{done}/{total}] seed={seed}  n={node_count}  Erdos-Renyi p={er_p}", flush=True)
                    log.write(f"Seed {seed} | n={node_count} | Erdos-Renyi (p={er_p})\n")
                    problem_er = generate_erdos_renyi_steiner_tree(
                        node_count=node_count,
                        terminal_count=TERMINAL_COUNT,
                        edge_probability=er_p,
                        weight_range=(1, MAX_WEIGHT),
                        seed=seed,
                    )
                    log.write(f"  |V|={len(problem_er.nodes)}  |E|={len(problem_er.edges)}  terminals={problem_er.terminals}\n")
                    _run_one(problem_er, log)
                    log.write("\n")

            log.write("-" * 70 + "\n")
            log.flush()

        log.write("\nDone.\n")

    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
