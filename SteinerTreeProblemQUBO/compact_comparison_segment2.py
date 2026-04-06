"""
Segment 2: Run OJ SQA solver on the cluster, comparing against Gurobi results.
Reads the JSON produced by segment 1 to get optimal costs for comparison.
"""
import glob
import json
import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_geometric_steiner_tree,
    generate_erdos_renyi_steiner_tree,
)
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
OJ_BATCH_SIZE = 100
OJ_MAX_READS = 10000

# Each config: (label, version, num_sweeps, trotter)
# None means use openjij defaults
OJ_CONFIGS = [
    ("v1_tuned",   1, 2000, 8),
    ("v2_tuned",   2, 2000, 8),
    ("v4_tuned",   4, 2000, 8),
    ("v1_default", 1, None, None),
    ("v2_default", 2, None, None),
    ("v4_default", 4, None, None),
]
# ────────────────────────────────────────────────────────────────────


def _make_key(family, seed, node_count, param):
    return f"{family}|seed={seed}|n={node_count}|param={param}"


def _load_gurobi_results():
    """Find the most recent gurobi results JSON."""
    results_dir = os.path.join(os.path.dirname(__file__), "logs", "gurobi_results")
    files = sorted(glob.glob(os.path.join(results_dir, "gurobi_*.json")))
    if not files:
        raise FileNotFoundError(
            f"No gurobi results found in {results_dir}. Run segment 1 first."
        )
    latest = files[-1]
    print(f"Loading Gurobi results from {latest}")
    with open(latest) as f:
        return json.load(f)


def _run_oj(problem, optimal_cost, label, version, num_sweeps, trotter, log):
    """Run OJ SQA in batches, stop early if it matches optimal."""
    t0 = time.time()
    total_reads = 0
    best_oj_cost = float("inf")
    matched_at = None

    extra_kwargs = {}
    if num_sweeps is not None:
        extra_kwargs["num_sweeps"] = num_sweeps
    if trotter is not None:
        extra_kwargs["trotter"] = trotter

    while total_reads < OJ_MAX_READS:
        r_oj = solve_with_sqa(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            version=version,
            num_reads=OJ_BATCH_SIZE,
            show_stats=False,
            show_progress=False,
            **extra_kwargs,
        )
        total_reads += OJ_BATCH_SIZE
        oj_cost = r_oj["best_energy_with_offset"]
        if oj_cost < best_oj_cost:
            best_oj_cost = oj_cost

        if optimal_cost is not None and abs(best_oj_cost - optimal_cost) < 1e-6:
            matched_at = total_reads
            break

    t_oj = time.time() - t0

    sweep_str = str(num_sweeps) if num_sweeps is not None else "default"
    trot_str = str(trotter) if trotter is not None else "default"
    match_str = f"matched at {matched_at} reads" if matched_at else f"no match in {total_reads} reads"
    log.write(f"  OJ {label:<14s} | cost: {best_oj_cost:<10.1f} | time: {t_oj:.4f}s | sweeps={sweep_str} trotter={trot_str} | {match_str}\n")


def main():
    gurobi_data = _load_gurobi_results()
    instances = gurobi_data["instances"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"comparison_{timestamp}.txt")

    num_seeds = SEED_END - SEED_START + 1
    num_per_seed = len(NODE_COUNT_LIST) * (len(GEO_K_LIST) + len(ER_EDGE_PROB_LIST))
    total = num_seeds * num_per_seed
    done = 0

    config_desc = ", ".join(f"{l}(v{v} sw={s} tr={t})" for l, v, s, t in OJ_CONFIGS)

    with open(log_path, "w") as log:
        log.write(f"Steiner Tree Comparison  |  {timestamp}\n")
        log.write(f"node_counts: {NODE_COUNT_LIST}  terminals: {TERMINAL_COUNT}  max_weight: {MAX_WEIGHT}\n")
        log.write(f"OJ params: constraint_weight={CONSTRAINT_WEIGHT}  batch={OJ_BATCH_SIZE}  max_reads={OJ_MAX_READS}\n")
        log.write(f"OJ configs: {config_desc}\n")
        log.write(f"Geometric params: connectivity={GEO_CONNECTIVITY}  k_values={GEO_K_LIST}\n")
        log.write(f"Erdos-Renyi params: edge_probabilities={ER_EDGE_PROB_LIST}\n")
        log.write(f"Seeds: {SEED_START}-{SEED_END}\n")
        log.write("=" * 70 + "\n\n")

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

                    g = instances[key]
                    log.write(f"Seed {seed} | n={node_count} | Geometric ({GEO_CONNECTIVITY}, k={geo_k})\n")
                    log.write(f"  |V|={g['num_nodes']}  |E|={g['num_edges']}  terminals={g['terminals']}\n")
                    log.write(f"  Gurobi ILP        | cost: {g['ilp_cost']:<10} | time: {g['ilp_time']}s\n")
                    log.write(f"  Gurobi Binary ILP | cost: {g['bin_cost']:<10} | time: {g['bin_time']}s\n")

                    for label, version, num_sweeps, trotter in OJ_CONFIGS:
                        _run_oj(problem, g["ilp_cost"], label, version, num_sweeps, trotter, log)

                    log.write("\n")
                    log.flush()

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

                    g = instances[key]
                    log.write(f"Seed {seed} | n={node_count} | Erdos-Renyi (p={er_p})\n")
                    log.write(f"  |V|={g['num_nodes']}  |E|={g['num_edges']}  terminals={g['terminals']}\n")
                    log.write(f"  Gurobi ILP        | cost: {g['ilp_cost']:<10} | time: {g['ilp_time']}s\n")
                    log.write(f"  Gurobi Binary ILP | cost: {g['bin_cost']:<10} | time: {g['bin_time']}s\n")

                    for label, version, num_sweeps, trotter in OJ_CONFIGS:
                        _run_oj(problem, g["ilp_cost"], label, version, num_sweeps, trotter, log)

                    log.write("\n")
                    log.flush()

            log.write("-" * 70 + "\n")
            log.flush()

        log.write("\nDone.\n")

    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
