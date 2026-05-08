"""
New models benchmark - Segment 2: benchmark the hybrid and alex QUBO formulations
against the Gurobi optima produced by Segment 1.

For each (n, k, p, seed) instance both models are run with
NUM_TRIALS x NUM_READS_PER_TRIAL reads (default 10 x 100 = 1000 total).
Every individual read is decoded and checked so we can report:

    prob_optimum        fraction of instances where optimum found within budget
    mean_first_hit      mean reads-to-first-hit (solved instances only)
    mean_approx_ratio   mean(best_cost / opt_cost) across all instances
    feasible_read_rate  mean fraction of individual reads that are feasible
    mean_tts            mean time-to-solution (wall-clock seconds)
                        — time to first hit if solved, total elapsed otherwise

Results are saved per instance and aggregated per (p, n, k) combo, with separate
sub-dicts for "hybrid" and "alex" so the two models can be compared directly.
"""
import glob
import json
import os
import time
from collections import defaultdict
from datetime import datetime

from SteinerTreeProblemQUBO.sparsity_problem_generator import (
    generate_sparsity_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.oj_solver_hybrid import (
    solve_with_sqa as solve_hybrid,
)
from SteinerTreeProblemQUBO.MyFormulization.oj_solver_alex import (
    solve_with_sqa as solve_alex,
)

# ── Parameters ──────────────────────────────────────────────────────
# Must match Segment 1.
NODE_COUNT_LIST = [4, 6, 8, 10, 12]
TERMINAL_COUNT_LIST = [2, 3, 4, 5, 6]
EDGE_PROBABILITY_LIST = [0.1, 0.3, 0.6]
NUM_INSTANCES_PER_COMBO = 10
MAX_WEIGHT = 100

CONSTRAINT_WEIGHT = 5 * MAX_WEIGHT
NUM_TRIALS = 10
NUM_READS_PER_TRIAL = 100       # 10 x 100 = 1000 total reads per instance
STOP_ON_FIRST_HIT = True        # skip remaining trials once optimum is matched

SQA_NUM_SWEEPS = 4000
SQA_TROTTER = 16
# ────────────────────────────────────────────────────────────────────

MODELS = {
    "hybrid": solve_hybrid,
    "alex": solve_alex,
}


def _make_key(p, node_count, terminal_count, seed):
    return f"sparsity|p={p}|n={node_count}|k={terminal_count}|seed={seed}"


def _load_gurobi_results():
    results_dir = os.path.join(
        os.path.dirname(__file__), "logs", "gurobi_new_models_results"
    )
    files = sorted(
        glob.glob(os.path.join(results_dir, "new_models_gurobi_*.json"))
    )
    if not files:
        raise FileNotFoundError(
            f"No new_models_gurobi_*.json found in {results_dir}. "
            "Run new_models_benchmark_segment1.py first."
        )
    latest = files[-1]
    print(f"Loading Gurobi results from {latest}")
    with open(latest) as f:
        return json.load(f), latest


def _is_feasible(problem, sample):
    """Return True if the directed-arc sample encodes a valid Steiner tree.

    An undirected edge {u,v} is selected when any direction variable
    e::u::v or e::v::u equals 1.  The selected edge set must be connected,
    acyclic, and span every terminal.
    """
    adj = defaultdict(set)
    for a, b, _ in problem.edges:
        if sample.get(f"e::{a}::{b}", 0) == 1 or sample.get(f"e::{b}::{a}", 0) == 1:
            adj[a].add(b)
            adj[b].add(a)

    nodes_in_tree = set(adj.keys())

    if not nodes_in_tree:
        return False

    if not set(problem.terminals).issubset(nodes_in_tree):
        return False

    root = problem.terminals[0]
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)

    if visited != nodes_in_tree:
        return False

    edge_count = sum(len(nbrs) for nbrs in adj.values()) // 2
    return edge_count == len(nodes_in_tree) - 1


def _run_model(problem, opt_cost, solve_fn):
    """Run one model on one instance and return all five metrics."""
    best_cost = float("inf")
    first_hit_reads = None
    time_to_solution = None
    feasible_count = 0
    cumulative_reads = 0
    t_start = time.time()

    for _trial in range(NUM_TRIALS):
        r = solve_fn(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            num_reads=NUM_READS_PER_TRIAL,
            show_stats=False,
            show_progress=False,
            num_sweeps=SQA_NUM_SWEEPS,
            trotter=SQA_TROTTER,
        )

        offset = r["offset"]

        for sample, energy, _ in r["response"].data():
            cumulative_reads += 1
            cost = float(energy) + offset

            if cost < best_cost:
                best_cost = cost

            if _is_feasible(problem, sample):
                feasible_count += 1

            if (
                first_hit_reads is None
                and opt_cost is not None
                and best_cost <= opt_cost + 1e-6
            ):
                first_hit_reads = cumulative_reads
                time_to_solution = time.time() - t_start

        if first_hit_reads is not None and STOP_ON_FIRST_HIT:
            break

    elapsed = time.time() - t_start

    approx_ratio = None
    if opt_cost is not None and opt_cost > 1e-9 and best_cost < float("inf"):
        approx_ratio = best_cost / opt_cost

    return {
        "solved": first_hit_reads is not None,
        "first_hit_reads": first_hit_reads,
        "best_cost": best_cost if best_cost < float("inf") else None,
        "approx_ratio": approx_ratio,
        "feasible_reads": feasible_count,
        "feasible_read_rate": feasible_count / cumulative_reads if cumulative_reads else 0.0,
        "time_to_solution": time_to_solution if time_to_solution is not None else elapsed,
        "total_reads": cumulative_reads,
        "total_time": round(elapsed, 4),
    }


def _aggregate(runs):
    """Aggregate per-instance run dicts into a summary dict for one (p,n,k) cell."""
    n = len(runs)
    if n == 0:
        return {}

    solved = [r for r in runs if r["solved"]]
    first_hits = [r["first_hit_reads"] for r in solved]
    approx_ratios = [r["approx_ratio"] for r in runs if r["approx_ratio"] is not None]
    feasible_rates = [r["feasible_read_rate"] for r in runs]
    tts = [r["time_to_solution"] for r in runs]

    return {
        "num_instances": n,
        "num_solved": len(solved),
        "success_rate": len(solved) / n,
        "mean_first_hit_reads": sum(first_hits) / len(first_hits) if first_hits else None,
        "mean_approx_ratio": sum(approx_ratios) / len(approx_ratios) if approx_ratios else None,
        "mean_feasible_read_rate": sum(feasible_rates) / len(feasible_rates),
        "mean_time_to_solution": sum(tts) / len(tts),
    }


def main():
    gurobi_data, gurobi_path = _load_gurobi_results()
    gurobi_instances = gurobi_data["instances"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)
    json_path = os.path.join(log_dir, f"new_models_qubo_{timestamp}.json")
    txt_path = os.path.join(log_dir, f"new_models_qubo_{timestamp}.txt")

    results = {
        "_meta": {
            "timestamp": timestamp,
            "gurobi_source": os.path.basename(gurobi_path),
            "node_count_list": NODE_COUNT_LIST,
            "terminal_count_list": TERMINAL_COUNT_LIST,
            "edge_probability_list": EDGE_PROBABILITY_LIST,
            "num_instances_per_combo": NUM_INSTANCES_PER_COMBO,
            "max_weight": MAX_WEIGHT,
            "models": list(MODELS.keys()),
            "constraint_weight": CONSTRAINT_WEIGHT,
            "num_trials": NUM_TRIALS,
            "num_reads_per_trial": NUM_READS_PER_TRIAL,
            "total_reads_budget": NUM_TRIALS * NUM_READS_PER_TRIAL,
            "stop_on_first_hit": STOP_ON_FIRST_HIT,
            "sqa_num_sweeps": SQA_NUM_SWEEPS,
            "sqa_trotter": SQA_TROTTER,
        },
        "instances": {},
        "summary": {},
    }

    total = sum(
        NUM_INSTANCES_PER_COMBO
        for p in EDGE_PROBABILITY_LIST
        for n in NODE_COUNT_LIST
        for k in TERMINAL_COUNT_LIST
        if k <= n
    )
    done = 0

    with open(txt_path, "w") as log:
        log.write(f"New models QUBO benchmark  |  {timestamp}\n")
        log.write(f"Gurobi source: {gurobi_path}\n")
        log.write(
            f"node_counts={NODE_COUNT_LIST}  terminals={TERMINAL_COUNT_LIST}  "
            f"edge_probs={EDGE_PROBABILITY_LIST}\n"
        )
        log.write(
            f"instances/combo={NUM_INSTANCES_PER_COMBO}  "
            f"trials={NUM_TRIALS} x {NUM_READS_PER_TRIAL} reads  "
            f"constraint_weight={CONSTRAINT_WEIGHT}  "
            f"sweeps={SQA_NUM_SWEEPS}  trotter={SQA_TROTTER}\n"
        )
        log.write("=" * 78 + "\n\n")

        for p in EDGE_PROBABILITY_LIST:
            for n in NODE_COUNT_LIST:
                for k in TERMINAL_COUNT_LIST:
                    if k > n:
                        continue

                    combo_label = f"p={p} n={n} k={k}"
                    log.write(f"--- {combo_label} ---\n")

                    # per-model lists of per-instance run dicts
                    model_runs = {name: [] for name in MODELS}

                    for seed in range(NUM_INSTANCES_PER_COMBO):
                        done += 1
                        key = _make_key(p, n, k, seed)
                        print(
                            f"[{done}/{total}] {key}", flush=True
                        )

                        if key not in gurobi_instances:
                            log.write(
                                f"  seed={seed}: MISSING from Gurobi results, skipping\n"
                            )
                            log.flush()
                            continue

                        problem = generate_sparsity_steiner_tree(
                            node_count=n,
                            terminal_count=k,
                            extra_edge_probability=p,
                            weight_range=(1, MAX_WEIGHT),
                            seed=seed,
                        )
                        g = gurobi_instances[key]
                        opt = g["ilp_cost"]

                        instance_entry = {
                            "edge_probability": p,
                            "node_count": n,
                            "terminal_count": k,
                            "seed": seed,
                            "num_nodes": g["num_nodes"],
                            "num_edges": g["num_edges"],
                            "ilp_cost": opt,
                            "models": {},
                        }

                        log.write(
                            f"  seed={seed:<2d}  "
                            f"|V|={g['num_nodes']:<3d}  "
                            f"|E|={g['num_edges']:<3d}  "
                            f"opt={opt}\n"
                        )

                        for model_name, solve_fn in MODELS.items():
                            run = _run_model(problem, opt, solve_fn)
                            model_runs[model_name].append(run)
                            instance_entry["models"][model_name] = run

                            hit_str = (
                                f"hit @ {run['first_hit_reads']} reads"
                                if run["solved"]
                                else f"NOT solved ({run['total_reads']} reads)"
                            )
                            log.write(
                                f"    {model_name:6s}: best={run['best_cost']}  "
                                f"ratio={run['approx_ratio']:.4f}  "
                                f"{hit_str}  "
                                f"feasible {run['feasible_reads']}/{run['total_reads']}  "
                                f"tts={run['time_to_solution']:.2f}s\n"
                            )
                            log.flush()

                        results["instances"][key] = instance_entry

                        with open(json_path, "w") as jf:
                            json.dump(results, jf, indent=2)

                    # Aggregate per (p, n, k)
                    combo_summary = {
                        "edge_probability": p,
                        "node_count": n,
                        "terminal_count": k,
                    }
                    for model_name in MODELS:
                        combo_summary[model_name] = _aggregate(model_runs[model_name])

                    results["summary"][f"p={p}|n={n}|k={k}"] = combo_summary

                    log.write(f"  SUMMARY {combo_label}:\n")
                    for model_name in MODELS:
                        s = combo_summary[model_name]
                        log.write(
                            f"    {model_name:6s}: "
                            f"solved={s.get('num_solved', 0)}/{s.get('num_instances', 0)}  "
                            f"success_rate={s.get('success_rate', 0):.2f}  "
                            f"mean_first_hit={s.get('mean_first_hit_reads')}  "
                            f"mean_ratio={s.get('mean_approx_ratio'):.4f}  "
                            f"mean_feasible_rate={s.get('mean_feasible_read_rate'):.3f}  "
                            f"mean_tts={s.get('mean_time_to_solution'):.2f}s\n"
                        )
                    log.write("\n")
                    log.flush()

                    with open(json_path, "w") as jf:
                        json.dump(results, jf, indent=2)

        log.write("\nDone.\n")

    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)

    print(f"\nJSON results saved to {json_path}")
    print(f"Text log saved to {txt_path}")


if __name__ == "__main__":
    main()
