"""
Heatmap benchmark - Segment 2: run the OJ SQA solver (BQM version 6) on every
instance produced by Segment 1 and log the first-hit statistics.

For each instance we perform ``NUM_TRIALS`` trials of ``NUM_READS_PER_TRIAL``
reads each (default 10 x 100 = 1000 total reads).  After every trial we check
whether the best energy seen so far matches the Gurobi optimum.  We log:

    - first_hit_reads: total reads at which the optimum was first found
                       (a multiple of NUM_READS_PER_TRIAL, or None if never).
    - solved:          bool, whether the optimum was found within the budget.
    - best_cost:       best energy-with-offset reached across all trials.
    - trial_costs:     the best energy-with-offset after each trial
                       (useful for plotting cumulative success curves).

Results are aggregated per (p, n, k) combination so that the downstream
plotting script can colour a (n, k) heatmap by

    success_rate = (# instances solved within budget) / NUM_INSTANCES_PER_COMBO

for each p.
"""
import glob
import json
import os
import time
from datetime import datetime

from SteinerTreeProblemQUBO.sparsity_problem_generator import (
    generate_sparsity_steiner_tree,
)
from SteinerTreeProblemQUBO.MyFormulization.oj_solver import solve_with_sqa

# ── Parameters ──────────────────────────────────────────────────────
# Must match segment 1 so we regenerate the same instances.
NODE_COUNT_LIST = [4, 6, 8, 10, 12]
TERMINAL_COUNT_LIST = [2, 3, 4, 5, 6]
EDGE_PROBABILITY_LIST = [0.1, 0.3, 0.6]
NUM_INSTANCES_PER_COMBO = 10
MAX_WEIGHT = 100

# QUBO solver configuration (BQM version 6).
BQM_VERSION = 6
CONSTRAINT_WEIGHT = 2 * MAX_WEIGHT
NUM_TRIALS = 10                # number of independent trials per instance
NUM_READS_PER_TRIAL = 100      # reads per trial (10 x 100 = 1000 total)
STOP_ON_FIRST_HIT = False       # if True, break out of the trial loop the first
                               # time the optimum is matched (saves cluster time
                               # on easy instances; unsolved instances still use
                               # the full NUM_TRIALS x NUM_READS_PER_TRIAL budget)

# openjij SQA kwargs — match the "tuned" config from compact_comparison_segment2.
SQA_NUM_SWEEPS = 4000
SQA_TROTTER = 16

# Set to a path (string) of a partial heatmap_qubo_*.json to resume that run:
# every instance already recorded under results["instances"] is skipped and
# new instances are appended to the SAME file (text log gets a fresh file).
# Leave as None to start a brand-new run.
RESUME_FROM = None
# ────────────────────────────────────────────────────────────────────


def _make_key(p, node_count, terminal_count, seed):
    return f"sparsity|p={p}|n={node_count}|k={terminal_count}|seed={seed}"


def _load_gurobi_results():
    """Load the most recent heatmap_gurobi_*.json written by segment 1."""
    results_dir = os.path.join(os.path.dirname(__file__), "logs", "gurobi_results")
    files = sorted(glob.glob(os.path.join(results_dir, "heatmap_gurobi_*.json")))
    if not files:
        raise FileNotFoundError(
            f"No heatmap_gurobi_*.json found in {results_dir}. "
            "Run heatmap_benchmark_segment1.py first."
        )
    latest = files[-1]
    print(f"Loading Gurobi results from {latest}")
    with open(latest) as f:
        return json.load(f), latest


def _run_qubo_with_first_hit(problem, optimal_cost):
    """Run NUM_TRIALS x NUM_READS_PER_TRIAL reads, tracking first-hit.

    Returns a dict describing the run.
    """
    best_cost = float("inf")
    trial_costs = []
    first_hit_reads = None
    t0 = time.time()

    for trial in range(NUM_TRIALS):
        r_oj = solve_with_sqa(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            version=BQM_VERSION,
            num_reads=NUM_READS_PER_TRIAL,
            show_stats=False,
            show_progress=False,
            num_sweeps=SQA_NUM_SWEEPS,
            trotter=SQA_TROTTER,
        )
        cost = float(r_oj["best_energy_with_offset"])
        if cost < best_cost:
            best_cost = cost
        trial_costs.append(best_cost)

        cumulative_reads = (trial + 1) * NUM_READS_PER_TRIAL
        if (
            first_hit_reads is None
            and optimal_cost is not None
            and best_cost <= optimal_cost + 1e-6
        ):
            first_hit_reads = cumulative_reads
            if STOP_ON_FIRST_HIT:
                break

    elapsed = time.time() - t0
    total_reads = NUM_TRIALS * NUM_READS_PER_TRIAL

    # Flag possible bugs: BQM energy below the ILP optimum means the
    # penalty landscape admits a cheaper feasible-looking solution than
    # the true optimum (or the optimum was wrong).  Worth surfacing.
    buggy = bool(
        optimal_cost is not None
        and best_cost < optimal_cost - 1e-6
    )

    return {
        "first_hit_reads": first_hit_reads,
        "solved": first_hit_reads is not None,
        "best_cost": best_cost,
        "trial_costs": trial_costs,
        "total_reads": total_reads,
        "time_seconds": round(elapsed, 4),
        "buggy_below_optimal": buggy,
    }


def main():
    gurobi_data, gurobi_path = _load_gurobi_results()
    gurobi_instances = gurobi_data["instances"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    fresh_meta = {
        "timestamp": timestamp,
        "gurobi_source": os.path.basename(gurobi_path),
        "node_count_list": NODE_COUNT_LIST,
        "terminal_count_list": TERMINAL_COUNT_LIST,
        "edge_probability_list": EDGE_PROBABILITY_LIST,
        "num_instances_per_combo": NUM_INSTANCES_PER_COMBO,
        "max_weight": MAX_WEIGHT,
        "bqm_version": BQM_VERSION,
        "constraint_weight": CONSTRAINT_WEIGHT,
        "num_trials": NUM_TRIALS,
        "num_reads_per_trial": NUM_READS_PER_TRIAL,
        "total_reads_per_instance": NUM_TRIALS * NUM_READS_PER_TRIAL,
        "stop_on_first_hit": STOP_ON_FIRST_HIT,
        "sqa_num_sweeps": SQA_NUM_SWEEPS,
        "sqa_trotter": SQA_TROTTER,
    }

    if RESUME_FROM is not None:
        resume_path = (
            RESUME_FROM
            if os.path.isabs(RESUME_FROM)
            else os.path.join(log_dir, RESUME_FROM)
        )
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"RESUME_FROM not found: {resume_path}")
        with open(resume_path) as f:
            results = json.load(f)
        results.setdefault("instances", {})
        results.setdefault("summary", {})
        # Append a resume marker but keep the original meta so downstream
        # tools can identify which instances came from which settings.
        results["_meta"].setdefault("resumes", []).append(
            {"resumed_at": timestamp, "with_settings": fresh_meta}
        )
        json_path = resume_path                                  # write back to same file
        txt_path = os.path.join(log_dir, f"heatmap_qubo_resume_{timestamp}.txt")
        print(
            f"Resuming {resume_path} — "
            f"{len(results['instances'])} instances already complete"
        )
    else:
        results = {"_meta": fresh_meta, "instances": {}, "summary": {}}
        json_path = os.path.join(log_dir, f"heatmap_qubo_{timestamp}.json")
        txt_path = os.path.join(log_dir, f"heatmap_qubo_{timestamp}.txt")

    # Count total instances (only k <= n combos).
    total = 0
    for p in EDGE_PROBABILITY_LIST:
        for n in NODE_COUNT_LIST:
            for k in TERMINAL_COUNT_LIST:
                if k <= n:
                    total += NUM_INSTANCES_PER_COMBO
    done = 0

    with open(txt_path, "w") as log:
        log.write(f"Heatmap QUBO benchmark  |  {timestamp}\n")
        log.write(f"Gurobi source: {gurobi_path}\n")
        log.write(
            f"node_counts: {NODE_COUNT_LIST}  terminals: {TERMINAL_COUNT_LIST}  "
            f"edge_probs: {EDGE_PROBABILITY_LIST}\n"
        )
        log.write(
            f"instances/combo: {NUM_INSTANCES_PER_COMBO}  "
            f"trials: {NUM_TRIALS} x {NUM_READS_PER_TRIAL} reads "
            f"(total {NUM_TRIALS * NUM_READS_PER_TRIAL})\n"
        )
        log.write(
            f"BQM version {BQM_VERSION}  constraint_weight={CONSTRAINT_WEIGHT}  "
            f"sqa_sweeps={SQA_NUM_SWEEPS}  trotter={SQA_TROTTER}\n"
        )
        log.write("=" * 74 + "\n\n")

        for p in EDGE_PROBABILITY_LIST:
            for n in NODE_COUNT_LIST:
                for k in TERMINAL_COUNT_LIST:
                    if k > n:
                        continue

                    first_hits = []   # per-instance first-hit read counts (or None)
                    solved_flags = []
                    combo_label = f"p={p} n={n} k={k}"
                    log.write(f"--- {combo_label} ---\n")

                    for seed in range(NUM_INSTANCES_PER_COMBO):
                        done += 1
                        key = _make_key(p, n, k, seed)
                        print(f"[{done}/{total}] {key}", flush=True)

                        if key not in gurobi_instances:
                            log.write(
                                f"  seed={seed}: MISSING from Gurobi results, "
                                f"skipping\n"
                            )
                            continue

                        # ── Resume: reuse already-completed instance ──
                        existing = results["instances"].get(key)
                        if existing is not None:
                            first_hits.append(existing.get("first_hit_reads"))
                            solved_flags.append(existing.get("solved", False))
                            log.write(
                                f"  seed={seed:<2d}  (resumed, already complete) "
                                f"first_hit={existing.get('first_hit_reads')} "
                                f"solved={existing.get('solved')}\n"
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

                        run = _run_qubo_with_first_hit(problem, opt)

                        results["instances"][key] = {
                            "edge_probability": p,
                            "node_count": n,
                            "terminal_count": k,
                            "seed": seed,
                            "num_nodes": g["num_nodes"],
                            "num_edges": g["num_edges"],
                            "ilp_cost": opt,
                            **run,
                        }

                        first_hits.append(run["first_hit_reads"])
                        solved_flags.append(run["solved"])

                        hit_str = (
                            f"first hit @ {run['first_hit_reads']} reads"
                            if run["solved"]
                            else f"NOT solved in {run['total_reads']} reads"
                        )
                        bug_str = "  [BUGGY<opt]" if run["buggy_below_optimal"] else ""
                        log.write(
                            f"  seed={seed:<2d}  |V|={g['num_nodes']:<3d} "
                            f"|E|={g['num_edges']:<3d}  opt={opt:<8}  "
                            f"best={run['best_cost']:<10.1f}  {hit_str}  "
                            f"t={run['time_seconds']:.2f}s{bug_str}\n"
                        )
                        log.flush()

                        # flush JSON intermittently so partial progress
                        # survives crashes on the cluster.
                        with open(json_path, "w") as jf:
                            json.dump(results, jf, indent=2)

                    solved_count = sum(solved_flags)
                    n_done = len(solved_flags)
                    success_rate = (solved_count / n_done) if n_done else None
                    solved_hits = [h for h in first_hits if h is not None]
                    avg_first_hit = (
                        sum(solved_hits) / len(solved_hits) if solved_hits else None
                    )

                    results["summary"][f"p={p}|n={n}|k={k}"] = {
                        "edge_probability": p,
                        "node_count": n,
                        "terminal_count": k,
                        "num_instances_run": n_done,
                        "num_instances_solved": solved_count,
                        "success_rate": success_rate,
                        "first_hits": first_hits,
                        "avg_first_hit_reads_when_solved": avg_first_hit,
                    }

                    log.write(
                        f"  SUMMARY {combo_label}: solved {solved_count}/{n_done}"
                    )
                    if avg_first_hit is not None:
                        log.write(
                            f"   avg first-hit = {avg_first_hit:.1f} reads (solved only)"
                        )
                    log.write("\n\n")
                    log.flush()

                    with open(json_path, "w") as jf:
                        json.dump(results, jf, indent=2)

        log.write("\nDone.\n")

    with open(json_path, "w") as jf:
        json.dump(results, jf, indent=2)

    print(f"JSON results saved to {json_path}")
    print(f"Text log saved to {txt_path}")


if __name__ == "__main__":
    main()
