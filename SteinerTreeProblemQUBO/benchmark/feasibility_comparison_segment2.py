"""
Feasibility comparison - Segment 2: run the OJ SQA solver with BQM versions 6
and 7 on every instance produced by Segment 1, stopping at the first optimum
hit within a 10 x 100 read budget.

For every instance and every version we:
    - run up to NUM_TRIALS trials of NUM_READS_PER_TRIAL reads each,
      breaking out the moment the best energy-with-offset matches the Gurobi
      optimum (``STOP_ON_FIRST_HIT``).
    - inspect the best sample (optimum or not) and check whether the edges
      with x_e = 1 form a feasible Steiner tree (connected, acyclic,
      spanning all terminals).
    - if the best sample is NOT feasible we also dump the raw x_e values and
      the directed flow values f_{u,v} (decoded from the z bits) so the user
      can inspect by hand what is going wrong.

Results are keyed per instance per version so downstream tooling can compare
v6 vs v7 directly.
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
from SteinerTreeProblemQUBO.MyFormulization.oj_solver import solve_with_sqa

# ── Parameters ──────────────────────────────────────────────────────
# Must match segment 1 so we regenerate the same instances.
NODE_COUNT_LIST = [4,6,8,10,12]
TERMINAL_COUNT_LIST = [2,3,4,5,6]
EDGE_PROBABILITY_LIST = [0.1, 0.3, 0.6]
NUM_INSTANCES_PER_COMBO = 5
MAX_WEIGHT = 100

# QUBO solver configuration.
BQM_VERSIONS = [8]
CONSTRAINT_WEIGHT = MAX_WEIGHT
NUM_TRIALS = 10                # number of independent trials per instance
NUM_READS_PER_TRIAL = 100      # reads per trial (10 x 100 = 1000 total)
STOP_ON_FIRST_HIT = True

# openjij SQA kwargs — match the "tuned" config used elsewhere.
SQA_NUM_SWEEPS = 4000
SQA_TROTTER = 16
# ────────────────────────────────────────────────────────────────────


def _make_key(p, node_count, terminal_count, seed):
    return f"sparsity|p={p}|n={node_count}|k={terminal_count}|seed={seed}"


def _load_gurobi_results():
    """Load the most recent feasibility_gurobi_*.json written by segment 1."""
    results_dir = os.path.join(
        os.path.dirname(__file__), "logs", "gurobi_feasibility_results"
    )
    files = sorted(
        glob.glob(os.path.join(results_dir, "feasibility_gurobi_*.json"))
    )
    if not files:
        raise FileNotFoundError(
            f"No feasibility_gurobi_*.json found in {results_dir}. "
            "Run feasibility_comparison_segment1.py first."
        )
    latest = files[-1]
    print(f"Loading Gurobi results from {latest}")
    with open(latest) as f:
        return json.load(f), latest


def _decode_sample(problem, sample):
    """Decode the best_sample dict into x_edges, flows and per-edge bits.

    Returns:
        x_edges:   list of (u, v) undirected edges where x_{u,v} = 1
        flows:     dict (u, v) -> integer flow value (decoded from z bits)
        x_values:  dict (u, v) -> 0/1 for every edge in problem.edges
        flow_bits: dict (u, v) -> dict(bit -> 0/1) for every directed edge
    """
    import math

    B = math.ceil(math.log2(len(problem.terminals)))

    x_values = {}
    x_edges = []
    flows = defaultdict(int)
    flow_bits = defaultdict(dict)

    for a, b, _ in problem.edges:
        label = f"x::{a}::{b}"
        val = int(sample.get(label, 0))
        x_values[(a, b)] = val
        if val == 1:
            x_edges.append((a, b))

        for (u, v) in ((a, b), (b, a)):
            bit_map = {}
            total = 0
            for bit in range(B):
                z_label = f"z::{u}::{v}::{bit}"
                bit_val = int(sample.get(z_label, 0))
                bit_map[bit] = bit_val
                if bit_val == 1:
                    total += 2 ** bit
            flows[(u, v)] = total
            flow_bits[(u, v)] = bit_map

    return x_edges, dict(flows), x_values, dict(flow_bits)


def _check_feasibility(problem, x_edges):
    """Check whether the x-edge set is a feasible Steiner tree.

    Feasible iff the selected edges form a single connected acyclic subgraph
    that contains every terminal.

    Returns (is_feasible: bool, reason: str).
    """
    terminals = problem.terminals

    if not x_edges:
        if len(terminals) <= 1:
            return True, "trivial (<=1 terminal, 0 edges)"
        return False, "no edges selected but >1 terminal"

    adj = defaultdict(set)
    for u, v in x_edges:
        adj[u].add(v)
        adj[v].add(u)

    nodes_in_tree = set(adj.keys())

    missing = [t for t in terminals if t not in nodes_in_tree]
    if missing:
        return False, f"terminals missing from selected edges: {missing}"

    root = terminals[0]
    visited = {root}
    stack = [root]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in visited:
                visited.add(v)
                stack.append(v)

    if visited != nodes_in_tree:
        disconnected = nodes_in_tree - visited
        return False, f"not connected ({len(disconnected)} nodes unreachable from {root})"

    if len(x_edges) != len(nodes_in_tree) - 1:
        return (
            False,
            f"contains cycle (|E|={len(x_edges)}, |V|={len(nodes_in_tree)})",
        )

    return True, "tree spanning all terminals"


def _run_qubo_with_first_hit(problem, optimal_cost, version):
    """Run up to NUM_TRIALS x NUM_READS_PER_TRIAL reads, tracking first-hit.

    Always returns the best sample seen across trials so feasibility can be
    inspected regardless of optimality.
    """
    best_cost = float("inf")
    best_sample = None
    trial_costs = []
    first_hit_reads = None
    t0 = time.time()

    for trial in range(NUM_TRIALS):
        r_oj = solve_with_sqa(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            version=version,
            num_reads=NUM_READS_PER_TRIAL,
            show_stats=False,
            show_progress=False,
            num_sweeps=SQA_NUM_SWEEPS,
            trotter=SQA_TROTTER,
        )
        cost = float(r_oj["best_energy_with_offset"])
        if cost < best_cost:
            best_cost = cost
            best_sample = r_oj["best_sample"]
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
    total_reads = (trial + 1) * NUM_READS_PER_TRIAL  # reads actually performed

    return {
        "first_hit_reads": first_hit_reads,
        "solved": first_hit_reads is not None,
        "best_cost": best_cost,
        "best_sample": best_sample,
        "trial_costs": trial_costs,
        "total_reads": total_reads,
        "time_seconds": round(elapsed, 4),
    }


def main():
    gurobi_data, gurobi_path = _load_gurobi_results()
    gurobi_instances = gurobi_data["instances"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(__file__), "logs")
    os.makedirs(log_dir, exist_ok=True)

    results = {
        "_meta": {
            "timestamp": timestamp,
            "gurobi_source": os.path.basename(gurobi_path),
            "node_count_list": NODE_COUNT_LIST,
            "terminal_count_list": TERMINAL_COUNT_LIST,
            "edge_probability_list": EDGE_PROBABILITY_LIST,
            "num_instances_per_combo": NUM_INSTANCES_PER_COMBO,
            "max_weight": MAX_WEIGHT,
            "bqm_versions": BQM_VERSIONS,
            "constraint_weight": CONSTRAINT_WEIGHT,
            "num_trials": NUM_TRIALS,
            "num_reads_per_trial": NUM_READS_PER_TRIAL,
            "total_reads_per_instance": NUM_TRIALS * NUM_READS_PER_TRIAL,
            "stop_on_first_hit": STOP_ON_FIRST_HIT,
            "sqa_num_sweeps": SQA_NUM_SWEEPS,
            "sqa_trotter": SQA_TROTTER,
        },
        "instances": {},
        "summary": {},
    }

    json_path = os.path.join(log_dir, f"feasibility_qubo_{timestamp}.json")
    txt_path = os.path.join(log_dir, f"feasibility_qubo_{timestamp}.txt")

    total = 0
    for p in EDGE_PROBABILITY_LIST:
        for n in NODE_COUNT_LIST:
            for k in TERMINAL_COUNT_LIST:
                if k <= n:
                    total += NUM_INSTANCES_PER_COMBO
    done = 0

    with open(txt_path, "w") as log:
        log.write(f"Feasibility comparison QUBO benchmark  |  {timestamp}\n")
        log.write(f"Gurobi source: {gurobi_path}\n")
        log.write(
            f"node_counts: {NODE_COUNT_LIST}  terminals: {TERMINAL_COUNT_LIST}  "
            f"edge_probs: {EDGE_PROBABILITY_LIST}\n"
        )
        log.write(
            f"instances/combo: {NUM_INSTANCES_PER_COMBO}  "
            f"trials: {NUM_TRIALS} x {NUM_READS_PER_TRIAL} reads "
            f"(total budget {NUM_TRIALS * NUM_READS_PER_TRIAL})\n"
        )
        log.write(
            f"versions: {BQM_VERSIONS}  constraint_weight={CONSTRAINT_WEIGHT}  "
            f"sqa_sweeps={SQA_NUM_SWEEPS}  trotter={SQA_TROTTER}\n"
        )
        log.write("=" * 78 + "\n\n")

        for p in EDGE_PROBABILITY_LIST:
            for n in NODE_COUNT_LIST:
                for k in TERMINAL_COUNT_LIST:
                    if k > n:
                        continue

                    combo_label = f"p={p} n={n} k={k}"
                    log.write(f"--- {combo_label} ---\n")
                    combo_stats = {
                        v: {"solved": 0, "feasible": 0, "no_penalty": 0, "run": 0}
                        for v in BQM_VERSIONS
                    }

                    for seed in range(NUM_INSTANCES_PER_COMBO):
                        done += 1
                        key = _make_key(p, n, k, seed)
                        print(f"[{done}/{total}] {key}", flush=True)

                        if key not in gurobi_instances:
                            log.write(
                                f"  seed={seed}: MISSING from Gurobi results, "
                                f"skipping\n"
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
                            "versions": {},
                        }

                        log.write(
                            f"  seed={seed:<2d}  |V|={g['num_nodes']:<3d} "
                            f"|E|={g['num_edges']:<3d}  opt={opt}\n"
                        )

                        for version in BQM_VERSIONS:
                            combo_stats[version]["run"] += 1
                            run = _run_qubo_with_first_hit(
                                problem, opt, version=version
                            )

                            x_edges, flows, x_values, flow_bits = _decode_sample(
                                problem, run["best_sample"] or {}
                            )
                            feasible, reason = _check_feasibility(
                                problem, x_edges
                            )

                            # Raw objective contribution = sum of weights of
                            # edges with x_e = 1.  Anything on top of that
                            # in best_cost is a penalty from violated
                            # constraints (flow / capacity / use / tree / etc).
                            edge_weight_sum = sum(
                                w for (a, b, w) in problem.edges
                                if x_values.get((a, b), 0) == 1
                            )
                            penalty = run["best_cost"] - edge_weight_sum
                            has_penalty = abs(penalty) > 1e-6

                            version_entry = {
                                "first_hit_reads": run["first_hit_reads"],
                                "solved": run["solved"],
                                "best_cost": run["best_cost"],
                                "edge_weight_sum": edge_weight_sum,
                                "penalty": penalty,
                                "has_penalty": has_penalty,
                                "trial_costs": run["trial_costs"],
                                "total_reads": run["total_reads"],
                                "time_seconds": run["time_seconds"],
                                "feasible": feasible,
                                "feasibility_reason": reason,
                                "selected_edges": x_edges,
                            }

                            if not feasible:
                                version_entry["x_values"] = {
                                    f"{u},{v}": val
                                    for (u, v), val in x_values.items()
                                }
                                version_entry["flows"] = {
                                    f"{u}->{v}": f
                                    for (u, v), f in flows.items()
                                }
                                version_entry["flow_bits"] = {
                                    f"{u}->{v}": bits
                                    for (u, v), bits in flow_bits.items()
                                }

                            instance_entry["versions"][f"v{version}"] = \
                                version_entry

                            if run["solved"]:
                                combo_stats[version]["solved"] += 1
                            if feasible:
                                combo_stats[version]["feasible"] += 1
                            if not has_penalty:
                                combo_stats[version]["no_penalty"] += 1

                            hit_str = (
                                f"first hit @ {run['first_hit_reads']} reads"
                                if run["solved"]
                                else f"NOT solved in {run['total_reads']} reads"
                            )
                            feas_str = (
                                "FEASIBLE"
                                if feasible
                                else f"INFEASIBLE ({reason})"
                            )
                            pen_str = (
                                f"NO PENALTY (cost = edge_sum = {edge_weight_sum})"
                                if not has_penalty
                                else f"PENALIZED (edge_sum={edge_weight_sum} "
                                     f"penalty={penalty:+.2f})"
                            )
                            log.write(
                                f"    v{version}: best={run['best_cost']:<10.1f} "
                                f"{hit_str}  {feas_str}  {pen_str}  "
                                f"t={run['time_seconds']:.2f}s\n"
                            )

                            if not feasible:
                                log.write(
                                    f"      x_values (edge -> x_e):\n"
                                )
                                for (u, v), val in x_values.items():
                                    log.write(f"        ({u},{v}) = {val}\n")
                                log.write(f"      flows (directed u->v = f):\n")
                                for (u, v), f_val in flows.items():
                                    if f_val != 0:
                                        log.write(
                                            f"        {u}->{v} = {f_val}\n"
                                        )
                            log.flush()

                        results["instances"][key] = instance_entry

                        with open(json_path, "w") as jf:
                            json.dump(results, jf, indent=2)

                    for version in BQM_VERSIONS:
                        s = combo_stats[version]
                        results["summary"].setdefault(
                            f"p={p}|n={n}|k={k}", {}
                        )[f"v{version}"] = {
                            "num_run": s["run"],
                            "num_solved": s["solved"],
                            "num_feasible": s["feasible"],
                            "num_no_penalty": s["no_penalty"],
                            "solved_rate": (
                                s["solved"] / s["run"] if s["run"] else None
                            ),
                            "feasible_rate": (
                                s["feasible"] / s["run"] if s["run"] else None
                            ),
                            "no_penalty_rate": (
                                s["no_penalty"] / s["run"] if s["run"] else None
                            ),
                        }

                    log.write(f"  SUMMARY {combo_label}:\n")
                    for version in BQM_VERSIONS:
                        s = combo_stats[version]
                        log.write(
                            f"    v{version}: solved {s['solved']}/{s['run']}  "
                            f"feasible {s['feasible']}/{s['run']}  "
                            f"no_penalty {s['no_penalty']}/{s['run']}\n"
                        )
                    log.write("\n")
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
