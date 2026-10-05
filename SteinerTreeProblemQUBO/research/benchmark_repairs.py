"""Reproducible pilot, not a replacement for a broad performance study.

Run from repository root: venv/bin/python -m
SteinerTreeProblemQUBO.research.benchmark_repairs --output ...
Uses fixed complete read budgets, actual edge cost, and separate directed
feasibility / zero-QUBO-penalty metrics. Exact STP reference by subset/MST.
"""
import argparse
import math
from itertools import combinations
import json
from pathlib import Path
import random
import time
from types import SimpleNamespace

import dimod
import neal
import networkx as nx
import numpy as np

from SteinerTreeProblemQUBO.MyFormulation import steiner_to_bqm_hybrid as old
from SteinerTreeProblemQUBO.AlexFowler.steiner_to_bqm_alex import steiner_to_bqm_ordering
from .steiner_to_bqm_chordal import steiner_to_bqm_chordal, chordal_completion


def make_instance(family, n, seed, integer_costs=False):
    rng = random.Random(seed + 1000 * n)
    if family == "ladder":
        g = nx.convert_node_labels_to_integers(nx.ladder_graph(n // 2))
    else:
        for attempt in range(100):
            g = nx.random_regular_graph(3, n, seed=seed + attempt * 100)
            if nx.is_connected(g):
                break
    # Integer costs are the same instance scaled by 20. They keep every QUBO and
    # Ising coefficient exactly representable, so cancelled terms are exactly 0.
    scale = 1 if integer_costs else 20
    edges = [(str(u), str(v), rng.randint(1, 20) / scale) for u, v in sorted(g.edges)]
    terminals = ["0"] + [str(v) for v in rng.sample(list(range(1, n)), max(2, n // 3) - 1)]
    return SimpleNamespace(nodes=[str(v) for v in range(n)], edges=edges, terminals=terminals)


def reference(problem, integer_costs=False):
    g = nx.Graph()
    g.add_nodes_from(problem.nodes)
    g.add_weighted_edges_from(problem.edges)
    terminals = set(problem.terminals)
    optional = [v for v in problem.nodes if v not in terminals]
    upper_tree = nx.minimum_spanning_tree(g)
    # Prune all nonterminal leaves: feasible bound independent of exact optimum.
    changed = True
    while changed:
        leaves = [v for v in upper_tree if upper_tree.degree(v) <= 1 and v not in terminals]
        changed = bool(leaves)
        upper_tree.remove_nodes_from(leaves)
    upper = upper_tree.size(weight="weight")
    if len(optional) > 16:
        from SteinerTreeProblemQUBO.MyFormulation.gurobi_solver import solve_ilp
        result = solve_ilp(problem)
        if result["status"] != "OPTIMAL":
            raise RuntimeError("Reference ILP did not certify optimality")
        cost = float(result["cost"])
        return (float(round(cost)) if integer_costs else cost), upper
    best = upper
    for mask in range(1 << len(optional)):
        nodes = terminals | {v for i, v in enumerate(optional) if mask & (1 << i)}
        sub = g.subgraph(nodes)
        if nx.is_connected(sub):
            best = min(best, nx.minimum_spanning_tree(sub).size(weight="weight"))
    return best, upper


def big_m(problem, P):
    # Build the current report's formulation with explicit uniform weight,
    # without mutating the user's in-progress module-level settings.
    ctx = old._HybridContext(problem)
    linear, quadratic = {}, {}
    old._initialize_variables(ctx, linear)
    old.add_H_cost(problem, ctx, linear)
    offset = old.add_H_terminal_parent(ctx, linear, quadratic, P)
    offset += old.add_H_nonterminal_parent(ctx, linear, quadratic, P)
    old.add_H_no_fake_root(ctx, linear, quadratic, P)
    offset += old.add_H_root_depth(ctx, linear, quadratic, P)
    offset += old.add_H_depth(ctx, linear, quadratic, P)
    return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)


def decode(problem, sample):
    arcs = []
    cost = 0.0
    for u, v, w in problem.edges:
        for a, b in ((u, v), (v, u)):
            if sample.get(("e", a, b), 0):
                arcs.append((a, b))
                cost += w
    g = nx.DiGraph()
    g.add_node(problem.terminals[0])
    g.add_edges_from(arcs)
    root = problem.terminals[0]
    valid = (set(problem.terminals) <= set(g) and nx.is_directed_acyclic_graph(g)
             and g.in_degree(root) == 0
             and all(g.in_degree(v) == 1 for v in g if v != root)
             and len(nx.descendants(g, root)) == len(g) - 1)
    return valid, cost


def stats(bqm, P):
    spin = bqm.change_vartype(dimod.SPIN, inplace=False)
    return dict(variables=bqm.num_variables,
                interactions=int(sum(bias != 0 for bias in bqm.quadratic.values())),
                max_abs_qubo_linear=max(map(abs, bqm.linear.values()), default=0),
                max_abs_qubo_quadratic=max(map(abs, bqm.quadratic.values()), default=0),
                max_abs_ising_field=max(map(abs, spin.linear.values()), default=0),
                max_abs_ising_coupler=max(map(abs, spin.quadratic.values()), default=0),
                penalty=P)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="SteinerTreeProblemQUBO/research/pilot_results.json")
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--sampler", choices=["neal", "openjij_sa", "openjij_sqa"], default="neal")
    parser.add_argument("--penalty-factor", type=float, default=1.05)
    parser.add_argument("--integer-costs", action="store_true",
                        help="scale costs by 20 and round P up to an integer")
    parser.add_argument("--models", nargs="+", default=["big_m", "fowler", "full_order_p", "chordal", "circuit", "domain_wall"])
    args = parser.parse_args()
    from .steiner_to_bqm_circuit import steiner_to_bqm_circuit
    from .steiner_to_bqm_domain_wall import steiner_to_bqm_domain_wall
    from .steiner_to_bqm_binary_order import steiner_to_bqm_binary_order
    builders = {"big_m": big_m, "fowler": steiner_to_bqm_ordering,
                "full_order_p": lambda p, P: steiner_to_bqm_chordal(p, P, full=True),
                "chordal": steiner_to_bqm_chordal,
                "chordal_fowler": lambda p, P: steiner_to_bqm_chordal(p, P, parent_encoding="fowler"),
                "chordal_local": lambda p, P: steiner_to_bqm_chordal(p, P, parent_encoding="local"),
                "full_order_local": lambda p, P: steiner_to_bqm_chordal(p, P, full=True, parent_encoding="local"),
                "binary_order": steiner_to_bqm_binary_order,
                "circuit": lambda p, P: steiner_to_bqm_circuit(p, penalty_weight=P, parent_encoding="square"),
                "domain_wall": steiner_to_bqm_domain_wall}
    output = dict(settings=vars(args), instances=[], records=[])
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for family in ("ladder", "regular3"):
        for n in args.sizes:
            for seed in range(args.seeds):
                problem = make_instance(family, n, seed, args.integer_costs)
                optimum, upper = reference(problem, args.integer_costs)
                P = max(upper * args.penalty_factor, 1e-6)
                if args.integer_costs:
                    P = float(math.ceil(P))
                instance_id = f"{family}_{n}_{seed}"
                h, _, width = chordal_completion(problem)
                output["instances"].append(dict(id=instance_id, nodes=problem.nodes, edges=problem.edges,
                    terminals=problem.terminals, optimum=optimum, feasible_upper=upper,
                    chordal_edges=h.number_of_edges(), elimination_width=width))
                for name in args.models:
                    bqm = builders[name](problem, P)
                    beta_range = [float(b) for b in neal.default_beta_range(bqm)]
                    started = time.perf_counter()
                    sampler_seed = 4321 + seed + n * 100
                    if args.sampler == "neal":
                        response = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=args.reads,
                            num_sweeps=args.sweeps, seed=sampler_seed)
                    else:
                        import openjij
                        labels = list(bqm.variables)
                        indexed = bqm.relabel_variables({v: i for i, v in enumerate(labels)}, inplace=False)
                        sampler = openjij.SASampler() if args.sampler == "openjij_sa" else openjij.SQASampler()
                        # OpenJij 0.10.17 restarts initialization and the sampler
                        # with the SAME seed for every read in a seeded batch.
                        # Give each read a distinct deterministic seed instead.
                        batches = [sampler.sample(indexed, num_reads=1,
                            num_sweeps=args.sweeps, seed=sampler_seed + i * 104729)
                            for i in range(args.reads)]
                        response = dimod.concatenate(batches)
                        response.relabel_variables(dict(enumerate(labels)))
                    elapsed = time.perf_counter() - started
                    feasible = optimal = zero = 0
                    best = None
                    for row in response.data(fields=["sample", "energy", "num_occurrences"], sorted_by=None):
                        valid, cost = decode(problem, row.sample)
                        num = int(row.num_occurrences)
                        if valid:
                            feasible += num
                            best = cost if best is None else min(best, cost)
                            optimal += num * int(abs(cost - optimum) < 1e-7)
                        zero += num * int(abs(float(row.energy) - cost) < 1e-6)
                    record = dict(instance=instance_id, model=name, **stats(bqm, P),
                        feasible_reads=feasible, optimal_reads=optimal, zero_penalty_reads=zero,
                        best_feasible_cost=best, optimum=optimum, seconds=elapsed, beta_range=beta_range,
                        reads=int(sum(response.record.num_occurrences)),
                        unique_states=int(len(np.unique(response.record.sample, axis=0))),
                        seed_policy="per-read distinct seed" if args.sampler.startswith("openjij") else "seeded batch")
                    output["records"].append(record)
                    dest.write_text(json.dumps(output, indent=2))
                    print(f"{instance_id:16} {name:12} nv={bqm.num_variables:4} "
                          f"feas={feasible:3} opt={optimal:3} zero={zero:3} sec={elapsed:.2f}", flush=True)
    for name in args.models:
        rows = [r for r in output["records"] if r["model"] == name]
        total = sum(r["reads"] for r in rows)
        print(name, dict(instances_solved=sum(r["optimal_reads"] > 0 for r in rows),
            feasible_rate=sum(r["feasible_reads"] for r in rows)/total,
            optimal_rate=sum(r["optimal_reads"] for r in rows)/total,
            zero_penalty_rate=sum(r["zero_penalty_reads"] for r in rows)/total), flush=True)


if __name__ == "__main__":
    main()
