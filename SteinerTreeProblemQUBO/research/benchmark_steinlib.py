"""Fowler vs chordal ordering on SteinLib instances (test set B, J. E. Beasley).

Run from repository root:
    venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_steinlib b01 b02 ...

Instances are read from research/steinlib/B/*.stp. Reference optima are
SteinLib's published (proven) values; the project's Gurobi flow ILP was too
slow on these instances. Any sampled tree cheaper than the optimum raises.
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace

from .benchmark_general import MODELS, run_sampler
from .benchmark_repairs import decode
from .steiner_to_bqm_chordal import chordal_completion

here = Path(__file__).parent
# Published optimal values for SteinLib test set B.
KNOWN_OPT = dict(b01=82, b02=83, b03=138, b04=59, b05=61, b06=122, b07=111, b08=104, b09=220,
                 b10=86, b11=88, b12=174, b13=165, b14=235, b15=318, b16=127, b17=131, b18=218)


def read_stp(name):
    edges, terminals, nodes = [], [], 0
    for line in (here / "steinlib" / "B" / f"{name}.stp").read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "Nodes":
            nodes = int(parts[1])
        elif parts[0] == "E":
            edges.append((f"v{parts[1]}", f"v{parts[2]}", int(parts[3])))
        elif parts[0] == "T":
            terminals.append(f"v{parts[1]}")
    return SimpleNamespace(nodes=[f"v{i}" for i in range(1, nodes + 1)], edges=edges, terminals=terminals)


def pruned_mst_upper(problem):
    import networkx as nx
    g = nx.Graph()
    g.add_weighted_edges_from(problem.edges)
    tree = nx.minimum_spanning_tree(g)
    terminals = set(problem.terminals)
    while True:
        leaves = [v for v in tree if tree.degree(v) <= 1 and v not in terminals]
        if not leaves:
            return tree.size(weight="weight")
        tree.remove_nodes_from(leaves)


def run_job(job):
    name, model_name, solver, reads, sweeps = job
    problem = read_stp(name)
    optimum = float(KNOWN_OPT[name])
    upper = pruned_mst_upper(problem)
    P = float(math.ceil(1.05 * upper))
    bqm = MODELS[model_name](problem, P)
    import time
    started = time.perf_counter()
    response = run_sampler(solver, bqm, reads, sweeps, 7919 + int(name[1:]))
    elapsed = time.perf_counter() - started
    feasible = optimal = 0
    best = None
    costs = []
    for row in response.data(fields=["sample", "num_occurrences"], sorted_by=None):
        valid, cost = decode(problem, row.sample)
        if valid:
            feasible += int(row.num_occurrences)
            optimal += int(row.num_occurrences) * int(abs(cost - optimum) < 1e-6)
            if cost < optimum - 1e-6:
                raise RuntimeError(f"{name}: sampled tree {cost} below published optimum {optimum}")
            best = cost if best is None else min(best, cost)
            costs.extend([cost] * int(row.num_occurrences))
    h, _, width = chordal_completion(problem)
    return dict(instance=name, model=model_name, solver=solver, n=len(problem.nodes), edges=len(problem.edges),
                terminals=len(problem.terminals), optimum=optimum,
                feasible_upper=upper, penalty=P, chordal_edges=h.number_of_edges(),
                complete_edges=(len(problem.nodes) - 1) * (len(problem.nodes) - 2) // 2, elimination_width=width,
                variables=bqm.num_variables, interactions=bqm.num_interactions, reads=reads, sweeps=sweeps,
                feasible_reads=feasible, optimal_reads=optimal, best_feasible_cost=best,
                median_feasible_cost=sorted(costs)[len(costs) // 2] if costs else None, seconds=elapsed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("instances", nargs="+")
    parser.add_argument("--solvers", nargs="+", default=["neal", "openjij_sa"])
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="SteinerTreeProblemQUBO/research/steinlib_results.json")
    args = parser.parse_args()
    dest = Path(args.output)
    output = json.loads(dest.read_text()) if dest.exists() else dict(records=[])
    done = {(r["instance"], r["model"], r["solver"]) for r in output["records"]}
    jobs = [(i, m, s, args.reads, args.sweeps) for i in args.instances for m in MODELS for s in args.solvers
            if (i, m, s) not in done]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for k, record in enumerate(pool.imap_unordered(run_job, jobs), 1):
            output["records"].append(record)
            dest.write_text(json.dumps(output, indent=1))
            print(f"[{k}/{len(jobs)}] {record['instance']} {record['model']} {record['solver']} "
                  f"feas={record['feasible_reads']} opt={record['optimal_reads']} best={record['best_feasible_cost']} "
                  f"(opt {record['optimum']:.0f}) {record['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
