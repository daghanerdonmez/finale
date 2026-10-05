"""General benchmark: Fowler vs chordal ordering across graph families and solvers.

Run from repository root:
    venv/bin/python -m SteinerTreeProblemQUBO.research.benchmark_general

Instances come from the project's own generators (integer weights), so every
QUBO/Ising coefficient is exact and Neal's automatic schedule is not degraded
by floating-point residue. Exact optima come from subset/MST enumeration.
Models form a 2x2 ablation: {full, chordal} ordering x {Fowler n, local} parents.
Solvers: Neal SA (auto schedule), OpenJij SA (auto schedule), and OpenJij SQA
on the Ising model normalised to max |bias| = 1, as hardware autoscaling does.
OpenJij reads are run one per call with distinct seeds (see diagnose_openjij_seed).
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import time

import dimod
import neal
import numpy as np

from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_erdos_renyi_steiner_tree,
    generate_geometric_steiner_tree,
    generate_grid_steiner_tree,
)
from SteinerTreeProblemQUBO.sparsity_problem_generator import generate_sparsity_steiner_tree
from SteinerTreeProblemQUBO.AlexFowler.steiner_to_bqm_alex import steiner_to_bqm_ordering
from .benchmark_repairs import decode, reference
from .steiner_to_bqm_chordal import chordal_completion, steiner_to_bqm_chordal

FAMILIES = ("sparse", "er30", "er60", "geo_knn4", "grid")
GRID_SHAPES = {8: (2, 4), 12: (3, 4), 16: (4, 4), 20: (4, 5), 24: (4, 6), 32: (4, 8), 40: (5, 8)}
MODELS = {
    "fowler": steiner_to_bqm_ordering,
    "full_order_local": lambda p, P: steiner_to_bqm_chordal(p, P, full=True, parent_encoding="local"),
    "chordal_fowler": lambda p, P: steiner_to_bqm_chordal(p, P, parent_encoding="fowler"),
    "chordal_local": lambda p, P: steiner_to_bqm_chordal(p, P, parent_encoding="local"),
}


def make_general(family, n, fraction, seed):
    k = max(2, round(fraction * n))
    s = seed + 1000 * n + int(1000 * fraction) + 100000 * FAMILIES.index(family)
    if family == "sparse":
        return generate_sparsity_steiner_tree(n, k, 0.1, seed=s)
    if family == "er30":
        return generate_erdos_renyi_steiner_tree(n, k, 0.3, seed=s)
    if family == "er60":
        return generate_erdos_renyi_steiner_tree(n, k, 0.6, seed=s)
    if family == "geo_knn4":
        return generate_geometric_steiner_tree(n, k, connectivity="knn", k=4, seed=s)
    rows, cols = GRID_SHAPES[n]
    return generate_grid_steiner_tree(rows, cols, k, seed=s)


def run_sampler(solver, bqm, reads, sweeps, seed):
    if solver == "neal":
        return neal.SimulatedAnnealingSampler().sample(bqm, num_reads=reads, num_sweeps=sweeps, seed=seed)
    import openjij
    labels = list(bqm.variables)
    model = bqm.relabel_variables({v: i for i, v in enumerate(labels)}, inplace=False)
    if solver == "openjij_sqa":
        model = model.change_vartype(dimod.SPIN, inplace=False)
        model.normalize()
        sampler, extra = openjij.SQASampler(), {"trotter": 4}
    else:
        sampler, extra = openjij.SASampler(), {}
    batches = [sampler.sample(model, num_reads=1, num_sweeps=sweeps, seed=seed + i * 104729, **extra)
               for i in range(reads)]
    response = dimod.concatenate(batches).change_vartype(dimod.BINARY, inplace=False)
    response.relabel_variables(dict(enumerate(labels)))
    return response


def run_instance(job):
    family, n, fraction, seed, solvers, reads, sweeps = job
    problem = make_general(family, n, fraction, seed)
    optimum, upper = reference(problem, integer_costs=True)
    P = float(math.ceil(1.05 * upper))
    h, _, width = chordal_completion(problem)
    instance_id = f"{family}_n{n}_t{int(100 * fraction)}_s{seed}"
    instance = dict(id=instance_id, family=family, n=n, terminal_fraction=fraction, seed=seed,
                    edges=len(problem.edges), terminals=len(problem.terminals), optimum=optimum,
                    feasible_upper=upper, penalty=P, chordal_edges=h.number_of_edges(),
                    complete_edges=(n - 1) * (n - 2) // 2, elimination_width=width)
    records = []
    for model_name, builder in MODELS.items():
        bqm = builder(problem, P)
        for solver in solvers:
            started = time.perf_counter()
            response = run_sampler(solver, bqm, reads, sweeps, 7919 + seed + 31 * n)
            elapsed = time.perf_counter() - started
            feasible = optimal = 0
            best = None
            for row in response.data(fields=["sample", "num_occurrences"], sorted_by=None):
                valid, cost = decode(problem, row.sample)
                num = int(row.num_occurrences)
                if valid:
                    feasible += num
                    best = cost if best is None else min(best, cost)
                    optimal += num * int(abs(cost - optimum) < 1e-6)
            records.append(dict(instance=instance_id, family=family, n=n, terminal_fraction=fraction,
                model=model_name, solver=solver, variables=bqm.num_variables,
                interactions=bqm.num_interactions, reads=int(sum(response.record.num_occurrences)),
                feasible_reads=feasible, optimal_reads=optimal, best_feasible_cost=best,
                optimum=optimum, seconds=elapsed,
                beta_range=[float(b) for b in neal.default_beta_range(bqm)] if solver == "neal" else None))
    return instance, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="SteinerTreeProblemQUBO/research/general_results.json")
    parser.add_argument("--families", nargs="+", default=list(FAMILIES))
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    parser.add_argument("--terminal-fractions", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--solvers", nargs="+", default=["neal", "openjij_sa", "openjij_sqa"])
    parser.add_argument("--reads", type=int, default=100)
    parser.add_argument("--sweeps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    args = parser.parse_args()
    jobs = [(f, n, t, s, args.solvers, args.reads, args.sweeps)
            for f in args.families for n in args.sizes for t in args.terminal_fractions for s in range(args.seeds)]
    output = dict(settings=vars(args), instances=[], records=[])
    dest = Path(args.output)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (instance, records) in enumerate(pool.imap_unordered(run_instance, jobs), 1):
            output["instances"].append(instance)
            output["records"].extend(records)
            dest.write_text(json.dumps(output, indent=1))
            print(f"[{i}/{len(jobs)}] {instance['id']}", flush=True)


if __name__ == "__main__":
    main()
