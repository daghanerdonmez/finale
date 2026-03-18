import sys

import dimod
import openjij as oj
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.MyFormulization.steiner_to_oj_qubo_daghan import (
    steiner_to_oj_qubo_daghan,
)
from tqdm import tqdm


def solve_with_sqa(
    problem: SteinerTree,
    constraint_weight: float,
    num_reads: int = 1000,
    show_stats: bool = False,
    show_progress: bool = False,
    **sampler_kwargs,
):
    if num_reads < 1:
        raise ValueError("num_reads must be at least 1")

    qubo, offset = steiner_to_oj_qubo_daghan(problem, constraint_weight)

    if show_stats:
        bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset=offset)
        print("Problem converted to QUBO", flush=True)
        print(f"Number of variables {bqm.num_variables}", flush=True)
        print(f"Number of interactions {bqm.num_interactions}", flush=True)

    sampler = oj.SQASampler()
    responses = []
    sampling_runs = range(num_reads)
    if show_progress:
        sampling_runs = tqdm(
            sampling_runs,
            total=num_reads,
            desc="Sampling",
            unit="read",
            file=sys.stdout,
        )

    for _ in sampling_runs:
        responses.append(sampler.sample_qubo(qubo, num_reads=1, **sampler_kwargs))

    response = dimod.concatenate(responses)
    best = response.first

    return {
        "qubo": qubo,
        "offset": offset,
        "response": response,
        "best_sample": best.sample,
        "best_energy_without_offset": best.energy,
        "best_energy_with_offset": best.energy + offset,
    }


if __name__ == "__main__":
    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [
        ("a", "b", 2),
        ("b", "c", 4),
        ("b", "d", 1),
        ("a", "d", 6),
        ("a", "e", 7),
        ("d", "f", 1),
        ("d", "g", 2),
        ("f", "g", 4),
        ("e", "f", 3),
    ]
    terminals = ["a", "c", "f", "g"]

    problem = SteinerTree(nodes, edges, terminals)
    print("SteinerTree object created")
    result = solve_with_sqa(
        problem,
        constraint_weight=100,
        num_reads=100000,
        show_stats=True,
        show_progress=True,
    )

    print("best energy:", result["best_energy_with_offset"])
    print("best energy without offset:", result["best_energy_without_offset"])
    print("best sample:")
    for var, value in result["best_sample"].items():
        if value == 1:
            print(var, value)
