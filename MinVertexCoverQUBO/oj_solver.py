import sys

import dimod
import openjij as oj
from MinVertexCoverQUBO.MinVertexCover import MinVertexCover
from MinVertexCoverQUBO.exact_solver import solve as solve_exact
from MinVertexCoverQUBO.mvc_to_oj_qubo import mvc_to_oj_qubo
from MinVertexCoverQUBO.random_problem_generator import generate_random_min_vertex_cover
from tqdm import tqdm


def solve_with_sqa(
    problem: MinVertexCover,
    penalty: float,
    num_reads: int = 1000,
    show_stats: bool = False,
    show_progress: bool = False,
    **sampler_kwargs,
):
    if num_reads < 1:
        raise ValueError("num_reads must be at least 1")

    qubo, offset = mvc_to_oj_qubo(problem, penalty)

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
    selected_vertices = problem.selected_vertices_from_sample(best.sample)
    uncovered_edges = problem.uncovered_edges(selected_vertices)

    return {
        "qubo": qubo,
        "offset": offset,
        "response": response,
        "best_sample": best.sample,
        "best_energy_without_offset": best.energy,
        "best_energy_with_offset": best.energy + offset,
        "selected_vertices": selected_vertices,
        "cover_weight": problem.cover_weight(selected_vertices),
        "is_valid_cover": not uncovered_edges,
        "uncovered_edges": uncovered_edges,
    }


if __name__ == "__main__":
    problem = generate_random_min_vertex_cover(
        vertex_count=80,
        edge_probability=0.6,
        weighted=False,
        seed=42,
    )
    penalty = sum(problem.weights.values()) + 1.0

    print("Random MinVertexCover object created")
    #print(f"Vertices: {problem.vertices}")
    #print(f"Edges: {problem.edges}")
    #print(f"Weights: {problem.weights}")
    #print(f"Penalty: {penalty}")

    exact_result = solve_exact(problem)
    print("true optimal energy:", exact_result["optimal_energy"])
    print("true optimal cover:", exact_result["cover"])

    result = solve_with_sqa(
        problem,
        penalty=penalty,
        num_reads=1,
        show_stats=True,
        show_progress=True,
    )

    print("best energy:", result["best_energy_with_offset"])
    print("best energy without offset:", result["best_energy_without_offset"])
    print("selected vertices:", result["selected_vertices"])
    print("cover weight:", result["cover_weight"])
    print("valid cover:", result["is_valid_cover"])
    print("energy gap:", result["best_energy_with_offset"] - exact_result["optimal_energy"])
    if not result["is_valid_cover"]:
        print("uncovered edges:", result["uncovered_edges"])
