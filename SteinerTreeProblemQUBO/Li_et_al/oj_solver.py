from SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.Li_et_al.steiner_to_oj_qubo import steiner_to_oj_qubo_Li_et_al
import openjij as oj


def solve_with_sqa(
    problem: SteinerTree,
    constraint_weight: float,
    num_reads: int = 1000,
    **sampler_kwargs,
):
    qubo, offset = steiner_to_oj_qubo_Li_et_al(problem, constraint_weight)

    sampler = oj.SQASampler()
    response = sampler.sample_qubo(qubo, num_reads=num_reads, **sampler_kwargs)
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
    #nodes = ["a", "b", "c", "d"]
    #edges = [("a","b",10),("b","c",10),("a","c",10),("a","d",5),("b","d",5),("c","d",5)]
    #terminals = ["a","b","c"]

    nodes = ["a", "b", "c"]
    edges = [("a","b",10), ("a", "c", 10)]
    terminals = ["a","b", "c"]

    problem = SteinerTree(nodes, edges, terminals)
    result = solve_with_sqa(problem, constraint_weight=100, num_reads=10000)

    print("best energy without offset:", result["best_energy_without_offset"])
    print("best energy with offset:", result["best_energy_with_offset"])
    print("best sample:")
    for var, value in result["best_sample"].items():
        if value == 1:
            print(var, value)
