import math

from SteinerTree import SteinerTree


def x_label(node: str, step: int, terminal: str) -> str:
    return f"x::{node}::{step}::{terminal}"


def y_label(terminal: str, bit_index: int) -> str:
    return f"h6_y::{terminal}::{bit_index}"


def build_pyqubo_model(problem: SteinerTree, constraint_weight: float):
    from pyqubo import Binary

    n = len(problem.nodes)
    S = n

    x_vars = {
        (node, step, terminal): Binary(x_label(node, step, terminal))
        for node in problem.nodes
        for step in range(S + 1)
        for terminal in problem.terminals
    }

    H_A = 0.0
    for i in problem.nodes:
        for j in problem.nodes:
            weight = problem.check_edge(i, j)
            if weight == 0:
                continue

            for terminal in problem.terminals:
                for step in range(S):
                    H_A += (
                        weight
                        * x_vars[(i, step, terminal)]
                        * x_vars[(j, step + 1, terminal)]
                    )

    root_node = problem.terminals[0]
    first_target = problem.terminals[0]
    H_1 = (1 - x_vars[(root_node, 0, first_target)]) ** 2

    H_2 = 0.0
    for terminal in problem.terminals:
        H_2 += (1 - x_vars[(terminal, S, terminal)]) ** 2

    H_3 = 0.0
    for terminal in problem.terminals:
        for step in range(S + 1):
            occupancy = sum(
                (x_vars[(node, step, terminal)] for node in problem.nodes),
                0.0,
            )
            H_3 += occupancy ** 2 - occupancy

    H_4 = 0.0
    for terminal in problem.terminals:
        for step in range(S):
            now_occ = sum(
                (x_vars[(node, step, terminal)] for node in problem.nodes),
                0.0,
            )
            next_occ = sum(
                (x_vars[(node, step + 1, terminal)] for node in problem.nodes),
                0.0,
            )
            H_4 += now_occ - now_occ * next_occ

    H_5 = 0.0
    for terminal in problem.terminals:
        for node in problem.nodes:
            if node in problem.terminals:
                continue
            for step in range(S):
                H_5 += x_vars[(node, step, terminal)] * x_vars[(node, step + 1, terminal)]

    H_6 = 0.0
    for terminal in problem.terminals:
        overlap_terms = []
        for other_terminal in problem.terminals:
            if terminal == other_terminal:
                continue
            for node in problem.nodes:
                for step in range(S + 1):
                    overlap_terms.append(
                        x_vars[(node, step, terminal)] * x_vars[(node, step, other_terminal)]
                    )

        if not overlap_terms:
            continue

        max_overlap = len(overlap_terms)
        max_slack = max_overlap - 1
        num_slack_bits = 0 if max_slack <= 0 else math.ceil(math.log2(max_slack + 1))
        slack = 0.0
        for bit_index in range(num_slack_bits):
            slack += (2 ** bit_index) * Binary(y_label(terminal, bit_index))

        H_6 += (sum(overlap_terms, 0.0) - slack - 1) ** 2

    H_constraints = 0.0
    H_constraints += H_1
    H_constraints += H_2
    H_constraints += H_3
    H_constraints += H_4
    H_constraints += H_5
    H_constraints += H_6

    H = H_A + constraint_weight * H_constraints
    model = H.compile(strength=constraint_weight)
    return model


def solve_with_pyqubo(problem: SteinerTree, constraint_weight: float, num_reads: int = 1000, **sampler_kwargs):
    import openjij as oj

    model = build_pyqubo_model(problem, constraint_weight)
    bqm = model.to_bqm()
    qubo, offset = model.to_qubo()

    sampler = oj.SQASampler()
    response = sampler.sample_qubo(qubo, num_reads=num_reads, **sampler_kwargs)
    decoded = model.decode_sampleset(response)
    best_decoded = min(decoded, key=lambda sample: sample.energy)

    return {
        "model": model,
        "bqm": bqm,
        "qubo": qubo,
        "offset": offset,
        "response": response,
        "decoded_samples": decoded,
        "best_decoded": best_decoded,
    }


if __name__ == "__main__":
    nodes = ["a", "b", "c"]
    edges = [("a", "b", 10), ("a", "c", 10)]
    terminals = ["a", "b", "c"]

    problem = SteinerTree(nodes, edges, terminals)
    result = solve_with_pyqubo(problem, constraint_weight=100, num_reads=1000)

    print("best decoded energy:", result["best_decoded"].energy)
    print("best decoded sample:")
    for var, value in sorted(result["best_decoded"].sample.items()):
        if value == 1:
            print(var, value)
