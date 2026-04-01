from SteinerTreeProblemQUBO.random_problem_generator import generate_random_steiner_tree
from SteinerTreeProblemQUBO.exact_solver import solve as exact_solve
from SteinerTreeProblemQUBO.MyFormulization.oj_solver import solve_with_sqa

# ── Parameters ──────────────────────────────────────────────────────
SEED_START = 11
SEED_END = 20

NODE_COUNT = 10
WEIGHT_RANGE = (10, 100)
TERMINAL_COUNT = 3
EXTRA_EDGE_PROB = 0.3

CONSTRAINT_WEIGHT = 100
OJ_VERSION = 2
NUM_READS = 10000
# ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"{'Seed':<6} {'Exact':>8} {'OJ':>8} {'Match':>6}")
    print("-" * 32)

    for seed in range(SEED_START, SEED_END + 1):
        problem = generate_random_steiner_tree(
            NODE_COUNT, WEIGHT_RANGE, TERMINAL_COUNT, EXTRA_EDGE_PROB, seed
        )

        exact_result = exact_solve(problem)
        exact_cost = exact_result["cost"]

        oj_result = solve_with_sqa(
            problem,
            constraint_weight=CONSTRAINT_WEIGHT,
            version=OJ_VERSION,
            num_reads=NUM_READS,
            show_stats=False,
            show_progress=False,
        )
        oj_energy = oj_result["best_energy_with_offset"]

        match = "✓" if exact_cost == oj_energy else "✗"
        print(f"{seed:<6} {exact_cost:>8} {oj_energy:>8.1f} {match:>6}")
