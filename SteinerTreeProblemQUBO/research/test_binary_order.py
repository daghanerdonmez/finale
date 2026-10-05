"""Exact and exhaustive checks of the shared binary-order candidate."""

import itertools
import unittest

import dimod

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.research.steiner_to_bqm_binary_order import (
    complete_sample,
    steiner_to_bqm_binary_order,
)


class BinaryOrderTests(unittest.TestCase):
    def setUp(self):
        self.problem = SteinerTree(
            ["r", "s", "t"], [("r", "s", 2), ("s", "t", 3), ("r", "t", 7)], ["r", "t"]
        )

    def test_all_triangle_arcs_and_ranks_including_ties(self):
        allowed = [("r", "s"), ("s", "t"), ("t", "s"), ("r", "t")]
        for parents in ("square", "sequential"):
            bqm = steiner_to_bqm_binary_order(self.problem, cost_scale=0, parent_encoding=parents)
            for values in itertools.product((0, 1), repeat=4):
                selected = {a for a, value in zip(allowed, values) if value}
                ns = sum(v == "s" for u, v in selected)
                nt = sum(v == "t" for u, v in selected)
                structural = nt == 1 and ns <= 1
                structural &= not any(u == "s" for u, v in selected) or ns == 1
                for s, t in itertools.product((0, 1), repeat=2):
                    ranks = {"r": 0, "s": s, "t": t}
                    ordered = ("s", "t") not in selected or s <= t
                    ordered &= ("t", "s") not in selected or t < s
                    sample = complete_sample(self.problem, selected, ranks, parents)
                    self.assertEqual(bqm.energy(sample) == 0, structural and ordered)

    def test_exact_global_optima_and_no_invalid_ground_states(self):
        for parents in ("square", "sequential"):
            bqm = steiner_to_bqm_binary_order(self.problem, penalty_weight=6, parent_encoding=parents)
            solutions = dimod.ExactSolver().sample(bqm).lowest()
            self.assertEqual(solutions.first.energy, 5)
            for datum in solutions.data():
                selected = {var[1:] for var, bit in datum.sample.items() if var[0] == "e" and bit}
                self.assertEqual(selected, {("r", "s"), ("s", "t")})

    def test_two_vertex_zero_bit_boundary(self):
        problem = SteinerTree(["r", "t"], [("r", "t", 5)], ["r", "t"])
        bqm = steiner_to_bqm_binary_order(problem, penalty_weight=6)
        self.assertEqual(bqm.num_variables, 1)
        self.assertEqual(bqm.energy(complete_sample(problem, [("r", "t")])), 5)
        self.assertEqual(dimod.ExactSolver().sample(bqm).first.energy, 5)

    def test_root_need_not_be_first_node(self):
        problem = SteinerTree(["x", "t", "r"], [("r", "x", 1), ("x", "t", 2)], ["r", "t"])
        bqm = steiner_to_bqm_binary_order(problem, penalty_weight=4)
        sample = complete_sample(problem, [("r", "x"), ("x", "t")])
        self.assertEqual(bqm.energy(sample), 3)

    def test_dense_tree_completion_bit_barrier_and_quadratic_bound(self):
        nodes = [str(i) for i in range(9)]
        problem = SteinerTree(nodes, [(u, v, 2) for i, u in enumerate(nodes) for v in nodes[i + 1 :]], ["0", "8"])
        selected = list(zip(nodes, nodes[1:]))
        bqm = steiner_to_bqm_binary_order(problem, penalty_weight=3)
        sample = complete_sample(problem, selected)
        base = bqm.energy(sample)
        self.assertEqual(base, 16)
        self.assertLessEqual(max(map(abs, bqm.quadratic.values())), 12)
        # A rank bit occurs once per nonroot incident edge, not 2**bit times.
        for bit in range(3):
            modified = dict(sample)
            modified[("o", "4", bit)] ^= 1
            self.assertEqual(bqm.energy(modified) - base, 3 * 7)


if __name__ == "__main__":
    unittest.main()
