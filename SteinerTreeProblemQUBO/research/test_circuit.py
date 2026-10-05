"""Correctness tests for the experimental ripple-borrow Steiner QUBO.

Run from the repository root:
    venv/bin/python -m unittest SteinerTreeProblemQUBO.research.test_circuit
"""

import itertools
import random
import unittest

import dimod

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.research.steiner_to_bqm_circuit import (
    _context,
    complete_sample,
    steiner_to_bqm_circuit,
)


class CircuitTests(unittest.TestCase):
    def test_subtractor_truth_table_unique_outputs_and_unit_gap(self):
        for x, y, borrow_in in itertools.product((0, 1), repeat=3):
            zero = []
            for difference, borrow_out in itertools.product((0, 1), repeat=2):
                penalty = (x - y - borrow_in - difference + 2 * borrow_out) ** 2
                if not penalty:
                    zero.append((difference, borrow_out))
                else:
                    self.assertGreaterEqual(penalty, 1)
            residual = x - y - borrow_in
            expected_borrow = int(residual < 0)
            self.assertEqual(zero, [(residual + 2 * expected_borrow, expected_borrow)])

    def test_ripple_borrow_is_strict_unsigned_comparison(self):
        for bits in range(1, 7):
            for x, y in itertools.product(range(1 << bits), repeat=2):
                borrow = 0
                difference_word = 0
                for i in range(bits):
                    residual = ((x >> i) & 1) - ((y >> i) & 1) - borrow
                    borrow = int(residual < 0)
                    difference_word += (residual + 2 * borrow) << i
                self.assertEqual(borrow, int(x < y))
                self.assertEqual(x - y, difference_word - (borrow << bits))

    def test_all_triangle_projections_match_tree_and_rank_constraints(self):
        problem = SteinerTree(["r", "s", "t"], [("r", "s", 2), ("s", "t", 3), ("r", "t", 7)], ["r", "t"])
        allowed = [("r", "s"), ("s", "t"), ("t", "s"), ("r", "t")]
        for parent_encoding in ("sequential", "square"):
            bqm = steiner_to_bqm_circuit(problem, cost_scale=0, parent_encoding=parent_encoding)
            for choices in itertools.product((0, 1), repeat=len(allowed)):
                selected = {arc for arc, value in zip(allowed, choices) if value}
                parents_s = sum(v == "s" for u, v in selected)
                parents_t = sum(v == "t" for u, v in selected)
                structural = parents_t == 1 and parents_s <= 1
                structural &= not any(u == "s" for u, v in selected) or parents_s == 1
                for rank_s, rank_t in itertools.product(range(4), repeat=2):
                    ranks = {"r": 0, "s": rank_s, "t": rank_t}
                    expected = structural and all(ranks[u] < ranks[v] for u, v in selected)
                    sample = complete_sample(problem, selected, depths=ranks, parent_encoding=parent_encoding)
                    self.assertEqual(bqm.energy(sample) == 0, expected, (selected, ranks, parent_encoding))

    def test_full_bqm_expansion_matches_independent_local_energy(self):
        problem = SteinerTree(["r", "s", "t"], [("r", "s", 2), ("s", "t", 3), ("r", "t", 7)], ["r", "t"])
        nodes, terminals, root, bits, arcs, incoming = _context(problem)
        penalty, scale = 2.5, 0.7
        rng = random.Random(912)
        for parent_encoding in ("sequential", "square"):
            bqm = steiner_to_bqm_circuit(problem, penalty, scale, parent_encoding)
            for _ in range(500):
                s = {var: rng.randrange(2) for var in bqm.variables}
                cost = sum(scale * cost * s[("e", u, v)] for u, v, cost in arcs)
                violations = 0
                for v in nodes:
                    if v == root:
                        continue
                    target = 1 if v in terminals else s[("p", v)]
                    if parent_encoding == "square" or not incoming[v]:
                        violations += (target - sum(s[e] for e in incoming[v])) ** 2
                    else:
                        previous = 0
                        for i, e in enumerate(incoming[v], start=1):
                            current = target if i == len(incoming[v]) else s[("cparent", v, i)]
                            violations += (current - previous - s[e]) ** 2
                            previous = current
                for u, v, _cost in arcs:
                    e = s[("e", u, v)]
                    if u not in terminals:
                        violations += e * (1 - s[("p", u)])
                    borrow = 0
                    for i in range(bits):
                        x = 0 if u == root else s[("o", u, i)]
                        y = s[("o", v, i)]
                        next_borrow = s[("cborrow", u, v, i + 1)]
                        difference = s[("cdiff", u, v, i)]
                        violations += (x - y - borrow - difference + 2 * next_borrow) ** 2
                        borrow = next_borrow
                    violations += e * (1 - borrow)
                self.assertAlmostEqual(bqm.energy(s), cost + penalty * violations)

    def test_exact_global_optimum_with_sufficient_penalty(self):
        problem = SteinerTree(["r", "t"], [("r", "t", 5)], ["r", "t"])
        bqm = steiner_to_bqm_circuit(problem, penalty_weight=6)
        ground = dimod.ExactSolver().sample(bqm).lowest()
        self.assertEqual(ground.first.energy, 5)
        for record in ground.data():
            self.assertEqual(record.sample[("e", "r", "t")], 1)

    def test_tree_completion_and_quadratic_bound_on_high_degree_graph(self):
        nodes = [str(i) for i in range(10)]
        edges = [(u, v, 2) for i, u in enumerate(nodes) for v in nodes[i + 1 :]]
        problem = SteinerTree(nodes, edges, ["0", "8", "9"])
        selected = [("0", "1"), ("1", "2"), ("2", "8"), ("2", "9")]
        for parent_encoding in ("sequential", "square"):
            bqm = steiner_to_bqm_circuit(problem, penalty_weight=3, parent_encoding=parent_encoding)
            sample = complete_sample(problem, selected, parent_encoding=parent_encoding)
            self.assertEqual(set(sample), set(bqm.variables))
            self.assertEqual(bqm.energy(sample), 8)
            self.assertLessEqual(max(map(abs, bqm.quadratic.values())), 12)

    def test_disconnected_terminal_has_positive_penalty(self):
        problem = SteinerTree(["r", "s", "t"], [("r", "s", 1)], ["r", "t"])
        bqm = steiner_to_bqm_circuit(problem, cost_scale=0)
        self.assertGreaterEqual(bqm.energy(complete_sample(problem, [])), 1)


if __name__ == "__main__":
    unittest.main()
