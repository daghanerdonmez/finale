import itertools
import random
import unittest

import dimod

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.AlexFowler.steiner_to_bqm_alex import steiner_to_bqm_ordering
from SteinerTreeProblemQUBO.research.steiner_to_bqm_chordal import steiner_to_bqm_chordal


class LocalParentTests(unittest.TestCase):
    def test_nonnegative_exact_zero_set(self):
        for indegree in range(9):
            for outdegree in range(9):
                alpha = outdegree + 1
                for q in range(indegree + 1):
                    for s in range(outdegree + 1):
                        value = alpha * q * (q - 1) // 2 + (1 - q) * s
                        self.assertGreaterEqual(value, 0)
                        self.assertEqual(value == 0, q == 1 or (q == 0 and s == 0))

    def test_full_order_fowler_is_existing_bqm(self):
        problem = SteinerTree(["a", "b", "r", "t"],
            [("r", "a", 2), ("a", "b", 1), ("b", "t", 1), ("a", "t", 3)], ["r", "t"])
        old = steiner_to_bqm_ordering(problem, 8)
        new = steiner_to_bqm_chordal(problem, 8, full=True, parent_encoding="fowler")
        self.assertEqual(set(old.variables), set(new.variables))
        for bits in itertools.product((0, 1), repeat=old.num_variables):
            sample = dict(zip(old.variables, bits))
            self.assertEqual(old.energy(sample), new.energy(sample))

    def test_exact_local_optimum(self):
        problem = SteinerTree(["r", "s", "t"], [("r", "s", 2), ("s", "t", 3), ("r", "t", 7)], ["r", "t"])
        for encoding in ("fowler", "local"):
            bqm = steiner_to_bqm_chordal(problem, 6, parent_encoding=encoding)
            for row in dimod.ExactSolver().sample(bqm).lowest().data():
                self.assertEqual(row.energy, 5)
                selected = {v[1:] for v, bit in row.sample.items() if v[0] == "e" and bit}
                self.assertEqual(selected, {("r", "s"), ("s", "t")})
