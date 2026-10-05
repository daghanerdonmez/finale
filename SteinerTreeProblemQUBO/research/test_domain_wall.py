"""Mathematical and exhaustive small-instance checks for the research encoding."""

import itertools
import unittest

import dimod

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.research.steiner_to_bqm_domain_wall import (
    encode_domain_wall_tree,
    steiner_to_bqm_domain_wall,
)


class DomainWallTests(unittest.TestCase):
    def test_clause_quadratization_truth_table(self):
        for e, z, w in itertools.product((0, 1), repeat=3):
            values = [e * z + a * (2 - e - z - w) for a in (0, 1)]
            self.assertGreaterEqual(min(values), 0)
            self.assertEqual(min(values), e * z * (1 - w))

    def test_all_small_assignments_and_optimum(self):
        problem = SteinerTree(
            ["r", "s", "t"],
            [("r", "s", 1), ("s", "t", 1), ("r", "t", 5)],
            ["r", "t"],
        )
        P = 8
        bqm = steiner_to_bqm_domain_wall(problem, penalty_weight=P)
        samples = dimod.ExactSolver().sample(bqm)
        self.assertAlmostEqual(samples.first.energy, 2)
        zero_penalty_count = 0
        for record in samples.data(["sample", "energy"]):
            sample = record.sample
            selected = [(var[1], var[2]) for var in bqm.variables if var[0] == "e" and sample[var]]
            costs = {frozenset((u, v)): w for u, v, w in problem.edges}
            cost = sum(costs[frozenset((u, v))] for u, v in selected)
            penalty = (record.energy - cost) / P
            self.assertGreaterEqual(penalty, 0)
            self.assertAlmostEqual(penalty, round(penalty))
            if penalty == 0:
                zero_penalty_count += 1
                # This completion routine independently verifies tree structure.
                encoded = encode_domain_wall_tree(problem, selected, bqm=bqm)
                self.assertAlmostEqual(bqm.energy(encoded), cost)
                incoming = {v: sum(b == v for _, b in selected) for v in problem.nodes}
                self.assertEqual(incoming["r"], 0)
                self.assertEqual(incoming["t"], 1)
                self.assertLessEqual(incoming["s"], 1)
        self.assertGreater(zero_penalty_count, 0)

    def test_long_path_completion_and_coefficient_bound(self):
        nodes = [str(i) for i in range(12)]
        edges = [(nodes[i], nodes[i + 1], i + 1) for i in range(11)]
        problem = SteinerTree(nodes, edges, [nodes[0], nodes[-1]])
        P = 80
        bqm = steiner_to_bqm_domain_wall(problem, penalty_weight=P)
        sample = encode_domain_wall_tree(problem, edges, bqm=bqm)
        self.assertAlmostEqual(bqm.energy(sample), sum(w for _, _, w in edges))
        self.assertLessEqual(max(map(abs, bqm.quadratic.values())), 2 * P)
        D, arc_count, root_degree = 11, 21, 1
        expected = arc_count + 10 + 11 * D + (arc_count - root_degree) * (D - 1)
        self.assertEqual(bqm.num_variables, expected)

    def test_cycle_cannot_have_strict_depths(self):
        D = 3
        # All monotone wall assignments correspond to these integer depths.
        for depths in itertools.product(range(D + 1), repeat=3):
            penalty = 0
            for u, v in ((0, 1), (1, 2), (2, 0)):
                penalty += int(depths[v] == 0) + int(depths[u] == D)
                penalty += sum(int(depths[u] >= k and depths[v] < k + 1) for k in range(1, D))
            self.assertGreater(penalty, 0)


if __name__ == "__main__":
    unittest.main()
