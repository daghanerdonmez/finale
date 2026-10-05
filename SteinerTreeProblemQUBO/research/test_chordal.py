"""Independent checks of chordal sparse-ordering exactness and counts."""

import itertools
import unittest

import dimod
import networkx as nx

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.research.steiner_to_bqm_chordal import (
    chordal_completion,
    complete_sample,
    steiner_to_bqm_chordal,
)


class ChordalTests(unittest.TestCase):
    def assert_valid_tree(self, problem, sample):
        root = problem.terminals[0]
        arcs = [(var[1], var[2]) for var, bit in sample.items() if var[0] == "e" and bit]
        directed = nx.DiGraph()
        directed.add_node(root)
        directed.add_edges_from(arcs)
        self.assertTrue(set(problem.terminals).issubset(directed.nodes))
        self.assertTrue(nx.is_arborescence(directed))
        self.assertEqual(directed.in_degree(root), 0)
        self.assertTrue(nx.is_tree(directed.to_undirected()))

    def test_four_cycle_completion_all_orientations(self):
        problem = SteinerTree(
            ["r", "a", "b", "c", "d"],
            [("r", "a", 1), ("a", "b", 1), ("b", "c", 1), ("c", "d", 1), ("d", "a", 1)],
            ["r", "d"],
        )
        h, _, _ = chordal_completion(problem)
        self.assertEqual(h.number_of_edges(), 5)
        self.assertTrue(nx.is_chordal(h))
        bqm = steiner_to_bqm_chordal(problem)
        xs = [var for var in bqm.variables if var[0] == "x_order"]
        for bits in itertools.product((0, 1), repeat=len(xs)):
            sample = dict.fromkeys(bqm.variables, 0)
            sample.update(zip(xs, bits))
            dag = nx.DiGraph()
            dag.add_nodes_from(h)
            for (_, u, v), bit in zip(xs, bits):
                dag.add_edge(u, v) if bit else dag.add_edge(v, u)
            # All edge/use variables vanish; subtract the terminal constant.
            triangle_penalty = bqm.energy(sample) - bqm.offset
            self.assertGreaterEqual(triangle_penalty, 0)
            self.assertEqual(triangle_penalty == 0, nx.is_directed_acyclic_graph(dag))

    def test_exact_tiny_bqm_and_zero_set(self):
        problem = SteinerTree(
            ["r", "s", "t"],
            [("r", "s", 1), ("s", "t", 1), ("r", "t", 5)],
            ["r", "t"],
        )
        costs = {frozenset((u, v)): w for u, v, w in problem.edges}
        for full in (False, True):
            bqm = steiner_to_bqm_chordal(problem, penalty_weight=8, full=full)
            samples = dimod.ExactSolver().sample(bqm)
            self.assertAlmostEqual(samples.first.energy, 2)
            feasible_count = 0
            for row in samples.data(["sample", "energy"]):
                cost = sum(costs[frozenset(var[1:])] for var, bit in row.sample.items() if var[0] == "e" and bit)
                penalty = (row.energy - cost) / 8
                self.assertGreaterEqual(penalty, 0)
                self.assertAlmostEqual(penalty, round(penalty))
                if penalty == 0:
                    feasible_count += 1
                    self.assert_valid_tree(problem, row.sample)
            self.assertGreater(feasible_count, 0)

    def test_random_completion_tree_extension_and_counts(self):
        for seed in range(12):
            n = 5 + seed % 6
            graph = nx.gnp_random_graph(n, 0.15 + 0.05 * (seed % 4), seed=seed)
            graph.add_edges_from((v, v + 1) for v in range(n - 1))
            nodes = [str(v) for v in graph]
            edges = [(str(u), str(v), 1 + (u + v) % 5) for u, v in graph.edges]
            problem = SteinerTree(nodes, edges, ["0", str(n - 1)])
            h, elimination, width = chordal_completion(problem)
            self.assertTrue(nx.is_chordal(h))
            original = {frozenset((str(u), str(v))) for u, v in graph.edges if u != 0 and v != 0}
            self.assertTrue(original.issubset({frozenset(e) for e in h.edges}))
            later = set(elimination)
            triangles = 0
            for v in elimination:
                neighbors = set(h[v]) & later
                self.assertLessEqual(len(neighbors), width)
                self.assertTrue(all(h.has_edge(a, b) for a, b in itertools.combinations(neighbors, 2)))
                triangles += len(neighbors) * (len(neighbors) - 1) // 2
                later.remove(v)
            self.assertLessEqual(h.number_of_edges(), (n - 1) * width)
            self.assertLessEqual(triangles, (n - 1) * width * (width - 1) // 2)
            bqm = steiner_to_bqm_chordal(problem, penalty_weight=20)
            dr = graph.degree(0)
            nt = set(range(1, n - 1))
            parent_pairs = sum(graph.degree(v) * (graph.degree(v) - 1) // 2 for v in range(1, n))
            usage_pairs = 2 * sum(graph.degree(v) for v in nt) - len(set(graph[0]) & nt)
            expected_interactions = 3 * triangles + 2 * (graph.number_of_edges() - dr) + parent_pairs + usage_pairs
            self.assertEqual(bqm.num_interactions, expected_interactions)
            selected = [(str(u), str(v)) for u, v in nx.bfs_edges(graph, 0)]
            costs = {frozenset((u, v)): w for u, v, w in edges}
            cost = sum(costs[frozenset(arc)] for arc in selected)
            for full in (False, True):
                bqm = steiner_to_bqm_chordal(problem, penalty_weight=20, full=full)
                sample = complete_sample(problem, selected, full=full)
                self.assert_valid_tree(problem, sample)
                self.assertAlmostEqual(bqm.energy(sample), cost)
                self.assertLessEqual(max(map(abs, bqm.quadratic.values())), 40)

    def test_transitivity_has_no_ising_fields(self):
        bqm = dimod.BinaryQuadraticModel(
            {"x": 0.0, "y": 0.0, "z": 1.0},
            {("x", "y"): 1.0, ("x", "z"): -1.0, ("y", "z"): -1.0},
            0.0,
            dimod.BINARY,
        )
        h, J, offset = bqm.to_ising()
        self.assertTrue(all(value == 0 for value in h.values()))
        self.assertTrue(all(abs(value) == 0.25 for value in J.values()))
        self.assertEqual(offset, 0.25)


if __name__ == "__main__":
    unittest.main()
