"""Random Steiner Tree generator parameterised by (n, k, p).

For a given (node_count n, terminal_count k, extra_edge_probability p, seed s),
the procedure is:

    1. Build a uniformly random spanning tree on n nodes.
    2. For every non-tree pair of nodes, include an edge independently with
       probability p.  This gives roughly m = (n-1) + p*(C(n,2) - (n-1)) edges.
    3. Sample k terminals uniformly without replacement.

To guarantee that instances with the same seed but different (n, k) are NOT
trivial continuations of each other (e.g. k=3 being a subset of k=5), the
actual RNG seed is a deterministic hash of (seed, n, k).  Two instances that
differ in any of those values therefore use entirely different random streams.
"""
import hashlib
import random
from typing import Optional, Tuple

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def _derive_seed(*parts) -> int:
    """Deterministically derive a 64-bit integer seed from the given parts."""
    key = "|".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big")


def generate_sparsity_steiner_tree(
    node_count: int,
    terminal_count: int,
    extra_edge_probability: float,
    weight_range: Tuple[int, int] = (1, 100),
    seed: Optional[int] = None,
) -> SteinerTree:
    """Generate a random connected graph with a given extra-edge probability.

    The graph is constructed as a random spanning tree plus independent
    Bernoulli(p) samples over the remaining possible edges.  Weights are
    integers drawn uniformly from ``weight_range``.

    The RNG is seeded from a hash of ``(seed, node_count, terminal_count,
    extra_edge_probability)`` so that e.g. (n=10, k=3) and (n=10, k=5) with
    the same ``seed`` are unrelated random graphs, not the same graph with
    different terminal sets.
    """
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not (1 <= terminal_count <= node_count):
        raise ValueError("terminal_count must be between 1 and node_count")
    if not (0.0 <= extra_edge_probability <= 1.0):
        raise ValueError("extra_edge_probability must be in [0, 1]")
    if weight_range[0] > weight_range[1] or weight_range[0] < 0:
        raise ValueError("invalid weight_range")

    rng = random.Random(
        _derive_seed(seed, node_count, terminal_count, extra_edge_probability)
    )

    nodes = [f"v{i}" for i in range(node_count)]

    # 1. Random spanning tree: draw a random permutation of nodes, then
    #    connect each node to a uniformly random earlier node in the
    #    permutation.  This is a standard way to get a uniformly chosen
    #    labelled tree structure (up to labelling) while staying O(n).
    order = list(range(node_count))
    rng.shuffle(order)

    edges = []
    existing = set()
    for i in range(1, node_count):
        u_idx = order[i]
        v_idx = order[rng.randrange(i)]
        key = tuple(sorted((u_idx, v_idx)))
        w = rng.randint(weight_range[0], weight_range[1])
        edges.append((nodes[key[0]], nodes[key[1]], w))
        existing.add(key)

    # 2. Extra edges with probability p on every remaining non-tree pair.
    for i in range(node_count):
        for j in range(i + 1, node_count):
            key = (i, j)
            if key in existing:
                continue
            if rng.random() < extra_edge_probability:
                w = rng.randint(weight_range[0], weight_range[1])
                edges.append((nodes[i], nodes[j], w))
                existing.add(key)

    # 3. Sample terminals uniformly without replacement.
    terminals = rng.sample(nodes, terminal_count)

    return SteinerTree(nodes, edges, terminals)
