import random
from typing import Tuple, Optional
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


def generate_random_steiner_tree(
    node_count: int,
    weight_range: Tuple[int, int] = (1, 10),
    terminal_count: int = 2,
    extra_edge_probability: float = 0.3,
    seed: Optional[int] = None,
) -> SteinerTree:
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not (1 <= terminal_count <= node_count):
        raise ValueError("terminal_count must be between 1 and node_count")
    if weight_range[0] > weight_range[1]:
        raise ValueError("invalid weight_range")
    if not (0.0 <= extra_edge_probability <= 1.0):
        raise ValueError("extra_edge_probability must be between 0 and 1")

    rng = random.Random(seed)

    nodes = [f"v{i}" for i in range(node_count)]
    edges = []

    # First make a connected random tree
    for i in range(1, node_count):
        u = nodes[i]
        v = rng.choice(nodes[:i])
        w = rng.randint(weight_range[0], weight_range[1])
        edges.append((u, v, w))

    # Then add extra random edges
    existing = {tuple(sorted((u, v))) for u, v, _ in edges}

    for i in range(node_count):
        for j in range(i + 1, node_count):
            u, v = nodes[i], nodes[j]
            key = (u, v)
            if key not in existing and rng.random() < extra_edge_probability:
                w = rng.randint(weight_range[0], weight_range[1])
                edges.append((u, v, w))
                existing.add(key)

    terminals = rng.sample(nodes, terminal_count)

    return SteinerTree(nodes, edges, terminals)