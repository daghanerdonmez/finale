import random
from typing import Optional, Tuple

from MinVertexCoverQUBO.MinVertexCover import MinVertexCover


def _edge_key(u: str, v: str) -> Tuple[str, str]:
    return tuple(sorted((u, v)))


def generate_random_min_vertex_cover(
    vertex_count: int,
    edge_probability: float = 0.3,
    weighted: bool = False,
    weight_range: Tuple[int, int] = (1, 10),
    ensure_connected: bool = True,
    seed: Optional[int] = None,
) -> MinVertexCover:
    if vertex_count < 2:
        raise ValueError("vertex_count must be at least 2")
    if not (0.0 <= edge_probability <= 1.0):
        raise ValueError("edge_probability must be between 0 and 1")
    if weight_range[0] > weight_range[1]:
        raise ValueError("invalid weight_range")

    rng = random.Random(seed)
    vertices = [f"v{i}" for i in range(vertex_count)]
    edges = []
    existing_edges = set()

    if ensure_connected:
        for i in range(1, vertex_count):
            u = vertices[i]
            v = rng.choice(vertices[:i])
            key = _edge_key(u, v)
            edges.append(key)
            existing_edges.add(key)

    for i in range(vertex_count):
        for j in range(i + 1, vertex_count):
            key = _edge_key(vertices[i], vertices[j])
            if key in existing_edges:
                continue
            if rng.random() < edge_probability:
                edges.append(key)
                existing_edges.add(key)

    if not edges:
        fallback_vertices = rng.sample(vertices, 2)
        fallback_edge = _edge_key(fallback_vertices[0], fallback_vertices[1])
        edges.append(fallback_edge)

    weights = None
    if weighted:
        weights = {
            vertex: rng.randint(weight_range[0], weight_range[1])
            for vertex in vertices
        }

    return MinVertexCover(vertices, edges, weights)
