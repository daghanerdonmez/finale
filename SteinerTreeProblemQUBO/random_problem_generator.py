import math
import random
from typing import Tuple, Optional, Literal
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


def generate_geometric_steiner_tree(
    node_count: int,
    terminal_count: int = 3,
    region_size: float = 100.0,
    connectivity: Literal["complete", "knn", "radius"] = "knn",
    k: int = 5,
    radius: float = 40.0,
    noise_std: float = 0.0,
    seed: Optional[int] = None,
) -> SteinerTree:
    """Generate a Steiner tree instance from random points in a 2D square.

    Nodes are placed uniformly at random in [0, region_size]^2.
    Edge weights are (rounded) Euclidean distances, optionally with additive
    Gaussian noise.  The graph is built using one of three connectivity models:

      - "complete": all pairs connected (only practical for small n).
      - "knn": each node connects to its k nearest neighbors (undirected).
      - "radius": nodes within Euclidean distance `radius` are connected.

    After building the graph a BFS check ensures the graph is connected.  If it
    is not (possible with knn/radius), extra edges are added greedily until
    connectivity is achieved.

    Parameters
    ----------
    node_count : int
        Number of nodes (>= 2).
    terminal_count : int
        Number of terminal nodes (1 <= terminal_count <= node_count).
    region_size : float
        Side length of the square region.
    connectivity : {"complete", "knn", "radius"}
        How to decide which pairs of nodes get an edge.
    k : int
        Number of nearest neighbors (only used when connectivity="knn").
    radius : float
        Connection radius (only used when connectivity="radius").
    noise_std : float
        Standard deviation of Gaussian noise added to each edge weight.
        The final weight is max(1, round(dist + noise)).  Set to 0 for
        pure Euclidean weights.
    seed : int or None
        Random seed for reproducibility.
    """
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not (1 <= terminal_count <= node_count):
        raise ValueError("terminal_count must be between 1 and node_count")

    rng = random.Random(seed)

    # --- place nodes ---
    nodes = [f"v{i}" for i in range(node_count)]
    coords = [(rng.uniform(0, region_size), rng.uniform(0, region_size))
              for _ in range(node_count)]

    def _dist(i: int, j: int) -> float:
        dx = coords[i][0] - coords[j][0]
        dy = coords[i][1] - coords[j][1]
        return math.sqrt(dx * dx + dy * dy)

    def _weight(i: int, j: int) -> int:
        d = _dist(i, j)
        if noise_std > 0:
            d += rng.gauss(0, noise_std)
        return max(1, round(d))

    # --- build edge set ---
    edge_set: set = set()

    if connectivity == "complete":
        for i in range(node_count):
            for j in range(i + 1, node_count):
                edge_set.add((i, j))

    elif connectivity == "knn":
        for i in range(node_count):
            dists = sorted(range(node_count), key=lambda j: _dist(i, j))
            for j in dists[1: k + 1]:
                pair = (min(i, j), max(i, j))
                edge_set.add(pair)

    elif connectivity == "radius":
        for i in range(node_count):
            for j in range(i + 1, node_count):
                if _dist(i, j) <= radius:
                    edge_set.add((i, j))
    else:
        raise ValueError(f"unknown connectivity model: {connectivity}")

    # --- ensure connectivity via BFS + greedy patching ---
    def _components():
        parent = list(range(node_count))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            a, b = find(a), find(b)
            if a != b:
                parent[a] = b

        for i, j in edge_set:
            union(i, j)

        comp: dict = {}
        for i in range(node_count):
            r = find(i)
            comp.setdefault(r, []).append(i)
        return list(comp.values())

    components = _components()
    while len(components) > 1:
        # connect two closest components
        best_d = float("inf")
        best_pair = None
        for ci in range(len(components)):
            for cj in range(ci + 1, len(components)):
                for ni in components[ci]:
                    for nj in components[cj]:
                        d = _dist(ni, nj)
                        if d < best_d:
                            best_d = d
                            best_pair = (min(ni, nj), max(ni, nj))
        edge_set.add(best_pair)
        components = _components()

    edges = [(nodes[i], nodes[j], _weight(i, j)) for i, j in sorted(edge_set)]
    terminals = rng.sample(nodes, terminal_count)
    return SteinerTree(nodes, edges, terminals)


def generate_erdos_renyi_steiner_tree(
    node_count: int,
    terminal_count: int = 3,
    edge_probability: float = 0.3,
    weight_range: Tuple[int, int] = (1, 100),
    seed: Optional[int] = None,
) -> SteinerTree:
    """Generate a Steiner tree instance on a connected Erdős–Rényi G(n,p) graph.

    Each possible edge is included independently with probability
    `edge_probability`.  If the resulting graph is not connected, extra edges
    are added between components (chosen uniformly at random) until it is.

    Parameters
    ----------
    node_count : int
        Number of nodes (>= 2).
    terminal_count : int
        Number of terminal nodes.
    edge_probability : float
        Probability that any given edge exists (0 < p <= 1).
    weight_range : (int, int)
        Inclusive range for uniform random integer edge weights.
    seed : int or None
        Random seed for reproducibility.
    """
    if node_count < 2:
        raise ValueError("node_count must be at least 2")
    if not (1 <= terminal_count <= node_count):
        raise ValueError("terminal_count must be between 1 and node_count")
    if not (0 < edge_probability <= 1.0):
        raise ValueError("edge_probability must be in (0, 1]")
    if weight_range[0] > weight_range[1]:
        raise ValueError("invalid weight_range")

    rng = random.Random(seed)
    nodes = [f"v{i}" for i in range(node_count)]

    edge_set: set = set()
    for i in range(node_count):
        for j in range(i + 1, node_count):
            if rng.random() < edge_probability:
                edge_set.add((i, j))

    # --- ensure connectivity (union-find + random bridging) ---
    parent = list(range(node_count))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[a] = b
            return True
        return False

    for i, j in edge_set:
        union(i, j)

    # collect components
    comp: dict = {}
    for i in range(node_count):
        comp.setdefault(find(i), []).append(i)
    comp_list = list(comp.values())

    while len(comp_list) > 1:
        # pick two distinct components and bridge them with a random edge
        c0 = comp_list[0]
        c1 = comp_list[1]
        ni = rng.choice(c0)
        nj = rng.choice(c1)
        pair = (min(ni, nj), max(ni, nj))
        edge_set.add(pair)
        union(ni, nj)
        # rebuild components
        comp = {}
        for i in range(node_count):
            comp.setdefault(find(i), []).append(i)
        comp_list = list(comp.values())

    edges = [(nodes[i], nodes[j], rng.randint(*weight_range))
             for i, j in sorted(edge_set)]
    terminals = rng.sample(nodes, terminal_count)
    return SteinerTree(nodes, edges, terminals)


def generate_grid_steiner_tree(
    rows: int,
    cols: int,
    terminal_count: int = 3,
    weight_range: Tuple[int, int] = (1, 100),
    diagonal_edges: bool = False,
    random_removal_prob: float = 0.0,
    seed: Optional[int] = None,
) -> SteinerTree:
    """Generate a Steiner tree instance on a 2D grid graph.

    Grid graphs have a known, regular structure that is useful as a baseline.
    Nodes are arranged on an (rows x cols) grid with edges to their 4-neighbors
    (or 8-neighbors if `diagonal_edges=True`).  Edges can be randomly removed
    to create sparser variants while keeping the graph connected.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.  Total nodes = rows * cols (must be >= 2).
    terminal_count : int
        Number of terminal nodes.
    weight_range : (int, int)
        Inclusive range for uniform random integer edge weights.
    diagonal_edges : bool
        If True, include diagonal (8-connectivity) edges as well.
    random_removal_prob : float
        Probability of removing each non-bridge edge after building the grid.
        0 means keep the full grid; higher values produce sparser graphs.
        Edges are only removed if the graph stays connected.
    seed : int or None
        Random seed for reproducibility.
    """
    node_count = rows * cols
    if node_count < 2:
        raise ValueError("grid must have at least 2 nodes")
    if not (1 <= terminal_count <= node_count):
        raise ValueError("terminal_count must be between 1 and node_count")
    if weight_range[0] > weight_range[1]:
        raise ValueError("invalid weight_range")
    if not (0.0 <= random_removal_prob < 1.0):
        raise ValueError("random_removal_prob must be in [0, 1)")

    rng = random.Random(seed)
    nodes = [f"v{r}_{c}" for r in range(rows) for c in range(cols)]

    def idx(r, c):
        return r * cols + c

    edge_set: set = set()
    for r in range(rows):
        for c in range(cols):
            # right neighbor
            if c + 1 < cols:
                edge_set.add((idx(r, c), idx(r, c + 1)))
            # down neighbor
            if r + 1 < rows:
                edge_set.add((idx(r, c), idx(r + 1, c)))
            if diagonal_edges:
                # down-right
                if r + 1 < rows and c + 1 < cols:
                    edge_set.add((idx(r, c), idx(r + 1, c + 1)))
                # down-left
                if r + 1 < rows and c - 1 >= 0:
                    edge_set.add((idx(r, c), idx(r + 1, c - 1)))

    # --- optional random edge removal (keep connectivity) ---
    if random_removal_prob > 0:
        # build adjacency for BFS connectivity check
        adj: dict = {i: set() for i in range(node_count)}
        for i, j in edge_set:
            adj[i].add(j)
            adj[j].add(i)

        def _is_connected():
            visited = set()
            stack = [0]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                stack.extend(adj[n] - visited)
            return len(visited) == node_count

        removable = list(edge_set)
        rng.shuffle(removable)
        for i, j in removable:
            if rng.random() < random_removal_prob:
                edge_set.discard((i, j))
                adj[i].discard(j)
                adj[j].discard(i)
                if not _is_connected():
                    # put it back
                    edge_set.add((i, j))
                    adj[i].add(j)
                    adj[j].add(i)

    edges = [(nodes[i], nodes[j], rng.randint(*weight_range))
             for i, j in sorted(edge_set)]
    terminals = rng.sample(nodes, terminal_count)
    return SteinerTree(nodes, edges, terminals)