from itertools import combinations
from SteinerTree import SteinerTree
from typing import Set

# Had ChatGPT write it


def solve(problem: SteinerTree) -> dict:
    nodes = problem.nodes
    terminals = set(problem.terminals)
    non_terminals = [v for v in nodes if v not in terminals]

    best_cost = float("inf")
    best_edges = None
    best_steiner_nodes = None

    # Try every subset of non-terminals
    for r in range(len(non_terminals) + 1):
        for subset in combinations(non_terminals, r):
            chosen_vertices = terminals | set(subset)

            mst_cost, mst_edges = _mst_on_chosen_vertices(problem, chosen_vertices)

            if mst_edges is not None and mst_cost < best_cost:
                best_cost = mst_cost
                best_edges = mst_edges
                best_steiner_nodes = list(subset)

    if best_edges is None:
        raise ValueError("No Steiner tree exists that connects all terminals.")

    return {
        "cost": best_cost,
        "edges": best_edges,
        "steiner_nodes": best_steiner_nodes,
    }


def _mst_on_chosen_vertices(problem: SteinerTree, chosen_vertices: Set[str]):
    # Keep only edges whose both endpoints are in chosen_vertices
    candidate_edges = []
    for u, v, w in problem.edges:
        if u in chosen_vertices and v in chosen_vertices:
            candidate_edges.append((u, v, w))

    # Kruskal
    candidate_edges.sort(key=lambda e: e[2])

    parent = {v: v for v in chosen_vertices}
    rank = {v: 0 for v in chosen_vertices}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False

        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    mst_edges = []
    mst_cost = 0

    for u, v, w in candidate_edges:
        if union(u, v):
            mst_edges.append((u, v, w))
            mst_cost += w

    # A spanning tree on k vertices must have k-1 edges
    if len(mst_edges) != len(chosen_vertices) - 1:
        return None, None

    return mst_cost, mst_edges

if __name__ == "__main__":
    nodes = ["a", "b", "c", "d", "e", "f", "g"]
    edges = [
        ("a", "b", 2),
        ("b", "c", 4),
        ("b", "d", 1),
        ("a", "d", 6),
        ("a", "e", 7),
        ("d", "f", 1),
        ("d", "g", 2),
        ("f", "g", 4),
        ("e", "f", 3),
    ]
    terminals = ["a", "c", "f", "g"]

    problem = SteinerTree(nodes, edges, terminals)
    solution = solve(problem)
    print(solution["cost"])
    print(solution["edges"])
