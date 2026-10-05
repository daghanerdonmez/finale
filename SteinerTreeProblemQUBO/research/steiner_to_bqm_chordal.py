"""Exact sparse ordering QUBO using a chordal completion of G minus the root.

Triangle consistency suffices on a chordal graph: a shortest directed cycle
of length >= 4 has a chord, which gives a shorter directed cycle. This is
not true if triangle penalties are restricted to the original sparse graph.
The completion is heuristic; low degree alone does not imply low treewidth.
"""
from itertools import combinations
import math

import dimod
import networkx as nx


def chordal_completion(problem, full=False):
    nodes = [v for v in problem.nodes if v != problem.terminals[0]]
    index = {v: i for i, v in enumerate(nodes)}
    h = nx.Graph()
    h.add_nodes_from(nodes)
    h.add_edges_from((u, v) for u, v, _ in problem.edges if u in index and v in index)
    if full:
        h.add_edges_from(combinations(nodes, 2))
    work = h.copy()
    elimination = []
    width = 0
    while work:
        def score(v):
            nbr = list(work[v])
            missing = sum(not work.has_edge(a, b) for a, b in combinations(nbr, 2))
            return missing, len(nbr), index[v]
        v = min(work, key=score)
        nbr = list(work[v])
        width = max(width, len(nbr))
        h.add_edges_from(combinations(nbr, 2))
        work.add_edges_from(combinations(nbr, 2))
        elimination.append(v)
        work.remove_node(v)
    return h, elimination, width


def steiner_to_bqm_chordal(
    problem, penalty_weight=1.0, full=False, parent_encoding="usage"
):
    """Return BQM; full=True isolates the effect of sparsifying order variables.

    parent_encoding="usage" uses the report's p-variable parent constraints.
    "fowler" removes p and uses n*choose(indegree, 2)+(1-indegree)*outdegree.
    "local" replaces n by the number of allowed outgoing arcs plus one.
    Each grouped vertex penalty is a nonnegative integer: with q incoming and
    s outgoing selected arcs, it is (q-1)*(alpha*q/2-s). At q=0 this is s;
    at q=1 it vanishes; at q>=2 it is at least one when alpha>d_out>=s.
    The fake-root factor alone can be negative and must not be independently
    downweighted without rechecking this bound.

    All grouped penalties are nonnegative integers. If costs are nonnegative
    and U is any feasible tree cost, penalty_weight > U is sufficient for exact
    optimization (not necessary). A user-supplied smaller weight is experimental.
    """
    if not math.isfinite(penalty_weight) or penalty_weight <= 0:
        raise ValueError("penalty_weight must be positive and finite")
    if parent_encoding not in {"usage", "fowler", "local"}:
        raise ValueError("parent_encoding must be 'usage', 'fowler', or 'local'")
    if len(set(problem.terminals)) < 2:
        raise ValueError("At least two terminals are required")
    if any(not math.isfinite(w) or w < 0 for _, _, w in problem.edges):
        raise ValueError("Costs must be finite and nonnegative")
    root = problem.terminals[0]
    terminals = set(problem.terminals)
    index = {v: i for i, v in enumerate(problem.nodes)}
    h, elimination, _ = chordal_completion(problem, full=full)
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    incoming = {v: [] for v in problem.nodes}
    outgoing = {v: [] for v in problem.nodes}
    P = float(penalty_weight)

    def order(u, v):
        a, b = sorted((u, v), key=index.get)
        return ("x_order", a, b)

    for u, v in h.edges:
        bqm.add_variable(order(u, v), 0.0)
    for u, v, cost in problem.edges:
        for a, b in ((u, v), (v, u)):
            if b == root:
                continue
            e = ("e", a, b)
            bqm.add_variable(e, float(cost))
            incoming[b].append(e)
            outgoing[a].append(e)
            if a != root:
                x = order(a, b)
                if index[a] < index[b]:
                    bqm.add_linear(e, P)
                    bqm.add_quadratic(e, x, -P)
                else:
                    bqm.add_quadratic(e, x, P)

    # Each triangle is enumerated once at its earliest eliminated vertex.
    later = set(elimination)
    for v in elimination:
        nbr = [u for u in h[v] if u in later]
        for a, b in combinations(nbr, 2):
            u, mid, w = sorted((v, a, b), key=index.get)
            x, y, z = order(u, mid), order(mid, w), order(u, w)
            bqm.add_linear(z, P)
            bqm.add_quadratic(x, y, P)
            bqm.add_quadratic(x, z, -P)
            bqm.add_quadratic(z, y, -P)
        later.remove(v)

    for v in problem.nodes:
        if v == root:
            continue
        inc = incoming[v]
        if v in terminals:
            bqm.offset += P
            for e in inc:
                bqm.add_linear(e, -P)
            for e, f in combinations(inc, 2):
                bqm.add_quadratic(e, f, 2 * P)
        elif parent_encoding == "usage":
            p = ("p", v)
            bqm.add_variable(p, P)
            for e in inc:
                bqm.add_linear(e, P)
                bqm.add_quadratic(e, p, -2 * P)
            for e in outgoing[v]:
                bqm.add_linear(e, P)
                bqm.add_quadratic(e, p, -P)
            for e, f in combinations(inc, 2):
                bqm.add_quadratic(e, f, 2 * P)
        else:
            alpha = len(problem.nodes) if parent_encoding == "fowler" else len(outgoing[v]) + 1
            for e, f in combinations(inc, 2):
                bqm.add_quadratic(e, f, alpha * P)
            for e in outgoing[v]:
                bqm.add_linear(e, P)
                for f in inc:
                    bqm.add_quadratic(e, f, -P)
    bqm.remove_interactions_from([pair for pair, bias in bqm.quadratic.items() if bias == 0])
    return bqm


def complete_sample(problem, selected_arcs, full=False, parent_encoding="usage"):
    """Extend a valid rooted tree to an exactly penalty-free sample."""
    bqm = steiner_to_bqm_chordal(
        problem, full=full, parent_encoding=parent_encoding
    )
    dag = nx.DiGraph()
    dag.add_nodes_from(problem.nodes)
    dag.add_edges_from(selected_arcs)
    ranks = {v: i for i, v in enumerate(nx.topological_sort(dag))}
    selected_arcs = set(selected_arcs)
    used = {v for _, v in selected_arcs}
    sample = {}
    for var in bqm.variables:
        if var[0] == "e":
            sample[var] = int((var[1], var[2]) in selected_arcs)
        elif var[0] == "p":
            sample[var] = int(var[1] in used)
        else:
            sample[var] = int(ranks[var[1]] < ranks[var[2]])
    return sample
