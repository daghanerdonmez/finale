"""Experimental shared binary-order comparator formulation.

Assign each NONROOT vertex a B-bit rank, B=ceil(log2(n-1)), and order vertices
lexicographically by (rank, position in problem.nodes).  Numerical rank ties
are safe because the second component breaks ties.  For a nonroot edge u,v
with index(u)<index(v), one ripple subtraction rank[v]-rank[u] computes final
borrow c = [rank[v]<rank[u]].  Add e[u,v]*c + e[v,u]*(1-c).

This shares one circuit between both directions.  Root arcs need no ordering
constraint because incoming root arcs do not exist.  A zero penalty implies
acyclicity; conversely every rooted tree has suitable ranks 0,...,n-2.
Primitive quadratic coefficients have magnitude <=4 times penalty_weight.
The same caveats as the unshared comparator model apply: auxiliary barriers
remain and linear/Ising coefficients may grow with graph degree.
"""

from collections import deque
from math import isfinite

import dimod

from SteinerTreeProblemQUBO.research.steiner_to_bqm_circuit import _context, _square


def steiner_to_bqm_binary_order(
    problem,
    penalty_weight=1.0,
    cost_scale=1.0,
    parent_encoding="square",
):
    """Build a BQM with one binary comparator per nonroot undirected edge."""
    if not isfinite(penalty_weight) or penalty_weight <= 0:
        raise ValueError("penalty_weight must be finite and positive.")
    if not isfinite(cost_scale) or cost_scale < 0:
        raise ValueError("cost_scale must be finite and nonnegative.")
    if parent_encoding not in {"square", "sequential"}:
        raise ValueError("parent_encoding must be 'square' or 'sequential'.")
    nodes, terminals, root, _bits, arcs, incoming = _context(problem)
    bits = (len(nodes) - 2).bit_length()
    index = {v: i for i, v in enumerate(nodes)}
    weight = float(penalty_weight)
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    for v in nodes:
        if v != root:
            for i in range(bits):
                bqm.add_variable(("o", v, i), 0.0)
        if v not in terminals:
            bqm.add_variable(("p", v), 0.0)
    for u, v, cost in arcs:
        bqm.add_variable(("e", u, v), cost_scale * cost)

    for v in nodes:
        if v == root:
            continue
        parents = incoming[v]
        if parent_encoding == "square" or not parents:
            terms = [(e, -1) for e in parents]
            if v not in terminals:
                terms.append((("p", v), 1))
            _square(bqm, terms, int(v in terminals), weight)
        else:
            for i, e in enumerate(parents, start=1):
                terms = [(e, -1)]
                if i > 1:
                    terms.append((("cparent", v, i - 1), -1))
                constant = 0
                if i < len(parents):
                    terms.append((("cparent", v, i), 1))
                elif v in terminals:
                    constant = 1
                else:
                    terms.append((("p", v), 1))
                _square(bqm, terms, constant, weight)
    for u, v, _ in arcs:
        if u not in terminals:
            bqm.add_linear(("e", u, v), weight)
            bqm.add_quadratic(("e", u, v), ("p", u), -weight)

    for first, second, _ in problem.edges:
        if root in (first, second):
            continue
        u, v = sorted((first, second), key=index.__getitem__)
        for i in range(bits):
            terms = [
                (("o", v, i), 1),
                (("o", u, i), -1),
                (("bdiff", u, v, i), -1),
                (("bborrow", u, v, i + 1), 2),
            ]
            if i:
                terms.append((("bborrow", u, v, i), -1))
            _square(bqm, terms, 0, weight)
        final_borrow = ("bborrow", u, v, bits)
        bqm.add_quadratic(("e", u, v), final_borrow, weight)
        bqm.add_linear(("e", v, u), weight)
        bqm.add_quadratic(("e", v, u), final_borrow, -weight)
    for u, v, bias in list(bqm.iter_quadratic()):
        if bias == 0:
            bqm.remove_interaction(u, v)
    return bqm


def complete_sample(problem, selected_arcs, ranks=None, parent_encoding="square"):
    """Create consistent comparator witnesses; tree ranks default to depth-1."""
    nodes, terminals, root, _bits, arcs, incoming = _context(problem)
    bits = (len(nodes) - 2).bit_length()
    index = {v: i for i, v in enumerate(nodes)}
    selected = set(selected_arcs)
    if not selected.issubset({(u, v) for u, v, _ in arcs}):
        raise ValueError("selected_arcs contains an unavailable arc.")
    if ranks is None:
        depth = {root: 0}
        queue = deque([root])
        outgoing = {v: [] for v in nodes}
        for u, v in selected:
            outgoing[u].append(v)
        while queue:
            u = queue.popleft()
            for v in outgoing[u]:
                if v not in depth:
                    depth[v] = depth[u] + 1
                    queue.append(v)
        rank = {v: max(0, depth.get(v, 0) - 1) for v in nodes}
    else:
        rank = dict(ranks)
        rank.setdefault(root, 0)
    if set(rank) != set(nodes):
        raise ValueError("ranks must specify every nonroot vertex.")
    if any(not isinstance(rank[v], int) or not 0 <= rank[v] < (1 << bits) for v in nodes if v != root):
        raise ValueError("Nonroot ranks must be integers in the encoded range.")
    bqm = steiner_to_bqm_binary_order(problem, parent_encoding=parent_encoding)
    sample = {var: 0 for var in bqm.variables}
    for v in nodes:
        if v != root:
            for i in range(bits):
                sample[("o", v, i)] = (rank[v] >> i) & 1
        if v not in terminals:
            sample[("p", v)] = int(any((e[1], e[2]) in selected for e in incoming[v]))
        if parent_encoding == "sequential":
            prefix = 0
            for i, e in enumerate(incoming[v], start=1):
                prefix += int((e[1], e[2]) in selected)
                if i < len(incoming[v]):
                    sample[("cparent", v, i)] = int(prefix > 0)
    for u, v, _ in arcs:
        sample[("e", u, v)] = int((u, v) in selected)
    for first, second, _ in problem.edges:
        if root in (first, second):
            continue
        u, v = sorted((first, second), key=index.__getitem__)
        borrow = 0
        for i in range(bits):
            residual = ((rank[v] >> i) & 1) - ((rank[u] >> i) & 1) - borrow
            borrow = int(residual < 0)
            sample[("bdiff", u, v, i)] = residual + 2 * borrow
            sample[("bborrow", u, v, i + 1)] = borrow
    return sample
