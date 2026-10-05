"""Experimental Steiner-tree QUBO using ripple-borrow comparison circuits.

For each allowed arc u -> v, compute unsigned subtraction o_u - o_v:

    o[u,i] - o[v,i] - b[i] - d[i] + 2*b[i+1] = 0,
    b[0] = 0.

All arithmetic equations are squared *locally*, one bit at a time.  The final
borrow b[B] is one exactly when o_u < o_v.  The additional quadratic penalty
e[u,v]*(1-b[B]) therefore forbids a selected arc with nonincreasing depth.
Root-depth bits are eliminated, not penalized.  There are no big-M constants
or binary place values in a penalty.  Every primitive quadratic coefficient
has absolute value at most 4*penalty_weight.  Shared depth bits accumulate
linear coefficients proportional to degree; this is not a bounded-degree
Ising construction, nor a claim of annealing superiority.

The sequential parent encoding uses binary prefix sums and costs O(m+n)
variables and interactions.  The alternative square encoding is closer to
the report, but retains O(sum_v degree(v)^2) interactions.  With sequential
parents, the complete model has O((m+n)*log(n)) variables and interactions.

All unweighted penalties are nonnegative integers.  For nonnegative edge
costs, any penalty_weight greater than a known feasible tree's scaled cost
guarantees a feasible global optimum.  This is a sufficient conservative
bound, not a prescription for annealing tuning.  Setting the penalty to one
has no such general guarantee.

WARNING: the subtraction circuit is enforced even on unselected arcs.
Auxiliary consistency barriers and binary carry barriers can still impede
single-bit annealing.  This module is a research prototype.
"""

from __future__ import annotations

from collections import deque
from math import ceil, isfinite, log2
from typing import Iterable, Mapping

import dimod


def _context(problem):
    nodes = list(problem.nodes)
    terminals = set(problem.terminals)
    if len(terminals) < 2:
        raise ValueError("At least two terminals are required.")
    if len(nodes) != len(set(nodes)) or not terminals.issubset(nodes):
        raise ValueError("Nodes must be unique and contain every terminal.")
    root = problem.terminals[0]
    incoming = {v: [] for v in nodes}
    arcs = []
    seen = set()
    for u, v, cost in problem.edges:
        edge = frozenset((u, v))
        if u == v or u not in incoming or v not in incoming or edge in seen:
            raise ValueError("Edges must be simple, undirected, and use known nodes.")
        if not isfinite(float(cost)) or cost < 0:
            raise ValueError("Finite nonnegative edge costs are required.")
        seen.add(edge)
        for source, target in ((u, v), (v, u)):
            if target != root:
                arc = (source, target, float(cost))
                arcs.append(arc)
                incoming[target].append(("e", source, target))
    return nodes, terminals, root, ceil(log2(len(nodes))), arcs, incoming


def _square(bqm, coefficients, constant, weight):
    """Add a square while handling repeated variables and binary diagonals."""
    terms = {}
    for var, coefficient in coefficients:
        terms[var] = terms.get(var, 0) + coefficient
    items = [(var, coeff) for var, coeff in terms.items() if coeff]
    bqm.offset += weight * constant * constant
    for i, (var, coefficient) in enumerate(items):
        bqm.add_linear(var, weight * (coefficient**2 + 2 * constant * coefficient))
        for other, other_coefficient in items[i + 1 :]:
            bqm.add_quadratic(var, other, 2 * weight * coefficient * other_coefficient)


def steiner_to_bqm_circuit(
    problem,
    penalty_weight: float = 1.0,
    cost_scale: float = 1.0,
    parent_encoding: str = "sequential",
) -> dimod.BinaryQuadraticModel:
    """Build the ripple-borrow model; arc labels match the existing models.

    ``parent_encoding='sequential'`` enforces binary prefix sums of incoming
    arcs.  ``'square'`` uses the report's squared parent constraints.
    ``cost_scale`` can scale the objective without changing primitive penalties.
    """
    if not isfinite(penalty_weight) or penalty_weight <= 0:
        raise ValueError("penalty_weight must be finite and positive.")
    if not isfinite(cost_scale) or cost_scale < 0:
        raise ValueError("cost_scale must be finite and nonnegative.")
    if parent_encoding not in {"sequential", "square"}:
        raise ValueError("parent_encoding must be 'sequential' or 'square'.")
    nodes, terminals, root, bits, arcs, incoming = _context(problem)
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    weight = float(penalty_weight)

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
            constant = 1 if v in terminals else 0
            if v not in terminals:
                terms.append((("p", v), 1))
            _square(bqm, terms, constant, weight)
        else:
            # s_i = sum_{j<=i} e_j; the last state is 1 or p_v.
            # Each state is binary, so every prefix is automatically <= 1.
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
        e = ("e", u, v)
        if u not in terminals:
            # A used outgoing arc needs an active, parented source.
            bqm.add_linear(e, weight)
            bqm.add_quadratic(e, ("p", u), -weight)

        for i in range(bits):
            terms = [(("cdiff", u, v, i), -1), (("cborrow", u, v, i + 1), 2)]
            if u != root:
                terms.append((("o", u, i), 1))
            terms.append((("o", v, i), -1))
            if i:
                terms.append((("cborrow", u, v, i), -1))
            _square(bqm, terms, 0, weight)

        bqm.add_linear(e, weight)
        bqm.add_quadratic(e, ("cborrow", u, v, bits), -weight)

    # dimod can retain cancellations as zero interactions; remove those so
    # reported interaction counts refer to actual nonzero coefficients.
    for u, v, bias in list(bqm.iter_quadratic()):
        if bias == 0:
            bqm.remove_interaction(u, v)
    return bqm


def complete_sample(
    problem,
    selected_arcs: Iterable[tuple],
    depths: Mapping | None = None,
    parent_encoding: str = "sequential",
) -> dict:
    """Extend selected arcs/ranks with the deterministic arithmetic witnesses.

    If ranks are omitted, use distance from the root along the selected arcs,
    and zero for unreachable vertices.  Every rooted directed Steiner tree
    therefore receives a zero-penalty extension.  Invalid arcs/ranks are also
    accepted for diagnostics: their parent/usage/comparison penalties remain.
    A saturated parent prefix is used when a vertex has several parents.
    """
    nodes, terminals, root, bits, arcs, incoming = _context(problem)
    selected = set(selected_arcs)
    allowed = {(u, v) for u, v, _ in arcs}
    if not selected.issubset(allowed):
        raise ValueError("selected_arcs contains an unavailable arc.")
    if depths is None:
        rank = {root: 0}
        queue = deque([root])
        outgoing = {v: [] for v in nodes}
        for u, v in selected:
            outgoing[u].append(v)
        while queue:
            u = queue.popleft()
            for v in outgoing[u]:
                if v not in rank:
                    rank[v] = rank[u] + 1
                    queue.append(v)
        rank.update({v: 0 for v in nodes if v not in rank})
    else:
        rank = dict(depths)
        rank.setdefault(root, 0)
    if set(rank) != set(nodes) or rank[root] != 0:
        raise ValueError("depths must specify every vertex and set the root to zero.")
    if any(not isinstance(d, int) or not 0 <= d < (1 << bits) for d in rank.values()):
        raise ValueError("Depths must be integers in the encoded range.")

    bqm = steiner_to_bqm_circuit(problem, parent_encoding=parent_encoding)
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
        borrow = 0
        for i in range(bits):
            residual = ((rank[u] >> i) & 1) - ((rank[v] >> i) & 1) - borrow
            borrow = int(residual < 0)
            sample[("cdiff", u, v, i)] = residual + 2 * borrow
            sample[("cborrow", u, v, i + 1)] = borrow
    return sample
