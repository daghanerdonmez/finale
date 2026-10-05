"""A coefficient-bounded domain-wall alternative to binary big-M depths.

For non-root v, z[v,k] = [depth(v) >= k], 1 <= k <= D=n-1.
Monotonicity costs z[v,k+1]*(1-z[v,k]).  A selected non-root arc
u -> v must satisfy e*(1-z[v,1])=0, e*z[u,D]=0, and
e*z[u,k]*(1-z[v,k+1])=0 for 1 <= k < D.  The latter is reduced by

    e*z + a*(2-e-z-w),

whose minimum over a is e*z*(1-w).  Crucially this quadratic expression
is nonnegative for EVERY assignment, not just after minimizing a.

Together with the report's parent/usage constraints, zero penalty is
equivalent to a rooted tree spanning the terminals.  Unused vertices may
have arbitrary legal depths.  D=n-1 admits every tree.  The number of bits
is |A| + (n-|T|) + (n-1)*D + (|A|-degree(root))*(D-1).
The interaction count is O(m*D + n*D + sum_v degree(v)**2).
Every quadratic coefficient has magnitude <= 2*penalty_weight.  This
repairs precision, but does not promise a favorable single-bit landscape.
"""

from collections import deque
from itertools import combinations
import math

import dimod


def _problem_data(problem):
    nodes = list(problem.nodes)
    terminals = set(problem.terminals)
    if len(nodes) != len(set(nodes)):
        raise ValueError("Duplicate nodes are not supported.")
    if not problem.terminals or not terminals.issubset(nodes):
        raise ValueError("At least one valid terminal is required.")
    root = problem.terminals[0]
    arcs = []
    seen = set()
    for u, v, weight in problem.edges:
        if u == v or u not in nodes or v not in nodes:
            raise ValueError("Edges must have distinct endpoints in problem.nodes.")
        key = frozenset((u, v))
        if key in seen:
            raise ValueError("Parallel or duplicate edges are not supported.")
        seen.add(key)
        weight = float(weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Finite nonnegative edge costs are required.")
        if v != root:
            arcs.append((u, v, weight))
        if u != root:
            arcs.append((v, u, weight))
    return nodes, terminals, root, arcs


def steiner_to_bqm_domain_wall(problem, penalty_weight=None):
    """Return an exact Steiner-tree BQM with unit-scale depth gadgets.

    A sufficient penalty is any value greater than a known feasible tree
    cost; the default (n-1)*max_edge_cost+1 is safe for connected graphs.
    Explicit smaller penalties remain supported for controlled experiments.
    Arc labels are ('e', u, v), matching the existing formulation API.
    """
    nodes, terminals, root, arcs = _problem_data(problem)
    D = len(nodes) - 1
    max_cost = max((w for _, _, w in arcs), default=0.0)
    P = float(penalty_weight) if penalty_weight is not None else D * max_cost + 1.0
    if not math.isfinite(P) or P <= 0:
        raise ValueError("penalty_weight must be finite and strictly positive.")
    bqm = dimod.BinaryQuadraticModel({}, {}, 0.0, dimod.BINARY)
    incoming = {v: [] for v in nodes}
    outgoing = {v: [] for v in nodes}
    for u, v, cost in arcs:
        e = ("e", u, v)
        bqm.add_variable(e, cost)
        incoming[v].append(e)
        outgoing[u].append(e)

    for v in nodes:
        if v == root:
            continue
        ins = incoming[v]
        if v in terminals:
            bqm.offset += P
            for e in ins:
                bqm.add_linear(e, -P)
        else:
            p = ("p", v)
            bqm.add_variable(p, P)
            for e in ins:
                bqm.add_linear(e, P)
                bqm.add_quadratic(p, e, -2 * P)
            for e in outgoing[v]:
                bqm.add_linear(e, P)
                bqm.add_quadratic(p, e, -P)
        for e1, e2 in combinations(ins, 2):
            bqm.add_quadratic(e1, e2, 2 * P)
        for k in range(1, D + 1):
            bqm.add_variable(("z_dw", v, k), 0.0)
        for k in range(1, D):
            z, zp = ("z_dw", v, k), ("z_dw", v, k + 1)
            bqm.add_linear(zp, P)
            bqm.add_quadratic(z, zp, -P)

    for u, v, _ in arcs:
        e = ("e", u, v)
        # k=0, with z[u,0]=1: a selected child must have positive depth.
        bqm.add_linear(e, P)
        bqm.add_quadratic(e, ("z_dw", v, 1), -P)
        if u == root:
            continue
        # k=D, with z[v,D+1]=0: a vertex at depth D has no child.
        bqm.add_quadratic(e, ("z_dw", u, D), P)
        for k in range(1, D):
            z, w = ("z_dw", u, k), ("z_dw", v, k + 1)
            a = ("a_dw", u, v, k)
            bqm.add_variable(a, 2 * P)
            bqm.add_quadratic(e, z, P)
            for x in (e, z, w):
                bqm.add_quadratic(a, x, -P)
    return bqm


def encode_domain_wall_tree(problem, tree_edges, bqm=None):
    """Complete an undirected feasible tree to an exact zero-penalty sample.

    tree_edges contains endpoint pairs (or triples whose third entry is
    ignored).  Rejects missing terminals, cycles and disconnected components.
    This is useful for checking energies and generating feasible initial states.
    """
    nodes, terminals, root, arcs = _problem_data(problem)
    if bqm is None:
        bqm = steiner_to_bqm_domain_wall(problem)
    allowed = {frozenset((u, v)) for u, v, _ in arcs}
    adj = {v: [] for v in nodes}
    seen = set()
    for edge in tree_edges:
        u, v = edge[:2]
        key = frozenset((u, v))
        if key not in allowed or key in seen:
            raise ValueError("Tree edges must be distinct edges of the problem.")
        seen.add(key)
        adj[u].append(v)
        adj[v].append(u)
    depths = {root: 0}
    selected_arcs = set()
    queue = deque([root])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v in depths:
                continue
            depths[v] = depths[u] + 1
            selected_arcs.add((u, v))
            queue.append(v)
    used = {root} | {u for key in seen for u in key}
    if not terminals.issubset(depths) or used != set(depths) or len(seen) != len(depths) - 1:
        raise ValueError("tree_edges must form a connected tree spanning every terminal.")
    sample = {var: 0 for var in bqm.variables}
    for u, v in selected_arcs:
        sample[("e", u, v)] = 1
    for v, depth in depths.items():
        if v == root:
            continue
        if v not in terminals:
            sample[("p", v)] = 1
        for k in range(1, depth + 1):
            sample[("z_dw", v, k)] = 1
    for var in sample:
        if var[0] == "a_dw":
            _, u, v, k = var
            sample[var] = int(
                (u, v) in selected_arcs
                and depths[u] >= k
                and depths[v] >= k + 1
            )
    return sample
