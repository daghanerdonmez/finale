import math
from typing import Dict, Hashable, List, Optional, Tuple

import dimod
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree


Var = tuple


def steiner_to_bqm_hybrid(
        problem: SteinerTree,
        constraint_weight: float,
) -> dimod.BinaryQuadraticModel:
    """
    Plain hybrid Steiner Tree QUBO formulation.

    Idea:
        - Use directed edge variables e_{u,v}, as in the ordering-based model.
        - Replace global ordering variables x_{u,v} with binary depth variables o_v.
        - Use non-terminal usage variables p_v.
        - This is the plain hybrid model, NOT the sequential-counter version.

    Variables:
        e_{u,v}: directed selected edge u -> v
        p_v: non-terminal v is used
        o_{v,i}: binary expansion bits of depth o_v
        g_{u,v,i}: slack bits for the conditional depth inequality
    """
    linear: Dict[Var, float] = {}
    quadratic: Dict[Tuple[Var, Var], float] = {}
    offset = 0.0

    ctx = _HybridContext(problem)

    _initialize_variables(ctx, linear)

    add_H_cost(problem, ctx, linear)

    offset += add_H_terminal_parent(ctx, linear, quadratic, constraint_weight)
    offset += add_H_nonterminal_parent(ctx, linear, quadratic, constraint_weight)
    add_H_no_fake_root(ctx, linear, quadratic, constraint_weight)
    offset += add_H_root_depth(ctx, linear, quadratic, constraint_weight)
    offset += add_H_depth(ctx, linear, quadratic, constraint_weight)

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)


class _HybridContext:
    def __init__(self, problem: SteinerTree):
        self.problem = problem
        self.nodes = list(problem.nodes)
        self.terminals = set(problem.terminals)

        if len(self.terminals) < 2:
            raise ValueError("Steiner Tree formulation requires at least two terminals.")

        self.root = problem.terminals[0]

        if self.root not in self.nodes:
            raise ValueError("Root terminal is not in problem.nodes.")

        self.node_index = {v: i for i, v in enumerate(self.nodes)}

        if len(self.node_index) != len(self.nodes):
            raise ValueError("problem.nodes contains duplicate nodes.")

        self.n = len(self.nodes)

        # Number of bits for depth variables.
        self.B_o = math.ceil(math.log2(self.n))

        # Depth values are encoded from 0 to R.
        self.R = (2 ** self.B_o) - 1

        # Big-M chosen with respect to the encoded depth range.
        self.M = self.R + 1

        # Slack for inactive arcs may need to reach 2R.
        self.B_g = math.ceil(math.log2(2 * self.R + 1))

        self.edge_set = set()
        for a, b, _ in problem.edges:
            if a == b:
                raise ValueError("Self-loops are not supported by this formulation.")

            if a not in self.node_index or b not in self.node_index:
                raise ValueError(f"Edge ({a}, {b}) contains a node not in problem.nodes.")

            key = self.edge_key(a, b)
            if key in self.edge_set:
                raise ValueError("Parallel/duplicate undirected edges are not supported.")

            self.edge_set.add(key)

    def edge_key(self, a: Hashable, b: Hashable) -> Tuple[Hashable, Hashable]:
        if self.node_index[a] < self.node_index[b]:
            return a, b
        return b, a

    def arc_var(self, u: Hashable, v: Hashable) -> Optional[Var]:
        """
        Directed edge variable e_{u,v}.

        We do not create variables pointing into the root.
        For root-incident edges, only root -> other is allowed.
        """
        if v == self.root:
            return None
        return ("e", u, v)

    def all_arcs(self) -> List[Tuple[Hashable, Hashable, Var]]:
        arcs = []

        for a, b, _ in self.problem.edges:
            if a == self.root:
                var = self.arc_var(a, b)
                if var is not None:
                    arcs.append((a, b, var))

            elif b == self.root:
                var = self.arc_var(b, a)
                if var is not None:
                    arcs.append((b, a, var))

            else:
                var_ab = self.arc_var(a, b)
                var_ba = self.arc_var(b, a)
                arcs.append((a, b, var_ab))
                arcs.append((b, a, var_ba))

        return arcs

    def incoming_vars(self, v: Hashable) -> List[Var]:
        result = []

        for a, b, _ in self.problem.edges:
            if b == v:
                var = self.arc_var(a, b)
                if var is not None:
                    result.append(var)

            elif a == v:
                var = self.arc_var(b, a)
                if var is not None:
                    result.append(var)

        return result

    def outgoing_vars(self, v: Hashable) -> List[Var]:
        result = []

        for a, b, _ in self.problem.edges:
            if a == v:
                var = self.arc_var(a, b)
                if var is not None:
                    result.append(var)

            elif b == v:
                var = self.arc_var(b, a)
                if var is not None:
                    result.append(var)

        return result


def _quad_key(var1: Var, var2: Var) -> Tuple[Var, Var]:
    """
    Stable key for quadratic dictionary.

    Uses repr so node labels can be ints, strings, or mixed hashables.
    """
    if repr(var1) <= repr(var2):
        return var1, var2
    return var2, var1


def add_linear(linear: dict, var: Var, coeff: float) -> None:
    linear[var] = linear.get(var, 0.0) + coeff


def add_quadratic(linear: dict, quadratic: dict, var1: Var, var2: Var, coeff: float) -> None:
    if var1 == var2:
        add_linear(linear, var1, coeff)
        return

    key = _quad_key(var1, var2)
    quadratic[key] = quadratic.get(key, 0.0) + coeff


def squared_linear_expression(
        expr: Dict[Var, float],
        constant: float,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    """
    Adds constraint_weight * (sum_i c_i x_i + constant)^2.
    """
    offset = constraint_weight * (constant ** 2)
    items = list(expr.items())

    for i, (var1, coeff1) in enumerate(items):
        add_linear(
            linear,
            var1,
            constraint_weight * (coeff1 ** 2 + 2.0 * constant * coeff1),
        )

        for var2, coeff2 in items[i + 1:]:
            add_quadratic(
                linear,
                quadratic,
                var1,
                var2,
                constraint_weight * 2.0 * coeff1 * coeff2,
            )

    return offset


def _initialize_variables(ctx: _HybridContext, linear: dict) -> None:
    # Directed edge variables.
    for _, _, var in ctx.all_arcs():
        add_linear(linear, var, 0.0)

    # Non-terminal usage variables.
    for v in ctx.nodes:
        if v not in ctx.terminals:
            add_linear(linear, ("p", v), 0.0)

    # Depth variables.
    for v in ctx.nodes:
        for bit in range(ctx.B_o):
            add_linear(linear, ("o", v, bit), 0.0)

    # Depth slack variables.
    for u, v, _ in ctx.all_arcs():
        for bit in range(ctx.B_g):
            add_linear(linear, ("g", u, v, bit), 0.0)


def add_H_cost(
        problem: SteinerTree,
        ctx: _HybridContext,
        linear: dict,
) -> None:
    """
    H_cost = sum_{e={u,v}} c_e (e_{u,v} + e_{v,u})

    For root-incident edges, only the root-outgoing direction exists.
    """
    for a, b, w in problem.edges:
        cost = float(w)

        if a == ctx.root:
            add_linear(linear, ("e", a, b), cost)

        elif b == ctx.root:
            add_linear(linear, ("e", b, a), cost)

        else:
            add_linear(linear, ("e", a, b), cost)
            add_linear(linear, ("e", b, a), cost)


def add_H_terminal_parent(
        ctx: _HybridContext,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    """
    For every terminal v != root:

        sum_{u:(u,v) in A} e_{u,v} = 1

    Hamiltonian:

        H_terminal_parent =
            sum_{v in T \\ {root}} (1 - sum_u e_{u,v})^2
    """
    offset = 0.0

    for v in ctx.terminals:
        if v == ctx.root:
            continue

        expr = {}

        for var in ctx.incoming_vars(v):
            expr[var] = expr.get(var, 0.0) - 1.0

        offset += squared_linear_expression(
            expr=expr,
            constant=1.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight,
        )

    return offset


def add_H_nonterminal_parent(
        ctx: _HybridContext,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    """
    For every non-terminal v:

        p_v = sum_{u:(u,v) in A} e_{u,v}

    Since p_v is binary, this means a non-terminal either has no parent
    and is unused, or has exactly one parent and is used.

    Hamiltonian:

        H_nonterminal_parent =
            sum_{v in V \\ T} (p_v - sum_u e_{u,v})^2
    """
    offset = 0.0

    for v in ctx.nodes:
        if v in ctx.terminals:
            continue

        expr = {("p", v): 1.0}

        for var in ctx.incoming_vars(v):
            expr[var] = expr.get(var, 0.0) - 1.0

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight,
        )

    return offset


def add_H_no_fake_root(
        ctx: _HybridContext,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> None:
    """
    For every non-terminal v and outgoing arc v -> w:

        e_{v,w} <= p_v

    Hamiltonian:

        H_no_fake_root =
            sum_{v in V \\ T} sum_{w:(v,w) in A} e_{v,w}(1 - p_v)

    This prevents an unused non-terminal from starting a disconnected component.
    """
    for v in ctx.nodes:
        if v in ctx.terminals:
            continue

        p_var = ("p", v)

        for out_var in ctx.outgoing_vars(v):
            add_linear(linear, out_var, constraint_weight)
            add_quadratic(linear, quadratic, out_var, p_var, -constraint_weight)


def add_H_root_depth(
        ctx: _HybridContext,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    """
    Root depth constraint:

        o_root = 0

    Hamiltonian:

        H_root_depth = o_root^2
    """
    expr = {}

    for bit in range(ctx.B_o):
        expr[("o", ctx.root, bit)] = 2 ** bit

    return squared_linear_expression(
        expr=expr,
        constant=0.0,
        linear=linear,
        quadratic=quadratic,
        constraint_weight=constraint_weight,
    )


def add_H_depth(
        ctx: _HybridContext,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    """
    Conditional depth/acyclicity constraint.

    For every directed edge variable e_{u,v}:

        e_{u,v} = 1  =>  o_v >= o_u + 1

    Big-M inequality:

        o_v - o_u - 1 + M(1 - e_{u,v}) >= 0

    With slack g_{u,v} >= 0:

        o_v - o_u - 1 + M(1 - e_{u,v}) - g_{u,v} = 0

    Hamiltonian:

        H_depth =
            sum_{(u,v) in A}
            (o_v - o_u - 1 + M(1 - e_{u,v}) - g_{u,v})^2
    """
    offset = 0.0

    for u, v, e_var in ctx.all_arcs():
        expr = {}

        # + o_v
        for bit in range(ctx.B_o):
            expr[("o", v, bit)] = expr.get(("o", v, bit), 0.0) + (2 ** bit)

        # - o_u
        for bit in range(ctx.B_o):
            expr[("o", u, bit)] = expr.get(("o", u, bit), 0.0) - (2 ** bit)

        # - M e_{u,v}
        expr[e_var] = expr.get(e_var, 0.0) - ctx.M

        # - g_{u,v}
        for bit in range(ctx.B_g):
            expr[("g", u, v, bit)] = expr.get(("g", u, v, bit), 0.0) - (2 ** bit)

        # constant: -1 + M
        offset += squared_linear_expression(
            expr=expr,
            constant=ctx.M - 1.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight,
        )

    return offset
