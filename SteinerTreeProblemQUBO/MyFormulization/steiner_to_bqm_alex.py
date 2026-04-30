import dimod
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from typing import Dict, Hashable, List, Tuple, Optional


Var = tuple


def steiner_to_bqm_ordering(
    problem: SteinerTree,
    penalty_weight: Optional[float] = None,
) -> dimod.BinaryQuadraticModel:
    """
    Improved ordering-based Steiner Tree QUBO formulation.

    This implements the formulation:

        H_I(e, x) = P_I * F_I(e, x) + O_I(e)

    where

        F_I = F_I1 + F_I2 + F_I3 + |V| F_I4 + F_I5.

    Variables
    ---------
    e_{u,v}:
        Directed edge variable. For edges not incident to the root, both
        e_{u,v} and e_{v,u} exist. For edges incident to the root v0, only
        e_{v0,u} exists.

    x_{u,v}:
        Ordering variable for non-root vertices u, v with u < v.
        x_{u,v} = 1 means u comes before v in the chosen ordering.

    Parameters
    ----------
    penalty_weight:
        The penalty P_I. If None, uses the paper's safe value
        P_I = (|V| - 1) * max_edge_cost + 1.
    """
    linear: Dict[Var, float] = {}
    quadratic: Dict[Tuple[Var, Var], float] = {}
    offset = 0.0

    root = problem.terminals[0]
    terminals = set(problem.terminals)
    nodes = list(problem.nodes)
    n = len(nodes)

    if len(terminals) < 2:
        raise ValueError("Steiner Tree formulation requires at least two terminals.")

    if root not in nodes:
        raise ValueError("Root terminal is not in problem.nodes.")

    max_cost = max(float(w) for _, _, w in problem.edges) if problem.edges else 0.0
    P = float(penalty_weight) if penalty_weight is not None else (n - 1) * max_cost + 1.0

    ctx = _Context(problem=problem, root=root, terminals=terminals, nodes=nodes)

    _initialize_variables(ctx, linear)

    add_OI(problem, ctx, linear)
    add_FI1(ctx, linear, quadratic, P)
    add_FI2(problem, ctx, linear, quadratic, P)
    offset += add_FI3(ctx, linear, quadratic, P)
    add_FI4(ctx, quadratic, P * n)
    add_FI5(ctx, linear, quadratic, P)

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, dimod.BINARY)


class _Context:
    def __init__(self, problem: SteinerTree, root: Hashable, terminals: set, nodes: List[Hashable]):
        self.problem = problem
        self.root = root
        self.terminals = terminals
        self.nodes = nodes
        self.node_index = {v: i for i, v in enumerate(nodes)}

        if len(self.node_index) != len(nodes):
            raise ValueError("problem.nodes contains duplicate nodes.")

        self.non_root_nodes = [v for v in nodes if v != root]

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
        return (a, b) if self.node_index[a] < self.node_index[b] else (b, a)

    def order_var(self, u: Hashable, v: Hashable) -> Var:
        """
        x_{u,v} exists only for non-root u,v with u < v according to the
        fixed node ordering.
        """
        if u == self.root or v == self.root:
            raise ValueError("Ordering variables are not defined for the root.")

        if self.node_index[u] < self.node_index[v]:
            return ("x_order", u, v)
        return ("x_order", v, u)

    def arc_var(self, u: Hashable, v: Hashable) -> Optional[Var]:
        """
        Directed edge variable e_{u,v}.

        No variable points into the root. If an undirected edge is incident
        to the root, only root -> other is allowed.
        """
        if v == self.root:
            return None
        if u == self.root:
            return ("e", self.root, v)
        return ("e", u, v)

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


def _quad_key(v1: Var, v2: Var) -> Tuple[Var, Var]:
    if v1 == v2:
        raise ValueError("Use add_linear for identical variables.")
    return (v1, v2) if repr(v1) < repr(v2) else (v2, v1)


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


def _initialize_variables(ctx: _Context, linear: dict) -> None:
    """
    Adds all variables with zero bias so they appear in the BQM even if
    some variable is only weakly used in a tiny graph.
    """
    # Directed edge variables.
    for a, b, _ in ctx.problem.edges:
        if a == ctx.root:
            add_linear(linear, ("e", ctx.root, b), 0.0)
        elif b == ctx.root:
            add_linear(linear, ("e", ctx.root, a), 0.0)
        else:
            add_linear(linear, ("e", a, b), 0.0)
            add_linear(linear, ("e", b, a), 0.0)

    # Ordering variables between all pairs of non-root vertices.
    non_root = ctx.non_root_nodes
    for i in range(len(non_root)):
        for j in range(i + 1, len(non_root)):
            u = non_root[i]
            v = non_root[j]
            add_linear(linear, ctx.order_var(u, v), 0.0)


def add_OI(
    problem: SteinerTree,
    ctx: _Context,
    linear: dict,
) -> None:
    """
    Objective term:

        O_I(e) =
            sum_{(u,v) in E, u,v != root} c(u,v)(e_{u,v}+e_{v,u})
            + sum_{(root,u) in E} c(root,u)e_{root,u}.
    """
    for a, b, w in problem.edges:
        cost = float(w)

        if a == ctx.root:
            add_linear(linear, ("e", ctx.root, b), cost)
        elif b == ctx.root:
            add_linear(linear, ("e", ctx.root, a), cost)
        else:
            add_linear(linear, ("e", a, b), cost)
            add_linear(linear, ("e", b, a), cost)


def add_FI1(
    ctx: _Context,
    linear: dict,
    quadratic: dict,
    constraint_weight: float,
) -> None:
    """
    Ordering consistency term:

        F_I1(x) =
            sum_{u<v<w, u,v,w != root}
            x_{u,w} + x_{u,v}x_{v,w}
            - x_{u,v}x_{u,w}
            - x_{u,w}x_{v,w}.

    This makes the pairwise ordering variables transitive.
    """
    non_root = sorted(ctx.non_root_nodes, key=lambda v: ctx.node_index[v])

    for i in range(len(non_root)):
        for j in range(i + 1, len(non_root)):
            for k in range(j + 1, len(non_root)):
                u = non_root[i]
                v = non_root[j]
                w = non_root[k]

                x_uv = ctx.order_var(u, v)
                x_uw = ctx.order_var(u, w)
                x_vw = ctx.order_var(v, w)

                add_linear(linear, x_uw, constraint_weight)
                add_quadratic(linear, quadratic, x_uv, x_vw, constraint_weight)
                add_quadratic(linear, quadratic, x_uv, x_uw, -constraint_weight)
                add_quadratic(linear, quadratic, x_uw, x_vw, -constraint_weight)


def add_FI2(
    problem: SteinerTree,
    ctx: _Context,
    linear: dict,
    quadratic: dict,
    constraint_weight: float,
) -> None:
    """
    Edge-order consistency term:

        F_I2(e,x) =
            sum_{(u,v) in E, u<v, u,v != root}
            e_{u,v}(1-x_{u,v}) + e_{v,u}x_{u,v}.

    If x_{u,v}=1, the allowed direction is u -> v.
    If x_{u,v}=0, the allowed direction is v -> u.
    """
    for a, b, _ in problem.edges:
        if a == ctx.root or b == ctx.root:
            continue

        if ctx.node_index[a] < ctx.node_index[b]:
            u, v = a, b
        else:
            u, v = b, a

        x_uv = ctx.order_var(u, v)
        e_uv = ("e", u, v)
        e_vu = ("e", v, u)

        # e_uv(1 - x_uv)
        add_linear(linear, e_uv, constraint_weight)
        add_quadratic(linear, quadratic, e_uv, x_uv, -constraint_weight)

        # e_vu x_uv
        add_quadratic(linear, quadratic, e_vu, x_uv, constraint_weight)


def add_FI3(
    ctx: _Context,
    linear: dict,
    quadratic: dict,
    constraint_weight: float,
) -> float:
    """
    Terminal incoming-edge term:

        F_I3(e) =
            sum_{v in U \\ {root}}
            (1 - sum_{(u,v) in E} e_{u,v})^2.

    Every terminal except the root must have exactly one incoming arc.
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


def add_FI4(
    ctx: _Context,
    quadratic: dict,
    constraint_weight: float,
) -> None:
    """
    Non-terminal multiple-incoming penalty:

        F_I4(e) =
            sum_{v in V \\ U}
            sum_{distinct incoming arc pairs into v}
            e_{u,v} e_{w,v}.

    This enforces that every non-terminal has at most one incoming arc.

    Note:
        The paper writes the inner summation compactly. Here we count each
        unordered pair once. Counting twice would only rescale this penalty.
    """
    dummy_linear = {}

    for v in ctx.nodes:
        if v in ctx.terminals:
            continue

        incoming = ctx.incoming_vars(v)

        for i in range(len(incoming)):
            for j in range(i + 1, len(incoming)):
                add_quadratic(
                    dummy_linear,
                    quadratic,
                    incoming[i],
                    incoming[j],
                    constraint_weight,
                )


def add_FI5(
    ctx: _Context,
    linear: dict,
    quadratic: dict,
    constraint_weight: float,
) -> None:
    """
    Non-terminal fake-root penalty:

        F_I5(e) =
            sum_{v in V \\ U}
            (1 - sum incoming_to_v) (sum outgoing_from_v).

    If a non-terminal has outgoing arcs but no incoming arc, it acts like a
    disconnected root. This term penalizes that.

    Expansion:
        (1 - IN) OUT = OUT - IN*OUT.
    """
    for v in ctx.nodes:
        if v in ctx.terminals:
            continue

        incoming = ctx.incoming_vars(v)
        outgoing = ctx.outgoing_vars(v)

        # + OUT
        for out_var in outgoing:
            add_linear(linear, out_var, constraint_weight)

        # - IN * OUT
        for in_var in incoming:
            for out_var in outgoing:
                add_quadratic(linear, quadratic, in_var, out_var, -constraint_weight)
