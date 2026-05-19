import dimod
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from typing import List, Tuple, Dict
import math

def steiner_to_bqm_daghan(
    problem: SteinerTree,
    constraint_weight: float,
    version = 2
) -> dimod.BinaryQuadraticModel:
    linear = {}
    quadratic = {}
    vartype = dimod.BINARY
    offset = 0.0

    if version == 1: #base model
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
    if version == 2: #additional constraints
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
        offset += add_H_use(problem, linear, quadratic, constraint_weight, constraint_weight)
    if version == 3: # base model but the capacity constraint is implemented in an alternative way
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap_alternative(problem, linear, quadratic, constraint_weight)
    if version == 4: #version 2 but it is not using the faulty constraint of H_use
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
        #offset += add_H_use(problem, linear, quadratic, constraint_weight, constraint_weight)
    if version == 5:
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, 2*constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
    if version == 6: #additional constraints
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
        offset += add_H_use_correct_version(problem, linear, quadratic, constraint_weight)
    if version == 7: #version 6 + add H_tree, H_node constraint 
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
        offset += add_H_use_correct_version(problem, linear, quadratic, constraint_weight)
        offset += add_H_tree(problem, linear, quadratic, constraint_weight)
        offset += add_H_node(problem, linear, quadratic, constraint_weight)
    if version == 8: #version 6 + add H_root, H_depth, H_activation constraint
        add_H_cost(problem, linear)
        offset += add_H_flow(problem, linear, quadratic, constraint_weight)
        offset += add_H_cap(problem, linear, quadratic, constraint_weight)
        add_H_opp(problem, quadratic, constraint_weight)
        offset += add_H_use_correct_version(problem, linear, quadratic, constraint_weight)
        offset += add_H_root(problem, linear, quadratic, constraint_weight)
        offset += add_H_depth(problem, linear, quadratic, constraint_weight)
        offset += add_H_activation(problem, linear, quadratic, constraint_weight)
    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)

# this is a helper function
# it takes a linear expression in the form of a dictionary {var_name: coefficient} and a constant term
# and adds the squared form of the expression to the linear and quadratic dictionaries with the given constraint
# simple example:
# if expr = {x: 2, y: 3}, constant = 1, constraint_weight = w
# then it adds w*(2x + 3y + 1)^2 to the linear and quadratic dictionaries
def squared_linear_expression(
        expr: Dict[tuple, float],
        constant: float,
        linear: dict,
        quadratic: dict,
        constraint_weight: float
) -> float:
    offset = constraint_weight * (constant ** 2)
    items = list(expr.items())

    for i, (var_name1, coefficient1) in enumerate(items):
        linear[var_name1] = linear.get(var_name1, 0.0) + constraint_weight * (coefficient1 ** 2 + 2 * constant * coefficient1)

        for var_name2, coefficient2 in items[i+1:]:
            key = tuple(sorted((var_name1, var_name2)))
            quadratic[key] = quadratic.get(key, 0.0) + constraint_weight * (2 * coefficient1 * coefficient2)
    
    return offset


# for all edges e,
# cost c_e is added if x_e = 1
# hamiltonian form: c_e x_e
def add_H_cost(
        problem: SteinerTree,
        linear: dict,
) -> None:
    for edge in problem.edges:
        var_name = ("x", edge[0], edge[1])
        linear[var_name] = linear.get(var_name, 0.0) + edge[2]

# for all nodes v,
# ∑f_uv - ∑f_vu = d_v,
# where d_v = 1 if v is a terminal other than the root, d_v = - (|K| - 1) if v is the root, and d_v = 0 otherwise
# hamiltonian form: (∑f_uv - ∑f_vu - d_v)^2
def add_H_flow(
        problem: SteinerTree,
        linear:dict,
        quadratic: dict,
        constraint_weight: float
) -> float:
    root = problem.terminals[0]
    B = math.ceil(math.log2(len(problem.terminals)))
    offset = 0.0

    for v in problem.nodes:
        expr = {}

        for a, b, _ in problem.edges:
            if a == v:
                for bit in range(B):
                    var_name = ("z", a, b, bit)
                    expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)
            elif b == v:
                for bit in range(B):
                    var_name = ("z", b, a, bit)
                    expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        for a, b, _ in problem.edges:
            if b == v:
                for bit in range(B):
                    var_name = ("z", a, b, bit)
                    expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)
            if a == v:
                for bit in range(B):
                    var_name = ("z", b, a, bit)
                    expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        if v == root:
            d_v = len(problem.terminals) - 1
        elif v in problem.terminals:
            d_v = -1
        else:
            d_v = 0

        offset += squared_linear_expression(
            expr=expr,
            constant=-d_v,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    return offset

# for all edges e,
# f_uv + f_vu <= (|K| - 1) x_e
# equivalent to: f_uv + f_vu - (|K| - 1) x_e - s_e = 0, where s_e is a slack variable
# hamiltonian form: (f_uv + f_vu - (|K| - 1) x_e - s_e)^2
def add_H_cap(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float
) -> float:
    B = math.ceil(math.log2(len(problem.terminals)))
    offset = 0.0
    max_flow = len(problem.terminals) - 1

    for a, b, _ in problem.edges:
        expr = {}

        # f_uv
        for bit in range(B):
            var_name = ("z", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # f_vu
        for bit in range(B):
            var_name = ("z", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # slack s_e
        for bit in range(B):
            var_name = ("s", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # -( |K| - 1 ) x_e
        x_var = ("x", a, b)
        expr[x_var] = expr.get(x_var, 0.0) - max_flow

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    return offset

#performs poorly compared to the initial version
def add_H_cap_alternative(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float
) -> float:
    B = math.ceil(math.log2(len(problem.terminals)))
    offset = 0.0

    for a, b, _ in problem.edges:
        x_var = ("x", a, b)

        # (f_uv + f_vu)(1 - x_e)
        # = (sum bits of f_uv + sum bits of f_vu)
        #   - x_e(sum bits of f_uv + sum bits of f_vu)

        for bit in range(B):
            coeff = 2 ** bit

            z_uv = ("z", a, b, bit)
            z_vu = ("z", b, a, bit)

            # + f_uv + f_vu
            linear[z_uv] = linear.get(z_uv, 0.0) + constraint_weight * coeff
            linear[z_vu] = linear.get(z_vu, 0.0) + constraint_weight * coeff

            # - x_e f_uv
            key1 = tuple(sorted((x_var, z_uv)))
            quadratic[key1] = quadratic.get(key1, 0.0) - constraint_weight * coeff

            # - x_e f_vu
            key2 = tuple(sorted((x_var, z_vu)))
            quadratic[key2] = quadratic.get(key2, 0.0) - constraint_weight * coeff

    return offset


# for all edges e,
# f_uv * f_vu = 0
# hamiltonian form: f_uv * f_vu
def add_H_opp(
        problem: SteinerTree,
        quadratic: dict,
        constraint_weight: float
) -> None:
    B = math.ceil(math.log2(len(problem.terminals)))

    for a, b, _ in problem.edges:
        for bit1 in range(B):
            var_name1 = ("z", a, b, bit1)
            coeff1 = 2 ** bit1

            for bit2 in range(B):
                var_name2 = ("z", b, a, bit2)
                coeff2 = 2 ** bit2

                key = tuple(sorted((var_name1, var_name2)))
                quadratic[key] = quadratic.get(key, 0.0) + constraint_weight * coeff1 * coeff2


# ay bunu silmedim ama doğru mu yanlış mı okumaya da çok üşeniyorum öf
# neyse altta doğrusu var galiba onu kullanıyorum
def add_H_use(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight_1: float,
        constraint_weight_2: float
) -> float:
    B = math.ceil(math.log2(len(problem.terminals)))
    max_flow = len(problem.terminals) - 1
    offset = 0.0

    for a, b, _ in problem.edges:
        expr = {}

        # f_uv
        for bit in range(B):
            var_name = ("z", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # f_vu
        for bit in range(B):
            var_name = ("z", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # slack t_e
        for bit in range(B):
            var_name = ("t", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # -( |K| - 1 ) y_e
        y_var = ("y", a, b)
        expr[y_var] = expr.get(y_var, 0.0) - max_flow

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight_1
        )

        # x_e (1 - y_e) = x_e - x_e y_e
        x_var = ("x", a, b)

        linear[x_var] = linear.get(x_var, 0.0) + constraint_weight_2

        key = tuple(sorted((x_var, y_var)))
        quadratic[key] = quadratic.get(key, 0.0) - constraint_weight_2

    return offset

# for all edges e,
# x_e <= f_uv + f_vu
# equivalent to: f_uv + f_vu - x_e - t_e = 0, where t_e is a slack variable
# hamiltonian form: (f_uv + f_vu - x_e - t_e)^2
def add_H_use_correct_version(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight_1: float,
) -> float:
    B = math.ceil(math.log2(len(problem.terminals)))
    offset = 0.0

    for a, b, _ in problem.edges:
        expr = {}

        # f_uv
        for bit in range(B):
            var_name = ("z", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # f_vu
        for bit in range(B):
            var_name = ("z", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # -x_e
        x_var = ("x", a, b)
        expr[x_var] = expr.get(x_var, 0.0) - 1.0

        # -t_e   where t_e is the slack for x_e <= f_uv + f_vu
        # so F_e - x_e - t_e = 0
        for bit in range(B):
            var_name = ("t", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight_1
        )

    return offset

def add_H_node(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    offset = 0.0

    for v in problem.nodes:
        expr = {}

        # number of incident edges on v
        num_incident_edges = len([1 for a, b, _ in problem.edges if a == v or b == v])
        B_v = math.ceil(math.log2(num_incident_edges + 1))

        # q_v
        for bit in range(B_v):
            var_name = ("q", v, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # y_v
        y_var = ("y", v)
        expr[y_var] = expr.get(y_var, 0.0) + 1.0

        # -x_e for each edge e incident on v
        for a, b, _ in problem.edges:
            if a == v or b == v:
                x_var = ("x", a, b)
                expr[x_var] = expr.get(x_var, 0.0) - 1.0

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    return offset
    
def add_H_tree(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    offset = 0.0

    expr = {}

    # number of used edges in the solution
    for a, b, _ in problem.edges:
        x_var = ("x", a, b)
        expr[x_var] = expr.get(x_var, 0.0) + 1.0

    # - number of used nodes in the solution
    for v in problem.nodes:
        y_var = ("y", v)
        expr[y_var] = expr.get(y_var, 0.0) - 1.0

    offset += squared_linear_expression(
        expr=expr,
        constant=1.0,
        linear=linear,
        quadratic=quadratic,
        constraint_weight=constraint_weight
    )

    return offset

# for all edges e,
# f_uv - M*a_uv <= 0
# f_vu - M*a_vu <= 0
# and
# a_uv <= f_uv
# a_vu <= f_vu
# where M = |K| - 1
# hamiltonian form: (f_uv - M*a_uv + r_uv1)^2 + (f_vu - M*a_vu + r_vu1)^2 + (a_uv - f_uv + r_uv2)^2 + (a_vu - f_vu + r_vu2)^2
def add_H_activation(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    offset = 0.0
    M = len(problem.terminals) - 1
    B = math.ceil(math.log2(len(problem.terminals)))

    for a, b, _ in problem.edges:
        expr = {}
        
        # f_uv - M*a_uv <= 0
        # f_uv
        for bit in range(B):
            var_name = ("z", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - M*a_uv
        a_var = ("a", a, b)
        expr[a_var] = expr.get(a_var, 0.0) - M

        # +r_uv1
        for bit in range(B):
            var_name = ("r", a, b, 1, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

        expr = {}

        # f_vu - M*a_vu <= 0
        # f_vu
        for bit in range(B):
            var_name = ("z", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - M*a_vu
        a_var = ("a", b, a)
        expr[a_var] = expr.get(a_var, 0.0) - M

        # +r_vu1
        for bit in range(B):
            var_name = ("r", b, a, 1, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    for a, b, _ in problem.edges:
        expr = {}

        # a_uv <= f_uv
        # f_uv
        for bit in range(B):
            var_name = ("z", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - a_uv
        a_var = ("a", a, b)
        expr[a_var] = expr.get(a_var, 0.0) - 1

        # -r_uv2
        for bit in range(B):
            var_name = ("r", a, b, 2, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

        expr = {}

        # a_vu <= f_vu
        # f_vu
        for bit in range(B):
            var_name = ("z", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - a_vu
        a_var = ("a", b, a)
        expr[a_var] = expr.get(a_var, 0.0) - 1

        # -r_vu2
        for bit in range(B):
            var_name = ("r", b, a, 2, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=0.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    return offset

# for the root node r,
# o_r = 0
# hamiltonian form: o_r^2
def add_H_root(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    offset = 0.0
    root = problem.terminals[0]
    B_o = math.ceil(math.log2(len(problem.nodes)))

    expr = {}

    # o_r
    for bit in range(B_o):
        var_name = ("o", root, bit)
        expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

    offset += squared_linear_expression(
        expr=expr,
        constant=0.0,
        linear=linear,
        quadratic=quadratic,
        constraint_weight=constraint_weight
    )

    return offset

# for all edges e = (u, v),
# o_v - o_u - 1 + N(1 - a_uv) >= 0
# o_u - o_v - 1 + N(1 - a_vu) >= 0
# where N = |V|
# hamiltonian form: (o_v - o_u - 1 + N(1 - a_uv) - g_uv)^2 + (o_u - o_v - 1 + N(1 - a_vu) - g_vu)^2,
# where g_uv and g_vu are slack variables for the respective constraints
def add_H_depth(
        problem: SteinerTree,
        linear: dict,
        quadratic: dict,
        constraint_weight: float,
) -> float:
    offset = 0.0
    B_o = math.ceil(math.log2(len(problem.nodes)))
    N = len(problem.nodes)
    B_g = math.ceil(math.log2(2 * N - 1))

    for a, b, _ in problem.edges:
        expr = {}

        # o_v - o_u - 1 + N(1 - a_uv) >= 0
        # o_v
        for bit in range(B_o):
            var_name = ("o", b, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - o_u
        for bit in range(B_o):
            var_name = ("o", a, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        # N(1 - a_uv) 
        a_var = ("a", a, b)
        expr[a_var] = expr.get(a_var, 0.0) - N

        # -g_uv
        for bit in range(B_g):
            var_name = ("g", a, b, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=N-1.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

        expr = {}

        # o_u - o_v - 1 + N(1 - a_vu) >= 0
        # o_u
        for bit in range(B_o):
            var_name = ("o", a, bit)
            expr[var_name] = expr.get(var_name, 0.0) + (2 ** bit)

        # - o_v
        for bit in range(B_o):
            var_name = ("o", b, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        # N(1 - a_vu)
        a_var = ("a", b, a)
        expr[a_var] = expr.get(a_var, 0.0) - N

        # -g_vu
        for bit in range(B_g):
            var_name = ("g", b, a, bit)
            expr[var_name] = expr.get(var_name, 0.0) - (2 ** bit)

        offset += squared_linear_expression(
            expr=expr,
            constant=N-1.0,
            linear=linear,
            quadratic=quadratic,
            constraint_weight=constraint_weight
        )

    return offset

