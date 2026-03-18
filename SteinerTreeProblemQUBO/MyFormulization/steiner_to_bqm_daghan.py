import dimod
from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from typing import List, Tuple, Dict
import math

def steiner_to_bqm_daghan(
    problem: SteinerTree,
    constraint_weight: float,
) -> dimod.BinaryQuadraticModel:
    linear = {}
    quadratic = {}
    vartype = dimod.BINARY
    offset = 0.0

    add_H_cost(problem, linear)
    offset += add_H_flow(problem, linear, quadratic, constraint_weight)
    offset += add_H_cap(problem, linear, quadratic, constraint_weight)

    return dimod.BinaryQuadraticModel(linear, quadratic, offset, vartype)


def add_H_cost(
        problem: SteinerTree,
        linear: dict,
) -> None:
    for edge in problem.edges:
        var_name = ("x", edge[0], edge[1])
        linear[var_name] = linear.get(var_name, 0.0) + edge[2]

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

