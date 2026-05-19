from __future__ import annotations

import csv
import math
import sys
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import dimod
import numpy as np
import openjij as oj
from tqdm import tqdm


THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SteinerTreeProblemQUBO.SteinerTree import SteinerTree
from SteinerTreeProblemQUBO.AlexFowler import steiner_to_bqm_alex as alex_mod
from SteinerTreeProblemQUBO.MyFormulization import steiner_to_bqm_hybrid as hybrid_mod
from SteinerTreeProblemQUBO.exact_solver import solve as solve_exact
from SteinerTreeProblemQUBO.random_problem_generator import (
    generate_erdos_renyi_steiner_tree,
    generate_geometric_steiner_tree,
    generate_grid_steiner_tree,
    generate_random_steiner_tree,
)
from SteinerTreeProblemQUBO.sparsity_problem_generator import (
    generate_sparsity_steiner_tree,
)


# Edit these defaults or import this module and call the helpers directly.
PROBLEM_SPEC = {
    "generator": "sparsity",
    "params": {
        "node_count": 10,
        "terminal_count": 4,
        "extra_edge_probability": 0.3,
        "weight_range": (1, 100),
        "seed": 0,
    },
}
BASE_CONSTRAINT_WEIGHT = None
TOP_K_SAMPLES = 10
NUM_READS = 100
PENALTY_GRID = [1, 3, 10, 30, 100, 300, 1000]
SQA_KWARGS = {
    "num_sweeps": 4000,
    "trotter": 16,
}
OUTPUT_DIR = REPO_ROOT / "SteinerTreeProblemQUBO" / "logs"
TRY_GUROBI_FIRST = True
EXACT_SOLVER_NODE_LIMIT = 16


GENERATORS = {
    "sparsity": generate_sparsity_steiner_tree,
    "random_connected": generate_random_steiner_tree,
    "erdos_renyi": generate_erdos_renyi_steiner_tree,
    "geometric": generate_geometric_steiner_tree,
    "grid": generate_grid_steiner_tree,
}

HYBRID_TERM_ORDER = [
    "H_cost",
    "H_terminal_parent",
    "H_nonterminal_parent",
    "H_no_fake_root",
    "H_root_depth",
    "H_depth",
]

ALEX_TERM_ORDER = [
    "O_I",
    "F_I1",
    "F_I2",
    "F_I3",
    "F_I4",
    "F_I5",
]

HYBRID_ABLATIONS = OrderedDict(
    [
        ("cost_only", ["H_cost"]),
        ("plus_terminal_parent", ["H_cost", "H_terminal_parent"]),
        (
            "plus_nonterminal_parent",
            ["H_cost", "H_terminal_parent", "H_nonterminal_parent"],
        ),
        (
            "plus_no_fake_root",
            [
                "H_cost",
                "H_terminal_parent",
                "H_nonterminal_parent",
                "H_no_fake_root",
            ],
        ),
        (
            "plus_root_depth",
            [
                "H_cost",
                "H_terminal_parent",
                "H_nonterminal_parent",
                "H_no_fake_root",
                "H_root_depth",
            ],
        ),
        ("full_hybrid", HYBRID_TERM_ORDER),
    ]
)


@dataclass
class ModelBundle:
    model_name: str
    constraint_weight: float
    enabled_terms: List[str]
    full_bqm: dimod.BinaryQuadraticModel
    labeled_bqm: dimod.BinaryQuadraticModel
    term_bqms: Dict[str, dimod.BinaryQuadraticModel]
    labeled_term_bqms: Dict[str, dimod.BinaryQuadraticModel]


def _format_variable_label(variable) -> str:
    if isinstance(variable, tuple):
        return "::".join(str(part) for part in variable)
    return str(variable)


def _relabel_bqm(bqm: dimod.BinaryQuadraticModel) -> dimod.BinaryQuadraticModel:
    relabeling = {var: _format_variable_label(var) for var in bqm.variables}
    return bqm.relabel_variables(relabeling, inplace=False)


def _percentile_summary(values: Sequence[float], prefix: str) -> Dict[str, Optional[float]]:
    if not values:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_p25": None,
            f"{prefix}_p50": None,
            f"{prefix}_p75": None,
            f"{prefix}_p90": None,
            f"{prefix}_p95": None,
            f"{prefix}_p99": None,
        }

    array = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_p25": float(np.percentile(array, 25)),
        f"{prefix}_p50": float(np.percentile(array, 50)),
        f"{prefix}_p75": float(np.percentile(array, 75)),
        f"{prefix}_p90": float(np.percentile(array, 90)),
        f"{prefix}_p95": float(np.percentile(array, 95)),
        f"{prefix}_p99": float(np.percentile(array, 99)),
    }


def default_constraint_weight(problem: SteinerTree) -> float:
    return 5 * max(float(weight) for _, _, weight in problem.edges) + 1.0


def make_problem(problem_spec: Mapping[str, object]) -> SteinerTree:
    generator_name = problem_spec["generator"]
    params = dict(problem_spec["params"])
    return GENERATORS[generator_name](**params)


def _combine_bqms(
    term_bqms: Mapping[str, dimod.BinaryQuadraticModel],
    enabled_terms: Iterable[str],
) -> dimod.BinaryQuadraticModel:
    linear: Dict[Tuple[str, ...], float] = defaultdict(float)
    quadratic: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], float] = defaultdict(float)
    offset = 0.0

    for term_name in enabled_terms:
        bqm = term_bqms[term_name]
        for var, bias in bqm.linear.items():
            linear[var] += float(bias)
        for key, bias in bqm.quadratic.items():
            quadratic[key] += float(bias)
        offset += float(bqm.offset)

    return dimod.BinaryQuadraticModel(dict(linear), dict(quadratic), offset, dimod.BINARY)


def build_hybrid_term_bqms(
    problem: SteinerTree,
    constraint_weight: float,
) -> Dict[str, dimod.BinaryQuadraticModel]:
    ctx = hybrid_mod._HybridContext(problem)
    n = len(problem.nodes)
    term_bqms: Dict[str, dimod.BinaryQuadraticModel] = OrderedDict()

    for term_name in HYBRID_TERM_ORDER:
        linear = {}
        quadratic = {}
        offset = 0.0

        if term_name == "H_cost":
            hybrid_mod.add_H_cost(problem, ctx, linear)
        elif term_name == "H_terminal_parent":
            offset += hybrid_mod.add_H_terminal_parent(
                ctx, linear, quadratic, constraint_weight
            )
        elif term_name == "H_nonterminal_parent":
            offset += hybrid_mod.add_H_nonterminal_parent(
                ctx, linear, quadratic, constraint_weight * n
            )
        elif term_name == "H_no_fake_root":
            hybrid_mod.add_H_no_fake_root(ctx, linear, quadratic, constraint_weight)
        elif term_name == "H_root_depth":
            offset += hybrid_mod.add_H_root_depth(
                ctx, linear, quadratic, constraint_weight
            )
        elif term_name == "H_depth":
            offset += hybrid_mod.add_H_depth(ctx, linear, quadratic, constraint_weight)
        else:
            raise ValueError(f"Unknown hybrid term {term_name}")

        term_bqms[term_name] = dimod.BinaryQuadraticModel(
            linear, quadratic, offset, dimod.BINARY
        )

    return term_bqms


def build_alex_term_bqms(
    problem: SteinerTree,
    penalty_weight: float,
) -> Dict[str, dimod.BinaryQuadraticModel]:
    root = problem.terminals[0]
    terminals = set(problem.terminals)
    nodes = list(problem.nodes)
    n = len(nodes)
    ctx = alex_mod._Context(problem=problem, root=root, terminals=terminals, nodes=nodes)

    term_bqms: Dict[str, dimod.BinaryQuadraticModel] = OrderedDict()

    for term_name in ALEX_TERM_ORDER:
        linear = {}
        quadratic = {}
        offset = 0.0

        if term_name == "O_I":
            alex_mod.add_OI(problem, ctx, linear)
        elif term_name == "F_I1":
            alex_mod.add_FI1(ctx, linear, quadratic, penalty_weight)
        elif term_name == "F_I2":
            alex_mod.add_FI2(problem, ctx, linear, quadratic, penalty_weight)
        elif term_name == "F_I3":
            offset += alex_mod.add_FI3(ctx, linear, quadratic, penalty_weight)
        elif term_name == "F_I4":
            alex_mod.add_FI4(ctx, quadratic, penalty_weight * n)
        elif term_name == "F_I5":
            alex_mod.add_FI5(ctx, linear, quadratic, penalty_weight)
        else:
            raise ValueError(f"Unknown alex term {term_name}")

        term_bqms[term_name] = dimod.BinaryQuadraticModel(
            linear, quadratic, offset, dimod.BINARY
        )

    return term_bqms


def build_model_bundle(
    problem: SteinerTree,
    model_name: str,
    constraint_weight: float,
    enabled_terms: Optional[Sequence[str]] = None,
) -> ModelBundle:
    if model_name == "hybrid":
        all_terms = HYBRID_TERM_ORDER
        term_bqms = build_hybrid_term_bqms(problem, constraint_weight)
        full_builder = hybrid_mod.steiner_to_bqm_hybrid
    elif model_name == "alex":
        all_terms = ALEX_TERM_ORDER
        term_bqms = build_alex_term_bqms(problem, constraint_weight)
        full_builder = alex_mod.steiner_to_bqm_ordering
    else:
        raise ValueError(f"Unknown model_name {model_name}")

    chosen_terms = list(enabled_terms) if enabled_terms is not None else list(all_terms)
    if chosen_terms == list(all_terms):
        full_bqm = full_builder(problem, constraint_weight)
    else:
        full_bqm = _combine_bqms(term_bqms, chosen_terms)
    labeled_bqm = _relabel_bqm(full_bqm)
    labeled_term_bqms = {
        name: _relabel_bqm(term_bqms[name])
        for name in chosen_terms
    }

    return ModelBundle(
        model_name=model_name,
        constraint_weight=float(constraint_weight),
        enabled_terms=chosen_terms,
        full_bqm=full_bqm,
        labeled_bqm=labeled_bqm,
        term_bqms={name: term_bqms[name] for name in chosen_terms},
        labeled_term_bqms=labeled_term_bqms,
    )


def coefficient_stats(
    bqm: dimod.BinaryQuadraticModel,
    *,
    experiment: str,
    model_name: str,
    constraint_weight: float,
    ablation_stage: str = "",
    enabled_terms: str = "",
) -> Dict[str, object]:
    linear_values = [float(v) for v in bqm.linear.values()]
    quadratic_values = [float(v) for v in bqm.quadratic.values()]
    abs_values = [abs(v) for v in linear_values + quadratic_values]

    row = {
        "experiment": experiment,
        "model": model_name,
        "constraint_weight": constraint_weight,
        "ablation_stage": ablation_stage,
        "enabled_terms": enabled_terms,
        "num_variables": int(bqm.num_variables),
        "num_quadratic_interactions": int(bqm.num_interactions),
        "linear_min": float(min(linear_values)) if linear_values else None,
        "linear_max": float(max(linear_values)) if linear_values else None,
        "quadratic_min": float(min(quadratic_values)) if quadratic_values else None,
        "quadratic_max": float(max(quadratic_values)) if quadratic_values else None,
        "max_abs_coefficient": float(max(abs_values)) if abs_values else 0.0,
        "offset": float(bqm.offset),
    }
    row.update(_percentile_summary([abs(v) for v in linear_values], "linear_abs"))
    row.update(_percentile_summary([abs(v) for v in quadratic_values], "quadratic_abs"))
    row.update(_percentile_summary(abs_values, "all_abs"))
    return row


def sample_with_openjij(
    labeled_bqm: dimod.BinaryQuadraticModel,
    num_reads: int,
    *,
    show_progress: bool,
    desc: str,
    **sampler_kwargs,
) -> dimod.SampleSet:
    qubo, offset = labeled_bqm.to_qubo()
    if abs(offset - labeled_bqm.offset) > 1e-9:
        raise ValueError("BQM offset changed during QUBO conversion.")

    sampler = oj.SQASampler()
    responses = []
    runs = range(num_reads)
    if show_progress:
        runs = tqdm(runs, total=num_reads, desc=desc, unit="read", file=sys.stdout)

    for _ in runs:
        responses.append(sampler.sample_qubo(qubo, num_reads=1, **sampler_kwargs))

    return dimod.concatenate(responses)


def aggregate_samples(response: dimod.SampleSet) -> List[Dict[str, object]]:
    aggregated = response.aggregate()
    rows = []
    for datum in aggregated.data(fields=["sample", "energy", "num_occurrences"]):
        sample = {str(var): int(value) for var, value in datum.sample.items()}
        rows.append(
            {
                "sample": sample,
                "energy_without_offset": float(datum.energy),
                "num_occurrences": int(datum.num_occurrences),
            }
        )

    rows.sort(key=lambda row: row["energy_without_offset"])
    return rows


def _edge_weight_lookup(problem: SteinerTree) -> Dict[Tuple[str, str], float]:
    weights = {}
    for a, b, weight in problem.edges:
        weights[(a, b)] = float(weight)
        weights[(b, a)] = float(weight)
    return weights


def _edge_key(problem: SteinerTree, a: str, b: str) -> Tuple[str, str]:
    node_index = {v: i for i, v in enumerate(problem.nodes)}
    if node_index[a] <= node_index[b]:
        return a, b
    return b, a


def decode_sample(
    problem: SteinerTree,
    model_name: str,
    sample: Mapping[str, int],
) -> Dict[str, object]:
    edge_weights = _edge_weight_lookup(problem)

    selected_arcs = []
    selected_arc_labels = []
    selected_undirected = OrderedDict()
    p_values = {}
    order_values = {}
    depth_bits = defaultdict(dict)
    slack_bits = defaultdict(dict)

    for label, raw_value in sample.items():
        value = int(raw_value)
        if value not in (0, 1):
            raise ValueError(f"Non-binary sample value {value} for {label}")

        parts = label.split("::")
        head = parts[0]

        if head == "e" and value == 1:
            _, u, v = parts
            selected_arcs.append((u, v, edge_weights[(u, v)]))
            selected_arc_labels.append(f"{u}->{v}({edge_weights[(u, v)]})")
            undirected_key = _edge_key(problem, u, v)
            if undirected_key not in selected_undirected:
                selected_undirected[undirected_key] = (
                    undirected_key[0],
                    undirected_key[1],
                    edge_weights[(u, v)],
                )
        elif head == "p":
            _, node = parts
            p_values[node] = value
        elif head == "x_order":
            _, u, v = parts
            order_values[(u, v)] = value
        elif head == "o":
            _, node, bit = parts
            depth_bits[node][int(bit)] = value
        elif head == "g":
            _, u, v, bit = parts
            slack_bits[(u, v)][int(bit)] = value

    depths = {}
    for node in problem.nodes:
        bits = depth_bits.get(node, {})
        depths[node] = sum((2 ** bit) * bits.get(bit, 0) for bit in bits)

    slacks = {}
    for edge_key, bits in slack_bits.items():
        slacks[edge_key] = sum((2 ** bit) * bits.get(bit, 0) for bit in bits)

    indegree = defaultdict(int)
    outdegree = defaultdict(int)
    for u, v, _ in selected_arcs:
        indegree[v] += 1
        outdegree[u] += 1
        indegree.setdefault(u, indegree.get(u, 0))
        outdegree.setdefault(v, outdegree.get(v, 0))

    undirected_edges = list(selected_undirected.values())
    undirected_labels = [f"{a}-{b}({w})" for a, b, w in undirected_edges]

    return {
        "model": model_name,
        "selected_arcs": selected_arcs,
        "selected_arc_labels": selected_arc_labels,
        "selected_undirected_edges": undirected_edges,
        "selected_undirected_labels": undirected_labels,
        "selected_arc_cost": float(sum(weight for _, _, weight in selected_arcs)),
        "selected_tree_cost": float(sum(weight for _, _, weight in undirected_edges)),
        "active_bit_count": int(sum(int(v) for v in sample.values())),
        "p_values": p_values,
        "order_values": order_values,
        "depths": depths,
        "slacks": slacks,
        "indegree": dict(indegree),
        "outdegree": dict(outdegree),
        "sample": dict(sample),
    }


def _comes_before(
    order_values: Mapping[Tuple[str, str], int],
    node_index: Mapping[str, int],
    a: str,
    b: str,
) -> bool:
    if a == b:
        raise ValueError("Ordering relation is undefined for identical nodes.")
    if node_index[a] < node_index[b]:
        return bool(order_values.get((a, b), 0))
    return not bool(order_values.get((b, a), 0))


def _directed_reachable(
    root: str,
    selected_arcs: Sequence[Tuple[str, str, float]],
) -> Tuple[set, Dict[str, List[str]]]:
    adj = defaultdict(list)
    for u, v, _ in selected_arcs:
        adj[u].append(v)

    visited = {root}
    stack = [root]
    while stack:
        node = stack.pop()
        for neighbor in adj[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                stack.append(neighbor)

    return visited, {node: list(neighbors) for node, neighbors in adj.items()}


def _has_directed_cycle(adj: Mapping[str, Sequence[str]]) -> bool:
    color = {}

    def visit(node: str) -> bool:
        state = color.get(node, 0)
        if state == 1:
            return True
        if state == 2:
            return False

        color[node] = 1
        for neighbor in adj.get(node, []):
            if visit(neighbor):
                return True
        color[node] = 2
        return False

    for node in adj:
        if color.get(node, 0) == 0 and visit(node):
            return True
    return False


def check_feasibility(
    problem: SteinerTree,
    model_name: str,
    sample: Mapping[str, int],
    decoded: Optional[Mapping[str, object]] = None,
) -> Dict[str, object]:
    decoded = decoded or decode_sample(problem, model_name, sample)
    root = problem.terminals[0]
    terminals = set(problem.terminals)
    node_index = {v: i for i, v in enumerate(problem.nodes)}
    edge_lookup = {_edge_key(problem, a, b) for a, b, _ in problem.edges}
    violations = []

    def add_violation(kind: str, magnitude: float, detail: str) -> None:
        violations.append(
            {
                "type": kind,
                "magnitude": float(magnitude),
                "detail": detail,
            }
        )

    selected_arcs = decoded["selected_arcs"]
    indegree = defaultdict(int, decoded["indegree"])
    outdegree = defaultdict(int, decoded["outdegree"])
    reachable, adj = _directed_reachable(root, selected_arcs)
    selected_nodes = {root}
    for u, v, _ in selected_arcs:
        selected_nodes.add(u)
        selected_nodes.add(v)

    terminals_not_reached = sorted(terminals - reachable)
    if terminals_not_reached:
        add_violation(
            "terminal_not_reached",
            float(len(terminals_not_reached)),
            f"Unreachable terminals from root {root}: {terminals_not_reached}",
        )

    detached_nodes = sorted(selected_nodes - reachable)
    if detached_nodes:
        add_violation(
            "detached_component",
            float(len(detached_nodes)),
            f"Selected nodes not reachable from root {root}: {detached_nodes}",
        )

    selected_arc_set = {(u, v) for u, v, _ in selected_arcs}
    for a, b, _ in problem.edges:
        if (a, b) in selected_arc_set and (b, a) in selected_arc_set:
            add_violation(
                "opposite_arc",
                1.0,
                f"Both directions selected on edge {a}-{b}",
            )

    if _has_directed_cycle(adj):
        add_violation(
            "directed_cycle",
            1.0,
            "Selected directed arcs contain a cycle.",
        )

    selected_undirected = {
        _edge_key(problem, u, v) for u, v, _ in selected_arcs
    }
    if len(selected_undirected) != len(selected_nodes) - 1:
        add_violation(
            "not_tree_edge_count",
            abs(len(selected_undirected) - (len(selected_nodes) - 1)),
            f"|E|={len(selected_undirected)} while |V|-1={len(selected_nodes) - 1}",
        )

    for terminal in terminals:
        if terminal == root:
            continue
        incoming = indegree.get(terminal, 0)
        if incoming != 1:
            add_violation(
                "terminal_parent",
                abs(incoming - 1),
                f"Terminal {terminal} has indegree {incoming}, expected 1",
            )

    if model_name == "hybrid":
        p_values = decoded["p_values"]
        depths = decoded["depths"]
        slacks = decoded["slacks"]
        ctx = hybrid_mod._HybridContext(problem)

        root_depth = depths.get(root, 0)
        if root_depth != 0:
            add_violation(
                "root_depth",
                abs(root_depth),
                f"Root depth is {root_depth}, expected 0",
            )

        for node in problem.nodes:
            if node in terminals:
                continue

            p_value = int(p_values.get(node, 0))
            incoming = indegree.get(node, 0)
            if p_value != incoming:
                add_violation(
                    "nonterminal_parent",
                    abs(p_value - incoming),
                    f"Node {node} has p={p_value}, indegree={incoming}",
                )

            if outdegree.get(node, 0) > 0 and p_value == 0:
                add_violation(
                    "no_fake_root",
                    float(outdegree[node]),
                    f"Node {node} has outgoing arcs but p={p_value}",
                )

        for u, v, _ in ctx.all_arcs():
            e_value = int(sample.get(f"e::{u}::{v}", 0))
            slack = int(slacks.get((u, v), 0))
            residual = (
                depths.get(v, 0)
                - depths.get(u, 0)
                - 1
                + ctx.M * (1 - e_value)
                - slack
            )
            if residual != 0:
                add_violation(
                    "depth_residual",
                    abs(residual),
                    f"Arc {u}->{v} residual={residual}",
                )

    elif model_name == "alex":
        order_values = decoded["order_values"]
        non_root_nodes = [v for v in problem.nodes if v != root]

        for node in problem.nodes:
            if node in terminals or node == root:
                continue
            incoming = indegree.get(node, 0)
            if incoming > 1:
                add_violation(
                    "multiple_incoming_nonterminal",
                    incoming - 1,
                    f"Node {node} has indegree {incoming}, expected at most 1",
                )
            if outdegree.get(node, 0) > 0 and incoming == 0:
                add_violation(
                    "fake_root_nonterminal",
                    float(outdegree[node]),
                    f"Node {node} has outgoing arcs but indegree 0",
                )

        sorted_non_root = sorted(non_root_nodes, key=node_index.get)
        for i in range(len(sorted_non_root)):
            for j in range(i + 1, len(sorted_non_root)):
                for k in range(j + 1, len(sorted_non_root)):
                    u = sorted_non_root[i]
                    v = sorted_non_root[j]
                    w = sorted_non_root[k]
                    uv = _comes_before(order_values, node_index, u, v)
                    vw = _comes_before(order_values, node_index, v, w)
                    uw = _comes_before(order_values, node_index, u, w)
                    if uv == vw and uw != uv:
                        add_violation(
                            "ordering_transitivity",
                            1.0,
                            f"Inconsistent order among ({u}, {v}, {w})",
                        )

        for a, b, _ in problem.edges:
            if a == root or b == root:
                continue
            u, v = _edge_key(problem, a, b)
            x_value = int(order_values.get((u, v), 0))
            e_uv = int(sample.get(f"e::{u}::{v}", 0))
            e_vu = int(sample.get(f"e::{v}::{u}", 0))
            if x_value == 1 and e_vu == 1:
                add_violation(
                    "edge_order_consistency",
                    1.0,
                    f"Arc {v}->{u} selected while order says {u} before {v}",
                )
            if x_value == 0 and e_uv == 1:
                add_violation(
                    "edge_order_consistency",
                    1.0,
                    f"Arc {u}->{v} selected while order says {v} before {u}",
                )

    else:
        raise ValueError(f"Unknown model_name {model_name}")

    return {
        "feasible": not violations,
        "violations": violations,
        "violation_types": [item["type"] for item in violations],
        "model_specific_constraints_checked": (
            "hybrid_parent_depth_use" if model_name == "hybrid" else "alex_order_parent"
        ),
        "flow_conservation_applicable": False,
        "flow_conservation_status": "not_applicable",
    }


def classify_sample(
    feasible: bool,
    decoded_tree_cost: float,
    optimal_cost: Optional[float],
) -> str:
    if not feasible:
        return "infeasible"
    if optimal_cost is not None and abs(decoded_tree_cost - optimal_cost) <= 1e-6:
        return "feasible optimal"
    return "feasible suboptimal"


def analyze_sample(
    problem: SteinerTree,
    bundle: ModelBundle,
    aggregated_row: Mapping[str, object],
    *,
    experiment: str,
    optimal_cost: Optional[float],
    ablation_stage: str = "",
    rank: Optional[int] = None,
) -> Dict[str, object]:
    sample = aggregated_row["sample"]
    decoded = decode_sample(problem, bundle.model_name, sample)
    feasibility = check_feasibility(problem, bundle.model_name, sample, decoded)
    total_energy_with_offset = float(
        aggregated_row["energy_without_offset"] + bundle.labeled_bqm.offset
    )
    term_contributions = {
        term_name: float(term_bqm.energy(sample))
        for term_name, term_bqm in bundle.labeled_term_bqms.items()
    }
    contribution_sum = float(sum(term_contributions.values()))

    analysis = {
        "experiment": experiment,
        "model": bundle.model_name,
        "constraint_weight": bundle.constraint_weight,
        "ablation_stage": ablation_stage,
        "sample_rank": rank,
        "num_occurrences": int(aggregated_row["num_occurrences"]),
        "energy_without_offset": float(aggregated_row["energy_without_offset"]),
        "total_energy_with_offset": total_energy_with_offset,
        "term_contribution_sum": contribution_sum,
        "term_balance_error": float(total_energy_with_offset - contribution_sum),
        "active_bit_count": decoded["active_bit_count"],
        "selected_arcs": "; ".join(decoded["selected_arc_labels"]),
        "selected_undirected_edges": "; ".join(decoded["selected_undirected_labels"]),
        "selected_arc_cost": decoded["selected_arc_cost"],
        "decoded_tree_cost": decoded["selected_tree_cost"],
        "feasible": bool(feasibility["feasible"]),
        "classification": classify_sample(
            bool(feasibility["feasible"]),
            float(decoded["selected_tree_cost"]),
            optimal_cost,
        ),
        "violation_types": "; ".join(feasibility["violation_types"]),
        "violation_details": " | ".join(
            f"{item['type']}[{item['magnitude']}]: {item['detail']}"
            for item in feasibility["violations"]
        ),
        "flow_conservation_status": feasibility["flow_conservation_status"],
    }
    analysis.update(term_contributions)
    return analysis


def summarize_analyses(
    analyses: Sequence[Mapping[str, object]],
    *,
    total_reads: Optional[int] = None,
    optimal_cost: Optional[float] = None,
) -> Dict[str, object]:
    if not analyses:
        return {
            "num_feasible": 0,
            "num_infeasible": 0,
            "best_feasible_energy": None,
            "best_infeasible_energy": None,
            "best_decoded_tree_cost": None,
            "feasible_sample_rate": 0.0,
            "optimal_hit_rate": None if optimal_cost is None else 0.0,
            "most_common_violation_type": "",
            "violation_frequency": {},
        }

    weighted_feasible = 0
    weighted_total = 0
    best_feasible_energy = None
    best_infeasible_energy = None
    best_tree_cost = None
    optimal_hits = 0
    violation_counter = Counter()

    for row in analyses:
        occurrences = int(row["num_occurrences"])
        weighted_total += occurrences

        if row["feasible"]:
            weighted_feasible += occurrences
            if best_feasible_energy is None or row["total_energy_with_offset"] < best_feasible_energy:
                best_feasible_energy = row["total_energy_with_offset"]
            if best_tree_cost is None or row["decoded_tree_cost"] < best_tree_cost:
                best_tree_cost = row["decoded_tree_cost"]
            if optimal_cost is not None and abs(row["decoded_tree_cost"] - optimal_cost) <= 1e-6:
                optimal_hits += occurrences
        else:
            if best_infeasible_energy is None or row["total_energy_with_offset"] < best_infeasible_energy:
                best_infeasible_energy = row["total_energy_with_offset"]
            for violation_type in filter(None, row["violation_types"].split("; ")):
                violation_counter[violation_type] += occurrences

    denominator = total_reads if total_reads is not None else weighted_total
    most_common_violation = violation_counter.most_common(1)[0][0] if violation_counter else ""

    return {
        "num_feasible": int(sum(1 for row in analyses if row["feasible"])),
        "num_infeasible": int(sum(1 for row in analyses if not row["feasible"])),
        "best_feasible_energy": best_feasible_energy,
        "best_infeasible_energy": best_infeasible_energy,
        "best_decoded_tree_cost": best_tree_cost,
        "feasible_sample_rate": (weighted_feasible / denominator) if denominator else 0.0,
        "optimal_hit_rate": (
            (optimal_hits / denominator) if (optimal_cost is not None and denominator) else None
        ),
        "most_common_violation_type": most_common_violation,
        "violation_frequency": dict(violation_counter),
    }


def _try_gurobi_optimum(problem: SteinerTree) -> Tuple[Optional[float], str]:
    try:
        from SteinerTreeProblemQUBO.MyFormulization.gurobi_solver import solve_ilp

        result = solve_ilp(problem)
        if result["cost"] is not None:
            return float(result["cost"]), "gurobi"
        return None, f"gurobi_status={result['status']}"
    except Exception as exc:
        return None, f"gurobi_unavailable: {exc}"


def compute_optimal_cost(problem: SteinerTree) -> Tuple[Optional[float], str]:
    if TRY_GUROBI_FIRST:
        optimal_cost, source = _try_gurobi_optimum(problem)
        if optimal_cost is not None:
            return optimal_cost, source

    if len(problem.nodes) <= EXACT_SOLVER_NODE_LIMIT:
        try:
            result = solve_exact(problem)
            return float(result["cost"]), "exact"
        except Exception as exc:
            return None, f"exact_failed: {exc}"

    return None, "not_available"


def print_problem_summary(
    problem: SteinerTree,
    optimal_cost: Optional[float],
    optimal_source: str,
    constraint_weight: float,
) -> None:
    print(f"nodes={len(problem.nodes)} edges={len(problem.edges)} terminals={len(problem.terminals)}")
    print(f"root={problem.terminals[0]} terminals={problem.terminals}")
    print(f"constraint_weight={constraint_weight}")
    print(f"optimal_cost={optimal_cost} source={optimal_source}")
    print()


def print_top_sample_summary(
    model_name: str,
    analyses: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    print(f"[{model_name}] top-{len(analyses)} unique samples")
    for row in analyses:
        print(
            f"  rank={row['sample_rank']} occ={row['num_occurrences']} "
            f"energy={row['total_energy_with_offset']:.3f} "
            f"class={row['classification']} "
            f"tree_cost={row['decoded_tree_cost']:.3f} "
            f"active_bits={row['active_bit_count']}"
        )
        term_bits = [
            f"{term}={row[term]:.3f}"
            for term in row
            if term in HYBRID_TERM_ORDER or term in ALEX_TERM_ORDER
        ]
        print(f"    terms: {' | '.join(term_bits)}")
        print(f"    edges: {row['selected_undirected_edges'] or '(none)'}")
        print(f"    violations: {row['violation_details'] or '(none)'}")
    print(
        f"  summary: feasible={summary['num_feasible']} "
        f"infeasible={summary['num_infeasible']} "
        f"best_feasible_energy={summary['best_feasible_energy']} "
        f"best_infeasible_energy={summary['best_infeasible_energy']} "
        f"best_tree_cost={summary['best_decoded_tree_cost']} "
        f"common_violation={summary['most_common_violation_type'] or '(none)'}"
    )
    print()


def print_coefficient_summary(row: Mapping[str, object]) -> None:
    label = f"{row['experiment']}:{row['model']}"
    if row["ablation_stage"]:
        label += f":{row['ablation_stage']}"
    print(
        f"[{label}] coeffs "
        f"vars={row['num_variables']} "
        f"quads={row['num_quadratic_interactions']} "
        f"lin=[{row['linear_min']}, {row['linear_max']}] "
        f"quad=[{row['quadratic_min']}, {row['quadratic_max']}] "
        f"maxabs={row['max_abs_coefficient']} "
        f"offset={row['offset']}"
    )


def run_model_diagnostics(
    problem: SteinerTree,
    *,
    model_name: str,
    constraint_weight: float,
    num_reads: int,
    top_k: int,
    experiment: str,
    optimal_cost: Optional[float],
    coefficient_rows: List[Dict[str, object]],
    sample_rows: List[Dict[str, object]],
    ablation_stage: str = "",
    enabled_terms: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    bundle = build_model_bundle(
        problem,
        model_name=model_name,
        constraint_weight=constraint_weight,
        enabled_terms=enabled_terms,
    )
    coeff_row = coefficient_stats(
        bundle.labeled_bqm,
        experiment=experiment,
        model_name=model_name,
        constraint_weight=constraint_weight,
        ablation_stage=ablation_stage,
        enabled_terms="; ".join(bundle.enabled_terms),
    )
    coefficient_rows.append(coeff_row)
    print_coefficient_summary(coeff_row)

    response = sample_with_openjij(
        bundle.labeled_bqm,
        num_reads,
        show_progress=True,
        desc=f"{experiment}:{model_name}",
        **SQA_KWARGS,
    )
    aggregated = aggregate_samples(response)

    all_analyses = [
        analyze_sample(
            problem,
            bundle,
            row,
            experiment=experiment,
            optimal_cost=optimal_cost,
            ablation_stage=ablation_stage,
        )
        for row in aggregated
    ]

    top_analyses = []
    for rank, analysis in enumerate(all_analyses[:top_k], start=1):
        enriched = dict(analysis)
        enriched["sample_rank"] = rank
        top_analyses.append(enriched)
        sample_rows.append(enriched)

    top_summary = summarize_analyses(top_analyses, optimal_cost=optimal_cost)
    all_summary = summarize_analyses(
        all_analyses,
        total_reads=num_reads,
        optimal_cost=optimal_cost,
    )

    return {
        "bundle": bundle,
        "response": response,
        "aggregated": aggregated,
        "top_analyses": top_analyses,
        "top_summary": top_summary,
        "all_summary": all_summary,
        "coefficient_row": coeff_row,
    }


def _flatten_violation_frequency(counter_dict: Mapping[str, int]) -> str:
    if not counter_dict:
        return ""
    return "; ".join(f"{key}={value}" for key, value in sorted(counter_dict.items()))


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_DIR / f"qubo_diagnostics_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = make_problem(PROBLEM_SPEC)
    constraint_weight = (
        float(BASE_CONSTRAINT_WEIGHT)
        if BASE_CONSTRAINT_WEIGHT is not None
        else default_constraint_weight(problem)
    )
    optimal_cost, optimal_source = compute_optimal_cost(problem)

    sample_rows: List[Dict[str, object]] = []
    coefficient_rows: List[Dict[str, object]] = []
    penalty_rows: List[Dict[str, object]] = []
    ablation_rows: List[Dict[str, object]] = []

    print_problem_summary(problem, optimal_cost, optimal_source, constraint_weight)

    baseline_results = {}
    for model_name in ("hybrid", "alex"):
        result = run_model_diagnostics(
            problem,
            model_name=model_name,
            constraint_weight=constraint_weight,
            num_reads=NUM_READS,
            top_k=TOP_K_SAMPLES,
            experiment="baseline",
            optimal_cost=optimal_cost,
            coefficient_rows=coefficient_rows,
            sample_rows=sample_rows,
        )
        baseline_results[model_name] = result
        print_top_sample_summary(
            model_name,
            result["top_analyses"],
            result["top_summary"],
        )

    print("[penalty_sweep] hybrid")
    for penalty in PENALTY_GRID:
        result = run_model_diagnostics(
            problem,
            model_name="hybrid",
            constraint_weight=float(penalty),
            num_reads=NUM_READS,
            top_k=TOP_K_SAMPLES,
            experiment="penalty_sweep",
            optimal_cost=optimal_cost,
            coefficient_rows=coefficient_rows,
            sample_rows=sample_rows,
        )
        summary = result["all_summary"]
        row = {
            "constraint_weight": float(penalty),
            "best_total_energy": result["aggregated"][0]["energy_without_offset"]
            + result["bundle"].labeled_bqm.offset,
            "num_feasible_unique": summary["num_feasible"],
            "num_infeasible_unique": summary["num_infeasible"],
            "best_feasible_energy": summary["best_feasible_energy"],
            "best_infeasible_energy": summary["best_infeasible_energy"],
            "best_feasible_tree_cost": summary["best_decoded_tree_cost"],
            "feasible_sample_rate": summary["feasible_sample_rate"],
            "optimal_hit_rate": summary["optimal_hit_rate"],
            "most_common_violation_type": summary["most_common_violation_type"],
            "violation_frequency": _flatten_violation_frequency(
                summary["violation_frequency"]
            ),
        }
        penalty_rows.append(row)
        print(
            f"  penalty={penalty:<5} "
            f"best_total_energy={row['best_total_energy']:.3f} "
            f"best_feasible_tree_cost={row['best_feasible_tree_cost']} "
            f"feasible_rate={row['feasible_sample_rate']:.3f} "
            f"optimal_hit_rate={row['optimal_hit_rate']} "
            f"common_violation={row['most_common_violation_type'] or '(none)'}"
        )
    print()

    print("[ablation] hybrid")
    for stage_name, enabled_terms in HYBRID_ABLATIONS.items():
        result = run_model_diagnostics(
            problem,
            model_name="hybrid",
            constraint_weight=constraint_weight,
            num_reads=NUM_READS,
            top_k=TOP_K_SAMPLES,
            experiment="ablation",
            optimal_cost=optimal_cost,
            coefficient_rows=coefficient_rows,
            sample_rows=sample_rows,
            ablation_stage=stage_name,
            enabled_terms=enabled_terms,
        )
        summary = result["all_summary"]
        row = {
            "ablation_stage": stage_name,
            "enabled_terms": "; ".join(enabled_terms),
            "constraint_weight": constraint_weight,
            "best_total_energy": result["aggregated"][0]["energy_without_offset"]
            + result["bundle"].labeled_bqm.offset,
            "num_feasible_unique": summary["num_feasible"],
            "num_infeasible_unique": summary["num_infeasible"],
            "best_feasible_energy": summary["best_feasible_energy"],
            "best_infeasible_energy": summary["best_infeasible_energy"],
            "best_feasible_tree_cost": summary["best_decoded_tree_cost"],
            "feasible_sample_rate": summary["feasible_sample_rate"],
            "optimal_hit_rate": summary["optimal_hit_rate"],
            "most_common_violation_type": summary["most_common_violation_type"],
            "violation_frequency": _flatten_violation_frequency(
                summary["violation_frequency"]
            ),
        }
        ablation_rows.append(row)
        print(
            f"  stage={stage_name:<24} "
            f"best_total_energy={row['best_total_energy']:.3f} "
            f"best_feasible_tree_cost={row['best_feasible_tree_cost']} "
            f"feasible_rate={row['feasible_sample_rate']:.3f} "
            f"common_violation={row['most_common_violation_type'] or '(none)'}"
        )
    print()

    write_csv(output_dir / "sample_diagnostics.csv", sample_rows)
    write_csv(output_dir / "penalty_sweep.csv", penalty_rows)
    write_csv(output_dir / "ablation_results.csv", ablation_rows)
    write_csv(output_dir / "coefficient_stats.csv", coefficient_rows)

    print("Wrote diagnostics:")
    print(f"  {output_dir / 'sample_diagnostics.csv'}")
    print(f"  {output_dir / 'penalty_sweep.csv'}")
    print(f"  {output_dir / 'ablation_results.csv'}")
    print(f"  {output_dir / 'coefficient_stats.csv'}")


if __name__ == "__main__":
    main()
